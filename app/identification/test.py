"""External CRISM inference/evaluation using a training checkpoint.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm

from .crism_common import (
    TileMeta,
    align_label_to_tiles,
    all_image_points,
    all_labeled_points,
    attach_labels,
    build_patches_from_prepared,
    discover_single_tile,
    discover_tiles,
    extract_tile_points,
    load_json,
    load_label_map,
    load_mat_data,
    prepare_tile_cube,
    resolve_device,
)
from .lsga import lsga_hsi


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "External CRISM test/inference"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="External test JSON configuration.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="",
        help="Optional CLI checkpoint override.",
    )
    return parser


def update_confusion(
    cm: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> None:
    valid = (
        (targets >= 0)
        & (targets < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )
    encoded = (
        targets[valid] * num_classes
        + predictions[valid]
    )
    cm += np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: np.ndarray) -> Dict:
    total = int(cm.sum())
    correct = int(np.trace(cm))
    oa = correct / max(total, 1)

    class_total = cm.sum(axis=1)
    class_correct = np.diag(cm)
    present = class_total > 0

    per_class = np.full(
        len(class_total),
        np.nan,
        dtype=np.float64,
    )
    per_class[present] = (
        class_correct[present]
        / class_total[present]
    )

    aa = (
        float(np.nanmean(per_class))
        if np.any(present)
        else np.nan
    )

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    pe = (
        float(np.dot(row_sum, col_sum))
        / max(float(total * total), 1.0)
    )
    kappa = (
        (oa - pe) / (1.0 - pe)
        if abs(1.0 - pe) > 1e-12
        else 0.0
    )

    return {
        "OA": float(oa),
        "AA": float(aa),
        "kappa": float(kappa),
        "per_class_acc": per_class,
        "class_total": class_total.astype(np.int64),
        "class_correct": class_correct.astype(np.int64),
    }


def save_metrics(
    output_dir: Path,
    prefix: str,
    cm: np.ndarray,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics_from_confusion(cm)

    np.savetxt(
        output_dir / f"{prefix}_confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=",",
    )

    with open(
        output_dir / f"{prefix}_metrics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["OA", metrics["OA"]])
        writer.writerow(["AA", metrics["AA"]])
        writer.writerow(["Kappa", metrics["kappa"]])
        writer.writerow([])
        writer.writerow(
            ["class_id", "accuracy", "correct", "total"]
        )

        for index, accuracy in enumerate(
            metrics["per_class_acc"],
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    "" if np.isnan(accuracy) else float(accuracy),
                    int(metrics["class_correct"][index - 1]),
                    int(metrics["class_total"][index - 1]),
                ]
            )

    return metrics


def merge_checkpoint_args(
    checkpoint: Dict,
    runtime: Dict,
) -> Dict:
    checkpoint_args = dict(checkpoint.get("args", {}))
    if not checkpoint_args:
        raise KeyError(
            "Checkpoint does not contain training args"
        )

    args = checkpoint_args

    # Only runtime paths and execution resources may override training settings.
    override_keys = {
        "device",
        "batch_size",
        "test_mode",
        "test_img",
        "test_label",
        "tile_dir",
        "label_path",
        "tile_pattern",
        "tile_w",
        "tile_position_mode",
        "output_dir",
        "data_key",
        "label_key",
        "exclude_train_points",
        "split_path",
    }
    for key in override_keys:
        if key in runtime and runtime[key] not in (None, ""):
            args[key] = runtime[key]

    if not args.get("split_path") and checkpoint.get("split_path"):
        args["split_path"] = checkpoint["split_path"]

    args["norm_mode"] = "none"
    args["use_spectral_features"] = False

    required = [
        "patch_size",
        "num_classes",
        "input_channels",
    ]
    missing = [key for key in required if key not in args]
    if missing:
        raise KeyError(
            f"Checkpoint args missing: {missing}"
        )

    return args


def filter_points_by_excluding_train(
    points: np.ndarray,
    label_shape: Tuple[int, int],
    split_path: str | Path,
) -> Tuple[np.ndarray, int]:
    """Remove training coordinates from labeled tile-mode test points."""
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"split_path not found: {split_path}"
        )

    split = np.load(split_path)
    if "train" not in split:
        raise KeyError(
            f"{split_path} does not contain a 'train' array"
        )

    train_points = np.asarray(split["train"], dtype=np.int64)
    if train_points.size == 0 or len(points) == 0:
        return points, 0

    in_bounds = (
        (train_points[:, 0] >= 0)
        & (train_points[:, 0] < label_shape[0])
        & (train_points[:, 1] >= 0)
        & (train_points[:, 1] < label_shape[1])
    )
    train_points = train_points[in_bounds]
    if len(train_points) == 0:
        return points, 0

    point_key = np.ravel_multi_index(
        (points[:, 0], points[:, 1]),
        dims=label_shape,
    )
    train_key = np.ravel_multi_index(
        (train_points[:, 0], train_points[:, 1]),
        dims=label_shape,
    )
    keep = ~np.isin(point_key, train_key)

    return points[keep].astype(np.int64, copy=False), int(
        np.sum(~keep)
    )


def build_scene(
    args: Dict,
) -> Tuple[List[TileMeta], np.ndarray, np.ndarray, bool]:
    mode = str(args.get("test_mode", "single")).lower()
    data_key = args.get("data_key", "data")
    label_key = args.get("label_key")
    num_classes = int(args["num_classes"])

    if mode == "single":
        if not args.get("test_img"):
            raise ValueError("single mode requires test_img")

        tiles = discover_single_tile(
            args["test_img"],
            data_key=data_key,
            data_layout=args.get("data_layout", "HWB"),
        )
        height = tiles[0].height
        width = tiles[0].width

        if args.get("test_label"):
            label_map = load_label_map(
                args["test_label"],
                key=label_key,
            )
            label_map = align_label_to_tiles(
                label_map,
                tiles,
            )
            points = all_labeled_points(
                label_map,
                num_classes,
            )
            has_label = True
        else:
            label_map = np.zeros(
                (height, width),
                dtype=np.int64,
            )
            points = all_image_points(height, width)
            has_label = False

        return tiles, label_map, points, has_label

    if mode == "tile":
        if not args.get("tile_dir"):
            raise ValueError("tile mode requires tile_dir")

        tiles = discover_tiles(
            args["tile_dir"],
            int(args.get("tile_w", 600)),
            args.get("tile_pattern", "*"),
            data_key=data_key,
            position_mode=args.get(
                "tile_position_mode",
                "tile_id",
            ),
            data_layout=args.get("data_layout", "HWB"),
        )

        height = tiles[0].height
        width = max(
            tile.start_col + tile.width for tile in tiles
        )

        if args.get("label_path"):
            label_map = load_label_map(
                args["label_path"],
                key=label_key,
            )
            label_map = align_label_to_tiles(
                label_map,
                tiles,
            )
            points = all_labeled_points(
                label_map,
                num_classes,
            )
            if bool(args.get("exclude_train_points", False)):
                if not args.get("split_path"):
                    raise ValueError(
                        "exclude_train_points=true requires split_path "
                        "or a checkpoint saved with split_path"
                    )
                before = len(points)
                points, removed = filter_points_by_excluding_train(
                    points,
                    label_map.shape,
                    args["split_path"],
                )
                print(
                    "Tile-mode labeled points: "
                    f"{before} total, removed {removed} training "
                    f"points, evaluating {len(points)} points."
                )
            has_label = True
        else:
            label_map = np.zeros(
                (height, width),
                dtype=np.int64,
            )
            points = all_image_points(height, width)
            has_label = False

        return tiles, label_map, points, has_label

    raise ValueError("test_mode must be 'single' or 'tile'")


def predict_scene(
    model,
    tiles: List[TileMeta],
    points: np.ndarray,
    label_map: np.ndarray,
    has_label: bool,
    args: Dict,
    device,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    num_classes = int(args["num_classes"])
    patch_size = int(args["patch_size"])
    batch_size = int(args.get("batch_size", 256))

    prediction_map = np.zeros_like(
        label_map,
        dtype=np.int16,
    )
    cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    all_predictions: List[np.ndarray] = []
    all_points: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for tile in tqdm(tiles, desc="External inference"):
            tile_points = extract_tile_points(
                points,
                tile.start_col,
                tile.width,
            )
            if len(tile_points) == 0:
                continue

            raw = load_mat_data(
                tile.path,
                key=args.get("data_key", "data"),
                prefer_3d=True,
            )
            prepared = prepare_tile_cube(raw, args)

            if prepared.shape[-1] != int(
                args["input_channels"]
            ):
                raise ValueError(
                    f"Input channel mismatch for {tile.path}: "
                    f"prepared={prepared.shape[-1]}, "
                    f"checkpoint={args['input_channels']}"
                )

            tile_predictions: List[np.ndarray] = []

            for start in range(0, len(tile_points), batch_size):
                end = min(
                    start + batch_size,
                    len(tile_points),
                )
                batch_points = tile_points[start:end]
                patches = build_patches_from_prepared(
                    prepared,
                    batch_points,
                    tile.start_col,
                    patch_size,
                )
                x = torch.from_numpy(patches).float().to(
                    device,
                    non_blocking=True,
                )
                pred = model(x).argmax(dim=1).cpu().numpy()
                tile_predictions.append(
                    pred.astype(np.int16)
                )

                if has_label:
                    gt = attach_labels(
                        batch_points,
                        label_map,
                    )
                    update_confusion(
                        cm,
                        gt,
                        pred,
                        num_classes,
                    )

            tile_pred = np.concatenate(tile_predictions)
            prediction_map[
                tile_points[:, 0],
                tile_points[:, 1],
            ] = tile_pred + 1

            all_predictions.append(tile_pred)
            all_points.append(tile_points)
            del raw, prepared

    if all_predictions:
        ordered_predictions = np.concatenate(all_predictions)
        ordered_points = np.concatenate(all_points)
    else:
        ordered_predictions = np.empty(
            (0,), dtype=np.int16
        )
        ordered_points = np.empty(
            (0, 2), dtype=np.int64
        )

    metrics = (
        metrics_from_confusion(cm)
        if has_label
        else {
            "OA": np.nan,
            "AA": np.nan,
            "kappa": np.nan,
            "per_class_acc": np.full(
                num_classes,
                np.nan,
            ),
        }
    )

    return prediction_map, (cm if has_label else None), {
        **metrics,
        "points": ordered_points,
        "predictions": ordered_predictions,
    }


def evaluate_checkpoint(
    runtime_args: Dict,
    checkpoint_path: Path,
) -> Dict:
    initial_device = resolve_device(
        runtime_args.get("device", 0)
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=initial_device,
    )

    args = merge_checkpoint_args(
        checkpoint,
        runtime_args,
    )
    device = resolve_device(args.get("device", 0))
    output_dir = Path(
        args.get("output_dir", "./external_test")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles, label_map, points, has_label = build_scene(args)

    model = lsga_hsi(args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prediction_map, cm, result = predict_scene(
        model,
        tiles,
        points,
        label_map,
        has_label,
        args,
        device,
    )

    mode_name = str(args.get("test_mode", "single")).lower()
    no_train_suffix = (
        "_no_train"
        if mode_name == "tile"
        and bool(args.get("exclude_train_points", False))
        else ""
    )
    prefix = (
        f"{args.get('dataset', 'crism')}_"
        f"{mode_name}_external{no_train_suffix}"
    )

    save_payload = {
        "pre": prediction_map,
        "points": result["points"],
        "predictions_0based": result["predictions"],
        "predictions_1based": (
            result["predictions"].astype(np.int64) + 1
        ),
        "exclude_train_points": bool(
            args.get("exclude_train_points", False)
        ),
    }
    if args.get("split_path"):
        save_payload["split_path"] = str(args["split_path"])
    if has_label:
        save_payload["gt"] = label_map
        save_payload["confusion_matrix"] = cm

    savemat(
        output_dir / f"{prefix}_prediction.mat",
        save_payload,
        do_compression=True,
    )

    if has_label and cm is not None:
        metrics = save_metrics(
            output_dir,
            prefix,
            cm,
        )
        print(
            f"External test | OA={metrics['OA'] * 100:.2f}% | "
            f"AA={metrics['AA'] * 100:.2f}% | "
            f"Kappa={metrics['kappa']:.4f}"
        )
        return metrics

    print("Prediction completed without labels.")
    return result


def main() -> None:
    cli = setup_parser().parse_args()
    runtime_args = load_json(cli.config)

    checkpoint_path = Path(
        cli.checkpoint_path
        or runtime_args.get("checkpoint_path", "")
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    evaluate_checkpoint(
        runtime_args,
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
