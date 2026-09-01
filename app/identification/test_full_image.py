"""Full-image CRISM inference, labeled-region metrics, and visualizations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import Patch
import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm

from .crism_common import (
    TileMeta,
    align_label_to_tiles,
    discover_single_tile,
    discover_tiles,
    load_json,
    load_label_map,
    load_mat_data,
    normalize_label_map,
    prepare_tile_cube,
    relayout_tiles_for_label,
    resolve_device,
)
from .io import load_wavelengths
from .lsga import lsga_hsi, prepare_lsga_for_eval


# Discrete high-saturation colors:
# index 0 = background; indices 1..24 = CRISM classes 1..24.
BASE_COLORS = [
    "#000000",
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#FFFF33",
    "#A65628",
    "#F781BF",
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#E5C494",
    "#B3B3B3",
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#666666",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CRISM full-image external test")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", default="")
    return p


def merge_checkpoint_args(checkpoint: Dict, runtime: Dict) -> Dict:
    args = dict(checkpoint.get("args", {}))
    if not args:
        raise KeyError("Checkpoint does not contain training args.")

    runtime_keys = {
        "device", "batch_size", "use_amp",
        "test_mode", "test_img", "test_label",
        "tile_dir", "label_path", "tile_pattern", "tile_w",
        "tile_position_mode", "data_key", "label_key",
        "output_dir", "scene_name", "confidence_threshold",
        "save_confidence_map", "save_mat", "save_png",
        "crop_labeled_region", "crop_padding", "visual_dpi",
        "class_names",
    }

    for key in runtime_keys:
        if key in runtime and runtime[key] not in (None, ""):
            args[key] = runtime[key]

    args["norm_mode"] = "none"
    args["use_spectral_features"] = False

    required = ["patch_size", "num_classes", "input_channels"]
    missing = [key for key in required if key not in args]
    if missing:
        raise KeyError(f"Checkpoint args missing: {missing}")
    return args


def discover_scene(
    args: Dict,
) -> Tuple[List[TileMeta], Optional[np.ndarray], int, int]:
    mode = str(args.get("test_mode", "single")).lower()
    data_key = args.get("data_key", "data")
    label_key = args.get("label_key")

    if mode == "single":
        if not args.get("test_img"):
            raise ValueError("single mode requires test_img")
        tiles = discover_single_tile(args["test_img"], data_key=data_key,
                                    data_layout=args.get("data_layout", "HWB"))
        height, width = tiles[0].height, tiles[0].width
        label_map = None
        if args.get("test_label"):
            label_map = load_label_map(args["test_label"], key=label_key)
            label_map = align_label_to_tiles(label_map, tiles)
            label_map, _, notes = normalize_label_map(
                label_map, int(args["num_classes"]), adjust_num_classes=False
            )
            for note in notes:
                print(note)
        return tiles, label_map, height, width

    if mode == "tile":
        if not args.get("tile_dir"):
            raise ValueError("tile mode requires tile_dir")
        tiles = discover_tiles(
            args["tile_dir"],
            int(args.get("tile_w", 600)),
            args.get("tile_pattern", "*"),
            data_key=data_key,
            position_mode=args.get("tile_position_mode", "tile_id"),
            data_layout=args.get("data_layout", "HWB"),
        )
        height = max(tile.height for tile in tiles)
        width = max(tile.start_col + tile.width for tile in tiles)
        label_map = None
        if args.get("label_path"):
            label_map = load_label_map(args["label_path"], key=label_key)
            tiles, label_map, layout_mode = relayout_tiles_for_label(
                tiles,
                label_map,
                int(args.get("tile_w", 600)),
                requested_mode=str(args.get("tile_position_mode", "tile_id")),
            )
            args["tile_position_mode"] = layout_mode
            height = max(tile.height for tile in tiles)
            width = max(tile.start_col + tile.width for tile in tiles)
            label_map, _, notes = normalize_label_map(
                label_map, int(args["num_classes"]), adjust_num_classes=False
            )
            for note in notes:
                print(note)
        return tiles, label_map, height, width

    raise ValueError("test_mode must be 'single' or 'tile'")


def patch_batch(
    padded: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    offsets = np.arange(patch_size, dtype=np.int64)
    x = padded[
        rows[:, None, None] + offsets[None, :, None],
        cols[:, None, None] + offsets[None, None, :],
        :,
    ]
    return x.transpose(0, 3, 1, 2).astype(np.float32, copy=False)


def update_cm(
    cm: np.ndarray,
    gt0: np.ndarray,
    pred0: np.ndarray,
    num_classes: int,
) -> None:
    valid = (
        (gt0 >= 0) & (gt0 < num_classes)
        & (pred0 >= 0) & (pred0 < num_classes)
    )
    if not np.any(valid):
        return
    encoded = gt0[valid] * num_classes + pred0[valid]
    cm += np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def metrics_from_cm(cm: np.ndarray) -> Dict:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    total = int(cm.sum())

    recall = np.divide(
        tp, support,
        out=np.full_like(tp, np.nan),
        where=support > 0,
    )
    precision = np.divide(
        tp, predicted,
        out=np.full_like(tp, np.nan),
        where=predicted > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.full_like(tp, np.nan),
        where=(precision + recall) > 0,
    )
    union = support + predicted - tp
    iou = np.divide(
        tp, union,
        out=np.full_like(tp, np.nan),
        where=union > 0,
    )

    present = support > 0
    oa = float(tp.sum() / max(total, 1))
    aa = float(np.nanmean(recall[present])) if np.any(present) else np.nan
    macro_f1 = float(np.nanmean(f1[present])) if np.any(present) else np.nan
    macro_iou = float(np.nanmean(iou[present])) if np.any(present) else np.nan

    row_sum, col_sum = cm.sum(axis=1), cm.sum(axis=0)
    pe = float(np.dot(row_sum, col_sum)) / max(float(total * total), 1.0)
    kappa = (oa - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else 0.0

    return {
        "OA": oa,
        "AA": aa,
        "Kappa": float(kappa),
        "macro_F1": macro_f1,
        "macro_IoU": macro_iou,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IoU": iou,
        "support": support.astype(np.int64),
        "predicted": predicted.astype(np.int64),
        "correct": tp.astype(np.int64),
        "total": total,
    }


def default_names(k: int) -> List[str]:
    known = [
        "nontronite", "montmorillonite", "none", "chlorite",
        "epidote", "bassanite", "polyhydrated_sulfate",
        "monohydrated_sulfate", "saponite", "pyroxene",
        "kaolinite_halloysite", "alunite", "gypsum", "magnesite",
        "jarosite", "prehnite", "calcite_si--configderite",
        "illite_muscovite", "serpentine", "hydrated_silica",
        "margarite", "olivine", "analcime", "vermiculite",
    ]
    return known if k == 24 else [f"class_{i}" for i in range(1, k + 1)]


def class_names(args: Dict, k: int) -> List[str]:
    names = args.get("class_names")
    if names is None:
        return default_names(k)
    names = [str(x) for x in names]
    if len(names) != k:
        raise ValueError(f"class_names={len(names)}, num_classes={k}")
    return names


def class_tick_labels(
    names: Sequence[str],
    include_background: bool,
) -> List[str]:
    """Build readable colorbar labels such as '1  nontronite'."""
    labels = [
        f"{index + 1}  {str(name).replace('_', ' ')}"
        for index, name in enumerate(names)
    ]
    if include_background:
        return ["0  background"] + labels
    return labels


def save_metrics(
    output_dir: Path,
    prefix: str,
    cm: np.ndarray,
    names: Sequence[str],
) -> Dict:
    metrics = metrics_from_cm(cm)

    with open(
        output_dir / f"{prefix}_confusion_matrix.csv",
        "w", newline="", encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["gt\\pred"]
            + [f"{i+1}:{names[i]}" for i in range(len(names))]
            + ["gt_total"]
        )
        for i, row in enumerate(cm):
            w.writerow([f"{i+1}:{names[i]}"] + row.tolist() + [int(row.sum())])
        w.writerow(["pred_total"] + cm.sum(axis=0).tolist() + [int(cm.sum())])

    with open(
        output_dir / f"{prefix}_metrics.csv",
        "w", newline="", encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in ["OA", "AA", "Kappa", "macro_F1", "macro_IoU", "total"]:
            w.writerow([key, metrics[key]])
        w.writerow([])
        w.writerow([
            "class_id", "class_name", "precision", "recall",
            "F1", "IoU", "correct", "gt_total", "predicted_total",
        ])
        for i, name in enumerate(names):
            w.writerow([
                i + 1, name,
                metrics["precision"][i], metrics["recall"][i],
                metrics["F1"][i], metrics["IoU"][i],
                int(metrics["correct"][i]), int(metrics["support"][i]),
                int(metrics["predicted"][i]),
            ])
    return metrics


def make_cmap(k: int):
    """Return a fixed discrete color map for background + classes 1..k."""
    required = k + 1
    if required > len(BASE_COLORS):
        raise ValueError(
            f"BASE_COLORS provides {len(BASE_COLORS) - 1} class colors, "
            f"but num_classes={k}. Add more colors before testing."
        )

    # Convert all hex values to RGBA tuples so later code can safely replace
    # the background color with white for labeled-region visualization.
    colors = [to_rgba(color) for color in BASE_COLORS[:required]]
    cmap = ListedColormap(
        colors,
        name=f"crism_discrete_{k}_classes",
    )
    norm = BoundaryNorm(
        np.arange(-0.5, k + 1.5, 1.0),
        cmap.N,
    )
    return cmap, norm


def save_class_map(
    array: np.ndarray,
    path: Path,
    title: str,
    k: int,
    dpi: int,
    names: Sequence[str],
    mask_zero: bool = False,
) -> None:
    cmap, norm = make_cmap(k)
    shown = array
    if mask_zero:
        shown = np.ma.masked_where(array == 0, array)
        cmap = cmap.copy()
        cmap.set_bad((0, 0, 0, 1)) # mask 区域纯黑色

    # Leave extra horizontal room for long mineral names on the colorbar.
    fig, ax = plt.subplots(figsize=(15, 8))
    image = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()

    include_background = not mask_zero
    ticks = (
        np.arange(0, k + 1)
        if include_background
        else np.arange(1, k + 1)
    )
    tick_labels = class_tick_labels(
        names,
        include_background=include_background,
    )

    cb = fig.colorbar(
        image,
        ax=ax,
        fraction=0.04,
        pad=0.025,
        ticks=ticks,
    )
    cb.set_ticklabels(tick_labels)
    cb.set_label("Class ID / Class name")
    cb.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_confidence(confidence: np.ndarray, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(
        confidence, cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_title("Maximum softmax confidence")
    ax.set_axis_off()
    cb = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Confidence")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def crop_slices(label: np.ndarray, padding: int):
    rows, cols = np.where(label > 0)
    if len(rows) == 0:
        return slice(None), slice(None)
    return (
        slice(max(int(rows.min()) - padding, 0),
              min(int(rows.max()) + padding + 1, label.shape[0])),
        slice(max(int(cols.min()) - padding, 0),
              min(int(cols.max()) + padding + 1, label.shape[1])),
    )


def save_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    path: Path,
    k: int,
    dpi: int,
    names: Sequence[str],
    crop: bool,
    padding: int,
) -> None:
    valid = (gt >= 1) & (gt <= k)
    if not np.any(valid):
        return

    gt_show = np.where(valid, gt, 0)
    pred_show = np.where(valid, pred, 0)
    correct = np.zeros_like(gt, dtype=np.uint8)
    correct[valid & (gt == pred)] = 1
    correct[valid & (gt != pred)] = 2

    if crop:
        rs, cs = crop_slices(gt_show, padding)
        gt_show, pred_show, correct = gt_show[rs, cs], pred_show[rs, cs], correct[rs, cs]

    cmap, norm = make_cmap(k)
    cmap = cmap.copy()
    corr_cmap = ListedColormap([
        (0, 0, 0, 0), (0.1, 0.65, 0.25, 1), (0.85, 0.15, 0.15, 1)
    ])
    corr_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], corr_cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(23, 8), constrained_layout=True)
    im = axes[0].imshow(gt_show, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("Ground truth on labeled pixels")
    axes[1].imshow(pred_show, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Prediction on labeled pixels")
    axes[2].imshow(correct, cmap=corr_cmap, norm=corr_norm, interpolation="nearest")
    axes[2].set_title("Correctness")
    for ax in axes:
        ax.set_axis_off()

    axes[2].legend(handles=[
        Patch(facecolor=corr_cmap(1), label="Correct"),
        Patch(facecolor=corr_cmap(2), label="Incorrect"),
    ], loc="lower right")
    cb = fig.colorbar(
        im,
        ax=[axes[0], axes[1]],
        fraction=0.04,
        pad=0.025,
        ticks=np.arange(1, k + 1),
    )
    cb.set_ticklabels(
        class_tick_labels(
            names,
            include_background=False,
        )
    )
    cb.set_label("Class ID / Class name")
    cb.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def predict_full(
    model,
    tiles: List[TileMeta],
    label_map: Optional[np.ndarray],
    height: int,
    width: int,
    args: Dict,
    device,
) -> Dict:
    k = int(args["num_classes"])
    patch_size = int(args["patch_size"])
    batch_size = int(args.get("batch_size", 1024))
    threshold = float(args.get("confidence_threshold", 0.0))
    use_amp = bool(args.get("use_amp", True)) and device.type == "cuda"

    raw_map = np.zeros((height, width), dtype=np.int16)
    display_map = np.zeros((height, width), dtype=np.int16)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    cm = np.zeros((k, k), dtype=np.int64)

    prepare_lsga_for_eval(model)
    with torch.inference_mode():
        for tile in tqdm(tiles, desc="Full-image inference"):
            raw = load_mat_data(
                tile.path,
                key=args.get("data_key", "data"),
                prefer_3d=True,
                data_layout=str(args.get("data_layout", "HWB")),
            )
            cube = prepare_tile_cube(
                raw,
                args,
                wavelengths=load_wavelengths(tile.path),
            )
            if cube.shape[-1] != int(args["input_channels"]):
                raise ValueError(
                    f"Channel mismatch: {cube.shape[-1]} vs "
                    f"{args['input_channels']} in {tile.path}"
                )

            pad = patch_size // 2
            padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            n = tile.height * tile.width

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                flat = np.arange(start, end, dtype=np.int64)
                rows = flat // tile.width
                cols = flat % tile.width
                patches = patch_batch(padded, rows, cols, patch_size)
                x = torch.from_numpy(patches).to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    probabilities = torch.softmax(model(x), dim=1)
                    confidence, prediction = probabilities.max(dim=1)

                pred0 = prediction.cpu().numpy()
                pred1 = pred0.astype(np.int16) + 1
                conf = confidence.float().cpu().numpy()
                global_cols = cols + tile.start_col

                raw_map[rows, global_cols] = pred1
                confidence_map[rows, global_cols] = conf
                shown = pred1.copy()
                if threshold > 0:
                    shown[conf < threshold] = 0
                display_map[rows, global_cols] = shown

                if label_map is not None:
                    gt1 = label_map[rows, global_cols]
                    valid = (gt1 >= 1) & (gt1 <= k)
                    if np.any(valid):
                        update_cm(cm, gt1[valid] - 1, pred0[valid], k)

            del raw, cube, padded

    return {
        "raw_prediction": raw_map,
        "display_prediction": display_map,
        "confidence": confidence_map,
        "confusion_matrix": cm,
    }


def save_outputs(
    result: Dict,
    label_map: Optional[np.ndarray],
    args: Dict,
    output_dir: Path,
    scene_name: str,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    k = int(args["num_classes"])
    names = class_names(args, k)
    dpi = int(args.get("visual_dpi", 200))
    raw = result["raw_prediction"]
    shown = result["display_prediction"]
    confidence = result["confidence"]
    cm = result["confusion_matrix"]

    if bool(args.get("save_mat", True)):
        payload = {
            "prediction_raw_1based": raw,
            "prediction_display_1based": shown,
            "confidence": confidence,
            "confidence_threshold": float(args.get("confidence_threshold", 0.0)),
        }
        if label_map is not None:
            payload["gt"] = label_map
            payload["labeled_mask"] = (
                (label_map >= 1) & (label_map <= k)
            ).astype(np.uint8)
            payload["confusion_matrix"] = cm
        savemat(
            output_dir / f"{scene_name}_full_prediction.mat",
            payload,
            do_compression=True,
        )

    if bool(args.get("save_envi", True)):
        from .io import write_envi_class_map

        write_envi_class_map(
            output_dir / f"{scene_name}_full_prediction.img",
            shown,
            names,
        )

    if bool(args.get("save_png", True)):
        save_class_map(
            shown,
            output_dir / f"{scene_name}_full_prediction.png",
            "Full-image prediction",
            k,
            dpi,
            names,
        )
        if bool(args.get("save_confidence_map", True)):
            save_confidence(
                confidence,
                output_dir / f"{scene_name}_full_confidence.png",
                dpi,
            )

    metrics = {"OA": np.nan, "AA": np.nan, "Kappa": np.nan, "total": 0}
    if label_map is not None:
        valid = (label_map >= 1) & (label_map <= k)
        if np.any(valid):
            metrics = save_metrics(
                output_dir,
                f"{scene_name}_labeled_region",
                cm,
                names,
            )
            if bool(args.get("save_png", True)):
                gt_show = np.where(valid, label_map, 0)
                pred_show = np.where(valid, raw, 0)
                save_class_map(
                    gt_show,
                    output_dir / f"{scene_name}_labeled_gt.png",
                    "Ground truth on labeled pixels",
                    k,
                    dpi,
                    names,
                    mask_zero=True,
                )
                save_class_map(
                    pred_show,
                    output_dir / f"{scene_name}_labeled_prediction.png",
                    "Prediction on labeled pixels",
                    k,
                    dpi,
                    names,
                    mask_zero=True,
                )
                save_comparison(
                    label_map,
                    raw,
                    output_dir / f"{scene_name}_labeled_comparison.png",
                    k,
                    dpi,
                    names,
                    bool(args.get("crop_labeled_region", True)),
                    int(args.get("crop_padding", 10)),
                )
            print(
                f"[{scene_name}] labeled samples={metrics['total']} | "
                f"OA={metrics['OA']*100:.2f}% | "
                f"AA={metrics['AA']*100:.2f}% | "
                f"macro_F1={metrics['macro_F1']*100:.2f}% | "
                f"Kappa={metrics['Kappa']:.4f}"
            )
        else:
            print(f"[{scene_name}] no valid labels in 1..{k}.")
    else:
        print(f"[{scene_name}] prediction completed without labels.")
    return metrics


def run_scene(
    base_runtime: Dict,
    scene: Dict,
    checkpoint: Dict,
    checkpoint_path: Path,
) -> Dict:
    runtime = dict(base_runtime)
    runtime.update(scene)
    scene_name = str(
        runtime.get("scene_name")
        or runtime.get("name")
        or Path(runtime.get("test_img") or runtime.get("tile_dir") or "scene").stem
    )
    runtime["scene_name"] = scene_name
    args = merge_checkpoint_args(checkpoint, runtime)
    device = resolve_device(args.get("device", "cpu"))

    root = Path(runtime.get("output_dir") or "./full_image_test")
    output_dir = root / scene_name if len(base_runtime.get("scenes", [])) > 1 else root

    tiles, label_map, height, width = discover_scene(args)
    model = lsga_hsi(args).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    prepare_lsga_for_eval(model)

    print(
        f"\nScene={scene_name} | shape={height}x{width} | "
        f"patch={args['patch_size']} | channels={args['input_channels']} | "
        f"label={'yes' if label_map is not None else 'no'}"
    )
    result = predict_full(model, tiles, label_map, height, width, args, device)
    metrics = save_outputs(result, label_map, args, output_dir, scene_name)
    names = class_names(args, int(args["num_classes"]))
    return {
        "scene_name": scene_name,
        "output_dir": str(output_dir),
        "display_prediction": result["display_prediction"],
        "raw_prediction": result["raw_prediction"],
        "num_classes": int(args["num_classes"]),
        "class_names": names,
        **{
            key: (
                float(value) if isinstance(value, (float, np.floating))
                else int(value) if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in metrics.items()
            if key in {"OA", "AA", "Kappa", "macro_F1", "macro_IoU", "total"}
        },
    }


def main() -> None:
    cli = parser().parse_args()
    runtime = load_json(cli.config)
    checkpoint_path = Path(
        cli.checkpoint_path or runtime.get("checkpoint_path", "")
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(runtime.get("device", "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    scenes = runtime.get("scenes")
    if scenes is None:
        scenes = [runtime]
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes must be a non-empty list.")

    summaries = [
        run_scene(runtime, scene, checkpoint, checkpoint_path)
        for scene in scenes
    ]

    summary_path = Path(
        runtime.get("output_dir", "./full_image_test")
    ) / "full_image_test_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summaries,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
