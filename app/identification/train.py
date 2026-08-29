from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, minimum_filter
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from tqdm import tqdm

from .crism_common import (
    TileMeta,
    attach_labels,
    build_patches_from_prepared,
    discover_tiles,
    extract_tile_points,
    get_band_mask,
    load_json,
    load_label_map,
    load_mat_data,
    normalize_label_map,
    prepare_tile_cube,
    relayout_tiles_for_label,
    resolve_device,
)
from .lsga import lsga_hsi, prepare_lsga_for_eval


SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Leakage-free CRISM merged-image training"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training JSON configuration.",
    )
    return parser


def set_random(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Benchmark is much faster than the old deterministic setting.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def validate_label_ids(
    label_map: np.ndarray,
    num_classes: int,
) -> None:
    positive = np.unique(label_map[label_map > 0])
    if len(positive) == 0:
        raise ValueError("No positive labels were found")

    too_large = positive[positive > num_classes]
    if len(too_large) > 0:
        raise ValueError(
            f"Label map contains class IDs {too_large.tolist()} above "
            f"num_classes={num_classes}. With labels 1..24, set "
            "num_classes=24 unless the labels were explicitly remapped."
        )

    missing = [
        cls for cls in range(1, num_classes + 1)
        if not np.any(label_map == cls)
    ]
    if missing:
        raise ValueError(
            f"Classes missing from merged label map: {missing}"
        )


def class_train_target(
    pixel_count: int,
    args: Dict,
) -> int:
    """Choose a rich but bounded training count for one class."""
    large_threshold = int(
        args.get("large_class_threshold", 500)
    )
    mid_threshold = int(
        args.get("mid_class_threshold", 100)
    )

    if pixel_count > large_threshold:
        return min(
            pixel_count,
            int(args.get("large_class_train_max", 1500)),
        )

    if pixel_count >= mid_threshold:
        ratio = float(
            args.get("mid_class_train_ratio", 0.70)
        )
        optional_cap = int(
            args.get("mid_class_train_max", 0)
        )
        target = max(1, int(round(pixel_count * ratio)))
        if optional_cap > 0:
            target = min(target, optional_cap)
        return min(pixel_count, target)

    ratio = float(
        args.get("rare_class_train_ratio", 0.60)
    )
    return min(
        pixel_count,
        max(1, int(round(pixel_count * ratio))),
    )


def split_targets(
    class_counts: np.ndarray,
    args: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Targets used when choosing held-out spatial blocks."""
    val_target = np.full(
        len(class_counts),
        int(args.get("val_per_class", 20)),
        dtype=np.int64,
    )
    test_target = np.full(
        len(class_counts),
        int(args.get("test_per_class", 100)),
        dtype=np.int64,
    )

    # Do not request more held-out points than a class can reasonably provide,
    # but always leave at least one pixel for validation when the class has
    # two or more labeled pixels.
    for index, count in enumerate(class_counts):
        count = int(count)
        train_min = class_train_target(count, args)
        if count >= 2:
            holdout = 1 if count < 10 else min(
                int(args.get("val_per_class", 20)),
                max(1, count // 10),
            )
            train_min = min(train_min, count - holdout)
        remaining = max(count - train_min, 0)

        val_target[index] = min(
            val_target[index],
            remaining,
        )
        if count >= 2 and val_target[index] < 1:
            val_target[index] = 1
            remaining = max(count - 1, 0)

        remaining_after_val = max(
            remaining - int(val_target[index]),
            0,
        )
        test_target[index] = min(
            test_target[index],
            remaining_after_val,
        )

    return val_target, test_target


def build_block_class_counts(
    label_map: np.ndarray,
    num_classes: int,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Count each class in every global spatial block."""
    points = np.argwhere(
        (label_map >= 1) & (label_map <= num_classes)
    ).astype(np.int64)
    labels = label_map[points[:, 0], points[:, 1]].astype(
        np.int64
    )

    block_rows = int(
        math.ceil(label_map.shape[0] / block_size)
    )
    block_cols = int(
        math.ceil(label_map.shape[1] / block_size)
    )
    block_ids = (
        (points[:, 0] // block_size) * block_cols
        + (points[:, 1] // block_size)
    ).astype(np.int64)

    block_count = block_rows * block_cols
    counts = np.zeros(
        (block_count, num_classes),
        dtype=np.int32,
    )
    np.add.at(counts, (block_ids, labels - 1), 1)

    return (
        points,
        labels,
        counts,
        block_rows,
        block_cols,
    )


def greedy_choose_blocks(
    block_counts: np.ndarray,
    available: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose compact blocks that cover per-class held-out targets."""
    chosen = np.zeros(len(block_counts), dtype=bool)
    current = np.zeros(block_counts.shape[1], dtype=np.int64)
    target_safe = np.maximum(targets, 1)

    while np.any(current < targets):
        need = np.maximum(targets - current, 0)
        useful_classes = need > 0

        candidate_ids = np.where(
            available
            & (~chosen)
            & np.any(
                block_counts[:, useful_classes] > 0,
                axis=1,
            )
        )[0]

        if len(candidate_ids) == 0:
            break

        candidate_counts = block_counts[candidate_ids]
        gains = np.minimum(
            candidate_counts,
            need[None, :],
        )

        normalized_gain = np.sum(
            gains / target_safe[None, :],
            axis=1,
        )

        block_total = np.sum(candidate_counts, axis=1)
        # Prefer blocks that satisfy needs without bringing a huge amount of
        # unrelated held-out data.
        score = normalized_gain / (
            1.0 + 0.002 * block_total
        )
        score += rng.uniform(
            0.0,
            1e-8,
            size=len(score),
        )

        best = candidate_ids[int(np.argmax(score))]
        if score.max() <= 0:
            break

        chosen[best] = True
        current += block_counts[best]

    return chosen


def choose_spatial_blocks(
    block_counts: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    seed: int,
    restarts: int,
) -> Tuple[np.ndarray, Dict]:
    """Search several greedy assignments and keep the best coverage."""
    occupied = np.sum(block_counts, axis=1) > 0
    reserved_train = _blocks_reserved_for_train(block_counts)
    best_assignment = None
    best_info = None
    best_score = -np.inf

    for restart in range(max(1, restarts)):
        rng = np.random.default_rng(
            seed + restart * 1009
        )
        available = occupied & (~reserved_train)

        test_blocks = greedy_choose_blocks(
            block_counts,
            available,
            test_targets,
            rng,
        )
        available = available & (~test_blocks)

        val_blocks = greedy_choose_blocks(
            block_counts,
            available,
            val_targets,
            rng,
        )

        assignment = np.full(
            len(block_counts),
            SPLIT_TRAIN,
            dtype=np.int8,
        )
        assignment[val_blocks] = SPLIT_VAL
        assignment[test_blocks] = SPLIT_TEST
        assignment = repair_block_assignment(assignment, block_counts)

        val_counts = block_counts[assignment == SPLIT_VAL].sum(axis=0)
        test_counts = block_counts[assignment == SPLIT_TEST].sum(axis=0)

        val_coverage = np.minimum(
            val_counts,
            val_targets,
        ) / np.maximum(val_targets, 1)
        test_coverage = np.minimum(
            test_counts,
            test_targets,
        ) / np.maximum(test_targets, 1)

        coverage = (
            float(np.sum(val_coverage))
            + float(np.sum(test_coverage))
        )
        selected_size = int(
            block_counts[assignment != SPLIT_TRAIN].sum()
        )
        score = coverage - 1e-7 * selected_size

        if score > best_score:
            best_score = score
            best_assignment = assignment
            best_info = {
                "val_block_counts_per_class": val_counts,
                "test_block_counts_per_class": test_counts,
                "val_blocks": np.where(assignment == SPLIT_VAL)[0],
                "test_blocks": np.where(assignment == SPLIT_TEST)[0],
                "coverage_score": coverage,
            }

    assert best_assignment is not None
    assert best_info is not None
    return best_assignment, best_info


def _blocks_reserved_for_train(block_counts: np.ndarray) -> np.ndarray:
    """Keep the only spatial block of a rare class in the training split."""
    reserved = np.zeros(len(block_counts), dtype=bool)
    occupied = np.sum(block_counts, axis=1) > 0
    for cls in range(block_counts.shape[1]):
        ids = np.where(occupied & (block_counts[:, cls] > 0))[0]
        if len(ids) == 1:
            reserved[ids[0]] = True
    return reserved


def repair_block_assignment(
    assignment: np.ndarray,
    block_counts: np.ndarray,
) -> np.ndarray:
    """Ensure every present class keeps at least one training block."""
    assignment = np.asarray(assignment, dtype=np.int8).copy()
    for cls in range(block_counts.shape[1]):
        has = block_counts[:, cls] > 0
        if not np.any(has):
            continue
        train_pixels = int(
            block_counts[(assignment == SPLIT_TRAIN) & has, cls].sum()
        )
        if train_pixels > 0:
            continue
        for split in (SPLIT_VAL, SPLIT_TEST):
            ids = np.where(has & (assignment == split))[0]
            if len(ids) == 0:
                continue
            best = ids[int(np.argmax(block_counts[ids, cls]))]
            assignment[best] = SPLIT_TRAIN
            break
    return assignment


def subtract_points(pool: np.ndarray, taken: np.ndarray) -> np.ndarray:
    if len(pool) == 0 or len(taken) == 0:
        return pool
    width_shift = np.int64(1) << 32
    pool_key = pool[:, 0].astype(np.int64) * width_shift + pool[:, 1].astype(
        np.int64
    )
    taken_key = taken[:, 0].astype(np.int64) * width_shift + taken[:, 1].astype(
        np.int64
    )
    return pool[~np.isin(pool_key, taken_key)]


def build_patch_safe_split_map(
    label_shape: Tuple[int, int],
    block_assignment: np.ndarray,
    block_rows: int,
    block_cols: int,
    block_size: int,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand block assignments and remove boundary centers.

    A center is safe only when every pixel in its patch belongs to the same
    global split. Therefore train/val/test patches cannot overlap.
    """
    grid = block_assignment.reshape(
        block_rows,
        block_cols,
    )
    full = np.repeat(
        np.repeat(grid, block_size, axis=0),
        block_size,
        axis=1,
    )
    full = full[
        : label_shape[0],
        : label_shape[1],
    ].astype(np.int8, copy=False)

    local_min = minimum_filter(
        full,
        size=patch_size,
        mode="nearest",
    )
    local_max = maximum_filter(
        full,
        size=patch_size,
        mode="nearest",
    )
    safe = local_min == local_max

    return full, safe


def random_take(
    points: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or len(points) == 0:
        return np.empty((0, 2), dtype=np.int64)
    if len(points) <= count:
        return points.astype(np.int64, copy=True)

    indices = rng.choice(
        len(points),
        size=count,
        replace=False,
    )
    return points[indices].astype(np.int64, copy=True)


def make_spatial_split(
    label_map: np.ndarray,
    num_classes: int,
    args: Dict,
    seed: int,
) -> Dict:
    """Create train, small val, held-out test and full-image test_all.

    ``test_all`` contains every labeled pixel in the entire label map. It
    therefore includes train/val/test samples and must be treated as a
    full-map diagnostic rather than an unbiased held-out metric.
    """
    block_size = int(args.get("spatial_block_size", 32))
    patch_size = int(args["patch_size"])
    restarts = int(args.get("split_search_restarts", 20))

    (
        all_points,
        all_labels,
        block_counts,
        block_rows,
        block_cols,
    ) = build_block_class_counts(
        label_map,
        num_classes,
        block_size,
    )

    class_counts = np.bincount(
        all_labels,
        minlength=num_classes + 1,
    )[1:]
    val_targets, test_targets = split_targets(
        class_counts,
        args,
    )

    block_assignment, block_info = choose_spatial_blocks(
        block_counts,
        val_targets,
        test_targets,
        seed,
        restarts,
    )

    full_split_map, safe_map = build_patch_safe_split_map(
        label_map.shape,
        block_assignment,
        block_rows,
        block_cols,
        block_size,
        patch_size,
    )

    point_splits = full_split_map[
        all_points[:, 0],
        all_points[:, 1],
    ]
    point_safe = safe_map[
        all_points[:, 0],
        all_points[:, 1],
    ]

    rng = np.random.default_rng(seed)

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    stats: List[Dict] = []

    for class_id in range(1, num_classes + 1):
        class_mask = all_labels == class_id
        class_points = all_points[class_mask]
        n_class = int(class_counts[class_id - 1])

        train_safe = class_points[
            point_safe[class_mask] & (point_splits[class_mask] == SPLIT_TRAIN)
        ]
        val_safe = class_points[
            point_safe[class_mask] & (point_splits[class_mask] == SPLIT_VAL)
        ]
        test_safe = class_points[
            point_safe[class_mask] & (point_splits[class_mask] == SPLIT_TEST)
        ]
        train_any = class_points[point_splits[class_mask] == SPLIT_TRAIN]
        val_any = class_points[point_splits[class_mask] == SPLIT_VAL]
        test_any = class_points[point_splits[class_mask] == SPLIT_TEST]

        train_target = class_train_target(n_class, args)
        val_target = int(val_targets[class_id - 1])
        test_target = int(test_targets[class_id - 1])

        train_pool = train_safe if len(train_safe) > 0 else train_any
        val_pool = val_safe if len(val_safe) > 0 else val_any
        test_pool = test_safe if len(test_safe) > 0 else test_any

        train_selected = random_take(
            train_pool,
            min(train_target, len(train_pool)),
            rng,
        )
        val_selected = random_take(
            val_pool,
            min(max(val_target, 1 if n_class >= 2 else 0), len(val_pool)),
            rng,
        )
        test_selected = random_take(
            test_pool,
            min(test_target, len(test_pool)),
            rng,
        )
        val_note = "spatial"

        if len(val_selected) == 0 and n_class >= 2:
            leftover = subtract_points(class_points, train_selected)
            if len(leftover) == 0:
                leftover = class_points
            n_val = min(
                max(val_target, 1),
                max(1, n_class // 5),
                20,
                len(leftover),
            )
            if n_class - n_val < 1:
                n_val = max(1, n_class - 1)
                leftover = class_points
            val_selected = random_take(leftover, n_val, rng)
            train_selected = subtract_points(train_selected, val_selected)
            val_note = "fallback"
            print(
                f"类别 {class_id} 在空间块划分后没有 patch-safe 验证样本"
                f"（该类共 {n_class} 像元，往往挤在很小的区域）。"
                f"已从同类像元中补了 {len(val_selected)} 个验证样本，"
                "训练可以继续。"
            )

        if len(train_selected) == 0 and n_class >= 1:
            leftover = subtract_points(class_points, val_selected)
            if len(leftover) == 0 and n_class == 1:
                leftover = class_points
                val_selected = np.empty((0, 2), dtype=np.int64)
            if len(leftover) > 0:
                train_selected = random_take(
                    leftover,
                    min(max(train_target, 1), len(leftover)),
                    rng,
                )
                print(
                    f"类别 {class_id} 空间划分后没有训练样本，"
                    f"已改从该类 {n_class} 个像元中取 {len(train_selected)} 个用于训练。"
                )

        if n_class >= 2 and (len(train_selected) == 0 or len(val_selected) == 0):
            # Last resort: random split of the class pixels.
            shuffled = class_points.copy()
            rng.shuffle(shuffled)
            n_val = max(1, min(len(shuffled) // 5, 20, len(shuffled) - 1))
            val_selected = shuffled[:n_val]
            train_selected = shuffled[n_val:]
            if train_target < len(train_selected):
                train_selected = train_selected[: max(train_target, 1)]
            val_note = "random"
            print(
                f"类别 {class_id} 改为随机划分："
                f"train={len(train_selected)}, val={len(val_selected)}。"
            )

        train_parts.append(train_selected)
        val_parts.append(val_selected)
        test_parts.append(test_selected)

        stats.append(
            {
                "class_id": class_id,
                "pixel_count": n_class,
                "train_pool": len(train_pool),
                "train_target": train_target,
                "train_count": len(train_selected),
                "val_pool": len(val_pool),
                "val_target": val_target,
                "val_count": len(val_selected),
                "test_pool": len(test_pool),
                "test_target": test_target,
                "test_count": len(test_selected),
                "test_all_count": n_class,
                "val_source": val_note,
            }
        )

        if len(test_pool) == 0 and n_class > 0 and test_target > 0:
            print(
                f"类别 {class_id} 没有独立的空间测试块"
                f"（像元太少或太集中）。内部 test 可能为空，"
                "仍可用 train/val 训练；整图 test_all 会包含该类。"
            )

    def concatenate(parts: List[np.ndarray]) -> np.ndarray:
        nonempty = [part for part in parts if len(part) > 0]
        if not nonempty:
            return np.empty((0, 2), dtype=np.int64)
        output = np.concatenate(nonempty, axis=0)
        rng.shuffle(output)
        return output.astype(np.int64)

    train = concatenate(train_parts)
    val = concatenate(val_parts)
    test = concatenate(test_parts)
    if len(val) == 0 and len(train) > 1:
        rng.shuffle(train)
        n_move = max(1, min(32, len(train) // 10))
        val = train[:n_move].copy()
        train = train[n_move:]
        print(
            f"验证集仍为空，已从训练样本中划出 {len(val)} 个做验证，"
            "以便早停可以计算 val_AA。"
        )

    return {
        "train": train,
        "val": val,
        "test": test,
        # Full-image diagnostic: every labeled pixel, without split-safe
        # filtering and without class balancing.
        "test_all": all_points.astype(np.int64, copy=True),
        "class_stats": stats,
        "block_assignment": block_assignment,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "block_size": block_size,
        "block_info": block_info,
    }


def save_split(
    result_dir: Path,
    seed: int,
    split: Dict,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        result_dir / f"split_seed{seed}.npz",
        train=split["train"],
        val=split["val"],
        test=split["test"],
        test_all=split["test_all"],
        block_assignment=split["block_assignment"],
        block_rows=np.int32(split["block_rows"]),
        block_cols=np.int32(split["block_cols"]),
        block_size=np.int32(split["block_size"]),
    )

    fields = [
        "class_id",
        "pixel_count",
        "train_pool",
        "train_target",
        "train_count",
        "val_pool",
        "val_target",
        "val_count",
        "test_pool",
        "test_target",
        "test_count",
        "test_all_count",
        "val_source",
    ]
    with open(
        result_dir / f"split_statistics_seed{seed}.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(split["class_stats"])


def cache_signature(
    tiles: List[TileMeta],
    points: np.ndarray,
    args: Dict,
) -> str:
    hasher = hashlib.sha1()
    hasher.update(points.astype(np.int64).tobytes())
    hasher.update(str(args["patch_size"]).encode())
    hasher.update(
        json.dumps(
            args.get("exclude_bands_1based", []),
            sort_keys=True,
        ).encode()
    )

    for tile in tiles:
        stat = tile.path.stat()
        hasher.update(str(tile.path.resolve()).encode())
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime_ns).encode())

    return hasher.hexdigest()


def create_patch_cache(
    cache_dir: Path,
    cache_name: str,
    tiles: List[TileMeta],
    points: np.ndarray,
    label_map: np.ndarray,
    args: Dict,
) -> Tuple[Path, Path]:
    """Read every relevant tile once and create a persistent patch memmap."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    patch_path = cache_dir / f"{cache_name}_patches.npy"
    label_path = cache_dir / f"{cache_name}_labels.npy"
    meta_path = cache_dir / f"{cache_name}_meta.json"

    signature = cache_signature(tiles, points, args)

    if (
        not bool(args.get("rebuild_patch_cache", False))
        and patch_path.exists()
        and label_path.exists()
        and meta_path.exists()
    ):
        with open(meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)
        if meta.get("signature") == signature:
            print(f"Reusing patch cache: {cache_name}")
            return patch_path, label_path

    if len(points) == 0:
        raise ValueError(f"No points supplied for cache {cache_name}")

    first_cube = load_mat_data(
        tiles[0].path,
        key=args.get("data_key", "data"),
        prefer_3d=True,
    )
    band_mask = get_band_mask(args, first_cube.shape[-1])
    input_channels = int(np.sum(band_mask))
    patch_size = int(args["patch_size"])

    patches = np.lib.format.open_memmap(
        patch_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(points),
            input_channels,
            patch_size,
            patch_size,
        ),
    )
    labels = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.int64,
        shape=(len(points),),
    )

    write_index = 0
    cache_batch = int(args.get("cache_batch_size", 4096))

    for tile in tqdm(
        tiles,
        desc=f"Building {cache_name} cache",
        leave=False,
    ):
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

        for start in range(0, len(tile_points), cache_batch):
            end = min(
                start + cache_batch,
                len(tile_points),
            )
            batch_points = tile_points[start:end]
            batch_patches = build_patches_from_prepared(
                prepared,
                batch_points,
                tile.start_col,
                patch_size,
            )
            batch_labels = attach_labels(
                batch_points,
                label_map,
            )

            count = len(batch_points)
            patches[
                write_index : write_index + count
            ] = batch_patches
            labels[
                write_index : write_index + count
            ] = batch_labels
            write_index += count

        del raw, prepared

    if write_index != len(points):
        raise RuntimeError(
            f"Cache {cache_name}: wrote {write_index} samples, "
            f"expected {len(points)}"
        )

    patches.flush()
    labels.flush()
    del patches, labels

    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "signature": signature,
                "sample_count": len(points),
                "input_channels": input_channels,
                "patch_size": patch_size,
            },
            file,
            indent=2,
        )

    return patch_path, label_path


class CachedPatchDataset(Dataset):
    """Memory-mapped patches; source tiles are not reread each epoch."""

    def __init__(
        self,
        patch_path: Path,
        label_path: Path,
        augment: bool,
    ):
        self.patch_path = Path(patch_path)
        self.label_path = Path(label_path)
        self.patches = np.load(
            self.patch_path,
            mmap_mode="r",
        )
        self.labels = np.load(
            self.label_path,
            mmap_mode="r",
        )
        self.augment = augment

        if len(self.patches) != len(self.labels):
            raise ValueError("Patch/label cache length mismatch")

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def spatial_augment(x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[1])
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(()) < 0.5:
            k = int(torch.randint(1, 4, ()).item())
            x = torch.rot90(x, k=k, dims=[1, 2])
        return x

    def __getitem__(self, index: int):
        # Copy one sample so rotations/flips never modify the memmap.
        x = torch.from_numpy(
            np.array(self.patches[index], copy=True)
        ).float()
        y = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long,
        )

        if self.augment:
            x = self.spatial_augment(x)

        return x, y


def make_train_loader(
    dataset: CachedPatchDataset,
    args: Dict,
) -> DataLoader:
    batch_size = int(args["batch_size"])
    num_workers = int(args.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    if bool(args.get("use_balanced_sampler", True)):
        labels = np.asarray(dataset.labels, dtype=np.int64)
        class_counts = np.bincount(labels)
        sample_weights = 1.0 / np.maximum(
            class_counts[labels],
            1,
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(
                sample_weights,
                dtype=torch.double,
            ),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                num_workers > 0
                and bool(args.get("persistent_workers", True))
            ),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
            and bool(args.get("persistent_workers", True))
        ),
    )


def make_eval_loader(
    dataset: CachedPatchDataset,
    args: Dict,
) -> DataLoader:
    num_workers = int(args.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(args["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            num_workers > 0
            and bool(args.get("persistent_workers", True))
        ),
    )


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
        else 0.0
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
        "AA": aa,
        "kappa": float(kappa),
        "per_class_acc": per_class,
        "class_total": class_total.astype(np.int64),
        "class_correct": class_correct.astype(np.int64),
    }


def _use_amp(device, args: Optional[Dict] = None) -> bool:
    if getattr(device, "type", None) != "cuda":
        return False
    if args is None:
        return True
    return bool(args.get("use_amp", True))


def _autocast(device, enabled: bool):
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _prepare_model_for_eval(model) -> None:
    prepare_lsga_for_eval(model)


def evaluate_loader(
    model,
    loader: DataLoader,
    device,
    num_classes: int,
    use_amp: bool = False,
) -> Tuple[np.ndarray, Dict]:
    _prepare_model_for_eval(model)
    cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with _autocast(device, use_amp):
                logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            update_confusion(
                cm,
                y.numpy(),
                pred,
                num_classes,
            )

    return cm, metrics_from_confusion(cm)


def evaluate_points_streaming(
    model,
    tiles: List[TileMeta],
    points: np.ndarray,
    label_map: np.ndarray,
    args: Dict,
    device,
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray]:
    """Evaluate full-image test_all by reading each relevant tile once.

    The supplied points may include training and validation pixels. This
    function is intended for complete labeled-map diagnostics and mapping,
    not for unbiased model selection.
    """
    num_classes = int(args["num_classes"])
    patch_size = int(args["patch_size"])
    batch_size = int(args["batch_size"])
    use_amp = _use_amp(device, args)

    cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )
    predictions_parts: List[np.ndarray] = []
    points_parts: List[np.ndarray] = []

    _prepare_model_for_eval(model)
    with torch.no_grad():
        for tile in tqdm(
            tiles,
            desc="Evaluating full-image test_all",
            leave=False,
        ):
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
                with _autocast(device, use_amp):
                    pred = model(x).argmax(dim=1).cpu().numpy()
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
                tile_predictions.append(
                    pred.astype(np.int16)
                )

            predictions_parts.append(
                np.concatenate(tile_predictions)
            )
            points_parts.append(tile_points)
            del raw, prepared

    if predictions_parts:
        predictions = np.concatenate(predictions_parts)
        ordered_points = np.concatenate(points_parts)
    else:
        predictions = np.empty((0,), dtype=np.int16)
        ordered_points = np.empty((0, 2), dtype=np.int64)

    return (
        cm,
        metrics_from_confusion(cm),
        ordered_points,
        predictions,
    )


def save_metrics(
    result_dir: Path,
    prefix: str,
    cm: np.ndarray,
) -> Dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics_from_confusion(cm)

    np.savetxt(
        result_dir / f"{prefix}_confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=",",
    )

    with open(
        result_dir / f"{prefix}_metrics.csv",
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


def train_one_seed(args: Dict, seed: int) -> Dict:
    set_random(seed)

    # Enforce the simplified network input.
    args["norm_mode"] = "none"
    args["use_spectral_features"] = False

    device = resolve_device(args.get("device", 0))
    result_dir = (
        Path(args.get("result_dir", "./result"))
        / str(args["dataset"])
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    tiles = discover_tiles(
        args["tile_dir"],
        int(args.get("tile_w", 600)),
        args.get("tile_pattern", "tile_*.mat"),
        data_key=args.get("data_key", "data"),
        position_mode=args.get(
            "tile_position_mode",
            "tile_id",
        ),
        data_layout=args.get("data_layout", "HWB"),
    )

    label_map = load_label_map(
        args["label_path"],
        key=args.get("label_key"),
    )
    tiles, label_map, layout_mode = relayout_tiles_for_label(
        tiles,
        label_map,
        int(args.get("tile_w", 600)),
        requested_mode=str(args.get("tile_position_mode", "sequential")),
    )
    args["tile_position_mode"] = layout_mode
    print(
        f"Tiles={len(tiles)} | mosaic="
        f"{tiles[0].height}x{max(t.start_col + t.width for t in tiles)} | "
        f"layout={layout_mode}"
    )

    label_map, num_classes, label_notes = normalize_label_map(
        label_map,
        int(args["num_classes"]),
    )
    args["num_classes"] = num_classes
    for note in label_notes:
        print(note)

    split = make_spatial_split(
        label_map,
        num_classes,
        args,
        seed,
    )
    save_split(result_dir, seed, split)

    print(
        f"Split sizes | train={len(split['train'])}, "
        f"val={len(split['val'])}, "
        f"test={len(split['test'])}, "
        f"test_all={len(split['test_all'])}"
    )
    if len(split["val"]) > 0:
        val_labels = label_map[
            split["val"][:, 0],
            split["val"][:, 1],
        ]
        val_hist = np.bincount(val_labels, minlength=num_classes + 1)
        majority = int(val_hist[1:].max()) if num_classes > 0 else 0
        print(
            f"验证集多数类占比 {100.0 * majority / max(len(split['val']), 1):.1f}% "
            f"（只猜这一类也能到这个 OA）。"
            f"24 类随机约 {100.0 / max(num_classes, 1):.1f}%。"
            "日志里的 train=% 是平衡抽样+增强后的训练批次精度，"
            "不能当成分类图精度；请看 val_OA / val_AA。"
        )

    use_amp = _use_amp(device, args)
    print(
        f"Device={device} | AMP={'on' if use_amp else 'off'} | "
        f"batch={int(args['batch_size'])} | epochs={int(args.get('epochs', 130))}"
    )
    if device.type == "cpu":
        print(
            "当前在用 CPU，会很慢。有 NVIDIA 显卡时请把设备改成 cuda:0，"
            "batch size 建议 128 或 256。"
        )

    cache_dir = (
        result_dir
        / str(args.get("cache_subdir", "patch_cache"))
        / f"seed{seed}"
    )

    train_paths = create_patch_cache(
        cache_dir,
        "train",
        tiles,
        split["train"],
        label_map,
        args,
    )
    if len(split["val"]) == 0:
        raise RuntimeError(
            "验证集为空，无法训练。请检查标签是否与影像对齐、各类是否有足够像元。"
        )
    val_paths = create_patch_cache(
        cache_dir,
        "val",
        tiles,
        split["val"],
        label_map,
        args,
    )
    train_set = CachedPatchDataset(
        *train_paths,
        augment=bool(args.get("spatial_augment", True)),
    )
    val_set = CachedPatchDataset(
        *val_paths,
        augment=False,
    )

    args["input_channels"] = int(
        train_set.patches.shape[1]
    )
    # lsga_hsi keeps pca_num as a backward-compatible input-channel key.
    args["pca_num"] = args["input_channels"]

    train_loader = make_train_loader(train_set, args)
    val_loader = make_eval_loader(val_set, args)
    test_loader = None
    if len(split["test"]) > 0:
        test_paths = create_patch_cache(
            cache_dir,
            "test",
            tiles,
            split["test"],
            label_map,
            args,
        )
        test_set = CachedPatchDataset(
            *test_paths,
            augment=False,
        )
        test_loader = make_eval_loader(test_set, args)
    else:
        print("内部 held-out test 为空，跳过测试缓存。")

    model = lsga_hsi(args).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.get("lr", 5e-4)),
        weight_decay=float(
            args.get("weight_decay", 1e-2)
        ),
    )
    epochs = int(args.get("epochs", 130))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(args.get("min_lr", 1e-6)),
    )

    label_smoothing = float(
        args.get("label_smoothing", 0.02)
    )
    grad_clip = float(args.get("grad_clip", 5.0))
    early_patience = int(
        args.get("early_stopping_patience", 30)
    )

    best_aa = -np.inf
    epochs_without_improvement = 0
    best_path = (
        result_dir
        / f"{args['dataset']}_seed{seed}_best.pth"
    )
    history: List[Dict] = []

    scaler = _grad_scaler(use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        sample_count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = model(x)
                loss = F.cross_entropy(
                    logits,
                    y,
                    label_smoothing=label_smoothing,
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )
                optimizer.step()

            loss_sum += float(loss.item()) * len(y)
            correct += int(
                (logits.argmax(dim=1) == y).sum().item()
            )
            sample_count += len(y)

        scheduler.step()

        # val_loader was built once before the epoch loop. No tile is reread.
        val_cm, val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            num_classes,
            use_amp=use_amp,
        )

        train_loss = loss_sum / max(sample_count, 1)
        train_acc = correct / max(sample_count, 1)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_OA": val_metrics["OA"],
            "val_AA": val_metrics["AA"],
            "val_kappa": val_metrics["kappa"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.5f} | "
            f"train={train_acc * 100:.2f}% | "
            f"val_OA={val_metrics['OA'] * 100:.2f}% | "
            f"val_AA={val_metrics['AA'] * 100:.2f}% | "
            f"kappa={val_metrics['kappa']:.4f}"
        )
        chance = 1.0 / max(num_classes, 1)
        if epoch == 1:
            pred_hist = val_cm.sum(axis=0)
            gt_hist = val_cm.sum(axis=1)
            total = max(int(val_cm.sum()), 1)
            top_pred = int(np.argmax(pred_hist)) + 1
            print(
                f"val 像元={total} | 模型最常预测第 {top_pred} 类 "
                f"({100.0 * pred_hist.max() / total:.1f}%) | "
                f"验证标签最多的类占比 {100.0 * gt_hist.max() / total:.1f}%"
            )
        if epoch in (3, 8) and train_acc < chance + 0.08:
            print(
                "训练精度仍接近随机。请检查："
                "① 标签是否与影像空间对齐；"
                "② 立方体是否为 I/F（0–1），而不是 DN 或 I/F×10000；"
                "③ 「类别数」是否等于标签中的 1..K；"
                "④ 数据排布 HWB/BHW 是否选对。"
            )
        elif epoch in (3, 4, 8) and val_metrics["AA"] <= chance + 0.03 and train_acc > 0.4:
            print(
                "train 高、val 接近随机：模型在记训练 patch 的位置/纹理，"
                "没有学到能推广到其他区域的矿物光谱。"
                "val_AA≈1/类别数 属于空间划分下的真实泛化结果，不是训练没跑起来。"
                "可继续看到 20–40 轮；若 val_AA 一直不变，需要更多不同区域的标注，"
                "而不是再加大 batch。"
            )

        if val_metrics["AA"] > best_aa:
            best_aa = val_metrics["AA"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": args,
                    "seed": seed,
                    "best_val_AA": float(best_aa),
                    "split_path": str(
                        result_dir / f"split_seed{seed}.npz"
                    ),
                    "train_source": "merged_img_spatial_blocks",
                },
                best_path,
            )
            save_metrics(
                result_dir,
                f"val_seed{seed}_best",
                val_cm,
            )
        else:
            epochs_without_improvement += 1

        if (
            early_patience > 0
            and epochs_without_improvement >= early_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"no val_AA improvement for {early_patience} epochs."
            )
            break

    with open(
        result_dir / f"train_history_seed{seed}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)

    # Final internal tests use only the best checkpoint.
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _prepare_model_for_eval(model)

    if test_loader is not None:
        test_cm, test_metrics = evaluate_loader(
            model,
            test_loader,
            device,
            num_classes,
            use_amp=use_amp,
        )
        save_metrics(
            result_dir,
            f"test_seed{seed}",
            test_cm,
        )
        print(
            "Balanced held-out test | "
            f"OA={test_metrics['OA'] * 100:.2f}% | "
            f"AA={test_metrics['AA'] * 100:.2f}%"
        )
    else:
        print("内部 held-out test 为空，跳过 test 指标。")

    if bool(args.get("eval_full_map", False)):
        (
            test_all_cm,
            test_all_metrics,
            test_all_points,
            test_all_predictions,
        ) = evaluate_points_streaming(
            model,
            tiles,
            split["test_all"],
            label_map,
            args,
            device,
        )
        save_metrics(
            result_dir,
            f"test_all_seed{seed}",
            test_all_cm,
        )
        np.savez_compressed(
            result_dir
            / f"test_all_predictions_seed{seed}.npz",
            points=test_all_points,
            predictions_0based=test_all_predictions,
            predictions_1based=(
                test_all_predictions.astype(np.int64) + 1
            ),
            ground_truth_1based=label_map[
                test_all_points[:, 0],
                test_all_points[:, 1],
            ] if len(test_all_points) else np.empty(
                (0,), dtype=np.int64
            ),
        )
        print(
            "Full-image labeled test_all (includes training pixels) | "
            f"OA={test_all_metrics['OA'] * 100:.2f}% | "
            f"AA={test_all_metrics['AA'] * 100:.2f}%"
        )
    else:
        print(
            "已跳过整图像元诊断 test_all（很慢）。"
            "分类图请用菜单「模型应用」；若需要训练结束时的全图指标，"
            "可在配置中设置 eval_full_map=true。"
        )

    print(f"Best checkpoint: {best_path}")
    return args


def main() -> None:
    cli = setup_parser().parse_args()
    args = load_json(cli.config)

    if str(args.get("mode", "train")).lower() != "train":
        raise ValueError(
            "main.py is training-only. Use test.py for external scenes."
        )

    seeds = args.get("seed_list", [0])
    if isinstance(seeds, int):
        seeds = [seeds]

    for seed in seeds:
        print(
            f"\n========== Training seed={int(seed)} =========="
        )
        train_one_seed(args, int(seed))


if __name__ == "__main__":
    main()
