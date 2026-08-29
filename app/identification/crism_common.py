"""Shared CRISM utilities for training and external inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from .io import load_array, load_cube, load_label_array, list_input_files, natural_sort_key


@dataclass(frozen=True)
class TileMeta:
    path: Path
    tile_id: int
    start_col: int
    width: int
    height: int
    bands: int


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_device(device_cfg):
    import torch

    if isinstance(device_cfg, int):
        if device_cfg >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{device_cfg}")
        return torch.device("cpu")

    if isinstance(device_cfg, str):
        if device_cfg.lower() == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        value = device_cfg
        if value.startswith("cuda:"):
            return torch.device(value)
        return torch.device(f"cuda:{value}")

    if isinstance(device_cfg, Sequence) and len(device_cfg) > 0:
        return resolve_device(device_cfg[0])

    return torch.device("cpu")


def load_mat_data(
    path: str | Path,
    *,
    key: Optional[str] = "data",
    prefer_3d: bool = True,
    data_layout: str = "HWB",
) -> np.ndarray:
    """Load a cube or array; .mat/.img/.dat/ENVI/PDS/NumPy/TIFF are accepted."""
    if prefer_3d:
        try:
            return load_cube(
                path,
                key=key,
                data_layout=data_layout,
                prefer_3d=True,
            )
        except ValueError:
            array = np.squeeze(load_array(path, key=key, prefer_3d=False))
            if array.ndim == 3:
                return load_cube(
                    path,
                    key=key,
                    data_layout=data_layout,
                    prefer_3d=True,
                )
            raise
    return load_array(path, key=key, prefer_3d=False)


def parse_tile_id(path: Path, default_id: int) -> int:
    """Parse the last numeric token in a filename."""
    for token in reversed(path.stem.replace("-", "_").split("_")):
        if token.isdigit():
            return int(token)
    return default_id


def discover_tiles(
    tile_dir: str | Path,
    tile_width: int,
    pattern: str,
    *,
    data_key: Optional[str] = "data",
    position_mode: str = "tile_id",
    data_layout: str = "HWB",
) -> List[TileMeta]:
    """Discover preprocessed tiles and their coordinates in merged_img."""
    tile_dir = Path(tile_dir)
    try:
        path_strs = list_input_files(tile_dir, pattern)
    except FileNotFoundError:
        paths = sorted(tile_dir.glob(pattern))
        path_strs = [str(p) for p in paths]
    if not path_strs:
        raise FileNotFoundError(
            f"No files matching {pattern!r} in {tile_dir}"
        )
    path_strs = sorted(path_strs, key=natural_sort_key)

    tiles: List[TileMeta] = []
    for order, path_str in enumerate(path_strs):
        path = Path(path_str)
        cube = load_mat_data(
            path,
            key=data_key,
            prefer_3d=True,
            data_layout=data_layout,
        )
        if cube.ndim != 3:
            raise ValueError(f"Expected [H,W,B], got {cube.shape}: {path}")

        height, width, bands = cube.shape
        tiles.append(
            TileMeta(
                path=path,
                tile_id=parse_tile_id(path, order),
                start_col=0,
                width=width,
                height=height,
                bands=bands,
            )
        )

    tiles = sorted(tiles, key=lambda tile: (tile.tile_id, natural_sort_key(tile.path)))
    return apply_tile_positions(tiles, position_mode, int(tile_width))


def apply_tile_positions(
    tiles: List[TileMeta],
    position_mode: str,
    tile_width: int,
) -> List[TileMeta]:
    """Assign mosaic column offsets for a tile list."""
    if not tiles:
        return tiles

    mode = str(position_mode or "sequential").lower()
    if mode == "sequential":
        cumulative = 0
        laid: List[TileMeta] = []
        for tile in tiles:
            laid.append(replace(tile, start_col=cumulative))
            cumulative += tile.width
        return laid

    if mode == "tile_id":
        return [
            replace(tile, start_col=int(tile.tile_id) * int(tile_width))
            for tile in tiles
        ]

    raise ValueError("tile_position_mode must be 'tile_id' or 'sequential'")


def mosaic_shape(tiles: List[TileMeta]) -> Tuple[int, int]:
    if not tiles:
        return (0, 0)
    height = tiles[0].height
    width = max(tile.start_col + tile.width for tile in tiles)
    return height, width


def relayout_tiles_for_label(
    tiles: List[TileMeta],
    label_map: np.ndarray,
    tile_width: int,
    requested_mode: str = "sequential",
) -> Tuple[List[TileMeta], np.ndarray, str]:
    """
    Choose tile column layout so the mosaic matches the label map.

    Filename lexicographic order (tile_10 before tile_2) used to scramble
    mosaics and yield near-chance accuracy. Try sequential and tile_id
    layouts, including a transpose of the label if needed.
    """
    if not tiles:
        return tiles, label_map, requested_mode

    label = np.asarray(label_map)
    modes = []
    for mode in (requested_mode, "sequential", "tile_id"):
        if mode not in modes:
            modes.append(mode)

    for mode in modes:
        laid = apply_tile_positions(tiles, mode, tile_width)
        height, width = mosaic_shape(laid)
        if label.shape == (height, width):
            if mode != requested_mode:
                print(
                    f"Tile layout adjusted: {requested_mode} -> {mode} "
                    f"to match label {tuple(label.shape)}"
                )
            return laid, label, mode
        if label.T.shape == (height, width):
            print(
                f"Label map appears transposed; using label.T "
                f"with tile layout '{mode}'"
            )
            return laid, label.T, mode

    height, width = mosaic_shape(
        apply_tile_positions(tiles, requested_mode, tile_width)
    )
    raise ValueError(
        "Label/tile shape mismatch: "
        f"label={tuple(label.shape)}, mosaic=({height}, {width}). "
        "若训练的是拼接后的多块 tile，请确认文件按 tile 编号排序，"
        "且标签图是整幅拼接影像而不是单块。"
    )


def discover_single_tile(
    image_path: str | Path,
    *,
    data_key: Optional[str] = "data",
    data_layout: str = "HWB",
) -> List[TileMeta]:
    path = Path(image_path)
    cube = load_mat_data(
        path,
        key=data_key,
        prefer_3d=True,
        data_layout=data_layout,
    )
    if cube.ndim != 3:
        raise ValueError(f"Expected [H,W,B], got {cube.shape}: {path}")

    height, width, bands = cube.shape
    return [
        TileMeta(
            path=path,
            tile_id=0,
            start_col=0,
            width=width,
            height=height,
            bands=bands,
        )
    ]


def load_label_map(
    path: str | Path,
    *,
    key: Optional[str] = None,
) -> np.ndarray:
    """Load a 2D label map from .mat / ENVI / IMG / DAT / NumPy / TIFF."""
    return load_label_array(path, key=key or None)


def align_label_to_tiles(
    label_map: np.ndarray,
    tiles: List[TileMeta],
) -> np.ndarray:
    if not tiles:
        return label_map

    expected_height = tiles[0].height
    expected_width = max(
        tile.start_col + tile.width for tile in tiles
    )

    if label_map.shape == (expected_height, expected_width):
        return label_map

    if label_map.T.shape == (expected_height, expected_width):
        print("Label map appears transposed; using label.T")
        return label_map.T

    raise ValueError(
        "Label/tile shape mismatch: "
        f"label={label_map.shape}, expected="
        f"({expected_height}, {expected_width})"
    )


def normalize_label_map(
    label_map: np.ndarray,
    num_classes: int,
    *,
    adjust_num_classes: bool = True,
) -> Tuple[np.ndarray, int, List[str]]:
    """Fix common label encodings that otherwise yield near-chance accuracy.

    - 255 / 65535 nodata -> 0
    - contiguous 0..K-1 class ids (0 is a real class, not background) -> 1..K
    - GUI default K=24 when the map only contains 1..K' -> use K'
    """
    messages: List[str] = []
    lab = np.asarray(label_map, dtype=np.int64)
    requested_k = int(num_classes)

    for nodata in (255, 254, 65535, -1, -9999):
        if np.any(lab == nodata):
            lab = lab.copy()
            lab[lab == nodata] = 0
            messages.append(f"将 nodata={nodata} 视为背景 0")

    unique = np.unique(lab)
    nonnegative = unique[unique >= 0]
    positive = unique[unique > 0]
    if len(positive) == 0:
        raise ValueError("标签图中没有大于 0 的类别，无法训练。")

    frac0 = float(np.mean(lab == 0))
    max_nonneg = int(nonnegative.max()) if len(nonnegative) else 0
    contiguous_from_zero = (
        len(nonnegative) == max_nonneg + 1
        and set(int(v) for v in nonnegative) == set(range(max_nonneg + 1))
    )

    # 0-based classes: unique ids are exactly 0..K-1 and 0 is not a huge
    # unlabeled region.
    if (
        contiguous_from_zero
        and max_nonneg >= 1
        and frac0 < 0.40
        and (
            max_nonneg == requested_k - 1
            or requested_k == 24
        )
    ):
        lab = lab + 1
        new_k = max_nonneg + 1
        messages.append(
            f"标签编号为 0..{max_nonneg}（0 也是类别，不是背景），"
            f"已自动 +1 映射为 1..{new_k}。"
        )
        requested_k = new_k
        positive = np.unique(lab[lab > 0])

    max_id = int(positive.max())
    too_large = positive[positive > requested_k]
    if len(too_large) > 0:
        raise ValueError(
            f"标签含有类别 {too_large.tolist()}，超过当前类别数 "
            f"num_classes={requested_k}。请把对话框中的「类别数」改成 "
            f"{max_id}，或把标签重映射为 1..K（0=背景）。"
        )

    if max_id < requested_k:
        missing = [
            cls for cls in range(1, requested_k + 1)
            if not np.any(lab == cls)
        ]
        if adjust_num_classes:
            messages.append(
                f"标签最大编号为 {max_id}，但类别数设为 {requested_k}。"
                f"缺少 {missing}。已将类别数改为 {max_id}，否则多出来的空类"
                f"会把精度拉到接近随机（1/{requested_k} ≈ {100.0 / requested_k:.1f}%）。"
            )
            requested_k = max_id
        else:
            messages.append(
                f"标签最大编号为 {max_id}，模型类别数为 {requested_k}，"
                f"缺少 {missing}。指标只在标签出现的类上计算。"
            )

    still_missing = [
        cls for cls in range(1, requested_k + 1)
        if not np.any(lab == cls)
    ]
    if still_missing:
        messages.append(
            f"警告：标签仍缺少类别 {still_missing}，这些类不会被学习。"
        )

    counts = np.bincount(lab.ravel(), minlength=requested_k + 1)
    parts = [
        f"{cls}:{int(counts[cls])}"
        for cls in range(1, requested_k + 1)
        if counts[cls] > 0
    ]
    messages.append(
        "各类像元数：" + ", ".join(parts[:24])
        + (" ..." if len(parts) > 24 else "")
    )
    messages.append(
        f"随机猜的总体精度约为 {100.0 / requested_k:.1f}% "
        f"（1/{requested_k}）。若结果只略高于此，优先检查标签对齐与 I/F 数值范围。"
    )
    return lab, requested_k, messages


def all_labeled_points(
    label_map: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    mask = (label_map >= 1) & (label_map <= num_classes)
    return np.argwhere(mask).astype(np.int64)


def all_image_points(height: int, width: int) -> np.ndarray:
    rows, cols = np.indices((height, width), dtype=np.int64)
    return np.stack([rows.ravel(), cols.ravel()], axis=1)


def extract_tile_points(
    points: np.ndarray,
    start_col: int,
    width: int,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int64)

    keep = (
        (points[:, 1] >= start_col)
        & (points[:, 1] < start_col + width)
    )
    return points[keep]


def get_band_mask(args: Dict, bands: int) -> np.ndarray:
    """Build the manually controlled network input band mask."""
    if bands <= 0:
        raise ValueError("bands must be positive")

    mask = np.ones(bands, dtype=bool)

    explicit_mask = args.get("use_band_mask")
    if explicit_mask is not None:
        explicit_mask = np.asarray(explicit_mask).reshape(-1).astype(bool)
        if len(explicit_mask) != bands:
            raise ValueError(
                f"use_band_mask length={len(explicit_mask)}, bands={bands}"
            )
        mask &= explicit_mask

    excluded = args.get("exclude_bands_1based", [])
    if excluded is None:
        excluded = []

    for band in excluded:
        band = int(band)
        if not 1 <= band <= bands:
            raise ValueError(
                f"Excluded band {band} is outside 1..{bands}"
            )
        mask[band - 1] = False

    if not np.any(mask):
        raise ValueError("All spectral bands were excluded")

    return mask


def prepare_tile_cube(
    tile_data: np.ndarray,
    args: Dict,
) -> np.ndarray:
    """Apply the exact network-input path shared by train and test."""
    norm_mode = str(args.get("norm_mode", "none")).lower()
    if norm_mode != "none":
        raise ValueError(
            "This version expects preprocessed L2 spectra and requires "
            "norm_mode='none'."
        )

    if bool(args.get("use_spectral_features", False)):
        raise ValueError(
            "Extra engineered spectral features are disabled in this version."
        )

    cube = tile_data.astype(np.float32, copy=False)
    if cube.ndim != 3:
        raise ValueError(f"Expected [H,W,B], got {cube.shape}")

    if not np.all(np.isfinite(cube)):
        raise ValueError(
            "Non-finite values remain in a preprocessed cube. "
            "Run preprocessing again before training/testing."
        )

    band_mask = get_band_mask(args, cube.shape[-1])
    return cube[:, :, band_mask].astype(np.float32, copy=False)


def mirror_hsi(cube: np.ndarray, patch_size: int) -> np.ndarray:
    pad = patch_size // 2
    return np.pad(
        cube,
        ((pad, pad), (pad, pad), (0, 0)),
        mode="reflect",
    )


def build_patches_from_prepared(
    prepared_cube: np.ndarray,
    global_points: np.ndarray,
    start_col: int,
    patch_size: int,
) -> np.ndarray:
    """Extract [N,C,P,P] patches from an already prepared cube."""
    bands = prepared_cube.shape[-1]

    if len(global_points) == 0:
        return np.empty(
            (0, bands, patch_size, patch_size),
            dtype=np.float32,
        )

    local = global_points.astype(np.int64, copy=True)
    local[:, 1] -= int(start_col)

    if (
        np.any(local[:, 0] < 0)
        or np.any(local[:, 0] >= prepared_cube.shape[0])
        or np.any(local[:, 1] < 0)
        or np.any(local[:, 1] >= prepared_cube.shape[1])
    ):
        raise IndexError("One or more points fall outside the tile")

    mirrored = mirror_hsi(prepared_cube, patch_size)
    patches = np.empty(
        (len(local), bands, patch_size, patch_size),
        dtype=np.float32,
    )

    for index, (row, col) in enumerate(local):
        patches[index] = mirrored[
            row : row + patch_size,
            col : col + patch_size,
            :,
        ].transpose(2, 0, 1)

    return patches


def build_patches(
    tile_data: np.ndarray,
    global_points: np.ndarray,
    start_col: int,
    patch_size: int,
    args: Dict,
) -> np.ndarray:
    prepared = prepare_tile_cube(tile_data, args)
    return build_patches_from_prepared(
        prepared,
        global_points,
        start_col,
        patch_size,
    )


def attach_labels(
    points: np.ndarray,
    label_map: np.ndarray,
) -> np.ndarray:
    """Return zero-based class labels."""
    return (
        label_map[points[:, 0], points[:, 1]] - 1
    ).astype(np.int64)
