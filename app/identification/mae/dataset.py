"""Windowed CRISM crops for MAE pretrain / fine-tune (no full-cube load)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from identification.bands import cube_to_identification_range
from identification.crism_common import (
    TileMeta,
    discover_tiles,
    extract_tile_points,
    resolve_tiles_and_label_map,
    normalize_label_map,
    all_labeled_points,
)
from identification.io import (
    list_input_files,
    load_cube_window,
    load_wavelengths,
    probe_cube_shape,
)
from identification.preprocess import l2_normalize_cube, prepare_identification_cube

from .defaults import MAE_BANDS, MAE_CROP


def prepare_mae_cube(
    cube: np.ndarray,
    wavelengths=None,
    mode: str = "crop",
    source_name: str = "",
) -> np.ndarray:
    """Return H×W×240 float32. ``crop`` skips spatial despike for unlabeled speed."""
    if mode == "full":
        processed, _, _ = prepare_identification_cube(
            cube, wavelengths, source_name=source_name
        )
        return processed
    cropped, _wl = cube_to_identification_range(cube, wavelengths)
    fill = 1e-4
    cropped = np.nan_to_num(cropped, nan=fill, posinf=fill, neginf=fill)
    cropped = np.clip(cropped, 0.0, None)
    out = l2_normalize_cube(cropped, unresolved_bad_mask=None, fill_value=fill)
    return out.astype(np.float32, copy=False)


def _reflect_crop(
    cube: np.ndarray, row: int, col: int, size: int
) -> np.ndarray:
    h, w, _ = cube.shape
    half = size // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="reflect")
    r0 = int(row)
    c0 = int(col)
    return np.ascontiguousarray(padded[r0 : r0 + size, c0 : c0 + size])


class UnlabeledWindowDataset(Dataset):
    """Each item is a random crop from a random unlabeled cube file."""

    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        crop: int = MAE_CROP,
        samples_per_epoch: int = 2048,
        data_key=None,
        data_layout: str = "HWB",
        preprocess_mode: str = "crop",
        seed: int = 0,
    ):
        self.files = [Path(p) for p in files]
        if not self.files:
            raise FileNotFoundError("预训练没有立方体文件。")
        self.crop = int(crop)
        self.samples_per_epoch = int(samples_per_epoch)
        self.data_key = data_key
        self.data_layout = data_layout
        self.preprocess_mode = preprocess_mode
        self.rng = np.random.default_rng(seed)
        self.meta: List[Tuple[Path, int, int, int]] = []
        skipped = 0
        for path in self.files:
            try:
                h, w, b = probe_cube_shape(
                    path, key=data_key, data_layout=data_layout
                )
            except Exception:
                skipped += 1
                continue
            if h < 8 or w < 8:
                skipped += 1
                continue
            self.meta.append((path, h, w, b))
        if not self.meta:
            raise ValueError("没有可读取尺寸的立方体（需要 ENVI/PDS/.npy 头信息）。")
        self.skipped = skipped

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_crop(self, path: Path, height: int, width: int) -> np.ndarray:
        crop = self.crop
        row = int(self.rng.integers(0, max(1, height)))
        col = int(self.rng.integers(0, max(1, width)))
        half = crop // 2 + 4
        r0 = max(0, row - half)
        r1 = min(height, row + half)
        c0 = max(0, col - half)
        c1 = min(width, col + half)
        raw = load_cube_window(
            path, r0, r1, c0, c1,
            key=self.data_key, data_layout=self.data_layout,
        )
        prepared = prepare_mae_cube(
            raw,
            load_wavelengths(path),
            mode=self.preprocess_mode,
            source_name=path.name,
        )
        local_r = min(max(row - r0, 0), prepared.shape[0] - 1)
        local_c = min(max(col - c0, 0), prepared.shape[1] - 1)
        crop_arr = _reflect_crop(prepared, local_r, local_c, crop)
        if crop_arr.shape[0] != crop or crop_arr.shape[1] != crop:
            raise RuntimeError(f"crop shape {crop_arr.shape} from {path}")
        if crop_arr.shape[-1] != MAE_BANDS:
            raise ValueError(
                f"{path.name} 预处理后波段为 {crop_arr.shape[-1]}，需要 {MAE_BANDS}"
            )
        return crop_arr

    def __getitem__(self, index: int) -> torch.Tensor:
        path, height, width, _bands = self.meta[int(self.rng.integers(0, len(self.meta)))]
        cube = self._load_crop(path, height, width)
        return torch.from_numpy(np.ascontiguousarray(cube, dtype=np.float32))


def collect_labeled_records(
    data_path: str | Path,
    label_path: str | Path,
    *,
    num_classes: int,
    data_key=None,
    label_key=None,
    data_layout: str = "HWB",
    input_pattern: str = "*",
    max_per_class: int = 0,
    seed: int = 0,
) -> Tuple[List[TileMeta], np.ndarray, np.ndarray]:
    """Return tiles, label mosaic, and labeled points (row, col) 0-based class later."""
    data_path = Path(data_path)
    if data_path.is_file():
        tile_dir, pattern, mode = data_path.parent, data_path.name, "sequential"
    else:
        tile_dir, pattern, mode = data_path, input_pattern, "sequential"
    tiles = discover_tiles(
        tile_dir,
        tile_width=600,
        pattern=pattern,
        data_key=data_key,
        position_mode=mode,
        data_layout=data_layout,
    )
    tiles, label_map, _layout = resolve_tiles_and_label_map(
        tiles,
        label_path,
        600,
        requested_mode=mode,
        label_key=label_key,
    )
    label_map, num_classes, notes = normalize_label_map(
        label_map, int(num_classes), adjust_num_classes=False
    )
    for note in notes:
        print(note)
    points = all_labeled_points(label_map, int(num_classes))
    if max_per_class and int(max_per_class) > 0:
        rng = np.random.default_rng(seed)
        kept = []
        labels = label_map[points[:, 0], points[:, 1]]
        for cls in range(1, int(num_classes) + 1):
            idx = np.where(labels == cls)[0]
            if idx.size == 0:
                continue
            take = min(int(max_per_class), int(idx.size))
            chosen = rng.choice(idx, size=take, replace=False)
            kept.append(points[chosen])
        if not kept:
            raise ValueError("按每类上限抽样后没有标注像元。")
        points = np.concatenate(kept, axis=0)
        print(f"少样本：每类最多 {max_per_class}，共 {len(points)} 个像元")
    return tiles, label_map, points.astype(np.int64)


class LabeledCropDataset(Dataset):
    """32×32 crops centered on labeled pixels, plus 4×4 block labels."""

    def __init__(
        self,
        tiles: List[TileMeta],
        label_map: np.ndarray,
        points: np.ndarray,
        *,
        crop: int = MAE_CROP,
        spatial_patch: int = 8,
        data_key=None,
        data_layout: str = "HWB",
        preprocess_mode: str = "full",
        num_classes: int = 24,
    ):
        self.tiles = tiles
        self.label_map = np.asarray(label_map)
        self.points = np.asarray(points, dtype=np.int64)
        self.crop = int(crop)
        self.spatial_patch = int(spatial_patch)
        self.data_key = data_key
        self.data_layout = data_layout
        self.preprocess_mode = preprocess_mode
        self.num_classes = int(num_classes)
        self._owners = self._assign_tiles()

    def _assign_tiles(self) -> np.ndarray:
        owners = np.full(len(self.points), -1, dtype=np.int32)
        lookup = {
            (int(row), int(col)): index
            for index, (row, col) in enumerate(self.points.tolist())
        }
        for i, tile in enumerate(self.tiles):
            local = extract_tile_points(self.points, tile.start_col, tile.width)
            for row, col in local.tolist():
                owners[lookup[(int(row), int(col))]] = i
        if np.any(owners < 0):
            raise RuntimeError("部分标注像元没有落到任何 tile 上")
        return owners

    def __len__(self) -> int:
        return len(self.points)

    def _block_labels(self, window_labels: np.ndarray) -> np.ndarray:
        n = self.crop // self.spatial_patch
        blocks = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                sl = window_labels[
                    i * self.spatial_patch : (i + 1) * self.spatial_patch,
                    j * self.spatial_patch : (j + 1) * self.spatial_patch,
                ]
                valid = sl[(sl >= 1) & (sl <= self.num_classes)]
                if valid.size == 0:
                    blocks[i, j] = -1
                else:
                    counts = np.bincount(valid.ravel(), minlength=self.num_classes + 1)
                    blocks[i, j] = int(np.argmax(counts[1:]) + 1) - 1
        return blocks

    def __getitem__(self, index: int):
        row, col = self.points[index]
        tile = self.tiles[int(self._owners[index])]
        local_c = int(col) - int(tile.start_col)
        local_r = int(row)
        half = self.crop // 2 + 4
        r0 = max(0, local_r - half)
        r1 = min(tile.height, local_r + half)
        c0 = max(0, local_c - half)
        c1 = min(tile.width, local_c + half)
        raw = load_cube_window(
            tile.path, r0, r1, c0, c1,
            key=self.data_key, data_layout=self.data_layout,
        )
        prepared = prepare_mae_cube(
            raw,
            load_wavelengths(tile.path),
            mode=self.preprocess_mode,
            source_name=tile.path.name,
        )
        lr = min(max(local_r - r0, 0), prepared.shape[0] - 1)
        lc = min(max(local_c - c0, 0), prepared.shape[1] - 1)
        crop = _reflect_crop(prepared, lr, lc, self.crop)
        pad = self.crop // 2
        lab_pad = np.pad(self.label_map, ((pad, pad), (pad, pad)), mode="constant")
        win_lab = lab_pad[int(row) : int(row) + self.crop, int(col) : int(col) + self.crop]
        y = int(self.label_map[int(row), int(col)]) - 1
        blocks = self._block_labels(win_lab)
        return (
            torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32)),
            torch.tensor(y, dtype=torch.long),
            torch.from_numpy(blocks.astype(np.int64)),
        )


def discover_unlabeled_files(data_path: str | Path, pattern: str = "*") -> List[str]:
    return list_input_files(data_path, pattern)
