"""Shared CRISM utilities for training and external inference.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from .io import (
    format_cube_memory,
    format_file_pairs,
    load_array,
    load_cube,
    load_label_array,
    list_input_files,
    natural_sort_key,
    pair_files_by_closest_name,
    probe_cube_shape,
    should_load_cube_in_memory,
)


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


def python_executable() -> str:
    return sys.executable or "python"


def torch_cuda_status() -> Dict:
    """Inspect the PyTorch actually imported by this process (not nvidia-smi)."""
    import torch

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    gpu_name = None
    if available and count > 0:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {
        "python": python_executable(),
        "torch_file": getattr(torch, "__file__", None),
        "torch_version": getattr(torch, "__version__", "?"),
        "cuda_built": getattr(torch.version, "cuda", None),
        "available": available,
        "device_count": count,
        "gpu_name": gpu_name,
    }


def format_torch_runtime() -> str:
    info = torch_cuda_status()
    gpu = info["gpu_name"] or "-"
    return (
        f"启动用的 Python: {info['python']}\n"
        f"PyTorch {info['torch_version']}  file={info['torch_file']}  "
        f"CUDA_built={info['cuda_built']}  "
        f"cuda.is_available={info['available']}  "
        f"gpu_count={info['device_count']}  gpu0={gpu}"
    )


def cuda_unavailable_message(requested) -> str:
    info = torch_cuda_status()
    py = info["python"]
    quoted = f'"{py}"' if " " in py else py
    return (
        f"已选择设备 {requested}，但启动本软件的这个 Python 里 "
        f"PyTorch 检测不到 CUDA。\n"
        "任务管理器 / nvidia-smi 里有显卡，不等于当前 PyTorch 能用它："
        "pip 默认装的经常是 CPU 版。\n"
        f"  启动用的 Python: {py}\n"
        f"  torch 文件: {info['torch_file']}\n"
        f"  torch 版本: {info['torch_version']}\n"
        f"  torch.version.cuda: {info['cuda_built']}\n"
        f"  cuda.is_available(): {info['available']}\n"
        "请用上面这一条 python 安装 GPU 版（不要另开一个 python / 不要只用 pip）：\n"
        f"  {quoted} -m pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu128\n"
        "装完后用同一条检查：\n"
        f"  {quoted} -c \"import torch; print(torch.__file__, "
        "torch.cuda.is_available(), torch.version.cuda)\""
    )


def is_cuda_request(device_cfg) -> bool:
    if device_cfg is None or isinstance(device_cfg, bool):
        return False
    if isinstance(device_cfg, int):
        return device_cfg >= 0
    text = str(device_cfg).strip().lower()
    if text.startswith("cuda"):
        return True
    if text.lstrip("+-").isdigit():
        return int(text) >= 0
    return False


def cpu_device_warning(requested) -> str:
    if is_cuda_request(requested):
        return (
            "请求了 GPU，但实际落到了 CPU。这通常是当前 Python 装了 CPU 版 "
            "PyTorch，不是下拉框没选 cuda:0。"
        )
    return (
        "当前计算设备是 CPU，训练会很慢。"
        "本机有 NVIDIA 显卡时：先确认启动软件的同一个 python 已安装 GPU 版 "
        "PyTorch（torch.cuda.is_available() 为 True），再把计算设备改成 cuda:0。"
        "只改下拉框、PyTorch 仍是 CPU 版是不够的。batch size 建议 128 或 256。"
    )


def resolve_device(device_cfg):
    try:
        import torch
    except ImportError as exc:
        py = python_executable()
        quoted = f'"{py}"' if " " in py else py
        raise RuntimeError(
            f"当前 Python 没有安装 PyTorch：{py}\n"
            "请用这一条安装 GPU 版：\n"
            f"  {quoted} -m pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128"
        ) from exc

    def _require_cuda(requested, index: Optional[int] = None):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            raise RuntimeError(cuda_unavailable_message(requested))
        if index is None:
            text = str(requested).strip()
            if text.lower() in ("cuda", "cuda:0"):
                return torch.device("cuda:0")
            return torch.device(text)
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"已选择 cuda:{index}，但当前可见 GPU 只有 "
                f"{torch.cuda.device_count()} 张。"
            )
        return torch.device(f"cuda:{index}")

    if device_cfg is None:
        return torch.device("cpu")

    if isinstance(device_cfg, torch.device):
        if device_cfg.type == "cuda":
            idx = 0 if device_cfg.index is None else int(device_cfg.index)
            return _require_cuda(str(device_cfg), idx)
        return device_cfg

    if isinstance(device_cfg, bool):
        raise ValueError(f"Unknown device: {device_cfg!r}")

    if isinstance(device_cfg, (int, np.integer)):
        index = int(device_cfg)
        if index < 0:
            return torch.device("cpu")
        return _require_cuda(f"cuda:{index}", index)

    if isinstance(device_cfg, str):
        text = device_cfg.strip()
        if not text or text.lower() == "cpu":
            return torch.device("cpu")
        if text.lower().startswith("cuda"):
            if text.lower() == "cuda":
                return _require_cuda(text, 0)
            if ":" in text:
                suffix = text.split(":", 1)[1]
                if suffix.isdigit():
                    return _require_cuda(text, int(suffix))
            return _require_cuda(text)
        if text.lstrip("+-").isdigit():
            return resolve_device(int(text))
        raise ValueError(f"Unknown device: {device_cfg}")

    if isinstance(device_cfg, Sequence) and not isinstance(device_cfg, (str, bytes)):
        if len(device_cfg) > 0:
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
        try:
            height, width, bands = probe_cube_shape(
                path,
                key=data_key,
                data_layout=data_layout,
            )
        except Exception:
            cube = load_mat_data(
                path,
                key=data_key,
                prefer_3d=True,
                data_layout=data_layout,
            )
            if cube.ndim != 3:
                raise ValueError(f"Expected [H,W,B], got {cube.shape}: {path}")
            height, width, bands = cube.shape
            del cube
        if not should_load_cube_in_memory(height, width, bands):
            print(
                f"{path.name}: {height}×{width}×{bands} 约 "
                f"{format_cube_memory(height, width, bands)}，"
                "训练时按空间窗口读取，避免整幅载入内存。"
            )
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
    height = max(tile.height for tile in tiles)
    width = max(tile.start_col + tile.width for tile in tiles)
    return height, width


def assemble_paired_label_map(
    tiles: List[TileMeta],
    label_paths: Sequence[str | Path],
    *,
    key: Optional[str] = None,
) -> np.ndarray:
    """Place each per-file label at the corresponding tile's mosaic columns."""
    if len(tiles) != len(label_paths):
        raise ValueError(
            f"立方体数量 ({len(tiles)}) 与标签数量 ({len(label_paths)}) 不一致。"
        )
    height, width = mosaic_shape(tiles)
    mosaic = np.zeros((height, width), dtype=np.int64)
    for tile, lp in zip(tiles, label_paths):
        lab = load_label_map(lp, key=key)
        if lab.shape == (tile.width, tile.height) and lab.shape != (
            tile.height,
            tile.width,
        ):
            print(f"标签 {Path(lp).name} 看起来是转置的，已使用转置。")
            lab = lab.T
        if lab.shape != (tile.height, tile.width):
            raise ValueError(
                f"标签 {Path(lp)} 尺寸 {tuple(int(x) for x in lab.shape)} "
                f"与立方体 {tile.path.name} ({tile.height}, {tile.width}) 不一致。"
                "每个标签文件必须与对应训练文件同高同宽。"
            )
        mosaic[
            0 : tile.height,
            tile.start_col : tile.start_col + tile.width,
        ] = lab
    return mosaic


def resolve_tiles_and_label_map(
    tiles: List[TileMeta],
    label_path: str | Path,
    tile_width: int,
    requested_mode: str = "sequential",
    *,
    label_key: Optional[str] = None,
    label_pattern: str = "*",
) -> Tuple[List[TileMeta], np.ndarray, str]:
    """Load one mosaic label, or pair a label folder to tiles by filename."""
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(f"标签路径不存在：{path}")

    if path.is_dir():
        try:
            label_files = [Path(p) for p in list_input_files(path, label_pattern)]
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"标签文件夹中没有可用标签文件：{path}"
            ) from exc

        if len(label_files) == 1 and len(tiles) > 1:
            print(
                f"标签文件夹中只有 1 个文件 {label_files[0].name}，"
                "按整幅拼接标签使用（与多块立方体对齐）。"
            )
            label_map = load_label_map(label_files[0], key=label_key)
            return relayout_tiles_for_label(
                tiles, label_map, tile_width, requested_mode
            )

        pairs = pair_files_by_closest_name(
            [tile.path for tile in tiles], label_files
        )
        print(format_file_pairs(pairs))
        used = {Path(lab).resolve() for _, lab, _ in pairs}
        unused = [p for p in label_files if p.resolve() not in used]
        if unused:
            print("未使用的标签文件：" + ", ".join(p.name for p in unused))
        for cube, lab, score in pairs:
            if score < 0.5:
                print(
                    f"警告：{cube.name} 与 {lab.name} 文件名相似度仅 "
                    f"{score:.3f}，请确认配对是否正确。"
                )
        tiles = apply_tile_positions(tiles, "sequential", tile_width)
        label_map = assemble_paired_label_map(
            tiles,
            [lab for _, lab, _ in pairs],
            key=label_key,
        )
        return tiles, label_map, "paired"

    label_map = load_label_map(path, key=label_key)
    return relayout_tiles_for_label(
        tiles, label_map, tile_width, requested_mode
    )


def describe_multi_file_vs_stitched(tiles: List[TileMeta]) -> str:
    """Explain why folder-of-files training differs from one stitched cube."""
    if len(tiles) <= 1:
        return ""
    height, width = mosaic_shape(tiles)
    lines = [
        "多文件训练与「先把这些文件拼成一张再训练」结果一般不会相同，这是预期行为：",
        "  1. 预处理（去尖峰 / 空间坏点修补 / SG）只在每个文件内部做；"
        "拼成一张后，邻域会跨过原来的接缝。",
        "  2. 接缝附近的 patch 不同：分开训练时边缘用本文件镜像填充；"
        "拼起来后用的是邻幅真实像元。",
        "  3. train/val 按当前标签拼接图做 32×32 空间块划分。"
        f"本轮拼接为 {height}×{width}，文件按下面顺序从左到右排列。"
        "若你的拼图是竖拼、按地理位置，或文件顺序不同，划分会完全不同。",
        "  请对照日志里的 Split sizes。划分接近而精度仍差，主要来自 1 和 2；"
        "划分差很多，说明拼图几何/顺序与下面不一致。",
        "  本轮从左到右：",
    ]
    for index, tile in enumerate(tiles, start=1):
        lines.append(
            f"    {index}. {tile.path.name}  {tile.height}×{tile.width}  "
            f"列 {tile.start_col}–{tile.start_col + tile.width - 1}"
        )
    return "\n".join(lines)


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
        "标签图尺寸必须与拼接影像一致，才能计算检验精度。"
        f" 标签={tuple(label.shape)}，拼接影像=({height}, {width})。"
        "若测试的是多块 tile，请提供整幅拼接标签，或提供与各立方体文件名对应的标签文件夹。"
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
        "标签图尺寸必须与影像一致，才能计算检验精度。"
        f" 标签={tuple(int(x) for x in label_map.shape)}，"
        f"影像=({expected_height}, {expected_width})。"
        "请提供与测试立方体同高同宽的二维标签（类别 1..K，0 为未标注）。"
    )


def _fmt_pct(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number * 100:.2f}%"


def _fmt_num(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number:.{digits}f}"


def format_evaluation_report(metrics: Dict, names: Optional[Sequence[str]] = None) -> str:
    """Human-readable OA / AA / Kappa block for the test log."""
    total = int(metrics.get("total") or 0)
    lines = [
        "======== 检验精度 ========",
        f"有效标注像元数：{total}",
        f"OA  = {_fmt_pct(metrics.get('OA'))}",
        f"AA  = {_fmt_pct(metrics.get('AA'))}",
        f"Kappa = {_fmt_num(metrics.get('Kappa'))}",
        f"macro-F1 = {_fmt_pct(metrics.get('macro_F1'))}",
    ]
    recall = metrics.get("recall")
    support = metrics.get("support")
    if recall is not None:
        rec = np.asarray(recall, dtype=np.float64).reshape(-1)
        if support is None:
            sup = np.ones(rec.shape[0], dtype=np.int64)
        else:
            sup = np.asarray(support).reshape(-1)
        label_names = [str(n) for n in (names or [])]
        lines.append("各类别召回率（有标注的类）：")
        any_class = False
        for index, value in enumerate(rec):
            count = int(sup[index]) if index < sup.size else 0
            if count <= 0:
                continue
            any_class = True
            name = (
                label_names[index]
                if index < len(label_names)
                else f"class_{index + 1}"
            )
            lines.append(f"  {index + 1:2d} {name}: {_fmt_pct(value)}  (n={count})")
        if not any_class:
            lines.append("  （没有 1..K 的有效标注像元）")
    return "\n".join(lines)


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


def iter_tile_windows(
    tile: TileMeta,
    tile_points: np.ndarray,
    window: int,
    halo: int,
):
    """Yield spatial load windows that cover each labeled point once."""
    if len(tile_points) == 0:
        return
    local_r = tile_points[:, 0].astype(np.int64)
    local_c = tile_points[:, 1].astype(np.int64) - int(tile.start_col)
    window = max(int(window), 32)
    origins = np.stack(
        [(local_r // window) * window, (local_c // window) * window],
        axis=1,
    )
    unique = np.unique(origins, axis=0)
    for r0, c0 in unique:
        r0 = int(r0)
        c0 = int(c0)
        r1 = min(r0 + window, int(tile.height))
        c1 = min(c0 + window, int(tile.width))
        load_r0 = max(0, r0 - halo)
        load_c0 = max(0, c0 - halo)
        load_r1 = min(int(tile.height), r1 + halo)
        load_c1 = min(int(tile.width), c1 + halo)
        mask = (
            (local_r >= r0)
            & (local_r < r1)
            & (local_c >= c0)
            & (local_c < c1)
        )
        pts = tile_points[mask]
        if len(pts) == 0:
            continue
        yield load_r0, load_r1, load_c0, load_c1, pts


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
    wavelengths=None,
) -> np.ndarray:
    """Wavelength crop + inline CRISM preprocess, then optional band mask."""
    from .preprocess import prepare_identification_cube

    cube, _, _ = prepare_identification_cube(
        tile_data,
        wavelengths,
        source_name=str(args.get("dataset", "")),
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
    rows = local[:, 0]
    cols = local[:, 1]
    offsets = np.arange(patch_size, dtype=np.int64)
    patches = mirrored[
        rows[:, None, None] + offsets[None, :, None],
        cols[:, None, None] + offsets[None, None, :],
        :,
    ]
    return np.ascontiguousarray(
        patches.transpose(0, 3, 1, 2),
        dtype=np.float32,
    )


def build_patches(
    tile_data: np.ndarray,
    global_points: np.ndarray,
    start_col: int,
    patch_size: int,
    args: Dict,
    wavelengths=None,
) -> np.ndarray:
    prepared = prepare_tile_cube(tile_data, args, wavelengths=wavelengths)
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
