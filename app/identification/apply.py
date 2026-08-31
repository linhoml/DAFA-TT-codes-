"""Apply a trained CRISM classifier to an in-memory hyperspectral cube."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .crism_common import prepare_tile_cube, resolve_device
from .defaults import default_class_names, find_preprocess_model
from .lsga import lsga_hsi, prepare_lsga_for_eval
from .preprocess import load_process_model, preprocess_cube
from .test_full_image import make_cmap, merge_checkpoint_args
from .bands import cube_to_crism_240


ProgressCb = Optional[Callable[[int, int, str], None]]


def load_preprocess_model(path: str | Path) -> Dict:
    return load_process_model(path)


def torch_load_checkpoint(path, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(path: str | Path, device=None) -> Dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"分类模型不存在：{path}")
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch_load_checkpoint(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint 格式无效，需要包含 model_state_dict 的字典。")
    return checkpoint


def predict_prepared_cube(
    cube: np.ndarray,
    args: Dict,
    checkpoint: Dict,
    device=None,
    progress_cb: ProgressCb = None,
) -> Dict:
    """Run LSGA inference on a preprocessed HWB cube."""
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected H×W×B, got {cube.shape}")

    args = dict(args)
    args["norm_mode"] = "none"
    args["use_spectral_features"] = False
    prepared = prepare_tile_cube(cube, args)
    height, width, channels = prepared.shape
    expected = int(args["input_channels"])
    if channels != expected:
        raise ValueError(
            f"通道数不匹配：预处理后 {channels}，模型期望 {expected}"
        )

    device = device or resolve_device(args.get("device", "cpu"))
    model = lsga_hsi(args).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    prepare_lsga_for_eval(model)

    patch_size = int(args["patch_size"])
    batch_size = int(args.get("batch_size", 256))
    threshold = float(args.get("confidence_threshold", 0.0))
    pad = patch_size // 2
    padded = np.pad(prepared, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    offsets = np.arange(patch_size, dtype=np.int64)

    k = int(args["num_classes"])
    raw_map = np.zeros((height, width), dtype=np.int16)
    display_map = np.zeros((height, width), dtype=np.int16)
    confidence_map = np.zeros((height, width), dtype=np.float32)

    total = height * width
    use_amp = bool(args.get("use_amp", True)) and device.type == "cuda"

    ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with ctx():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            flat = np.arange(start, end, dtype=np.int64)
            rows = flat // width
            cols = flat % width
            patches = padded[
                rows[:, None, None] + offsets[None, :, None],
                cols[:, None, None] + offsets[None, None, :],
                :,
            ].transpose(0, 3, 1, 2)
            x = torch.from_numpy(np.ascontiguousarray(patches)).to(
                device, non_blocking=True
            )
            if use_amp:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=True,
                ):
                    probabilities = torch.softmax(model(x), dim=1)
                    confidence, prediction = probabilities.max(dim=1)
            else:
                probabilities = torch.softmax(model(x), dim=1)
                confidence, prediction = probabilities.max(dim=1)
            pred0 = prediction.cpu().numpy()
            pred1 = pred0.astype(np.int16) + 1
            conf = confidence.float().cpu().numpy()
            raw_map[rows, cols] = pred1
            confidence_map[rows, cols] = conf
            shown = pred1.copy()
            if threshold > 0:
                shown[conf < threshold] = 0
            display_map[rows, cols] = shown
            if progress_cb is not None:
                progress_cb(end, total, "分类推理")

    names = args.get("class_names") or default_class_names(k)
    return {
        "raw_prediction": raw_map,
        "display_prediction": display_map,
        "confidence": confidence_map,
        "num_classes": k,
        "class_names": list(names),
        "patch_size": patch_size,
        "input_channels": channels,
    }


def apply_to_opened_cube(
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray],
    checkpoint_path: str | Path,
    preprocess_model_path: str | Path,
    device_cfg="cpu",
    batch_size: int = 256,
    confidence_threshold: float = 0.0,
    progress_cb: ProgressCb = None,
) -> Dict:
    """End-to-end application: 1.02–2.58 μm → preprocess → LSGA map."""
    if progress_cb:
        progress_cb(0, 3, "波段截取 1.02–2.58 μm")
    cube_240, wl_240 = cube_to_crism_240(cube, wavelengths)

    if progress_cb:
        progress_cb(1, 3, "预处理（去尖峰 / SG / L2）")
    resolved_prep = find_preprocess_model(checkpoint_path, preprocess_model_path)
    if resolved_prep is None:
        raise FileNotFoundError(
            "找不到预处理模型 preprocess_model.pkl。\n"
            "训练时它写在输出目录（与 result 文件夹同级），"
            "不能留空，清空路径也不会跳过预处理。\n"
            f"当前分类模型：{checkpoint_path}\n"
            f"当前预处理路径：{preprocess_model_path!r}"
        )
    preprocess_model_path = resolved_prep
    prep_model = load_preprocess_model(preprocess_model_path)
    processed, summary = preprocess_cube(cube_240, prep_model, source_name="opened_cube")
    fill = float(prep_model.get("invalid_fill_value", 1e-4))
    if not np.all(np.isfinite(processed)):
        processed = np.nan_to_num(processed, nan=fill, posinf=fill, neginf=fill)

    device = resolve_device(device_cfg)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    runtime = {
        "device": device_cfg,
        "batch_size": int(batch_size),
        "confidence_threshold": float(confidence_threshold),
    }
    args = merge_checkpoint_args(checkpoint, runtime)
    args["batch_size"] = int(batch_size)
    args["confidence_threshold"] = float(confidence_threshold)
    if "class_names" not in args:
        args["class_names"] = default_class_names(int(args["num_classes"]))

    result = predict_prepared_cube(
        processed,
        args,
        checkpoint,
        device=device,
        progress_cb=progress_cb,
    )
    result["preprocess_summary"] = summary
    result["wavelengths"] = wl_240
    result["checkpoint_path"] = str(checkpoint_path)
    result["preprocess_model_path"] = str(preprocess_model_path)
    return result


def classification_colormap(num_classes: int):
    return make_cmap(int(num_classes))
