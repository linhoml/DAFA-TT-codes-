"""Apply a fine-tuned MAE classifier to CRISM cubes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from identification.crism_common import (
    discover_single_tile,
    iter_prepared_coverage_windows,
    resolve_device,
    should_window_tile,
)
from identification.defaults import default_class_names
from identification.io import (
    RASTER_EXTENSIONS,
    classification_stem,
    format_cube_memory,
    is_classification_output,
    load_cube,
    load_wavelengths,
    write_envi_class_map,
)
from identification.preprocess import prepare_identification_cube

from .model import MineralMAEClassifier, encoder_from_config


ProgressCb = Optional[Callable[[int, int, str], None]]


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_classifier(checkpoint_path, device) -> tuple:
    payload = _torch_load(checkpoint_path, map_location=device)
    config = dict(payload.get("config") or {})
    encoder = encoder_from_config(config)
    num_classes = int(config.get("num_classes") or 24)
    model = MineralMAEClassifier(encoder, num_classes)
    state = payload.get("model_state_dict")
    if state is None:
        raise ValueError(f"{checkpoint_path} 不是微调后的 MAE 分类模型（缺少 model_state_dict）")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, payload, config


def _tile_cube(cube: np.ndarray, crop: int) -> tuple:
    h, w, _c = cube.shape
    pad_h = (crop - h % crop) % crop
    pad_w = (crop - w % crop) % crop
    padded = np.pad(cube, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, h, w


def predict_prepared_cube(
    cube: np.ndarray,
    model: MineralMAEClassifier,
    device,
    batch_size: int = 8,
    progress_cb: ProgressCb = None,
) -> Dict:
    cube = np.asarray(cube, dtype=np.float32)
    crop = model.encoder.crop
    padded, height, width = _tile_cube(cube, crop)
    ph, pw = padded.shape[0], padded.shape[1]
    raw_map = np.zeros((ph, pw), dtype=np.int16)
    conf_map = np.zeros((ph, pw), dtype=np.float32)
    tiles = []
    coords = []
    for row in range(0, ph, crop):
        for col in range(0, pw, crop):
            tiles.append(padded[row : row + crop, col : col + crop])
            coords.append((row, col))
    total = len(tiles)
    ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with ctx():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = np.stack(tiles[start:end], axis=0)
            x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            maps, cmap = model.predict_tile(x)
            maps_np = maps.cpu().numpy()
            conf_np = cmap.cpu().numpy()
            for i, (row, col) in enumerate(coords[start:end]):
                raw_map[row : row + crop, col : col + crop] = maps_np[i]
                conf_map[row : row + crop, col : col + crop] = conf_np[i]
            if progress_cb is not None:
                progress_cb(end, total, "MAE 推理")
    raw_map = raw_map[:height, :width]
    conf_map = conf_map[:height, :width]
    k = model.num_classes
    return {
        "raw_prediction": raw_map,
        "display_prediction": raw_map.copy(),
        "confidence": conf_map,
        "num_classes": k,
        "class_names": default_class_names(k),
        "patch_size": crop,
    }


def apply_to_cube_array(
    cube: np.ndarray,
    wavelengths,
    checkpoint_path,
    device_cfg="cpu",
    batch_size: int = 8,
    progress_cb: ProgressCb = None,
    source_name: str = "cube",
    model=None,
) -> Dict:
    device = resolve_device(device_cfg)
    if model is None:
        model, _payload, config = load_classifier(checkpoint_path, device)
    else:
        config = {}
    names = list(config.get("class_names") or default_class_names(model.num_classes))
    if progress_cb:
        progress_cb(0, 3, "截取 1.02–2.6 μm 并预处理")
    processed, _wl, summary = prepare_identification_cube(
        cube, wavelengths, source_name=source_name
    )
    if progress_cb:
        progress_cb(1, 3, "MAE 分类")
    result = predict_prepared_cube(
        processed, model, device, batch_size=batch_size, progress_cb=progress_cb
    )
    result["class_names"] = names
    result["preprocess_summary"] = summary
    result["checkpoint_path"] = str(checkpoint_path)
    return result


def apply_paths(
    paths,
    checkpoint_path,
    save_dir,
    device_cfg="cpu",
    batch_size: int = 8,
    data_key=None,
    data_layout: str = "HWB",
    progress_cb: ProgressCb = None,
    log=None,
) -> Dict:
    device = resolve_device(device_cfg)
    model, _payload, config = load_classifier(checkpoint_path, device)
    names = list(config.get("class_names") or default_class_names(model.num_classes))
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path_list = [Path(p) for p in paths]
    saved = []
    last = None
    total = len(path_list)
    if total == 0:
        raise FileNotFoundError("没有可分类的立方体文件。")
    runtime = {
        "data_key": data_key or "data",
        "data_layout": data_layout,
        "patch_size": int(config.get("crop", 32)),
        "dataset": "mae",
    }
    for index, path in enumerate(path_list):
        if path.suffix.lower() in RASTER_EXTENSIONS and is_classification_output(path):
            if log:
                log(f"跳过分类结果文件：{path}")
            continue
        if log:
            log(f"MAE 分类 {index + 1}/{total}：{path}")
        tile = discover_single_tile(
            path, data_key=data_key, data_layout=data_layout
        )[0]
        height, width = int(tile.height), int(tile.width)
        if should_window_tile(tile, runtime):
            if log:
                log(
                    f"{path.name} 约 "
                    f"{format_cube_memory(height, width, tile.bands)}，按窗口分类"
                )
            raw_map = np.zeros((height, width), dtype=np.int16)
            conf_map = np.zeros((height, width), dtype=np.float32)
            for prepared, r0, r1, c0, c1, load_r0, load_c0 in iter_prepared_coverage_windows(
                tile, runtime
            ):
                part = predict_prepared_cube(
                    prepared, model, device, batch_size=batch_size
                )
                sr, sc = r0 - load_r0, c0 - load_c0
                er, ec = r1 - load_r0, c1 - load_c0
                raw_map[r0:r1, c0:c1] = part["raw_prediction"][sr:er, sc:ec]
                conf_map[r0:r1, c0:c1] = part["confidence"][sr:er, sc:ec]
            last = {
                "raw_prediction": raw_map,
                "display_prediction": raw_map.copy(),
                "confidence": conf_map,
                "num_classes": model.num_classes,
                "class_names": names,
            }
        else:
            cube = load_cube(path, key=data_key, data_layout=data_layout)
            last = apply_to_cube_array(
                cube,
                load_wavelengths(path),
                checkpoint_path,
                device_cfg,
                batch_size=batch_size,
                source_name=path.stem,
                model=model,
            )
            last["class_names"] = names
        envi_path = out_dir / f"{classification_stem(path.stem, 'MAE')}.img"
        last["envi_path"] = str(
            write_envi_class_map(envi_path, last["display_prediction"], names)
        )
        last["source_path"] = str(path)
        last["checkpoint_path"] = str(checkpoint_path)
        last["num_classes"] = model.num_classes
        saved.append(str(last["envi_path"]))
        if log:
            log(f"已保存 {last['envi_path']}")
        if progress_cb:
            progress_cb(index + 1, total, path.name)
    if last is None:
        raise FileNotFoundError("没有可分类的立方体文件。")
    return {
        "saved": saved,
        "last": last,
        "count": len(saved),
        "save_dir": str(out_dir),
        "num_classes": model.num_classes,
        "class_names": names,
    }
