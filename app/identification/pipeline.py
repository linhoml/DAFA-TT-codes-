"""GUI-facing training / testing orchestration for CRISM identification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from .crism_common import load_json, resolve_device
from .defaults import default_train_args, save_last_trained
from .preprocess import build_preprocess_model, transform_inputs
from .test_full_image import run_scene
from .train import train_one_seed


LogFn = Callable[[str], None]


def _log(log: Optional[LogFn], message: str) -> None:
    if log is not None:
        log(message)
    else:
        print(message)


def run_training(config: Dict, log: Optional[LogFn] = None) -> Dict:
    """
    Fit preprocess model (optional), write preprocessed cubes, then train LSGA.

    config keys:
        data_path, already_preprocessed, data_key, data_layout, input_pattern,
        label_path, label_key, output_dir, num_classes, patch_size, batch_size,
        epochs, seed, device, extra_json
    """
    data_path = Path(config["data_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = output_dir / "preprocessed"
    preprocess_model_path = output_dir / "preprocess_model.pkl"
    result_dir = output_dir / "result"

    args = default_train_args()
    extra = config.get("extra_json")
    if extra:
        args.update(load_json(extra))

    args["dataset"] = str(config.get("dataset", args["dataset"]))
    args["data_key"] = config.get("data_key") or args.get("data_key", "data")
    args["label_path"] = str(config["label_path"])
    args["label_key"] = config.get("label_key") or None
    args["num_classes"] = int(config.get("num_classes", args["num_classes"]))
    args["patch_size"] = int(config.get("patch_size", args["patch_size"]))
    args["batch_size"] = int(config.get("batch_size", args["batch_size"]))
    args["epochs"] = int(config.get("epochs", args["epochs"]))
    args["device"] = config.get("device", args["device"])
    args["result_dir"] = str(result_dir)
    args["tile_pattern"] = config.get("input_pattern") or "*.mat"
    args["tile_w"] = int(config.get("tile_w", args["tile_w"]))
    seed = int(config.get("seed", 0))
    args["seed_list"] = [seed]

    if data_path.is_file():
        args["tile_position_mode"] = "sequential"
        args["tile_pattern"] = data_path.name
        tile_input = data_path.parent
        input_pattern = data_path.name
    else:
        args["tile_position_mode"] = str(
            config.get("tile_position_mode", "sequential")
        )
        tile_input = data_path
        input_pattern = args["tile_pattern"]

    already = bool(config.get("already_preprocessed", False))
    data_key = config.get("data_key") or None
    data_layout = str(config.get("data_layout", "HWB"))

    if already:
        _log(log, "跳过预处理：使用已预处理的 .mat（变量名 data）。")
        if data_path.is_file():
            args["tile_dir"] = str(data_path.parent)
        else:
            args["tile_dir"] = str(data_path)
        if not preprocess_model_path.exists() and config.get("preprocess_model_path"):
            preprocess_model_path = Path(config["preprocess_model_path"])
    else:
        _log(log, f"拟合预处理模型：{preprocess_model_path}")
        build_preprocess_model(
            train_input_path=tile_input if data_path.is_dir() else data_path,
            model_save_path=preprocess_model_path,
            input_pattern=input_pattern,
            data_key=data_key,
            data_layout=data_layout,
        )
        _log(log, f"写出预处理立方体：{preprocessed_dir}")
        transform_inputs(
            input_path=tile_input if data_path.is_dir() else data_path,
            save_dir=preprocessed_dir,
            model_path=preprocess_model_path,
            input_pattern=input_pattern,
            data_key=data_key,
            data_layout=data_layout,
            output_prefix="preprocessed",
        )
        args["tile_dir"] = str(preprocessed_dir)
        args["data_key"] = "data"
        args["tile_pattern"] = "preprocessed_*.mat"
        args["tile_position_mode"] = "sequential"

    _log(log, f"开始训练 seed={seed}  device={args['device']}")
    train_one_seed(args, seed)

    checkpoint = (
        result_dir
        / str(args["dataset"])
        / f"{args['dataset']}_seed{seed}_best.pth"
    )
    record = {
        "checkpoint_path": str(checkpoint),
        "preprocess_model_path": str(preprocess_model_path),
        "result_dir": str(result_dir / str(args["dataset"])),
        "num_classes": int(args["num_classes"]),
        "patch_size": int(args["patch_size"]),
        "dataset": args["dataset"],
        "seed": seed,
    }
    save_last_trained(record)
    _log(log, f"训练完成。最佳模型：{checkpoint}")
    return record


def run_testing(config: Dict, log: Optional[LogFn] = None) -> Dict:
    """Preprocess test cubes if needed, then run full-image inference + metrics."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(config.get("device", "cpu"))
    from .apply import torch_load_checkpoint

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)

    data_path = Path(config["data_path"])
    already = bool(config.get("already_preprocessed", False))
    data_key = config.get("data_key") or None
    data_layout = str(config.get("data_layout", "HWB"))
    input_pattern = config.get("input_pattern") or "*.mat"
    preprocess_model_path = Path(
        config.get("preprocess_model_path") or ""
    )

    if already:
        test_input = data_path
        runtime_data_key = data_key or "data"
    else:
        if not preprocess_model_path.exists():
            raise FileNotFoundError(
                "测试未预处理数据时必须提供训练阶段的 process_model.pkl"
            )
        preprocessed_dir = output_dir / "preprocessed"
        _log(log, f"复用预处理模型：{preprocess_model_path}")
        transform_inputs(
            input_path=data_path,
            save_dir=preprocessed_dir,
            model_path=preprocess_model_path,
            input_pattern=input_pattern if data_path.is_dir() else data_path.name,
            data_key=data_key,
            data_layout=data_layout,
            output_prefix="preprocessed",
        )
        test_input = preprocessed_dir
        runtime_data_key = "data"
        input_pattern = "preprocessed_*.mat"

    runtime = {
        "device": config.get("device", "cpu"),
        "batch_size": int(config.get("batch_size", 256)),
        "output_dir": str(output_dir),
        "data_key": runtime_data_key,
        "label_key": config.get("label_key") or None,
        "scene_name": config.get("scene_name") or test_input.stem,
        "save_png": True,
        "save_mat": True,
        "save_confidence_map": True,
    }

    label_path = config.get("label_path") or ""
    if data_path.is_file() and already:
        runtime["test_mode"] = "single"
        runtime["test_img"] = str(test_input)
        if label_path:
            runtime["test_label"] = label_path
    elif test_input.is_file():
        runtime["test_mode"] = "single"
        runtime["test_img"] = str(test_input)
        if label_path:
            runtime["test_label"] = label_path
    else:
        # After preprocessing a single file, save_dir contains one mat.
        mats = sorted(Path(test_input).glob(input_pattern)) if test_input.is_dir() else []
        if len(mats) == 1:
            runtime["test_mode"] = "single"
            runtime["test_img"] = str(mats[0])
            if label_path:
                runtime["test_label"] = label_path
        else:
            runtime["test_mode"] = "tile"
            runtime["tile_dir"] = str(test_input if test_input.is_dir() else test_input.parent)
            runtime["tile_pattern"] = input_pattern
            runtime["tile_w"] = int(config.get("tile_w", 600))
            runtime["tile_position_mode"] = str(
                config.get("tile_position_mode", "sequential")
            )
            if label_path:
                runtime["label_path"] = label_path

    _log(log, "开始整图测试 / 推理…")
    summary = run_scene(runtime, runtime, checkpoint, checkpoint_path)
    _log(log, f"测试完成。输出目录：{summary.get('output_dir')}")
    return summary
