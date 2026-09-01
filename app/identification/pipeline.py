"""GUI-facing training / testing orchestration for CRISM identification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from .crism_common import (
    format_evaluation_report,
    format_torch_runtime,
    load_json,
    resolve_device,
)
from .defaults import default_train_args, save_last_trained
from .io import DEFAULT_INPUT_PATTERN


LogFn = Callable[[str], None]


def _log(log: Optional[LogFn], message: str) -> None:
    if log is not None:
        log(message)
    else:
        print(message)


def run_training(config: Dict, log: Optional[LogFn] = None) -> Dict:
    """Read cubes, inline 1.02–2.6 μm preprocess, then train LSGA."""
    data_path = Path(config["data_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "result"

    args = default_train_args()
    extra = config.get("extra_json")
    if extra:
        args.update(load_json(extra))

    args["dataset"] = str(config.get("dataset", args["dataset"]))
    data_key = config.get("data_key") or None
    args["data_key"] = data_key
    args["label_path"] = str(config["label_path"])
    args["label_key"] = config.get("label_key") or None
    args["num_classes"] = int(config.get("num_classes", args["num_classes"]))
    args["patch_size"] = int(config.get("patch_size", args["patch_size"]))
    args["batch_size"] = int(config.get("batch_size", args["batch_size"]))
    args["epochs"] = int(config.get("epochs", args["epochs"]))
    args["device"] = config.get("device", args["device"])
    resolve_device(args["device"])
    args["result_dir"] = str(result_dir)
    args["tile_pattern"] = config.get("input_pattern") or DEFAULT_INPUT_PATTERN
    args["tile_w"] = int(config.get("tile_w", args["tile_w"]))
    args["data_layout"] = str(config.get("data_layout", "HWB"))
    seed = int(config.get("seed", 0))
    args["seed_list"] = [seed]

    if data_path.is_file():
        args["tile_position_mode"] = "sequential"
        args["tile_pattern"] = data_path.name
        args["tile_dir"] = str(data_path.parent)
    else:
        args["tile_position_mode"] = str(
            config.get("tile_position_mode", "sequential")
        )
        args["tile_dir"] = str(data_path)

    _log(
        log,
        "读取数据后自动截取 1.02–2.6 μm，并做无效值填充 / 去尖峰 / "
        "空间坏点修补 / SG 平滑 / L2 归一化（不保存预处理模型）。",
    )
    _log(log, format_torch_runtime())
    _log(log, f"开始训练 seed={seed}  请求设备={args['device']!r}")
    from .train import train_one_seed

    trained = train_one_seed(args, seed)
    metrics = trained.get("metrics") or {}

    checkpoint = (
        result_dir
        / str(args["dataset"])
        / f"{args['dataset']}_seed{seed}_best.pth"
    )
    record = {
        "checkpoint_path": str(checkpoint),
        "result_dir": str(result_dir / str(args["dataset"])),
        "num_classes": int(args["num_classes"]),
        "patch_size": int(args["patch_size"]),
        "dataset": args["dataset"],
        "seed": seed,
        "metrics": metrics,
    }
    save_last_trained(record)
    if metrics.get("test_all_OA") is not None:
        _log(
            log,
            "原脚本口径 test_all OA="
            f"{metrics['test_all_OA'] * 100:.2f}%  "
            f"AA={metrics['test_all_AA'] * 100:.2f}%",
        )
    _log(log, f"训练完成。最佳模型：{checkpoint}")
    return record


def run_testing(config: Dict, log: Optional[LogFn] = None) -> Dict:
    """Run labeled full-image evaluation with the same inline preprocess as training."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    label_path = str(config.get("label_path") or "").strip()
    if not label_path or not Path(label_path).exists():
        raise ValueError(
            "模型测试必须提供与影像空间尺寸一致的标签图，才能计算检验精度。"
        )

    device = resolve_device(config.get("device", "cpu"))
    _log(log, format_torch_runtime())
    _log(log, f"请求设备={config.get('device')!r} → 实际设备={device}")
    _log(log, f"检验标签：{label_path}")
    from .apply import torch_load_checkpoint

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)

    data_path = Path(config["data_path"])
    data_key = config.get("data_key") or None
    data_layout = str(config.get("data_layout", "HWB"))
    input_pattern = config.get("input_pattern") or DEFAULT_INPUT_PATTERN
    test_input = data_path

    runtime = {
        "device": config.get("device", "cpu"),
        "batch_size": int(config.get("batch_size", 256)),
        "output_dir": str(output_dir),
        "data_key": data_key,
        "data_layout": data_layout,
        "label_key": config.get("label_key") or None,
        "scene_name": config.get("scene_name") or test_input.stem,
        "save_png": True,
        "save_mat": True,
        "save_envi": True,
        "save_confidence_map": True,
        "require_label": True,
    }

    if data_path.is_file():
        runtime["test_mode"] = "single"
        runtime["test_img"] = str(test_input)
        runtime["test_label"] = label_path
    else:
        from .io import list_input_files

        try:
            mats = list_input_files(test_input, input_pattern)
        except FileNotFoundError:
            mats = []
        if len(mats) == 1:
            runtime["test_mode"] = "single"
            runtime["test_img"] = str(mats[0])
            runtime["test_label"] = label_path
        else:
            runtime["test_mode"] = "tile"
            runtime["tile_dir"] = str(test_input)
            runtime["tile_pattern"] = input_pattern
            runtime["tile_w"] = int(config.get("tile_w", 600))
            runtime["tile_position_mode"] = str(
                config.get("tile_position_mode", "sequential")
            )
            runtime["label_path"] = label_path

    _log(log, "开始整图测试（自动 1.02–2.6 μm 截取与预处理，对照标签计算检验精度）…")
    from .test_full_image import run_scene

    summary = run_scene(runtime, runtime, checkpoint, checkpoint_path)
    report = str(summary.get("accuracy_report") or "").strip()
    if not report:
        report = format_evaluation_report(summary, summary.get("class_names"))
    _log(log, report)
    _log(log, f"测试完成。输出目录：{summary.get('output_dir')}")
    return summary
