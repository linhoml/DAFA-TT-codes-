"""GUI orchestration for MAE pretrain / fine-tune / test / apply."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from identification.crism_common import format_evaluation_report
from identification.defaults import default_class_names
from identification.io import load_label_array, list_input_files
from identification.train import metrics_from_confusion, update_confusion

from .apply import apply_paths
from .defaults import load_last_finetune, load_last_pretrain, mae_data_dir
from .finetune import run_finetune
from .pretrain import run_pretrain


LogFn = Callable[[str], None]


def _log(log: Optional[LogFn], message: str) -> None:
    if log is not None:
        log(message)
    else:
        print(message)


def pretrain(config: Dict, log: Optional[LogFn] = None) -> Dict:
    _log(log, "======== MAE 自监督预训练（spatial+spectral）========")
    return run_pretrain(config, log=log)


def finetune(config: Dict, log: Optional[LogFn] = None) -> Dict:
    _log(log, "======== MAE 少量样本微调 ========")
    return run_finetune(config, log=log)


def evaluate_prediction(pred: np.ndarray, label_map: np.ndarray, num_classes: int) -> Dict:
    k = int(num_classes)
    cm = np.zeros((k, k), dtype=np.int64)
    gt = np.asarray(label_map)
    pred0 = np.asarray(pred, dtype=np.int16) - 1
    valid = (gt >= 1) & (gt <= k)
    update_confusion(cm, gt[valid] - 1, pred0[valid], k)
    metrics = metrics_from_confusion(cm)
    metrics["Kappa"] = metrics.get("kappa")
    metrics["total"] = int(np.sum(metrics.get("class_total", 0)))
    metrics["recall"] = metrics.get("per_class_acc")
    metrics["support"] = metrics.get("class_total")
    return {"confusion_matrix": cm, **metrics}


def run_test(config: Dict, log: Optional[LogFn] = None) -> Dict:
    checkpoint = Path(config["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到微调模型：{checkpoint}")
    data_path = config["data_path"]
    paths = list_input_files(data_path, config.get("input_pattern") or "*")
    save_dir = Path(config.get("output_dir") or mae_data_dir() / "test")
    batch = apply_paths(
        paths,
        checkpoint_path=checkpoint,
        save_dir=save_dir,
        device_cfg=config.get("device", "cpu"),
        batch_size=int(config.get("batch_size", 8)),
        data_key=config.get("data_key"),
        data_layout=config.get("data_layout") or "HWB",
        log=lambda m: _log(log, m),
    )
    last = batch["last"]
    label_path = str(config.get("label_path") or "").strip()
    if not label_path:
        raise ValueError("测试必须提供与影像同尺寸的标签图。")
    label = np.squeeze(load_label_array(label_path, key=config.get("label_key")))
    pred = np.asarray(last["display_prediction"])
    if label.shape != pred.shape:
        if label.T.shape == pred.shape:
            label = label.T
            _log(log, "标签看起来是转置的，已使用 label.T")
        else:
            raise ValueError(
                f"标签尺寸 {tuple(label.shape)} 与分类图 {tuple(pred.shape)} 不一致"
            )
    k = int(last.get("num_classes") or 24)
    metrics = evaluate_prediction(pred, label, k)
    names = last.get("class_names") or default_class_names(k)
    report = format_evaluation_report(metrics, names)
    _log(log, report)
    last.update(metrics)
    last["report"] = report
    last["label_path"] = label_path
    return last


def default_encoder_path() -> str:
    rec = load_last_pretrain() or {}
    path = rec.get("checkpoint_path") or str(mae_data_dir() / "pretrain" / "checkpoints" / "encoder.pt")
    return path


def default_finetune_checkpoint() -> str:
    rec = load_last_finetune() or {}
    return rec.get("checkpoint_path") or str(mae_data_dir() / "finetune" / "model_best.pth")
