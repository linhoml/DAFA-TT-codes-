"""Defaults and help text for CRISM spatial+spectral MAE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from identification.defaults import (
    default_class_names,
    identification_data_dir,
)

MAE_CROP = 32
MAE_SPATIAL_PATCH = 8
MAE_SPECTRAL_PATCH = 16
MAE_BANDS = 240
MAE_D_MODEL = 256
MAE_ENCODER_DEPTH = 8
MAE_ENCODER_HEADS = 8
MAE_DECODER_DIM = 128
MAE_DECODER_DEPTH = 4
MAE_DECODER_HEADS = 4

# Hydration / mineral diagnostic intervals (μm), analogous to LIBS emission lines.
FEATURE_WL_WINDOWS = (
    (1.35, 1.45),
    (1.88, 2.02),
    (2.14, 2.36),
    (2.38, 2.54),
)

MAE_HELP = """\
MAE 自监督（Masked Autoencoder）— CRISM 空间+光谱

论文里的 LIBS 只有一条光谱，用 1D 波长 patch + 谱线感知掩码。
CRISM 是 H×W×波段立方体，矿物既看吸收位置，也看空间邻域，因此这里用
spatial+spectral 3D MAE：把 32×32×240 的窗口切成 8×8×16 的三维块再掩码重建。

流程：
1. 自监督预训练：文件夹里上万幅无标签 CRISM（I/F）。随机抽 32×32 窗口，
   掩掉约 75% 的三维块（诊断波段 1.4 / 1.9 / 2.3 μm 附近掩得更狠），
   只在被掩块上算重建损失。不需要标签。
2. 少量样本微调：加载预训练编码器，用有标签像元（可限制每类条数）训练分类头。
   可选冻结编码器（linear probe）。
3. 测试 / 应用：与 LSGA 一样输出矿物分类图（ENVI *_MAE_classification.img）。

输入立方体格式与 Identification 其它方法相同（ENVI / PDS / .npy / .mat）。
大图按窗口读取，不会整幅载入。预训练建议用 ENVI .hdr/.img。
"""


def mae_data_dir() -> Path:
    return identification_data_dir() / "mae"


def last_pretrain_path() -> Path:
    return identification_data_dir() / "last_mae_pretrain.json"


def last_finetune_path() -> Path:
    return identification_data_dir() / "last_mae_finetune.json"


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)


def load_last_pretrain() -> Optional[Dict]:
    return _read_json(last_pretrain_path())


def save_last_pretrain(record: Dict) -> None:
    _write_json(last_pretrain_path(), record)


def load_last_finetune() -> Optional[Dict]:
    return _read_json(last_finetune_path())


def save_last_finetune(record: Dict) -> None:
    _write_json(last_finetune_path(), record)


def default_pretrain_args() -> Dict:
    return {
        "crop": MAE_CROP,
        "spatial_patch": MAE_SPATIAL_PATCH,
        "spectral_patch": MAE_SPECTRAL_PATCH,
        "bands": MAE_BANDS,
        "d_model": MAE_D_MODEL,
        "encoder_depth": MAE_ENCODER_DEPTH,
        "encoder_heads": MAE_ENCODER_HEADS,
        "decoder_dim": MAE_DECODER_DIM,
        "decoder_depth": MAE_DECODER_DEPTH,
        "decoder_heads": MAE_DECODER_HEADS,
        "p_feat": 0.85,
        "p_cont": 0.70,
        "mask_ratio": 0.75,
        "epochs": 50,
        "batch_size": 16,
        "samples_per_epoch": 2048,
        "lr": 1.5e-4,
        "min_lr": 1e-6,
        "weight_decay": 0.05,
        "warmup_frac": 0.05,
        "device": "cuda:0",
        "seed": 0,
        "preprocess_mode": "crop",
        "data_layout": "HWB",
        "input_pattern": "*",
    }


def default_finetune_args() -> Dict:
    args = default_pretrain_args()
    args.update(
        {
            "num_classes": 24,
            "class_names": default_class_names(24),
            "epochs": 40,
            "batch_size": 32,
            "lr": 1e-4,
            "head_lr": 1e-3,
            "freeze_encoder": False,
            "max_per_class": 0,
            "val_fraction": 0.2,
            "label_smoothing": 0.02,
            "spatial_loss_weight": 0.5,
            "confidence_threshold": 0.0,
        }
    )
    return args
