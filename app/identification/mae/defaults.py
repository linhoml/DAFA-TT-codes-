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
大图按窗口读取，不会整幅载入。预训练建议用 ENVI .hdr/.img，并放在本地硬盘。
RaiDrive/WebDAV 上若只有 .hdr、.img 是 0 字节，程序会跳过这些文件；
全部都是空影像时会报错，请先拷到本地或等网盘下载完。

预训练轮数 × 每轮随机窗口数 = 本轮预训练看到的窗口总数（每个窗口 32×32）。
每轮窗口不必等于文件数：程序每步从全部立方体里随机抽一幅再随机裁一块。
建议每轮窗口 ≥ 文件数，让多数影像每轮至少被抽到一次。

推荐（GPU，预处理选 crop）：
  试跑 / 通路径：轮数 5，每轮 256，batch 16
  几百幅：轮数 50，每轮 2048，batch 32（软件默认）
  约 1000–3000 幅：轮数 80，每轮 4096，batch 32–64
  约 1 万幅：轮数 100–200，每轮 8192–16384，batch 32–64
  CPU 试跑：轮数 2，每轮 64，batch=4，读盘线程 1

这三个数分别干什么：
  轮数、每轮窗口 = 一共看多少块，只决定训多久、学多少，加大不会更快。
  batch size = 每步送进 GPU 多少块。显存够就加到 32 或 64，计算段更饱。
  单独加大 batch 若读盘仍是串行，利用率往往仍只有百分之十几。
  GPU 利用率 <20% 通常是读盘饿着 GPU：把「读盘线程」加到 4–8，
  看日志「读盘 Xs 计算 Ys」，目标是读盘 ≤ 计算。

看 loss：前几轮应明显下降，后期变缓即可停。每 10 轮会存 encoder_epN.pt。

怎么确认在用 GPU：
  日志开头应有 cuda.is_available=True 和 gpu0=显卡名。
  「编码器参数」只说明模型建好了，还没开始算。
  下一行「实际训练设备：cuda:0（显卡名）」才表示模型已上 GPU。
  出现 [ep 1/… batch 1/…] loss= … 设备 cuda:0 显存 xxx MiB，才是第一次前向完成。
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
        "batch_size": 32,
        "samples_per_epoch": 2048,
        "num_readers": 4,
        "crops_per_read": 4,
        "prefetch_batches": 2,
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
