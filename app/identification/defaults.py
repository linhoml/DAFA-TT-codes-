"""Default CRISM identification paths, class names, and training settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

APPLY_WL_MIN = 1.02
APPLY_WL_MAX = 2.58
TARGET_BAND_NUM = 240

DEFAULT_CLASS_NAMES = [
    "nontronite",
    "montmorillonite",
    "none",
    "chlorite",
    "epidote",
    "bassanite",
    "polyhydrated_sulfate",
    "monohydrated_sulfate",
    "saponite",
    "pyroxene",
    "kaolinite_halloysite",
    "alunite",
    "gypsum",
    "magnesite",
    "jarosite",
    "prehnite",
    "calcite_siderite",
    "illite_muscovite",
    "serpentine",
    "hydrated_silica",
    "margarite",
    "olivine",
    "analcime",
    "vermiculite",
]

DATA_FORMAT_HELP = """\
输入数据格式（训练 / 测试）

1. 高光谱立方体：MATLAB .mat 三维数组
   • 排布可选 Height×Width×Bands（HWB）或 Bands×Height×Width（BHW）
   • 变量名默认识别第一个三维数组；预处理输出统一使用变量名 data
   • 438 波段：自动保留原始 1-based 第 4–243 波段（共 240 波段）
   • 240 波段：保持不变
   • 可选择单个 .mat，或包含多个 tile_*.mat 的文件夹

2. 标签（训练必填；测试可选）
   • 二维 .mat，与图像空间尺寸一致
   • 类别编号 1..K，0 为背景/未标注
   • 默认 24 类火星矿物

3. 预处理
   • 训练时在训练数据上拟合 process_model.pkl，再写出预处理立方体
   • 测试时复用同一 process_model.pkl，不可重新拟合

4. 模型应用
   • 使用软件中已打开的高光谱影像（I/F）
   • 仅取 1.02–2.58 μm 波段，重采样到 240 通道后预处理并分类
"""


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _package_root().parents[1]


def identification_data_dir() -> Path:
    return _repo_root() / "data" / "identification"


def builtin_dir() -> Path:
    return identification_data_dir() / "builtin"


def builtin_preprocess_path() -> Path:
    return builtin_dir() / "preprocess_model.pkl"


def builtin_checkpoint_path() -> Path:
    return builtin_dir() / "model_best.pth"


def last_trained_record_path() -> Path:
    return identification_data_dir() / "last_trained.json"


def default_class_names(num_classes: int = 24) -> List[str]:
    if num_classes == len(DEFAULT_CLASS_NAMES):
        return list(DEFAULT_CLASS_NAMES)
    return [f"class_{i}" for i in range(1, num_classes + 1)]


def default_train_args() -> Dict:
    return {
        "dataset": "crism",
        "mode": "train",
        "data_key": "data",
        "label_key": None,
        "tile_pattern": "*.mat",
        "tile_w": 600,
        "tile_position_mode": "sequential",
        "num_classes": 24,
        "patch_size": 9,
        "batch_size": 64,
        "epochs": 130,
        "lr": 5e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-2,
        "label_smoothing": 0.02,
        "grad_clip": 5.0,
        "early_stopping_patience": 30,
        "num_workers": 0,
        "use_balanced_sampler": True,
        "spatial_augment": True,
        "norm_mode": "none",
        "use_spectral_features": False,
        "device": "cpu",
        "seed_list": [0],
        "class_names": default_class_names(24),
    }


def load_last_trained() -> Optional[Dict]:
    path = last_trained_record_path()
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_last_trained(record: Dict) -> None:
    path = last_trained_record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
