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

1. 高光谱立方体（单个文件或文件夹）
   支持多种格式：
   • MATLAB：.mat（三维数组）
   • ENVI / CRISM：.img、.dat、.hdr、.bsq/.bil/.bip
   • PDS 标签：.lbl（配合同目录 .img）
   • NumPy：.npy / .npz；GeoTIFF：.tif / .tiff
   • 选 .img/.dat 时会自动寻找同名 .hdr 或 .lbl
   • 排布：Height×Width×Bands（HWB）或 Bands×Height×Width（BHW）
     （ENVI/PDS 一般为 HWB，此项主要给 .mat/.npy 用）
   • .mat/.npz 变量名可留空（自动取第一个三维数组）；预处理输出变量名为 data
   • 438 波段：自动保留原始 1-based 第 4–243 波段（共 240 波段）
   • 240 波段：保持不变
   • 文件夹匹配模式默认 * ，会收集上述扩展名（同名 .hdr+.img 只读一套）

2. 标签（训练必填；测试可选）
   • 同样支持 .mat / .img / .dat / .hdr / .lbl / .npy / .tif
   • 二维图，与图像空间尺寸一致；1 波段 ENVI 也可
   • 类别编号 1..K，0 为背景/未标注。不要用 0..K-1 当类别号（软件会尝试自动 +1）
   • 「类别数」必须等于标签中的最大类别号。默认 24 只适用于 24 类矿物图
   • 立方体应为 I/F（大约 0–1）。若数值经常大于 1（例如 I/F×10000），软件会自动缩放
   • 多块 tile 训练时，标签必须是整幅拼接图；文件按 tile_2、tile_10 数字顺序读取

3. 预处理
   • 训练时在训练数据上拟合 process_model.pkl，再写出预处理立方体（.mat, data）
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
        "tile_pattern": "*",
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
