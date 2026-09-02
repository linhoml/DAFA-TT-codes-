"""HBM mineral identification (Plebani et al. 2022 / Banus/crism_ml)."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[3] / "third_party" / "crism_ml"


def ensure_crism_ml() -> Path:
    """Put vendored crism_ml on sys.path and return its root."""
    root = _VENDOR.resolve()
    if not (root / "crism_ml" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"未找到嵌入的 crism_ml 包：{root}\n"
            "请确认 third_party/crism_ml/crism_ml 存在。"
        )
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root
