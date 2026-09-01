"""Convert app cubes to the CRISM-ML IF layout and remap class ids for display."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Same selection as crism_ml.io.crism_to_mat (438-band TRR3 I/F).
CRISM_BAND_SELECT = np.r_[433:185:-1, 170:-1:68]


def cube_to_if_mat(cube: np.ndarray, data_layout: str = "HWB") -> Dict:
    """Build the Matlab-like dict crism_ml.train expects (IF, x, y)."""
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(f"HBM 需要三维立方体，当前形状 {arr.shape}")
    layout = (data_layout or "HWB").upper()
    if layout == "BHW":
        arr = np.transpose(arr, (1, 2, 0))
    height, width, bands = arr.shape
    if bands >= 434:
        selected = np.ascontiguousarray(arr[:, :, CRISM_BAND_SELECT])
    elif bands == 248:
        selected = np.ascontiguousarray(arr)
    else:
        raise ValueError(
            "HBM 需要 CRISM TRR3 I/F（约 438 波段，内部取 248 通道）"
            f"或已经是 248 波段的立方体，当前波段数={bands}。"
        )
    rows, cols = np.mgrid[0:height, 0:width]
    return {
        "IF": selected.reshape(-1, selected.shape[-1]).astype(np.float32, copy=False),
        "x": (cols.ravel() + 1).astype(np.int32),
        "y": (rows.ravel() + 1).astype(np.int32),
    }


def mineral_names(class_ids: Sequence[int]) -> List[str]:
    try:
        from crism_ml.lab import BROAD_NAMES, FULL_NAMES
    except ImportError:
        BROAD_NAMES, FULL_NAMES = {}, {}
    names = []
    for code in class_ids:
        code = int(code)
        name = BROAD_NAMES.get(code) or FULL_NAMES.get(code) or f"class_{code}"
        names.append(f"{code} {name}")
    return names


def remap_prediction(pred: np.ndarray) -> Tuple[np.ndarray, List[str], List[int]]:
    """Map sparse HBM mineral codes to 1..K for the existing overlay widget."""
    shown = np.asarray(pred, dtype=np.int32)
    codes = sorted({int(v) for v in np.unique(shown) if int(v) > 0})
    display = np.zeros(shown.shape, dtype=np.int16)
    for index, code in enumerate(codes, start=1):
        display[shown == code] = index
    return display, mineral_names(codes), codes
