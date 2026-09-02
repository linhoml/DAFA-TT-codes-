"""Convert app cubes to the CRISM-ML IF layout and remap class ids for display."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Same selection as crism_ml.io.crism_to_mat (438-band TRR3 I/F).
CRISM_BAND_SELECT = np.r_[433:185:-1, 170:-1:68]

# Neutral / bland (and the Leask spike artifact) are not mineral detections.
BACKGROUND_HBM_CODES = {38, 39}


def _lab_names() -> Tuple[dict, dict]:
    try:
        from identification.hbm import ensure_crism_ml

        ensure_crism_ml()
        from crism_ml.lab import BROAD_NAMES, FULL_NAMES

        return dict(BROAD_NAMES), dict(FULL_NAMES)
    except Exception:
        return {}, {}


def normalize_if_values(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Scale CRISM I/F stored as 0–100 / 0–10000 down to ~0–1."""
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return data, 1.0
    med = float(np.median(np.abs(finite)))
    if med > 500:
        scale = 10000.0
    elif med > 5:
        scale = 100.0
    else:
        return data, 1.0
    return data / np.float32(scale), scale


def select_hbm_cube(
    cube_hwb: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Take the 248 channels crism_ml expects from a HWB cube."""
    arr = np.ascontiguousarray(cube_hwb)
    height, width, bands = arr.shape
    if wavelengths is not None and len(wavelengths) == bands:
        try:
            from identification.hbm import ensure_crism_ml

            ensure_crism_ml()
            from crism_ml.preprocessing import BANDS
        except Exception:
            BANDS = None
        if BANDS is not None:
            wl = np.asarray(wavelengths, dtype=float).reshape(-1)
            if np.nanmax(wl) > 100:
                wl = wl / 1000.0
            idx = [int(np.nanargmin(np.abs(wl - float(target)))) for target in BANDS]
            if len(set(idx)) > 200:
                return np.ascontiguousarray(arr[:, :, idx])
    if bands >= 434:
        return np.ascontiguousarray(arr[:, :, CRISM_BAND_SELECT])
    if bands == 248:
        return arr
    raise ValueError(
        "HBM 需要 CRISM TRR3 I/F（约 438 波段，内部取 248 通道）"
        f"或已经是 248 波段的立方体，当前波段数={bands}。"
    )


def cube_to_if_mat(
    cube: np.ndarray,
    data_layout: str = "HWB",
    wavelengths: Optional[np.ndarray] = None,
) -> Dict:
    """Build the Matlab-like dict crism_ml.train expects (IF, x, y)."""
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(f"HBM 需要三维立方体，当前形状 {arr.shape}")
    layout = (data_layout or "HWB").upper()
    if layout == "BHW":
        arr = np.transpose(arr, (1, 2, 0))
    height, width, _bands = arr.shape
    selected = select_hbm_cube(arr, wavelengths=wavelengths)
    selected, _scale = normalize_if_values(selected)
    rows, cols = np.mgrid[0:height, 0:width]
    return {
        "IF": selected.reshape(-1, selected.shape[-1]).astype(np.float32, copy=False),
        "x": (cols.ravel() + 1).astype(np.int32),
        "y": (rows.ravel() + 1).astype(np.int32),
    }


def mineral_names(class_ids: Sequence[int]) -> List[str]:
    broad, full = _lab_names()
    names = []
    for code in class_ids:
        code = int(code)
        name = broad.get(code) or full.get(code) or f"class_{code}"
        names.append(f"{code} {name}")
    return names


def hbm_full_class_names(max_id: int = 40) -> List[str]:
    """ENVI class-name table indexed by HBM mineral code (id 1..max_id)."""
    return mineral_names(range(1, int(max_id) + 1))


def remap_prediction(pred: np.ndarray) -> Tuple[np.ndarray, List[str], List[int]]:
    """Map sparse HBM mineral codes to 1..K for the existing overlay widget."""
    shown = np.asarray(pred, dtype=np.int32)
    codes = sorted({int(v) for v in np.unique(shown) if int(v) > 0})
    display = np.zeros(shown.shape, dtype=np.int16)
    for index, code in enumerate(codes, start=1):
        display[shown == code] = index
    return display, mineral_names(codes), codes


def _zero_background(code_map: np.ndarray) -> np.ndarray:
    out = np.asarray(code_map, dtype=np.int32).copy()
    out[np.isin(out, list(BACKGROUND_HBM_CODES))] = 0
    return out


def build_hbm_display(
    filtered: np.ndarray,
    unfiltered: Optional[np.ndarray] = None,
    region: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str], List[int], str]:
    """Choose a mineral map that actually has pixels, then remap to 1..K.

    Preference: detected patches → confidence-filtered pixels → unfiltered
    argmax with bland/artifact hidden. Overlay and ENVI must share the same
    1..K ids so class names are not attached to empty HBM codes.
    """
    candidates = [
        (region, "regions"),
        (filtered, "filtered"),
        (None if unfiltered is None else _zero_background(unfiltered), "unfiltered"),
    ]
    for candidate, mode in candidates:
        if candidate is None:
            continue
        arr = np.asarray(candidate)
        if np.any(arr > 0):
            display, names, codes = remap_prediction(arr)
            return display, names, codes, mode
    display, names, codes = remap_prediction(np.asarray(filtered))
    return display, names, codes, "empty"


def class_pixel_counts(display: np.ndarray, names: Sequence[str]) -> List[str]:
    shown = np.asarray(display)
    lines = []
    for index, name in enumerate(names, start=1):
        lines.append(f"{index}  {name}: {int(np.sum(shown == index))} px")
    return lines
