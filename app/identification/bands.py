"""Select and resample the 1.02–2.6 μm identification range."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .defaults import APPLY_WL_MAX, APPLY_WL_MIN, TARGET_BAND_NUM


def crism_target_wavelengths(n: int = TARGET_BAND_NUM) -> np.ndarray:
    return np.linspace(APPLY_WL_MIN, APPLY_WL_MAX, int(n), dtype=np.float64)


def _as_um(wavelengths: np.ndarray) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    finite = wl[np.isfinite(wl)]
    if finite.size and float(np.nanmax(np.abs(finite))) > 100:
        wl = wl / 1000.0
    return wl


def cube_to_identification_range(
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep 1.02–2.6 μm and return a 240-channel HWB cube.

    If wavelengths are present they are used to crop, then interpolated onto a
    uniform 240-point grid so train and apply share the same channel count.
    Without wavelengths, CRISM 438-band cubes use original 1-based bands 4–243.
    """
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected H×W×B cube, got shape={cube.shape}")

    bands = cube.shape[-1]
    wl = None if wavelengths is None else _as_um(wavelengths)
    if wl is not None and wl.size != bands:
        raise ValueError(
            f"Wavelength count {wl.size} does not match cube bands {bands}"
        )

    target_wl = crism_target_wavelengths()
    if wl is not None:
        mask = (wl >= APPLY_WL_MIN) & (wl <= APPLY_WL_MAX) & np.isfinite(wl)
        if int(np.sum(mask)) >= 8:
            sub = cube[:, :, mask]
            sub_wl = wl[mask]
            if sub.shape[-1] == TARGET_BAND_NUM and np.allclose(
                sub_wl, target_wl, rtol=0, atol=1e-3
            ):
                return sub.astype(np.float32, copy=False), sub_wl.astype(np.float64)
            return _interp_bands(sub, sub_wl, target_wl), target_wl

    if bands == 438:
        sliced = cube[:, :, 3:243]
        return sliced.astype(np.float32, copy=False), target_wl
    if bands == TARGET_BAND_NUM:
        return cube.astype(np.float32, copy=False), (
            wl.astype(np.float64) if wl is not None else target_wl
        )

    if wl is not None and wl.size == bands:
        return _interp_bands(cube, wl, target_wl), target_wl

    raise ValueError(
        f"无法将 {bands} 个波段映射到 {APPLY_WL_MIN}–{APPLY_WL_MAX} μm 的 "
        f"{TARGET_BAND_NUM} 通道。请确认影像含波长信息，或本身为 CRISM 438/240 波段。"
    )


def cube_to_crism_240(
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias."""
    return cube_to_identification_range(cube, wavelengths)


def _interp_bands(
    cube: np.ndarray,
    src_wl: np.ndarray,
    tgt_wl: np.ndarray,
) -> np.ndarray:
    order = np.argsort(src_wl)
    src = np.asarray(src_wl, dtype=np.float64)[order]
    data = np.asarray(cube, dtype=np.float32)[:, :, order]
    height, width, _ = data.shape
    flat = data.reshape(-1, data.shape[-1])
    out = np.empty((flat.shape[0], tgt_wl.size), dtype=np.float32)
    finite_src = np.isfinite(src)
    if int(np.sum(finite_src)) < 2:
        raise ValueError(
            f"有效波长不足，无法重采样到 {APPLY_WL_MIN}–{APPLY_WL_MAX} μm。"
        )
    src_ok = src[finite_src]
    for i, spectrum in enumerate(flat):
        spec = spectrum[finite_src]
        good = np.isfinite(spec)
        if int(np.sum(good)) < 2:
            out[i] = np.nan
            continue
        out[i] = np.interp(tgt_wl, src_ok[good], spec[good]).astype(np.float32)
    return out.reshape(height, width, tgt_wl.size)
