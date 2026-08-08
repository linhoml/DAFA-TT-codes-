"""
Sparse unmixing in single-scattering albedo (SSA) space (SUNSAL).

Workflow
--------
1. Endmember Excel reflectance (REFF) → SSA via Hapke (lab geometry).
2. Image I/F → REFF = I/F / cos(i) → SSA (observation geometry).
3. Sparse unmixing (SUNSAL) in SSA space.
4. Reconstruct SSA → REFF → I/F for display.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .hapke import iff_to_reff, reflectance_to_ssa, reff_to_iff, ssa_to_reflectance
from .sunsal import sunsal


def endmember_reff_to_ssa(
    spectra: np.ndarray,
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
) -> np.ndarray:
    """Convert endmember REFF matrix (bands, n_em) to SSA."""
    A = np.asarray(spectra, dtype=float)
    out = np.empty_like(A)
    for j in range(A.shape[1]):
        out[:, j] = reflectance_to_ssa(A[:, j], incidence_deg, emission_deg)
    return out


def iff_spectrum_to_ssa(
    iff: np.ndarray,
    incidence_deg: float,
    emission_deg: float = 0.0,
) -> np.ndarray:
    """Image I/F spectrum → REFF → SSA."""
    reff = iff_to_reff(iff, incidence_deg)
    return reflectance_to_ssa(reff, incidence_deg, emission_deg)


def ssa_to_iff(
    ssa: np.ndarray,
    incidence_deg: float,
    emission_deg: float = 0.0,
) -> np.ndarray:
    """SSA → REFF → I/F for plotting against the image."""
    reff = ssa_to_reflectance(ssa, incidence_deg, emission_deg)
    return reff_to_iff(reff, incidence_deg)


def sparse_unmix_ssa(
    observed_iff: np.ndarray,
    endmember_ssa: np.ndarray,
    incidence_deg: float,
    emission_deg: float = 0.0,
    lambda_: float = 1e-4,
    positivity: bool = True,
    addone: bool = True,
    al_iters: int = 100,
    band_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert one I/F spectrum to SSA and unmix with SUNSAL against endmember SSA.

    endmember_ssa : (bands, n_em)
    """
    y_iff = np.asarray(observed_iff, dtype=float).ravel()
    A = np.asarray(endmember_ssa, dtype=float)
    n_em = A.shape[1]
    if band_mask is None:
        band_mask = np.isfinite(y_iff) & np.all(np.isfinite(A), axis=1)
    band_mask = np.asarray(band_mask, dtype=bool)
    if band_mask.sum() < max(3, n_em):
        return {
            "abundance": np.full(n_em, np.nan),
            "reconstructed_iff": np.full_like(y_iff, np.nan),
            "reconstructed_ssa": np.full_like(y_iff, np.nan),
            "observed_ssa": np.full_like(y_iff, np.nan),
            "rmse": np.array(np.nan),
            "method": "sunsal_ssa",
        }

    y_ssa_full = iff_spectrum_to_ssa(y_iff, incidence_deg, emission_deg)
    y_m = y_ssa_full[band_mask]
    A_m = A[band_mask, :]
    # Drop non-finite SSA bands
    good = np.isfinite(y_m) & np.all(np.isfinite(A_m), axis=1)
    if good.sum() < max(3, n_em):
        return {
            "abundance": np.full(n_em, np.nan),
            "reconstructed_iff": np.full_like(y_iff, np.nan),
            "reconstructed_ssa": np.full_like(y_iff, np.nan),
            "observed_ssa": y_ssa_full,
            "rmse": np.array(np.nan),
            "method": "sunsal_ssa",
        }

    res = sunsal(
        A_m[good, :],
        y_m[good],
        lambda_=float(lambda_),
        positivity=positivity,
        addone=addone,
        al_iters=al_iters,
    )
    abund = np.asarray(res["abundance"], dtype=float).ravel()
    recon_ssa = A @ abund
    recon_iff = ssa_to_iff(recon_ssa, incidence_deg, emission_deg)
    resid = y_iff - recon_iff
    mask_rmse = band_mask & np.isfinite(y_iff) & np.isfinite(recon_iff)
    rmse = (
        float(np.sqrt(np.mean((y_iff[mask_rmse] - recon_iff[mask_rmse]) ** 2)))
        if mask_rmse.any()
        else float("nan")
    )
    return {
        "abundance": abund,
        "reconstructed_iff": recon_iff,
        "reconstructed_ssa": recon_ssa,
        "observed_ssa": y_ssa_full,
        "residual": resid,
        "rmse": np.array(rmse),
        "res_p": res["res_p"],
        "res_d": res["res_d"],
        "n_iter": res["n_iter"],
        "method": "sunsal_ssa",
        "incidence_deg": np.array(incidence_deg),
        "emission_deg": np.array(emission_deg),
        "lambda": np.array(float(lambda_)),
    }


def sparse_unmix_cube_ssa(
    cube_iff: np.ndarray,
    endmember_ssa: np.ndarray,
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
    lambda_: float = 1e-4,
    positivity: bool = True,
    addone: bool = True,
    al_iters: int = 100,
    spatial_stride: int = 1,
    band_mask: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    per_pixel_geometry: Optional[Callable[[int, int], Tuple[float, float]]] = None,
) -> Dict[str, np.ndarray]:
    """Whole-image SUNSAL unmixing in SSA space. Cube is I/F."""
    cube = np.asarray(cube_iff, dtype=float)
    A = np.asarray(endmember_ssa, dtype=float)
    rows, cols, bands = cube.shape
    if A.shape[0] != bands:
        raise ValueError(f"波段数不匹配：cube={bands}, A={A.shape[0]}")
    n_em = A.shape[1]
    abund = np.full((rows, cols, n_em), np.nan, dtype=float)
    rmse = np.full((rows, cols), np.nan, dtype=float)
    step = max(1, int(spatial_stride))
    coords = [(r, c) for r in range(0, rows, step) for c in range(0, cols, step)]
    total = len(coords)
    for i, (r, c) in enumerate(coords):
        y = cube[r, c, :]
        if np.nanmean(np.isfinite(y)) < 0.3:
            continue
        if per_pixel_geometry is not None:
            try:
                inc, emi = per_pixel_geometry(r, c)
            except Exception:
                continue
            if not (np.isfinite(inc) and np.isfinite(emi)):
                continue
            if abs(float(inc)) >= 89.5 or abs(float(emi)) >= 89.5:
                continue
        else:
            inc, emi = incidence_deg, emission_deg
        try:
            res = sparse_unmix_ssa(
                y, A,
                incidence_deg=float(inc),
                emission_deg=float(emi),
                lambda_=lambda_,
                positivity=positivity,
                addone=addone,
                al_iters=al_iters,
                band_mask=band_mask,
            )
            abund[r, c, :] = res["abundance"]
            rmse[r, c] = float(res["rmse"])
        except Exception:
            continue
        if progress_cb is not None and (i % 20 == 0 or i + 1 == total):
            progress_cb(i + 1, total)
    return {
        "abundance": abund,
        "rmse": rmse,
        "method": "sunsal_ssa",
        "stride": step,
        "lambda": float(lambda_),
    }

