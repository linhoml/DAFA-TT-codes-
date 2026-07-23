"""
Hapke isotropic single-scattering albedo (SSA) conversion.

Used to linearize mineral mixtures in SSA space (Mustard & Pieters style),
then map abundances back to reflectance.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .solvers import unmix_spectrum


def _H(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Chandrasekhar H-function approximation (Hapke)."""
    return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma)


def ssa_to_reflectance(
    w: np.ndarray,
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
    phase_deg: Optional[float] = None,
) -> np.ndarray:
    """
    Bidirectional reflectance factor (≈ radiance factor / I/F for collimated beam)
    for isotropic scatterers, no opposition surge / shadow hiding.
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, 0.999999)
    mu0 = np.cos(np.radians(float(incidence_deg)))
    mu = np.cos(np.radians(float(emission_deg)))
    mu0 = max(mu0, 1e-6)
    mu = max(mu, 1e-6)
    gamma = np.sqrt(np.maximum(1.0 - w, 0.0))
    # phase function P=1 (isotropic); ignore B(g) opposition
    r = (w / 4.0) * (mu0 / (mu0 + mu)) * _H(mu0, gamma) * _H(mu, gamma)
    return r


def reflectance_to_ssa(
    r: np.ndarray,
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
    max_iter: int = 40,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Invert Hapke isotropic model for single-scattering albedo w from reflectance r.
    Uses a robust scalar Newton / bisection per band.
    """
    r = np.asarray(r, dtype=float)
    out = np.full_like(r, np.nan, dtype=float)
    flat = r.ravel()
    res = np.empty_like(flat)
    for i, ri in enumerate(flat):
        if not np.isfinite(ri) or ri <= 0:
            res[i] = np.nan
            continue
        # physical upper bound: r max approaches 1 for high w; clamp input
        ri = float(min(max(ri, 1e-8), 0.99))
        lo, hi = 0.0, 0.999999
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fmid = float(ssa_to_reflectance(np.array([mid]), incidence_deg, emission_deg)[0] - ri)
            if abs(fmid) < tol:
                lo = hi = mid
                break
            if fmid > 0:
                hi = mid
            else:
                lo = mid
        res[i] = 0.5 * (lo + hi)
    return res.reshape(r.shape)


def hapke_unmix_spectrum(
    reflectance: np.ndarray,
    endmember_refl: np.ndarray,
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
    method: str = "nnls",
    sparsity: int = 3,
    sum_to_one: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert observation + endmembers to SSA, unmix linearly, reconstruct reflectance.
    """
    y_w = reflectance_to_ssa(reflectance, incidence_deg, emission_deg)
    A = np.asarray(endmember_refl, dtype=float)
    if A.ndim != 2:
        raise ValueError("endmember_refl must be (bands, n_em)")
    A_w = np.empty_like(A)
    for j in range(A.shape[1]):
        A_w[:, j] = reflectance_to_ssa(A[:, j], incidence_deg, emission_deg)

    mix = unmix_spectrum(
        y_w, A_w, method=method, sparsity=sparsity, sum_to_one=sum_to_one
    )
    # reconstructed SSA → reflectance
    recon_w = mix["reconstructed"]
    recon_r = ssa_to_reflectance(recon_w, incidence_deg, emission_deg)
    resid = reflectance - recon_r
    mask = np.isfinite(reflectance) & np.isfinite(recon_r)
    rmse = (
        float(np.sqrt(np.nanmean((reflectance[mask] - recon_r[mask]) ** 2)))
        if mask.any()
        else np.nan
    )
    return {
        "abundance": mix["abundance"],
        "reconstructed": recon_r,
        "reconstructed_ssa": recon_w,
        "observed_ssa": y_w,
        "residual": resid,
        "rmse": np.array(rmse),
        "support": mix["support"],
        "method": f"hapke+{mix['method']}",
        "incidence_deg": np.array(incidence_deg),
        "emission_deg": np.array(emission_deg),
    }
