"""
Hapke isotropic single-scattering albedo (SSA) conversion.

Used to linearize mineral mixtures in SSA space (Mustard & Pieters style),
then map abundances back to reflectance.

Quantity conventions (this project)
---------------------------------
- Endmember lab spectra: reflectance factor **REFF**
- Hyperspectral image pixels: radiance factor **I/F**
- Conversion (solar incidence angle i from aux cube):
      I/F = REFF × cos(i)
      REFF = I/F / cos(i)

`ssa_to_reflectance` / `reflectance_to_ssa` operate on **REFF**.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

from .solvers import unmix_spectrum

ArrayLike = Union[float, np.ndarray]


def _H(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Chandrasekhar H-function approximation (Hapke)."""
    return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma)


def cos_incidence(incidence_deg: ArrayLike, min_mu0: float = 1e-6) -> ArrayLike:
    """μ0 = cos(i), clipped away from zero for numerical safety."""
    mu0 = np.cos(np.radians(np.asarray(incidence_deg, dtype=float)))
    return np.maximum(mu0, min_mu0)


def reff_to_iff(reff: np.ndarray, incidence_deg: ArrayLike) -> np.ndarray:
    """Radiance factor I/F from reflectance factor REFF: I/F = REFF × cos(i)."""
    return np.asarray(reff, dtype=float) * cos_incidence(incidence_deg)


def iff_to_reff(iff: np.ndarray, incidence_deg: ArrayLike) -> np.ndarray:
    """Reflectance factor REFF from radiance factor I/F: REFF = I/F / cos(i)."""
    return np.asarray(iff, dtype=float) / cos_incidence(incidence_deg)


def ssa_to_reflectance(
    w: np.ndarray,
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
    phase_deg: Optional[float] = None,
) -> np.ndarray:
    """
    Hapke bidirectional **reflectance factor (REFF)** for isotropic scatterers
    (no opposition surge / shadow hiding).

    Not I/F. Convert with ``reff_to_iff(reff, incidence_deg)`` when comparing
    to CRISM-style radiance factor.
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, 0.999999)
    mu0 = float(cos_incidence(incidence_deg))
    mu = float(cos_incidence(emission_deg))
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
    Invert Hapke isotropic model for SSA from **REFF** (not I/F).
    Uses a robust scalar bisection per band.
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
    Convert observation + endmembers (REFF) to SSA, unmix linearly, reconstruct REFF.
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
    # reconstructed SSA → REFF
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
