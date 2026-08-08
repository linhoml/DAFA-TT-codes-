"""Port of GETMOM / BDRF helpers from main.f."""

from __future__ import annotations

import numpy as np


def getmom(iphas: int, gg: float, nmom: int) -> np.ndarray:
    """
    Legendre expansion coefficients of the phase function.
    IPHAS=3: Henyey-Greenstein with asymmetry gg (as in Fortran GETMOM).
    Returns array shape (nmom+1,) with PMOM[0]=1.
    """
    pmom = np.zeros(nmom + 1, dtype=np.float64)
    pmom[0] = 1.0
    if iphas == 1:
        return pmom
    if iphas == 2:
        pmom[2] = 0.1
        return pmom
    if iphas == 3:
        if not (-1.0 < gg < 1.0):
            gg = float(np.clip(gg, -0.999999, 0.999999))
        for k in range(1, nmom + 1):
            pmom[k] = gg**k
        return pmom
    raise ValueError(f"Unsupported IPHAS={iphas}")


def bdrf_hapke(albedo: float, mu: float, mup: float, dphi_deg: float) -> float:
    """Port of BDRF Hapke model from main.f (degrees in dphi as in Fortran)."""
    cc = 0.3
    bb = 0.26
    ff = bb**2
    cos_dphi = np.cos(np.deg2rad(dphi_deg))
    ctheta = (1.0 - bb**2) / ((1.0 - 2 * bb * cos_dphi + bb**2) ** 1.5)
    theta = (1.0 - bb**2) * ff / ((1.0 + 2 * bb * cos_dphi + bb**2) ** 1.5)
    p = (1.0 + cc) / 2.0 * ctheta + (1.0 - cc) / 2.0 * theta
    hh = 0.06
    b0 = 1.0
    b = b0 / (1.0 + (1.0 / hh) * np.tan(np.deg2rad(dphi_deg)))
    gamma = np.sqrt(max(1.0 - albedo, 0.0))
    h0 = (1.0 + 2.0 * mup) / (1.0 + 2.0 * mup * gamma)
    h = (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * gamma)
    return albedo / 8.0 * mup / (mu + mup + 1e-12) * ((1.0 + b) * p + h0 * h - 1.0) * 15.0
