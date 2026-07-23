"""
DISORT intensity engine.

Original Fortran DISORT.f / LINPAK / ErrPack are retained under fortran/.
This module provides a Python-callable solver using PythonicDISORT
(pure-Python DISORT reimplementation) with an interface matching the
Fortran driver usage: return TOA intensity UU(1,1,1) for one geometry.
"""

from __future__ import annotations

import numpy as np

try:
    from PythonicDISORT import pydisort
    from PythonicDISORT import subroutines as pd_sub
except Exception as exc:  # pragma: no cover
    pydisort = None
    pd_sub = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


def disort_toa_intensity(
    dtauc: np.ndarray,
    ssalb: np.ndarray,
    pmom: np.ndarray,
    mu0: float,
    phi0_deg: float,
    umu: float,
    phi_deg: float,
    fbeam: float,
    albedo: float,
    nstr: int = 16,
) -> float:
    """
    Compute TOA intensity for a multi-layer atmosphere + Lambertian surface.

    Parameters
    ----------
    dtauc : (nlyr,) layer optical depths (top to bottom)
    ssalb : (nlyr,) single-scattering albedo
    pmom  : (nmom+1, nlyr) or (nlyr, nmom+1) Legendre coeffs, PMOM[0]=1
    mu0   : cos(solar zenith)
    umu   : cos(view zenith)
    fbeam : beam flux at TOA (Fortran FBEAM)
    albedo: Lambertian surface albedo (Fortran LAMBER=.TRUE.)
    """
    if pydisort is None:
        raise ImportError(
            "PythonicDISORT is required for DISORT correction. "
            f"Install with: pip install PythonicDISORT\nOriginal error: {_IMPORT_ERR}"
        )

    dtauc = np.asarray(dtauc, dtype=np.float64).ravel()
    ssalb = np.asarray(ssalb, dtype=np.float64).ravel()
    nlyr = int(dtauc.size)
    alb = float(np.clip(albedo, 0.0, 1.0))
    mu0 = float(np.clip(abs(mu0), 1e-6, 1.0))
    umu = float(np.clip(abs(umu), 1e-6, 1.0))
    phi0 = float(np.deg2rad(phi0_deg))
    phi = float(np.deg2rad(phi_deg))
    i0 = float(max(fbeam, 0.0))

    if nlyr == 0:
        # No atmosphere: Lambert intensity I = F0 * a * mu0 / pi
        return float(i0 * alb * mu0 / np.pi)

    if ssalb.size != nlyr:
        raise ValueError("DTAUC and SSALB size mismatch")

    dtauc = np.maximum(dtauc, 1e-12)
    ssalb = np.clip(ssalb, 0.0, 0.999999)
    tau_arr = np.cumsum(dtauc)

    pm = np.asarray(pmom, dtype=np.float64)
    if pm.ndim != 2:
        raise ValueError("PMOM must be 2-D")
    # Accept (nmom+1, nlyr) Fortran layout or (nlyr, nmom+1)
    if pm.shape[1] == nlyr and pm.shape[0] != nlyr:
        leg = pm.T.copy()
    elif pm.shape[0] == nlyr:
        leg = pm.copy()
    else:
        raise ValueError(f"PMOM shape {pm.shape} incompatible with nlyr={nlyr}")

    # PythonicDISORT expects unweighted Legendre coeffs in [0, 1]
    leg = np.clip(leg, 0.0, 1.0)
    nstr = int(nstr)
    if nstr % 2:
        nstr += 1
    if leg.shape[1] < nstr + 1:
        pad = np.zeros((nlyr, nstr + 1 - leg.shape[1]))
        leg = np.hstack([leg, pad])
    else:
        leg = leg[:, : nstr + 1]

    # Lambertian BDRF: only Fourier mode 0 = albedo/π (docs §1.5)
    bdrf_modes = [alb / np.pi]

    _mu_arr, _flux_up, _flux_down, _u0, u = pydisort(
        tau_arr,
        ssalb,
        nstr,
        leg,
        mu0,
        i0,
        phi0,
        only_flux=False,
        BDRF_Fourier_modes=bdrf_modes,
        NT_cor=False,
    )

    u_interp = pd_sub.interpolate(u)
    # Intensity at TOA (tau=0), upward view +umu, azimuth phi
    val = u_interp(np.array([umu]), 0.0, np.array([phi]))
    return float(np.asarray(val).ravel()[0])
