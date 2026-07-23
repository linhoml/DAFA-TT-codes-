"""
Port of main.f CRISM DISORT atmospheric correction driver.

For each wavelength: build layered optical properties, binary-search
Lambertian albedo so modeled TOA radiance (DISORT intensity UU)
matches the observed radiance spectrum.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .engine import disort_toa_intensity
from .io_input import load_input_bundle
from .optical_data import load_optical_data
from .optical_properties import optical_calculate
from .phase_function import getmom


def radiance_to_if(radiance: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """
    Convert TOA radiance to I/F reflectance factor: I/F = π * L / F0.

    ``s0`` is the solar spectral flux used as DISORT FBEAM (same units as L*sr).
    This matches the common CRISM-style convention where a Lambertian surface
    with albedo 1 viewed/illuminated at nadir has I/F ≈ 1.
    """
    rad = np.asarray(radiance, dtype=np.float64)
    flux = np.asarray(s0, dtype=np.float64)
    if flux.ndim == 0:
        flux = np.full(rad.shape, float(flux))
    out = np.full(rad.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(rad) & np.isfinite(flux) & (flux > 0)
    out[ok] = np.pi * rad[ok] / flux[ok]
    return out


def _build_layer_props(
    j_wave: int,
    atm: dict,
    opt: dict,
    kab_co2: np.ndarray,
    kab_h2o: np.ndarray,
    nlyr: int,
    nmom: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build DTAUC, SSALB, PMOM for one wavelength (Fortran layer loop)."""
    height = atm["height"]
    density = atm["density"]
    press = atm["press"]
    temp = atm["temp"]
    press_surf = atm["press_surf"]
    temp_surf = atm["temp_surf"]
    co2_mix = atm["co2_mixradio"]
    dust_mix = atm["dust_mixradio"]
    dust_re = atm["dust_re"]
    watice_mix = atm["watice_mixradio"]
    watice_re = atm["watice_re"]
    wv_mix = atm["wv_mixradio"]
    wavelen = atm["wavelen"]

    na = 6.02214129e23
    dtauc = np.zeros(nlyr - 1, dtype=np.float64)
    ssalb = np.zeros(nlyr - 1, dtype=np.float64)
    pmom = np.zeros((nmom + 1, nlyr - 1), dtype=np.float64)

    for iod in range(nlyr - 1):
        nk = (nlyr - 1) - iod  # 1-based NK in Fortran: NLYR-1-IOD+1
        # Convert to 0-based index for height arrays
        nk0 = nk - 1
        nk1 = nk
        if nk1 >= height.size:
            nk1 = height.size - 1
            nk0 = max(0, nk1 - 1)
        hz = height[nk1] - height[nk0]

        co2 = co2_mix[0, nk0] * density[0, nk0] * na / 440.1 * hz
        dust = dust_mix[0, nk0]
        watice = watice_mix[0, nk0]
        wv = wv_mix[0, nk0] * density[0, nk0] * na / 440.1 * hz

        ns = press_surf[0] / (8.314 * temp_surf[0])
        # Fortran: 6.7*10E-5 / (wavelen*10E-6)**4
        dtauc_rl = (
            (6.7 * 10e-5)
            * (press[nk0] / (8.314 * temp[0, nk0]) * hz)
            / (((wavelen[j_wave] * 10e-6) ** 4) * ns * na + 1e-30)
        )

        dtauc_co2 = co2 * kab_co2[j_wave, nk0]
        dtauc_wv = wv * kab_h2o[j_wave, nk0]
        dtauc_dust = (
            3.0
            * opt["ext_dust"][j_wave]
            * dust
            * density[0, nk0]
            * hz
            / (4.0 * 2400.0 * max(dust_re[0, nk0], 1e-30))
        )
        if watice_re[0, nk0] > 0.0:
            dtauc_watice = (
                3.0
                * opt["ext_watice"][j_wave]
                * watice
                * density[0, nk0]
                * hz
                / (4.0 * 917.0 * watice_re[0, nk0])
            )
        else:
            dtauc_watice = 0.0

        dtauc[iod] = dtauc_rl + dtauc_co2 + dtauc_wv + dtauc_dust + dtauc_watice
        dtauc[iod] = max(float(dtauc[iod]), 1e-12)

        ssalb_dust = opt["ww_dust"][j_wave]
        ssalb_watice = opt["ww_watice"][j_wave] if dtauc_watice > 0 else 0.0
        ssalb[iod] = (
            dtauc_rl * 0.99999
            + dtauc_dust * ssalb_dust
            + dtauc_watice * ssalb_watice
        ) / dtauc[iod]

        gg_dust = opt["g_dust"][j_wave]
        gg_watice = opt["g_watice"][j_wave] if dtauc_watice > 0 else 0.0
        denom_g = dtauc[iod] * ssalb[iod] + 1e-30
        gg = (
            dtauc_dust * ssalb_dust * gg_dust
            + dtauc_watice * ssalb_watice * gg_watice
        ) / denom_g
        gg = float(np.clip(gg, -0.999999, 0.999999))
        pmom[:, iod] = getmom(3, gg, nmom)

    return dtauc, ssalb, pmom


def run_disort_correction(
    data_root: str,
    observed_radiance: Optional[np.ndarray] = None,
    wavelengths_um: Optional[np.ndarray] = None,
    n_wave: Optional[int] = None,
    n_hours: int = 24,
    n_columns: int = 35,
    nstr: int = 16,
    nlyr: int = 35,
    soz_deg: Optional[float] = None,
    voz_deg: float = 7.878,
    pa_deg: float = 65.657,
    err_tol: float = 0.01,
    max_iter: int = 100,
    band_indices: Optional[np.ndarray] = None,
    band_step: int = 1,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    observed_if: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Run the Fortran main.f correction workflow.

    Parameters
    ----------
    data_root : directory containing ``input/`` and ``optical/`` (or the files themselves)
    observed_radiance : optional observed TOA radiance spectrum (overrides file).
        Compared directly to DISORT intensity UU (same physical quantity as Fortran rf_ra).
    wavelengths_um : if provided with observed_radiance, interpolate onto table wavelengths
    band_step : subsample table wavelengths when ``band_indices`` is None
    observed_if : legacy alias of ``observed_radiance`` (same array; not I/F units)
    """
    if observed_radiance is None:
        observed_radiance = observed_if
    if observed_radiance is None and "observed_if" in kwargs:
        observed_radiance = kwargs.pop("observed_if")
    kwargs.pop("observed_if", None)
    # Ignore unknown leftover kwargs from older GUI versions
    kwargs.clear()

    # Resolve table size from wavelength.txt first
    atm_probe = load_input_bundle(data_root, n_wave=None, n_hours=n_hours, n_columns=n_columns)
    table_wave = atm_probe["wavelen"]
    n_table = int(table_wave.size)

    if observed_radiance is not None:
        obs = np.asarray(observed_radiance, dtype=np.float64).ravel()
        if wavelengths_um is not None and len(wavelengths_um) == len(obs):
            # interpolate observed radiance onto DISORT table wavelengths
            obs_on_table = np.interp(
                table_wave,
                np.asarray(wavelengths_um, dtype=np.float64),
                obs,
                left=np.nan,
                right=np.nan,
            )
        else:
            obs_on_table = np.full(n_table, np.nan)
            n_copy = min(n_table, obs.size)
            obs_on_table[:n_copy] = obs[:n_copy]
        atm_probe["rf_ra"][0, 0, :] = obs_on_table

    atm = atm_probe
    n_wave = n_table
    opt_root = data_root
    opt = load_optical_data(opt_root, n_wave=n_wave)

    kab_co2, kab_h2o = optical_calculate(
        atm["wavelen"],
        atm["press"],
        atm["temp"],
        opt,
        atm["co2_mixradio"],
        atm["wv_mixradio"],
    )

    nmom = nstr
    if soz_deg is None:
        soz_deg = float(atm["soz"][0]) if np.any(atm["soz"]) else 70.725
    # Fortran hard-codes these in the distributed main; allow override
    ssoz = float(soz_deg)
    mu0 = np.cos(np.deg2rad(ssoz))
    umu = np.cos(np.deg2rad(voz_deg))

    if band_indices is None:
        step = max(int(band_step), 1)
        band_indices = np.arange(0, n_wave, step)
    else:
        band_indices = np.asarray(band_indices, dtype=int)
        band_indices = band_indices[(band_indices >= 0) & (band_indices < n_wave)]

    albedo_out = np.full(n_wave, np.nan, dtype=np.float64)
    model_radiance = np.full(n_wave, np.nan, dtype=np.float64)
    n_total = len(band_indices)

    for count, j in enumerate(band_indices):
        j = int(j)
        if progress_cb is not None:
            progress_cb(count + 1, n_total, f"wavelength {atm['wavelen'][j]:.4f} μm")

        rad_target = float(atm["rf_ra"][0, 0, j])
        if not np.isfinite(rad_target) or rad_target <= 0:
            continue

        dtauc, ssalb, pmom = _build_layer_props(
            j, atm, opt, kab_co2, kab_h2o, nlyr, nmom
        )
        fbeam = float(atm["s0"][j]) if j < atm["s0"].size else 1.0

        alb_low, alb_high = 0.1, 0.4
        albedo = 0.2
        rad_mod = np.nan
        for _ in range(max_iter):
            # DISORT UU is radiance/intensity; match observed radiance (not I/F)
            rad_mod = disort_toa_intensity(
                dtauc,
                ssalb,
                pmom,
                mu0=mu0,
                phi0_deg=0.0,
                umu=umu,
                phi_deg=pa_deg,
                fbeam=fbeam,
                albedo=albedo,
                nstr=nstr,
            )
            err = abs(rad_mod - rad_target) / (abs(rad_target) + 1e-12)
            if err < err_tol:
                break
            if rad_mod < rad_target:
                alb_low = albedo
            else:
                alb_high = albedo
            albedo = 0.5 * (alb_low + alb_high)

        albedo_out[j] = albedo
        model_radiance[j] = rad_mod

    observed = atm["rf_ra"][0, 0, :].copy()
    s0 = atm["s0"].copy()
    observed_if = radiance_to_if(observed, s0)
    model_if = radiance_to_if(model_radiance, s0)
    return {
        "wavelength": atm["wavelen"].copy(),
        "albedo": albedo_out,
        "model_radiance": model_radiance,
        "observed_radiance": observed,
        "model_if": model_if,
        "observed_if": observed_if,
        "s0": s0,
    }


def apply_disort_to_cube_spectrum(
    spectrum: np.ndarray,
    wavelengths_um: np.ndarray,
    data_root: str,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Convenience: correct one observed radiance spectrum using tables in data_root."""
    return run_disort_correction(
        data_root,
        observed_radiance=spectrum,
        wavelengths_um=wavelengths_um,
        n_wave=len(wavelengths_um),
        **kwargs,
    )
