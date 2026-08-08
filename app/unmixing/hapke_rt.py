"""
Hapke radiative-transfer unmixing with optical-constant inversion.

Workflow
--------
1. Load mineral endmember **REFF** spectra (Excel).
2. For each endmember provide: density ρ, real index n (scalar), mean grain size D.
3. Invert wavelength-dependent imaginary index k(λ) from Hapke RT (REFF space).
4. Fit mass fractions of a mixed **I/F** spectrum / image by nonlinear least squares
   of the Hapke intimate-mixture forward model.
   Image I/F and endmember REFF are linked by:
       I/F = REFF × cos(i)
   where solar incidence i comes from the auxiliary cube (per pixel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .hapke import reflectance_to_ssa, ssa_to_reflectance


@dataclass
class HapkeEndmember:
    name: str  # mineral name (Excel row 1)
    wavelengths: np.ndarray  # μm, shape (nb,)
    reflectance: np.ndarray  # lab reflectance factor REFF, (nb,)
    density: float = 3.0  # g cm^-3
    n: float = 1.7  # real refractive index (constant w.r.t. wavelength)
    grain_size_um: float = 50.0  # mean particle diameter, μm
    k: Optional[np.ndarray] = None  # imaginary index (nb,)
    ssa: Optional[np.ndarray] = None  # single-scattering albedo (nb,)
    spectrum_id: str = ""  # Excel row 2 spectrum ID (optional metadata)
    lab_incidence_deg: float = 30.0  # REFF measurement incidence i
    lab_emission_deg: float = 0.0  # REFF measurement emission e
    lab_phase_deg: float = 30.0  # REFF measurement phase angle g
    source: str = ""
    selected: bool = True  # whether used in unmixing
    is_background: bool = False  # image featureless-background endmember

    def resample(self, target_wavelengths: np.ndarray) -> "HapkeEndmember":
        tw = np.asarray(target_wavelengths, dtype=float).ravel()
        w = np.asarray(self.wavelengths, dtype=float).ravel()

        def _interp(y):
            y = np.asarray(y, dtype=float)
            good = np.isfinite(w) & np.isfinite(y)
            if good.sum() < 2:
                return np.full(tw.shape, np.nan)
            return np.interp(tw, w[good], y[good], left=np.nan, right=np.nan)

        return HapkeEndmember(
            name=self.name,
            wavelengths=tw.copy(),
            reflectance=_interp(self.reflectance),
            density=float(self.density),
            n=float(self.n),
            grain_size_um=float(self.grain_size_um),
            k=None if self.k is None else _interp(self.k),
            ssa=None if self.ssa is None else _interp(self.ssa),
            spectrum_id=self.spectrum_id or "",
            lab_incidence_deg=float(self.lab_incidence_deg),
            lab_emission_deg=float(self.lab_emission_deg),
            lab_phase_deg=float(getattr(self, "lab_phase_deg", 30.0)),
            source=self.source + "→resampled",
            selected=bool(self.selected),
            is_background=bool(getattr(self, "is_background", False)),
        )


def specular_coeffs(n: float) -> Tuple[float, float]:
    """
    Approximate Hapke external / internal specular reflection coefficients.
    n = real refractive index relative to surrounding medium.
    """
    n = float(max(n, 1.01))
    # Hapke (1993/2012) practical approximations
    se = ((n - 1.0) / (n + 1.0)) ** 2 + 0.05
    si = 1.0 - 4.0 / (n * (n + 1.0) ** 2)
    se = float(np.clip(se, 0.0, 0.95))
    si = float(np.clip(si, 0.0, 0.999))
    return se, si


def ssa_from_k(
    k: np.ndarray,
    wavelength_um: np.ndarray,
    n: float,
    grain_size_um: float,
) -> np.ndarray:
    """
    Single-scattering albedo from optical constants (equivalent-slab Hapke).

    α = 4π k / λ,  Θ = exp(-α D),
    w = Se + (1-Se)(1-Si) Θ / (1 - Si Θ)
    """
    k = np.asarray(k, dtype=float)
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    d = max(float(grain_size_um), 1e-6)
    se, si = specular_coeffs(n)
    # absorption coefficient [1/μm]
    alpha = 4.0 * np.pi * np.maximum(k, 0.0) / np.maximum(wavelength_um, 1e-9)
    # internal transmission (simple exponential path ≈ D)
    theta = np.exp(-alpha * d)
    theta = np.clip(theta, 0.0, 1.0)
    w = se + (1.0 - se) * (1.0 - si) * theta / (1.0 - si * theta)
    return np.clip(w, 0.0, 0.999999)


def k_from_ssa(
    w: np.ndarray,
    wavelength_um: np.ndarray,
    n: float,
    grain_size_um: float,
) -> np.ndarray:
    """Invert SSA → k(λ) given n and mean grain size D."""
    w = np.asarray(w, dtype=float)
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    d = max(float(grain_size_um), 1e-6)
    se, si = specular_coeffs(n)

    k = np.full_like(w, np.nan, dtype=float)
    for i, wi in enumerate(w.ravel()):
        if not np.isfinite(wi):
            continue
        # Physical SSA cannot be below Se (pure surface reflection)
        wi = float(np.clip(wi, se + 1e-8, 0.999999))
        beta = (wi - se) / ((1.0 - se) * (1.0 - si) + 1e-15)
        # Θ = β / (1 + β Si)
        theta = beta / (1.0 + beta * si)
        theta = float(np.clip(theta, 1e-12, 1.0 - 1e-12))
        alpha = -np.log(theta) / d
        lam = float(wavelength_um.ravel()[i])
        k.ravel()[i] = alpha * lam / (4.0 * np.pi)
    return np.maximum(k, 0.0)


def invert_k_from_reflectance(
    reflectance: np.ndarray,
    wavelength_um: np.ndarray,
    n: float,
    grain_size_um: float,
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lab **REFF** → SSA (Hapke) → k(λ).

    ``reflectance`` must be reflectance factor REFF (not image I/F).

    Returns (k, ssa).
    """
    ssa = reflectance_to_ssa(
        np.asarray(reflectance, dtype=float),
        incidence_deg=incidence_deg,
        emission_deg=emission_deg,
    )
    k = k_from_ssa(ssa, wavelength_um, n=n, grain_size_um=grain_size_um)
    return k, ssa


def prepare_endmembers_k(
    endmembers: Sequence[HapkeEndmember],
    lab_incidence_deg: Optional[float] = None,
    lab_emission_deg: Optional[float] = None,
) -> List[HapkeEndmember]:
    """
    Fill k(λ) and ssa for each endmember from its REFF + physical params.

    Uses each endmember's ``lab_incidence_deg`` / ``lab_emission_deg`` by default.
    Optional ``lab_incidence_deg`` / ``lab_emission_deg`` override all endmembers
    (kept for backward compatibility).
    """
    out: List[HapkeEndmember] = []
    for em in endmembers:
        inc = float(
            lab_incidence_deg
            if lab_incidence_deg is not None
            else getattr(em, "lab_incidence_deg", 30.0)
        )
        emi = float(
            lab_emission_deg
            if lab_emission_deg is not None
            else getattr(em, "lab_emission_deg", 0.0)
        )
        k, ssa = invert_k_from_reflectance(
            em.reflectance,
            em.wavelengths,
            n=em.n,
            grain_size_um=em.grain_size_um,
            incidence_deg=inc,
            emission_deg=emi,
        )
        em2 = HapkeEndmember(
            name=em.name,
            wavelengths=np.asarray(em.wavelengths, dtype=float).copy(),
            reflectance=np.asarray(em.reflectance, dtype=float).copy(),
            density=float(em.density),
            n=float(em.n),
            grain_size_um=float(em.grain_size_um),
            k=k,
            ssa=ssa,
            spectrum_id=em.spectrum_id or "",
            lab_incidence_deg=inc,
            lab_emission_deg=emi,
            lab_phase_deg=float(getattr(em, "lab_phase_deg", 30.0)),
            source=em.source,
            selected=bool(getattr(em, "selected", True)),
            is_background=bool(getattr(em, "is_background", False)),
        )
        out.append(em2)
    return out


def intimate_mixture_ssa(
    mass_fractions: np.ndarray,
    endmember_ssa: np.ndarray,
    densities: np.ndarray,
    grain_sizes_um: np.ndarray,
) -> np.ndarray:
    """
    Hapke intimate mixture SSA.

    w = Σ (m_i σ_i w_i) / Σ (m_i σ_i),  σ_i ∝ 1/(ρ_i D_i)
    endmember_ssa: (nb, n_em)
    """
    m = np.asarray(mass_fractions, dtype=float).ravel()
    m = np.maximum(m, 0.0)
    s = m.sum()
    if s <= 0:
        m = np.ones_like(m) / m.size
    else:
        m = m / s

    W = np.asarray(endmember_ssa, dtype=float)  # (nb, n)
    rho = np.asarray(densities, dtype=float).ravel()
    d = np.asarray(grain_sizes_um, dtype=float).ravel()
    sigma = 1.0 / (np.maximum(rho, 1e-6) * np.maximum(d, 1e-6))
    weights = m * sigma  # (n,)
    denom = float(np.sum(weights)) + 1e-15
    return (W * weights[np.newaxis, :]).sum(axis=1) / denom


def forward_mixture_reff(
    mass_fractions: np.ndarray,
    endmembers: Sequence[HapkeEndmember],
    incidence_deg: float,
    emission_deg: float,
) -> np.ndarray:
    """Predict mixture **REFF** from mass fractions and prepared endmembers."""
    ssa_mat = np.column_stack([em.ssa for em in endmembers])
    dens = np.array([em.density for em in endmembers], dtype=float)
    grains = np.array([em.grain_size_um for em in endmembers], dtype=float)
    w_mix = intimate_mixture_ssa(mass_fractions, ssa_mat, dens, grains)
    return ssa_to_reflectance(w_mix, incidence_deg, emission_deg)


def _iff_scale_from_abundances(
    mass_fractions: np.ndarray,
    endmembers: Sequence[HapkeEndmember],
    incidence_deg: float,
) -> float:
    """
    Scale Hapke REFF → comparable I/F.

    - Excel mineral endmembers (REFF): multiply by cos(i)
    - Image background endmember (already I/F): scale factor 1 (no cos)

    For mixed abundances use mass-fraction-weighted scale.
    """
    from .hapke import cos_incidence

    f = np.asarray(mass_fractions, dtype=float).ravel()
    f = np.maximum(f, 0.0)
    s = float(f.sum())
    if s <= 0:
        f = np.ones_like(f) / max(f.size, 1)
    else:
        f = f / s
    mu0 = float(cos_incidence(incidence_deg))
    scales = np.array(
        [1.0 if getattr(em, "is_background", False) else mu0 for em in endmembers],
        dtype=float,
    )
    return float(np.dot(f, scales))


def forward_mixture_iff(
    mass_fractions: np.ndarray,
    endmembers: Sequence[HapkeEndmember],
    incidence_deg: float,
    emission_deg: float,
) -> np.ndarray:
    """
    Predict mixture quantity comparable to image **I/F**.

    Hapke forward gives REFF; mineral endmembers then use I/F = REFF×cos(i).
    Image-background endmember is already I/F from the cube, so it does **not**
    get an extra ×cos(i). Mixed cases use abundance-weighted scale.
    """
    reff = forward_mixture_reff(mass_fractions, endmembers, incidence_deg, emission_deg)
    scale = _iff_scale_from_abundances(mass_fractions, endmembers, incidence_deg)
    return np.asarray(reff, dtype=float) * scale


def forward_mixture_reflectance(
    mass_fractions: np.ndarray,
    endmembers: Sequence[HapkeEndmember],
    incidence_deg: float,
    emission_deg: float,
) -> np.ndarray:
    """Alias of ``forward_mixture_reff`` (backward compatible)."""
    return forward_mixture_reff(mass_fractions, endmembers, incidence_deg, emission_deg)


def _softmax_params(x: np.ndarray) -> np.ndarray:
    """Map unconstrained vector → simplex (mass fractions)."""
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-15)


def fit_mass_fractions(
    observed_iff: np.ndarray,
    endmembers: Sequence[HapkeEndmember],
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
    band_mask: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, np.ndarray]:
    """
    Nonlinear least squares for Hapke intimate-mixture mass fractions.

    ``observed_iff`` is image radiance factor **I/F**.
    Excel mineral endmembers are **REFF** (k/SSA inverted in REFF space);
    forward uses I/F = REFF × cos(i).
    Image-background endmember is already **I/F** from the cube and does
    **not** receive an extra ×cos(i).

    Uses an unconstrained parameterization (softmax) so abundances stay on the
    probability simplex (non-negative, sum to 1).

    progress_cb(nfev, max_nfev): optional callback for UI progress.
    """
    if any(em.ssa is None for em in endmembers):
        raise ValueError("端元尚未反演 k/SSA，请先调用 prepare_endmembers_k。")

    y = np.asarray(observed_iff, dtype=float).ravel()
    n_em = len(endmembers)
    if band_mask is None:
        band_mask = np.isfinite(y)
        for em in endmembers:
            band_mask &= np.isfinite(em.ssa) & np.isfinite(em.reflectance)
    band_mask = np.asarray(band_mask, dtype=bool)
    if band_mask.sum() < max(3, n_em):
        return {
            "abundance": np.full(n_em, np.nan),
            "reconstructed": np.full_like(y, np.nan),
            "reconstructed_reff": np.full_like(y, np.nan),
            "residual": np.full_like(y, np.nan),
            "rmse": np.array(np.nan),
            "success": np.array(False),
            "method": "hapke_nls",
        }

    # Restrict endmember SSA to valid bands for speed in residual
    ssa_mat = np.column_stack([em.ssa for em in endmembers])
    dens = np.array([em.density for em in endmembers], dtype=float)
    grains = np.array([em.grain_size_um for em in endmembers], dtype=float)
    y_m = y[band_mask]
    ssa_m = ssa_mat[band_mask, :]

    if x0 is None:
        x0 = np.zeros(n_em, dtype=float)  # equal fractions after softmax
    else:
        # convert fractions to logits roughly
        f0 = np.clip(np.asarray(x0, dtype=float), 1e-6, 1.0)
        f0 = f0 / f0.sum()
        x0 = np.log(f0)

    max_nfev = 200 * n_em
    nfev_count = {"n": 0}

    def residual(x):
        nfev_count["n"] += 1
        if progress_cb is not None and (nfev_count["n"] % 5 == 0 or nfev_count["n"] <= 2):
            progress_cb(nfev_count["n"], max_nfev)
        f = _softmax_params(x)
        w = intimate_mixture_ssa(f, ssa_m, dens, grains)
        reff = ssa_to_reflectance(w, incidence_deg, emission_deg)
        # minerals: ×cos(i); image background: no extra ×cos(i)
        scale = _iff_scale_from_abundances(f, endmembers, incidence_deg)
        iff = reff * scale
        return iff - y_m

    if progress_cb is not None:
        progress_cb(0, max_nfev)
    result = least_squares(residual, x0, method="lm", max_nfev=max_nfev)
    if progress_cb is not None:
        progress_cb(max_nfev, max_nfev)
    abund = _softmax_params(result.x)
    recon_reff = forward_mixture_reff(abund, endmembers, incidence_deg, emission_deg)
    recon_iff = forward_mixture_iff(abund, endmembers, incidence_deg, emission_deg)
    resid = y - recon_iff
    rmse = float(np.sqrt(np.mean((recon_iff[band_mask] - y_m) ** 2)))
    return {
        "abundance": abund,
        "reconstructed": recon_iff,       # comparable to observed I/F
        "reconstructed_reff": recon_reff, # Hapke REFF before I/F scaling
        "residual": resid,
        "rmse": np.array(rmse),
        "success": np.array(bool(result.success)),
        "nfev": np.array(result.nfev),
        "method": "hapke_nls",
        "incidence_deg": np.array(incidence_deg),
        "emission_deg": np.array(emission_deg),
        "mu0": np.array(float(np.cos(np.radians(float(incidence_deg))))),
    }


def fit_cube_mass_fractions(
    cube: np.ndarray,
    endmembers: Sequence[HapkeEndmember],
    incidence_deg: float = 30.0,
    emission_deg: float = 0.0,
    spatial_stride: int = 1,
    band_mask: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    per_pixel_geometry: Optional[Callable[[int, int], Tuple[float, float]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Whole-image Hapke NLS unmixing on **I/F** cube.

    per_pixel_geometry(row, col) -> (incidence_deg, emission_deg) if provided.
    Incidence is required for I/F = REFF × cos(i).
    """
    cube = np.asarray(cube, dtype=float)
    rows, cols, bands = cube.shape
    n_em = len(endmembers)
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
            # Hapke uses cos(i), cos(e); skip extreme grazing / invalid
            if abs(float(inc)) >= 89.5 or abs(float(emi)) >= 89.5:
                continue
        else:
            inc, emi = incidence_deg, emission_deg
        try:
            res = fit_mass_fractions(
                y, endmembers, incidence_deg=inc, emission_deg=emi, band_mask=band_mask
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
        "method": "hapke_nls",
        "stride": step,
    }
