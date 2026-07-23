"""
Spectral unmixing utilities for SpectralApp.

- Sparse / linear abundance estimation against an endmember library
- Hapke RT: Excel reflectance → k(λ) inversion → nonlinear mass-fraction fit
"""

from .library import SpectralLibrary, load_library
from .solvers import unmix_spectrum, unmix_cube
from .hapke import reflectance_to_ssa, ssa_to_reflectance, hapke_unmix_spectrum
from .hapke_rt import (
    HapkeEndmember,
    prepare_endmembers_k,
    fit_mass_fractions,
    fit_cube_mass_fractions,
)
from .excel_endmembers import load_endmembers_excel

__all__ = [
    "SpectralLibrary",
    "load_library",
    "unmix_spectrum",
    "unmix_cube",
    "reflectance_to_ssa",
    "ssa_to_reflectance",
    "hapke_unmix_spectrum",
    "HapkeEndmember",
    "prepare_endmembers_k",
    "fit_mass_fractions",
    "fit_cube_mass_fractions",
    "load_endmembers_excel",
]
