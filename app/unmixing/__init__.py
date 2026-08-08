"""
Spectral unmixing utilities for SpectralApp.

- Sparse unmixing (SUNSAL) in SSA space from Excel endmembers
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
from .excel_endmembers import load_endmembers_excel, load_sparse_endmembers_excel
from .sunsal import sunsal, soft
from .sparse_ssa import sparse_unmix_ssa, sparse_unmix_cube_ssa, endmember_reff_to_ssa

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
    "load_sparse_endmembers_excel",
    "sunsal",
    "soft",
    "sparse_unmix_ssa",
    "sparse_unmix_cube_ssa",
    "endmember_reff_to_ssa",
]
