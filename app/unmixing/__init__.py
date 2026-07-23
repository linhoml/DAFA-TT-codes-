"""
Spectral unmixing utilities for SpectralApp.

- Linear / sparse abundance estimation against an endmember library
- Hapke isotropic single-scattering-albedo (SSA) conversion for nonlinear
  photometric unmixing in SSA space (common planetary remote-sensing approach)
"""

from .library import SpectralLibrary, load_library
from .solvers import unmix_spectrum, unmix_cube
from .hapke import reflectance_to_ssa, ssa_to_reflectance, hapke_unmix_spectrum

__all__ = [
    "SpectralLibrary",
    "load_library",
    "unmix_spectrum",
    "unmix_cube",
    "reflectance_to_ssa",
    "ssa_to_reflectance",
    "hapke_unmix_spectrum",
]
