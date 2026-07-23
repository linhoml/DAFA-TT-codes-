"""
DISORT atmospheric correction package (Python port of the CRISM Fortran driver).

Application modules ported from Fortran:
  - input_data.f       -> io_input.py
  - optical_data.f     -> optical_data.py
  - optical_propertise.f -> optical_properties.py
  - main.f (driver + GETMOM/BDRF) -> phase_function.py, correction.py

Radiative transfer solver:
  - Uses PythonicDISORT (pure-Python DISORT) via engine.py
  - Original Fortran DISORT sources kept under fortran/ for reference
"""

from .correction import run_disort_correction, radiance_to_if
from .mcd_client import fetch_mcd_profile, MCDProfileCache

__all__ = [
    "run_disort_correction",
    "radiance_to_if",
    "fetch_mcd_profile",
    "MCDProfileCache",
]
