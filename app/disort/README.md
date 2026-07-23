# DISORT atmospheric correction (Python port)

## Source mapping

| Fortran | Python |
|---------|--------|
| `main.f` (driver) | `correction.py` |
| `input_data.f` | `io_input.py` |
| `optical_data.f` | `optical_data.py` |
| `optical_propertise.f` | `optical_properties.py` |
| `GETMOM` / `BDRF` in `main.f` | `phase_function.py` |
| `DISORT.f` + `LINPAK.f` + … | `engine.py` via **PythonicDISORT** |

Original Fortran sources are kept in `fortran/` for reference.

## Data layout

Select a root folder containing:

```
root/
  input/
    wavelength.txt
    s0.txt
    CO2 column(kgm2).txt
    CO2 volume mixing ratio.txt
    Density(kgm3)day.txt
    ...
  optical/
    co2_hitran.txt
    h2o_hitran.txt
    mie_dust.dat
    mie_icewater.dat
    Qt_co2.txt
    Qt_h2o.txt
```

## Notes

- Observed spectra / cube values used by DISORT are **TOA radiance** (辐亮度),
  matched to DISORT intensity `UU` with beam flux `FBEAM = s0`.
  They are **not** I/F reflectance.
- `Tools → DISORT correction` in the hyperspectral app.
