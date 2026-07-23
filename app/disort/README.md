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

## GUI workflow (Tools → DISORT correction)

1. **加载辐亮度图像** → left-top false-color
2. **加载辅助信息图像** → left-bottom shows band 13 (local time)
   - band1 solar incidence, band2 emission, band3 phase,
     band4 lat, band5 lon, band13 local time (hours)
3. **单光谱计算** → click a radiance pixel; MCD profile from lat/lon/LT + user Ls
4. **图像处理** → whole image with spatial stride + MCD cache

Atmospheric profiles prefer local `fmcd`/`mcd-python`, then MCD web CGI,
then fall back to tables under `input/`.

## Notes

- Observed spectra used by the solver are **TOA radiance**,
  matched to DISORT intensity `UU` with beam flux `FBEAM = s0`.
- For display, radiance is converted to I/F as ``I/F = π · L / F0``
  (`s0`), and plotted with modeled I/F and retrieved surface albedo.
