"""Port of input_data.f — read atmospheric / CRISM input tables."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np


def _skip_header(f, n=10):
    for _ in range(n):
        f.readline()


def _read_two_col(path, n, skip=10):
    """Read n rows of two-column numeric data after optional header lines."""
    col0 = np.zeros(n, dtype=np.float64)
    col1 = np.zeros(n, dtype=np.float64)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        _skip_header(f, skip)
        for i in range(n):
            parts = f.readline().split()
            if len(parts) < 2:
                raise ValueError(f"Unexpected format in {path} at data row {i + 1}")
            col0[i] = float(parts[0])
            col1[i] = float(parts[1])
    return col0, col1


def load_input_bundle(
    input_dir: str,
    n_wave: int | None = None,
    n_hours: int = 24,
    n_columns: int = 35,
    samples: int = 1,
    lines: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Read the CRISM DISORT input directory (Fortran `input\\...` files).

    Parameters
    ----------
    input_dir : folder containing wavelength.txt, s0.txt and atmospheric tables
                (either directly or under an ``input`` subdirectory).
    n_wave : number of spectral bands; if None, inferred from wavelength.txt
    """
    base = input_dir
    nested = os.path.join(input_dir, "input")
    if os.path.isdir(nested):
        base = nested

    def p(*names):
        for name in names:
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"Missing input file among {names} under {base}")

    # wavelength / solar flux may live next to atmospheric tables
    wl_path = p("wavelength.txt")
    s0_path = p("s0.txt")

    wavelen = np.loadtxt(wl_path, dtype=np.float64)
    if wavelen.ndim > 1:
        wavelen = wavelen[:, 0]
    wavelen = np.asarray(wavelen, dtype=np.float64).ravel()
    if n_wave is None:
        n_wave = int(wavelen.size)
    if wavelen.size < n_wave:
        raise ValueError(f"wavelength.txt has {wavelen.size} bands, need {n_wave}")
    wavelen = wavelen[:n_wave]

    s0_raw = np.loadtxt(s0_path, dtype=np.float64)
    if s0_raw.ndim == 1:
        s0 = s0_raw[:n_wave]
    else:
        s0 = s0_raw[:n_wave, -1]

    height = np.zeros(n_columns, dtype=np.float64)
    co2_column = np.zeros(n_hours, dtype=np.float64)
    f0 = np.zeros(n_hours, dtype=np.float64)
    soz = np.zeros(n_hours, dtype=np.float64)
    press_surf = np.zeros(n_hours, dtype=np.float64)
    temp_surf = np.zeros(n_hours, dtype=np.float64)
    temp = np.zeros((n_hours, n_columns), dtype=np.float64)
    press = np.zeros(n_columns, dtype=np.float64)
    co2_mixradio = np.zeros((n_hours, n_columns), dtype=np.float64)
    density = np.zeros((n_hours, n_columns), dtype=np.float64)
    dust_re = np.zeros((n_hours, n_columns), dtype=np.float64)
    dust_mixradio = np.zeros((n_hours, n_columns), dtype=np.float64)
    watice_column = np.zeros(n_hours, dtype=np.float64)
    watice_mixradio = np.zeros((n_hours, n_columns), dtype=np.float64)
    watice_re = np.zeros((n_hours, n_columns), dtype=np.float64)
    wv_column = np.zeros(n_hours, dtype=np.float64)
    wv_mixradio = np.zeros((n_hours, n_columns), dtype=np.float64)
    vz = np.zeros((samples, lines), dtype=np.float64)
    pa = np.zeros((samples, lines), dtype=np.float64)
    rf_ra = np.zeros((samples, lines, n_wave), dtype=np.float64)

    _, co2_column[:] = _read_two_col(p("CO2 column(kgm2).txt"), n_hours)
    height[:], co2_mixradio[0, :] = _read_two_col(p("CO2 volume mixing ratio.txt"), n_columns)
    _, density[0, :] = _read_two_col(p("Density(kgm3)day.txt"), n_columns)
    height[:], dust_re[0, :] = _read_two_col(p("Dust effective radius(m).txt"), n_columns)
    height[:], dust_mixradio[0, :] = _read_two_col(
        p("Dust mass mixing ratio(kgkg).txt"), n_columns
    )

    # Pressure may be scientific format
    with open(p("Pressure(Pa)0h.txt"), "r", encoding="utf-8", errors="ignore") as f:
        _skip_header(f, 10)
        for i in range(n_columns):
            line = f.readline()
            parts = line.replace("D", "E").replace("d", "e").split()
            press[i] = float(parts[-1])

    _, soz[:] = _read_two_col(p("Solar zenith angle(deg).txt"), n_hours)
    _, press_surf[:] = _read_two_col(p("Surface Pressure(Pa).txt"), n_hours)
    _, temp_surf[:] = _read_two_col(p("Surface Temperature(K)day.txt"), n_hours)
    height[:], temp[0, :] = _read_two_col(p("Temperature(K)day.txt"), n_columns)

    htmp, watice_column[:] = _read_two_col(p("Water ice column(kgm2).txt"), n_hours)
    height[:], watice_mixradio[0, :] = _read_two_col(
        p("Water ice mixing ratio.txt"), n_columns
    )
    height[:], watice_re[0, :] = _read_two_col(
        p("Water ice effective radius(m).txt"), n_columns
    )
    _, wv_column[:] = _read_two_col(p("Water vapor column(kgm2).txt"), n_hours)
    height[:], wv_mixradio[0, :] = _read_two_col(
        p("Water vapor mixing ratio.txt"), n_columns
    )

    # Optional observed I/F cube / spectrum file (Fortran: c9dbrad2.txt)
    rf_candidates = [
        "c9dbrad2.txt",
        "rf_ra.txt",
        "observed_if.txt",
    ]
    rf_path = None
    for name in rf_candidates:
        cand = os.path.join(base, name)
        if os.path.isfile(cand):
            rf_path = cand
            break
    if rf_path is not None:
        raw = np.loadtxt(rf_path, dtype=np.float64)
        raw = np.atleast_1d(raw)
        if raw.ndim == 1:
            rf_ra[0, 0, : min(n_wave, raw.size)] = raw[:n_wave]
        elif raw.ndim == 2:
            # rows = bands or samples
            if raw.shape[0] >= n_wave:
                rf_ra[0, 0, :] = raw[:n_wave, 0]
            else:
                rf_ra[0, 0, : raw.shape[1]] = raw[0, :n_wave]

    return {
        "wavelen": wavelen,
        "s0": np.asarray(s0, dtype=np.float64),
        "height": height,
        "co2_column": co2_column,
        "f0": f0,
        "soz": soz,
        "press_surf": press_surf,
        "temp_surf": temp_surf,
        "temp": temp,
        "press": press,
        "co2_mixradio": co2_mixradio,
        "density": density,
        "dust_re": dust_re,
        "dust_mixradio": dust_mixradio,
        "watice_column": watice_column,
        "watice_mixradio": watice_mixradio,
        "watice_re": watice_re,
        "wv_column": wv_column,
        "wv_mixradio": wv_mixradio,
        "vz": vz,
        "pa": pa,
        "rf_ra": rf_ra,
        "input_dir": base,
    }
