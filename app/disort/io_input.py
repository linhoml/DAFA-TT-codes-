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
    allow_partial: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Read the CRISM DISORT input directory (Fortran `input\\...` files).

    Parameters
    ----------
    input_dir : folder containing wavelength.txt, s0.txt and atmospheric tables
                (either directly or under an ``input`` subdirectory).
    n_wave : number of spectral bands; if None, inferred from wavelength.txt
    allow_partial : if True, missing atmospheric tables are filled with NaN/defaults
                    (for MCD-driven workflows that only need wavelength/s0 from disk).
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

    def maybe_p(*names):
        try:
            return p(*names)
        except FileNotFoundError:
            if allow_partial:
                return None
            raise

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

    height = np.linspace(0.0, 80000.0, n_columns)
    co2_column = np.zeros(n_hours, dtype=np.float64)
    f0 = np.zeros(n_hours, dtype=np.float64)
    soz = np.zeros(n_hours, dtype=np.float64)
    press_surf = np.full(n_hours, 610.0, dtype=np.float64)
    temp_surf = np.full(n_hours, 220.0, dtype=np.float64)
    temp = np.full((n_hours, n_columns), 210.0, dtype=np.float64)
    press = np.logspace(np.log10(610.0), np.log10(0.1), n_columns)
    co2_mixradio = np.full((n_hours, n_columns), 0.95, dtype=np.float64)
    density = np.full((n_hours, n_columns), 0.02, dtype=np.float64)
    dust_re = np.full((n_hours, n_columns), 1.5e-6, dtype=np.float64)
    dust_mixradio = np.zeros((n_hours, n_columns), dtype=np.float64)
    watice_column = np.zeros(n_hours, dtype=np.float64)
    watice_mixradio = np.zeros((n_hours, n_columns), dtype=np.float64)
    watice_re = np.zeros((n_hours, n_columns), dtype=np.float64)
    wv_column = np.zeros(n_hours, dtype=np.float64)
    wv_mixradio = np.zeros((n_hours, n_columns), dtype=np.float64)
    vz = np.zeros((samples, lines), dtype=np.float64)
    pa = np.zeros((samples, lines), dtype=np.float64)
    rf_ra = np.zeros((samples, lines, n_wave), dtype=np.float64)

    def read_two(path, n, skip=10):
        if path is None:
            return None
        return _read_two_col(path, n, skip=skip)

    path = maybe_p("CO2 column(kgm2).txt")
    if path:
        _, co2_column[:] = read_two(path, n_hours)
    path = maybe_p("CO2 volume mixing ratio.txt")
    if path:
        height[:], co2_mixradio[0, :] = read_two(path, n_columns)
    path = maybe_p("Density(kgm3)day.txt")
    if path:
        _, density[0, :] = read_two(path, n_columns)
    path = maybe_p("Dust effective radius(m).txt")
    if path:
        height[:], dust_re[0, :] = read_two(path, n_columns)
    path = maybe_p("Dust mass mixing ratio(kgkg).txt")
    if path:
        height[:], dust_mixradio[0, :] = read_two(path, n_columns)

    path = maybe_p("Pressure(Pa)0h.txt")
    if path:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            _skip_header(f, 10)
            for i in range(n_columns):
                line = f.readline()
                parts = line.replace("D", "E").replace("d", "e").split()
                press[i] = float(parts[-1])

    path = maybe_p("Solar zenith angle(deg).txt")
    if path:
        _, soz[:] = read_two(path, n_hours)
    path = maybe_p("Surface Pressure(Pa).txt")
    if path:
        _, press_surf[:] = read_two(path, n_hours)
    path = maybe_p("Surface Temperature(K)day.txt")
    if path:
        _, temp_surf[:] = read_two(path, n_hours)
    path = maybe_p("Temperature(K)day.txt")
    if path:
        height[:], temp[0, :] = read_two(path, n_columns)

    path = maybe_p("Water ice column(kgm2).txt")
    if path:
        _, watice_column[:] = read_two(path, n_hours)
    path = maybe_p("Water ice mixing ratio.txt")
    if path:
        height[:], watice_mixradio[0, :] = read_two(path, n_columns)
    path = maybe_p("Water ice effective radius(m).txt")
    if path:
        height[:], watice_re[0, :] = read_two(path, n_columns)
    path = maybe_p("Water vapor column(kgm2).txt")
    if path:
        _, wv_column[:] = read_two(path, n_hours)
    path = maybe_p("Water vapor mixing ratio.txt")
    if path:
        height[:], wv_mixradio[0, :] = read_two(path, n_columns)

    # Optional observed radiance spectrum / cube file (Fortran: c9dbrad2.txt / rf_ra)
    rf_candidates = [
        "c9dbrad2.txt",
        "rf_ra.txt",
        "observed_radiance.txt",
        "observed_if.txt",  # legacy filename
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
