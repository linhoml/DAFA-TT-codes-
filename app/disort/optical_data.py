"""Port of optical_data.f — read gas/aerosol optical property tables."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np


def load_optical_data(optical_dir: str, n_wave: int) -> Dict[str, np.ndarray]:
    """
    Read optical property files (Fortran `optical\\...`).

    Looks in ``optical_dir`` or ``optical_dir/optical``.
    """
    base = optical_dir
    nested = os.path.join(optical_dir, "optical")
    if os.path.isdir(nested):
        base = nested
    # also allow parent/optical when user selected input folder
    parent_opt = os.path.join(os.path.dirname(optical_dir), "optical")
    if not os.path.isdir(base) and os.path.isdir(parent_opt):
        base = parent_opt

    def p(name: str) -> str:
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing optical file: {path}")
        return path

    wavnum_co2 = np.zeros(n_wave)
    sw_co2 = np.zeros(n_wave)
    gammaa_co2 = np.zeros(n_wave)
    gammas_co2 = np.zeros(n_wave)
    elower_co2 = np.zeros(n_wave)
    nn_co2 = np.zeros(n_wave)
    delta_co2 = np.zeros(n_wave)

    wavnum_h2o = np.zeros(n_wave)
    sw_h2o = np.zeros(n_wave)
    gammaa_h2o = np.zeros(n_wave)
    gammas_h2o = np.zeros(n_wave)
    elower_h2o = np.zeros(n_wave)
    nn_h2o = np.zeros(n_wave)
    delta_h2o = np.zeros(n_wave)

    ext_dust = np.zeros(n_wave)
    ww_dust = np.zeros(n_wave)
    g_dust = np.zeros(n_wave)
    ext_watice = np.zeros(n_wave)
    ww_watice = np.zeros(n_wave)
    g_watice = np.zeros(n_wave)

    # HITRAN-like fixed-width rows (fallback to free format)
    def read_hitran(path, outs):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i in range(n_wave):
                line = f.readline()
                if not line:
                    break
                try:
                    # F12.6,F12.6,E10.3,F5.4,F5.3,F10.4,F4.2,F8.6
                    wavenum = float(line[0:12])
                    outs[0][i] = float(line[12:24])
                    outs[1][i] = float(line[24:34])
                    outs[2][i] = float(line[34:39])
                    outs[3][i] = float(line[39:44])
                    outs[4][i] = float(line[44:54])
                    outs[5][i] = float(line[54:58])
                    outs[6][i] = float(line[58:66])
                except Exception:
                    parts = line.replace("D", "E").split()
                    # skip first wavenumber column if present
                    vals = [float(x) for x in parts]
                    if len(vals) >= 8:
                        (
                            outs[0][i],
                            outs[1][i],
                            outs[2][i],
                            outs[3][i],
                            outs[4][i],
                            outs[5][i],
                            outs[6][i],
                        ) = vals[1:8]
                    elif len(vals) >= 7:
                        (
                            outs[0][i],
                            outs[1][i],
                            outs[2][i],
                            outs[3][i],
                            outs[4][i],
                            outs[5][i],
                            outs[6][i],
                        ) = vals[:7]

    read_hitran(
        p("co2_hitran.txt"),
        [wavnum_co2, sw_co2, gammaa_co2, gammas_co2, elower_co2, nn_co2, delta_co2],
    )
    read_hitran(
        p("h2o_hitran.txt"),
        [wavnum_h2o, sw_h2o, gammaa_h2o, gammas_h2o, elower_h2o, nn_h2o, delta_h2o],
    )

    dust = np.loadtxt(p("mie_dust.dat"), dtype=np.float64)
    if dust.ndim == 1:
        dust = dust.reshape(1, -1)
    ext_dust[: dust.shape[0]] = dust[:n_wave, 1]
    ww_dust[: dust.shape[0]] = dust[:n_wave, 2]
    g_dust[: dust.shape[0]] = dust[:n_wave, 3]

    ice = np.loadtxt(p("mie_icewater.dat"), dtype=np.float64)
    if ice.ndim == 1:
        ice = ice.reshape(1, -1)
    ext_watice[: ice.shape[0]] = ice[:n_wave, 1]
    ww_watice[: ice.shape[0]] = ice[:n_wave, 2]
    g_watice[: ice.shape[0]] = ice[:n_wave, 3]

    return {
        "wavnum_co2": wavnum_co2,
        "sw_co2": sw_co2,
        "gammaa_co2": gammaa_co2,
        "gammas_co2": gammas_co2,
        "elower_co2": elower_co2,
        "nn_co2": nn_co2,
        "delta_co2": delta_co2,
        "wavnum_h2o": wavnum_h2o,
        "sw_h2o": sw_h2o,
        "gammaa_h2o": gammaa_h2o,
        "gammas_h2o": gammas_h2o,
        "elower_h2o": elower_h2o,
        "nn_h2o": nn_h2o,
        "delta_h2o": delta_h2o,
        "ext_dust": ext_dust,
        "ww_dust": ww_dust,
        "g_dust": g_dust,
        "ext_watice": ext_watice,
        "ww_watice": ww_watice,
        "g_watice": g_watice,
        "optical_dir": base,
    }
