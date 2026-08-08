"""Port of optical_propertise.f — gas absorption coefficient calculation."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def _load_qt_table(path: str, n: int = 301) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    return data[:n, 0], data[:n, 1]


def qe_co2(temp_row: np.ndarray, optical_dir: str) -> np.ndarray:
    """Look up CO2 partition function vs temperature (Fortran qe_co2)."""
    path = os.path.join(optical_dir, "Qt_co2.txt")
    tp, qq = _load_qt_table(path)
    qij = np.zeros(temp_row.size, dtype=np.float64)
    for j, t in enumerate(temp_row):
        idx = int(np.argmin(np.abs(tp - t)))
        # Fortran keeps last index with |tp-t|<0.5; closest is fine
        close = np.where(np.abs(tp - t) < 0.5)[0]
        if close.size:
            idx = int(close[-1])
        qij[j] = qq[idx]
    return qij


def qe_h2o(temp_row: np.ndarray, optical_dir: str) -> np.ndarray:
    path = os.path.join(optical_dir, "Qt_h2o.txt")
    tp, qq = _load_qt_table(path)
    qij = np.zeros(temp_row.size, dtype=np.float64)
    for j, t in enumerate(temp_row):
        idx = int(np.argmin(np.abs(tp - t)))
        close = np.where(np.abs(tp - t) < 0.5)[0]
        if close.size:
            idx = int(close[-1])
        qij[j] = qq[idx]
    return qij


def optical_calculate(
    wavelen: np.ndarray,
    press: np.ndarray,
    temp: np.ndarray,
    opt: dict,
    co2_mixradio: np.ndarray,
    wv_mixradio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute kab_co2 / kab_h2o with shape (n_wave, n_columns).
    Faithful port of optical_calculate (including H2O using CO2 line centers
    as in the original Fortran).
    """
    n_wave = wavelen.size
    n_columns = press.size
    temp_row = temp[0, :] if temp.ndim == 2 else temp

    kk = 1.38064e-15  # 1.38064*10E-16 in Fortran = 1.38064e-15
    cc = 2.99792458e10
    c2 = 1.4387769
    na = 6.02214129e23
    tref = 296.0

    kab_co2 = np.zeros((n_wave, n_columns), dtype=np.float64)
    kab_h2o = np.zeros((n_wave, n_columns), dtype=np.float64)

    qref_co2 = 282029.9106
    qij_co2 = qe_co2(temp_row, opt["optical_dir"])

    for i in range(n_wave):
        wavenum = 10000.0 / wavelen[i]
        for j in range(n_columns):
            waveij = opt["wavnum_co2"][i] + opt["delta_co2"][i] * press[j] / 101325.0
            ex_e1 = np.exp(-c2 * opt["elower_co2"][i] / temp_row[j])
            ex_e2 = np.exp(-c2 * opt["elower_co2"][i] / tref)
            ex_v1 = 1.0 - np.exp(-c2 * waveij / temp_row[j])
            ex_v2 = 1.0 - np.exp(-c2 * waveij / tref)
            sij = (
                opt["sw_co2"][i]
                * qref_co2
                * ex_e1
                * ex_v1
                / (qij_co2[j] * ex_e2 * ex_v2 + 1e-30)
            )

            afa_d = waveij / cc * np.sqrt(
                2.0 * na * kk * temp_row[j] * np.log(2.0) / 44.01
            )
            gamma = (tref / temp_row[j]) ** opt["nn_co2"][i] * (
                opt["gammaa_co2"][i] * (1.0 - co2_mixradio[0, j]) * press[j] / 101325.0
                + opt["gammas_co2"][i] * co2_mixradio[0, j] * press[j] / 101325.0
            )
            fij_l = 0.318 * gamma / (
                gamma**2
                + (
                    wavenum
                    - (waveij + opt["delta_co2"][i] * press[j] / 101325.0)
                )
                ** 2
                + 1e-30
            )
            fij_d = np.sqrt(np.log(2.0 / (np.pi * afa_d**2 + 1e-30))) * np.exp(
                -((wavenum - waveij) ** 2) * np.log(2.0) / (afa_d**2 + 1e-30)
            )
            fij = fij_l if press[j] > 13.33 else fij_d
            kab_co2[i, j] = sij * fij

    qref_h2o = 206297.69
    qij_h2o = qe_h2o(temp_row, opt["optical_dir"])

    for i in range(n_wave):
        wavenum = 10000.0 / wavelen[i]
        for j in range(n_columns):
            # NOTE: original Fortran uses CO2 line parameters here for H2O block
            waveij = opt["wavnum_co2"][i] + opt["delta_co2"][i] * press[j] / 101325.0
            ex_e1 = np.exp(-c2 * opt["elower_co2"][i] / temp_row[j])
            ex_e2 = np.exp(-c2 * opt["elower_co2"][i] / tref)
            ex_v1 = 1.0 - np.exp(-c2 * waveij / temp_row[j])
            ex_v2 = 1.0 - np.exp(-c2 * waveij / tref)
            sij = (
                opt["sw_h2o"][i]
                * qref_h2o
                * ex_e1
                * ex_v1
                / (qij_h2o[j] * ex_e2 * ex_v2 + 1e-30)
            )

            afa_d = waveij / cc * np.sqrt(
                2.0 * na * kk * temp_row[j] * np.log(2.0) / 18.0
            )
            gamma = (tref / temp_row[j]) ** opt["nn_h2o"][i] * (
                opt["gammaa_h2o"][i] * (1.0 - wv_mixradio[0, j]) * press[j] / 101325.0
                + opt["gammas_h2o"][i] * wv_mixradio[0, j] * press[j] / 101325.0
            )
            fij_l = 0.318 * gamma / (
                gamma**2
                + (
                    wavenum
                    - (waveij + opt["delta_h2o"][i] * press[j] / 101325.0)
                )
                ** 2
                + 1e-30
            )
            fij_d = np.sqrt(np.log(2.0 / (np.pi * afa_d**2 + 1e-30))) * np.exp(
                -((wavenum - waveij) ** 2) * np.log(2.0) / (afa_d**2 + 1e-30)
            )
            fij = fij_l if press[j] > 13.33 else fij_d
            kab_h2o[i, j] = sij * fij

    return kab_co2, kab_h2o
