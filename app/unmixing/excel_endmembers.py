"""Load Hapke endmember reflectance tables from Excel."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .hapke_rt import HapkeEndmember


def _require_excel_deps():
    """Import pandas/openpyxl with a clear install hint."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "缺少 pandas，无法读取 Excel。\n"
            "请在当前 Python 环境执行：\n"
            "  pip install pandas openpyxl\n"
            "或：\n"
            "  pip install -r requirements.txt"
        ) from exc
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "缺少 openpyxl，无法读取 .xlsx Excel。\n"
            "请在当前 Python 环境执行：\n"
            "  pip install openpyxl\n"
            "或：\n"
            "  pip install -r requirements.txt"
        ) from exc
    return pd


def _normalize_wave(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    if np.nanmax(wave) > 100.0 or np.nanmedian(wave) > 50.0:
        wave = wave / 1000.0  # nm → μm
    return wave


def load_endmembers_excel(path: str) -> List[HapkeEndmember]:
    """
    Read mineral endmember **reflectance factor (REFF)** from an Excel file.

    Accepted layouts
    ----------------
    A) Single sheet, wide table:
         col0 = wavelength (μm or nm)
         col1..N = REFF, header = mineral name
    B) Multiple sheets: each sheet is one mineral with two columns
         wavelength, REFF (header optional)

    Note: image pixels are I/F; conversion I/F = REFF × cos(i) is applied
    during Hapke fitting using aux-cube incidence.
    """
    pd = _require_excel_deps()
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "读取 Excel 失败：未安装 openpyxl。\n"
            "请执行：pip install openpyxl\n"
            "或：pip install -r requirements.txt"
        ) from exc
    endmembers: List[HapkeEndmember] = []

    if len(xl.sheet_names) == 1:
        df = xl.parse(xl.sheet_names[0])
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if df.shape[1] < 2:
            raise ValueError("Excel 至少需要两列：波长 + 至少一个端元反射率")
        wave = _normalize_wave(df.iloc[:, 0].to_numpy(dtype=float))
        for j in range(1, df.shape[1]):
            name = str(df.columns[j]).strip() or f"EM{j}"
            # skip unnamed / wavelength-like headers
            if name.lower().startswith("unnamed"):
                name = f"EM{j}"
            refl = df.iloc[:, j].to_numpy(dtype=float)
            order = np.argsort(wave)
            endmembers.append(
                HapkeEndmember(
                    name=name,
                    wavelengths=wave[order],
                    reflectance=refl[order],
                    source=f"excel:{os.path.basename(path)}:{name}",
                )
            )
    else:
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
            if df.shape[1] < 2:
                continue
            # Prefer columns named like wavelength / reflectance
            cols = [str(c).strip().lower() for c in df.columns]
            wcol, rcol = 0, 1
            for i, c in enumerate(cols):
                if "wave" in c or "λ" in c or "lambda" in c or c in ("wl", "wvl"):
                    wcol = i
                if "refl" in c or "r" == c or "i/f" in c or "radf" in c:
                    rcol = i
            wave = _normalize_wave(df.iloc[:, wcol].to_numpy(dtype=float))
            refl = df.iloc[:, rcol].to_numpy(dtype=float)
            order = np.argsort(wave)
            endmembers.append(
                HapkeEndmember(
                    name=str(sheet).strip() or f"EM{len(endmembers)+1}",
                    wavelengths=wave[order],
                    reflectance=refl[order],
                    source=f"excel:{os.path.basename(path)}:{sheet}",
                )
            )

    if not endmembers:
        raise ValueError(f"未能从 Excel 解析到端元：{path}")
    return endmembers


def write_endmember_template(path: str, wavelengths_um: Optional[np.ndarray] = None) -> str:
    """Write a starter Excel template with example minerals."""
    pd = _require_excel_deps()
    if wavelengths_um is None:
        wavelengths_um = np.arange(1.0, 2.601, 0.01)
    w = np.asarray(wavelengths_um, dtype=float)
    # Simple synthetic continua for template only
    olivine = 0.35 + 0.05 * np.sin(2 * np.pi * (w - 1.0) / 1.6)
    pyroxene = 0.30 + 0.08 * np.exp(-((w - 1.9) / 0.25) ** 2)
    plagioclase = 0.55 - 0.02 * (w - 1.0)
    df = pd.DataFrame(
        {
            "wavelength_um": w,
            "Olivine": np.clip(olivine, 0.05, 0.95),
            "Pyroxene": np.clip(pyroxene, 0.05, 0.95),
            "Plagioclase": np.clip(plagioclase, 0.05, 0.95),
        }
    )
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_excel(path, index=False, engine="openpyxl")
    return path
