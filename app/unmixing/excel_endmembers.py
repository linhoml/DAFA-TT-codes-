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


def _cell_str(v) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and np.isnan(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() in ("nan", "none", "nat"):
        return ""
    # Excel may store IDs as 1.0
    try:
        f = float(s)
        if abs(f - int(f)) < 1e-9:
            return str(int(f))
    except Exception:
        pass
    return s


def _cell_float(v, default: float) -> float:
    if v is None:
        return float(default)
    try:
        if isinstance(v, float) and np.isnan(v):
            return float(default)
        return float(v)
    except Exception:
        s = str(v).strip().replace(",", "")
        try:
            return float(s)
        except Exception:
            return float(default)


def _looks_like_metadata_layout(df) -> bool:
    """
    New layout (1-based Excel rows):
      row1: mineral names (optional labels in col0)
      row2: spectrum ID
      row3: mean grain size D (μm)
      row4: density ρ
      row5: real index n
      row6+: wavelength | REFF ...
    """
    if df.shape[0] < 6 or df.shape[1] < 2:
        return False
    wave = pd_to_float_array(df.iloc[5:, 0])
    if np.isfinite(wave).sum() < 3:
        return False
    # Wavelengths should be mostly positive and not look like reflectance-only noise
    finite = wave[np.isfinite(wave)]
    if finite.size < 3:
        return False
    if np.nanmin(finite) <= 0:
        return False
    # Prefer increasing wavelengths
    if np.nanmean(np.diff(finite) > 0) < 0.6:
        return False
    return True


def pd_to_float_array(series) -> np.ndarray:
    out = []
    for v in series:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append(np.nan)
            else:
                out.append(float(v))
        except Exception:
            try:
                out.append(float(str(v).strip().replace(",", "")))
            except Exception:
                out.append(np.nan)
    return np.asarray(out, dtype=float)


def _load_metadata_layout(df, path: str) -> List[HapkeEndmember]:
    """
    Parse:
      Excel row1 = mineral / column titles
      row2 = spectrum ID
      row3 = mean grain size D (μm)
      row4 = density ρ (g/cm³)
      row5 = refractive index n
      row6+ = wavelength | REFF
    """
    endmembers: List[HapkeEndmember] = []
    wave = _normalize_wave(pd_to_float_array(df.iloc[5:, 0]))
    order = np.argsort(wave)

    for j in range(1, df.shape[1]):
        mineral_name = _cell_str(df.iloc[0, j])  # Excel 第1行：矿物名称
        spectrum_id = _cell_str(df.iloc[1, j])   # Excel 第2行：光谱ID（元数据）
        grain = _cell_float(df.iloc[2, j], 50.0)
        dens = _cell_float(df.iloc[3, j], 3.0)
        n_val = _cell_float(df.iloc[4, j], 1.7)
        refl = pd_to_float_array(df.iloc[5:, j])

        name = mineral_name or spectrum_id or f"EM{j}"
        if not np.isfinite(refl).any():
            continue
        # Skip columns that are entirely empty / non-numeric in metadata+spectra
        if np.isfinite(refl).sum() < 3:
            continue

        endmembers.append(
            HapkeEndmember(
                name=name,
                wavelengths=wave[order],
                reflectance=refl[order],
                density=float(dens) if dens > 0 else 3.0,
                n=float(n_val) if n_val > 1.0 else 1.7,
                grain_size_um=float(grain) if grain > 0 else 50.0,
                spectrum_id=spectrum_id,
                lab_incidence_deg=30.0,
                lab_emission_deg=0.0,
                lab_phase_deg=30.0,
                source=f"excel:{os.path.basename(path)}:{name}",
            )
        )
    return endmembers


def _load_simple_wide_layout(df, path: str) -> List[HapkeEndmember]:
    """Legacy: header = mineral names, all rows = wavelength + REFF."""
    endmembers: List[HapkeEndmember] = []
    wave = _normalize_wave(df.iloc[:, 0].to_numpy(dtype=float))
    for j in range(1, df.shape[1]):
        name = str(df.columns[j]).strip() or f"EM{j}"
        if name.lower().startswith("unnamed"):
            name = f"EM{j}"
        refl = df.iloc[:, j].to_numpy(dtype=float)
        order = np.argsort(wave)
        endmembers.append(
            HapkeEndmember(
                name=name,
                wavelengths=wave[order],
                reflectance=refl[order],
                spectrum_id=name,
                source=f"excel:{os.path.basename(path)}:{name}",
            )
        )
    return endmembers


def load_endmembers_excel(path: str) -> List[HapkeEndmember]:
    """
    Read mineral endmember **reflectance factor (REFF)** from an Excel file.

    Preferred layout (single sheet)
    -------------------------------
    Row1: (label) | mineral titles ...
    Row2: 光谱ID | ID1 | ID2 | ...
    Row3: 平均粒径 | D1 | D2 | ...   (μm)
    Row4: 密度 | ρ1 | ρ2 | ...
    Row5: 折射率 n | n1 | n2 | ...
    Row6+: wavelength | REFF ...

    Legacy layouts are still accepted as fallback:
    - wide table with header = mineral names, all rows = spectra
    - multi-sheet (one mineral per sheet)

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
        # Read without header so Excel row numbers match the documented layout
        df_raw = xl.parse(xl.sheet_names[0], header=None)
        df_raw = df_raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if df_raw.shape[1] < 2:
            raise ValueError("Excel 至少需要两列：波长 + 至少一个端元")

        if _looks_like_metadata_layout(df_raw):
            endmembers = _load_metadata_layout(df_raw, path)
        else:
            # Fall back to header-based wide table
            df = xl.parse(xl.sheet_names[0])
            df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
            if df.shape[1] < 2:
                raise ValueError("Excel 至少需要两列：波长 + 至少一个端元反射率")
            endmembers = _load_simple_wide_layout(df, path)
    else:
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
            if df.shape[1] < 2:
                continue
            cols = [str(c).strip().lower() for c in df.columns]
            wcol, rcol = 0, 1
            for i, c in enumerate(cols):
                if "wave" in c or "λ" in c or "lambda" in c or c in ("wl", "wvl"):
                    wcol = i
                if "reff" in c or "refl" in c or c == "r" or "i/f" in c or "radf" in c:
                    rcol = i
            wave = _normalize_wave(df.iloc[:, wcol].to_numpy(dtype=float))
            refl = df.iloc[:, rcol].to_numpy(dtype=float)
            order = np.argsort(wave)
            name = str(sheet).strip() or f"EM{len(endmembers)+1}"
            endmembers.append(
                HapkeEndmember(
                    name=name,
                    wavelengths=wave[order],
                    reflectance=refl[order],
                    spectrum_id=name,
                    source=f"excel:{os.path.basename(path)}:{sheet}",
                )
            )

    if not endmembers:
        raise ValueError(f"未能从 Excel 解析到端元：{path}")
    return endmembers


def write_endmember_template(path: str, wavelengths_um: Optional[np.ndarray] = None) -> str:
    """
    Write a starter Excel template in the metadata layout:

      row1 titles, row2 spectrum ID, row3 D, row4 density, row5 n, row6+ spectra.
    """
    pd = _require_excel_deps()
    if wavelengths_um is None:
        wavelengths_um = np.arange(1.0, 2.601, 0.01)
    w = np.asarray(wavelengths_um, dtype=float)
    olivine = np.clip(0.35 + 0.05 * np.sin(2 * np.pi * (w - 1.0) / 1.6), 0.05, 0.95)
    pyroxene = np.clip(0.30 + 0.08 * np.exp(-((w - 1.9) / 0.25) ** 2), 0.05, 0.95)
    plagioclase = np.clip(0.55 - 0.02 * (w - 1.0), 0.05, 0.95)

    # Build without relying on DataFrame header so row order is exact
    rows = [
        ["mineral / wavelength", "Olivine", "Pyroxene", "Plagioclase"],
        ["光谱ID", "OL-01", "PX-01", "PL-01"],
        ["平均粒径_um", 50.0, 40.0, 80.0],
        ["密度_g_cm3", 3.32, 3.50, 2.69],
        ["折射率_n", 1.69, 1.70, 1.56],
    ]
    for i, wi in enumerate(w):
        rows.append([float(wi), float(olivine[i]), float(pyroxene[i]), float(plagioclase[i])])

    df = pd.DataFrame(rows)
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_excel(path, index=False, header=False, engine="openpyxl")
    return path
