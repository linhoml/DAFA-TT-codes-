"""Endmember spectral library loading and resampling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _as_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", "ignore")
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return _as_str(x.item() if x.size == 1 else x.ravel()[0])
        return str(x.tolist() if x.size > 1 else x.item())
    return str(x).strip()


@dataclass
class SpectralLibrary:
    wavelengths: np.ndarray  # (n_bands,) μm
    spectra: np.ndarray  # (n_bands, n_endmembers)
    names: List[str] = field(default_factory=list)
    source: str = ""

    @property
    def n_endmembers(self) -> int:
        return int(self.spectra.shape[1])

    def resample(self, target_wavelengths: np.ndarray) -> "SpectralLibrary":
        """Linearly interpolate each endmember onto target wavelengths (μm)."""
        tw = np.asarray(target_wavelengths, dtype=float).ravel()
        src_w = np.asarray(self.wavelengths, dtype=float).ravel()
        src = np.asarray(self.spectra, dtype=float)
        out = np.empty((tw.size, src.shape[1]), dtype=float)
        for j in range(src.shape[1]):
            y = src[:, j]
            good = np.isfinite(src_w) & np.isfinite(y)
            if good.sum() < 2:
                out[:, j] = np.nan
                continue
            out[:, j] = np.interp(tw, src_w[good], y[good], left=np.nan, right=np.nan)
        return SpectralLibrary(
            wavelengths=tw.copy(),
            spectra=out,
            names=list(self.names),
            source=self.source + "→resampled",
        )

    def subset(self, indices: Sequence[int]) -> "SpectralLibrary":
        idx = [int(i) for i in indices]
        return SpectralLibrary(
            wavelengths=self.wavelengths.copy(),
            spectra=self.spectra[:, idx].copy(),
            names=[self.names[i] if i < len(self.names) else f"EM{i+1}" for i in idx],
            source=self.source + f"→subset{len(idx)}",
        )

    def wavelength_mask(
        self, wmin: Optional[float] = None, wmax: Optional[float] = None
    ) -> np.ndarray:
        w = self.wavelengths
        m = np.isfinite(w)
        if wmin is not None:
            m &= w >= float(wmin)
        if wmax is not None:
            m &= w <= float(wmax)
        return m


def load_relab_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Two-column wavelength / reflectance text (nm or μm)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] in "#;%":
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            if len(parts) < 2:
                continue
            try:
                w, r = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if np.isfinite(w) and np.isfinite(r):
                rows.append((w, r))
    if not rows:
        raise ValueError(f"无有效光谱数据：{path}")
    arr = np.asarray(rows, dtype=float)
    wave, refl = arr[:, 0], arr[:, 1]
    if np.nanmax(wave) > 100.0 or np.nanmedian(wave) > 50.0:
        wave = wave / 1000.0
    order = np.argsort(wave)
    return wave[order], refl[order]


def load_library_from_txt_files(paths: Sequence[str]) -> SpectralLibrary:
    waves = []
    specs = []
    names = []
    for p in paths:
        w, r = load_relab_txt(p)
        waves.append(w)
        specs.append(r)
        names.append(os.path.splitext(os.path.basename(p))[0])
    # Use union grid of first file as reference; resample others
    ref_w = waves[0]
    mat = np.empty((ref_w.size, len(specs)), dtype=float)
    mat[:, 0] = specs[0]
    for j in range(1, len(specs)):
        mat[:, j] = np.interp(ref_w, waves[j], specs[j], left=np.nan, right=np.nan)
    return SpectralLibrary(
        wavelengths=ref_w,
        spectra=mat,
        names=names,
        source="txt:" + ";".join(os.path.basename(p) for p in paths),
    )


def load_library_from_mat(path: str) -> SpectralLibrary:
    """
    Load DAFA/TT-style MATLAB library:
      TargetLibrary[:,0] = wavelength (μm)
      TargetLibrary[:,1:] = endmember spectra
      TargetLibraryName / TargetLibraryFileName optional
    """
    from scipy.io import loadmat

    d = loadmat(path)
    if "TargetLibrary" not in d:
        # generic: first column wavelength
        key = next((k for k in d if not k.startswith("__") and isinstance(d[k], np.ndarray) and d[k].ndim == 2), None)
        if key is None:
            raise ValueError(f"MAT 文件中未找到 TargetLibrary：{path}")
        arr = np.asarray(d[key], dtype=float)
    else:
        arr = np.asarray(d["TargetLibrary"], dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("光谱库矩阵至少需要 2 列（波长 + 端元）")

    wave = arr[:, 0]
    specs = arr[:, 1:]
    names: List[str] = []
    if "TargetLibraryName" in d:
        raw = d["TargetLibraryName"]
        # shape (1, n) cells; index 0 is often 'wavelength'
        n = specs.shape[1]
        for i in range(1, n + 1):
            if raw.shape[1] > i:
                names.append(_as_str(raw[0, i]) or f"EM{i}")
            else:
                names.append(f"EM{i}")
    else:
        names = [f"EM{i+1}" for i in range(specs.shape[1])]

    order = np.argsort(wave)
    return SpectralLibrary(
        wavelengths=wave[order],
        spectra=specs[order, :],
        names=names,
        source=f"mat:{os.path.basename(path)}",
    )


def load_library(path: str) -> SpectralLibrary:
    """Load from .mat, single .txt, or a directory of .txt files."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith((".txt", ".asc", ".csv", ".dat"))
        )
        if not files:
            raise FileNotFoundError(f"目录中没有光谱 txt：{path}")
        return load_library_from_txt_files(files)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        return load_library_from_mat(path)
    if ext in (".txt", ".asc", ".csv", ".dat"):
        return load_library_from_txt_files([path])
    raise ValueError(f"不支持的光谱库格式：{path}")
