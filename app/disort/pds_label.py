"""
PDS3 detached-label (.lbl) + image (.img) loader for CRISM DDR products.

ODE / PDS Geosciences deliver DDR as:
  FRT########_##_DEnnnL_DDR1.IMG
  FRT########_##_DEnnnL_DDR1.LBL

Typical IMAGE object (14 bands, BSQ, PC_REAL):
  1 INA, 2 EMA, 3 phase, 4 lat, 5 lon, …, 13 local solar time (hours), 14 spare
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_KEY_RE = re.compile(
    r"^(?P<key>\^?[A-Za-z0-9_:/]+)\s*=\s*(?P<val>.*)$"
)


def _strip_units(val: str) -> str:
    return re.sub(r"\s*<[^>]+>\s*$", "", val).strip()


def _unquote(val: str) -> str:
    v = val.strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1]
    return v


def parse_pds3_label(text: str) -> Dict[str, Any]:
    """
    Lightweight PDS3 keyword parser (flat + nested OBJECT values flattened).

    Values that span braces {…} or parentheses (…) are kept as one string.
    """
    # Remove /* … */ comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = text.splitlines()
    meta: Dict[str, Any] = {}
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i].strip()
        i += 1
        if not raw or raw in ("END",):
            continue
        if raw.startswith("END_OBJECT"):
            continue
        m = _KEY_RE.match(raw)
        if not m:
            continue
        key = m.group("key").strip()
        val = m.group("val").strip()

        # Continuation / multi-line quoted or braced values
        if val.startswith("{") and "}" not in val:
            parts = [val]
            while i < n and "}" not in lines[i]:
                parts.append(lines[i].strip())
                i += 1
            if i < n:
                parts.append(lines[i].strip())
                i += 1
            val = " ".join(parts)
        elif val.startswith("(") and ")" not in val:
            parts = [val]
            while i < n and ")" not in lines[i]:
                parts.append(lines[i].strip())
                i += 1
            if i < n:
                parts.append(lines[i].strip())
                i += 1
            val = " ".join(parts)
        elif val.startswith('"') and not val.endswith('"'):
            parts = [val]
            while i < n:
                parts.append(lines[i].rstrip("\n"))
                if '"' in lines[i] and lines[i].rstrip().endswith('"'):
                    i += 1
                    break
                i += 1
            val = " ".join(p.strip() for p in parts)

        val = _strip_units(val)
        # Skip OBJECT = FILE / IMAGE structural markers as data
        if key.upper() == "OBJECT":
            continue
        meta[key] = _unquote(val)
        # Also store without pointer caret and without namespace prefix alias
        if key.startswith("^"):
            meta[key[1:]] = meta[key]
        if ":" in key:
            meta[key.split(":", 1)[-1]] = meta[key]
    return meta


def parse_pds3_label_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_pds3_label(f.read())


def _dtype_from_sample_type(sample_type: str, sample_bits: int) -> np.dtype:
    st = (sample_type or "").upper()
    bits = int(sample_bits or 32)
    if st in ("PC_REAL", "IEEE_REAL", "PC_IEEE_REAL") and bits == 32:
        return np.dtype("<f4")
    if st in ("MSB_REAL", "IEEE_REAL_MSB", "SUN_REAL", "MAC_REAL") and bits == 32:
        return np.dtype(">f4")
    if st in ("PC_REAL",) and bits == 64:
        return np.dtype("<f8")
    if st in ("MSB_INTEGER", "MSB_UNSIGNED_INTEGER") and bits == 16:
        return np.dtype(">i2") if "UNSIGNED" not in st else np.dtype(">u2")
    if st in ("LSB_INTEGER", "PC_INTEGER", "LSB_UNSIGNED_INTEGER", "PC_UNSIGNED_INTEGER") and bits == 16:
        return np.dtype("<i2") if "UNSIGNED" not in st else np.dtype("<u2")
    # Default: little-endian float32 (CRISM DDR)
    return np.dtype("<f4")


def _parse_band_names(val: Optional[str]) -> List[str]:
    if not val:
        return []
    s = val.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    names = []
    for part in re.findall(r'"([^"]*)"', s):
        names.append(part.strip())
    if names:
        return names
    return [p.strip().strip('"') for p in s.split(",") if p.strip()]


def resolve_lbl_img_paths(path: str) -> Tuple[str, str]:
    """
    From a user-selected .lbl or .img path, return (lbl_path, img_path).
    """
    path = os.path.abspath(path)
    root, ext = os.path.splitext(path)
    ext_l = ext.lower()
    directory = os.path.dirname(path)

    def _existing_pair(base: str) -> Optional[Tuple[str, str]]:
        for le in (".lbl", ".LBL", ".Lbl"):
            for ie in (".img", ".IMG", ".Img"):
                lp, ip = base + le, base + ie
                if os.path.isfile(lp) and os.path.isfile(ip):
                    return lp, ip
        return None

    if ext_l == ".lbl":
        meta = parse_pds3_label_file(path)
        img_name = meta.get("^IMAGE") or meta.get("IMAGE")
        if img_name:
            img_name = _unquote(str(img_name).strip())
            # Pointer may be "file.img" or ("file.img", offset)
            m = re.search(r'"([^"]+\.img)"', img_name, flags=re.I)
            if m:
                img_name = m.group(1)
            cand = os.path.join(directory, img_name)
            if os.path.isfile(cand):
                return path, cand
        pair = _existing_pair(root)
        if pair:
            return pair
        # Case-insensitive sibling search
        stem = os.path.basename(root).lower()
        for fn in os.listdir(directory):
            if fn.lower() == stem + ".img":
                return path, os.path.join(directory, fn)
        raise FileNotFoundError(f"找不到与标签对应的 .img：{path}")

    if ext_l == ".img":
        pair = _existing_pair(root)
        if pair:
            return pair
        stem = os.path.basename(root).lower()
        for fn in os.listdir(directory):
            if fn.lower() == stem + ".lbl":
                return os.path.join(directory, fn), path
        raise FileNotFoundError(f"找不到与图像对应的 .lbl：{path}")

    raise ValueError(f"需要 .lbl 或 .img 文件，收到：{path}")


_ENVI_HEADER_EXTS = (".hdr", ".HDR")
_ENVI_RASTER_EXTS = (
    ".img", ".IMG", ".dat", ".DAT",
    ".bsq", ".BSQ", ".bil", ".BIL", ".bip", ".BIP",
)
_ENVI_DTYPE = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def _sibling_with_exts(path: str, extensions: Tuple[str, ...]) -> Optional[str]:
    path = os.path.abspath(path)
    root, _ = os.path.splitext(path)
    directory = os.path.dirname(path)
    stem = os.path.basename(root)
    for ext in extensions:
        candidate = root + ext
        if os.path.isfile(candidate):
            return candidate
    wanted = {stem.lower() + ext.lower() for ext in extensions}
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for name in names:
        if name.lower() in wanted:
            return os.path.join(directory, name)
    return None


def _parse_envi_header(header_path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with open(header_path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("envi") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip().lower()] = value.strip().strip("{}").strip()
    return meta


def _load_envi_cube(
    header_path: str, raster_path: str
) -> Tuple[np.ndarray, Dict[str, Any], str, str]:
    header_path = os.path.abspath(header_path)
    raster_path = os.path.abspath(raster_path)
    meta = _parse_envi_header(header_path)
    try:
        samples = int(float(meta["samples"]))
        lines = int(float(meta["lines"]))
    except KeyError as exc:
        raise ValueError(f"ENVI 头文件缺少 samples/lines：{header_path}") from exc
    bands = int(float(meta.get("bands", "1")))
    dtype = _ENVI_DTYPE.get(int(float(meta.get("data type", "4"))))
    if dtype is None:
        raise ValueError(f"不支持的 ENVI data type：{header_path}")
    endian = ">" if int(float(meta.get("byte order", "0"))) == 1 else "<"
    dtype = np.dtype(dtype).newbyteorder(endian)
    offset = int(float(meta.get("header offset", "0")))
    interleave = str(meta.get("interleave", "bsq")).lower()
    count = lines * samples * bands
    with open(raster_path, "rb") as handle:
        handle.seek(offset)
        raw = np.fromfile(handle, dtype=dtype, count=count)
    if raw.size < count:
        raise ValueError(
            f"ENVI 图像过短：期望 {count} 个样点，实际 {raw.size}（{raster_path}）"
        )
    if interleave == "bip":
        cube = raw.reshape((lines, samples, bands))
    elif interleave == "bil":
        cube = raw.reshape((lines, bands, samples)).transpose(0, 2, 1)
    else:
        cube = raw.reshape((bands, lines, samples)).transpose(1, 2, 0)
    meta["_hdr_path"] = header_path
    meta["_img_path"] = raster_path
    return np.ascontiguousarray(cube, dtype=np.float32), meta, header_path, raster_path


def load_aux_cube(path: str) -> Tuple[np.ndarray, Dict[str, Any], str, str]:
    """Load a geometry/aux cube from PDS .lbl or ordinary ENVI (.hdr/.img/.dat)."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到辅助立方体：{path}")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".lbl":
        return load_pds_cube(path)

    header = path if ext == ".hdr" else _sibling_with_exts(path, _ENVI_HEADER_EXTS)
    if header:
        raster = (
            path
            if ext != ".hdr"
            else _sibling_with_exts(path, _ENVI_RASTER_EXTS)
        )
        if raster is None:
            raise FileNotFoundError(
                f"找到 ENVI 头文件但没有同名图像（.img/.dat/.bsq/.bil/.bip）：{header}"
            )
        return _load_envi_cube(header, raster)

    if ext in {".img", ".dat"}:
        label = _sibling_with_exts(path, (".lbl", ".LBL", ".Lbl"))
        if label:
            return load_pds_cube(path)

    raise ValueError(
        "无法识别辅助立方体格式。请选择 PDS 标签（.lbl），"
        "或普通 ENVI（.hdr，以及同名 .img / .dat / .bsq / .bil / .bip）。"
    )


def load_pds_cube(path: str) -> Tuple[np.ndarray, Dict[str, Any], str, str]:
    """
    Load a PDS3 cube selected by .lbl or .img path.

    Returns
    -------
    cube : (lines, samples, bands) float32
    metadata : parsed label keywords
    lbl_path, img_path
    """
    lbl_path, img_path = resolve_lbl_img_paths(path)
    meta = parse_pds3_label_file(lbl_path)

    try:
        lines = int(float(meta["LINES"]))
        samples = int(float(meta["LINE_SAMPLES"]))
        bands = int(float(meta.get("BANDS", 1)))
    except KeyError as exc:
        raise ValueError(f"PDS 标签缺少图像尺寸字段：{exc}") from exc

    dtype = _dtype_from_sample_type(
        str(meta.get("SAMPLE_TYPE", "PC_REAL")),
        int(float(meta.get("SAMPLE_BITS", 32))),
    )
    storage = str(meta.get("BAND_STORAGE_TYPE", "BAND_SEQUENTIAL")).upper()

    raw = np.fromfile(img_path, dtype=dtype)
    expected = lines * samples * bands
    if raw.size < expected:
        raise ValueError(
            f"图像文件过短：期望 {expected} 个样点，实际 {raw.size}（{img_path}）"
        )
    raw = raw[:expected]

    if storage in ("BAND_SEQUENTIAL", "BSQ"):
        cube = raw.reshape((bands, lines, samples)).transpose(1, 2, 0)
    elif storage in ("LINE_INTERLEAVED", "BIL"):
        cube = raw.reshape((lines, bands, samples)).transpose(0, 2, 1)
    elif storage in ("SAMPLE_INTERLEAVED", "BIP"):
        cube = raw.reshape((lines, samples, bands))
    else:
        cube = raw.reshape((bands, lines, samples)).transpose(1, 2, 0)

    meta["_band_names"] = _parse_band_names(meta.get("BAND_NAME"))
    meta["_lbl_path"] = lbl_path
    meta["_img_path"] = img_path
    return np.asarray(cube, dtype=np.float32), meta, lbl_path, img_path


def solar_longitude_from_label(meta: Dict[str, Any]) -> Optional[float]:
    """Return SOLAR_LONGITUDE (deg) from PDS metadata if present."""
    for key in ("SOLAR_LONGITUDE", "solar_longitude", "SOLAR_LONGITUDE_ANGLE"):
        if key in meta and meta[key] not in (None, "", "NULL", "N/A", '"N/A"'):
            try:
                return float(_strip_units(str(meta[key])))
            except ValueError:
                continue
    return None
