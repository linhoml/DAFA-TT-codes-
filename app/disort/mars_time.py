"""
Mars solar longitude (Ls) from Earth UTC / CRISM PDS labels.

Uses the NASA GISS Mars24 / Allison & McEwen (2000) algorithm
(https://www.giss.nasa.gov/tools/mars24/help/algorithm.html).

Also helpers to read observation time from:
  - ENVI .hdr metadata
  - CRISM DDR PDS3 detached labels (.lbl) — preferred source is the
    label keyword SOLAR_LONGITUDE; START_TIME is used as fallback.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Leap-second table: (UTC datetime inclusive start, TAI-UTC seconds)
# TT-UTC = TAI-UTC + 32.184
_LEAP_TAI_UTC = [
    (datetime(1972, 1, 1, tzinfo=timezone.utc), 10),
    (datetime(1972, 7, 1, tzinfo=timezone.utc), 11),
    (datetime(1973, 1, 1, tzinfo=timezone.utc), 12),
    (datetime(1974, 1, 1, tzinfo=timezone.utc), 13),
    (datetime(1975, 1, 1, tzinfo=timezone.utc), 14),
    (datetime(1976, 1, 1, tzinfo=timezone.utc), 15),
    (datetime(1977, 1, 1, tzinfo=timezone.utc), 16),
    (datetime(1978, 1, 1, tzinfo=timezone.utc), 17),
    (datetime(1979, 1, 1, tzinfo=timezone.utc), 18),
    (datetime(1980, 1, 1, tzinfo=timezone.utc), 19),
    (datetime(1981, 7, 1, tzinfo=timezone.utc), 20),
    (datetime(1982, 7, 1, tzinfo=timezone.utc), 21),
    (datetime(1983, 7, 1, tzinfo=timezone.utc), 22),
    (datetime(1985, 7, 1, tzinfo=timezone.utc), 23),
    (datetime(1988, 1, 1, tzinfo=timezone.utc), 24),
    (datetime(1990, 1, 1, tzinfo=timezone.utc), 25),
    (datetime(1991, 1, 1, tzinfo=timezone.utc), 26),
    (datetime(1992, 7, 1, tzinfo=timezone.utc), 27),
    (datetime(1993, 7, 1, tzinfo=timezone.utc), 28),
    (datetime(1994, 7, 1, tzinfo=timezone.utc), 29),
    (datetime(1996, 1, 1, tzinfo=timezone.utc), 30),
    (datetime(1997, 7, 1, tzinfo=timezone.utc), 31),
    (datetime(1999, 1, 1, tzinfo=timezone.utc), 32),
    (datetime(2006, 1, 1, tzinfo=timezone.utc), 33),
    (datetime(2009, 1, 1, tzinfo=timezone.utc), 34),
    (datetime(2012, 7, 1, tzinfo=timezone.utc), 35),
    (datetime(2015, 7, 1, tzinfo=timezone.utc), 36),
    (datetime(2017, 1, 1, tzinfo=timezone.utc), 37),
]


def _tt_minus_utc_seconds(dt_utc: datetime) -> float:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    if dt_utc < datetime(1972, 1, 1, tzinfo=timezone.utc):
        jd_ut = datetime_to_julian(dt_utc)
        t = (jd_ut - 2451545.0) / 36525.0
        return 64.184 + 59.0 * t - 51.2 * t**2 - 67.1 * t**3 - 16.4 * t**4
    tai_utc = 10.0
    for start, val in _LEAP_TAI_UTC:
        if dt_utc >= start:
            tai_utc = float(val)
        else:
            break
    return tai_utc + 32.184


def datetime_to_julian(dt: datetime) -> float:
    """Gregorian datetime → Julian Date (same timezone as dt; treat as UT/UTC)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    y, m = dt.year, dt.month
    d = (
        dt.day
        + (dt.hour + (dt.minute + (dt.second + dt.microsecond * 1e-6) / 60.0) / 60.0)
        / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    return float(jd)


def mars_ls_from_julian_ut(jd_ut: float, tt_minus_utc_s: Optional[float] = None) -> float:
    """
    Areocentric solar longitude Ls (degrees, 0–360) from Julian Date UT/UTC.

    Mars24 / Allison & McEwen (2000) recipe.
    """
    if tt_minus_utc_s is None:
        # Approximate TT-UTC for modern missions (~2000+)
        tt_minus_utc_s = 69.184  # post-2017 leap second state
        # Refine if we can invert JD to datetime roughly
        try:
            # Unix millis from JD
            millis = (jd_ut - 2440587.5) * 86400000.0
            dt = datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
            tt_minus_utc_s = _tt_minus_utc_seconds(dt)
        except Exception:
            pass

    jd_tt = jd_ut + tt_minus_utc_s / 86400.0
    dt_j2000 = jd_tt - 2451545.0

    m = math.radians((19.3871 + 0.52402073 * dt_j2000) % 360.0)
    alpha_fms = (270.3871 + 0.524038496 * dt_j2000) % 360.0

    pbs = 0.0
    table = [
        (0.0071, 2.2353, 49.409),
        (0.0057, 2.7543, 168.173),
        (0.0039, 1.1177, 191.837),
        (0.0037, 15.7866, 21.736),
        (0.0021, 2.1354, 15.704),
        (0.0020, 2.4694, 95.528),
        (0.0018, 32.8493, 49.095),
    ]
    for ai, taui, phii in table:
        pbs += ai * math.cos(math.radians(0.985626 * dt_j2000 / taui + phii))

    nu_m = (
        (10.691 + 3.0e-7 * dt_j2000) * math.sin(m)
        + 0.623 * math.sin(2 * m)
        + 0.050 * math.sin(3 * m)
        + 0.005 * math.sin(4 * m)
        + 0.0005 * math.sin(5 * m)
        + pbs
    )
    ls = (alpha_fms + nu_m) % 360.0
    return float(ls)


def mars_ls_from_utc(dt_utc: datetime) -> float:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    jd = datetime_to_julian(dt_utc)
    return mars_ls_from_julian_ut(jd, _tt_minus_utc_seconds(dt_utc))


_UTC_KEY_CANDIDATES = [
    "start_time",          # CRISM / PDS3
    "start time",
    "stop_time",
    "stop time",
    "acquisition time",
    "acquisition_time",
    "observation time",
    "observation_time",
    "observation start time",
    "observation_start_time",
    "utc",
    "utc time",
    "utc_time",
    "time",
    "datetime",
    "date time",
    "date_time",
    "product_creation_time",
    "start_time_utc",
    "image_time",
    "closest_approach_time",
]


_DATE_PATTERNS = [
    # 2008-06-15T12:34:56.789Z / 2008-06-15 12:34:56
    re.compile(
        r"(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})[ T_](?P<H>\d{1,2}):(?P<M>\d{2})"
        r"(?::(?P<S>\d{2}(?:\.\d+)?))?"
    ),
    # 15-Jun-2008 12:34:56
    re.compile(
        r"(?P<d>\d{1,2})[- ](?P<mon>[A-Za-z]{3})[- ](?P<y>\d{4})[ T](?P<H>\d{1,2}):(?P<M>\d{2})"
        r"(?::(?P<S>\d{2}(?:\.\d+)?))?"
    ),
    # 20080615T123456
    re.compile(
        r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[T_ ]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"
    ),
]

_MON = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_utc_string(text: str) -> Optional[datetime]:
    if text is None:
        return None
    s = str(text).strip().strip('"').strip("'")
    if not s:
        return None
    # ISO first
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    for pat in _DATE_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        gd = m.groupdict()
        try:
            y = int(gd["y"])
            if "mon" in gd and gd["mon"]:
                month = _MON[gd["mon"][:3].lower()]
            else:
                month = int(gd["m"])
            d = int(gd["d"])
            H = int(gd["H"])
            M = int(gd["M"])
            S = float(gd["S"]) if gd.get("S") else 0.0
            sec = int(S)
            micro = int(round((S - sec) * 1e6))
            return datetime(y, month, d, H, M, sec, micro, tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _normalize_key(k: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(k).strip().lower())


def extract_utc_from_metadata(metadata: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[str]]:
    """
    Search ENVI/spectral metadata dict for an observation UTC.

    Returns (datetime_utc, source_key_or_None).
    """
    if not metadata:
        return None, None

    # Direct preferred keys
    norm_map = {_normalize_key(k): k for k in metadata.keys()}
    for cand in _UTC_KEY_CANDIDATES:
        nk = _normalize_key(cand)
        if nk in norm_map:
            raw = metadata[norm_map[nk]]
            if isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]
            dt = parse_utc_string(str(raw))
            if dt is not None:
                return dt, norm_map[nk]

    # Scan all string-like values
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            texts = [str(x) for x in v]
        else:
            texts = [str(v)]
        for t in texts:
            if not re.search(r"\d{4}", t):
                continue
            # Prefer values that look like timestamps
            if not re.search(r"\d{1,2}:\d{2}|\d{8}T\d{6}|T\d{2}", t, re.I):
                # still try if key hints time/date
                if not re.search(r"time|date|utc|acquisition", str(k), re.I):
                    continue
            dt = parse_utc_string(t)
            if dt is not None:
                return dt, str(k)
    return None, None


def extract_utc_from_hdr_file(hdr_path: str) -> Tuple[Optional[datetime], Optional[str]]:
    """Parse a raw ENVI .hdr or PDS3 .lbl text file for UTC-like fields."""
    if not hdr_path or not os.path.isfile(hdr_path):
        return None, None
    try:
        with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return None, None

    # Prefer explicit START_TIME (CRISM DDR PDS label)
    for m in re.finditer(
        r"^(START_TIME|STOP_TIME)\s*=\s*(.+)$", text, flags=re.M | re.I
    ):
        key = m.group(1).strip()
        val = m.group(2).strip().strip('"')
        # strip PDS units if any
        val = re.sub(r"\s*<[^>]+>\s*$", "", val)
        dt = parse_utc_string(val)
        if dt is not None:
            return dt, key

    # key = value lines
    for m in re.finditer(
        r"^([A-Za-z0-9 _\-/:]+)\s*=\s*(.+)$", text, flags=re.M
    ):
        key = m.group(1).strip()
        val = m.group(2).strip().strip("{}").strip()
        nk = _normalize_key(key)
        if any(c in nk for c in ("utc", "time", "date", "acquisition")):
            dt = parse_utc_string(val)
            if dt is not None:
                return dt, key

    # Fallback: any ISO-like timestamp in file
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            dt = parse_utc_string(m.group(0))
            if dt is not None:
                return dt, "hdr_text"
    return None, None


def ls_from_label_source(
    metadata: Optional[Dict[str, Any]] = None,
    label_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get Mars Ls from CRISM DDR / ENVI label sources.

    Preference order:
      1. SOLAR_LONGITUDE in metadata / .lbl  (official SPICE value in DDR)
      2. START_TIME / other UTC → Mars24 Ls

    Returns dict: ok, ls_deg, utc, utc_iso, source_key, message, ls_source
    """
    from .pds_label import parse_pds3_label_file, solar_longitude_from_label

    meta = dict(metadata or {})
    if label_path and os.path.isfile(label_path) and label_path.lower().endswith(".lbl"):
        try:
            meta.update(parse_pds3_label_file(label_path))
        except Exception:
            pass

    # 1) Direct SOLAR_LONGITUDE from PDS DDR label
    ls_direct = solar_longitude_from_label(meta)
    dt, src = (None, None)
    if meta:
        dt, src = extract_utc_from_metadata(meta)
    if dt is None and label_path:
        dt2, src2 = extract_utc_from_hdr_file(label_path)
        if dt2 is not None:
            dt, src = dt2, src2

    if ls_direct is not None:
        utc_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None
        msg = f"由 PDS 标签 SOLAR_LONGITUDE 读取 Ls = {ls_direct:.3f}°"
        if utc_iso:
            msg += f"（START_TIME={utc_iso}）"
        return {
            "ok": True,
            "ls_deg": float(ls_direct) % 360.0,
            "utc": dt,
            "utc_iso": utc_iso,
            "source_key": "SOLAR_LONGITUDE",
            "ls_source": "SOLAR_LONGITUDE",
            "message": msg,
        }

    if dt is None:
        return {
            "ok": False,
            "ls_deg": None,
            "utc": None,
            "utc_iso": None,
            "source_key": None,
            "ls_source": None,
            "message": "头文件中未找到 SOLAR_LONGITUDE 或可用的 UTC（START_TIME）。",
        }

    ls = mars_ls_from_utc(dt)
    return {
        "ok": True,
        "ls_deg": float(ls),
        "utc": dt,
        "utc_iso": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_key": src,
        "ls_source": "UTC_COMPUTED",
        "message": (
            f"由 UTC {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC 计算得 Ls = {ls:.3f}°"
        ),
    }


# Backwards-compatible alias
def ls_from_envi_source(
    metadata: Optional[Dict[str, Any]] = None,
    hdr_path: Optional[str] = None,
) -> Dict[str, Any]:
    return ls_from_label_source(metadata=metadata, label_path=hdr_path)
