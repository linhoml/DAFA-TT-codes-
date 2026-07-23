"""
Mars Climate Database (MCD) profile client.

Backends (tried in order):
1. Local compiled ``fmcd`` / ``mcd`` Python bindings (recommended)
2. MCD web CGI at www-mars.lmd.jussieu.fr (best-effort)
3. Fallback: reuse atmospheric tables under an existing ``input/`` folder

Returned profile is a dict compatible with ``correction.run_disort_correction``
``atm_overrides`` (n_columns vertical levels).
"""

from __future__ import annotations

import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np

MCD_CGI = "https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py"

# MCD web variable codes (see listvar.js)
_WEB_VAR_BATCHES = [
    ("t", "p", "rho", "vmr_co2"),
    ("vmr_h2o", "dust_mmr", "dust_reff", "vmr_h2oice"),
    ("h2oice_reff", "col_h2ovapor", "col_h2oice", "col_co2"),
    ("tsurf", "ps", "solzenang", "t"),
]


def _linspace_heights(n_columns: int, z_top_m: float = 80000.0) -> np.ndarray:
    # Surface → TOA (matches DISORT layer construction using increasing height)
    return np.linspace(0.0, z_top_m, n_columns)


def empty_profile(n_columns: int = 35) -> Dict[str, np.ndarray]:
    height = _linspace_heights(n_columns)
    z = np.zeros(n_columns, dtype=np.float64)
    return {
        "height": height,
        "density": z.copy(),
        "press": z.copy(),
        "temp": np.full(n_columns, 210.0),
        "co2_mixradio": z.copy(),
        "dust_mixradio": z.copy(),
        "dust_re": np.full(n_columns, 1.5e-6),
        "watice_mixradio": z.copy(),
        "watice_re": z.copy(),
        "wv_mixradio": z.copy(),
        "co2_column": np.array([0.0]),
        "wv_column": np.array([0.0]),
        "watice_column": np.array([0.0]),
        "press_surf": np.array([610.0]),
        "temp_surf": np.array([220.0]),
        "soz": np.array([np.nan]),
        "source": "empty",
    }


def profile_from_input_dir(input_root: str, n_columns: int = 35) -> Dict[str, np.ndarray]:
    """Fallback: load atmospheric tables from DISORT input/ folder."""
    from .io_input import load_input_bundle

    atm = load_input_bundle(input_root, n_wave=None, n_columns=n_columns)
    out = {
        "height": atm["height"].copy(),
        "density": atm["density"][0].copy(),
        "press": atm["press"].copy(),
        "temp": atm["temp"][0].copy(),
        "co2_mixradio": atm["co2_mixradio"][0].copy(),
        "dust_mixradio": atm["dust_mixradio"][0].copy(),
        "dust_re": atm["dust_re"][0].copy(),
        "watice_mixradio": atm["watice_mixradio"][0].copy(),
        "watice_re": atm["watice_re"][0].copy(),
        "wv_mixradio": atm["wv_mixradio"][0].copy(),
        "co2_column": np.array([float(atm["co2_column"][0])]),
        "wv_column": np.array([float(atm["wv_column"][0])]),
        "watice_column": np.array([float(atm["watice_column"][0])]),
        "press_surf": np.array([float(atm["press_surf"][0])]),
        "temp_surf": np.array([float(atm["temp_surf"][0])]),
        "soz": np.array([float(atm["soz"][0]) if np.isfinite(atm["soz"][0]) else np.nan]),
        "source": f"input_dir:{atm['input_dir']}",
    }
    return out


def _try_local_fmcd(
    lat: float,
    lon: float,
    loct: float,
    ls: float,
    n_columns: int,
    dust_scenario: int = 1,
) -> Optional[Dict[str, np.ndarray]]:
    """Use local f2py MCD bindings if present."""
    call_mcd = None
    try:
        from fmcd import call_mcd as _cm  # type: ignore

        call_mcd = _cm
    except Exception:
        try:
            from mcd import mcd as McdClass  # type: ignore

            query = McdClass()
            query.lat = float(lat)
            query.lon = float(lon)
            query.loct = float(loct)
            query.xdate = float(ls)
            query.datekey = 1
            query.dust = int(dust_scenario)
            query.hrkey = 0
            query.zkey = 3  # m above surface
            heights = _linspace_heights(n_columns)
            query.profile(nd=n_columns, tabperso=heights)
            # Attribute names vary by mcd-python version; collect defensively
            temp = np.asarray(getattr(query, "temp", getattr(query, "t", None)), dtype=float)
            press = np.asarray(getattr(query, "pres", getattr(query, "p", None)), dtype=float)
            dens = np.asarray(getattr(query, "dens", getattr(query, "rho", None)), dtype=float)
            if temp.size != n_columns:
                return None
            out = empty_profile(n_columns)
            out["height"] = heights
            out["temp"] = temp
            out["press"] = press
            out["density"] = dens
            # Extended variables if available
            meanvar = getattr(query, "meanvar", None)
            extvar = getattr(query, "extvar", None)
            out["source"] = "local_mcd_python"
            out["temp_surf"] = np.array([float(temp[0])])
            out["press_surf"] = np.array([float(press[0])])
            return out
        except Exception:
            return None

    if call_mcd is None:
        return None

    # Direct fmcd.call_mcd loop over altitudes
    heights = _linspace_heights(n_columns)
    out = empty_profile(n_columns)
    out["height"] = heights
    try:
        dset = os.environ.get("MCD_DATA", "")
        if dset and not dset.endswith("/"):
            dset = dset + "/"
        for i, z in enumerate(heights):
            # Signature follows MCD manual; unused extras zeroed
            pres, dens, temp, *_rest = call_mcd(
                3, float(z), float(lon), float(lat), 0,
                1, float(ls), float(loct), dset, int(dust_scenario),
                0, 1, 0.0, np.ones(100, dtype=np.int32),
            )
            out["press"][i] = float(pres)
            out["density"][i] = float(dens)
            out["temp"][i] = float(temp)
        out["press_surf"] = np.array([float(out["press"][0])])
        out["temp_surf"] = np.array([float(out["temp"][0])])
        out["source"] = "local_fmcd"
        return out
    except Exception:
        return None


def _parse_ascii_table(text: str) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Parse MCD ASCII dump: first column altitude/x, following columns variables."""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        # skip non-numeric headers
        parts = s.replace(",", " ").split()
        try:
            vals = [float(x) for x in parts]
        except Exception:
            continue
        if len(vals) >= 2:
            lines.append(vals)
    if len(lines) < 3:
        return None
    arr = np.asarray(lines, dtype=float)
    x = arr[:, 0]
    cols = {f"c{i}": arr[:, i] for i in range(1, arr.shape[1])}
    return x, cols


def _web_fetch_one(
    lat: float,
    lon: float,
    loct: float,
    ls: float,
    var1: str,
    var2: str = "none",
    var3: str = "none",
    var4: str = "none",
    dust_scenario: int = 1,
    timeout: float = 60.0,
) -> Optional[str]:
    params = {
        "datekeyhtml": "1",
        "ls": f"{ls:.4f}",
        "localtime": f"{loct:.4f}",
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "altitude": "all",
        "zkey": "3",
        "var1": var1,
        "var2": var2,
        "var3": var3,
        "var4": var4,
        "isfixedlt": "on",
        "dust": str(int(dust_scenario)),
        "hrkey": "0",
        "averaging": "off",
        "animation": "off",
        "spacecraft": "none",
        "format": "png",
        "islog": "off",
        "colorm": "jet",
        "minval": "",
        "maxval": "",
        "proj": "",
        "zonmean": "off",
        "diumean": "off",
        "iswind": "off",
        "istherepoint": "off",
        "betatest": "off",
        "maths": "",
        "dpi": "",
        "trans": "",
        "palt": "",
        "plat": "",
        "plon": "",
        "animframes": "",
        "year": "",
        "month": "",
        "day": "",
        "hours": "",
        "minutes": "",
        "seconds": "",
        "julian": "",
        "martianyear": "",
        "sol": "",
        "latpoint": "",
        "lonpoint": "",
    }
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        MCD_CGI,
        data=data,
        method="POST",
        headers={
            "User-Agent": "SpectralApp-DISORT/1.0",
            "Referer": "https://www-mars.lmd.jussieu.fr/mcd_python/",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _extract_ascii_from_html(html: str) -> Optional[str]:
    # Prefer linked .txt / .asc / .dat
    for pat in (
        r'href=["\']([^"\']+\.(?:txt|asc|dat|csv))["\']',
        r'href=["\'](https?://[^"\']+/tmp/[^"\']+)["\']',
    ):
        m = re.search(pat, html, flags=re.I)
        if m:
            url = m.group(1)
            if url.startswith("./"):
                url = "https://www-mars.lmd.jussieu.fr/mcd_python/" + url[2:]
            elif url.startswith("/"):
                url = "https://www-mars.lmd.jussieu.fr" + url
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    return r.read().decode("utf-8", "replace")
            except Exception:
                continue
    # Sometimes ASCII is embedded in <pre>
    m = re.search(r"<pre[^>]*>(.*?)</pre>", html, flags=re.I | re.S)
    if m:
        return m.group(1)
    return None


def _try_web_mcd(
    lat: float,
    lon: float,
    loct: float,
    ls: float,
    n_columns: int,
    dust_scenario: int = 1,
) -> Optional[Dict[str, np.ndarray]]:
    collected = {}
    x_ref = None
    try:
        for batch in _WEB_VAR_BATCHES[:2]:  # limit requests; CGI is fragile
            html = _web_fetch_one(lat, lon, loct, ls, *batch, dust_scenario=dust_scenario)
            if not html:
                continue
            ascii_txt = _extract_ascii_from_html(html)
            if not ascii_txt:
                continue
            parsed = _parse_ascii_table(ascii_txt)
            if not parsed:
                continue
            x, cols = parsed
            if x_ref is None:
                x_ref = x
            # Map columns in request order
            names = [b for b in batch if b != "none"]
            for i, name in enumerate(names):
                key = f"c{i+1}"
                if key in cols:
                    collected[name] = cols[key]
    except Exception:
        return None

    if x_ref is None or "t" not in collected:
        return None

    heights = np.asarray(x_ref, dtype=float)
    # Interpolate onto fixed n_columns grid (surface→top)
    z_grid = _linspace_heights(n_columns, z_top_m=float(np.nanmax(heights)))
    out = empty_profile(n_columns)
    out["height"] = z_grid

    def _interp(name, default=0.0):
        if name not in collected:
            return np.full(n_columns, default)
        y = np.asarray(collected[name], dtype=float)
        order = np.argsort(heights)
        return np.interp(z_grid, heights[order], y[order], left=y[order][0], right=y[order][-1])

    out["temp"] = _interp("t", 210.0)
    out["press"] = _interp("p", 100.0)
    out["density"] = _interp("rho", 0.01)
    out["co2_mixradio"] = _interp("vmr_co2", 0.95)
    out["wv_mixradio"] = _interp("vmr_h2o", 0.0)
    out["dust_mixradio"] = _interp("dust_mmr", 0.0)
    out["dust_re"] = _interp("dust_reff", 1.5e-6)
    out["watice_mixradio"] = _interp("vmr_h2oice", 0.0)
    out["watice_re"] = _interp("h2oice_reff", 0.0)
    out["press_surf"] = np.array([float(out["press"][0])])
    out["temp_surf"] = np.array([float(out["temp"][0])])
    if "col_co2" in collected:
        out["co2_column"] = np.array([float(np.nanmean(collected["col_co2"]))])
    if "col_h2ovapor" in collected:
        out["wv_column"] = np.array([float(np.nanmean(collected["col_h2ovapor"]))])
    if "col_h2oice" in collected:
        out["watice_column"] = np.array([float(np.nanmean(collected["col_h2oice"]))])
    out["source"] = "mcd_web"
    return out


def fetch_mcd_profile(
    lat: float,
    lon: float,
    local_time_h: float,
    ls_deg: float,
    n_columns: int = 35,
    dust_scenario: int = 1,
    fallback_input_root: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Fetch a vertical atmospheric profile for one location/time.

    Parameters
    ----------
    lat, lon : degrees (planetocentric; east longitude)
    local_time_h : martian hours (0–24)
    ls_deg : solar longitude (degrees)
    """
    lat = float(lat)
    lon = float(((lon + 180) % 360) - 180)
    loct = float(local_time_h) % 24.0
    ls = float(ls_deg) % 360.0

    prof = _try_local_fmcd(lat, lon, loct, ls, n_columns, dust_scenario)
    if prof is not None:
        return prof

    prof = _try_web_mcd(lat, lon, loct, ls, n_columns, dust_scenario)
    if prof is not None:
        return prof

    if fallback_input_root:
        prof = profile_from_input_dir(fallback_input_root, n_columns=n_columns)
        prof["source"] = "fallback_input:" + str(prof.get("source", ""))
        prof["warning"] = (
            "未能连接本地/在线 MCD，已改用 input/ 大气表。"
            "几何角仍使用辅助图像像元值。"
        )
        return prof

    raise RuntimeError(
        "无法从 Mars Climate Database 读取大气廓线。\n"
        "可选方案：\n"
        "1) 安装并配置本地 MCD + fmcd/mcd-python，设置环境变量 MCD_DATA\n"
        "2) 确保可访问 https://www-mars.lmd.jussieu.fr/mcd_python/\n"
        "3) 提供含大气表的 input/ 目录作为回退"
    )


def apply_profile_to_atm(atm: dict, profile: Dict[str, np.ndarray]) -> dict:
    """Write MCD profile fields into an atm bundle from load_input_bundle."""
    n = atm["height"].size
    h = np.asarray(profile["height"], dtype=float)
    if h.size != n:
        # resample profile onto atm height grid if sizes differ
        z_dst = atm["height"]
        order = np.argsort(h)

        def rsz(key, default=0.0):
            y = np.asarray(profile[key], dtype=float)
            if y.size == 1:
                return np.full(n, float(y.ravel()[0]))
            return np.interp(z_dst, h[order], y[order], left=y[order][0], right=y[order][-1])

        atm["density"][0] = rsz("density")
        atm["press"][:] = rsz("press")
        atm["temp"][0] = rsz("temp")
        atm["co2_mixradio"][0] = rsz("co2_mixradio")
        atm["dust_mixradio"][0] = rsz("dust_mixradio")
        atm["dust_re"][0] = rsz("dust_re")
        atm["watice_mixradio"][0] = rsz("watice_mixradio")
        atm["watice_re"][0] = rsz("watice_re")
        atm["wv_mixradio"][0] = rsz("wv_mixradio")
    else:
        atm["height"][:] = profile["height"]
        atm["density"][0] = profile["density"]
        atm["press"][:] = profile["press"]
        atm["temp"][0] = profile["temp"]
        atm["co2_mixradio"][0] = profile["co2_mixradio"]
        atm["dust_mixradio"][0] = profile["dust_mixradio"]
        atm["dust_re"][0] = profile["dust_re"]
        atm["watice_mixradio"][0] = profile["watice_mixradio"]
        atm["watice_re"][0] = profile["watice_re"]
        atm["wv_mixradio"][0] = profile["wv_mixradio"]

    atm["co2_column"][0] = float(np.asarray(profile["co2_column"]).ravel()[0])
    atm["wv_column"][0] = float(np.asarray(profile["wv_column"]).ravel()[0])
    atm["watice_column"][0] = float(np.asarray(profile["watice_column"]).ravel()[0])
    atm["press_surf"][0] = float(np.asarray(profile["press_surf"]).ravel()[0])
    atm["temp_surf"][0] = float(np.asarray(profile["temp_surf"]).ravel()[0])
    return atm


class MCDProfileCache:
    """Cache MCD profiles by quantized lat/lon/lt/Ls to speed up image mode."""

    def __init__(self, lat_q=0.5, lon_q=0.5, lt_q=0.5, ls_q=5.0):
        self.lat_q = lat_q
        self.lon_q = lon_q
        self.lt_q = lt_q
        self.ls_q = ls_q
        self._cache = {}

    def _key(self, lat, lon, lt, ls):
        return (
            round(lat / self.lat_q) * self.lat_q,
            round(lon / self.lon_q) * self.lon_q,
            round(lt / self.lt_q) * self.lt_q,
            round(ls / self.ls_q) * self.ls_q,
        )

    def get(self, lat, lon, lt, ls, **kwargs):
        k = self._key(lat, lon, lt, ls)
        if k not in self._cache:
            self._cache[k] = fetch_mcd_profile(lat, lon, lt, ls, **kwargs)
        return self._cache[k]
