"""
Resolve local Mars Climate Database install paths.

Looks for (in order):
  1. Environment variable MCD_DATA
  2. data/mcd/MCD_DATA.path written by scripts/install_mcd.py
  3. data/mcd/MCD/data/ or data/mcd/data/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_APP_ROOT = Path(__file__).resolve().parents[2]  # repo root (…/app/disort → …)
_DEFAULT_DIR = _APP_ROOT / "data" / "mcd"


def mcd_install_dir() -> Path:
    return Path(os.environ.get("MCD_INSTALL_DIR", str(_DEFAULT_DIR)))


def resolve_mcd_data(explicit: Optional[str] = None) -> Optional[str]:
    """Return MCD data directory path with trailing separator, or None."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("MCD_DATA")
    if env:
        candidates.append(env)

    marker = mcd_install_dir() / "MCD_DATA.path"
    if marker.is_file():
        try:
            candidates.append(marker.read_text(encoding="utf-8").strip())
        except OSError:
            pass

    for rel in (
        mcd_install_dir() / "MCD" / "data",
        mcd_install_dir() / "data",
    ):
        candidates.append(str(rel))

    for c in candidates:
        if not c:
            continue
        p = Path(c.rstrip("/\\"))
        if p.is_dir():
            return str(p.resolve()) + os.sep
    return None


def ensure_mcd_data_env() -> Optional[str]:
    """Set os.environ['MCD_DATA'] if a local install is found."""
    path = resolve_mcd_data()
    if path:
        os.environ["MCD_DATA"] = path
    return path
