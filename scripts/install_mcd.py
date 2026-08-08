#!/usr/bin/env python3
"""
Install / configure a local Mars Climate Database (MCD) tree for SpectralApp.

The full MCD (NetCDF data + Fortran access software) is distributed by LMD
upon request only:
  https://www-mars.lmd.jussieu.fr/MCD_pro/mcd_pro.html
  Contact: millour@lmd.jussieu.fr , forget@lmd.jussieu.fr

Once you receive a download URL or a local .tar/.zip archive, run:

  python scripts/install_mcd.py --archive /path/to/MCD.tar.gz
  # or
  python scripts/install_mcd.py --url 'https://...(link from LMD email)...'

This extracts into data/mcd/ and writes data/mcd/MCD_DATA.path plus a
shell snippet that exports MCD_DATA for fmcd / mcd-python.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "mcd"
MARKER = "MCD_DATA.path"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  downloading {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
    else:
        print(f"\r  downloading {downloaded / (1024 * 1024):.1f} MB", end="", flush=True)


def download(url: str, dest_file: Path) -> Path:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading:\n  {url}\n→ {dest_file}")
    req = Request(url, headers={"User-Agent": "SpectralApp-MCD-Installer/1.0"})
    # urlretrieve does not take Request; use urlopen manually for headers
    from urllib.request import urlopen

    with urlopen(req, timeout=600) as resp, open(dest_file, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            if total:
                print(
                    f"\r  downloading {got/1024/1024:.1f}/{total/1024/1024:.1f} MB "
                    f"({100.0*got/total:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r  downloading {got/1024/1024:.1f} MB", end="", flush=True)
    print()
    return dest_file


def _looks_like_mcd_root(path: Path) -> bool:
    """True if path contains data/ and mcd/ as in the official distribution."""
    return (path / "data").is_dir() and (path / "mcd").is_dir()


def find_mcd_root(extract_dir: Path) -> Path:
    if _looks_like_mcd_root(extract_dir):
        return extract_dir
    # One nesting level (common for tar.gz)
    for child in sorted(extract_dir.iterdir()):
        if child.is_dir() and _looks_like_mcd_root(child):
            return child
    # Broader search
    for data_dir in extract_dir.rglob("data"):
        parent = data_dir.parent
        if _looks_like_mcd_root(parent):
            return parent
    raise FileNotFoundError(
        f"在 {extract_dir} 中未找到标准 MCD 目录结构（需要同时有 data/ 与 mcd/）。"
    )


def extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / "_extract_tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    name = archive.name.lower()
    print(f"Extracting {archive} …")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(staging)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz")):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(staging)
    else:
        # Try tar then zip
        try:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(staging)
        except Exception:
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(staging)

    root = find_mcd_root(staging)
    # Move into dest (replace previous install contents carefully)
    final = dest / "MCD"
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(root), str(final))
    shutil.rmtree(staging, ignore_errors=True)
    return final


def write_env_files(mcd_root: Path, dest: Path) -> Path:
    """
    MCD access software expects dset = path to the data/ directory
    (with trailing slash recommended).
    """
    data_dir = mcd_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"缺少 data/ 目录：{data_dir}")

    data_path = str(data_dir.resolve()) + os.sep
    marker = dest / MARKER
    marker.write_text(data_path + "\n", encoding="utf-8")

    sh = dest / "env.sh"
    sh.write_text(
        f"# Source this file before running SpectralApp / fmcd\n"
        f"export MCD_DATA=\"{data_path}\"\n"
        f"export MCD_ROOT=\"{mcd_root.resolve()}\"\n"
        f'export PYTHONPATH="{ROOT / "third_party" / "mcd-python"}'
        f':{mcd_root / "mcd" / "interfaces" / "python"}:$PYTHONPATH"\n',
        encoding="utf-8",
    )

    # Convenience symlink at dest/data → real data dir
    link = dest / "data"
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            # keep existing directory if it is the real one
            pass
    if not link.exists():
        try:
            link.symlink_to(data_dir.resolve(), target_is_directory=True)
        except OSError:
            pass

    print(f"Wrote {marker}")
    print(f"Wrote {sh}")
    print(f"MCD_DATA={data_path}")
    return marker


def status(dest: Path = DEFAULT_DEST) -> int:
    marker = dest / MARKER
    print(f"Install dir: {dest}")
    if marker.is_file():
        path = marker.read_text(encoding="utf-8").strip()
        ok = os.path.isdir(path.rstrip("/\\"))
        print(f"MCD_DATA marker: {path}  ({'OK' if ok else 'MISSING'})")
        env = os.environ.get("MCD_DATA")
        print(f"env MCD_DATA: {env or '(not set)'}")
        return 0 if ok else 1
    print("Not installed yet. Register at:")
    print("  https://www-mars.lmd.jussieu.fr/MCD_pro/mcd_pro.html")
    print("Then run: python scripts/install_mcd.py --archive <file> | --url <link>")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Install root (default: data/mcd)")
    p.add_argument("--archive", type=Path, help="Local MCD .tar.gz / .zip from LMD")
    p.add_argument("--url", type=str, help="Download URL provided by LMD after registration")
    p.add_argument("--status", action="store_true", help="Show local MCD install status")
    p.add_argument("--keep-download", action="store_true", help="Keep downloaded archive under dest/downloads")
    args = p.parse_args(argv)

    if args.status or (not args.archive and not args.url):
        return status(args.dest)

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    archive: Optional[Path] = args.archive

    if args.url:
        dl_dir = dest / "downloads"
        dl_dir.mkdir(parents=True, exist_ok=True)
        fname = args.url.rstrip("/").split("/")[-1] or "mcd_download.tar.gz"
        # strip query string from filename
        fname = fname.split("?")[0] or "mcd_download.tar.gz"
        archive = download(args.url, dl_dir / fname)

    assert archive is not None
    if not archive.is_file():
        print(f"Archive not found: {archive}", file=sys.stderr)
        return 2

    mcd_root = extract_archive(archive, dest)
    write_env_files(mcd_root, dest)

    if args.url and archive and not args.keep_download:
        try:
            archive.unlink()
        except OSError:
            pass

    print("\nNext steps:")
    print(f"  1) source {dest / 'env.sh'}")
    print("  2) Compile Fortran interface (needs gfortran + NetCDF), see:")
    print("       third_party/mcd-python/README.md")
    print(f"       and {mcd_root / 'mcd'} (official python/fmcd scripts)")
    print("  3) Re-run SpectralApp; DISORT will prefer local fmcd / MCD_DATA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
