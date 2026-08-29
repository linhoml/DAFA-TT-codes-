"""Load identification cubes and labels from several raster formats."""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

CUBE_EXTENSIONS = (
    ".mat",
    ".img",
    ".dat",
    ".hdr",
    ".lbl",
    ".bsq",
    ".bil",
    ".bip",
    ".npy",
    ".npz",
    ".tif",
    ".tiff",
)

LABEL_EXTENSIONS = CUBE_EXTENSIONS

HEADER_EXTENSIONS = (".hdr", ".lbl")
RASTER_EXTENSIONS = (".img", ".dat", ".bsq", ".bil", ".bip")

FILE_FILTER_CUBE = (
    "Hyperspectral (*.mat *.img *.dat *.hdr *.lbl *.bsq *.bil *.bip *.npy *.tif);;"
    "MATLAB (*.mat);;"
    "ENVI / PDS (*.hdr *.img *.dat *.lbl *.bsq *.bil *.bip);;"
    "NumPy (*.npy *.npz);;"
    "GeoTIFF (*.tif *.tiff);;"
    "All Files (*)"
)

FILE_FILTER_LABEL = (
    "Label maps (*.mat *.img *.dat *.hdr *.lbl *.npy *.tif);;"
    "MATLAB (*.mat);;"
    "ENVI / PDS (*.hdr *.img *.dat *.lbl);;"
    "NumPy (*.npy *.npz);;"
    "GeoTIFF (*.tif *.tiff);;"
    "All Files (*)"
)

DEFAULT_INPUT_PATTERN = "*"


def _suffix(path: Path) -> str:
    return path.suffix.lower()


def natural_sort_key(path: str | Path) -> Tuple:
    """Sort tile_2 before tile_10 (lexicographic order would reverse them)."""
    text = str(Path(path))
    parts = re.split(r"(\d+)", text.lower())
    key = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part)))
        elif part:
            key.append((0, part))
    return tuple(key)


def is_supported_cube(path: str | Path) -> bool:
    return _suffix(Path(path)) in CUBE_EXTENSIONS


def is_supported_label(path: str | Path) -> bool:
    return _suffix(Path(path)) in LABEL_EXTENSIONS


def _sidecar(path: Path, extensions: Sequence[str]) -> Optional[Path]:
    for ext in extensions:
        candidate = path.with_suffix(ext)
        if candidate.exists():
            return candidate
        alt = path.with_suffix(ext.upper())
        if alt.exists():
            return alt
    return None


def resolve_open_path(path: str | Path) -> Path:
    """Prefer an ENVI/PDS header when the user picked .img/.dat."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    ext = _suffix(path)
    if ext in RASTER_EXTENSIONS:
        header = _sidecar(path, HEADER_EXTENSIONS)
        if header is not None:
            return header
    return path


def _load_mat(
    path: Path,
    *,
    key: Optional[str],
    prefer_3d: bool,
) -> np.ndarray:
    from scipy.io import loadmat

    mat = loadmat(path)
    if key:
        if key not in mat:
            available = [name for name in mat if not name.startswith("__")]
            raise KeyError(
                f"Key {key!r} not found in {path}. Available keys: {available}"
            )
        value = mat[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key!r} in {path} is not an ndarray")
        return value

    arrays = [
        value
        for name, value in mat.items()
        if not name.startswith("__") and isinstance(value, np.ndarray)
    ]
    if not arrays:
        raise ValueError(f"No ndarray found in {path}")
    if prefer_3d:
        for value in arrays:
            if value.ndim == 3:
                return value
    return arrays[0]


def _load_npy_npz(
    path: Path,
    *,
    key: Optional[str],
    prefer_3d: bool,
) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if _suffix(path) == ".npy":
        return np.asarray(payload)
    if key:
        if key not in payload.files:
            raise KeyError(
                f"Key {key!r} not found in {path}. Available keys: {list(payload.files)}"
            )
        return np.asarray(payload[key])
    arrays = [(name, np.asarray(payload[name])) for name in payload.files]
    if prefer_3d:
        for _, value in arrays:
            if value.ndim == 3:
                return value
    return arrays[0][1]


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


def _parse_envi_header(header_path: Path) -> dict:
    meta = {}
    for raw in header_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("envi") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip().lower()] = value.strip().strip("{}").strip()
    return meta


def _load_envi_binary(header_path: Path, raster_path: Path) -> np.ndarray:
    meta = _parse_envi_header(header_path)
    samples = int(meta["samples"])
    lines = int(meta["lines"])
    bands = int(float(meta.get("bands", "1")))
    dtype = _ENVI_DTYPE.get(int(float(meta.get("data type", "4"))))
    if dtype is None:
        raise ValueError(f"Unsupported ENVI data type in {header_path}")
    offset = int(float(meta.get("header offset", "0")))
    interleave = meta.get("interleave", "bsq").lower()
    count = lines * samples * bands
    with open(raster_path, "rb") as handle:
        handle.seek(offset)
        raw = np.fromfile(handle, dtype=dtype, count=count)
    if raw.size < count:
        raise ValueError(
            f"ENVI binary too short: {raster_path} need {count}, got {raw.size}"
        )
    if interleave == "bip":
        cube = raw.reshape(lines, samples, bands)
    elif interleave == "bil":
        cube = raw.reshape(lines, bands, samples).transpose(0, 2, 1)
    else:
        cube = raw.reshape(bands, lines, samples).transpose(1, 2, 0)
    return np.ascontiguousarray(cube)


def _load_envi_or_pds(path: Path) -> np.ndarray:
    open_path = resolve_open_path(path)
    ext = _suffix(open_path)

    if ext in {".lbl", ".img"}:
        try:
            from disort.pds_label import load_pds_cube

            cube, *_ = load_pds_cube(str(open_path if ext == ".lbl" else path))
            return np.asarray(cube)
        except Exception:
            pass

    header = open_path
    raster = None
    if _suffix(header) in HEADER_EXTENSIONS:
        raster = _sidecar(header, RASTER_EXTENSIONS)
    else:
        raster = header
        found = _sidecar(header, HEADER_EXTENSIONS)
        if found is not None:
            header = found

    try:
        import spectral.io.envi as envi

        if raster is not None and _suffix(header) in HEADER_EXTENSIONS:
            img = envi.open(str(header), image=str(raster))
        else:
            img = envi.open(str(header))
        return np.array(img.load())
    except Exception:
        if raster is None or not Path(header).exists():
            raise
        return _load_envi_binary(Path(header), Path(raster))


def _load_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(str(path)))
    except ImportError:
        pass
    try:
        from PIL import Image

        return np.asarray(Image.open(path))
    except Exception as exc:
        raise ImportError(
            f"Reading TIFF {path} needs tifffile or Pillow."
        ) from exc


def load_array(
    path: str | Path,
    *,
    key: Optional[str] = None,
    prefer_3d: bool = True,
) -> np.ndarray:
    """Load a cube or map from .mat / ENVI / PDS / NumPy / TIFF."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    ext = _suffix(path)

    if ext == ".mat":
        value = _load_mat(path, key=key, prefer_3d=prefer_3d)
    elif ext in {".npy", ".npz"}:
        value = _load_npy_npz(path, key=key, prefer_3d=prefer_3d)
    elif ext in {".tif", ".tiff"}:
        value = _load_tiff(path)
    elif ext in {".img", ".dat", ".hdr", ".lbl", ".bsq", ".bil", ".bip"}:
        value = _load_envi_or_pds(path)
    else:
        raise ValueError(
            f"Unsupported format {ext} ({path}). "
            "Use .mat, .img, .dat, .hdr, .lbl, .bsq/.bil/.bip, .npy/.npz, or .tif."
        )

    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"Empty array in {path}")
    return array


def load_cube(
    path: str | Path,
    *,
    key: Optional[str] = None,
    data_layout: str = "HWB",
    prefer_3d: bool = True,
) -> np.ndarray:
    """Load a 3D cube and return Height×Width×Bands."""
    cube = load_array(path, key=key, prefer_3d=prefer_3d)
    cube = np.squeeze(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape={cube.shape}: {path}")

    layout = str(data_layout or "HWB").upper()
    if layout not in {"HWB", "BHW"}:
        raise ValueError("data_layout must be 'HWB' or 'BHW'")

    ext = _suffix(Path(path))
    envi_like = ext in {".img", ".dat", ".hdr", ".lbl", ".bsq", ".bil", ".bip"}
    # ENVI/PDS cubes from spectral are already rows × cols × bands.
    if layout == "BHW" and not envi_like:
        cube = np.transpose(cube, (1, 2, 0))
    return cube


def load_label_array(
    path: str | Path,
    *,
    key: Optional[str] = None,
) -> np.ndarray:
    """Load a 2D integer label map."""
    label = np.squeeze(load_array(path, key=key, prefer_3d=False))
    if label.ndim == 3 and label.shape[-1] == 1:
        label = label[:, :, 0]
    if label.ndim == 3 and label.shape[0] == 1:
        label = label[0]
    if label.ndim != 2:
        raise ValueError(
            f"Label map must be 2D, got {label.shape}: {path}"
        )
    return label.astype(np.int64)


def _matches_supported(path: Path, pattern: str) -> bool:
    if not is_supported_cube(path):
        return False
    name = path.name
    if pattern in {"*", "*.*", "", "all"}:
        return True
    return path.match(pattern) or Path(name).match(pattern)


def unique_dataset_paths(paths: Sequence[Path]) -> List[Path]:
    """Drop .img/.dat when the same stem already has .hdr/.lbl."""
    ordered = sorted({Path(p).resolve() for p in paths})
    used = set()
    result: List[Path] = []

    def stem_key(path: Path):
        return (str(path.parent), path.stem.lower())

    for path in ordered:
        if _suffix(path) in {".mat", ".npy", ".npz", ".tif", ".tiff", ".hdr", ".lbl"}:
            result.append(path)
            used.add(stem_key(path))

    for path in ordered:
        key = stem_key(path)
        if key in used:
            continue
        if _suffix(path) in RASTER_EXTENSIONS:
            result.append(path)
            used.add(key)
    return sorted(result, key=natural_sort_key)


def list_input_files(
    input_path: str | Path,
    input_pattern: str = DEFAULT_INPUT_PATTERN,
) -> List[str]:
    """One file, or a folder of supported cubes (mat/img/dat/hdr/…)."""
    path = Path(input_path)
    pattern = (input_pattern or DEFAULT_INPUT_PATTERN).strip() or DEFAULT_INPUT_PATTERN

    if path.is_file():
        if not is_supported_cube(path):
            raise ValueError(
                f"Unsupported input file: {path}\n"
                "Supported: .mat, .img, .dat, .hdr, .lbl, .bsq/.bil/.bip, .npy/.npz, .tif"
            )
        return [str(path)]

    if path.is_dir():
        if pattern in {"*", "*.*", "all"}:
            raw = []
            for ext in CUBE_EXTENSIONS:
                raw.extend(path.glob(f"*{ext}"))
                raw.extend(path.glob(f"*{ext.upper()}"))
        else:
            raw = [Path(p) for p in sorted(glob.glob(str(path / pattern)))]
            if not raw:
                # Allow patterns like "*.img" while a folder also has headers.
                raw = [p for p in path.iterdir() if p.is_file() and _matches_supported(p, pattern)]

        files = unique_dataset_paths([p for p in raw if p.is_file()])
        if not files:
            raise FileNotFoundError(
                f"No supported cubes matching {pattern!r} in {path}"
            )
        return [str(p) for p in files]

    raise FileNotFoundError(f"Input path not found: {path}")
