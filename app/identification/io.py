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


def _as_um(wavelengths: np.ndarray) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()
    finite = wl[np.isfinite(wl)]
    if finite.size and float(np.nanmax(np.abs(finite))) > 100:
        wl = wl / 1000.0
    return wl


def load_wavelengths(path: str | Path) -> Optional[np.ndarray]:
    """Read wavelength vector in μm from ENVI/PDS headers or .mat/.npz keys."""
    path = Path(path)
    ext = _suffix(path)

    if ext in {".img", ".dat", ".hdr", ".lbl", ".bsq", ".bil", ".bip"}:
        try:
            import spectral.io.envi as envi

            open_path = resolve_open_path(path)
            header = open_path
            raster = None
            if _suffix(header) in HEADER_EXTENSIONS:
                raster = _sidecar(header, RASTER_EXTENSIONS)
            else:
                raster = header
                found = _sidecar(header, HEADER_EXTENSIONS)
                if found is not None:
                    header = found
            if raster is not None and _suffix(header) in HEADER_EXTENSIONS:
                img = envi.open(str(header), image=str(raster))
            else:
                img = envi.open(str(header))
            centers = getattr(getattr(img, "bands", None), "centers", None)
            if centers is not None and len(centers) > 0:
                return _as_um(np.asarray(centers, dtype=np.float64))
            meta = getattr(img, "metadata", {}) or {}
            raw = meta.get("wavelength") or meta.get("Wavelength")
            if raw:
                values = [float(item) for item in raw]
                return _as_um(np.asarray(values, dtype=np.float64))
        except Exception:
            pass

    if ext == ".mat":
        try:
            from scipy.io import loadmat

            mat = loadmat(path)
            for key in ("wavelengths", "wavelength", "wl", "wvl", "lambda"):
                if key not in mat:
                    continue
                arr = np.squeeze(np.asarray(mat[key]))
                if arr.ndim == 1 and arr.size >= 8 and np.issubdtype(arr.dtype, np.number):
                    return _as_um(arr)
        except Exception:
            pass

    if ext == ".npz":
        try:
            payload = np.load(path, allow_pickle=False)
            for key in ("wavelengths", "wavelength", "wl", "wvl"):
                if key not in payload.files:
                    continue
                arr = np.squeeze(np.asarray(payload[key]))
                if arr.ndim == 1 and arr.size >= 8:
                    return _as_um(arr)
        except Exception:
            pass

    return None


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
        files = [p for p in files if not is_classification_output(p)]
        if not files:
            raise FileNotFoundError(
                f"No supported cubes matching {pattern!r} in {path}"
            )
        return [str(p) for p in files]

    raise FileNotFoundError(f"Input path not found: {path}")


def filter_class_map(class_map, class_id) -> np.ndarray:
    """Keep one 1-based class; 0 / None / '' keeps every class."""
    shown = np.asarray(class_map).copy()
    if class_id is None or str(class_id).strip() in {"", "0"}:
        return shown
    keep = int(class_id)
    shown[shown != keep] = 0
    return shown


_OUTPUT_STRIP = (
    "_lsga_classification",
    "_hbm_classification_codes",
    "_hbm_classification",
    "_classification_codes",
    "_classification",
    "_hbm_codes",
    "_hbm_class",
    "_class",
)


def classification_stem(source_name: str | Path, method: str) -> str:
    """``<input>_<LSGA|HBM>_classification``; does not stack extra suffixes."""
    tag = str(method or "").strip()
    if tag.lower() == "lsga":
        tag = "LSGA"
    elif tag.lower() == "hbm":
        tag = "HBM"
    else:
        tag = tag.upper() or "CLASS"
    stem = Path(source_name).stem
    lower = stem.lower()
    for suffix in _OUTPUT_STRIP:
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem.strip("_").strip() or "scene"
    return f"{stem}_{tag}_classification"


def is_classification_output(path: str | Path) -> bool:
    """True for previously saved classification rasters (skip when batching)."""
    stem = Path(path).stem.lower()
    return any(stem.endswith(suffix) for suffix in _OUTPUT_STRIP)


# Same palette as the overlay (index 0 = Unclassified / black).
_CLASS_LOOKUP_HEX = [
    "#000000", "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#66C2A5",
    "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F",
    "#E5C494", "#B3B3B3", "#1B9E77", "#D95F02", "#7570B3",
    "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666",
]


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hsv_rgb(hue: float, sat: float = 0.85, val: float = 0.95) -> tuple[int, int, int]:
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    i %= 6
    r, g, b = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)][i]
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _class_lookup_rgb(n_classes: int) -> list[int]:
    """Flattened RGB triplets, one per class, as ENVI `class lookup` requires."""
    flat: list[int] = []
    for index in range(int(n_classes)):
        if index < len(_CLASS_LOOKUP_HEX):
            flat.extend(_hex_rgb(_CLASS_LOOKUP_HEX[index]))
        else:
            hue = ((index * 0.61803398875) % 1.0)
            flat.extend(_hsv_rgb(hue))
    return flat


def _sanitize_envi_class_name(name: str) -> str:
    text = str(name).replace("{", "").replace("}", "").replace(",", " ")
    text = " ".join(text.split())
    return text or "class"


def write_envi_class_map(
    path: str | Path,
    class_map: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """Write a 1-band ENVI classification (.img + .hdr).

    ENVI requires ``classes``, ``class names``, and ``class lookup`` (RGB)
    to open the file as a classification image. Values fit in unsigned
    8-bit (data type 1), matching ENVI's usual classification format.
    """
    img_path = Path(path)
    if img_path.suffix.lower() != ".img":
        img_path = img_path.with_suffix(".img")
    hdr_path = img_path.with_suffix(".hdr")
    raw = np.asarray(class_map)
    if raw.ndim != 2:
        raise ValueError(f"Classification map must be 2D, got {raw.shape}")
    if np.nanmax(raw) >= 256 or np.nanmin(raw) < 0:
        raise ValueError(
            "ENVI 分类图需要类别号在 0–255。当前 "
            f"min={int(np.nanmin(raw))}, max={int(np.nanmax(raw))}。"
        )
    arr = np.ascontiguousarray(np.nan_to_num(raw, nan=0.0), dtype=np.uint8)
    height, width = arr.shape
    img_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(img_path)

    names = [_sanitize_envi_class_name(n) for n in (class_names or [])]
    max_id = int(arr.max()) if arr.size else 0
    n_classes = max(max_id, len(names), 0) + 1
    class_list = ["Unclassified"]
    for i in range(1, n_classes):
        if i - 1 < len(names):
            class_list.append(names[i - 1])
        else:
            class_list.append(f"class_{i}")
    joined_names = ", ".join(class_list)
    lookup = ", ".join(str(v) for v in _class_lookup_rgb(n_classes))
    header = (
        "ENVI\n"
        "description = {Identification class map}\n"
        f"samples = {width}\n"
        f"lines = {height}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Classification\n"
        "data type = 1\n"
        "interleave = bsq\n"
        "byte order = 0\n"
        f"classes = {n_classes}\n"
        f"class names = {{{joined_names}}}\n"
        f"class lookup = {{{lookup}}}\n"
        "band names = {class id}\n"
    )
    hdr_path.write_text(header, encoding="ascii", errors="replace")
    return img_path
