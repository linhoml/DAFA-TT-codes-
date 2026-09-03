"""Load identification cubes and labels from several raster formats."""

from __future__ import annotations

import glob
import re
from difflib import SequenceMatcher
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


MAX_INCORE_BYTES = 512 * 1024 * 1024
DEFAULT_CUBE_WINDOW = 512


def format_cube_memory(height: int, width: int, bands: int, itemsize: int = 4) -> str:
    nbytes = int(height) * int(width) * int(bands) * int(itemsize)
    if nbytes >= 1024 ** 3:
        return f"{nbytes / (1024 ** 3):.1f} GiB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / (1024 ** 2):.1f} MiB"
    return f"{nbytes / 1024:.1f} KiB"


def cube_nbytes(height: int, width: int, bands: int, itemsize: int = 4) -> int:
    return int(height) * int(width) * int(bands) * int(itemsize)


def should_load_cube_in_memory(
    height: int,
    width: int,
    bands: int,
    *,
    max_bytes: int = MAX_INCORE_BYTES,
) -> bool:
    return cube_nbytes(height, width, bands) <= int(max_bytes)


def _envi_header_and_raster(path: Path) -> Tuple[Path, Optional[Path]]:
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
    return header, raster


def _envi_dtype(meta: dict) -> np.dtype:
    code = int(float(meta.get("data type", "4")))
    base = _ENVI_DTYPE.get(code)
    if base is None:
        raise ValueError(f"Unsupported ENVI data type {code}")
    endian = ">" if int(float(meta.get("byte order", "0"))) == 1 else "<"
    return np.dtype(base).newbyteorder(endian)


def probe_cube_shape(
    path: str | Path,
    *,
    key: Optional[str] = None,
    data_layout: str = "HWB",
) -> Tuple[int, int, int]:
    """Return (height, width, bands) without loading the full cube when possible."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    ext = _suffix(path)
    layout = str(data_layout or "HWB").upper()

    if ext in {".img", ".dat", ".hdr", ".lbl", ".bsq", ".bil", ".bip"}:
        open_path = resolve_open_path(path)
        if _suffix(open_path) == ".lbl" or (
            ext in {".img", ".lbl"} and _sidecar(path, (".lbl",)) is not None
            and _sidecar(path, (".hdr",)) is None
        ):
            try:
                from disort.pds_label import parse_pds3_label_file, resolve_lbl_img_paths

                lbl_path, _img_path = resolve_lbl_img_paths(str(path))
                meta = parse_pds3_label_file(lbl_path)
                height = int(float(meta["LINES"]))
                width = int(float(meta["LINE_SAMPLES"]))
                bands = int(float(meta.get("BANDS", 1)))
                return height, width, bands
            except Exception:
                pass
        header, _raster = _envi_header_and_raster(path)
        if _suffix(header) == ".hdr":
            meta = _parse_envi_header(header)
            height = int(float(meta["lines"]))
            width = int(float(meta["samples"]))
            bands = int(float(meta.get("bands", "1")))
            return height, width, bands

    if ext == ".npy":
        array = np.load(path, mmap_mode="r")
        shape = tuple(int(x) for x in array.shape)
        del array
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D cube, got shape={shape}: {path}")
        if layout == "BHW":
            bands, height, width = shape
            return height, width, bands
        height, width, bands = shape
        return height, width, bands

    if ext == ".mat":
        from scipy.io import whosmat

        entries = whosmat(str(path))
        if key:
            match = [item for item in entries if item[0] == key]
            if not match:
                raise KeyError(f"Key {key!r} not found in {path}")
            shape = tuple(int(x) for x in match[0][1])
        else:
            ranked = sorted(
                entries,
                key=lambda item: (
                    0 if len(item[1]) == 3 else 1,
                    -int(np.prod(item[1])) if item[1] else 0,
                ),
            )
            if not ranked:
                raise ValueError(f"No arrays in {path}")
            shape = tuple(int(x) for x in ranked[0][1])
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D cube, got shape={shape}: {path}")
        if layout == "BHW":
            bands, height, width = shape
            return height, width, bands
        height, width, bands = shape
        return height, width, bands

    if ext in {".tif", ".tiff"}:
        try:
            import tifffile

            with tifffile.TiffFile(str(path)) as tif:
                shape = tuple(int(x) for x in tif.series[0].shape)
            if len(shape) == 2:
                height, width = shape
                return height, width, 1
            if len(shape) == 3:
                if layout == "BHW":
                    bands, height, width = shape
                    return height, width, bands
                height, width, bands = shape
                return height, width, bands
        except Exception:
            pass

    if path.stat().st_size <= MAX_INCORE_BYTES:
        cube = load_cube(path, key=key, data_layout=data_layout)
        height, width, bands = cube.shape
        del cube
        return int(height), int(width), int(bands)

    raise MemoryError(
        f"立方体 {path.name} 约 {path.stat().st_size / (1024 ** 3):.1f} GiB，"
        "无法整幅载入以探测尺寸。请使用带 .hdr 的 ENVI、PDS .lbl 或 .npy。"
    )


def _clip_window(
    height: int, width: int, row0: int, row1: int, col0: int, col1: int
) -> Tuple[int, int, int, int]:
    r0 = max(0, int(row0))
    r1 = min(int(height), int(row1))
    c0 = max(0, int(col0))
    c1 = min(int(width), int(col1))
    if r1 <= r0 or c1 <= c0:
        raise ValueError(
            f"Empty cube window rows={row0}:{row1} cols={col0}:{col1} "
            f"in {height}x{width}"
        )
    return r0, r1, c0, c1


def _read_envi_window(
    header_path: Path,
    raster_path: Path,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
) -> np.ndarray:
    meta = _parse_envi_header(header_path)
    samples = int(float(meta["samples"]))
    lines = int(float(meta["lines"]))
    bands = int(float(meta.get("bands", "1")))
    dtype = _envi_dtype(meta)
    offset = int(float(meta.get("header offset", "0")))
    interleave = str(meta.get("interleave", "bsq")).lower()
    r0, r1, c0, c1 = _clip_window(lines, samples, row0, row1, col0, col1)
    height, width = r1 - r0, c1 - c0
    itemsize = int(dtype.itemsize)
    out = np.empty((height, width, bands), dtype=np.float32)

    with open(raster_path, "rb") as handle:
        if interleave == "bip":
            line_stride = samples * bands * itemsize
            for i, row in enumerate(range(r0, r1)):
                handle.seek(offset + row * line_stride + c0 * bands * itemsize)
                chunk = np.fromfile(handle, dtype=dtype, count=width * bands)
                if chunk.size < width * bands:
                    raise ValueError(f"ENVI BIP window short: {raster_path}")
                out[i] = chunk.reshape(width, bands).astype(np.float32, copy=False)
            return out

        if interleave == "bil":
            line_stride = bands * samples * itemsize
            for i, row in enumerate(range(r0, r1)):
                for band in range(bands):
                    handle.seek(
                        offset
                        + row * line_stride
                        + band * samples * itemsize
                        + c0 * itemsize
                    )
                    chunk = np.fromfile(handle, dtype=dtype, count=width)
                    if chunk.size < width:
                        raise ValueError(f"ENVI BIL window short: {raster_path}")
                    out[i, :, band] = chunk.astype(np.float32, copy=False)
            return out

        band_stride = lines * samples * itemsize
        row_stride = samples * itemsize
        for band in range(bands):
            handle.seek(offset + band * band_stride + r0 * row_stride)
            plane = np.fromfile(handle, dtype=dtype, count=height * samples)
            if plane.size < height * samples:
                raise ValueError(f"ENVI BSQ window short: {raster_path}")
            out[:, :, band] = plane.reshape(height, samples)[:, c0:c1].astype(
                np.float32, copy=False
            )
    return out


def _read_pds_window(
    path: Path, row0: int, row1: int, col0: int, col1: int
) -> np.ndarray:
    from disort.pds_label import (
        _dtype_from_sample_type,
        parse_pds3_label_file,
        resolve_lbl_img_paths,
    )

    lbl_path, img_path = resolve_lbl_img_paths(str(path))
    meta = parse_pds3_label_file(lbl_path)
    lines = int(float(meta["LINES"]))
    samples = int(float(meta["LINE_SAMPLES"]))
    bands = int(float(meta.get("BANDS", 1)))
    dtype = _dtype_from_sample_type(
        str(meta.get("SAMPLE_TYPE", "PC_REAL")),
        int(float(meta.get("SAMPLE_BITS", 32))),
    )
    storage = str(meta.get("BAND_STORAGE_TYPE", "BAND_SEQUENTIAL")).upper()
    r0, r1, c0, c1 = _clip_window(lines, samples, row0, row1, col0, col1)
    height, width = r1 - r0, c1 - c0
    itemsize = int(np.dtype(dtype).itemsize)
    out = np.empty((height, width, bands), dtype=np.float32)
    with open(img_path, "rb") as handle:
        if storage in ("SAMPLE_INTERLEAVED", "BIP"):
            line_stride = samples * bands * itemsize
            for i, row in enumerate(range(r0, r1)):
                handle.seek(row * line_stride + c0 * bands * itemsize)
                chunk = np.fromfile(handle, dtype=dtype, count=width * bands)
                out[i] = chunk.reshape(width, bands).astype(np.float32, copy=False)
            return out
        if storage in ("LINE_INTERLEAVED", "BIL"):
            line_stride = bands * samples * itemsize
            for i, row in enumerate(range(r0, r1)):
                for band in range(bands):
                    handle.seek(
                        row * line_stride + band * samples * itemsize + c0 * itemsize
                    )
                    chunk = np.fromfile(handle, dtype=dtype, count=width)
                    out[i, :, band] = chunk.astype(np.float32, copy=False)
            return out
        band_stride = lines * samples * itemsize
        row_stride = samples * itemsize
        for band in range(bands):
            handle.seek(band * band_stride + r0 * row_stride)
            plane = np.fromfile(handle, dtype=dtype, count=height * samples)
            out[:, :, band] = plane.reshape(height, samples)[:, c0:c1].astype(
                np.float32, copy=False
            )
    return out


def load_cube_window(
    path: str | Path,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    *,
    key: Optional[str] = None,
    data_layout: str = "HWB",
) -> np.ndarray:
    """Load one spatial window as Height×Width×Bands float32."""
    path = Path(path)
    height, width, bands = probe_cube_shape(
        path, key=key, data_layout=data_layout
    )
    r0, r1, c0, c1 = _clip_window(height, width, row0, row1, col0, col1)
    ext = _suffix(path)
    layout = str(data_layout or "HWB").upper()

    if should_load_cube_in_memory(height, width, bands):
        cube = load_cube(path, key=key, data_layout=data_layout)
        return np.ascontiguousarray(cube[r0:r1, c0:c1, :])

    if ext == ".npy":
        array = np.load(path, mmap_mode="r")
        if layout == "BHW":
            window = np.transpose(array[:, r0:r1, c0:c1], (1, 2, 0))
        else:
            window = array[r0:r1, c0:c1, :]
        return np.ascontiguousarray(window, dtype=np.float32)

    if ext in {".img", ".dat", ".hdr", ".lbl", ".bsq", ".bil", ".bip"}:
        open_path = resolve_open_path(path)
        hdr_sidecar = _sidecar(path, (".hdr",))
        lbl_sidecar = _sidecar(path, (".lbl",))
        use_pds = _suffix(open_path) == ".lbl" or (
            hdr_sidecar is None and lbl_sidecar is not None
        )
        if use_pds:
            try:
                return _read_pds_window(path, r0, r1, c0, c1)
            except Exception:
                pass
        header, raster = _envi_header_and_raster(path)
        if raster is None or _suffix(header) != ".hdr":
            raise ValueError(
                f"无法按窗口读取 {path}：需要 ENVI .hdr 或 PDS .lbl。"
                f"整幅约 {format_cube_memory(height, width, bands)}，"
                "请转换为 ENVI 后再训练。"
            )
        return _read_envi_window(header, raster, r0, r1, c0, c1)

    raise MemoryError(
        f"立方体 {path.name} 尺寸 {height}×{width}×{bands} "
        f"（约 {format_cube_memory(height, width, bands)}）无法整幅载入内存。"
        "请把数据转成 ENVI .hdr/.img（或 .npy）后重试；训练会按空间窗口读取。"
    )


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


_LABEL_NAME_SUFFIXES = (
    "_classification",
    "_labels",
    "_label",
    "_masks",
    "_mask",
    "_classes",
    "_class",
    "_targets",
    "_target",
    "_gts",
    "_gt",
    "_lbls",
    "_lbl",
    "_map",
    "-labels",
    "-label",
    "-gt",
    "-lbl",
    "-mask",
)

_MATCH_EXTRA_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _strip_known_extensions(name: str) -> str:
    text = name.lower()
    while True:
        suffix = Path(text).suffix
        if suffix in CUBE_EXTENSIONS or suffix in _MATCH_EXTRA_EXTS:
            text = text[: -len(suffix)]
            continue
        break
    return text


def canonical_match_key(path: str | Path) -> str:
    """Filename key for pairing cubes to labels (suffixes and zero-padding stripped)."""
    stem = _strip_known_extensions(Path(path).name)
    for suffix in _LABEL_NAME_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = re.split(r"(\d+)", stem)
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            out.append(str(int(part)))
        else:
            out.append(re.sub(r"[^a-z0-9]+", "", part.lower()))
    return "".join(out)


def filename_match_score(cube: str | Path, label: str | Path) -> float:
    """Higher is closer. 1.0 means the same key after stripping _label/_gt/etc."""
    cube_p, label_p = Path(cube), Path(label)
    raw_a = _strip_known_extensions(cube_p.name)
    raw_b = _strip_known_extensions(label_p.name)
    key_a = canonical_match_key(cube_p)
    key_b = canonical_match_key(label_p)
    if not key_a or not key_b:
        return SequenceMatcher(None, raw_a, raw_b).ratio()
    if key_a == key_b:
        return 1.0
    if raw_a == raw_b:
        return 0.99
    contain = 0.0
    if key_a in key_b or key_b in key_a:
        contain = 0.82 * (
            min(len(key_a), len(key_b)) / max(len(key_a), len(key_b))
        )
    seq = max(
        SequenceMatcher(None, key_a, key_b).ratio(),
        SequenceMatcher(None, raw_a, raw_b).ratio(),
    )
    return float(max(contain, seq))


def pair_files_by_closest_name(
    cubes: Sequence[str | Path],
    labels: Sequence[str | Path],
) -> List[Tuple[Path, Path, float]]:
    """One-to-one match: each cube gets the unused label with the closest name.

    Returns pairs in the same order as ``cubes``. Raises if there are fewer
    labels than cubes.
    """
    cubes_p = [Path(p) for p in cubes]
    labels_p = [Path(p) for p in labels]
    if not cubes_p:
        raise ValueError("没有训练数据文件可与标签配对。")
    if not labels_p:
        raise ValueError("标签文件夹里没有可用的标签文件。")
    if len(labels_p) < len(cubes_p):
        raise ValueError(
            f"标签文件数量 ({len(labels_p)}) 少于训练数据文件 ({len(cubes_p)})。"
            "每个立方体都需要一份对应标签。"
        )
    if len(cubes_p) == 1 and len(labels_p) == 1:
        score = filename_match_score(cubes_p[0], labels_p[0])
        return [(cubes_p[0], labels_p[0], score)]

    ranked: List[Tuple[float, int, int]] = []
    for i, cube in enumerate(cubes_p):
        for j, lab in enumerate(labels_p):
            ranked.append((filename_match_score(cube, lab), i, j))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

    assigned: dict = {}
    used_label = set()
    for score, i, j in ranked:
        if i in assigned or j in used_label:
            continue
        assigned[i] = (j, score)
        used_label.add(j)
        if len(assigned) == len(cubes_p):
            break

    unmatched = [cubes_p[i] for i in range(len(cubes_p)) if i not in assigned]
    if unmatched:
        leftover = [
            labels_p[j] for j in range(len(labels_p)) if j not in used_label
        ]
        detail = "\n".join(f"  {p.name}" for p in unmatched)
        extra = ""
        if leftover:
            extra = "\n未使用的标签：\n" + "\n".join(f"  {p.name}" for p in leftover)
        raise ValueError(
            "有训练文件没有配对到标签：\n"
            f"{detail}{extra}\n"
            "请让标签文件名与立方体文件名尽量接近。"
        )

    pairs: List[Tuple[Path, Path, float]] = []
    for i, cube in enumerate(cubes_p):
        j, score = assigned[i]
        pairs.append((cube, labels_p[j], float(score)))
    return pairs


def format_file_pairs(pairs: Sequence[Tuple[Path, Path, float]]) -> str:
    lines = [f"按文件名最近匹配，共 {len(pairs)} 对："]
    for cube, lab, score in pairs:
        flag = "  ⚠ 相似度偏低" if score < 0.5 else ""
        lines.append(f"  {cube.name}  ↔  {lab.name}  ({score:.3f}){flag}")
    return "\n".join(lines)


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
    "_mae_classification",
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
    elif tag.lower() == "mae":
        tag = "MAE"
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
