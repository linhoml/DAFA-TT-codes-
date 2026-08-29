"""
CRISM 240-band preprocessing.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import savemat
from scipy.ndimage import convolve, median_filter
from scipy.signal import savgol_filter

from .io import load_array, load_cube, list_input_files

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


RAW_BAND_NUM = 438
TARGET_BAND_NUM = 240

# Original CRISM 1-based bands 4--243.
BAND_START_0BASED = 3
BAND_END_0BASED = 242
BAND_SLICE = slice(BAND_START_0BASED, BAND_END_0BASED + 1)

DEFAULT_FILL_VALUE = 1e-4


def dump_process_model(path: str | Path, model: Dict) -> None:
    """Save a preprocess model with stdlib pickle (no joblib required)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_process_model(path: str | Path) -> Dict:
    """Load a preprocess model saved by pickle or (legacy) joblib."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"预处理模型不存在：{path}")
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            try:
                import joblib
            except ImportError as exc:
                raise RuntimeError(
                    f"无法读取预处理模型 {path}。"
                    "请确认文件是本软件保存的 preprocess_model.pkl。"
                ) from exc
            return joblib.load(f)


# =============================================================================
# I/O and layout
# =============================================================================

def read_mat_array(
    path: str | Path,
    key: Optional[str] = None,
    prefer_3d: bool = True,
) -> np.ndarray:
    """Read a cube/array from .mat, ENVI/IMG/DAT, NumPy, or TIFF."""
    return load_array(path, key=key or None, prefer_3d=prefer_3d)


def to_hwb(data: np.ndarray, data_layout: str = "HWB") -> np.ndarray:
    """Convert a cube to [height, width, bands]."""
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape={data.shape}")

    layout = data_layout.upper()
    if layout == "HWB":
        return data
    if layout == "BHW":
        return np.transpose(data, (1, 2, 0))

    raise ValueError("data_layout must be 'HWB' or 'BHW'")


def normalize_crism_band_count(
    data: np.ndarray,
    source_name: str = "",
    verbose: bool = False,
) -> np.ndarray:
    """
    Convert a CRISM cube to the common 240-band representation.

    - 438 bands: keep original 1-based bands 4--243.
    - 240 bands: keep unchanged.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected [H, W, B], got shape={data.shape}")

    band_num = data.shape[-1]
    prefix = f"{source_name}: " if source_name else ""

    if band_num == RAW_BAND_NUM:
        output = data[:, :, BAND_SLICE]
        if verbose:
            print(
                f"{prefix}{RAW_BAND_NUM} -> {TARGET_BAND_NUM} bands; "
                f"Python 0-based [{BAND_START_0BASED}, {BAND_END_0BASED}]"
            )
        return output

    if band_num == TARGET_BAND_NUM:
        if verbose:
            print(f"{prefix}{TARGET_BAND_NUM} bands; no slicing")
        return data

    raise ValueError(
        f"{prefix}unsupported B={band_num}; expected "
        f"{RAW_BAND_NUM} or {TARGET_BAND_NUM}"
    )


def list_input_mat_files(
    input_path: str | Path,
    input_pattern: str = "*",
) -> List[str]:
    """Accept one cube or a folder of .mat/.img/.dat/ENVI files."""
    return list_input_files(input_path, input_pattern)


def parse_band_list_1based(text: Optional[str]) -> List[int]:
    """
    Parse comma-separated 1-based positions in the 240-band cube.

    Example:
        "1,2,17,239,240"
    """
    if text is None or not str(text).strip():
        return []

    values: List[int] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue

        value = int(token)
        if not 1 <= value <= TARGET_BAND_NUM:
            raise ValueError(
                f"Band {value} is outside 1..{TARGET_BAND_NUM}"
            )
        values.append(value)

    return sorted(set(values))


# =============================================================================
# Candidate bad-band audit: report only
# =============================================================================

def compute_bad_band_statistics(
    input_path: str | Path,
    input_pattern: str = "*",
    data_key: Optional[str] = None,
    data_layout: str = "HWB",
    row_block: int = 128,
    zero_ratio_thr: float = 0.95,
    invalid_ratio_thr: float = 0.01,
) -> Dict[str, np.ndarray | float | int | list]:
    """
    Scan merged/training data and report candidate bad bands.

    A candidate band is stored for manual review only. No band is removed and
    no special bad-band interpolation is applied in preprocessing.
    """
    paths = list_input_mat_files(input_path, input_pattern)

    total_count = np.zeros(TARGET_BAND_NUM, dtype=np.int64)
    zero_count = np.zeros(TARGET_BAND_NUM, dtype=np.int64)
    nan_count = np.zeros(TARGET_BAND_NUM, dtype=np.int64)
    inf_count = np.zeros(TARGET_BAND_NUM, dtype=np.int64)
    gt1_count = np.zeros(TARGET_BAND_NUM, dtype=np.int64)

    for file_id, path in enumerate(tqdm(paths, desc="Auditing bad bands")):
        data = load_cube(path, key=data_key, data_layout=data_layout)
        data = normalize_crism_band_count(
            data,
            source_name=os.path.basename(path),
            verbose=(file_id == 0),
        )

        height = data.shape[0]

        for row_start in range(0, height, row_block):
            row_end = min(row_start + row_block, height)
            block = data[row_start:row_end]

            finite = np.isfinite(block)
            pixel_count = block.shape[0] * block.shape[1]

            total_count += pixel_count
            zero_count += np.sum(
                finite & (block == 0), axis=(0, 1)
            ).astype(np.int64)
            nan_count += np.sum(
                np.isnan(block), axis=(0, 1)
            ).astype(np.int64)
            inf_count += np.sum(
                np.isinf(block), axis=(0, 1)
            ).astype(np.int64)
            gt1_count += np.sum(
                finite & (block > 1), axis=(0, 1)
            ).astype(np.int64)

    total_safe = np.maximum(total_count, 1)

    zero_ratio = zero_count / total_safe
    nan_ratio = nan_count / total_safe
    inf_ratio = inf_count / total_safe
    gt1_ratio = gt1_count / total_safe
    invalid_ratio = (nan_count + inf_count + gt1_count) / total_safe

    candidate_bad_band = (
        (zero_ratio >= zero_ratio_thr)
        | (invalid_ratio >= invalid_ratio_thr)
    )

    candidate_0based = np.where(candidate_bad_band)[0]
    candidate_240_1based = candidate_0based + 1
    candidate_source_1based = (
        candidate_0based + BAND_START_0BASED + 1
    )

    print("\n========== Candidate bad-band report ==========")
    print(f"Total bands: {TARGET_BAND_NUM}")
    print(f"Candidate bad bands: {len(candidate_0based)}")
    print(
        "240-band positions, 1-based:",
        candidate_240_1based.tolist(),
    )
    print(
        "Original CRISM positions, 1-based:",
        candidate_source_1based.tolist(),
    )
    print("No candidate band is dropped or specially repaired.")

    return {
        "total_count": total_count,
        "zero_count": zero_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "gt1_count": gt1_count,
        "zero_ratio": zero_ratio.astype(np.float32),
        "nan_ratio": nan_ratio.astype(np.float32),
        "inf_ratio": inf_ratio.astype(np.float32),
        "gt1_ratio": gt1_ratio.astype(np.float32),
        "invalid_ratio": invalid_ratio.astype(np.float32),
        "candidate_bad_band_mask": candidate_bad_band,
        "candidate_bad_bands_240_1based": (
            candidate_240_1based.astype(int).tolist()
        ),
        "candidate_bad_bands_source_1based": (
            candidate_source_1based.astype(int).tolist()
        ),
        "zero_ratio_thr": float(zero_ratio_thr),
        "invalid_ratio_thr": float(invalid_ratio_thr),
    }


# =============================================================================
# Invalid values
# =============================================================================

def replace_invalid_values(
    data: np.ndarray,
    fill_value: float = DEFAULT_FILL_VALUE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace NaN, +/-Inf and values > 1 with fill_value.

    Zero and negative finite values are not modified by this rule.
    Returns:
        cleaned cube
        per-pixel original invalid ratio
    """
    output = data.astype(np.float32, copy=True)

    invalid = (
        ~np.isfinite(output)
        | (np.isfinite(output) & (output > 1))
    )

    invalid_ratio = np.mean(invalid, axis=-1).astype(np.float32)
    output[invalid] = np.float32(fill_value)

    return output, invalid_ratio


# =============================================================================
# Spectral despiking
# =============================================================================

def _repair_candidate_runs(
    spectrum: np.ndarray,
    candidate: np.ndarray,
    local_median: np.ndarray,
    max_spike_width: int,
) -> Tuple[np.ndarray, int]:
    """
    Repair candidate runs whose width is <= max_spike_width.
    """
    repaired = spectrum.copy()
    repaired_count = 0
    band_num = spectrum.size

    padded = np.concatenate(
        [np.array([False]), candidate, np.array([False])]
    )
    transitions = np.diff(padded.astype(np.int8))

    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1

    for start, end in zip(starts, ends):
        width = end - start + 1

        if width > max_spike_width:
            continue

        left = start - 1
        right = end + 1

        if left >= 0 and right < band_num:
            repaired[start : end + 1] = np.linspace(
                repaired[left],
                repaired[right],
                width + 2,
                dtype=np.float32,
            )[1:-1]
        else:
            repaired[start : end + 1] = local_median[
                start : end + 1
            ]

        repaired_count += width

    return repaired, repaired_count


def despike_spectra_batch(
    spectra: np.ndarray,
    median_window: int = 11,
    max_spike_width: int = 5,
    mad_k: float = 6.0,
    absolute_threshold: float = 0.01,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repair isolated spectral spikes in [N, B] spectra.
    """
    if spectra.ndim != 2:
        raise ValueError(
            f"Expected spectra [N, B], got shape={spectra.shape}"
        )
    if median_window < 3 or median_window % 2 == 0:
        raise ValueError(
            "median_window must be an odd integer >= 3"
        )
    if max_spike_width < 1:
        raise ValueError("max_spike_width must be >= 1")
    if median_window < 2 * max_spike_width + 1:
        print(
            "Warning: median_window is smaller than "
            "2 * max_spike_width + 1. Wide spikes may be missed."
        )

    output = spectra.astype(np.float32, copy=True)
    band_num = output.shape[1]

    if band_num < median_window:
        raise ValueError(
            f"Band number {band_num} is smaller than "
            f"median_window={median_window}"
        )

    local_median = median_filter(
        output,
        size=(1, median_window),
        mode="nearest",
    )

    residual = output - local_median

    # Robust center and scale for each spectrum.
    residual_center = np.median(
        residual, axis=1, keepdims=True
    )
    robust_mad = 1.4826 * np.median(
        np.abs(residual - residual_center),
        axis=1,
        keepdims=True,
    )
    robust_mad = np.maximum(robust_mad, eps)

    threshold = np.maximum(
        mad_k * robust_mad,
        absolute_threshold,
    )

    candidate = (
        np.abs(residual - residual_center) > threshold
    )

    repaired_count = np.zeros(output.shape[0], dtype=np.uint16)

    candidate_rows = np.where(np.any(candidate, axis=1))[0]

    for row in candidate_rows:
        output[row], repaired_count[row] = _repair_candidate_runs(
            spectrum=output[row],
            candidate=candidate[row],
            local_median=local_median[row],
            max_spike_width=max_spike_width,
        )

    return output, repaired_count


def despike_cube(
    data: np.ndarray,
    batch_pixels: int = 50000,
    median_window: int = 11,
    max_spike_width: int = 5,
    mad_k: float = 6.0,
    absolute_threshold: float = 0.01,
) -> Tuple[np.ndarray, int]:
    """Apply despiking to a [H, W, B] cube in pixel batches."""
    height, width, band_num = data.shape
    spectra = data.reshape(-1, band_num)

    output = np.empty_like(spectra, dtype=np.float32)
    total_repaired_values = 0

    for start in range(0, spectra.shape[0], batch_pixels):
        end = min(start + batch_pixels, spectra.shape[0])

        cleaned, repaired_count = despike_spectra_batch(
            spectra[start:end],
            median_window=median_window,
            max_spike_width=max_spike_width,
            mad_k=mad_k,
            absolute_threshold=absolute_threshold,
        )

        output[start:end] = cleaned
        total_repaired_values += int(np.sum(repaired_count))

    return (
        output.reshape(height, width, band_num),
        total_repaired_values,
    )


# =============================================================================
# Spatial bad-pixel detection and repair
# =============================================================================

def spectral_angle_map(
    data: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute per-pixel spectral angle in radians."""
    dot = np.sum(data * reference, axis=-1)
    norm_data = np.linalg.norm(data, axis=-1)
    norm_reference = np.linalg.norm(reference, axis=-1)

    denominator = np.maximum(
        norm_data * norm_reference,
        eps,
    )
    cosine = np.clip(dot / denominator, -1.0, 1.0)

    return np.arccos(cosine).astype(np.float32)


def detect_spatial_bad_pixels(
    data: np.ndarray,
    hard_bad_mask: np.ndarray,
    relative_deviation_threshold: float = 0.25,
    spectral_angle_threshold: float = 0.12,
    isolated_cluster_max: int = 2,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """
    Detect isolated spatial spectral outliers.
    """
    if hard_bad_mask.shape != data.shape[:2]:
        raise ValueError(
            "hard_bad_mask shape does not match cube spatial shape"
        )

    spatial_median = median_filter(
        data,
        size=(3, 3, 1),
        mode="nearest",
    )

    median_abs_difference = np.median(
        np.abs(data - spatial_median),
        axis=-1,
    )
    local_scale = np.median(
        np.abs(spatial_median),
        axis=-1,
    )
    relative_deviation = (
        median_abs_difference
        / np.maximum(local_scale, eps)
    )

    angle = spectral_angle_map(data, spatial_median)

    outlier = (
        (relative_deviation > relative_deviation_threshold)
        & (angle > spectral_angle_threshold)
        & (~hard_bad_mask)
    )

    if isolated_cluster_max > 0:
        neighborhood_count = convolve(
            outlier.astype(np.int16),
            np.ones((3, 3), dtype=np.int16),
            mode="constant",
            cval=0,
        )
        outlier &= neighborhood_count <= isolated_cluster_max

    bad_pixel_mask = hard_bad_mask | outlier
    return bad_pixel_mask, int(np.sum(outlier))


def repair_spatial_bad_pixels(
    data: np.ndarray,
    bad_pixel_mask: np.ndarray,
    min_valid_neighbors: int = 3,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Repair only detected bad pixels using the per-band median of valid 3x3
    neighbors.
    """
    if bad_pixel_mask.shape != data.shape[:2]:
        raise ValueError(
            "bad_pixel_mask shape does not match cube spatial shape"
        )

    output = data.astype(np.float32, copy=True)
    height, width, _ = output.shape

    valid_source = ~bad_pixel_mask

    padded_data = np.pad(
        output,
        ((1, 1), (1, 1), (0, 0)),
        mode="edge",
    )
    padded_valid = np.pad(
        valid_source,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=False,
    )

    repaired_count = 0
    unresolved_mask = np.zeros((height, width), dtype=bool)

    for row, col in np.argwhere(bad_pixel_mask):
        neighborhood = padded_data[
            row : row + 3,
            col : col + 3,
            :,
        ]
        valid_neighbors = padded_valid[
            row : row + 3,
            col : col + 3,
        ].copy()

        # Do not use the center itself.
        valid_neighbors[1, 1] = False

        if int(np.sum(valid_neighbors)) >= min_valid_neighbors:
            output[row, col, :] = np.median(
                neighborhood[valid_neighbors],
                axis=0,
            )
            repaired_count += 1
        else:
            unresolved_mask[row, col] = True

    return output, repaired_count, unresolved_mask


# =============================================================================
# SG smoothing and L2 normalization
# =============================================================================

def apply_savgol(
    data: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along the spectral axis."""
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError(
            "SG window_length must be an odd integer >= 3"
        )
    if polyorder >= window_length:
        raise ValueError(
            "SG polyorder must be smaller than window_length"
        )
    if data.shape[-1] < window_length:
        raise ValueError(
            f"Band number {data.shape[-1]} is smaller than "
            f"SG window {window_length}"
        )

    return savgol_filter(
        data,
        window_length=window_length,
        polyorder=polyorder,
        axis=-1,
        mode="interp",
    ).astype(np.float32)


def l2_normalize_cube(
    data: np.ndarray,
    unresolved_bad_mask: Optional[np.ndarray] = None,
    fill_value: float = DEFAULT_FILL_VALUE,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply per-pixel L2 normalization.
    """
    norms = np.linalg.norm(data, axis=-1)

    valid = np.isfinite(norms) & (norms > eps)
    if unresolved_bad_mask is not None:
        valid &= ~unresolved_bad_mask

    output = np.full_like(
        data,
        fill_value=np.float32(fill_value),
        dtype=np.float32,
    )
    output[valid] = data[valid] / norms[valid, None]

    return output


# =============================================================================
# Frozen preprocessing model
# =============================================================================

def build_preprocess_model(
    train_input_path: str | Path,
    model_save_path: str | Path,
    input_pattern: str = "*",
    data_key: Optional[str] = None,
    data_layout: str = "HWB",
    manual_exclude_bands_1based: Optional[Sequence[int]] = None,
    zero_ratio_thr: float = 0.95,
    invalid_ratio_thr: float = 0.01,
    invalid_fill_value: float = DEFAULT_FILL_VALUE,
    invalid_pixel_ratio_thr: float = 0.20,
    despike: bool = True,
    despike_median_window: int = 11,
    max_spike_width: int = 5,
    spike_mad_k: float = 6.0,
    spike_absolute_threshold: float = 0.01,
    spatial_repair: bool = True,
    spatial_relative_deviation_threshold: float = 0.25,
    spatial_spectral_angle_threshold: float = 0.12,
    spatial_isolated_cluster_max: int = 2,
    spatial_min_valid_neighbors: int = 3,
    smooth: bool = True,
    sg_window: int = 5,
    sg_polyorder: int = 2,
    l2_eps: float = 1e-8,
) -> Dict:
    """
    Audit merged/training data and save one frozen preprocessing model.
    """
    if despike_median_window % 2 == 0:
        raise ValueError("despike_median_window must be odd")
    if (
        despike_median_window
        < 2 * max_spike_width + 1
    ):
        print(
            "Warning: recommended despike_median_window is at least "
            "2 * max_spike_width + 1."
        )

    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    band_stats = compute_bad_band_statistics(
        input_path=train_input_path,
        input_pattern=input_pattern,
        data_key=data_key,
        data_layout=data_layout,
        zero_ratio_thr=zero_ratio_thr,
        invalid_ratio_thr=invalid_ratio_thr,
    )

    manual_exclude = sorted(
        set(manual_exclude_bands_1based or [])
    )
    for band in manual_exclude:
        if not 1 <= band <= TARGET_BAND_NUM:
            raise ValueError(
                f"Manual excluded band {band} is outside "
                f"1..{TARGET_BAND_NUM}"
            )

    use_band_mask = np.ones(
        TARGET_BAND_NUM,
        dtype=bool,
    )
    if manual_exclude:
        use_band_mask[
            np.asarray(manual_exclude, dtype=int) - 1
        ] = False

    model = {
        "version": "crism_preprocess_240_no_pca_v2",
        "raw_band_num": RAW_BAND_NUM,
        "target_band_num": TARGET_BAND_NUM,
        "source_band_start_1based": 4,
        "source_band_end_1based": 243,

        # Candidate bad-band report.
        "bad_band_policy": (
            "report only; no band is dropped or specially repaired"
        ),
        "bad_band_statistics": band_stats,
        "candidate_bad_bands_240_1based": (
            band_stats["candidate_bad_bands_240_1based"]
        ),
        "candidate_bad_bands_source_1based": (
            band_stats["candidate_bad_bands_source_1based"]
        ),

        # Manual hyperparameter for the downstream training loader.
        # It does not change the saved 240-band preprocessing result.
        "manual_exclude_bands_240_1based": manual_exclude,
        "use_band_mask": use_band_mask,

        # Invalid values.
        "invalid_rule": "NaN, +/-Inf and values > 1 -> fill value",
        "invalid_fill_value": float(invalid_fill_value),
        "invalid_pixel_ratio_thr": float(
            invalid_pixel_ratio_thr
        ),

        # Despiking.
        "despike": bool(despike),
        "despike_median_window": int(
            despike_median_window
        ),
        "max_spike_width": int(max_spike_width),
        "spike_mad_k": float(spike_mad_k),
        "spike_absolute_threshold": float(
            spike_absolute_threshold
        ),
        "spike_repair_method": (
            "linear interpolation for interior runs; "
            "local median at spectral edges"
        ),

        # Spatial bad pixels.
        "spatial_repair": bool(spatial_repair),
        "spatial_relative_deviation_threshold": float(
            spatial_relative_deviation_threshold
        ),
        "spatial_spectral_angle_threshold": float(
            spatial_spectral_angle_threshold
        ),
        "spatial_isolated_cluster_max": int(
            spatial_isolated_cluster_max
        ),
        "spatial_min_valid_neighbors": int(
            spatial_min_valid_neighbors
        ),
        "spatial_repair_method": (
            "per-band median of valid 3x3 neighbors"
        ),

        # Smoothing and normalization.
        "smooth": bool(smooth),
        "sg_window": int(sg_window),
        "sg_polyorder": int(sg_polyorder),
        "spectral_normalization": "per-pixel L2",
        "l2_eps": float(l2_eps),

        "output_band_num": TARGET_BAND_NUM,
        "output_mat_user_keys": ["data"],
        "process_order": [
            "read_mat",
            "to_HWB",
            "438_to_240_or_keep_240",
            "replace_NaN_Inf_gt1_with_fill_value",
            "despike_local_median_MAD_width_constraint",
            "selective_spatial_bad_pixel_repair",
            "Savitzky_Golay",
            "per_pixel_L2",
        ],
    }

    dump_process_model(model_path, model)

    print(f"Saved preprocess model: {model_path}")
    print(
        "Candidate bad bands, 240-band 1-based:",
        model["candidate_bad_bands_240_1based"],
    )
    print(
        "Manual excluded bands, 240-band 1-based:",
        model["manual_exclude_bands_240_1based"],
    )
    print(
        "Output cubes remain 240 bands; manual exclusion is "
        "only stored as use_band_mask in the model."
    )

    return model


# =============================================================================
# Apply preprocessing
# =============================================================================

def preprocess_cube(
    data: np.ndarray,
    model: Dict,
    source_name: str = "",
    batch_pixels: int = 50000,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Apply the frozen model to one CRISM cube."""
    cube = normalize_crism_band_count(
        data,
        source_name=source_name,
        verbose=False,
    ).astype(np.float32, copy=False)

    if cube.shape[-1] != TARGET_BAND_NUM:
        raise RuntimeError(
            f"Expected {TARGET_BAND_NUM} bands, "
            f"got {cube.shape[-1]}"
        )

    fill_value = float(
        model.get("invalid_fill_value", DEFAULT_FILL_VALUE)
    )

    # 1. Explicit invalid-value repair.
    cube, original_invalid_ratio = replace_invalid_values(
        cube,
        fill_value=fill_value,
    )

    hard_bad_mask = (
        original_invalid_ratio
        >= float(
            model.get("invalid_pixel_ratio_thr", 0.20)
        )
    )

    # 2. Spectral despiking.
    repaired_spike_values = 0
    if bool(model.get("despike", True)):
        cube, repaired_spike_values = despike_cube(
            cube,
            batch_pixels=batch_pixels,
            median_window=int(
                model.get("despike_median_window", 11)
            ),
            max_spike_width=int(
                model.get("max_spike_width", 5)
            ),
            mad_k=float(
                model.get("spike_mad_k", 6.0)
            ),
            absolute_threshold=float(
                model.get(
                    "spike_absolute_threshold",
                    0.01,
                )
            ),
        )

    # 3. Selective spatial bad-pixel repair.
    detected_spatial_outliers = 0
    repaired_spatial_pixels = 0

    if bool(model.get("spatial_repair", True)):
        bad_pixel_mask, detected_spatial_outliers = (
            detect_spatial_bad_pixels(
                cube,
                hard_bad_mask=hard_bad_mask,
                relative_deviation_threshold=float(
                    model.get(
                        "spatial_relative_deviation_threshold",
                        0.25,
                    )
                ),
                spectral_angle_threshold=float(
                    model.get(
                        "spatial_spectral_angle_threshold",
                        0.12,
                    )
                ),
                isolated_cluster_max=int(
                    model.get(
                        "spatial_isolated_cluster_max",
                        2,
                    )
                ),
            )
        )

        (
            cube,
            repaired_spatial_pixels,
            unresolved_bad_mask,
        ) = repair_spatial_bad_pixels(
            cube,
            bad_pixel_mask=bad_pixel_mask,
            min_valid_neighbors=int(
                model.get(
                    "spatial_min_valid_neighbors",
                    3,
                )
            ),
        )
    else:
        unresolved_bad_mask = hard_bad_mask.copy()

    # 4. SG smoothing.
    if bool(model.get("smooth", True)):
        cube = apply_savgol(
            cube,
            window_length=int(
                model.get("sg_window", 5)
            ),
            polyorder=int(
                model.get("sg_polyorder", 2)
            ),
        )

    # Recheck because SG can create non-finite values or slight overshoot.
    cube, _ = replace_invalid_values(
        cube,
        fill_value=fill_value,
    )

    # 5. Per-pixel L2 normalization.
    cube = l2_normalize_cube(
        cube,
        unresolved_bad_mask=unresolved_bad_mask,
        fill_value=fill_value,
        eps=float(model.get("l2_eps", 1e-8)),
    )

    if cube.shape[-1] != TARGET_BAND_NUM:
        raise RuntimeError(
            f"Output must have {TARGET_BAND_NUM} bands, "
            f"got {cube.shape[-1]}"
        )

    summary = {
        "original_hard_bad_pixels": int(
            np.sum(hard_bad_mask)
        ),
        "repaired_spike_values": int(
            repaired_spike_values
        ),
        "detected_spatial_outliers": int(
            detected_spatial_outliers
        ),
        "repaired_spatial_pixels": int(
            repaired_spatial_pixels
        ),
        "unresolved_spatial_pixels": int(
            np.sum(unresolved_bad_mask)
        ),
    }

    return cube.astype(np.float32), summary


def make_output_name(
    input_path: str,
    output_prefix: str = "preprocessed",
) -> str:
    """Build a deterministic output filename."""
    stem = Path(input_path).stem

    if stem.startswith("image_tile_"):
        suffix = stem[len("image_tile_") :]
        return f"{output_prefix}_{suffix}.mat"

    if stem.startswith(f"{output_prefix}_"):
        return f"{stem}.mat"

    return f"{output_prefix}_{stem}.mat"


def transform_inputs(
    input_path: str | Path,
    save_dir: str | Path,
    model_path: str | Path,
    input_pattern: str = "*",
    data_key: Optional[str] = None,
    data_layout: str = "HWB",
    output_prefix: str = "preprocessed",
    batch_pixels: int = 50000,
) -> None:
    """
    Preprocess one MAT file or a folder using a frozen model.

    Every output MAT contains only:
        data
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_process_model(model_path)
    paths = list_input_mat_files(
        input_path,
        input_pattern,
    )

    print(f"Loaded preprocess model: {model_path}")
    print("PCA: disabled")
    print("Output bands:", model["output_band_num"])
    print(
        "Despike:",
        f"median_window={model['despike_median_window']},",
        f"max_spike_width={model['max_spike_width']},",
        f"MAD k={model['spike_mad_k']},",
        f"absolute_threshold={model['spike_absolute_threshold']}",
    )
    print(
        "Candidate bad bands, 240-band 1-based:",
        model["candidate_bad_bands_240_1based"],
    )
    print("Output MAT user key: data only")

    for path in tqdm(paths, desc="Preprocessing cubes"):
        raw = load_cube(
            path,
            key=data_key,
            data_layout=data_layout,
        )

        processed, summary = preprocess_cube(
            raw,
            model=model,
            source_name=os.path.basename(path),
            batch_pixels=batch_pixels,
        )

        output_name = make_output_name(
            path,
            output_prefix=output_prefix,
        )
        save_path = output_dir / output_name

        # Only one user variable is written to each MAT file.
        savemat(
            save_path,
            {"data": processed},
            do_compression=True,
        )

        print(
            f"Saved: {save_path} | shape={processed.shape} | "
            f"spike values repaired={summary['repaired_spike_values']} | "
            f"spatial pixels repaired={summary['repaired_spatial_pixels']} | "
            f"unresolved={summary['unresolved_spatial_pixels']}"
        )


# =============================================================================
# Command line
# =============================================================================

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "CRISM 240-band preprocessing without PCA"
    )

    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test", "transform"],
        help=(
            "train: audit candidate bad bands on merged/training data, "
            "save model, then preprocess the same input. "
            "test/transform: load the frozen model without refitting."
        ),
    )
    parser.add_argument(
        "--input_path",
        default="/home/xj/code/mineralHSI/data/CRISM4/应用CRISM/hrl000040ff_07_if183l_trr3_CAT_corr.mat",
        help="One MAT file or a folder containing MAT files.",
    )
    parser.add_argument(
        "--save_dir",
        default="/home/xj/code/mineralHSI/data/CRISM4/应用CRISM/preprocessed"
    )
    parser.add_argument(
        "--model_path",
        default="data/CRISM4/caijian/preprocessed/preprocess_model.pkl"
    )
    parser.add_argument(
        "--input_pattern",
        default="*",
        help="Glob when input_path is a directory. Default * matches .mat/.img/.dat/.hdr/…",
    )
    parser.add_argument(
        "--data_key",
        default="data",
        help=(
            "MAT variable name. If omitted, the first 3D ndarray "
            "is selected."
        ),
    )
    parser.add_argument(
        "--data_layout",
        default="HWB",
        choices=["HWB", "BHW"],
    )
    parser.add_argument(
        "--output_prefix",
        default="",
    )
    parser.add_argument(
        "--batch_pixels",
        type=int,
        default=50000,
    )

    # Candidate bad-band audit.
    parser.add_argument(
        "--zero_ratio_thr",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--invalid_ratio_thr",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--manual_exclude_bands",
        default="",
        help=(
            "Comma-separated 1-based positions in the 240-band cube. "
            "Stored only in model['use_band_mask']; output remains "
            "240 bands."
        ),
    )

    # Invalid values.
    parser.add_argument(
        "--invalid_fill_value",
        type=float,
        default=DEFAULT_FILL_VALUE,
    )
    parser.add_argument(
        "--invalid_pixel_ratio_thr",
        type=float,
        default=0.20,
        help=(
            "Pixels whose original NaN/Inf/>1 ratio reaches this "
            "threshold are hard spatial bad-pixel candidates."
        ),
    )

    # Spectral despiking.
    parser.add_argument(
        "--despike",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--despike_median_window",
        type=int,
        default=9,
        help=(
            "Odd local-median detection window. "
            "Default 11 supports spikes up to about 5 bands."
        ),
    )
    parser.add_argument(
        "--max_spike_width",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--spike_mad_k",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--spike_absolute_threshold",
        type=float,
        default=0.01,
    )

    # Spatial repair.
    parser.add_argument(
        "--spatial_repair",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--spatial_relative_deviation_threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--spatial_spectral_angle_threshold",
        type=float,
        default=0.12,
        help="Radians.",
    )
    parser.add_argument(
        "--spatial_isolated_cluster_max",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--spatial_min_valid_neighbors",
        type=int,
        default=3,
    )

    # SG and L2.
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sg_window",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sg_polyorder",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--l2_eps",
        type=float,
        default=1e-8,
    )

    return parser


def main() -> None:
    args = setup_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "train":
        manual_exclude = parse_band_list_1based(
            args.manual_exclude_bands
        )

        build_preprocess_model(
            train_input_path=args.input_path,
            model_save_path=args.model_path,
            input_pattern=args.input_pattern,
            data_key=args.data_key,
            data_layout=args.data_layout,
            manual_exclude_bands_1based=manual_exclude,
            zero_ratio_thr=args.zero_ratio_thr,
            invalid_ratio_thr=args.invalid_ratio_thr,
            invalid_fill_value=args.invalid_fill_value,
            invalid_pixel_ratio_thr=(
                args.invalid_pixel_ratio_thr
            ),
            despike=bool(args.despike),
            despike_median_window=(
                args.despike_median_window
            ),
            max_spike_width=args.max_spike_width,
            spike_mad_k=args.spike_mad_k,
            spike_absolute_threshold=(
                args.spike_absolute_threshold
            ),
            spatial_repair=bool(args.spatial_repair),
            spatial_relative_deviation_threshold=(
                args.spatial_relative_deviation_threshold
            ),
            spatial_spectral_angle_threshold=(
                args.spatial_spectral_angle_threshold
            ),
            spatial_isolated_cluster_max=(
                args.spatial_isolated_cluster_max
            ),
            spatial_min_valid_neighbors=(
                args.spatial_min_valid_neighbors
            ),
            smooth=bool(args.smooth),
            sg_window=args.sg_window,
            sg_polyorder=args.sg_polyorder,
            l2_eps=args.l2_eps,
        )

        # Process merged/training data using the model just saved.
        transform_inputs(
            input_path=args.input_path,
            save_dir=args.save_dir,
            model_path=args.model_path,
            input_pattern=args.input_pattern,
            data_key=args.data_key,
            data_layout=args.data_layout,
            output_prefix=args.output_prefix,
            batch_pixels=args.batch_pixels,
        )
        return

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Preprocess model not found: {args.model_path}"
        )

    # External CRISM/test scenes use exactly the same frozen model.
    transform_inputs(
        input_path=args.input_path,
        save_dir=args.save_dir,
        model_path=args.model_path,
        input_pattern=args.input_pattern,
        data_key=args.data_key,
        data_layout=args.data_layout,
        output_prefix=args.output_prefix,
        batch_pixels=args.batch_pixels,
    )


if __name__ == "__main__":
    main()
