"""Linear and sparse spectral unmixing solvers (NumPy + SciPy)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import nnls


def _valid_mask(y: np.ndarray, A: np.ndarray) -> np.ndarray:
    m = np.isfinite(y)
    m &= np.all(np.isfinite(A), axis=1)
    # drop near-zero / nodata in observation
    m &= np.abs(y) > 1e-12
    return m


def unmix_spectrum(
    y: np.ndarray,
    A: np.ndarray,
    method: str = "nnls",
    sparsity: int = 3,
    sum_to_one: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Unmix one spectrum.

    Parameters
    ----------
    y : (n_bands,)
        Observed spectrum.
    A : (n_bands, n_endmembers)
        Endmember matrix (columns = endmembers).
    method : 'nnls' | 'ucls' | 'fcls' | 'omp'
        - nnls: non-negative least squares
        - ucls: unconstrained least squares (pseudoinverse)
        - fcls: fully constrained (ANC + ASC) via iterative projection
        - omp: orthogonal matching pursuit (sparse non-negative)
    sparsity : max endmembers for OMP
    sum_to_one : if True and method is nnls/omp, renormalize abundances to sum 1
                 after solve (soft ASC). For hard ASC use method='fcls'.
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or y.size != A.shape[0]:
        raise ValueError(f"维度不匹配：y={y.shape}, A={A.shape}")

    mask = _valid_mask(y, A)
    if mask.sum() < max(3, A.shape[1]):
        n = A.shape[1]
        return {
            "abundance": np.full(n, np.nan),
            "reconstructed": np.full_like(y, np.nan),
            "residual": np.full_like(y, np.nan),
            "rmse": np.array(np.nan),
            "support": np.array([], dtype=int),
            "method": method,
        }

    ym = y[mask]
    Am = A[mask, :]
    method = (method or "nnls").lower()

    if method == "ucls":
        x, *_ = np.linalg.lstsq(Am, ym, rcond=None)
        support = np.arange(A.shape[1])
    elif method == "fcls":
        x = _fcls(Am, ym)
        support = np.where(x > 1e-8)[0]
    elif method == "omp":
        x, support = _omp_nn(Am, ym, sparsity=max(1, int(sparsity)))
    else:  # nnls
        x, _ = nnls(Am, ym)
        support = np.where(x > 1e-8)[0]

    if sum_to_one and method != "fcls":
        s = float(np.sum(np.maximum(x, 0.0)))
        if s > 1e-12:
            x = np.maximum(x, 0.0) / s

    recon = A @ x
    resid = y - recon
    rmse = float(np.sqrt(np.nanmean((ym - (Am @ x)) ** 2)))
    return {
        "abundance": np.asarray(x, dtype=float),
        "reconstructed": recon,
        "residual": resid,
        "rmse": np.array(rmse),
        "support": np.asarray(support, dtype=int),
        "method": method,
    }


def _fcls(A: np.ndarray, y: np.ndarray, max_iter: int = 500, tol: float = 1e-9) -> np.ndarray:
    """
    Fully constrained least squares (non-negative + sum-to-one).
    Active-set style: NNLS on augmented system with sum constraint via
    iterative renormalization + NNLS (Heinz & Chang style practical variant).
    """
    n = A.shape[1]
    # Augment for sum-to-one with a weight
    delta = 1.0 / (np.linalg.norm(A, ord="fro") / max(A.shape[0], 1) + 1e-12)
    A_aug = np.vstack([A, delta * np.ones((1, n))])
    y_aug = np.append(y, delta)
    x, _ = nnls(A_aug, y_aug)
    # Project onto simplex if needed
    x = _project_simplex(np.maximum(x, 0.0))
    # One refinement
    for _ in range(3):
        A_aug = np.vstack([A, delta * np.ones((1, n))])
        y_aug = np.append(y, delta)
        x, _ = nnls(A_aug, y_aug)
        x = _project_simplex(np.maximum(x, 0.0))
        if abs(x.sum() - 1.0) < tol:
            break
    return x


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto probability simplex."""
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    ind = np.arange(1, n + 1)
    cond = u - (cssv - 1) / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n)
    rho = int(ind[cond][-1])
    theta = (cssv[rho - 1] - 1.0) / rho
    return np.maximum(v - theta, 0.0)


def _omp_nn(A: np.ndarray, y: np.ndarray, sparsity: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Non-negative OMP: greedily add the endmember with largest positive correlation."""
    n = A.shape[1]
    x = np.zeros(n, dtype=float)
    residual = y.copy()
    support = []
    unused = list(range(n))
    # normalize columns for correlation
    norms = np.linalg.norm(A, axis=0) + 1e-12
    for _ in range(min(sparsity, n)):
        corr = (A.T @ residual) / norms
        # only consider unused and positive correlation
        best_i = None
        best_c = 0.0
        for i in unused:
            if corr[i] > best_c:
                best_c = float(corr[i])
                best_i = i
        if best_i is None or best_c <= 0:
            break
        support.append(best_i)
        unused.remove(best_i)
        As = A[:, support]
        coef, _ = nnls(As, y)
        x_full = np.zeros(n)
        for k, idx in enumerate(support):
            x_full[idx] = coef[k]
        x = x_full
        residual = y - A @ x
    return x, np.asarray(support, dtype=int)


def unmix_cube(
    cube: np.ndarray,
    A: np.ndarray,
    method: str = "nnls",
    sparsity: int = 3,
    sum_to_one: bool = False,
    spatial_stride: int = 1,
    progress_cb=None,
) -> Dict[str, np.ndarray]:
    """
    Unmix an image cube.

    Parameters
    ----------
    cube : (rows, cols, bands)
    A : (bands, n_endmembers)
    spatial_stride : compute every N pixels (rest left NaN)

    Returns
    -------
    dict with abundance (rows, cols, n_em), rmse (rows, cols), reconstructed optional not stored
    """
    cube = np.asarray(cube, dtype=float)
    A = np.asarray(A, dtype=float)
    rows, cols, bands = cube.shape
    if A.shape[0] != bands:
        raise ValueError(f"波段数不匹配：cube bands={bands}, A={A.shape}")
    n_em = A.shape[1]
    abund = np.full((rows, cols, n_em), np.nan, dtype=float)
    rmse = np.full((rows, cols), np.nan, dtype=float)

    step = max(1, int(spatial_stride))
    coords = [(r, c) for r in range(0, rows, step) for c in range(0, cols, step)]
    total = len(coords)
    for i, (r, c) in enumerate(coords):
        y = cube[r, c, :]
        if not np.any(np.isfinite(y)):
            continue
        # skip if mostly nodata
        if np.nanmean(np.isfinite(y)) < 0.3:
            continue
        res = unmix_spectrum(y, A, method=method, sparsity=sparsity, sum_to_one=sum_to_one)
        abund[r, c, :] = res["abundance"]
        rmse[r, c] = float(res["rmse"])
        if progress_cb is not None and (i % 50 == 0 or i + 1 == total):
            progress_cb(i + 1, total)

    return {"abundance": abund, "rmse": rmse, "method": method, "stride": step}
