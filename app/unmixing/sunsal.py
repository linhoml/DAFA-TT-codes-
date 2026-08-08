"""
SUNSAL: sparse unmixing via variable splitting and ADMM.

Python port of Bioucas-Dias & Figueiredo sunsal.m + soft.m
(soft_cecc.m / sunsal_9038.m).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np


def soft(x: np.ndarray, T: Union[float, np.ndarray]) -> np.ndarray:
    """Soft-thresholding operator (MATLAB soft.m)."""
    T = np.asarray(T, dtype=float) + np.finfo(float).eps
    y = np.maximum(np.abs(x) - T, 0.0)
    return (y / (y + T)) * x


def sunsal(
    M: np.ndarray,
    y: np.ndarray,
    lambda_: Union[float, np.ndarray] = 0.0,
    positivity: bool = False,
    addone: bool = False,
    al_iters: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
    x0: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Sparse unmixing via ADMM (SUNSAL).

    Solves:
        min_X  (1/2) ||M X - y||_F^2 + lambda ||X||_1
    with optional X >= 0 and/or 1^T X = 1.

    Parameters
    ----------
    M : (L, p) mixing matrix (endmembers as columns)
    y : (L,) or (L, N) observations
    lambda_ : scalar or length-N regularization
    positivity : enforce X >= 0
    addone : enforce sum-to-one per pixel
    al_iters : max ADMM iterations
    tol : primal/dual residual tolerance

    Returns
    -------
    dict with abundance (p,) or (p, N), res_p, res_d, n_iter
    """
    M = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    LM, p = M.shape
    L, N = y.shape
    if LM != L:
        raise ValueError(f"M ({LM}×{p}) and y ({L}×{N}) are inconsistent")

    al_iters = max(1, int(al_iters))
    lambda_arr = np.asarray(lambda_, dtype=float)
    if lambda_arr.ndim == 0 or lambda_arr.size == 1:
        lambda_mat = np.full((p, N), float(lambda_arr), dtype=float)
    elif lambda_arr.size == N:
        lambda_mat = np.tile(lambda_arr.ravel()[None, :], (p, 1))
    elif lambda_arr.shape == (p, N):
        lambda_mat = lambda_arr.astype(float)
    else:
        raise ValueError("lambda size is inconsistent with the data set")

    if np.any(lambda_mat < 0):
        raise ValueError("lambda must be non-negative")

    # Rescale like MATLAB
    norm_y = float(np.sqrt(np.mean(y ** 2)))
    if not np.isfinite(norm_y) or norm_y <= 0:
        norm_y = 1.0
    M = M / norm_y
    y = y / norm_y
    lambda_mat = lambda_mat / (norm_y ** 2)

    # Plain least squares
    if (not np.any(lambda_mat)) and (not positivity) and (not addone):
        z = np.linalg.pinv(M) @ y
        out = z[:, 0] if squeeze else z
        return {
            "abundance": out,
            "res_p": np.array(0.0),
            "res_d": np.array(0.0),
            "n_iter": np.array(0),
            "method": "sunsal_ls",
        }

    SMALL = 1e-12
    B = np.ones((1, p))
    a = np.ones((1, N))

    # Sum-to-one LS without positivity
    if addone and (not positivity) and (not np.any(lambda_mat)):
        F = M.T @ M
        if np.linalg.rcond(F) > SMALL:
            IF = np.linalg.inv(F)
            z = IF @ M.T @ y - IF @ B.T @ np.linalg.inv(B @ IF @ B.T) @ (B @ IF @ M.T @ y - a)
            out = z[:, 0] if squeeze else z
            return {
                "abundance": out,
                "res_p": np.array(0.0),
                "res_d": np.array(0.0),
                "n_iter": np.array(0),
                "method": "sunsal_sumls",
            }

    mu_AL = 0.01
    mu = 10.0 * float(np.mean(lambda_mat)) + mu_AL

    # Symmetric eigen-decomposition (equivalent to MATLAB svd on M'*M)
    sF, UF = np.linalg.eigh(M.T @ M)
    IF = UF @ np.diag(1.0 / (sF + mu)) @ UF.T
    Aux = IF @ B.T @ np.linalg.inv(B @ IF @ B.T)
    x_aux = Aux @ a
    IF1 = IF - Aux @ B @ IF
    yy = M.T @ y

    if x0 is None:
        x = IF @ yy
    else:
        x = np.asarray(x0, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape != (p, N):
            raise ValueError("initial X is inconsistent with M or Y")

    z = x.copy()
    d = np.zeros_like(z)

    tol1 = np.sqrt(N * p) * tol
    tol2 = np.sqrt(N * p) * tol
    res_p = np.inf
    res_d = np.inf
    i = 1
    mu_changed = False
    z0 = z.copy()

    while (i <= al_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
        if i % 10 == 1:
            z0 = z.copy()

        z = soft(x - d, lambda_mat / mu)
        if positivity:
            z = np.maximum(z, 0.0)

        if addone:
            x = IF1 @ (yy + mu * (z + d)) + x_aux
        else:
            x = IF @ (yy + mu * (z + d))

        d = d - (x - z)

        if i % 10 == 1:
            res_p = float(np.linalg.norm(x - z, "fro"))
            res_d = float(mu * np.linalg.norm(z - z0, "fro"))
            if verbose:
                print(f" i = {i}, res_p = {res_p}, res_d = {res_d}")
            if res_p > 10 * res_d:
                mu *= 2.0
                d /= 2.0
                mu_changed = True
            elif res_d > 10 * res_p:
                mu /= 2.0
                d *= 2.0
                mu_changed = True
            if mu_changed:
                IF = UF @ np.diag(1.0 / (sF + mu)) @ UF.T
                Aux = IF @ B.T @ np.linalg.inv(B @ IF @ B.T)
                x_aux = Aux @ a
                IF1 = IF - Aux @ B @ IF
                mu_changed = False
        i += 1

    out = z[:, 0] if squeeze else z
    return {
        "abundance": np.asarray(out, dtype=float),
        "res_p": np.array(res_p if np.isfinite(res_p) else 0.0),
        "res_d": np.array(res_d if np.isfinite(res_d) else 0.0),
        "n_iter": np.array(i - 1),
        "method": "sunsal",
    }
