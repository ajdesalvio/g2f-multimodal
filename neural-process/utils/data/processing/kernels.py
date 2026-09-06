"""Shared kernel computation, centering, and eigendecomposition utilities.

Shared by the kernel-producing processors (genomic, phenomic,
enviromic, weather_kernel, interaction) to avoid duplicating the
kernel/centering/eigendecomposition math.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ── Kernel computation ───────────────────────────────────────────


def compute_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    kernel: str = "linear",
    gamma: float | None = None,
) -> np.ndarray:
    """Compute kernel matrix K(X, Y).

    Parameters
    ----------
    X : array of shape (n_x, p)
    Y : array of shape (n_y, p), optional.  If None, computes K(X, X).
    kernel : ``"linear"`` or ``"rbf"``.
    gamma : RBF bandwidth.  Defaults to ``1 / p`` when None.

    Returns
    -------
    K : array of shape (n_x, n_y).
        For ``"linear"``: ``K = X @ Y.T / p`` (scaled dot product).
        For ``"rbf"``: ``K[i,j] = exp(-gamma * ||x_i - y_j||^2)``.
    """
    if Y is None:
        Y = X

    if X.shape[0] == 0:
        return np.empty((0, Y.shape[0]))

    p = X.shape[1]

    if kernel == "linear":
        return X @ Y.T / p
    elif kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / p
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-gamma * sq_dists)
    else:
        raise ValueError(
            f"Unknown kernel: {kernel!r}. Expected 'linear' or 'rbf'."
        )


# ── Double-centering ─────────────────────────────────────────────


def center_kernel_train(
    K_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Double-center a training kernel matrix.

    Parameters
    ----------
    K_train : array of shape (n, n)

    Returns
    -------
    K_centered : array of shape (n, n)
    col_means : array of shape (n,)
    grand_mean : float
    """
    col_means = K_train.mean(axis=0)
    grand_mean = float(K_train.mean())
    K_centered = (
        K_train
        - col_means[np.newaxis, :]
        - col_means[:, np.newaxis]
        + grand_mean
    )
    return K_centered, col_means, grand_mean


def center_kernel_cross(
    K_cross: np.ndarray,
    train_col_means: np.ndarray,
    train_grand_mean: float,
) -> np.ndarray:
    """Center a cross-kernel K(X_new, X_train) using training statistics.

    Parameters
    ----------
    K_cross : array of shape (n_new, n_train)
    train_col_means : array of shape (n_train,) from ``center_kernel_train``.
    train_grand_mean : float from ``center_kernel_train``.

    Returns
    -------
    K_cross_centered : array of shape (n_new, n_train)
    """
    row_means = K_cross.mean(axis=1, keepdims=True)
    return (
        K_cross
        - train_col_means[np.newaxis, :]
        - row_means
        + train_grand_mean
    )


# ── Eigendecomposition ───────────────────────────────────────────


def eigen_decompose(
    K_centered: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a centered kernel matrix.

    Returns the **full** decomposition (all positive eigenvalues
    above the relative threshold). ``n_components`` is applied as a
    post-load slice by callers — this function never truncates by
    component count.

    Parameters
    ----------
    K_centered : array of shape (n, n), symmetric positive semi-definite.
    eps : eigenvalues below ``eps * max_eigenvalue`` are discarded as
        numerical noise. Set ``eps=0`` to keep every strictly-positive
        mode — used by BGLR-bound processors under ``fit_scope='all'``
        so the eigenbasis matches the R reference pipeline, which applies
        no upstream filter and relies on BGLR's internal ``tolD=1e-10``
        absolute filter (finding F).

    Returns
    -------
    V : array of shape (n, m) — eigenvectors in descending eigenvalue order.
        ``m <= n`` after filtering near-zero eigenvalues.
    D : array of shape (m,) — positive eigenvalues in descending order.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

    # eigh returns ascending order; reverse to descending
    eigenvalues = eigenvalues[::-1].copy()
    eigenvectors = eigenvectors[:, ::-1].copy()

    # Discard non-positive eigenvalues. PSD matrices can have small
    # negatives from numerical noise; these are also what BGLR's
    # internal RKHS handler drops via tolD.
    pos_mask = eigenvalues > 0
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]

    # Optional relative threshold for cross-projection numerical
    # stability (project_eigen_cross divides by sqrt(d)). Skipped
    # when eps=0.
    if eps > 0 and len(eigenvalues) > 0:
        threshold = eps * eigenvalues[0]
        keep = eigenvalues >= threshold
        eigenvalues = eigenvalues[keep]
        eigenvectors = eigenvectors[:, keep]

    return eigenvectors, eigenvalues


def slice_components(
    V: np.ndarray,
    D: np.ndarray,
    n_components: int | None = None,
    n_components_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice eigenvectors/eigenvalues per the D1 two-knob contract.

    - Exactly one of ``n_components`` / ``n_components_max`` may be set.
    - ``n_components=K`` — hard fix. Uses exactly K components. Raises
      ``ValueError`` if ``K > len(D)``. This is the mode where "the
      code fails, it's the user's fault" — the user asked for an exact
      width and the data can't deliver it.
    - ``n_components_max=K`` — soft ceiling. Uses ``min(K, len(D))``
      components. Never raises on the ceiling; adapts to available
      rank.
    - both ``None`` — returns the full ``(V, D)``.
    - both set — ``ValueError`` (config error).
    """
    if n_components is not None and n_components_max is not None:
        raise ValueError(
            "n_components and n_components_max are mutually exclusive; "
            "set at most one."
        )
    if n_components is None and n_components_max is None:
        return V, D
    if n_components is not None:
        if n_components > len(D):
            raise ValueError(
                f"Requested n_components={n_components} (hard fix) but "
                f"only {len(D)} eigen-components are available after "
                f"the noise filter. Effective rank is determined by "
                f"the data; use n_components_max={n_components} for "
                f"adaptive clamping, or reduce n_components to "
                f"<= {len(D)}."
            )
        k = n_components
    else:
        k = min(n_components_max, len(D))
    return V[:, :k], D[:k]


def read_component_params(src: dict) -> tuple[int | None, int | None]:
    """Extract the ``(n_components, n_components_max)`` pair from a source dict.

    Small shared helper used by genomic / phenomic / enviromic processors
    to plumb the D1 two-knob contract uniformly. Returns a tuple suitable
    for direct forwarding to ``slice_components`` as keyword arguments
    (or to local ``n_comp`` / ``n_comp_max`` variables).
    """
    return src.get("n_components"), src.get("n_components_max")


# ── Projection ───────────────────────────────────────────────────


def project_eigen_train(V_m: np.ndarray, D_m: np.ndarray) -> np.ndarray:
    """Compute train eigen-features: ``Z = V_m * sqrt(D_m)``.

    Parameters
    ----------
    V_m : array of shape (n_train, m)
    D_m : array of shape (m,)

    Returns
    -------
    Z_train : array of shape (n_train, m)
    """
    return V_m * np.sqrt(D_m)[np.newaxis, :]


def project_eigen_cross(
    K_cross_centered: np.ndarray,
    V_m: np.ndarray,
    D_m: np.ndarray,
) -> np.ndarray:
    """Project new samples onto train eigenspace.

    ``Z = K_cross_centered @ V_m @ diag(1 / sqrt(D_m))``

    Parameters
    ----------
    K_cross_centered : array of shape (n_new, n_train)
    V_m : array of shape (n_train, m)
    D_m : array of shape (m,)

    Returns
    -------
    Z_cross : array of shape (n_new, m)
    """
    return K_cross_centered @ V_m * (1.0 / np.sqrt(D_m))[np.newaxis, :]
