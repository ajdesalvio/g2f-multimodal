"""Genomic feature extraction with configurable fitting scope.

Computes additive/dominance GRMs (Van Raden, Vitezica) and PCA on SNP
dosages. ``fit_scope`` controls which pedigrees are used to estimate
allele frequencies, GRMs, and eigendecompositions:

- ``"all"`` (default): fit on every pedigree in the dataset (train ∪ val
  ∪ test). Genotypes are observed prior to planting and are not yield-
  derived, so this introduces no leakage — the rationale mirrors the
  ``"all"`` scope already granted to weather and enviromic processors.
- ``"train"``: fit on train pedigrees only; project val/test via cross-
  kernel (the historical strict-train protocol).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import ClassVar

import numpy as np
import pandas as pd

from .base import BaseProcessor, OrchestratorContext
from .registry import register_processor
from .kernels import (
    center_kernel_cross,
    center_kernel_train,
    eigen_decompose,
    project_eigen_cross,
    project_eigen_train,
    read_component_params,
    slice_components,
)

logger = logging.getLogger(__name__)


# ── CSV loading ──────────────────────────────────────────────────


def load_genomic_csv(
    csv_path: str,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load a wide-format genomic dosage CSV.

    Expects a header row with the first column ``Pedigree`` and the
    remaining columns being SNP markers (integer dosages, typically
    {0, 1, 2}).

    Parameters
    ----------
    csv_path : path to the CSV.
    cache_dir : if provided, results are cached as a single ``.npz``
        keyed by absolute path + mtime so subsequent runs skip the
        ~30s pandas read of the 566 MB file.

    Returns
    -------
    (dosage_matrix, pedigree_names) where ``dosage_matrix`` has shape
    ``(n_pedigrees, n_snps)`` and ``pedigree_names`` is a list aligned
    with its rows.
    """
    abspath = os.path.abspath(csv_path)
    mtime = os.path.getmtime(abspath)

    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        # Filename encodes path basename + mtime so cache invalidates
        # automatically when the source CSV changes.
        tag = f"{os.path.basename(abspath)}_{int(mtime)}.npz"
        cache_path = os.path.join(cache_dir, "genomic_" + tag)
        if os.path.isfile(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            logger.info("Loaded genomic dosage from cache: %s", cache_path)
            return data["dosage"], data["pedigrees"].tolist()

    logger.info("Reading genomic CSV %s ...", csv_path)
    df = pd.read_csv(csv_path)
    if df.columns[0] != "Pedigree":
        raise ValueError(
            f"Genomic CSV {csv_path!r} must have 'Pedigree' as first "
            f"column, got {df.columns[0]!r}"
        )
    pedigrees = df["Pedigree"].astype(str).tolist()
    dosage = df.iloc[:, 1:].to_numpy()

    if cache_path is not None:
        tmp_path = cache_path + f".tmp.{os.getpid()}.npz"
        np.savez(
            tmp_path,
            dosage=dosage,
            pedigrees=np.array(pedigrees, dtype=object),
        )
        try:
            os.replace(tmp_path, cache_path)
        except FileNotFoundError:
            pass
        logger.info("Cached genomic dosage to %s", cache_path)

    return dosage, pedigrees


# ── GRM computation ──────────────────────────────────────────────


def _validate_dosage_matrix(M: np.ndarray, context: str = "genotype matrix") -> None:
    """Fail loud if a dosage matrix holds anything but hard calls {0, 1, 2}.

    The additive/dominance GRMs assume integer dosages. In particular
    ``_dominance_code`` maps ``M != 0`` and ``M != 1`` to the homozygous
    formula, so a fractional/imputed dosage would be silently miscoded;
    and ``compute_allele_frequencies`` would propagate a ``NaN`` straight
    into the GRM and ``eigh``. Rather than miscode or emit ``NaN``, reject
    any value outside ``{0, 1, 2}`` (NaN, fractional, or out-of-range).
    """
    arr = np.asarray(M)
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        raise ValueError(
            f"{context} contains NaN (missing genotypes). Dosages must be "
            f"hard-called integers in {{0, 1, 2}}; impute or drop missing "
            f"calls upstream before building the GRM."
        )
    valid = np.isin(arr, (0, 1, 2))
    if not valid.all():
        bad = np.unique(arr[~valid])
        raise ValueError(
            f"{context} contains values outside {{0, 1, 2}}: "
            f"{bad[:10].tolist()}"
            f"{' ...' if bad.size > 10 else ''}. Dosages must be "
            f"hard-called integers; fractional/imputed dosages are not "
            f"supported (they would be miscoded by the dominance formula)."
        )


def compute_allele_frequencies(M: np.ndarray) -> np.ndarray:
    """Compute per-locus allele frequencies from a dosage matrix.

    Parameters
    ----------
    M : array of shape (n, p), values in {0, 1, 2}.

    Returns
    -------
    p : array of shape (p,), allele frequencies in [0, 1].

    Raises
    ------
    ValueError
        If ``M`` contains ``NaN`` or any value outside ``{0, 1, 2}`` —
        rather than silently producing ``NaN`` / miscoded frequencies.
    """
    _validate_dosage_matrix(M, context="allele-frequency dosage matrix")
    return M.mean(axis=0) / 2.0


def _additive_center(
    M: np.ndarray, allele_freq: np.ndarray
) -> np.ndarray:
    """Center dosage matrix for additive GRM: Z = M - 2p."""
    return M - 2.0 * allele_freq[np.newaxis, :]


def _additive_scale(allele_freq: np.ndarray) -> float:
    """Scaling denominator for Van Raden additive GRM: 2 * sum(p*(1-p))."""
    pq = allele_freq * (1.0 - allele_freq)
    return 2.0 * pq.sum()


def compute_additive_grm(
    M: np.ndarray,
    allele_freq: np.ndarray,
) -> np.ndarray:
    """Additive GRM (Van Raden 2008): G = ZZ' / (2 * sum(p*(1-p))).

    Parameters
    ----------
    M : dosage matrix (n, p), values in {0, 1, 2}.
    allele_freq : array of shape (p,), allele frequencies estimated on
        the fitting set — every pedigree under ``fit_scope="all"``
        (the default), train pedigrees only under ``fit_scope="train"``.

    Returns
    -------
    G : array of shape (n, n).
    """
    Z = _additive_center(M, allele_freq)
    scale = _additive_scale(allele_freq)
    if scale == 0:
        logger.warning("Additive GRM scale is zero (all loci monomorphic).")
        return np.zeros((M.shape[0], M.shape[0]))
    return Z @ Z.T / scale


def compute_additive_grm_cross(
    M_new: np.ndarray,
    M_train: np.ndarray,
    allele_freq: np.ndarray,
) -> np.ndarray:
    """Cross additive GRM: K(new, train) = Z_new @ Z_train.T / scale.

    Both M_new and M_train are centered with the same train allele freqs.
    """
    Z_new = _additive_center(M_new, allele_freq)
    Z_train = _additive_center(M_train, allele_freq)
    scale = _additive_scale(allele_freq)
    if scale == 0:
        return np.zeros((M_new.shape[0], M_train.shape[0]))
    return Z_new @ Z_train.T / scale


def _dominance_code(
    M: np.ndarray, allele_freq: np.ndarray
) -> np.ndarray:
    """Dominance coding (Vitezica et al. 2013).

    W_ij = -2*p_j^2       if M_ij = 0
    W_ij =  2*p_j*(1-p_j) if M_ij = 1
    W_ij = -2*(1-p_j)^2   if M_ij = 2

    The nested ``np.where`` below sends every non-{0, 1} dosage down the
    ``M == 2`` branch, so a fractional/imputed/NaN value would be silently
    coded with the homozygous-alt formula. Validate up front and fail loud
    instead of miscoding.
    """
    _validate_dosage_matrix(M, context="dominance-coding dosage matrix")
    p = allele_freq[np.newaxis, :]
    W = np.where(
        M == 0,
        -2.0 * p**2,
        np.where(
            M == 1,
            2.0 * p * (1.0 - p),
            -2.0 * (1.0 - p) ** 2,
        ),
    )
    return W


def _dominance_scale(allele_freq: np.ndarray) -> float:
    """Scaling denominator for Vitezica dominance GRM: 4 * sum(p^2*(1-p)^2)."""
    pq_sq = (allele_freq * (1.0 - allele_freq)) ** 2
    return 4.0 * pq_sq.sum()


def compute_dominance_grm(
    M: np.ndarray,
    allele_freq: np.ndarray,
) -> np.ndarray:
    """Dominance GRM (Vitezica 2013): G = WW' / (4 * sum(p^2*(1-p)^2)).

    Parameters
    ----------
    M : dosage matrix (n, p), values in {0, 1, 2}.
    allele_freq : array of shape (p,), allele frequencies estimated on
        the fitting set — every pedigree under ``fit_scope="all"``
        (the default), train pedigrees only under ``fit_scope="train"``.

    Returns
    -------
    G : array of shape (n, n).
    """
    W = _dominance_code(M, allele_freq)
    scale = _dominance_scale(allele_freq)
    if scale == 0:
        logger.warning(
            "Dominance GRM scale is zero (all loci monomorphic)."
        )
        return np.zeros((M.shape[0], M.shape[0]))
    return W @ W.T / scale


def compute_dominance_grm_cross(
    M_new: np.ndarray,
    M_train: np.ndarray,
    allele_freq: np.ndarray,
) -> np.ndarray:
    """Cross dominance GRM: K(new, train) = W_new @ W_train.T / scale."""
    W_new = _dominance_code(M_new, allele_freq)
    W_train = _dominance_code(M_train, allele_freq)
    scale = _dominance_scale(allele_freq)
    if scale == 0:
        return np.zeros((M_new.shape[0], M_train.shape[0]))
    return W_new @ W_train.T / scale


def _raw_pca_effective_k(
    n_components: int | None,
    n_components_max: int | None,
    n_avail: int,
    src_name: str,
) -> int:
    """Resolve raw-PCA component count per the D1 two-knob contract.

    Mirrors ``kernels.slice_components`` semantics for the raw SVD path,
    which doesn't have a ``(V, D)`` pair to hand off. Raises on hard-fix
    overflow, clamps silently under the soft ceiling.
    """
    if n_components is not None and n_components_max is not None:
        raise ValueError(
            f"source={src_name!r}: n_components and n_components_max are "
            f"mutually exclusive; set at most one."
        )
    if n_components is None and n_components_max is None:
        return n_avail
    if n_components is not None:
        if n_components > n_avail:
            raise ValueError(
                f"source={src_name!r}: n_components={n_components} "
                f"(hard fix) exceeds raw-PCA rank {n_avail}. Use "
                f"n_components_max={n_components} for adaptive "
                f"clamping, or reduce n_components to <= {n_avail}."
            )
        return n_components
    return min(n_components_max, n_avail)


# ── GenomicFeatureBuilder ────────────────────────────────────────


@register_processor
class GenomicFeatureBuilder(BaseProcessor):
    """Genomic feature extraction with configurable fitting scope.

    Processes multiple sources simultaneously (e.g., additive + dominance GRM).
    Each source has a ``name`` that determines its key in the output dict.

    Parameters
    ----------
    sources : list of dicts, each with keys:
        - ``name``: str — output key (e.g., "genomic_add")
        - ``type``: str — one of "additive_grm", "dominance_grm",
          "additive_grm_rows", "dominance_grm_rows", "raw"
        - ``n_components``: int | None — hard-fix component count for
          eigen/PCA sources. Raises if greater than the effective rank
          of the filtered spectrum.
        - ``n_components_max``: int | None — soft ceiling for
          eigen/PCA sources. Mutually exclusive with ``n_components``.
          Clamps silently to ``min(K, effective_rank)``.
    fit_scope : ``"all"`` (default) or ``"train"``. ``"all"`` fits
        allele frequencies, GRMs, and eigendecompositions on every
        pedigree present in the dataset (train ∪ val ∪ test); val/test
        projection is then a direct index lookup. ``"train"`` restricts
        the fit to training pedigrees and projects val/test via cross-
        kernel.
    """

    # GRM types that need eigendecomposition
    _EIGEN_TYPES = {"additive_grm", "dominance_grm"}
    # GRM types that return raw centered kernel rows
    _ROW_TYPES = {"additive_grm_rows", "dominance_grm_rows"}
    # All GRM types (need GRM computation)
    _GRM_TYPES = _EIGEN_TYPES | _ROW_TYPES

    _VALID_FIT_SCOPES = ("all", "train")

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "genomic"
    priority: ClassVar[int] = 5
    has_eigen_sources: ClassVar[bool] = True
    eigen_source_types: ClassVar[frozenset[str]] = frozenset(
        {"additive_grm", "dominance_grm", "raw"}
    )

    @classmethod
    def from_config(cls, config: dict) -> "GenomicFeatureBuilder":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            sources=config.get("sources", []),
            fit_scope=config.get("fit_scope", "all"),
            center_kernel=config.get("center_kernel", False),
        )

    @classmethod
    def coverage_mask(cls, metadata_df, proc_config):
        """Drop samples whose pedigree is not in the genomic CSV.

        Reads the dosage CSV's pedigree index (cheap header read) and
        returns a boolean mask of the rows to keep. Returning `None`
        when disabled lets the orchestrator skip the filter entirely.
        """
        if not proc_config.get("enabled", False):
            return None
        from utils.data.dataset import _read_pedigree_index

        genomic_pedigrees = _read_pedigree_index(proc_config["csv_path"])
        return metadata_df["Pedigree"].isin(genomic_pedigrees).values

    def __init__(
        self,
        sources: list[dict],
        fit_scope: str = "all",
        center_kernel: bool = False,
    ):
        self.sources = sources
        if fit_scope not in self._VALID_FIT_SCOPES:
            raise ValueError(
                f"fit_scope must be one of {self._VALID_FIT_SCOPES}, "
                f"got {fit_scope!r}."
            )
        self.fit_scope = fit_scope
        # When False, the GRM is eigen-decomposed in its raw form
        # (no double-centering, no cross-kernel centering offsets) —
        # feeds the unmodified VanRaden / Vitezica K_A / K_D into
        # RKHS terms. True matches the textbook kernel-PCA recipe;
        # the shipped default is False (reference parity).
        self.center_kernel = bool(center_kernel)
        self._validate_sources()

        # Fitted state (populated by fit())
        self._allele_freq: np.ndarray | None = None
        self._fit_pedigrees: list[str] | None = None
        self._ped_to_idx: dict[str, int] | None = None
        self._fit_dosage: np.ndarray | None = None

        # Per-GRM-type fitted state
        self._grm_state: dict[str, dict] = {}
        # PCA fitted state (for "raw" source type)
        self._pca_state: dict | None = None

    def _validate_sources(self):
        valid_types = {
            "additive_grm", "dominance_grm",
            "additive_grm_rows", "dominance_grm_rows",
            "raw", "raw_dosage",
        }
        names = set()
        for src in self.sources:
            if src["type"] not in valid_types:
                raise ValueError(
                    f"Unknown genomic source type: {src['type']!r}. "
                    f"Valid: {sorted(valid_types)}"
                )
            if src["name"] in names:
                raise ValueError(
                    f"Duplicate source name: {src['name']!r}"
                )
            names.add(src["name"])

    @property
    def _needs_additive(self) -> bool:
        return any(
            s["type"] in ("additive_grm", "additive_grm_rows")
            for s in self.sources
        )

    @property
    def _needs_dominance(self) -> bool:
        return any(
            s["type"] in ("dominance_grm", "dominance_grm_rows")
            for s in self.sources
        )

    @property
    def _needs_raw_pca(self) -> bool:
        return any(s["type"] == "raw" for s in self.sources)

    def fit_impl(
        self,
        train_pedigrees: list[str],
        dosage_matrix: np.ndarray,
        pedigree_names: list[str],
        all_pedigrees: list[str] | None = None,
    ) -> None:
        """Fit allele frequencies, GRMs, and eigendecompositions.

        Parameters
        ----------
        train_pedigrees : per-sample pedigree list for the train split
            (may have duplicates).
        dosage_matrix : full dosage matrix (n_total_pedigrees, n_snps),
            rows correspond to ``pedigree_names``.
        pedigree_names : pedigree names corresponding to rows of
            ``dosage_matrix``.
        all_pedigrees : per-sample pedigree list spanning train ∪ val ∪
            test. Required when ``fit_scope="all"``; ignored otherwise.
        """
        fit_pedigrees = self._resolve_fit_pedigrees(
            train_pedigrees, all_pedigrees
        )
        # Deduplicate and sort for determinism
        unique_fit = sorted(set(fit_pedigrees))
        self._fit_pedigrees = unique_fit
        self._ped_to_idx = {p: i for i, p in enumerate(unique_fit)}

        # Subset dosage matrix to fit pedigrees
        name_to_row = {n: i for i, n in enumerate(pedigree_names)}
        fit_rows = [name_to_row[p] for p in unique_fit]
        M_fit = dosage_matrix[fit_rows].astype(np.float64)

        # Compute allele frequencies from the fitting set
        self._allele_freq = compute_allele_frequencies(M_fit)

        # Store fit dosage for cross-kernel computation (used only when
        # fit_scope="train" and a sample falls outside the fitting set)
        self._fit_dosage = M_fit

        # Compute GRMs as needed
        if self._needs_additive:
            G_add = compute_additive_grm(M_fit, self._allele_freq)
            self._grm_state["additive"] = self._build_grm_state(
                G_add,
                eigen_needed=any(
                    s["type"] == "additive_grm" for s in self.sources
                ),
                rows_needed=any(
                    s["type"] == "additive_grm_rows" for s in self.sources
                ),
            )

        if self._needs_dominance:
            G_dom = compute_dominance_grm(M_fit, self._allele_freq)
            self._grm_state["dominance"] = self._build_grm_state(
                G_dom,
                eigen_needed=any(
                    s["type"] == "dominance_grm" for s in self.sources
                ),
                rows_needed=any(
                    s["type"] == "dominance_grm_rows" for s in self.sources
                ),
            )

        # PCA on centered SNP matrix
        if self._needs_raw_pca:
            Z = _additive_center(M_fit, self._allele_freq)
            # Economy SVD: Z = U @ diag(S) @ Vt, where Vt is (min(n,p), p)
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            self._pca_state = {"U": U, "S": S, "Vt": Vt}

    def _build_grm_state(
        self,
        G: np.ndarray,
        eigen_needed: bool,
        rows_needed: bool,
    ) -> dict:
        """Construct per-GRM-type fitted state, honoring center_kernel.

        When ``self.center_kernel`` is True, G is double-
        centered before eigendecomposition. When False (default), G is used as-is
        and ``col_means`` / ``grand_mean`` are zero placeholders kept
        for cache-schema uniformity (cross-kernel projection in
        transform skips the centering step too).
        """
        state: dict = {}
        if eigen_needed or rows_needed:
            if self.center_kernel:
                K_used, col_means, grand_mean = center_kernel_train(G)
            else:
                K_used = G
                col_means = np.zeros(G.shape[0], dtype=np.float64)
                grand_mean = 0.0
            state["K_centered"] = K_used
            state["col_means"] = col_means
            state["grand_mean"] = grand_mean
            if eigen_needed:
                # Always filter numerical noise. R-reference parity for the
                # transductive (BGLR/GBLUP) path flows through
                # raw_kernel() — unfiltered, with BGLR's own tolD applied
                # downstream — not through this eigenbasis. eps=0 under
                # fit_scope='all' kept ~N/2 spurious ~1e-17 modes that
                # defeat the rank check and bloat cache and compute.
                eig_eps = 1e-10
                V, D = eigen_decompose(K_used, eps=eig_eps)
                state["V"] = V
                state["D"] = D
        return state

    def _resolve_fit_pedigrees(
        self,
        train_pedigrees: list[str],
        all_pedigrees: list[str] | None,
    ) -> list[str]:
        """Pick the pedigree list that defines the fitting set."""
        if self.fit_scope == "all":
            if all_pedigrees is None:
                raise ValueError(
                    "all_pedigrees is required when fit_scope='all'."
                )
            return all_pedigrees
        return train_pedigrees

    def _is_in_fitting_set(self, sample_pedigrees: list[str]) -> bool:
        """True iff every sample pedigree was seen at fit time."""
        return all(p in self._ped_to_idx for p in sample_pedigrees)

    def transform_impl(
        self,
        sample_pedigrees: list[str],
        split_name: str,
        dosage_matrix: np.ndarray | None = None,
        pedigree_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Transform pedigrees into feature arrays.

        Parameters
        ----------
        sample_pedigrees : per-sample pedigree list (may have duplicates).
        split_name : "train", "val", or "test".
        dosage_matrix : full dosage matrix, needed for cross-kernel and
            raw PCA projection on non-train splits.
        pedigree_names : pedigree names for ``dosage_matrix`` rows.

        Returns
        -------
        dict mapping source name → array of shape (n_samples, dim).
        """
        n_samples = len(sample_pedigrees)
        result = {}

        for src in self.sources:
            name = src["name"]
            stype = src["type"]
            n_comp, n_comp_max = read_component_params(src)

            if stype == "additive_grm":
                features = self._transform_grm_eigen(
                    sample_pedigrees, "additive",
                    n_comp, n_comp_max, dosage_matrix, pedigree_names,
                    grm_fn=compute_additive_grm_cross,
                )
            elif stype == "dominance_grm":
                features = self._transform_grm_eigen(
                    sample_pedigrees, "dominance",
                    n_comp, n_comp_max, dosage_matrix, pedigree_names,
                    grm_fn=compute_dominance_grm_cross,
                )
            elif stype == "additive_grm_rows":
                features = self._transform_grm_rows(
                    sample_pedigrees, "additive",
                    dosage_matrix, pedigree_names,
                    grm_fn=compute_additive_grm_cross,
                )
            elif stype == "dominance_grm_rows":
                features = self._transform_grm_rows(
                    sample_pedigrees, "dominance",
                    dosage_matrix, pedigree_names,
                    grm_fn=compute_dominance_grm_cross,
                )
            elif stype == "raw":
                features = self._transform_raw_pca(
                    sample_pedigrees, n_comp, n_comp_max,
                    dosage_matrix, pedigree_names,
                )
            elif stype == "raw_dosage":
                features = self._transform_raw_dosage(
                    sample_pedigrees, dosage_matrix, pedigree_names,
                )
            else:
                raise ValueError(f"Unknown source type: {stype!r}")

            assert features.shape[0] == n_samples, (
                f"Source {name!r}: expected {n_samples} rows, "
                f"got {features.shape[0]}"
            )
            result[name] = features

        return result

    def raw_kernel(
        self,
        ctx: OrchestratorContext,
        source_name: str,
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        """Return obs-level ``Z_a @ K_ped @ Z_a^T`` for the given rows.

        ``K_ped = self._grm_state[grm_key]["K_centered"]`` is the
        pedigree-level GRM (VanRaden / Vitezica, optionally double-
        centered). The obs-level kernel is built via ``np.ix_`` slicing
        (equivalent to ``Z_a K_ped Z_a^T`` but without materializing
        the indicator matrix).
        """
        src = next(s for s in self.sources if s["name"] == source_name)
        stype = src["type"]
        grm_key_by_type = {
            "additive_grm": "additive",
            "dominance_grm": "dominance",
            "additive_grm_rows": "additive",
            "dominance_grm_rows": "dominance",
        }
        grm_key = grm_key_by_type.get(stype)
        if grm_key is None:
            raise NotImplementedError(
                f"GenomicFeatureBuilder.raw_kernel does not support "
                f"source type {stype!r}."
            )
        K_ped = self._grm_state[grm_key]["K_centered"]
        sample_peds = ctx.metadata_df.loc[sample_indices, "Pedigree"].tolist()
        for ped in sample_peds:
            if ped not in self._ped_to_idx:
                raise KeyError(
                    f"GenomicFeatureBuilder.raw_kernel: pedigree {ped!r} "
                    f"not in fitted GRM (fit_scope={self.fit_scope!r})."
                )
        ped_idx = np.array(
            [self._ped_to_idx[p] for p in sample_peds], dtype=np.int64,
        )
        return K_ped[np.ix_(ped_idx, ped_idx)]

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Load dosage matrix from CSV (cached), pull pedigrees from
        `ctx`, delegate to `fit_impl`.
        """
        csv_path = ctx.proc_config["csv_path"]
        dosage_matrix, pedigree_names = load_genomic_csv(
            csv_path, cache_dir=ctx.cache_dir,
        )
        # Stash for transform() to reuse without re-reading the CSV.
        self._dosage_matrix = dosage_matrix
        self._pedigree_names = pedigree_names
        train_peds = ctx.pedigrees_for_split("train")
        all_peds = ctx.all_pedigrees()
        self.fit_impl(
            train_peds, dosage_matrix, pedigree_names,
            all_pedigrees=all_peds,
        )

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """Pull split's pedigrees from `ctx` and delegate to
        `transform_impl`. Lazily loads the dosage CSV in predict mode
        (after `load_fitted`, where `fit` was skipped).
        """
        idxs = ctx.split_indices.get(split, [])
        if not idxs:
            return None
        if getattr(self, "_dosage_matrix", None) is None:
            csv_path = ctx.proc_config["csv_path"]
            self._dosage_matrix, self._pedigree_names = load_genomic_csv(
                csv_path, cache_dir=ctx.cache_dir,
            )
        peds = ctx.pedigrees_for_split(split)
        return self.transform_impl(
            peds, split,
            dosage_matrix=self._dosage_matrix,
            pedigree_names=self._pedigree_names,
        )

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Recompute via `compute_cache_key` from ctx-derived inputs."""
        train_peds = ctx.pedigrees_for_split("train")
        all_peds = ctx.all_pedigrees()
        return self.compute_cache_key(
            train_peds, self._dosage_matrix, self._pedigree_names,
            all_pedigrees=all_peds,
        )

    @property
    def feature_dims(self) -> dict[str, int]:
        """Output dimensions per source. Available after fit().

        Reflects the **actually-used** width per D1:
        - ``n_components: K`` (hard fix): exactly K, or ``slice_components``
          raises during fit if ``K > effective_rank``.
        - ``n_components_max: K`` (soft ceiling): ``min(K, effective_rank)``.
        - both unset: effective rank.
        """
        dims = {}
        for src in self.sources:
            stype = src["type"]
            n_comp, n_comp_max = read_component_params(src)

            if stype in self._EIGEN_TYPES:
                grm_key = "additive" if "additive" in stype else "dominance"
                state = self._grm_state[grm_key]
                _, D_m = slice_components(
                    state["V"], state["D"],
                    n_components=n_comp,
                    n_components_max=n_comp_max,
                )
                dims[src["name"]] = len(D_m)
            elif stype in self._ROW_TYPES:
                dims[src["name"]] = len(self._fit_pedigrees)
            elif stype == "raw":
                S = self._pca_state["S"]
                dims[src["name"]] = _raw_pca_effective_k(
                    n_comp, n_comp_max, len(S), src["name"],
                )
            elif stype == "raw_dosage":
                # Literal marker dosage: width is the SNP count.
                dims[src["name"]] = int(self._fit_dosage.shape[1])
        return dims

    # ── Cache: save / load fitted state ──────────────────────────

    def compute_cache_key(
        self,
        train_pedigrees: list[str],
        dosage_matrix: np.ndarray,
        pedigree_names: list[str],
        all_pedigrees: list[str] | None = None,
    ) -> str:
        """Content-addressed cache key for the fitted state.

        Derived from: source config (name/type/normalization, excluding
        both ``n_components`` and ``n_components_max`` which are
        post-load slices), ``center_kernel``, ``fit_scope``, the
        deduplicated sorted fit pedigree list, and the fit-subset
        dosage matrix content.
        """
        fit_pedigrees = self._resolve_fit_pedigrees(
            train_pedigrees, all_pedigrees
        )
        h = hashlib.sha256()
        # v4 applies only to fit_scope='all' entries: their v3 eigen
        # spectra were computed with eps=0 (numerical-noise modes kept)
        # and must not be served. Train-scope entries are bit-identical
        # under both versions and stay warm on v3.
        h.update(
            b"genomic_v4" if self.fit_scope == "all" else b"genomic_v3"
        )

        src_repr = json.dumps(
            [
                {
                    "name": s["name"],
                    "type": s["type"],
                    "normalization": s.get("normalization"),
                }
                for s in self.sources
            ],
            sort_keys=True,
        ).encode()
        h.update(src_repr)
        h.update(f"|fit_scope={self.fit_scope}".encode())
        h.update(f"|center_kernel={self.center_kernel}".encode())

        unique_fit = sorted(set(fit_pedigrees))
        h.update(b"|peds=")
        h.update("|".join(unique_fit).encode())

        name_to_row = {n: i for i, n in enumerate(pedigree_names)}
        fit_rows = [name_to_row[p] for p in unique_fit]
        M_fit = dosage_matrix[fit_rows].astype(np.float64)
        h.update(b"|dosage=")
        h.update(M_fit.tobytes())

        return h.hexdigest()

    def save_cache(self, cache_dir: str, cache_key: str) -> None:
        """Persist fitted state to ``cache_dir/genomic_<key[:16]>.npz``.

        Writes atomically (tmp + rename). Raises ``RuntimeError`` if the
        builder has not been fitted yet.
        """
        if self._allele_freq is None:
            raise RuntimeError("Cannot save_cache before fit().")

        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"genomic_{cache_key[:16]}.npz")

        payload: dict[str, np.ndarray] = {
            "allele_freq": self._allele_freq,
            "fit_pedigrees": np.array(self._fit_pedigrees, dtype=object),
            "fit_dosage": self._fit_dosage,
            "config_json": np.array(
                json.dumps(
                    {
                        "sources": self.sources,
                        "fit_scope": self.fit_scope,
                        "center_kernel": self.center_kernel,
                    }
                ),
                dtype=object,
            ),
        }
        for grm_key, state in self._grm_state.items():
            for attr_name, arr in state.items():
                payload[f"grm_{grm_key}_{attr_name}"] = arr
        if self._pca_state is not None:
            for attr_name, arr in self._pca_state.items():
                payload[f"pca_{attr_name}"] = arr

        # np.savez appends ".npz" if the filename does not already end
        # in it — use a .tmp.npz tmp name so the file ends up where we
        # expect, then atomically rename. The pid is included to avoid
        # races when parallel jobs share the same cache key (same train
        # pedigrees, different folds or configs).
        tmp_path = path + f".tmp.{os.getpid()}.npz"
        np.savez(tmp_path, **payload)
        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:
            # Another process already wrote the final file and cleaned
            # up — the cache is warm, nothing to do.
            pass
        logger.info("Genomic cache saved: %s", path)

    def load_fitted(self, cache_dir: str, cache_key: str) -> None:
        """Restore fitted state from ``cache_dir/genomic_<key[:16]>.npz``.

        After this call, ``transform()`` can be used on any split without
        re-fitting. Raises ``FileNotFoundError`` if the cache entry does
        not exist, or ``ValueError`` if the cached config (``sources``
        or ``fit_scope``) does not match the current instance.
        """
        path = os.path.join(cache_dir, f"genomic_{cache_key[:16]}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Genomic cache not found: {path}")

        data = np.load(path, allow_pickle=True)

        saved = json.loads(str(data["config_json"].item()))
        if (
            saved.get("sources") != self.sources
            or saved.get("fit_scope") != self.fit_scope
            or saved["center_kernel"] != self.center_kernel
        ):
            raise ValueError(
                f"Genomic config mismatch: cache has sources="
                f"{saved.get('sources')}, fit_scope="
                f"{saved.get('fit_scope')!r}, center_kernel="
                f"{saved['center_kernel']}; current config has "
                f"sources={self.sources}, fit_scope={self.fit_scope!r}, "
                f"center_kernel={self.center_kernel}"
            )

        self._allele_freq = data["allele_freq"]
        self._fit_pedigrees = [str(p) for p in data["fit_pedigrees"]]
        self._ped_to_idx = {p: i for i, p in enumerate(self._fit_pedigrees)}
        self._fit_dosage = data["fit_dosage"]

        self._grm_state = {}
        for grm_key in ("additive", "dominance"):
            prefix = f"grm_{grm_key}_"
            matched = {
                k[len(prefix):]: data[k]
                for k in data.files
                if k.startswith(prefix)
            }
            if matched:
                self._grm_state[grm_key] = matched

        pca_prefix = "pca_"
        pca_matched = {
            k[len(pca_prefix):]: data[k]
            for k in data.files
            if k.startswith(pca_prefix)
        }
        self._pca_state = pca_matched if pca_matched else None

        logger.info("Genomic cache loaded: %s", path)

    # ── Private helpers ──────────────────────────────────────────

    def _get_per_pedigree_features(
        self, features_per_ped: np.ndarray, sample_pedigrees: list[str]
    ) -> np.ndarray:
        """Map per-unique-pedigree features to per-sample features."""
        indices = [self._ped_to_idx[p] for p in sample_pedigrees]
        return features_per_ped[indices]

    def _transform_grm_eigen(
        self, sample_pedigrees, grm_key,
        n_components, n_components_max,
        dosage_matrix, pedigree_names, grm_fn,
    ) -> np.ndarray:
        state = self._grm_state[grm_key]
        V_m, D_m = slice_components(
            state["V"], state["D"],
            n_components=n_components,
            n_components_max=n_components_max,
        )

        if self._is_in_fitting_set(sample_pedigrees):
            Z_per_ped = project_eigen_train(V_m, D_m)
            return self._get_per_pedigree_features(
                Z_per_ped, sample_pedigrees
            )
        else:
            # Cross-kernel (fit_scope="train" with out-of-fit samples)
            M_new = self._load_pedigree_dosage(
                sample_pedigrees, dosage_matrix, pedigree_names
            )
            K_cross = grm_fn(M_new, self._fit_dosage, self._allele_freq)
            K_for_proj = (
                center_kernel_cross(
                    K_cross, state["col_means"], state["grand_mean"]
                )
                if self.center_kernel
                else K_cross
            )
            return project_eigen_cross(K_for_proj, V_m, D_m)

    def _transform_grm_rows(
        self, sample_pedigrees, grm_key,
        dosage_matrix, pedigree_names, grm_fn,
    ) -> np.ndarray:
        state = self._grm_state[grm_key]

        if self._is_in_fitting_set(sample_pedigrees):
            return self._get_per_pedigree_features(
                state["K_centered"], sample_pedigrees
            )
        else:
            M_new = self._load_pedigree_dosage(
                sample_pedigrees, dosage_matrix, pedigree_names
            )
            K_cross = grm_fn(M_new, self._fit_dosage, self._allele_freq)
            return (
                center_kernel_cross(
                    K_cross, state["col_means"], state["grand_mean"]
                )
                if self.center_kernel
                else K_cross
            )

    def _transform_raw_pca(
        self, sample_pedigrees,
        n_components, n_components_max,
        dosage_matrix, pedigree_names,
    ) -> np.ndarray:
        state = self._pca_state
        U, S, Vt = state["U"], state["S"], state["Vt"]

        k = _raw_pca_effective_k(
            n_components, n_components_max, len(S), src_name="raw",
        )

        if self._is_in_fitting_set(sample_pedigrees):
            # Fit-set scores: U[:, :k] * S[:k]
            scores_per_ped = U[:, :k] * S[:k]
            return self._get_per_pedigree_features(
                scores_per_ped, sample_pedigrees
            )
        else:
            # Project new data: X_centered @ V[:k].T
            M_new = self._load_pedigree_dosage(
                sample_pedigrees, dosage_matrix, pedigree_names
            )
            Z_new = _additive_center(M_new, self._allele_freq)
            # Vt[:k] is (k, p), so Z_new @ Vt[:k].T = (n_new, k)
            return Z_new @ Vt[:k].T

    def _transform_raw_dosage(
        self, sample_pedigrees,
        dosage_matrix, pedigree_names,
    ) -> np.ndarray:
        """Per-sample raw SNP dosage vectors — no GRM, no PCA.

        Returns the marker dosage rows directly, i.e. the literal genomic
        input for a peer MLP. Fit-set pedigrees are served from the
        cached ``_fit_dosage`` (so no ``dosage_matrix`` is needed under
        the default ``fit_scope='all'``); out-of-fit pedigrees
        (``fit_scope='train'``) are looked up in ``dosage_matrix``.
        """
        if self._is_in_fitting_set(sample_pedigrees):
            return self._get_per_pedigree_features(
                self._fit_dosage, sample_pedigrees
            )
        return self._load_pedigree_dosage(
            sample_pedigrees, dosage_matrix, pedigree_names
        )

    def _load_pedigree_dosage(
        self,
        sample_pedigrees: list[str],
        dosage_matrix: np.ndarray | None,
        pedigree_names: list[str] | None,
    ) -> np.ndarray:
        """Load dosage matrix rows for the given pedigrees.

        Deduplicates pedigrees, subsets dosage_matrix, then expands back
        to per-sample ordering.
        """
        if dosage_matrix is None or pedigree_names is None:
            raise ValueError(
                "dosage_matrix and pedigree_names are required for "
                "non-train splits."
            )

        # Deduplicate to avoid redundant lookups
        unique_peds = sorted(set(sample_pedigrees))
        name_to_row = {n: i for i, n in enumerate(pedigree_names)}
        rows = [name_to_row[p] for p in unique_peds]
        M_unique = dosage_matrix[rows].astype(np.float64)

        # Map back to per-sample ordering
        ped_to_unique_idx = {p: i for i, p in enumerate(unique_peds)}
        sample_indices = [ped_to_unique_idx[p] for p in sample_pedigrees]
        return M_unique[sample_indices]
