"""VI FPCA feature extraction via R fdapace.

Wraps the existing R-based FPCA pipeline (``baselines/fpca_compute.R``) into a
processor that fits on train data only, projects all splits via CE/BLUP, and
caches results with content-addressed keys.

``n_components`` is excluded from the cache key: the R script outputs all
available FPC scores (fdapace computes all eigenfunctions internally), and
``n_components`` is applied as a cheap post-load slice in Python.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import warnings
from typing import ClassVar

import numpy as np
import pandas as pd
import torch

from .base import BaseProcessor, OrchestratorContext
from .registry import register_processor
from ..views import View, assemble_view

logger = logging.getLogger(__name__)

# Path to the R script (shared with baselines — single source of truth).
_R_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    os.pardir,
    "baselines",
    "fpca_compute.R",
)

N_VIS = 37  # Number of vegetation indices

# Allowed values for the VI FPCA missing-cell policy.
MISSING_VALUE_MODES: tuple[str, ...] = ("skip", "interpolate")

# Time axes selectable as the fdapace coordinate. ``dap`` is native;
# ``gdd`` / ``agdd`` are attached upstream by AxisSourceProcessor. Kept in
# sync with ``axis_source.AXIS_NAMES`` (duplicated to avoid import coupling).
AXIS_NAMES: tuple[str, ...] = ("dap", "gdd", "agdd")
# Derived axes whose per-sample values are not captured by the dap/channels
# content hash, so they must be hashed into the cache key when referenced
# (as a coord or an extra channel) by a non-trivial FPCA view.
DERIVED_AXES: tuple[str, ...] = ("gdd", "agdd")


# ── Deterministic sample-ID helpers ─────────────────────────────


def _sample_hash(sample: dict[str, torch.Tensor]) -> bytes:
    """SHA-256 of one sample's arrays (DAPs, VI values, NaN mask, yield).

    The NaN mask is included so two samples with identical interpolated
    values but different originally-missing cells produce different
    hashes — that distinction matters under
    ``missing_values='skip'``. Older sample dicts without ``vi_nan_mask``
    hash as if no cells were missing.
    """
    h = hashlib.sha256()
    h.update(sample["dap"].numpy().tobytes())
    h.update(sample["channels"].numpy().tobytes())
    nan_mask = sample.get("vi_nan_mask")
    if nan_mask is not None:
        h.update(nan_mask.numpy().tobytes())
    h.update(str(sample["yield_value"].item()).encode())
    return h.digest()


def _split_content_hash(data_list: list[dict[str, torch.Tensor]]) -> str:
    """Order-independent SHA-256 over all samples in a split."""
    per_sample = sorted(_sample_hash(s).hex() for s in data_list)
    h = hashlib.sha256()
    h.update(str(len(data_list)).encode())
    for fp in per_sample:
        h.update(fp.encode())
    return h.hexdigest()


def _hash_sort_order(data_list: list[dict[str, torch.Tensor]]) -> list[int]:
    """Return ``order`` such that ``order[k]`` is the original index of the
    sample that ends up at position k after sorting by content hash.

    Mirrors the logic inside :func:`_extract_sparse_curves` so that the row
    order produced by R can be reversed back to input order.
    """
    return sorted(range(len(data_list)), key=lambda i: _sample_hash(data_list[i]))


def _hash_to_input_perm(data_list: list[dict[str, torch.Tensor]]) -> np.ndarray:
    """Inverse of :func:`_hash_sort_order`.

    Returns ``perm`` of length ``len(data_list)`` such that
    ``X_hash_order[perm]`` yields rows in input-list order.  Equivalently,
    ``perm[i]`` is the hash-sort position of input sample ``i``.
    """
    order = _hash_sort_order(data_list)
    perm = np.empty(len(data_list), dtype=np.int64)
    for hash_pos, orig_idx in enumerate(order):
        perm[orig_idx] = hash_pos
    return perm


def _align_raw_scores_to_input(
    raw_scores: dict[str, tuple[np.ndarray, np.ndarray, int]],
    data_lists: dict[str, list[dict[str, torch.Tensor]] | None],
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Permute each split's score matrix from hash order to input order.

    R produces FPC scores in content-hash order (see
    :func:`_extract_sparse_curves`).  Downstream consumers expect features
    indexed by *input* position so they can attach ``derived_features[i]``
    to ``data_list[i]``.  This helper applies the inverse permutation per
    split using the current data lists.
    """
    aligned: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for split_name, payload in raw_scores.items():
        X, ids, max_k = payload
        data_list = data_lists.get(split_name)
        if not data_list:
            aligned[split_name] = payload
            continue
        if X.shape[0] != len(data_list):
            raise ValueError(
                f"VI FPCA: split {split_name!r} has {X.shape[0]} score rows "
                f"but {len(data_list)} input samples — cannot align."
            )
        perm = _hash_to_input_perm(data_list)
        aligned[split_name] = (X[perm], ids[perm] if ids.size else ids, max_k)
    return aligned


# ── Sparse curve extraction ─────────────────────────────────────


def _sample_curve_arrays(
    sample: dict[str, torch.Tensor],
    axis: str,
    view: "View | None",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(t[T], values[T, C], mask[T, C] | None)`` for one sample.

    Two paths:

    - **view is None** (trivial dap/agdd config): the legacy fast path —
      ``t = sample[axis]``, ``values = sample["channels"]``, ``mask =
      sample["vi_nan_mask"]``. Byte-identical to the pre-View bridge.
    - **view given** (dedup / warp / extra-channel): assemble via
      :func:`~utils.data.views.assemble_view` and read ``t =
      coords[:, 0]``, ``values = channels`` (each column an independent FPCA
      block), ``mask`` carried through the transform (G3).
    """
    if view is None:
        if axis not in sample:
            raise KeyError(
                f"VI FPCA axis={axis!r} but sample has no {axis!r} key. "
                f"Enable the `axis_source` processor (priority -1) so it "
                f"attaches the derived axis before vi_fpca runs."
            )
        t_vals = sample[axis].numpy().squeeze(-1)
        vis = sample["channels"].numpy()
        m = sample.get("vi_nan_mask")
        mask = None if m is None else m.numpy()
        return t_vals, vis, mask

    arr = assemble_view(sample, view)
    t_vals = arr.coords[:, 0].numpy()
    vis = arr.channels.numpy()
    mask = None if arr.mask is None else arr.mask.numpy()
    return t_vals, vis, mask


def _extract_sparse_curves(
    data_list: list[dict[str, torch.Tensor]],
    split_name: str,
    id_offset: int = 0,
    missing_values: str = "interpolate",
    axis: str = "dap",
    view: "View | None" = None,
    fit_indices: list[int] | None = None,
) -> pd.DataFrame:
    """Convert G2FDataset sample dicts to tall DataFrame for R fdapace.

    Samples are sorted by content hash before ID assignment so that IDs are
    ordering-invariant (content-addressed). The hash is always over the
    native ``dap`` axis (axis-independent), so the row order — and the
    inverse permutation used to realign R's output — is stable regardless
    of which time axis / view is used.

    Parameters
    ----------
    axis : {"dap", "gdd", "agdd"}
        Time axis for the trivial (``view is None``) fast path: the
        per-sample coordinate that becomes the fdapace time column ``t``.
    view : View | None
        A non-trivial FPCA view (dedup / warp / extra-channel). When given,
        the curve is assembled via ``assemble_view`` — coords[:, 0] is the
        time column ``t`` and each ``channels`` column is an independent FPCA
        block (``vi_index``). Overrides ``axis`` (the view's coords carry it).
    fit_indices : list[int] | None
        Column indices into each sample's VI/channel matrix selecting the
        *compute set* (the ``fit_vars`` VIs). When given, ``values`` and
        ``mask`` are column-sliced to these blocks **before** the tall frame
        is built, so R only fits FPCA on those VIs. ``None`` (default) keeps
        every block — byte-identical to the pre-``fit_vars`` path. The slice
        happens after :func:`_sample_curve_arrays`, so the content-hash sort
        order (computed from the un-sliced samples) is unaffected.
    missing_values : {"interpolate", "skip"}
        How to treat cells that were originally NaN (and have since been
        linearly interpolated by the reader).

        - ``"interpolate"`` (current scheme): pass every cell through to
          fdapace. The interpolated values are indistinguishable from
          true observations.
        - ``"skip"`` (the R reference's scheme): drop cells where the per-cell mask
          is True before emitting rows. fdapace only sees originally-real
          observations. Falls back to ``"interpolate"`` semantics for any
          sample/view with no mask.

    Returns DataFrame with columns:
    ``sample_id, vi_index, t, value, yield, split``
    """
    if missing_values not in MISSING_VALUE_MODES:
        raise ValueError(
            f"missing_values must be one of {MISSING_VALUE_MODES}, "
            f"got {missing_values!r}"
        )

    if not data_list:
        return pd.DataFrame(
            columns=["sample_id", "vi_index", "t", "value", "yield", "split"]
        )

    order = sorted(
        range(len(data_list)), key=lambda i: _sample_hash(data_list[i])
    )

    parts = []
    for new_id, orig_idx in enumerate(order):
        sample = data_list[orig_idx]
        daps, vis, nan_mask = _sample_curve_arrays(sample, axis, view)
        yield_val = sample["yield_value"].item()

        # Compute-set slice: keep only the fit_vars VI/channel columns so R
        # fits FPCA on those blocks alone. Done here (post curve-assembly,
        # post hash-sort) so the row order is identical to the full path.
        if fit_indices is not None:
            vis = vis[:, fit_indices]
            if nan_mask is not None:
                nan_mask = nan_mask[:, fit_indices]

        n_t, n_vis = vis.shape
        dap_col = np.tile(daps, n_vis)
        vi_col = np.repeat(np.arange(n_vis), n_t)
        val_col = vis.T.ravel()

        # Always drop cells whose TIME coordinate is non-finite (NaN). This
        # is the R reference's `is.finite(t)` filter: a VI obs on a DAP that the
        # deduped clean-AGDD axis collapsed away (axis_source `vi_dedup`)
        # carries NaN AGDD and must be excluded, exactly as the reference
        # `left_join(dap.gdd)` NA does. No-op when every time value is
        # finite (the default DAP / raw-AGDD path), so those stay
        # byte-identical.
        keep = np.isfinite(dap_col)
        if missing_values == "skip" and nan_mask is not None:
            # Mask is shape [T, n_blocks]; ravel in the same block-major
            # order as `vis.T.ravel()` so positions align.
            if nan_mask.shape != vis.shape:
                raise ValueError(
                    f"vi_nan_mask shape {nan_mask.shape} does not match "
                    f"channels shape {vis.shape} "
                    f"(sample id={new_id + id_offset})."
                )
            keep &= ~nan_mask.T.ravel()
        if not keep.all():
            dap_col = dap_col[keep]
            vi_col = vi_col[keep]
            val_col = val_col[keep]
        if val_col.size == 0:
            # No real observations left (every cell interpolated, or every
            # timepoint collapsed) — skip; fdapace errors on an empty Ly.
            # The skipped sample still consumed its enumerate() id, so id
            # allocation must advance by max()+1, not by emitted-id counts
            # (see the next_id updates in _run_r_fpca). The row-count
            # mismatch this creates is caught by _align_raw_scores_to_input;
            # log the identity here so that error is diagnosable.
            logger.warning(
                "vi_fpca: sample %d in split %r has no surviving "
                "observations (all timepoints non-finite/masked); it is "
                "excluded from the R input and will trip the downstream "
                "score/input alignment check.",
                new_id + id_offset,
                split_name,
            )
            continue

        parts.append(
            pd.DataFrame(
                {
                    "sample_id": new_id + id_offset,
                    "vi_index": vi_col,
                    "t": dap_col,
                    "value": val_col,
                    "yield": yield_val,
                    "split": split_name,
                }
            )
        )

    if not parts:
        return pd.DataFrame(
            columns=["sample_id", "vi_index", "t", "value", "yield", "split"]
        )
    return pd.concat(parts, ignore_index=True)


# ── FPC score reading ───────────────────────────────────────────


def _read_fpc_scores_all(
    csv_path: str,
    n_vis: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Read FPC scores CSV and return ALL available components.

    Returns dict mapping ``split_name -> (X, sample_ids, max_k)`` where
    ``X`` has shape ``(n_vis * max_k)`` per sample and ``max_k`` is the
    maximum number of FPC columns present in the CSV.

    ``n_vis`` is the number of independent FPCA blocks (``vi_index`` values).
    **B2 fix:** it is derived from the emitted tall frame
    (``max(vi_index) + 1``), *not* the hardcoded ``N_VIS=37``. A view that
    selects a different channel count (a VI subset at source, or an
    axis-as-channel extra block) yields a different block count and previously
    overflowed/misreshaped the score matrix. An explicit ``n_vis`` may still
    be passed to override (e.g. to assert a fixed width on an empty split).
    """
    df = pd.read_csv(csv_path)

    if n_vis is None:
        # Derive the block count from the data: vi_index is 0-based and dense.
        n_vis = (
            int(df["vi_index"].max()) + 1 if len(df) > 0 else 0
        )

    # Discover how many FPC columns the R script output
    fpc_cols = sorted(
        [c for c in df.columns if c.startswith("FPC")],
        key=lambda c: int(c[3:]),
    )
    max_k = len(fpc_cols)
    if max_k == 0:
        raise ValueError(f"No FPC columns found in {csv_path}")

    result = {}
    for split_name in df["split"].unique():
        split_df = df[df["split"] == split_name]
        sample_ids = sorted(split_df["sample_id"].unique())
        n_samples = len(sample_ids)

        X = np.zeros((n_samples, n_vis * max_k))
        for i, sid in enumerate(sample_ids):
            sample_df = split_df[split_df["sample_id"] == sid].sort_values(
                "vi_index"
            )
            for _, row in sample_df.iterrows():
                vi = int(row["vi_index"])
                for k, col in enumerate(fpc_cols):
                    X[i, vi * max_k + k] = row[col]

        result[split_name] = (X, np.array(sample_ids), max_k)

    return result


def _slice_fpc_scores(
    X_all: np.ndarray,
    max_k: int,
    n_components: int,
    n_vis: int = N_VIS,
) -> np.ndarray:
    """Slice all-component score matrix to ``n_components`` per VI.

    ``X_all`` has shape ``(N, n_vis * max_k)``.  Returns ``(N, n_vis * n_components)``.
    When ``n_components > max_k``, extra columns are zero (already zero-padded by R).
    """
    if n_components == max_k:
        return X_all
    if n_components > max_k:
        warnings.warn(
            f"Requested n_components={n_components} but R produced only "
            f"{max_k} — extra components will be zero-padded.",
            stacklevel=2,
        )
        # Pad each VI block with zeros
        N = X_all.shape[0]
        X_out = np.zeros((N, n_vis * n_components))
        for vi in range(n_vis):
            src = X_all[:, vi * max_k : vi * max_k + max_k]
            X_out[:, vi * n_components : vi * n_components + max_k] = src
        return X_out

    # n_components < max_k — slice each VI block
    N = X_all.shape[0]
    X_out = np.zeros((N, n_vis * n_components))
    for vi in range(n_vis):
        src = X_all[:, vi * max_k : vi * max_k + n_components]
        X_out[:, vi * n_components : vi * n_components + n_components] = src
    return X_out


# ── Cache key computation ───────────────────────────────────────


def _axis_content_hash(
    data_list: list[dict[str, torch.Tensor]],
    axis: str,
) -> str:
    """Order-independent SHA-256 over the per-sample ``axis`` arrays.

    The native ``dap`` axis is already covered by ``_sample_hash`` (which
    hashes ``sample["dap"]``), so this is only needed for derived axes
    (``gdd`` / ``agdd``): their values depend on the env→weather mapping,
    which the dap/channels/mask/yield content hash does **not** capture.
    Folding them into the key keeps dap-axis vs agdd-axis fits in
    distinct cache entries.
    """
    per_sample = sorted(
        hashlib.sha256(s[axis].numpy().tobytes()).hexdigest()
        for s in data_list
    )
    h = hashlib.sha256()
    for fp in per_sample:
        h.update(fp.encode())
    return h.hexdigest()


def _view_fingerprint(view: "View") -> str:
    """Stable JSON fingerprint of a non-trivial FPCA view (coords/channels/
    transform/args/target) for the cache key."""
    return json.dumps(
        {
            "coords": list(view.coords),
            "channels": list(view.channels),
            "transform": view.transform,
            "transform_args": view.transform_args,
            "target": view.target,
        },
        sort_keys=True,
    )


def compute_vi_fpca_cache_key(
    train_data: list[dict[str, torch.Tensor]],
    missing_values: str = "skip",
    axis: str = "dap",
    view: "View | None" = None,
    fit_vars: list[str] | None = None,
) -> str:
    """Content-addressed cache key for VI FPCA.

    Covers the training split content only. Excludes ``n_components``
    and ``vi_subset`` (both post-load slices) and test/val data (FPCA
    is fitted on train only). ``fit_vars`` (the compute set) IS folded in
    when set: it column-slices the channels before the R fit, so it changes
    what the cache stores. The full-channel content hash above does NOT
    capture that slice (the samples are hashed un-sliced), so this tag is
    load-bearing — distinct ``fit_vars`` must key distinct entries.
    ``None`` adds no bytes, keeping the historical full-fit key
    byte-identical so existing caches stay valid. ``missing_values`` is
    included because
    ``"skip"`` and ``"interpolate"`` produce different fdapace inputs
    on the same train data and therefore distinct fitted bases — the
    two modes must land in separate cache entries. ``axis`` is included
    (name + value fingerprint for derived axes) so dap-axis, agdd-axis
    and gdd-axis fits never collide on the same entry.

    ``view`` is the non-trivial FPCA view (dedup/warp/extra-channel). When
    given, its fingerprint **plus** the per-sample value hash of every
    derived axis it references (as a coord *or* an extra channel) are folded
    in, so e.g. ``channels:[vi.*, agdd]`` and a ``warp{to:agdd}`` view land in
    distinct entries. A trivial view (``coords:[axis] channels:[vi.*]
    identity``) is passed as ``None`` and keeps the byte-identical axis-only
    key, so existing dap/agdd cache entries are not invalidated.
    """
    if missing_values not in MISSING_VALUE_MODES:
        raise ValueError(
            f"missing_values must be one of {MISSING_VALUE_MODES}, "
            f"got {missing_values!r}"
        )
    h = hashlib.sha256()
    h.update(b"vi_fpca")
    h.update(_split_content_hash(train_data).encode())
    h.update(b"|missing=")
    h.update(missing_values.encode())
    # Always tag the axis name so every entry is self-describing on disk
    # (dap included). Derived axes (gdd/agdd) additionally fold in their
    # per-sample values, which depend on the env→weather mapping and are
    # NOT captured by the content hash above; the native dap values are
    # already in `_split_content_hash`, so dap needs no value fold-in.
    h.update(b"|axis=")
    h.update(axis.encode())
    if axis in DERIVED_AXES and train_data:
        h.update(b"|axis_values=")
        h.update(_axis_content_hash(train_data, axis).encode())
    if view is not None:
        h.update(b"|view=")
        h.update(_view_fingerprint(view).encode())
        if train_data:
            referenced = set(view.coords) | set(view.channels)
            for ax in sorted(a for a in referenced if a in DERIVED_AXES):
                h.update(f"|axisval:{ax}=".encode())
                h.update(_axis_content_hash(train_data, ax).encode())
    # Compute set: restricting the channels fed to R changes what the cache
    # stores, so it must key a distinct entry. Folded in ONLY when set, so
    # the default full-fit (fit_vars=None) key stays byte-identical to the
    # historical key — existing caches remain valid. Mirrors how
    # `weather_fpca` tags its `fit_vars`.
    if fit_vars is not None:
        h.update(b"|fit_vars=")
        h.update(",".join(sorted(fit_vars)).encode())
    return h.hexdigest()


# ── Explicit FIT-set support (D3) ───────────────────────────────


def _fit_subset(
    train_data: list[dict[str, torch.Tensor]],
    fit_flags: np.ndarray | None,
) -> list[dict[str, torch.Tensor]]:
    """The FIT-set rows of a train split — the rows the FPCA basis fits on.

    ``fit_flags`` is a boolean array aligned to ``train_data`` positions
    (``True`` = FIT). ``None`` means *no fit-mask*: the FIT set is the
    entire train split, i.e. behaviour byte-identical to today (the D3
    regression gate). Non-FIT (``OBSERVE``) rows are projected, not fit.
    """
    if fit_flags is None:
        return train_data
    if len(fit_flags) != len(train_data):
        raise ValueError(
            f"fit_flags length {len(fit_flags)} != train split size "
            f"{len(train_data)} — cannot align the VI-FPCA fit set."
        )
    return [s for s, keep in zip(train_data, fit_flags) if keep]


def _observe_subset(
    train_data: list[dict[str, torch.Tensor]],
    fit_flags: np.ndarray | None,
) -> list[dict[str, torch.Tensor]]:
    """The in-train-split ``OBSERVE`` rows (observed by the model but
    excluded from the FPCA basis and projected). Empty when ``fit_flags``
    is ``None`` (no fit-mask) or all-``True``.
    """
    if fit_flags is None:
        return []
    return [s for s, keep in zip(train_data, fit_flags) if not keep]


def vi_fpca_fit_cache_key(
    train_data: list[dict[str, torch.Tensor]],
    fit_flags: np.ndarray | None,
    missing_values: str,
    axis: str = "dap",
    view: "View | None" = None,
    fit_vars: list[str] | None = None,
) -> str:
    """Single source of truth for the FIT-set-addressed cache key (D3).

    Called from **all** sites that compute the key
    (``fit_and_transform_all`` and the ``cache_key`` lifecycle hook); the
    third site — the cache write in ``_save_cache`` — receives the key
    computed by ``fit_and_transform_all``, so routing both compute sites
    through this helper keeps the three provably in sync (no silent
    stale-cache reuse). The key hashes the **FIT set**, not the train
    split: when ``fit_flags is None`` the FIT set *is* the train split, so
    the key is byte-identical to today's.
    """
    return compute_vi_fpca_cache_key(
        _fit_subset(train_data, fit_flags),
        missing_values=missing_values,
        axis=axis,
        view=view,
        fit_vars=fit_vars,
    )


def _merge_observe_into_train(
    aligned: dict[str, tuple[np.ndarray, np.ndarray, int]],
    fit_flags: np.ndarray | None,
    n_train: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Recombine the R ``"train"`` (FIT) and ``"observe"`` (OBSERVE) score
    blocks into a single dataset ``"train"`` block, in train-split order.

    The FIT rows were handed to R as ``split=="train"`` (the only split R
    fits on) and the OBSERVE rows as ``split=="observe"`` (projected). R
    projects *both* via CE/BLUP (``fpca_compute.R`` projects every split
    uniformly), so this just scatters the two input-order blocks back into
    the original train positions using ``fit_flags``. No-op when there are
    no OBSERVE rows.
    """
    if fit_flags is None or "observe" not in aligned:
        return aligned
    train_X, _, max_k = aligned["train"]
    obs_X, _, _ = aligned["observe"]
    full = np.empty((n_train, train_X.shape[1]), dtype=train_X.dtype)
    fi = oi = 0
    for p in range(n_train):
        if fit_flags[p]:
            full[p] = train_X[fi]
            fi += 1
        else:
            full[p] = obs_X[oi]
            oi += 1
    out = {k: v for k, v in aligned.items() if k != "observe"}
    out["train"] = (full, np.arange(n_train), max_k)
    return out


def _slice_full_to_subset_k(
    X_all: np.ndarray,
    max_k: int,
    n_components: int,
    n_vis_full: int,
    subset_indices: list[int],
) -> np.ndarray:
    """Two-axis slice: VI subset (along vi block) + K (along component).

    ``X_all`` has shape ``(N, n_vis_full * max_k)`` laid out as the
    concatenation of per-VI score blocks. Reshape to
    ``(N, n_vis_full, max_k)``, take the subset rows along axis 1, then
    take the first ``n_components`` columns along axis 2. Mirrors
    ``_slice_fpc_scores``'s zero-pad behavior when
    ``n_components > max_k``.
    """
    N = X_all.shape[0]
    block = X_all.reshape(N, n_vis_full, max_k)
    block = block[:, subset_indices, :]
    n_subset = len(subset_indices)
    if n_components == max_k:
        return block.reshape(N, n_subset * max_k).copy()
    if n_components < max_k:
        return block[:, :, :n_components].reshape(
            N, n_subset * n_components
        ).copy()
    # n_components > max_k — pad each subset block with zeros.
    warnings.warn(
        f"Requested n_components={n_components} but R produced only "
        f"{max_k} — extra components will be zero-padded.",
        stacklevel=3,
    )
    out = np.zeros((N, n_subset * n_components), dtype=block.dtype)
    for j in range(n_subset):
        out[:, j * n_components : j * n_components + max_k] = block[:, j, :]
    return out


# ── VIFPCAProcessor ─────────────────────────────────────────────


@register_processor
class VIFPCAProcessor(BaseProcessor):
    """Compute FPC scores from sparse VI curves via R fdapace.

    Fitted on train split only.  Projects val/test via CE/BLUP.
    Results cached with content-addressed keys (``n_components`` excluded
    from key — applied as post-load slice).

    Parameters
    ----------
    n_components : int
        Number of FPC scores to keep per VI.  Applied as a post-load slice.
    cache_dir : str | None
        Directory for caching FPC scores and fitted R models.
        If None, caching is disabled.
    vi_subset : list[str] | None
        The **emit set** — vegetation indices whose scores are sliced out
        and handed downstream. ``None`` (default) emits every VI in the fit
        set. Pure **post-load slice** over the cached fit (NOT part of the
        cache key), so different emit subsets reuse the same fit. Must be a
        subset of the fit set (``vi_subset ⊆ fit_vars``). Per-VI FPCA
        independence makes the sliced output bit-identical to a
        fit-on-subset path.
    fit_vars : list[str] | None
        The **compute set** — VIs the R fdapace fit actually runs on.
        ``None`` (default) fits every VI in the per-sample tensor (the
        historical behaviour; the cache stores the full per-split score
        matrix and any emit subset reuses it). When set (e.g. ``[NGRDI]``),
        the channels tensor is column-sliced to these VIs **before** the
        tall CSV is built, so R only fits those blocks — a pure compute/cost
        saving. Mirrors ``weather_fpca.fit_vars``. Folded into the cache key
        (``None`` keeps the historical key byte-identical). By per-VI FPCA
        independence the scores of any VI are bit-identical whether it was
        fit alone or alongside others — this never changes outputs, only
        which VIs get computed and which cache entry is used.
    missing_values : {"skip", "interpolate"}
        How to handle VI cells that were originally NaN. ``"skip"``
        (default) drops them before fdapace fits — matches the R reference,
        which feeds tall-format observations with no NaN
        rows. ``"interpolate"`` keeps the linearly-interpolated values
        the reader produced and passes them to fdapace as if observed.
        The two modes produce different fitted bases and live under
        separate cache entries.
    """

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "vi_fpca"
    priority: ClassVar[int] = 1

    @classmethod
    def from_config(cls, config: dict) -> "VIFPCAProcessor":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            n_components=config.get("n_components", 4),
            cache_dir=config.get("cache_dir"),
            vi_subset=config.get("vi_subset"),
            fit_vars=config.get("fit_vars"),
            missing_values=config.get("missing_values", "skip"),
            axis=config.get("axis", "dap"),
            channels=config.get("channels"),
            transform=config.get("transform", "identity"),
            transform_args=config.get("transform_args"),
        )

    def __init__(
        self,
        n_components: int,
        cache_dir: str | None = None,
        vi_subset: list[str] | None = None,
        fit_vars: list[str] | None = None,
        missing_values: str = "skip",
        axis: str = "dap",
        channels: list[str] | None = None,
        transform: str = "identity",
        transform_args: dict | None = None,
    ):
        if n_components < 1:
            raise ValueError(
                f"n_components must be >= 1, got {n_components}"
            )
        if missing_values not in MISSING_VALUE_MODES:
            raise ValueError(
                f"missing_values must be one of {MISSING_VALUE_MODES}, "
                f"got {missing_values!r}"
            )
        if axis not in AXIS_NAMES:
            raise ValueError(
                f"axis must be one of {AXIS_NAMES}, got {axis!r}. "
                f"Non-'dap' axes require the axis_source processor."
            )
        self.n_components = n_components
        self.cache_dir = cache_dir
        self.vi_subset = list(vi_subset) if vi_subset is not None else None
        # Compute set (`fit_vars`) vs emit set (`vi_subset`), mirroring
        # `weather_fpca`. `fit_vars=None` → fit every VI in the per-sample
        # tensor (historical behaviour). When set, the channels are
        # column-sliced to these VIs before the R fit. `vi_subset` then
        # post-load-slices the (already restricted) fit. Enforce
        # `vi_subset ⊆ fit_vars` early when both are explicit so a misconfig
        # fails at construction, not after a multi-hour fit.
        if fit_vars is not None and len(fit_vars) == 0:
            raise ValueError(
                "fit_vars must be a non-empty list or None "
                "(None = fit every VI in the dataset)."
            )
        self.fit_vars = list(fit_vars) if fit_vars is not None else None
        if self.fit_vars is not None and self.vi_subset is not None:
            not_fit = [v for v in self.vi_subset if v not in set(self.fit_vars)]
            if not_fit:
                raise ValueError(
                    f"vi_subset {not_fit} are not in the fit set "
                    f"fit_vars={sorted(self.fit_vars)} — the emit set "
                    f"(vi_subset) must be a subset of the fit set (you can "
                    f"only slice out a VI that was fit)."
                )
        self.missing_values = missing_values
        self.axis = axis

        # View routing. The default config — single axis coord,
        # all VI channels, identity transform — is a *trivial* view: it takes
        # the legacy fast path (`self.view = None`) which is byte-identical to
        # the pre-View bridge and keeps existing dap/agdd cache entries. Any
        # non-trivial config (extra channels, dedup, warp) builds a `View` and
        # routes the curve through `assemble_view`, so dedup/warp/extra-FPCA
        # blocks flow into the tall frame. ``coords`` is always ``[axis]``.
        # NB: ``self.transform`` would shadow ``BaseProcessor.transform`` (the
        # lifecycle method), so the view-transform name lives on
        # ``self.view_transform``.
        self.channels = list(channels) if channels else ["vi.*"]
        self.view_transform = transform
        self.transform_args = dict(transform_args) if transform_args else {}
        is_trivial = (
            self.channels == ["vi.*"]
            and self.view_transform == "identity"
            and not self.transform_args
        )
        self.view: "View | None" = (
            None
            if is_trivial
            else View(
                name="vi_fpca",
                coords=[self.axis],
                channels=self.channels,
                transform=self.view_transform,
                transform_args=self.transform_args,
                target="fpca",
            )
        )

        # Fitted state
        self._max_k: int | None = None
        self._scores: dict[str, np.ndarray] | None = None
        self._last_cache_key: str | None = None
        self._n_available: dict[str, int] | None = None
        # Canonical VI ordering R fit on (every VI in the per-sample
        # tensor). Populated at fit time from `ctx.vi_names()` or from
        # the metadata.json on cache load. `vi_subset` is resolved
        # against this list at slice time.
        self._vi_names_full: list[str] | None = None

    # ── Subset resolution (post-load slice over the cached full fit) ──

    @property
    def _n_vis_full(self) -> int:
        if self._vi_names_full is None:
            raise RuntimeError(
                "VIFPCAProcessor: full VI list not yet known. Call "
                "fit_and_transform_all() or load_fitted() first."
            )
        return len(self._vi_names_full)

    @property
    def vi_names(self) -> list[str]:
        """Resolved subset (preserves canonical alphabetic order)."""
        if self._vi_names_full is None:
            raise RuntimeError(
                "VIFPCAProcessor: full VI list not yet known."
            )
        if self.vi_subset is None:
            return list(self._vi_names_full)
        requested = set(self.vi_subset)
        return [v for v in self._vi_names_full if v in requested]

    @property
    def n_vis(self) -> int:
        """Subset size — drives feature_dims."""
        return len(self.vi_names)

    def _resolve_subset_indices(self) -> list[int]:
        """Indices into ``_vi_names_full`` for the requested subset.

        Validates that every requested VI name is present.
        ``vi_subset=None`` returns every index in canonical order.
        """
        if self._vi_names_full is None:
            raise RuntimeError(
                "VIFPCAProcessor: full VI list not yet known."
            )
        if self.vi_subset is None:
            return list(range(len(self._vi_names_full)))
        idx_by_name = {n: i for i, n in enumerate(self._vi_names_full)}
        unknown = [v for v in self.vi_subset if v not in idx_by_name]
        if unknown:
            raise ValueError(
                f"vi_subset names not found in dataset: {sorted(unknown)}. "
                f"Available: {self._vi_names_full}"
            )
        return [idx_by_name[v] for v in self._vi_names_full
                if v in set(self.vi_subset)]

    def fit_and_transform_all(
        self,
        train_data: list[dict[str, torch.Tensor]],
        val_data: list[dict[str, torch.Tensor]] | None = None,
        test_data: list[dict[str, torch.Tensor]] | None = None,
        vi_names: list[str] | None = None,
        fit_flags: np.ndarray | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Fit FPCA on the FIT set, project all other rows via CE/BLUP.

        Returns ``{split_name: {"vi_fpc_scores": array(N, n_subset * n_components)}}``
        for each non-empty split, where ``n_subset = len(vi_subset or
        all_vis)``.

        Internally R receives samples in content-hash order (see
        ``_extract_sparse_curves``) so that ``sample_id`` assignment is
        order-invariant and the cache key is content-addressed. Before
        returning, scores are permuted back to the **input list order**
        so that ``derived_features`` attached to ``data_list[i]``
        actually corresponds to the i-th input sample.

        ``vi_names`` is the canonical list naming each column of
        ``channels[:, j]`` and is used to resolve
        ``self.vi_subset`` at slice time. ``None`` falls back to
        synthetic ``["VI_0", "VI_1", ...]`` names sized from the data;
        the orchestrator-driven path passes
        ``ctx.vi_names()``.

        ``fit_flags`` (D3) is a boolean array over ``train_data``
        positions selecting the FIT set (the rows the FPCA basis fits
        on). ``None`` means the whole train split is the FIT set
        (byte-identical to today). The non-FIT (``OBSERVE``) train rows
        are projected onto the FIT basis and merged back into the
        ``"train"`` result in train-split order.
        """
        cache_key = vi_fpca_fit_cache_key(
            train_data, fit_flags, self.missing_values, self.axis, self.view,
            fit_vars=self.fit_vars,
        )
        self._last_cache_key = cache_key

        fit_data = _fit_subset(train_data, fit_flags)
        observe_data = _observe_subset(train_data, fit_flags)

        # Payload-identity digests: the cache KEY is FIT-only (predict mode
        # depends on that), but the cached payload also carries observe/val/
        # test score blocks. A hit must therefore verify those splits'
        # content too — a same-size split with different composition would
        # otherwise be silently mis-permuted by the hash-order alignment
        # below (the stored ids are R-sequential, not content-addressed).
        split_digests = {
            "train": _split_content_hash(fit_data),
            "observe": (
                _split_content_hash(observe_data) if observe_data else None
            ),
            "val": _split_content_hash(val_data) if val_data else None,
            "test": _split_content_hash(test_data) if test_data else None,
        }

        # Try cache
        cached = self._load_cache(cache_key, split_digests=split_digests)
        if cached is not None:
            logger.info("VI FPCA cache hit — skipping R subprocess.")
            raw_scores = cached
        else:
            raw_scores = self._run_fpca(
                train_data, val_data, test_data, fit_flags=fit_flags,
                vi_names=vi_names,
            )
            self._set_vi_names_from_data(train_data, vi_names)
            if self.cache_dir is not None:
                self._save_cache(
                    cache_key, raw_scores, split_digests=split_digests
                )

        # Cache hit didn't go through `_save_cache`, so make sure
        # `_vi_names_full` is set (read from cache metadata in
        # `_load_cache`, or — if the cache predates the metadata field —
        # fall back to the same data-derived path).
        if self._vi_names_full is None:
            self._set_vi_names_from_data(train_data, vi_names)

        # Permute every R split's X from hash order back to input order so
        # that downstream consumers can index features by input position.
        # R sees the FIT rows as "train" and the OBSERVE rows as
        # "observe"; both are aligned to their own input lists, then
        # merged back into the dataset "train" split below.
        aligned = _align_raw_scores_to_input(
            raw_scores,
            {
                "train": fit_data,
                "observe": observe_data or None,
                "val": val_data,
                "test": test_data,
            },
        )
        aligned = _merge_observe_into_train(
            aligned, fit_flags, n_train=len(train_data)
        )
        return self._build_result(aligned)

    def _derive_full_names_fit(
        self,
        train_data: list[dict[str, torch.Tensor]],
        vi_names: list[str] | None,
    ) -> list[str]:
        """The FULL (pre-``fit_vars``) canonical block-name list at fit time.

        Exactly the historical ``_set_vi_names_from_data`` derivation: for a
        non-trivial view the blocks are the **assembled** channels'
        ``channel_names``; for the trivial path the names come from the
        passed ``vi_names`` (``ctx.vi_names()``), falling back to synthetic
        ``VI_i`` only when none are supplied. Keeping this
        unchanged is what makes ``fit_vars=None`` byte-identical to today.
        """
        if not train_data:
            return list(vi_names or [])
        if self.view is not None:
            return list(assemble_view(train_data[0], self.view).channel_names)
        n_vis = train_data[0]["channels"].shape[1]
        if vi_names is not None:
            if len(vi_names) != n_vis:
                raise ValueError(
                    f"vi_names length {len(vi_names)} does not match VI "
                    f"axis of per-sample tensor ({n_vis})."
                )
            return list(vi_names)
        return [f"VI_{i}" for i in range(n_vis)]

    def _fit_indices(self, full_names: list[str]) -> list[int]:
        """Column indices (into ``full_names``) of the ``fit_vars`` compute
        set, in canonical (``full_names``) order.

        ``fit_vars=None`` returns every index (a no-op slice). Every
        requested ``fit_vars`` name must be present, else a loud error.
        """
        if self.fit_vars is None:
            return list(range(len(full_names)))
        idx_by_name = {n: i for i, n in enumerate(full_names)}
        unknown = [v for v in self.fit_vars if v not in idx_by_name]
        if unknown:
            raise ValueError(
                f"vi_fpca.fit_vars names not found in dataset: "
                f"{sorted(unknown)}. Available: {full_names}"
            )
        requested = set(self.fit_vars)
        return [i for i, n in enumerate(full_names) if n in requested]

    def _split_block_names(
        self, split_data: list[dict[str, torch.Tensor]],
    ) -> list[str]:
        """The FULL block-name list for a split at PREDICT time.

        Unlike the fit path, predict mode has no ``vi_names`` hint, so the
        names come from the sample's persisted ``channel_names`` (or the
        assembled view's). Used only to map the loaded fit set
        (``_vi_names_full``) onto this split's channel columns.
        """
        sample = split_data[0]
        if self.view is not None:
            return list(assemble_view(sample, self.view).channel_names)
        n_vis = sample["channels"].shape[1]
        names = sample.get("channel_names")
        if names is not None and len(names) == n_vis:
            return [str(n) for n in names]
        return [f"VI_{i}" for i in range(n_vis)]

    def _set_vi_names_from_data(
        self,
        train_data: list[dict[str, torch.Tensor]],
        vi_names: list[str] | None,
    ) -> None:
        """Capture the canonical FPCA-block ordering R fit on, for slicing.

        With ``fit_vars`` set this is the **fit set** (the full list filtered
        to ``fit_vars``, canonical order), so ``_vi_names_full[j]`` names the
        j-th block of the (column-sliced) score matrix — the same invariant
        the post-load ``vi_subset`` slice and ``_n_vis_full`` rely on. With
        ``fit_vars=None`` it is the full list, byte-identical to before.
        """
        full = self._derive_full_names_fit(train_data, vi_names)
        if self.fit_vars is None or not full:
            self._vi_names_full = full
            return
        idx = self._fit_indices(full)
        self._vi_names_full = [full[i] for i in idx]

    @property
    def feature_dims(self) -> dict[str, int]:
        """Output dimensions: ``{"vi_fpc_scores": n_vis_subset * n_components}``.

        Reports the **subset** count (matches the slice shape).
        """
        if self._vi_names_full is None:
            return {"vi_fpc_scores": N_VIS * self.n_components}
        return {"vi_fpc_scores": self.n_vis * self.n_components}

    @property
    def max_k(self) -> int | None:
        """Maximum number of components available from R (after fit)."""
        return self._max_k

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Run R fdapace on train (+val+test if present) in one
        subprocess; stash sliced per-split scores for `transform`.

        Matches the existing `fit_and_transform_all` path — the R train
        mode call already projects every split it sees, so calling it
        once with all three splits is the minimum-cost cold-fit.
        Predict-mode evaluation paths use `load_fitted` + `transform`,
        which re-runs R per split.
        """
        train_data = ctx.split_data("train")
        val_data = ctx.split_data("val") or None
        test_data = ctx.split_data("test") or None
        self._fitted_per_split = self.fit_and_transform_all(
            train_data, val_data, test_data,
            vi_names=ctx.vi_names() or None,
            fit_flags=ctx.fit_flags_for_split("train"),
        )

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """Return this split's pre-computed scores from `fit`, or run R
        in predict mode against the loaded fdapace basis.
        """
        split_data = ctx.split_data(split)
        if not split_data:
            return None
        if hasattr(self, "_fitted_per_split"):
            features = self._fitted_per_split.get(split)
            if features is not None:
                return features
        # Predict-mode (post-load_fitted): R subprocess per split.
        return self.transform_split(split_data, split)

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Recompute the FIT-set-addressed key from the train split.

        Routes through ``vi_fpca_fit_cache_key`` — the same helper
        ``fit_and_transform_all`` uses — so both compute sites land on the
        same cache entry under any ``fit_mask`` (D3 no-stale-cache gate).
        """
        train_data = ctx.split_data("train")
        return vi_fpca_fit_cache_key(
            train_data,
            ctx.fit_flags_for_split("train"),
            self.missing_values,
            self.axis,
            self.view,
            fit_vars=self.fit_vars,
        )

    def save_cache(self, cache_dir: str, key: str) -> None:
        """No-op: the underlying cache write happens inside `_run_fpca`
        when `cache_dir` is configured at construction.
        """

    # ── Private: R execution ────────────────────────────────────

    def _run_fpca(
        self,
        train_data: list[dict[str, torch.Tensor]],
        val_data: list[dict[str, torch.Tensor]] | None,
        test_data: list[dict[str, torch.Tensor]] | None,
        fit_flags: np.ndarray | None = None,
        vi_names: list[str] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
        """Export curves, call R, read scores.

        R fits the FPCA basis on rows tagged ``split=="train"`` and
        projects every other split via CE/BLUP. To realize the D3
        fit-mask without touching the R script, the FIT rows are tagged
        ``"train"`` and the in-train-split ``OBSERVE`` rows ``"observe"``;
        the caller merges them back into the dataset ``"train"`` split.
        When ``fit_flags is None`` the FIT set is the whole train split
        and no ``"observe"`` split is emitted (byte-identical to today).

        ``vi_names`` (``ctx.vi_names()``) names ``channels[:, j]`` and is
        used only to resolve ``fit_vars`` → the compute-set column indices.
        The same indices slice every split (FIT/OBSERVE/val/test) so all
        share one block ordering. ``fit_vars=None`` → ``fit_indices=None``
        → no slicing (byte-identical to today).
        """
        fit_data = _fit_subset(train_data, fit_flags)
        observe_data = _observe_subset(train_data, fit_flags)
        # Compute-set columns: resolve once from the full block names so
        # every split's tall frame carries the same fit_vars blocks in the
        # same canonical order as `_vi_names_full` (set by the caller).
        fit_indices: list[int] | None = None
        if self.fit_vars is not None:
            full_names = self._derive_full_names_fit(train_data, vi_names)
            fit_indices = self._fit_indices(full_names)
        tmpdir = tempfile.mkdtemp(prefix="vi_fpca_")
        try:
            input_csv = os.path.join(tmpdir, "sparse_curves.csv")
            output_csv = os.path.join(tmpdir, "fpc_scores.csv")
            model_dir = os.path.join(tmpdir, "models")
            os.makedirs(model_dir)

            # Export sparse curves. FIT rows → "train" (R fits on these);
            # OBSERVE rows → "observe" (projected, then merged into train).
            df_train = _extract_sparse_curves(
                fit_data, "train", id_offset=0,
                missing_values=self.missing_values,
                axis=self.axis,
                view=self.view,
                fit_indices=fit_indices,
            )
            next_id = (
                int(df_train["sample_id"].max()) + 1 if len(df_train) > 0 else 0
            )

            dfs = [df_train]
            if observe_data:
                df_observe = _extract_sparse_curves(
                    observe_data, "observe", id_offset=next_id,
                    missing_values=self.missing_values,
                    axis=self.axis,
                    view=self.view,
                    fit_indices=fit_indices,
                )
                # Advance past the highest consumed id, not the emitted-id
                # count: a fully-masked sample is skipped from the frame but
                # still consumed an id, and a count-based advance would hand
                # its id to the next split (two curves merging in R).
                next_id = (
                    int(df_observe["sample_id"].max()) + 1
                    if len(df_observe) > 0
                    else next_id
                )
                dfs.append(df_observe)
            if val_data:
                df_val = _extract_sparse_curves(
                    val_data, "val", id_offset=next_id,
                    missing_values=self.missing_values,
                    axis=self.axis,
                    view=self.view,
                    fit_indices=fit_indices,
                )
                next_id = (
                    int(df_val["sample_id"].max()) + 1
                    if len(df_val) > 0
                    else next_id
                )
                dfs.append(df_val)
            if test_data:
                df_test = _extract_sparse_curves(
                    test_data, "test", id_offset=next_id,
                    missing_values=self.missing_values,
                    axis=self.axis,
                    view=self.view,
                    fit_indices=fit_indices,
                )
                dfs.append(df_test)

            combined = pd.concat(dfs, ignore_index=True)
            combined.to_csv(input_csv, index=False)

            # Use a large n_components to get all available scores from R.
            # fdapace will produce min(estimated, requested) components and
            # R's extract_scores() pads/slices accordingly.
            r_n_components = 50  # generous upper bound

            self._call_rscript(
                input_csv, output_csv, model_dir, r_n_components
            )

            # Copy fpca_models.rds for predict mode (if caching)
            self._model_dir = model_dir

            # Read all scores
            raw_scores = _read_fpc_scores_all(output_csv)

            # Record max_k
            if raw_scores:
                first = next(iter(raw_scores.values()))
                self._max_k = first[2]

            return raw_scores

        finally:
            # Clean up temp dir but keep models if we need to cache them
            if self.cache_dir is None:
                shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                self._tmpdir = tmpdir

    def _call_rscript(
        self,
        input_csv: str,
        output_csv: str,
        model_dir: str,
        n_components: int,
        mode: str = "train",
    ) -> None:
        """Call R fpca_compute.R in the requested mode.

        ``mode="train"`` fits fdapace on the training rows of
        ``input_csv`` and projects all splits. ``mode="predict"`` loads
        ``fpca_models.rds`` from ``model_dir`` and projects curves in
        ``input_csv`` against the existing basis.
        """
        if mode not in ("train", "predict"):
            raise ValueError(f"mode must be 'train' or 'predict', got {mode!r}")
        r_script = os.path.normpath(os.path.abspath(_R_SCRIPT))
        cmd = [
            "Rscript",
            r_script,
            "--mode",
            mode,
            "--input_csv",
            input_csv,
            "--output_csv",
            output_csv,
            "--model_dir",
            model_dir,
            "--n_components",
            str(n_components),
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout.rstrip())
        if result.stderr:
            logger.warning(result.stderr.rstrip())
        if result.returncode != 0:
            raise RuntimeError(
                f"Rscript failed (return code {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # The R exits 0 even when a per-VI FPCA fit failed (dropped VI) or a
        # per-split scoring call was caught into an all-zero score block
        # (is_zero_padded). Either turns silently into a bogus / all-zero
        # feature column downstream. It writes those markers into
        # fpca_summary.json; fail loud here instead of ingesting them.
        summary_path = os.path.join(model_dir, "fpca_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            failed_vis = summary.get("failed_vis") or []
            zero_padded = summary.get("zero_padded_blocks") or []
            if failed_vis or zero_padded:
                raise RuntimeError(
                    "FPCA "
                    f"{summary.get('mode', mode)} produced degenerate "
                    "features that would become silent all-zero columns "
                    f"(summary: {summary_path}). "
                    f"failed_vis={failed_vis!r} "
                    f"zero_padded_blocks={zero_padded!r}. "
                    "Refusing to continue; inspect the R log above."
                )

    # ── Predict mode: load fitted + transform new splits ─────────

    def load_fitted(self, cache_dir: str, cache_key: str) -> None:
        """Load fitted fdapace model from cache for predict-mode transforms.

        After this call, :meth:`transform_split` can project new sparse
        VI curves onto the fitted basis without re-fitting. Reads
        ``fpca_models.rds`` and ``metadata.json`` from the cache subdir
        written by :meth:`_save_cache`.

        Raises
        ------
        FileNotFoundError
            If the cache subdirectory, ``fpca_models.rds``, or
            ``metadata.json`` is missing.
        """
        cache_path = os.path.join(cache_dir, "vi_fpca", cache_key[:16])
        rds_path = os.path.join(cache_path, "fpca_models.rds")
        meta_path = os.path.join(cache_path, "metadata.json")
        if not os.path.isfile(rds_path):
            raise FileNotFoundError(
                f"VI FPCA cache missing fpca_models.rds at: {rds_path}"
            )
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"VI FPCA cache missing metadata.json at: {meta_path}"
            )

        with open(meta_path) as f:
            meta = json.load(f)
        self._max_k = int(meta["max_k"])
        self._vi_names_full = [str(v) for v in meta["vi_names_full"]]
        cached_mode = meta.get("missing_values")
        if cached_mode is not None and cached_mode != self.missing_values:
            raise ValueError(
                f"VIFPCAProcessor: cache at {cache_path} was fitted with "
                f"missing_values={cached_mode!r}, but processor is "
                f"configured with missing_values={self.missing_values!r}. "
                "Cache keys should have prevented this — check your config."
            )
        self._loaded_model_dir = cache_path
        self._last_cache_key = cache_key
        logger.info("VI FPCA fitted state loaded from %s", cache_path)

    def transform_split(
        self,
        split_data: list[dict[str, torch.Tensor]],
        split_name: str,
    ) -> dict[str, np.ndarray]:
        """Project one split's sparse VI curves onto the loaded FPCA basis.

        Requires :meth:`load_fitted` (or a prior :meth:`fit_and_transform_all`
        with ``cache_dir`` set) to have been called first. Runs R in
        ``predict`` mode against the loaded ``fpca_models.rds`` and
        returns features in *input list order* (not content-hash order).

        Returns an empty dict when ``split_data`` is empty.
        """
        if not split_data:
            n_subset = (
                self.n_vis if self._vi_names_full is not None else N_VIS
            )
            return {
                "vi_fpc_scores": np.zeros((0, n_subset * self.n_components))
            }

        model_dir = getattr(self, "_loaded_model_dir", None)
        if model_dir is None:
            raise RuntimeError(
                "VIFPCAProcessor.transform_split requires load_fitted() "
                "(or fit_and_transform_all with cache_dir set) first."
            )

        # Compute-set columns for predict mode. The fitted basis only knows
        # the fit_vars blocks, so feed R exactly those columns — and in the
        # SAME order as `_vi_names_full` (the loaded fit set) so block j of
        # the projected scores aligns with `_vi_names_full[j]`. We map by
        # NAME (not by the fit-time channel order), which keeps predict
        # correct even if this split's channel ordering differed. Skipped
        # when fit_vars is None (byte-identical to today).
        fit_indices: list[int] | None = None
        if self.fit_vars is not None:
            if self._vi_names_full is None:
                raise RuntimeError(
                    "VIFPCAProcessor.transform_split: fit set unknown — call "
                    "load_fitted() (or fit_and_transform_all) first."
                )
            block_names = self._split_block_names(split_data)
            col_by_name = {n: i for i, n in enumerate(block_names)}
            missing = [n for n in self._vi_names_full if n not in col_by_name]
            if missing:
                raise ValueError(
                    f"vi_fpca.fit_vars blocks {missing} absent from this "
                    f"split's channels {block_names} — cannot project."
                )
            fit_indices = [col_by_name[n] for n in self._vi_names_full]

        tmpdir = tempfile.mkdtemp(prefix="vi_fpca_predict_")
        try:
            input_csv = os.path.join(tmpdir, "sparse_curves.csv")
            output_csv = os.path.join(tmpdir, "fpc_scores.csv")

            df = _extract_sparse_curves(
                split_data, split_name, id_offset=0,
                missing_values=self.missing_values,
                axis=self.axis,
                view=self.view,
                fit_indices=fit_indices,
            )
            df.to_csv(input_csv, index=False)

            # Request more components than we need; the slice happens in
            # Python so that n_components remains a post-load knob.
            r_n_components = max(self._max_k or self.n_components, self.n_components)
            self._call_rscript(
                input_csv, output_csv, model_dir, r_n_components, mode="predict"
            )

            raw = _read_fpc_scores_all(output_csv)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if split_name not in raw:
            raise RuntimeError(
                f"R predict-mode output missing split {split_name!r}; "
                f"got {list(raw.keys())}"
            )

        aligned = _align_raw_scores_to_input(raw, {split_name: split_data})
        X_all, _, max_k = aligned[split_name]
        self._max_k = max(self._max_k or 0, max_k)
        # `_vi_names_full` was populated by `load_fitted` from the cache
        # metadata, so the subset slice has the right canonical context.
        X = _slice_full_to_subset_k(
            X_all, max_k, self.n_components,
            n_vis_full=self._n_vis_full,
            subset_indices=self._resolve_subset_indices(),
        )
        return {"vi_fpc_scores": X}

    # ── Private: cache ──────────────────────────────────────────

    def _cache_subdir(self, cache_key: str) -> str:
        tag = cache_key[:16]
        return os.path.join(self.cache_dir, "vi_fpca", tag)

    def _load_cache(
        self,
        cache_key: str,
        split_digests: dict[str, str | None] | None = None,
    ) -> dict | None:
        """Load cached scores if valid. Returns raw_scores dict or None.

        ``split_digests`` (split name → content digest, ``None`` for an
        absent split) is verified against the entry's stored
        ``split_content`` metadata: the cache key covers only the FIT set,
        so the observe/val/test blocks of the payload must be identity-
        checked here. A mismatch is a miss (recompute), never a
        mis-permuted hit.
        """
        if self.cache_dir is None:
            return None

        cache_path = self._cache_subdir(cache_key)
        fp_path = os.path.join(cache_path, "fingerprint")
        scores_path = os.path.join(cache_path, "fpc_scores.npz")

        if not (os.path.isfile(fp_path) and os.path.isfile(scores_path)):
            return None

        with open(fp_path) as f:
            cached_key = f.read().strip()
        if cached_key != cache_key:
            return None

        data = np.load(scores_path, allow_pickle=True)
        meta_path = os.path.join(cache_path, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)

        if split_digests is not None:
            stored = meta.get("split_content")
            if stored is None:
                logger.warning(
                    "VI FPCA cache entry %s predates payload-identity "
                    "metadata; the observe/val/test score blocks cannot be "
                    "verified against the current splits. Delete the entry "
                    "to refresh it with verifiable metadata.",
                    cache_path,
                )
            else:
                for split, digest in split_digests.items():
                    if stored.get(split) != digest:
                        logger.info(
                            "VI FPCA cache entry %s was built for a "
                            "different %r split composition — treating as "
                            "a miss and recomputing.",
                            cache_path,
                            split,
                        )
                        return None
        max_k = int(meta["max_k"])
        self._max_k = max_k
        # `vi_names_full` is the canonical column ordering R fit on —
        # the subset slice resolves against it.
        self._vi_names_full = [str(v) for v in meta["vi_names_full"]]

        # Reconstruct raw_scores dict (rows are still in content-hash order;
        # caller is responsible for permuting to input order).
        result = {}
        for key in data.keys():
            if key.endswith("_scores"):
                split_name = key[: -len("_scores")]
                ids_key = f"{split_name}_ids"
                X = data[key]
                ids = data[ids_key] if ids_key in data else np.arange(X.shape[0])
                result[split_name] = (X, ids, max_k)

        return result

    def _save_cache(
        self,
        cache_key: str,
        raw_scores: dict,
        split_digests: dict[str, str | None] | None = None,
    ) -> None:
        """Save scores and metadata to cache directory.

        Writes atomically (uniquely-tagged staging dir + rename) so parallel
        SLURM jobs sharing the same cache key cannot corrupt each other.
        """
        cache_path = self._cache_subdir(cache_key)
        # PID alone is unique only WITHIN a node. Two SLURM jobs on
        # different nodes can share a PID, and when they also share a
        # cache key — as env and env-free variants of the same model do,
        # since their upstream FPCA settings are identical — they would
        # write into the SAME staging dir on shared scratch, interleave,
        # and rename a corrupt mixture into place. The uuid makes the
        # staging path globally unique; the PID is kept for legibility
        # when tracing a stray directory back to its job.
        staging_path = cache_path + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:12]}"
        os.makedirs(staging_path, exist_ok=True)

        # Save scores as npz
        arrays = {}
        for split_name, (X, ids, _) in raw_scores.items():
            arrays[f"{split_name}_scores"] = X
            arrays[f"{split_name}_ids"] = ids
        np.savez(os.path.join(staging_path, "fpc_scores.npz"), **arrays)

        # Copy fpca_models.rds if available
        if hasattr(self, "_model_dir"):
            rds_src = os.path.join(self._model_dir, "fpca_models.rds")
            if os.path.isfile(rds_src):
                shutil.copy2(rds_src, os.path.join(staging_path, "fpca_models.rds"))

        # Metadata
        max_k = next(iter(raw_scores.values()))[2] if raw_scores else 0
        meta = {
            "max_k": max_k,
            "n_vis": len(self._vi_names_full or []),
            "vi_names_full": list(self._vi_names_full or []),
            "missing_values": self.missing_values,
            "splits": {
                name: {"n_samples": X.shape[0]}
                for name, (X, _, _) in raw_scores.items()
            },
            # Per-split content digests over the payload's non-FIT blocks
            # (the cache key itself covers only the FIT set) — verified by
            # _load_cache before a hit is served.
            "split_content": {
                name: digest
                for name, digest in (split_digests or {}).items()
                if digest is not None
            },
        }
        with open(os.path.join(staging_path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Fingerprint
        with open(os.path.join(staging_path, "fingerprint"), "w") as f:
            f.write(cache_key)

        # Atomically move staging dir to final path. If a sibling
        # process already placed the final dir, the rename fails —
        # the cache is warm regardless of which process won.
        try:
            os.rename(staging_path, cache_path)
        except OSError:
            # Another process won the race — clean up our staging dir.
            shutil.rmtree(staging_path, ignore_errors=True)

        logger.info("Cached VI FPCA to %s", cache_path)

        # Clean up temp dir
        if hasattr(self, "_tmpdir"):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            del self._tmpdir

    # ── Private: result building ────────────────────────────────

    def _build_result(
        self,
        raw_scores: dict[str, tuple[np.ndarray, np.ndarray, int]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Slice to ``vi_subset`` × ``n_components`` and format output."""
        subset_indices = self._resolve_subset_indices()
        n_vis_full = self._n_vis_full

        result: dict[str, dict[str, np.ndarray]] = {}
        max_k = None
        for split_name, (X_all, _, max_k) in raw_scores.items():
            X_sliced = _slice_full_to_subset_k(
                X_all, max_k, self.n_components,
                n_vis_full=n_vis_full,
                subset_indices=subset_indices,
            )
            result[split_name] = {"vi_fpc_scores": X_sliced}

        self._max_k = max_k if raw_scores else None
        self._scores = {
            split: data["vi_fpc_scores"] for split, data in result.items()
        }
        return result
