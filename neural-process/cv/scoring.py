"""Shared label-based scoring for both pipelines (D2 / D8).

The BGLR/FPCA path (``baselines.fpca_core``) and the DL eval path
(``utils.experiment.utils.evaluate_model``) score the **same** way:
scatter per-row predictions back to ``metadata_df`` positions, join the
per-row ``cv_label``, and compute ``cor``/RMSE once per label over its
masked slice — including the in-sample CV2/CV0 tested-line metrics. This
module is the single home for that logic so the two pipelines can never
drift on the label==position alignment convention (the single most
drift-prone contract in the pipeline).

Both pipelines emit the same artifacts:
  - ``metrics.json`` ``by_metric`` block (one entry per ``cv_label``):
    ``rmse`` / ``pearson_r`` / ``spearman_r`` over the label slice, plus
    Tiezzi et al. (2017)'s within-block inverse-variance weighted correlations
    ``r_w`` (Pearson) and ``rho_w`` (Spearman / rank) (block = ``Env.Year``
    from :func:`block_array`; see :func:`weighted_block_correlation`).
  - ``predictions.csv`` D8 all-row schema.

The only difference is how each pipeline produces the aligned full-length
prediction vectors:
  - BGLR is transductive — it returns ``y_pred_train`` (FIT∪OBSERVE) and
    ``y_pred_test`` (PREDICT); :func:`aligned_full_predictions` scatters
    both back by split order.
  - DL is inductive — it runs ``trainer.predict`` over **all** rows and
    accumulates one ``y_pred``/``y_true`` per row;
    :func:`scatter_full_predictions` scatters that single stream by the
    rows' ``metadata_df`` positions.

Both feed the identical :func:`score_by_label` / :func:`write_predictions`.
"""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr, spearmanr

if TYPE_CHECKING:  # avoid import cycle (dataset → cv.schemes → ...)
    from utils.data.dataset import G2FDataset


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, Pearson correlation, and Spearman rank correlation."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if len(y_true) > 1:
        pearson, _ = pearsonr(y_true, y_pred)
        spearman, _ = spearmanr(y_true, y_pred)
        pearson = float(pearson)
        spearman = float(spearman)
    else:
        pearson = float("nan")
        spearman = float("nan")
    return {"rmse": rmse, "pearson_r": pearson, "spearman_r": spearman}


def weighted_block_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    blocks: np.ndarray,
    *,
    method: str = "pearson",
) -> tuple[float, int]:
    """Inverse-variance weighted within-block correlation (Tiezzi et al., 2017).

    Their "predictive ability" metric: rather than one correlation over the
    whole validation set, a correlation is computed *within* each block
    (Tiezzi's block is the herd; here it is the environment, ``Env.Year``) and
    the per-block correlations are pooled by inverse-variance weight. This
    measures how well the model ranks genotypes *inside* an environment,
    removing between-environment mean differences.

    For block ``j`` with ``n_j`` rows, ``r_j`` is the correlation between
    ``y_true`` and ``y_pred`` over that block, with sampling variance
    ``V(r_j) = (1 - r_j**2) / (n_j - 2)``. The pooled estimate is

        r_w = Σ_j (r_j / V(r_j)) / Σ_j (1 / V(r_j)).

    ``method`` selects the per-block correlation:

    - ``"pearson"`` (default) → linear correlation; the metric reported as
      ``r_w`` in ``metrics.json``.
    - ``"spearman"`` → rank correlation. Spearman's ρ is Pearson on
      within-block ranks, and its significance test uses the *same*
      ``(1 - ρ**2)/(n - 2)`` variance approximation, so the identical pooling
      is statistically valid; reported as ``rho_w``. This measures within-env
      **rank** predictive ability (selection-relevant), separate from the
      linear ``r_w``.

    A block is skipped (does not contribute) when it cannot yield a finite,
    positive-variance correlation: ``n_j < 3`` (``V`` undefined), constant
    ``y_true`` or ``y_pred`` within the block (``r_j`` undefined), or
    ``|r_j| == 1`` (``V = 0`` → infinite weight). Returns ``(r_w, n_blocks)``
    where ``n_blocks`` is the number of contributing blocks; the correlation is
    ``nan`` when none qualify.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(
            f"weighted_block_correlation: method must be 'pearson' or "
            f"'spearman', got {method!r}."
        )
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    blk = np.asarray(blocks)
    if not (len(yt) == len(yp) == len(blk)):
        raise ValueError(
            f"weighted_block_correlation: length mismatch — y_true "
            f"({len(yt)}), y_pred ({len(yp)}), blocks ({len(blk)})."
        )

    num = 0.0
    den = 0.0
    n_blocks = 0
    for b in np.unique(blk):
        mask = blk == b
        n_j = int(mask.sum())
        if n_j < 3:
            continue
        a = yt[mask]
        p = yp[mask]
        if method == "spearman":
            # Spearman = Pearson on ranks (average ranks for ties, matching
            # scipy.stats.spearmanr); the variance / skip logic below is then
            # the standard Spearman t-test approximation.
            from scipy.stats import rankdata

            a = rankdata(a)
            p = rankdata(p)
        if np.std(a) == 0.0 or np.std(p) == 0.0:
            continue  # r_j undefined (constant within block)
        r_j = float(np.corrcoef(a, p)[0, 1])
        if not np.isfinite(r_j) or abs(r_j) >= 1.0:
            continue  # |r_j| == 1 → V = 0 → infinite weight
        v_j = (1.0 - r_j * r_j) / (n_j - 2)
        if v_j <= 0.0 or not np.isfinite(v_j):
            continue
        num += r_j / v_j
        den += 1.0 / v_j
        n_blocks += 1

    if n_blocks == 0 or den == 0.0:
        return float("nan"), 0
    return float(num / den), n_blocks


def label_array(dataset: "G2FDataset") -> np.ndarray:
    """Per-``metadata_df``-row ``cv_label`` (object array, ``None`` unscored).

    Role-based (``cv_spec``) builds expose ``dataset.cv_labels`` directly.
    The legacy ``test_filter_columns`` path (``env_year_loo`` reproduced via
    a filter) has no scheme labels, so its single held-out test split *is*
    the ``"test"`` label — the degenerate one-label case (D2).
    """
    n = len(dataset.metadata_df)
    if getattr(dataset, "cv_labels", None) is not None:
        return np.asarray(dataset.cv_labels, dtype=object)
    labels = np.full(n, None, dtype=object)
    for p in dataset.split_indices.get("test", []):
        labels[p] = "test"
    return labels


def block_array(dataset: "G2FDataset") -> np.ndarray:
    """Per-``metadata_df``-row block id (``Env.Year``) for the within-block r_w.

    The block is the environment (Tiezzi et al.'s herd analog), built from the
    canonical ``Env.Year`` identifier. Raises if ``metadata_df`` lacks the
    ``Env`` / ``Year`` columns needed to form it.
    """
    meta = dataset.metadata_df
    if "Env" not in meta.columns or "Year" not in meta.columns:
        raise KeyError(
            "block_array: metadata_df needs 'Env' and 'Year' columns to "
            "build the Env.Year block id for r_w."
        )
    from cv.schemes.identifiers import env_year_series

    return env_year_series(meta).to_numpy()


def scatter_full_predictions(
    n_rows: int,
    positions: np.ndarray | list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter a single prediction stream into full-length ``metadata_df`` arrays.

    Used by the inductive DL path: ``trainer.predict`` runs over an all-row
    loader (FIT∪OBSERVE∪PREDICT) and ``positions[i]`` is the ``metadata_df``
    row that produced ``y_true[i]``/``y_pred[i]``. Returns ``(y_true_all,
    y_pred_all)`` of length ``n_rows`` with ``NaN`` in any row that received
    no prediction. ``positions`` must be unique (each metadata row predicted
    at most once) — a duplicate is an alignment bug and raises.
    """
    positions = np.asarray(positions, dtype=int)
    if not (len(positions) == len(y_true) == len(y_pred)):
        raise RuntimeError(
            f"scatter_full_predictions: positions ({len(positions)}), "
            f"y_true ({len(y_true)}), y_pred ({len(y_pred)}) length mismatch."
        )
    if len(np.unique(positions)) != len(positions):
        raise RuntimeError(
            "scatter_full_predictions: duplicate metadata positions — each "
            "row must be predicted at most once (alignment bug)."
        )
    y_true_all = np.full(n_rows, np.nan, dtype=np.float64)
    y_pred_all = np.full(n_rows, np.nan, dtype=np.float64)
    y_true_all[positions] = y_true
    y_pred_all[positions] = y_pred
    return y_true_all, y_pred_all


def aligned_full_predictions(
    dataset: "G2FDataset",
    y_train: np.ndarray,
    y_pred_train: np.ndarray | None,
    y_test: np.ndarray,
    y_pred_test: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter transductive train/test predictions back to ``metadata_df`` positions.

    Returns ``(y_true_all, y_pred_all)`` of length ``len(metadata_df)`` with
    ``NaN`` in rows that received no prediction (e.g. a DL ``val`` carve,
    never present for BGLR where ``val_ratio=0``). Alignment is positional:
    ``split_indices['train'|'test']`` are ``metadata_df`` positions (the
    RangeIndex invariant), and ``y_*`` / ``y_pred_*`` are stacked in those
    same orders (``_extract`` follows ``split_indices``). This is the single
    label==position convention this module exists to protect.
    """
    n = len(dataset.metadata_df)
    y_true_all = np.full(n, np.nan, dtype=np.float64)
    y_pred_all = np.full(n, np.nan, dtype=np.float64)

    train_idx = np.asarray(dataset.split_indices.get("train", []), dtype=int)
    test_idx = np.asarray(dataset.split_indices.get("test", []), dtype=int)

    if y_pred_train is not None and len(train_idx):
        if len(train_idx) != len(y_pred_train):
            raise RuntimeError(
                f"train alignment: split_indices['train'] has "
                f"{len(train_idx)} rows but y_pred_train has "
                f"{len(y_pred_train)}."
            )
        y_true_all[train_idx] = y_train
        y_pred_all[train_idx] = y_pred_train
    if y_pred_test is not None and len(test_idx):
        if len(test_idx) != len(y_pred_test):
            raise RuntimeError(
                f"test alignment: split_indices['test'] has "
                f"{len(test_idx)} rows but y_pred_test has "
                f"{len(y_pred_test)}."
            )
        y_true_all[test_idx] = y_test
        y_pred_all[test_idx] = y_pred_test

    return y_true_all, y_pred_all


def score_by_label(
    labels: np.ndarray,
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    blocks: np.ndarray,
    y_logprob_all: np.ndarray | None = None,
) -> dict[str, dict]:
    """Per-``cv_label`` metrics over the rows carrying each label (D2).

    Every scored quadrant — including the in-sample CV2/CV0 tested-line
    metrics over FIT rows — is a ``cor``/RMSE over a label-masked slice of
    the concatenated prediction vector. Labels are scored in sorted order
    for deterministic ``metrics.json``. A label row missing a prediction is
    a hard error (a masking/alignment bug, not an expected state).

    ``blocks`` is the per-row block id aligned to ``labels`` (the ``Env.Year``
    id from :func:`block_array`). Each entry carries ``rmse`` / ``pearson_r``
    / ``spearman_r`` plus Tiezzi et al.'s within-block inverse-variance
    weighted correlations — ``r_w`` (Pearson) and ``rho_w`` (Spearman, rank
    within-env predictive ability) — each with its number of contributing
    blocks (``r_w_n_blocks`` / ``rho_w_n_blocks``), computed over that label's
    slice.

    ``y_logprob_all`` is the optional per-row predictive log-density (aligned
    to ``labels`` like the other vectors, ``NaN`` where unpredicted). When
    given — only the DL path with a distributional head supplies it — each
    entry also carries ``loglik``, the mean log-density over the label slice
    (the point/MSE BGLR path has no density and omits it).
    """
    by_metric: dict[str, dict] = {}
    present = sorted({lab for lab in labels.tolist() if lab is not None})
    blk = np.asarray(blocks)
    for label in present:
        mask = labels == label
        yt = y_true_all[mask]
        yp = y_pred_all[mask]
        if np.isnan(yp).any() or np.isnan(yt).any():
            raise RuntimeError(
                f"label {label!r}: {int(np.isnan(yp).sum())} of {len(yp)} "
                "rows have no prediction — masking/alignment bug."
            )
        r_w, n_blocks = weighted_block_correlation(yt, yp, blk[mask])
        rho_w, rho_n_blocks = weighted_block_correlation(
            yt, yp, blk[mask], method="spearman"
        )
        entry = {
            "n": int(mask.sum()),
            **compute_metrics(yt, yp),
            "r_w": r_w,
            "r_w_n_blocks": n_blocks,
            "rho_w": rho_w,
            "rho_w_n_blocks": rho_n_blocks,
        }
        if y_logprob_all is not None:
            lp = y_logprob_all[mask]
            if np.isnan(lp).any():
                raise RuntimeError(
                    f"label {label!r}: {int(np.isnan(lp).sum())} of {len(lp)} "
                    "rows have no log-density — masking/alignment bug."
                )
            entry["loglik"] = float(np.mean(lp))
        by_metric[label] = entry
    return by_metric


def write_predictions(
    out_dir: str,
    dataset: "G2FDataset",
    labels: np.ndarray,
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    filename: str = "predictions.csv",
) -> None:
    """Write the D8 all-row ``predictions.csv``.

    Schema: ``Pedigree.Env, Pedigree, Env, Female, Fold, role, cv_label,
    Actual, Predicted`` — one row per ``metadata_df`` row that received a
    prediction (so in-sample CV2 and pooled-global aggregation both work).
    ``Pedigree.Env`` mirrors the R reference's ``PredVals.csv`` key. ``Female`` /
    ``Fold`` / ``role`` / ``cv_label`` are present only on the role-based
    build; the legacy filter path leaves them blank (Female is still
    derivable but the legacy artifact has no fold/role concept).
    """
    meta = dataset.metadata_df
    pedigrees = meta["Pedigree"].astype(str).tolist()
    envs = meta["Env"].astype(str).tolist()
    years = meta["Year"].astype(str).tolist() if "Year" in meta.columns else None

    def _col(name: str) -> list:
        return meta[name].tolist() if name in meta.columns else [None] * len(meta)

    females = _col("Female")
    folds = _col("Fold")
    roles = _col("role")
    cv_labels = labels

    def _fmt(v: object) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float) and float(v).is_integer():
            return str(int(v))
        return str(v)

    path = os.path.join(out_dir, filename)
    n_written = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Pedigree.Env", "Pedigree", "Env", "Female", "Fold",
                "role", "cv_label", "Actual", "Predicted",
            ]
        )
        for i, (ped, env) in enumerate(zip(pedigrees, envs)):
            if np.isnan(y_pred_all[i]):
                continue  # row received no prediction (e.g. DL val carve)
            # If the env column is just `XXH1` (no year), reconstruct the
            # full Env.Year identifier the way the R reference's bundle does.
            if years is not None and "." not in env:
                env_full = f"{env}.{years[i]}"
            else:
                env_full = env
            pe = f"{ped}.{env_full}"
            writer.writerow([
                pe, ped, env_full, _fmt(females[i]), _fmt(folds[i]),
                _fmt(roles[i]), _fmt(cv_labels[i]),
                float(y_true_all[i]), float(y_pred_all[i]),
            ])
            n_written += 1
    print(f"[scoring] Saved {n_written} row predictions to {path}")
