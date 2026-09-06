import hashlib
import json
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from utils.experiment.lightning_wrapper import LitWrapper


def ensure_directory_exists(dirpath: Path | str) -> Path | None:
    """Create the directory if needed; returns None on a falsy path."""
    if dirpath:
        p = Path(dirpath)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"Directory created at: {p}")
        else:
            print(f"Directory already exists at: {p}")
        return p
    return None


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)


def _find_local_checkpoint(dirpath: Path) -> str | None:
    """Search a local directory for a checkpoint, preferring a best over last.

    Order: legacy ``best.ckpt`` (old artifacts), then any per-metric
    ``best_<metric>.ckpt`` (excluding ``best_frozen_*``), then ``last.ckpt``,
    then any ``.ckpt`` — so an eval still finds a checkpoint under either the
    legacy unsuffixed names or the per-metric naming.
    """
    best_path = dirpath / "best.ckpt"
    if best_path.exists():
        print(f"Loading best checkpoint: {best_path}")
        return str(best_path)
    best_metric = sorted(
        p
        for p in dirpath.glob("best_*.ckpt")
        if not p.stem.startswith("best_frozen")
    )
    if best_metric:
        print(f"Loading best checkpoint: {best_metric[0]}")
        return str(best_metric[0])
    last_path = dirpath / "last.ckpt"
    if last_path.exists():
        print(f"Loading last checkpoint: {last_path}")
        return str(last_path)
    candidates = sorted(dirpath.glob("*.ckpt"))
    if candidates:
        print(f"Loading checkpoint: {candidates[0]}")
        return str(candidates[0])
    return None


def _classify_checkpoint(stem: str) -> tuple[str, str] | None:
    """Map a checkpoint file stem to ``(kind, metric)``, or ``None`` if unknown.

    ``kind`` is ``"best"`` / ``"best_frozen"`` / ``"last"``; ``metric`` is the
    metric name for per-metric files (``best_rmse`` → ``"rmse"``) or ``""`` for
    legacy unsuffixed files (``best`` / ``best_frozen``). The test order is
    load-bearing: a frozen stem (``best_frozen_rmse``) also starts with
    ``best_``, so ``best_frozen_`` / ``best_frozen`` must be checked before
    ``best_`` / ``best``.
    """
    if stem == "last":
        return ("last", "")
    if stem.startswith("best_frozen_"):
        return ("best_frozen", stem[len("best_frozen_"):])
    if stem == "best_frozen":
        return ("best_frozen", "")
    if stem.startswith("best_"):
        return ("best", stem[len("best_"):])
    if stem == "best":
        return ("best", "")
    return None


def find_checkpoint_path(experiment_config: DictConfig) -> str:
    """Find a checkpoint from the local checkpoint directory."""
    checkpoint_dir = getattr(
        experiment_config.misc, "checkpoint_dirpath", None
    )
    if checkpoint_dir:
        path = _find_local_checkpoint(Path(checkpoint_dir))
        if path:
            return path
    raise RuntimeError("No checkpoint found locally")


def find_eval_checkpoints(
    experiment_config: DictConfig,
) -> list[tuple[str, str]]:
    """Checkpoints to evaluate, as ``(label, path)`` pairs.

    Discovers every per-metric ``best_<metric>.ckpt`` /
    ``best_frozen_<metric>.ckpt`` plus ``last.ckpt``. The label is the file
    stem, and EVERY checkpoint's outputs are written suffixed —
    ``metrics_<label>.json`` / ``predictions_<label>.csv`` (see ``run_eval``).
    There is no bare ``metrics.json`` for a DL run, so a metric-selected
    checkpoint can never be silently mistaken for another (the failure mode of
    an implicit "primary" that maps to the bare file).

    A checkpoint selected on a metric MUST name that metric in its filename. A
    bare ``best.ckpt`` / ``best_frozen.ckpt`` (no ``_<metric>`` suffix) is a
    hard error: its metrics/predictions could not be attributed to the metric
    that selected it. ``last.ckpt`` is exempt (it is not metric-selected).

    Ordering: all ``best_<metric>`` (alphabetical), then all
    ``best_frozen_<metric>`` (alphabetical), then ``last`` — deterministic, but
    with no privileged first element. Falls back to the single-checkpoint
    search only when ``checkpoint_dirpath`` is unset (a standalone eval against
    an explicitly supplied checkpoint).
    """
    checkpoint_dir = getattr(
        experiment_config.misc, "checkpoint_dirpath", None
    )
    if not checkpoint_dir:
        path = find_checkpoint_path(experiment_config)
        return [(Path(path).stem, path)]

    d = Path(checkpoint_dir)
    discovered: list[tuple[str, str, str]] = []  # (kind, metric, path)
    unlabeled: list[str] = []
    for p in sorted(d.glob("*.ckpt")):
        classified = _classify_checkpoint(p.stem)
        if classified is None:
            continue
        kind, metric = classified
        if kind in ("best", "best_frozen") and metric == "":
            unlabeled.append(p.name)
            continue
        discovered.append((kind, metric, str(p)))

    if unlabeled:
        raise ValueError(
            "Metric-selected checkpoint(s) missing a metric suffix: "
            f"{', '.join(sorted(unlabeled))}. Every best/best_frozen checkpoint "
            "must be named best_<metric>.ckpt / best_frozen_<metric>.ckpt so its "
            "metrics_<label>.json / predictions_<label>.csv can be attributed to "
            "the metric that selected it."
        )
    if not discovered:
        path = find_checkpoint_path(experiment_config)
        return [(Path(path).stem, path)]

    kind_rank = {"best": 0, "best_frozen": 1, "last": 2}
    discovered.sort(key=lambda it: (kind_rank[it[0]], it[2]))
    return [(Path(path).stem, path) for _, _, path in discovered]


def evaluate_model(
    model: LitWrapper,
    dataset: Any,
    experiment_config: DictConfig,
    *,
    collate_fn: Any,
    dl_cfg: Any,
    metrics_dir: Path | str | None = None,
    predictions_filename: str = "predictions.csv",
) -> dict[str, Any]:
    """Evaluate a DL model by predicting over **all** rows, score by label.

    D7: DL is inductive, but the role/label abstraction covers it. We run
    inference over the full FIT∪OBSERVE∪PREDICT set (not just the test
    split), then score ``cor``/RMSE per ``cv_label`` with the **identical**
    scorer as the BGLR path (``cv.scoring``).

    Crucially the metrics are computed from the **raw accumulated**
    ``y_pred``/``y_true`` that ``trainer.predict`` returns, scored once by the
    shared ``cv.scoring`` — so they are **batch-size invariant**: the batch
    split only changes how the prediction stream is chunked before it is
    concatenated, never the scored numbers (incl. the correlations and the
    within-block ``r_w``/``rho_w``).

    This path does **not** call ``trainer.test``. The ``LitWrapper`` val/test
    epoch reduction (``_eval_epoch_end``) is itself correct and batch-size
    invariant — it pools its ``(pred, true)`` buffers and reruns the metric
    once, not a per-batch weighted mean. The reason it is bypassed is *scope*,
    not bias: those buffers carry no ``Env.Year`` block id or ``cv_label``, and
    the test ``metric_fn`` is the generic regression set (RMSE/Pearson/
    Spearman) — no per-``cv_label`` split and no within-block ``r_w``/``rho_w``.
    A within-block correlation cannot be assembled from independent per-batch
    reductions (a block spans batches), so the by-label scoring has to run over
    the full accumulated stream here.

    Args:
        model: Loaded ``LitWrapper`` (its ``test_config`` selects the
            forward wrapper used by ``predict_step``).
        dataset: The built ``G2FDataset`` — provides the per-row sample
            list, ``split_indices`` (metadata positions), ``cv_labels`` and
            ``metadata_df`` for the scorer.
        experiment_config: Experiment configuration (for ``meta``).
        collate_fn: Collate function for the all-row dataloader.
        dl_cfg: The ``test`` dataloader config (batch_size / num_workers /
            pin_memory). ``batch_size=null`` ⇒ the whole predict set in one
            batch (the single-batch / accumulating loader D7 prefers).
        metrics_dir: Output dir for ``predictions.csv``. No-op when None.

    Returns:
        Results dict with ``meta`` and ``by_metric`` keys — the same schema
        the BGLR path emits, so downstream consumers read both identically.
    """
    # Local imports to avoid a setup.py ↔ utils.py import cycle.
    from cv.scoring import (
        block_array,
        label_array,
        scatter_full_predictions,
        score_by_label,
        write_predictions,
    )
    from utils.data.dataset import _SplitDataset
    from utils.experiment.setup import create_dataloader

    trainer = pl.Trainer(
        accelerator=(
            "cpu" if torch.backends.mps.is_available() else "auto"
        ),  # matrix inverse is not compatible with mps
        devices="auto",
        logger=False,
        enable_checkpointing=False,
    )

    # All-row loader: concatenate the three splits. ``split_indices`` give
    # each sample's ``metadata_df`` position (the RangeIndex invariant), so
    # predictions scatter back positionally. A DL ``val`` carve is included
    # here — it gets predicted but carries no ``cv_label`` (cleared in
    # ``assign_roles``), so the scorer ignores it (D7).
    all_samples = (
        list(dataset._train) + list(dataset._val) + list(dataset._test)
    )
    positions = (
        list(dataset.split_indices.get("train", []))
        + list(dataset.split_indices.get("val", []))
        + list(dataset.split_indices.get("test", []))
    )
    if len(all_samples) != len(positions):
        raise RuntimeError(
            f"evaluate_model: {len(all_samples)} samples but "
            f"{len(positions)} split positions — split bookkeeping bug."
        )

    batch_size = (
        dl_cfg.batch_size if dl_cfg.batch_size is not None else len(all_samples)
    )
    all_loader = create_dataloader(
        dataset=_SplitDataset(all_samples),
        batch_size=batch_size,
        shuffle=False,  # row order must track `positions`
        collate_fn=collate_fn,
        num_workers=dl_cfg.get("num_workers", 1),
        pin_memory=dl_cfg.get("pin_memory", True),
        drop_last=False,  # eval must keep every row — never drop the last batch
    )

    preds = trainer.predict(model, dataloaders=all_loader)
    y_pred = torch.cat([p["y_pred"] for p in preds]).numpy()
    y_true = torch.cat([p["y_true"] for p in preds]).numpy()
    # Per-sample predictive log-density, present only when the model's head is
    # distributional (``predict_step`` omits it for a point head). Threaded
    # through to the scorer for a per-``cv_label`` ``loglik``.
    has_logprob = bool(preds) and all("y_logprob" in p for p in preds)
    y_logprob = (
        torch.cat([p["y_logprob"] for p in preds]).numpy()
        if has_logprob
        else None
    )

    # D7: the predict loader must accumulate every row so per-label
    # Pearson/Spearman are over the full slice (no batch-boundary effects).
    if len(y_pred) != len(positions):
        raise RuntimeError(
            f"evaluate_model: predict returned {len(y_pred)} rows but "
            f"the all-row loader had {len(positions)} — accumulation bug."
        )

    n_rows = len(dataset.metadata_df)
    y_true_all, y_pred_all = scatter_full_predictions(
        n_rows, positions, y_true, y_pred
    )
    # Scatter the log-density on the same (validated-unique) positions.
    y_logprob_all = None
    if y_logprob is not None:
        y_logprob_all = np.full(n_rows, np.nan, dtype=np.float64)
        y_logprob_all[np.asarray(positions, dtype=int)] = y_logprob
    labels = label_array(dataset)
    by_metric = score_by_label(
        labels,
        y_true_all,
        y_pred_all,
        blocks=block_array(dataset),
        y_logprob_all=y_logprob_all,
    )

    # All-row predictions.csv (D8) for cross-fold pooling / in-sample rows.
    # A failed write must propagate (exit non-zero): swallowing it lets the
    # job touch its idempotency marker (curve.done / metrics_last.json
    # probe) with no artifacts on disk, permanently skipping the fold.
    if metrics_dir is not None:
        md = Path(metrics_dir)
        md.mkdir(parents=True, exist_ok=True)
        write_predictions(
            str(md), dataset, labels, y_true_all, y_pred_all,
            filename=predictions_filename,
        )

    return {
        "meta": {
            "model_name": experiment_config.model_name,
            # Architecture + subset tags (no seed/fold) — the
            # aggregation-stable identity. Falls back to the bare name
            # for configs composed without the misc.model_slug field.
            "model_slug": getattr(
                experiment_config.misc, "model_slug",
                experiment_config.model_name,
            ),
            "seed": getattr(experiment_config.misc, "seed", None),
            "n_test": len(dataset.split_indices.get("test", [])),
        },
        "by_metric": by_metric,
    }


def init_wandb_run(
    cfg: DictConfig,
) -> wandb.sdk.wandb_run.Run | None:
    """Initialize or resume a wandb run from the raw Hydra config.

    Takes ``cfg`` (pre-instantiation) so ``run_eval`` can hoist wandb init
    above ``initialize_experiment`` and capture instantiation stdout in
    the W&B run.
    """
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    job_type = "evaluation"
    project = cfg.misc.project

    try:
        if run_id_full := getattr(cfg.misc, "wandb_run_id", None):
            # Always prioritize using an existing run_id if available
            run_id = run_id_full.split("/")[-1]
            run = wandb.init(
                id=run_id,
                resume="allow",
                project=project,
                config=wandb_config,
                job_type=job_type,
            )
            print(f"Resumed W&B run {run.name} (ID: {run.id})")
        else:
            # Only create a new run if no run_id was provided
            run = wandb.init(
                project=project,
                name=cfg.misc.eval_name,
                config=wandb_config,
                job_type=job_type,
            )
            print(f"Created new W&B run {run.name} (ID: {run.id})")

        return run
    except Exception as e:
        print(f"Failed to initialize W&B run: {e}")
        return None


def _save_locally(
    results: dict[str, Any],
    metrics_dir: Path,
    filename: str = "metrics.json",
) -> None:
    """Dump the full results dict to ``filename`` under metrics_dir.

    Written atomically (tmp + ``Path.replace``) so a kill mid-dump cannot
    leave a truncated ``metrics.json`` — which is both the idempotency skip
    marker and JSON that aggregation must parse.
    """
    ensure_directory_exists(metrics_dir)
    filepath = metrics_dir / filename
    tmp_path = filepath.with_name(filepath.name + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    tmp_path.replace(filepath)  # atomic on POSIX (Path.replace → os.replace)
    print(f"Saved results to {filepath}")


def sanitize_wandb_tags(tags: Any) -> list[str]:
    """Truncate/hash W&B tags longer than 64 chars before ``wandb.init``.

    wandb rejects tags >64 chars and crashes — and in the FPCA path that
    crash happens *after* ``metrics.json`` is written, marking the SLURM
    job FAILED. The new ``cv_seed``/``fold``/``heldout_env`` slug tokens
    push slugs past 64 chars, so the 950-job ``cv_0_00`` array would trip
    this en masse. Over-long tags are replaced by a 55-char prefix + ``-``
    + 8-char content hash (total 64) so they stay distinct and traceable.
    """
    out: list[str] = []
    for tag in tags:
        tag = str(tag)
        if len(tag) > 64:
            digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
            tag = f"{tag[:55]}-{digest}"
        out.append(tag)
    return out


def sanitize_wandb_group(group: str) -> str:
    """Cap a W&B group name at wandb's 128-char hard limit.

    wandb rejects group names >128 chars with a 400 ("invalid parameters:
    128 limit exceeded for GroupName"), crashing ``wandb.init`` *after* the
    job has claimed its slot, so the SLURM job FAILs with no metrics. The
    resolved ``model_slug`` alone is ~110 chars, so a
    ``scheme/<slug>/cvseed/seed`` group trips it; even the shorter
    ``CONFIG``-based group is one rename away from doing so. Over-long
    groups are replaced by a 119-char prefix + ``-`` + 8-char content hash
    (total 128) — deterministic, so identical groups stay identical and
    distinct groups stay distinct (no wrong merges).
    """
    group = str(group)
    if len(group) > 128:
        digest = hashlib.sha1(group.encode("utf-8")).hexdigest()[:8]
        group = f"{group[:119]}-{digest}"
    return group


def _log_to_wandb(
    run: wandb.sdk.wandb_run.Run,
    metrics: dict[str, float],
) -> None:
    """Log metrics to the given wandb run summary."""
    run.summary.update(metrics)
    print(f"Logged {len(metrics)} metrics to W&B summary")


def log_results(
    results: dict[str, Any],
    experiment_config: DictConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    metrics_filename: str = "metrics.json",
    wandb_prefix: str = "",
) -> None:
    """
    Log evaluation results to Weights & Biases and/or a local file.

    Args:
        results: Results dict with 'meta' and 'by_metric' keys.
        experiment_config: Experiment configuration
        wandb_run: Optional existing wandb run to use
        metrics_filename: Local file to write under ``misc.metrics_dirpath``
            (``metrics_<label>.json`` — every evaluated checkpoint gets its
            own suffixed file; there is no bare ``metrics.json`` for a DL
            run).
        wandb_prefix: Prepended to every W&B summary key (``"<label>/"``
            per checkpoint) so the per-checkpoint scalars never collide in
            one run.
    """
    if (
        getattr(experiment_config.misc, "wandb_eval_logging_enabled", False)
        and wandb_run
    ):
        try:
            # Flatten the per-label metrics to ``<label>/<metric>`` scalars,
            # matching the BGLR path (baselines/fpca_core.py).
            flat = {
                f"{wandb_prefix}{label}/{k}": v
                for label, m in results["by_metric"].items()
                for k, v in m.items()
            }
            _log_to_wandb(wandb_run, flat)
        except Exception as e:
            print(f"Failed to log results to W&B: {e}")

    # The local metrics file is the run's completion evidence — a failed
    # write must propagate (exit non-zero) rather than let the job mark
    # the fold done with nothing on disk. (W&B logging above stays
    # best-effort: a telemetry flake shouldn't fail a fold whose local
    # artifacts are intact.)
    if metrics_dir := getattr(experiment_config.misc, "metrics_dirpath", None):
        _save_locally(results, Path(metrics_dir), filename=metrics_filename)
