"""Non-decorated entry-point helpers.

``train.py`` and ``eval.py`` are 3-line ``@hydra.main`` shims that
delegate to ``run_training(cfg)`` / ``run_eval(cfg)`` here. The split
lets callers import the helpers directly with an already-composed
config instead of spawning a subprocess.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from functools import partial

import lightning.pytorch as pl
import torch
import wandb
from omegaconf import DictConfig

from utils.data import (
    ContextProvider,
    TaskSampler,
    _identity_collate,
    g2f_collate_fn,
)
from set_func.models import TransformerNeuralProcess
from utils.experiment.lightning_wrapper import LitWrapper, NPLitWrapper
from utils.experiment.setup import (
    build_eval_stream_runtimes,
    create_dataloader,
    initialize_callbacks,
    initialize_experiment,
    initialize_logger,
    log_hardware_info,
    log_training_config,
    resolve_scheduler_monitor,
)
from utils.experiment.utils import (
    ensure_directory_exists,
    evaluate_model,
    find_eval_checkpoints,
    init_wandb_run,
    log_results,
)


def _resolve_resume_checkpoint(checkpointing_config) -> str | None:
    """Return the resume checkpoint path, or None.

    D1 shape: ``checkpointing.resume_from`` is a single ``Optional[str]``.
    Consumed by ``run_training``.
    """
    resume_from = checkpointing_config.get("resume_from", None)
    if resume_from is None:
        return None
    if not os.path.exists(resume_from):
        print(f"Warning: Local checkpoint not found at {resume_from}")
        return None
    return resume_from


def _checkpointing_active(checkpointing_config) -> bool:
    """Checkpointing is active if anything will be written or resumed.

    Lightning's ``enable_checkpointing`` trainer flag is True iff a
    ``ModelCheckpoint`` callback will be installed OR we're restoring
    from a checkpoint path.
    """
    if checkpointing_config.get("resume_from", None) is not None:
        return True
    if checkpointing_config.get("save_last", False):
        return True
    best = checkpointing_config.get("best", {}) or {}
    return bool(best.get("enabled", False))


def _write_processing_metadata(experiment) -> None:
    """Write ``processing_metadata.json`` to ``misc.metrics_dirpath``.

    Captures everything eval.py / predict mode needs to reconstruct the
    processor state without re-fitting on train data:
      - ``processor_cache_keys``: per-processor content-addressed cache keys
        (see ``FeatureProcessor.cache_keys``). Read by
        ``setup._resolve_processing_cache_keys`` in predict mode.
      - ``enabled_processors``: ordered list of processor names that ran.
      - ``feature_dims``: per-source output dim (matches the model's encoder
        ``in_dim`` values when configured correctly).
      - ``effective_y_dim``: VI tensor width after any raw_vi slicing /
        weather-concat widening.
      - ``effective_ranks``: per-source effective eigen ranks from the
        kernel processors.

    No-op when there is no processor (baselines without processing or
    configs that disable everything).
    """
    proc = getattr(experiment.dataset, "processor", None)
    if proc is None:
        return
    metrics_dirpath = getattr(experiment.misc, "metrics_dirpath", None)
    if metrics_dirpath is None:
        return
    ensure_directory_exists(metrics_dirpath)
    payload = {
        "processor_cache_keys": proc.cache_keys,
        "enabled_processors": list(proc.enabled_names),
        # Processors run with persist_cache=false: their keys above have
        # no backing cache file, and predict mode refuses to start from
        # this metadata (see setup._resolve_processing_cache_keys).
        "unpersisted_processors": list(
            getattr(proc, "unpersisted_processors", [])
        ),
        "feature_dims": dict(proc.feature_dims),
        "effective_ranks": dict(proc.effective_ranks),
        "effective_y_dim": proc.effective_y_dim,
    }
    out_path = os.path.join(metrics_dirpath, "processing_metadata.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, out_path)
    print(f"Wrote processing metadata: {out_path}")


def _build_split_dataloader(split, dl_cfg, collate_fn, *, shuffle_default: bool):
    """Build a dataloader for a named split, or return None when empty.

    Shared helper for train/val/test dataloader construction. Returns
    ``None`` when the split is empty (Lightning skips the corresponding
    loop); otherwise builds a dataloader, defaulting ``batch_size`` to
    ``len(split)`` when the config leaves it unset.
    """
    if len(split) == 0:
        return None
    batch_size = (
        dl_cfg.batch_size if dl_cfg.batch_size is not None else len(split)
    )
    return create_dataloader(
        dataset=split,
        batch_size=batch_size,
        shuffle=dl_cfg.get("shuffle", shuffle_default),
        collate_fn=collate_fn,
        num_workers=dl_cfg.get("num_workers", 1),
        pin_memory=dl_cfg.get("pin_memory", True),
        drop_last=dl_cfg.get("drop_last", False),
    )


def _train_pool_and_roles(dataset):
    """Train-visible sample rows + their aligned roles (for the TaskSampler).

    Rows come from the base ``train`` split (``FIT ∪ OBSERVE``). Roles, when the
    dataset carries them, are read from ``metadata_df["role"]`` indexed by
    ``split_indices["train"]`` (same order as the rows) so the sampler's role
    firewall can defensively exclude any ``PREDICT`` row.
    """
    train_split = dataset.get_split("train")
    rows = [train_split[i] for i in range(len(train_split))]

    roles = None
    md = getattr(dataset, "metadata_df", None)
    si = getattr(dataset, "split_indices", None)
    if (
        md is not None
        and si is not None
        and "role" in getattr(md, "columns", [])
        and "train" in si
    ):
        role_col = md["role"].to_numpy()
        roles = [role_col[p] for p in si["train"]]
    return rows, roles


def _build_np_train_dataloader(dataset, ts_cfg, views, pad_value, augment=None):
    """Train dataloader for the Transformer Neural Process.

    Wraps a :class:`TaskSampler` (the role firewall + shared-per-batch ``nc``/``nq``
    schedule; builds the :class:`PrecomputedPool` once internally) as
    ``DataLoader(batch_size=None, collate_fn=_identity_collate)`` — the sampler
    already yields a ready :class:`NPTaskBatch`, so collate is a pass-through.

    ``task_sampler.num_workers > 0`` prefetches the next batch while the GPU runs;
    the sampler is worker-shardable so workers never duplicate batches.
    """
    rows, roles = _train_pool_and_roles(dataset)

    def _g(key, default=None):
        return ts_cfg.get(key, default) if ts_cfg is not None else default

    sampler = TaskSampler(
        rows,
        views=views,
        pad_value=pad_value,
        augment=augment,
        tasks_per_batch=_g("tasks_per_batch", 1),
        batches_per_epoch=_g("batches_per_epoch", 100),
        nc_min=_g("nc_min"),
        nc_max=_g("nc_max"),
        nq_min=_g("nq_min"),
        nq_max=_g("nq_max"),
        nc_sampler=_g("nc_sampler", "uniform"),
        seed=_g("seed", 0),
        roles=roles,
    )
    num_workers = int(_g("num_workers", 0) or 0)
    return torch.utils.data.DataLoader(
        sampler,
        batch_size=None,
        collate_fn=_identity_collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=bool(_g("pin_memory", False)),
    )


def run_training(cfg: DictConfig) -> None:
    """Train a model from a pre-composed Hydra config.

    ``train.py`` is a 3-line ``@hydra.main`` shim delegating here, so
    callers can also import this function directly with an
    already-composed config (``trainer.fast_dev_run=true`` short-circuits
    the actual training loop).
    """
    # Initialize the W&B logger before instantiation so that any prints or
    # warnings emitted during initialize_experiment (dataset loading,
    # processor fitting, model construction) are captured in the W&B run.
    # Touching logger.experiment triggers wandb.init() immediately.
    logger = initialize_logger(cfg)
    if isinstance(logger, pl.loggers.WandbLogger):
        _ = logger.experiment

    try:
        experiment = initialize_experiment(cfg)
    except BaseException:
        if isinstance(logger, pl.loggers.WandbLogger):
            try:
                wandb.finish(exit_code=1)
            except Exception as finish_err:
                print(f"Warning: wandb.finish failed after instantiation error: {finish_err}")
        raise

    model = experiment.model
    dataset = experiment.dataset
    optimizer = experiment.optimizer(model.parameters())
    epochs = experiment.misc.epochs

    # Write processing metadata for predict-mode reconstruction. Done before
    # trainer.fit so the file exists even if training crashes mid-epoch.
    _write_processing_metadata(experiment)

    # Create dataloaders
    dl_cfg = experiment.dataloader
    pad_value = getattr(dl_cfg, "pad_value", 0.0)
    collate_fn = partial(
        g2f_collate_fn,
        pad_value=pad_value,
        views=getattr(dataset, "dl_views", None),
    )

    train_dl_cfg = getattr(dl_cfg, "train", None)
    validation_dl_cfg = getattr(dl_cfg, "val", None)

    train_split = dataset.get_split("train")

    if len(train_split) == 0:
        raise ValueError(
            "Train split is empty. Check dataset configuration — "
            "training requires a non-empty train split."
        )

    # The Transformer Neural Process trains on TASKS (context+query drawn per
    # batch by a TaskSampler), so its train dataloader differs from the per-sample
    # path. Val/test loaders stay query-only (plain g2f_collate) — the context is
    # supplied on the eval path by the NPLitWrapper/ContextProvider. The
    # encoder variant is the only model switch; the task layer is selected here by
    # the model type.
    is_np = isinstance(model, TransformerNeuralProcess)
    task_sampler_cfg = getattr(experiment, "task_sampler", None)

    if is_np:
        # The TaskSampler builds the PrecomputedPool once and is worker-shardable,
        # so num_workers (from task_sampler.num_workers) prefetches without
        # duplicating batches. Default 0 keeps the single-process path.
        train_dataloader = _build_np_train_dataloader(
            dataset,
            task_sampler_cfg,
            getattr(dataset, "dl_views", None),
            pad_value,
            augment=getattr(experiment, "augment", None),
        )
    else:
        if train_dl_cfg is None:
            raise ValueError(
                "Training requires a 'train' dataloader config. "
                "Add a 'train:' block under 'dataloader:' in your YAML."
            )
        # Per-modality timepoint subsampling for per-sample (non-task) models: applied
        # TRAIN-ONLY, as a picklable collate wrapper around g2f_collate_fn (the
        # val/test loaders keep the plain collate_fn, so "off at eval" is
        # structural). A no-op augment leaves the plain collate untouched.
        from utils.data.augment import augmented_collate_fn, is_noop
        aug_cfg = getattr(experiment, "augment", None)
        train_collate_fn = collate_fn
        if not is_noop(aug_cfg):
            train_collate_fn = partial(
                augmented_collate_fn, collate_fn=collate_fn, aug=aug_cfg
            )
        train_dataloader = _build_split_dataloader(
            train_split, train_dl_cfg, train_collate_fn, shuffle_default=True
        )

    # Resolve the eval streams (carve + observe) into runtimes — one resolved
    # PhaseConfig per stream from the shared `eval` template + the stream's own
    # metrics — and build one val dataloader per stream in the canonical order
    # (`dataloader_idx` maps to `eval_streams[idx]`). Empty / skipped streams
    # are dropped so the surviving runtimes stay index-aligned with the
    # dataloaders. `eval.py` (predict-mode) clears eval_streams, so a
    # training-only run never needs a test dataloader here.
    eval_runtimes = build_eval_stream_runtimes(experiment)
    val_dataloaders: list = []
    active_runtimes: list = []
    if eval_runtimes and validation_dl_cfg is None:
        raise ValueError(
            "eval_streams are configured but there is no 'val' dataloader "
            "config. Add a 'val:' block under 'dataloader:' in your YAML."
        )
    for rt in eval_runtimes:
        stream_split = dataset.get_split(rt.name)
        dl = _build_split_dataloader(
            stream_split, validation_dl_cfg, collate_fn, shuffle_default=False
        )
        if dl is None:
            if rt.monitor:
                # The monitor carve drives early stopping + best-checkpoint;
                # an empty one means those callbacks would watch a metric that
                # is never logged. Fail loudly rather than silently disabling
                # model selection (e.g. a `random` frac that rounds to zero, or
                # a group/env/fold spec that matches no train rows).
                raise ValueError(
                    f"Monitor eval stream {rt.name!r} produced an empty split "
                    "— early stopping / best-checkpoint would watch a metric "
                    "that is never logged. Check the carve strategy (a `frac` "
                    "that rounds to zero, or a group/env/fold spec matching no "
                    "train rows)."
                )
            warnings.warn(
                f"Eval stream {rt.name!r} is empty — its val dataloader is "
                "skipped this run.",
                UserWarning,
            )
            continue
        val_dataloaders.append(dl)
        active_runtimes.append(rt)

    if not val_dataloaders:
        warnings.warn(
            "No non-empty eval streams — the validation loop is disabled "
            "during training (no early-stopping / best-checkpoint monitor).",
            UserWarning,
        )
        val_dataloaders = None  # Lightning: no val loop

    eval_template = getattr(experiment.phase_configs, "eval", None)

    # Get checkpointing configuration (D1 shape).
    checkpointing_config = getattr(experiment.misc, "checkpointing", {})

    # NP wrapper: the eval-context provider is the only extra wiring vs the
    # per-sample LitWrapper. eval_context defaults to the train pool.
    wrapper_cls = NPLitWrapper if is_np else LitWrapper
    np_wrapper_kwargs: dict = {}
    if is_np:
        eval_context = (
            task_sampler_cfg.get("eval_context", "train")
            if task_sampler_cfg is not None
            else "train"
        )
        np_wrapper_kwargs["context_provider"] = ContextProvider(
            dataset, eval_context=eval_context, pad_value=pad_value
        )

    # Model Initialization
    ckpt_file = None
    # Full horizon (before any weights-only resume trims `epochs`) and the
    # number of epochs already completed by a resumed checkpoint. Used below to
    # keep the LR schedule continuous across a weights-only resume.
    original_epochs = epochs
    completed_epochs = 0

    resume_path = _resolve_resume_checkpoint(checkpointing_config)

    if resume_path is not None:
        try:
            ckpt_file = resume_path
            lit_model = wrapper_cls.load_from_checkpoint(
                ckpt_file,
                model=experiment.model,
                test_config=eval_template,
                eval_streams=active_runtimes,
                **np_wrapper_kwargs,
            )
            # Weights-only checkpoints (save_weights_only=True) lack
            # optimizer_states, so trainer.fit(ckpt_path=...) would crash.
            # Load weights only and adjust remaining epochs instead.
            _meta = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            if "optimizer_states" not in _meta:
                _saved_epoch = _meta.get("epoch", 0)
                completed_epochs = _saved_epoch + 1
                epochs = max(1, original_epochs - completed_epochs)
                ckpt_file = None  # prevent trainer.fit from restoring loop state
                print(f"Loaded weights from epoch {_saved_epoch} (weights-only checkpoint); {epochs} epochs remaining")
            else:
                print(f"Resuming full state from: {resume_path}")
            del _meta
        except Exception as e:
            print(f"Warning: Failed to load local checkpoint: {e}")
            ckpt_file = None

    # Create LR scheduler if configured. Built AFTER the resume block so a
    # weights-only resume (which reduces `epochs` and has completed
    # `completed_epochs` epochs) continues the schedule from where it left off
    # rather than restarting from peak LR. The horizon (T_max / total_steps) is
    # pinned to the ORIGINAL epoch budget and `last_epoch` is advanced by the
    # completed epochs, so a cosine decay stays continuous across the resume
    # For a fresh run completed_epochs==0 → last_epoch=-1 (default).
    scheduler = None
    scheduler_config = {}
    scheduler_factory = getattr(experiment, "scheduler", None)
    if scheduler_factory is not None:
        if completed_epochs > 0:
            # PyTorch requires `initial_lr` on every param group to build a
            # scheduler with last_epoch != -1 (no saved scheduler state exists
            # on a weights-only checkpoint). Seed it from the current lr.
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        total_steps = original_epochs * len(train_dataloader)
        step_last_epoch = (
            completed_epochs * len(train_dataloader) - 1
            if completed_epochs > 0
            else -1
        )
        epoch_last_epoch = completed_epochs - 1 if completed_epochs > 0 else -1
        try:
            # OneCycleLR and similar step-based schedulers need total_steps;
            # keep the horizon at the original budget and skip completed steps.
            scheduler = scheduler_factory(
                optimizer,
                total_steps=total_steps,
                last_epoch=step_last_epoch,
            )
        except TypeError:
            # Epoch-based schedulers (CosineAnnealingLR etc.) reject
            # total_steps; T_max already spans the original horizon (from
            # config), so only advance last_epoch by the completed epochs.
            scheduler = scheduler_factory(optimizer, last_epoch=epoch_last_epoch)
        raw_cfg = getattr(experiment, "scheduler_config", None)
        if raw_cfg is not None:
            # Sync a plateau scheduler's ``monitor`` with the Python-derived
            # monitor key (B7) so the LR scheduler, early stopping, and
            # best-checkpoint all watch the same metric (see
            # ``resolve_scheduler_monitor``).
            scheduler_config = resolve_scheduler_monitor(dict(raw_cfg), experiment)

    # If no checkpoint loaded, initialize a new model. ``test_config`` is the
    # shared ``eval`` template — it feeds the train-set eval metrics (B8) and
    # the predict/test hooks; ``run_training`` never calls ``trainer.test()``.
    # The val loop is driven by ``eval_streams`` (one PhaseConfig per stream).
    if ckpt_file is None:
        lit_model = wrapper_cls(
            model=model,
            optimizer=optimizer,
            train_config=experiment.phase_configs.train,
            test_config=eval_template,
            eval_streams=active_runtimes,
            scheduler=scheduler,
            lr_scheduler_config=scheduler_config,
            **np_wrapper_kwargs,
        )

    # Logger was already built from cfg at the top of run_training so that
    # instantiation stdout is captured; only callbacks remain.
    callbacks = initialize_callbacks(experiment)

    accelerator = (
        "cpu" if torch.backends.mps.is_available() else "auto"
    )  # matrix inverse is not compatible with mps for some reason
    epochs = epochs if torch.cuda.is_available() else min(epochs, 2)

    # trainer.fast_dev_run support: cfg.trainer.fast_dev_run=true
    # short-circuits the loop. Gated on the field's existence so configs
    # without a `trainer:` block stay valid.
    fast_dev_run = bool(
        getattr(getattr(cfg, "trainer", None), "fast_dev_run", False)
    )

    trainer_config = {
        "logger": logger,
        "max_epochs": epochs,
        # Lightning's `log_every_n_steps` only matters when a logger is
        # attached; apply `misc.log_interval` in that case (1 ⇒ every step) and
        # leave it at Lightning's default when there is no logger (no-op).
        "log_every_n_steps": (
            getattr(experiment.misc, "log_interval", 50)
            if logger
            else None
        ),
        "devices": "auto",
        "accelerator": accelerator,
        "gradient_clip_val": (
            getattr(experiment.misc, "gradient_clip_val", 0.5)
        ),
        "callbacks": callbacks,
        "enable_progress_bar": (
            getattr(experiment.misc, "enable_progress_bar", True)
        ),
        "enable_checkpointing": _checkpointing_active(checkpointing_config),
        "enable_model_summary": True,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": getattr(
            experiment.misc, "check_val_every_n_epoch", 1
        ),
        "fast_dev_run": fast_dev_run,
    }

    trainer = pl.Trainer(**trainer_config)

    log_hardware_info()
    log_training_config(
        trainer,
        epochs,
        train_dataloader=train_dataloader,
        val_dataloader=(val_dataloaders[0] if val_dataloaders else None),
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloaders,
        ckpt_path=ckpt_file,
    )


def _select_eval_wrapper(experiment, pad_value):
    """Pick the eval/predict Lightning wrapper class + load kwargs for ``experiment``.

    A TNP predicts query-only batches and needs the eval context injected at
    predict time (np_forward pairs it with each query), so it loads as
    NPLitWrapper + a ContextProvider — mirroring run_training. Otherwise
    on_predict_start has no context to stash and np_forward raises. Per-sample
    models load as the plain LitWrapper. Extracted so the choice is isolated from
    ``run_eval`` (which itself needs a checkpoint).
    """
    is_np = isinstance(experiment.model, TransformerNeuralProcess)
    load_kwargs: dict = {"model": experiment.model}
    if is_np:
        task_sampler_cfg = getattr(experiment, "task_sampler", None)
        eval_context = (
            task_sampler_cfg.get("eval_context", "train")
            if task_sampler_cfg is not None
            else "train"
        )
        load_kwargs["context_provider"] = ContextProvider(
            experiment.dataset, eval_context=eval_context, pad_value=pad_value
        )
    return (NPLitWrapper if is_np else LitWrapper), load_kwargs


def run_eval(cfg: DictConfig) -> None:
    """Evaluate a model from a pre-composed Hydra config.

    ``eval.py`` is a 3-line ``@hydra.main`` shim delegating here, so
    callers can also import this function directly with an
    already-composed config. ``predict_mode=True`` is hardcoded in the call to
    ``initialize_experiment`` because that's what eval semantically
    means; toggling it via a config field would smear the train-vs-eval
    distinction across every model yaml.
    """
    # Initialize the W&B run before instantiation so that any prints or
    # warnings emitted during initialize_experiment (predict-mode dataset
    # loading, processor reconstruction, model build) are captured.
    wandb_run = None
    if getattr(cfg.misc, "wandb_eval_logging_enabled", False):
        try:
            wandb_run = init_wandb_run(cfg)
            if wandb_run:
                print(f"Initialized wandb run: {wandb_run.name} (ID: {wandb_run.id})")
        except Exception as e:
            print(f"Error initializing wandb run: {e}")

    try:
        try:
            experiment = initialize_experiment(cfg, predict_mode=True)
        except BaseException:
            if wandb_run is not None:
                try:
                    wandb.finish(exit_code=1)
                except Exception as finish_err:
                    print(f"Warning: wandb.finish failed after instantiation error: {finish_err}")
            raise

        dl_cfg = experiment.dataloader
        pad_value = getattr(dl_cfg, "pad_value", 0.0)
        collate_fn = partial(
            g2f_collate_fn,
            pad_value=pad_value,
            views=getattr(experiment.dataset, "dl_views", None),
        )

        test_dl_cfg = getattr(dl_cfg, "test", None)
        # The PREDICT (test) split must be non-empty — there must be at
        # least one held-out row to score. ``evaluate_model`` predicts over
        # *all* rows (FIT∪OBSERVE∪PREDICT), so the test split is the scored
        # subset, not the whole inference set.
        test_split = experiment.dataset.get_split("test")
        if len(test_split) == 0:
            raise ValueError(
                "Test/PREDICT split is empty. Check dataset.test_filter_columns / "
                "dataset.test_ratio (legacy) or cv_spec (role-based) — "
                "evaluation requires a non-empty held-out split."
            )
        if test_dl_cfg is None:
            raise ValueError(
                "Evaluation requires a 'test' dataloader config. "
                "Add a 'test:' block under 'dataloader:' in your YAML."
            )

        if not hasattr(experiment.phase_configs, "eval"):
            raise ValueError(
                "Evaluation requires an 'eval' phase_config template to be "
                "defined. Add a 'phase_configs.eval' block to your model YAML "
                "config (it supplies the predict-time forward wrapper)."
            )

        # Evaluate every present checkpoint (best / best_frozen / last) in this
        # one process — the dataset + processing (genomic GRM, R) are built once
        # by initialize_experiment, so looping here is far cheaper than separate
        # eval.py invocations. Every checkpoint writes metrics_<label>.json /
        # predictions_<label>.csv (and W&B keys under a "<label>/" prefix) —
        # there is no bare metrics.json for a DL run.
        checkpoints = find_eval_checkpoints(experiment)
        print(
            f"\nEvaluating {len(checkpoints)} checkpoint(s): "
            f"{', '.join(label for label, _ in checkpoints)}"
        )

        # B12: the offline scorer reads only test_config[0].forward_wrapper; the
        # shared `eval` template supplies it (one source of truth for "how the
        # model runs in eval mode"). The attribute stays named `test_config`.
        wrapper_cls, load_kwargs = _select_eval_wrapper(experiment, pad_value)
        metrics_dirpath = getattr(experiment.misc, "metrics_dirpath", None)

        for label, checkpoint_path in checkpoints:
            # Every checkpoint's artifacts are suffixed by its label so a
            # metric-selected checkpoint is always attributable to its metric;
            # there is no bare metrics.json / predictions.csv for a DL run.
            suffix = f"_{label}"
            print(f"\nEvaluating '{label}' checkpoint: {checkpoint_path}")
            model = wrapper_cls.load_from_checkpoint(
                checkpoint_path,
                test_config=experiment.phase_configs.eval,
                **load_kwargs,
            )
            results = evaluate_model(
                model,
                experiment.dataset,
                experiment,
                collate_fn=collate_fn,
                dl_cfg=test_dl_cfg,
                metrics_dir=metrics_dirpath,
                predictions_filename=f"predictions{suffix}.csv",
            )
            log_results(
                results,
                experiment,
                wandb_run,
                metrics_filename=f"metrics{suffix}.json",
                wandb_prefix=f"{label}/",
            )
            print(f"\nResults ('{label}'):")
            for lbl, m in results["by_metric"].items():
                print(f"  {lbl}: {m}")

        if wandb_run is not None:
            wandb.finish()
            print(f"W&B run finished: {wandb_run.name} (ID: {wandb_run.id})")

        print("\nEvaluation complete!")

    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
