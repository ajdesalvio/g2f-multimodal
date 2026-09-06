import os
import json
import random
import psutil
import platform
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from utils.data.processing.rank_check import check_rank_consistency

from .lightning_wrapper import (
    EvalStreamRuntime,
    FreezableBestCheckpoint,
    LogPerformanceCallback,
    PhaseConfig,
    PlateauFreezeCallback,
    TrainingCurveCallback,
    WeightAveragingCallback,
)
from .metrics import build_metric, get_metric_class, metric_direction
from .utils import ensure_directory_exists


class _AttrDict(dict):
    """Dict with attribute access and recursive wrapping.

    wraps the result of ``instantiate(cfg, _convert_="all")``
    so the attribute-access sites (e.g.
    ``experiment.misc.seed``, ``getattr(experiment.misc, "checkpoint_dirpath",
    default)``) keep working after the D3 flip removed ``DictConfig`` from
    the runtime path.

    Mutation through ``__setattr__`` is required: ``initialize_callbacks``
    writes ``experiment.misc.default_lightning_logger = True/False`` at
    two sites in this module, and the predict-mode resume branch writes
    ``experiment.misc.wandb_run_id``. A read-only wrapper would silently
    drop those mutations.

    Recursive wrapping: any plain ``dict`` value is wrapped on construction
    so ``experiment.misc.checkpointing.resume_from`` chains work without
    re-wrapping at every level. Non-dict values (instantiated objects like
    ``TransformerNeuralProcess`` or ``functools.partial``) pass through
    untouched. ``DictConfig`` instances also pass through untouched —
    ``experiment.config`` keeps the original config alive for downstream
    sites that round-trip through ``OmegaConf.to_container`` (W&B).
    """

    def __init__(self, data: Any = None):
        super().__init__()
        if data is None:
            return
        for k, v in data.items():
            super().__setitem__(k, self._wrap(v))

    @classmethod
    def _wrap(cls, v: Any) -> Any:
        # Don't double-wrap our own instances.
        if isinstance(v, cls):
            return v
        # Wrap plain dicts only. DictConfig and other dict subclasses are
        # left alone so OmegaConf round-tripping keeps working.
        if type(v) is dict:
            return cls(v)
        if type(v) is list:
            return [cls._wrap(x) for x in v]
        return v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setitem__(name, self._wrap(value))

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _thread_cv_spec_into_dataset(cfg: DictConfig) -> None:
    """Copy a root-level ``cv_spec`` block into ``cfg.dataset.cv_spec``.

    The ``cv_spec`` config group (D9) is ``# @package _global_``, so an
    opt-in ``+cv_spec=cv_0_00`` lands at ``cfg.cv_spec`` (root) — but
    ``instantiate(cfg.dataset)`` only sees ``cfg.dataset.*``. This copies the
    block down so ``G2FDataset(cv_spec=...)`` receives it and flips onto the
    role-based split path. No-op when there is no root ``cv_spec`` (the
    legacy ``test_filter_columns`` path is unchanged) or when
    ``cfg.dataset.cv_spec`` was already set explicitly.

    ``force_add=True``: ``cv_spec`` is a runtime-injected dataset field not
    declared in the schema, so struct-mode would otherwise reject the key.
    """
    cv_spec = OmegaConf.select(cfg, "cv_spec", default=None)
    if cv_spec is None:
        return
    existing = OmegaConf.select(cfg, "dataset.cv_spec", default="__missing__")
    if existing not in (None, "__missing__"):
        return
    OmegaConf.update(cfg, "dataset.cv_spec", cv_spec, force_add=True)


def initialize_experiment(
    cfg: DictConfig,
    predict_mode: bool = False,
) -> _AttrDict:
    """Initialize an experiment from a pre-composed Hydra config.

    This function does not parse CLI arguments or load YAML files. The
    caller (``train.py``, ``eval.py``, or anything composing via
    ``hydra.compose``) is responsible for producing a fully-resolved
    ``DictConfig`` — no argparse path exists.

    Instantiation is split into two phases:
      1. Build ``cfg.dataset`` first. This runs DataReader, splitter, the
         FeatureProcessor (fit_transform in train mode, load_and_transform
         in predict mode), normalization, etc. Doing this before model
         construction lets ``_validate_config`` check the model config
         against actual processor outputs (feature_dims, effective_y_dim).
      2. Strip ``_target_`` from ``cfg.dataset`` and instantiate everything
         else (model, optimizer, scheduler, ...). Then attach the pre-built
         dataset onto the resulting experiment object.

    Args:
        cfg: Pre-composed Hydra DictConfig. Must contain a ``dataset``
            block with ``_target_``, a ``misc`` block with ``seed``, and
            the rest of the runtime keys (``model``, ``optimizer``,
            ``phase_configs``, ``dataloader``, ...) at root.
        predict_mode: When True, processing cache keys are resolved from
            training artifacts and injected into ``dataset.processing_cache_keys``
            so ``G2FDataset.__init__`` calls ``load_and_transform()`` instead
            of ``fit_transform()``. ``eval.py`` passes this; ``train.py`` does
            not.
    """
    pl.seed_everything(cfg.misc.seed)

    # Thread a root-level cv_spec (D9) into the dataset config so
    # instantiate(cfg.dataset) passes it to G2FDataset, flipping it onto the
    # role-based split path (coverage → roles, D6). The cv_spec group is
    # `# @package _global_`, so it lands at cfg.cv_spec (root); the dataset
    # block doesn't declare it. Mirrors the FPCA path's
    # ``_build_dataset`` (which reads root ``cv_spec`` and passes it through).
    _thread_cv_spec_into_dataset(cfg)

    # Validate the eval-streams config before building the dataset (the
    # dataset consumes the carve specs during construction), so a bad stream
    # config fails at the gate with a clear message rather than deep in the
    # splitter. No-op when no eval_streams are configured.
    _validate_eval_streams(cfg)

    # Eval (predict_mode): drop ALL eval streams so the scored label sets
    # cover *every* labeled row — in particular ``cv_2_1``'s in-sample CV2
    # over every retained FIT row, matching R/BGLR (no early-stopping
    # holdout). The carve is a training-only concern; at eval there is no
    # training and no validation loop. ``evaluate_model`` scores over all
    # rows independently of eval streams (B12), and clearing the carve does
    # not change the FIT set (the ``vi_fpca`` fit-mask is the FIT role,
    # independent of the carve), so the processing cache still hits.
    if predict_mode:
        OmegaConf.update(cfg, "dataset.eval_streams", [], force_add=True)

    # Predict mode: inject cache keys into dataset config *before* instantiation
    # so G2FDataset.__init__ takes the load_and_transform() path.
    #
    # Gate on *at least one processor actually enabled*, not just the
    # presence of the processing block. The default `dataset.processing`
    # block lists every processor with `enabled: false`, so a naive
    # truthy check would fire predict-mode cache resolution on vanilla
    # VI-only runs (which never wrote a processing_metadata.json).
    if predict_mode and _any_processor_enabled(cfg):
        cache_keys = _resolve_processing_cache_keys(cfg)
        # force_add=True: dataset.processing_cache_keys is intentionally
        # left out of the schema (it's a runtime-injected field), and
        # struct-mode DictConfig rejects new keys without this escape hatch.
        OmegaConf.update(
            cfg,
            "dataset.processing_cache_keys",
            cache_keys,
            force_add=True,
        )

    # Phase 1: build the dataset. _convert_="all" returns a plain G2FDataset
    # instance (the dataset's _target_ resolves to a Python class, and
    # _convert_ only affects nested dict/list values, not top-level
    # instantiation results).
    dataset = instantiate(cfg.dataset, _convert_="all")

    # Validate model/processing config consistency before constructing the model.
    if getattr(dataset, "processor", None) is not None:
        _validate_config(cfg, dataset)
        # Phase 2: fill params.y_dim from the processor's effective width.
        # Must run before the full-cfg instantiate at phase 2 below (so
        # the model sees resolved integers).
        _inject_set_encoder_ydims(cfg, dataset.processor)
        # Fill the VI branch's coord/channel dims (params.x_dim / y_dim) from
        # the assembled "main" view, so multi-axis coords (D>1) and
        # axis-as-channel selectors are reflected in the encoder shapes.
        _inject_view_dims(cfg, dataset)
        # Fill the TNP tokenizer's geno-MLP input width (params.geno_in_dim)
        # from the configured geno_key feature widths (eigen K or rows
        # n_train_pedigrees). No-op for non-TNP models (param absent).
        _inject_np_geno_dim(cfg, dataset.processor)

    # Phase 2: instantiate the rest of the experiment. Drop the dataset's
    # _target_ so Hydra leaves cfg.dataset as a plain dict instead of
    # re-running G2FDataset.__init__ (which would re-load all CSVs and
    # re-run the FeatureProcessor — minutes of wasted work).
    if "_target_" in cfg.dataset:
        # open_dict bypasses struct-mode deletion guard. The Hydra-composed
        # cfg is struct-mode-by-default (the root carries the MiscConfig
        # schema and propagates struct mode), so direct `del` raises.
        with open_dict(cfg.dataset):
            del cfg.dataset["_target_"]
    # Reseed IMMEDIATELY before the model is instantiated so weight-init draws
    # from a deterministic RNG state for a given `misc.seed`, independent of how
    # much RNG the dataset construction/feature processing consumed above.
    # Without this, two configs differing only in data processing would get
    # different weight inits at the same seed.
    pl.seed_everything(cfg.misc.seed)
    experiment_raw = instantiate(cfg, _convert_="all")
    experiment = _AttrDict(experiment_raw)
    experiment.dataset = dataset
    experiment.config = cfg
    pl.seed_everything(experiment.misc.seed)

    # The schema field defaults to None, so a non-None value means the
    # caller intentionally set it via `misc.wandb_run_id=` CLI override.
    if getattr(experiment.misc, "wandb_run_id", None):
        print(f"Using provided wandb run ID: {experiment.misc.wandb_run_id}")

    # mkdir is idempotent and cheap, so always create the checkpoint
    # dir when a path is configured.
    if hasattr(experiment.misc, "checkpoint_dirpath"):
        ensure_directory_exists(experiment.misc.checkpoint_dirpath)

    # Create metrics directory if needed and not using wandb
    if hasattr(experiment.misc, "metrics_dirpath"):
        ensure_directory_exists(experiment.misc.metrics_dirpath)

    return experiment


def _inject_set_encoder_ydims(config: DictConfig, processor) -> None:
    """Fill the VI set-encoder y-dim from the live dataset.

    The VI set-encoder branch reads ``params.y_dim`` for its ``y_encoder.in_dim``,
    which depends on the active variable subset (the ``vi_subset`` config
    group), so we fill it from the processor's just-computed width instead of
    hardcoding a count in any model or subset config.

    - ``params.y_dim`` ← ``processor.effective_y_dim`` (final ``channels``
      width: VI subset + any weather_concat tail). Present on every DL model;
      absent on baselines (skipped).

    The weather peer's ``params.weather_y_dim`` (and ``weather_x_dim``) are
    **no longer filled here** — the weather peer is a real
    ``dataset.views`` entry, so ``_inject_view_dims`` fills its coord/channel
    dims from the assembled view (per-peer). This function and
    ``_inject_view_dims`` agree on ``params.y_dim`` for the ``main`` view
    (``effective_y_dim == assemble_view(main).C``); the latter runs after and
    is the canonical source.
    """
    # VI y-dim (key always carries a real int default on DL models).
    if OmegaConf.select(config, "params.y_dim", default="__missing__") != "__missing__":
        eff = getattr(processor, "effective_y_dim", None)
        if eff is not None:
            OmegaConf.update(config, "params.y_dim", int(eff), force_add=False)


def _inject_np_geno_dim(config: DictConfig, processor) -> None:
    """Fill ``params.geno_in_dim`` for the TNP tokenizer's geno MLP.

    The :class:`~set_func.models.neural_process.SampleTokenizer` runs a single
    MLP over the concatenated ``geno_key`` derived features, so its input width
    is the SUM of those sources' feature dims. Switching the geno representation
    (eigen ↔ kernel rows ↔ raw dosage) is then pure config — ``geno_key`` plus
    the genomic processor's ``sources`` decide which keys exist, and this fills
    the concrete width (eigen ``K`` or rows ``n_train_pedigrees``) automatically.

    Reads ``params.geno_key`` (str or list); no-op when ``params.geno_in_dim`` is
    absent (non-TNP model) or already an int.
    """
    sentinel = "__missing__"
    if OmegaConf.select(config, "params.geno_in_dim", default=sentinel) == sentinel:
        return
    if OmegaConf.select(config, "params.geno_in_dim", default=None) is not None:
        return  # already resolved to an int
    geno_key = OmegaConf.select(config, "params.geno_key", default=None)
    if geno_key is None:
        return
    keys = [geno_key] if isinstance(geno_key, str) else list(geno_key)
    feature_dims = getattr(processor, "feature_dims", None) or {}
    total = sum(int(feature_dims[k]) for k in keys if k in feature_dims)
    if total > 0:
        OmegaConf.update(config, "params.geno_in_dim", total, force_add=False)


def _view_dim_params(view_name: str) -> tuple[str, str]:
    """The ``(x_dim_param, y_dim_param)`` config paths a peer bound to
    ``view_name`` reads its coord/channel widths from.

    Convention (matches the shipped model configs, zero churn):
    - the VI view ``"main"`` → the legacy globals ``params.x_dim`` /
      ``params.y_dim`` (read by the VI ``x_encoder`` / ``y_encoder``);
    - any other view ``V`` → ``params.{V}_x_dim`` / ``params.{V}_y_dim``
      (e.g. the weather peer already reads ``params.weather_x_dim`` /
      ``params.weather_y_dim``).
    """
    if view_name == "main":
        return "params.x_dim", "params.y_dim"
    return f"params.{view_name}_x_dim", f"params.{view_name}_y_dim"


def _inject_view_dims(config: DictConfig, dataset) -> None:
    """Fill each set-encoder peer's coord/channel dims from its bound view.

    A set encoder reads two scalar dims: its coordinate width (the
    ``x_encoder`` in_dim) and its
    channel width (``y_encoder`` in_dim). With config-driven Views these are no
    longer fixed at 1 / VI-count: a multi-axis ``coords:[dap,agdd]`` view has
    ``D==2``; an axis-as-channel ``channels:[vi.*, agdd]`` view widens ``C``.

    For **every DL-target view** in ``dataset.views`` we assemble it on one
    train sample and write the real ``(D, C)`` into that peer's dim params
    (see :func:`_view_dim_params` for the param-name convention). Each write is
    guarded by param-presence — a model only gets the dims it actually
    declares — so a view with no matching params is a no-op. This generalizes
    the former ``main``-only injection to the two-view (DAP+VI ‖ AGDD-set)
    case: each peer's D/C come from its own view.

    For the default ``coords:[dap] channels:[vi.*]`` main view this is a no-op
    numerically (``D==1`` matches the shipped ``x_dim``; ``C`` equals the
    ``effective_y_dim`` ``_inject_set_encoder_ydims`` also injects for the VI
    branch), so the legacy DAP path is unchanged. The weather
    peer is a real ``dataset.views`` entry (when ``raw_weather`` is enabled),
    so this loop fills its ``params.weather_x_dim`` / ``params.weather_y_dim``
    from the assembled weather view — it is the sole source for those (the old
    ``_inject_set_encoder_ydims`` weather branch was retired in 4d).
    """
    views = getattr(dataset, "views", None) or {}
    dl_views = [v for v in views.values() if v.target == "dl"]
    if not dl_views:
        return
    train = getattr(dataset, "_train", None)
    if not train:
        return

    from utils.data.views import assemble_view

    sample = train[0]
    for view in dl_views:
        arr = assemble_view(sample, view)
        x_param, y_param = _view_dim_params(view.name)
        if OmegaConf.select(config, x_param, default="__missing__") != "__missing__":
            OmegaConf.update(config, x_param, int(arr.D), force_add=False)
        if OmegaConf.select(config, y_param, default="__missing__") != "__missing__":
            OmegaConf.update(config, y_param, int(arr.C), force_add=False)


def _validate_config(config: DictConfig, dataset) -> None:
    """Validate model/processing config consistency.

    Runs after dataset construction (so `dataset.processor.feature_dims` and
    `effective_y_dim` are populated) and before model instantiation. Catches
    config errors early instead of letting them surface as opaque failures
    deep in `instantiate(config)` or the forward pass:

    - Kernel eigen sources declaring inconsistent ``n_components`` /
      ``n_components_max`` (both set, or hard-fix exceeds effective rank).

    The check runs for every config — baselines, probe runs, and DL models
    alike — so bad n_components declarations always fail fast.

    ``params.y_dim`` is deliberately NOT validated here. The set-encoder
    dims are data-driven: ``params.y_dim`` (VI) is overwritten from the
    processor's ``effective_y_dim`` by ``_inject_set_encoder_ydims``, and
    every peer's coord/channel dims (VI ``x_dim``/``y_dim``; the weather
    peer's ``weather_x_dim``/``weather_y_dim``) from the assembled views
    by ``_inject_view_dims`` — both run immediately after this validation.
    A stale model-config y_dim — e.g. the g2f default 37 under
    ``vi_subset=ngrdi`` — is corrected, not rejected.
    """
    proc = getattr(dataset, "processor", None)
    if proc is None:
        return

    # D1 rank consistency. Mutual-exclusion config errors are caught here
    # before any feature_dims access (which would re-raise from
    # slice_components without fold context). Shared with the FPCA path
    # via utils.data.processing.rank_check.
    check_rank_consistency(config, proc)


def _any_processor_enabled(config: DictConfig) -> bool:
    """True if at least one sub-block under ``dataset.processing`` has
    ``enabled: true``.

    The default ``dataset.processing`` block lists every processor with
    ``enabled: false``, so a naive truthy check on the block itself
    would fire predict-mode cache resolution on every vanilla VI-only
    run (which never writes a ``processing_metadata.json``). Narrowing
    the gate keeps predict-mode a no-op for the vanilla path while
    still firing when any multi-modal config enables a processor.
    """
    proc_cfg = OmegaConf.select(config, "dataset.processing", default=None)
    if proc_cfg is None:
        return False
    for key in proc_cfg:
        if OmegaConf.select(proc_cfg, f"{key}.enabled", default=False):
            return True
    return False


def _resolve_processing_cache_keys(
    config: DictConfig,
) -> dict[str, str]:
    """Resolve cache keys for predict mode from training artifacts.

    Resolution order:
      1. ``cfg.misc.processing_cache_keys`` (JSON string or dict, set by
         the caller — typically via ``misc.processing_cache_keys=`` Hydra
         override).
      2. ``processing_metadata.json`` in ``cfg.misc.metrics_dirpath``,
         written by ``train.py`` alongside ``metrics.json``.
      3. (Future) checkpoint ``hyper_parameters["processor_cache_keys"]``.

    Raises:
        FileNotFoundError: if no source provides cache keys.
    """
    cli_cache_keys = OmegaConf.select(
        config, "misc.processing_cache_keys", default=None
    )
    if cli_cache_keys:
        if isinstance(cli_cache_keys, str):
            try:
                keys = json.loads(cli_cache_keys)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"misc.processing_cache_keys must be valid JSON, "
                    f"got: {cli_cache_keys!r}"
                ) from e
        else:
            keys = OmegaConf.to_container(cli_cache_keys, resolve=True) \
                if isinstance(cli_cache_keys, DictConfig) else cli_cache_keys
        if not isinstance(keys, dict):
            raise ValueError(
                f"misc.processing_cache_keys must decode to a dict, "
                f"got {type(keys).__name__}"
            )
        return {str(k): str(v) for k, v in keys.items()}

    metrics_dirpath = OmegaConf.select(config, "misc.metrics_dirpath", default=None)
    if metrics_dirpath:
        meta_path = os.path.join(metrics_dirpath, "processing_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            unpersisted = meta.get("unpersisted_processors") or []
            if unpersisted:
                raise ValueError(
                    f"Predict mode cannot restore processor(s) "
                    f"{sorted(unpersisted)}: the training run set "
                    "persist_cache=false for them, so their recorded "
                    "cache keys have no backing file on disk. Re-run "
                    "training with persist_cache=true for these "
                    "processors (or pass misc.processing_cache_keys "
                    "pointing at existing cache entries)."
                )
            keys = meta.get("processor_cache_keys")
            if isinstance(keys, dict):
                return {str(k): str(v) for k, v in keys.items()}
            raise ValueError(
                f"{meta_path} is missing a 'processor_cache_keys' dict."
            )

    raise FileNotFoundError(
        "Could not resolve processing cache keys for predict mode. Pass "
        "--processing-cache-keys '<json>' or ensure processing_metadata.json "
        f"exists at {metrics_dirpath}/processing_metadata.json."
    )


def _eval_stream_specs_from_cfg(cfg: DictConfig) -> list:
    """Parse ``dataset.eval_streams`` from the raw config into specs."""
    from utils.data.eval_streams import parse_eval_streams

    raw = OmegaConf.select(cfg, "dataset.eval_streams", default=None)
    return parse_eval_streams(raw)


# Reserved phase prefixes a stream name must not collide with (the train-set
# eval metrics log under ``train_*``; hardware telemetry under ``hardware/``;
# any ``*_step`` key is the per-batch sibling of an epoch key).
_RESERVED_STREAM_NAMES = frozenset({"train", "hardware"})


def _monitor_required(cfg: DictConfig) -> bool:
    """Whether a monitor (val) carve is needed by a downstream consumer.

    ``True`` iff best-checkpoint selection, early stopping, or a plateau LR
    scheduler (a ``scheduler_config.monitor`` key) is active — the three
    consumers of the derived ``<stream>_<metric>`` monitor key (B7). When all
    are off the validation loop needs no monitor, so an observe-only
    eval-streams bundle (no carve) is valid; ``initialize_callbacks`` then
    attaches no best/early-stop callback and ``eval.py`` scores ``last.ckpt``.
    Mirror of the ``monitor is not None`` guards in ``initialize_callbacks``.
    """
    best_enabled = bool(
        OmegaConf.select(cfg, "misc.checkpointing.best.enabled", default=False)
    )
    es_enabled = bool(
        OmegaConf.select(cfg, "misc.early_stopping.enabled", default=False)
    )
    plateau_monitor = (
        OmegaConf.select(cfg, "scheduler_config.monitor", default=None)
        is not None
    )
    return best_enabled or es_enabled or plateau_monitor


def _validate_eval_streams(cfg: DictConfig) -> None:
    """Unified stream↔scheme VALIDATE gate (raises ``ValueError``).

    Runs *before* dataset construction (it reads only the config + the
    carve/observe/metric registries), so a malformed ``eval_streams`` fails
    here with a clear message instead of deep in the splitter. Enforces:

    - exactly one ``monitor: true`` stream, and it is ``source: carve``
      (observe streams are structurally barred from selection — the leakage
      guard);
    - every stream names a non-empty ``metrics`` list of *registered* metrics;
    - stream names are unique and not a reserved phase prefix
      (``train`` / ``hardware`` / ``*_step``);
    - a ``carve`` strategy is registered and scheme-compatible (B6: ``fold``
      needs a fold-bearing scheme and ``fold != test fold``);
    - an ``observe`` selector is registered and the active scheme is in its
      ``compatible_schemes``;
    - the FPCA-basis-with-carve guardrail (B1).
    """
    specs = _eval_stream_specs_from_cfg(cfg)
    if not specs:
        return

    from utils.data.eval_streams.carve.registry import get_val_strategy_class
    from utils.data.eval_streams.observe.registry import (
        get_observe_selector_class,
    )

    scheme = OmegaConf.select(cfg, "cv_spec.scheme", default=None)

    seen: set[str] = set()
    monitor_carves: list = []
    for s in specs:
        # name legality
        if s.name in seen:
            raise ValueError(f"duplicate eval stream name {s.name!r}.")
        seen.add(s.name)
        if s.name in _RESERVED_STREAM_NAMES or s.name.endswith("_step"):
            raise ValueError(
                f"eval stream name {s.name!r} is reserved — it would collide "
                f"with the {s.name}_* / *_step metric keys. Choose another."
            )
        # metrics present + registered
        if not s.metrics:
            raise ValueError(
                f"eval stream {s.name!r} names no metrics — every stream must "
                f"declare a non-empty `metrics` list (requesting the subset "
                f"alone is not sufficient)."
            )
        for m in s.metrics:
            try:
                get_metric_class(m)
            except KeyError as e:
                raise ValueError(str(e)) from e
        # monitor ⇒ carve (leakage guard)
        if s.monitor and not s.is_carve:
            raise ValueError(
                f"eval stream {s.name!r} is source={s.source!r} but "
                f"monitor=true; only a `carve` stream may be the monitor "
                f"(observe streams cannot influence selection)."
            )
        # in_sample ⇒ observe. in_sample exempts a stream from the
        # context∩query disjointness guard (it deliberately reconstructs rows
        # in the eval context). A carve is carved OUT of train, so it is
        # disjoint by construction — marking it in_sample is meaningless, and
        # marking the monitor in_sample would silently disable the leakage
        # guard on the metric that drives selection.
        if s.in_sample and not s.is_observe:
            raise ValueError(
                f"eval stream {s.name!r} sets in_sample=true but is "
                f"source={s.source!r}; in_sample is only valid on `observe` "
                f"streams (a carve is disjoint from train by construction)."
            )
        if s.monitor:
            monitor_carves.append(s)
            _validate_monitor_metrics(s)

        if s.is_carve:
            _validate_carve_stream(s, scheme, cfg, get_val_strategy_class)
        else:
            _validate_observe_stream(s, scheme, get_observe_selector_class)

    # At most one carve stream is supported: the materializer maps every carve
    # to the single merged val split, so >1 carve would silently give each the
    # same rows. (Multiple-carve support — e.g. an OOD val + an IID val_iid —
    # needs per-carve named splits in the splitter; not yet implemented.)
    n_carve = sum(1 for s in specs if s.is_carve)
    if n_carve > 1:
        raise ValueError(
            f"eval_streams supports at most one carve stream (the monitor); "
            f"found {n_carve}. Multiple carve streams are not yet implemented."
        )

    # A monitor (val) carve is REQUIRED only when something downstream consumes
    # it: best-checkpoint selection, early stopping, or a plateau LR scheduler
    # that monitors a metric. When all three are off — e.g. the curve-track
    # random_kfold regime (train to the last epoch, constant LR, eval the final
    # last.ckpt) — an observe-only bundle with NO carve is valid: the runtime
    # runs the val loop with no monitor and initialize_callbacks attaches no
    # best/early-stop callback. Requiring it conditionally also turns the "best
    # enabled but no monitor" misconfig into a loud error here instead of a
    # silent fall-through to last.ckpt.
    if _monitor_required(cfg):
        if len(monitor_carves) != 1:
            raise ValueError(
                "a monitor:true carve stream is required because best-"
                "checkpoint / early-stopping / a plateau LR scheduler reads "
                f"it, but found {len(monitor_carves)}. Add a carve monitor "
                "stream, or disable misc.checkpointing.best.enabled + "
                "misc.early_stopping.enabled (and use a non-plateau LR) for an "
                "observe-only run."
            )
    elif n_carve == 1 and len(monitor_carves) != 1:
        raise ValueError(
            "the lone carve stream must be the monitor (observe streams cannot "
            "influence selection); set monitor: true on the carve, or drop it "
            "for an observe-only bundle."
        )

    _check_fpca_basis_guardrail(cfg, has_carve=n_carve > 0)


def _validate_monitor_metrics(spec) -> None:
    """The monitor stream's monitored + checkpoint metrics must be computed here.

    The wrapper logs ``<stream>_<metric>`` only for metrics in the stream's
    ``metrics`` list (``_eval_epoch_end``); a best-checkpoint / early-stop
    monitor keyed off a metric NOT in that list would silently never fire
    (Lightning only warns). So:

    - ``monitor_metric`` (default ``pearson_r``) must be in ``metrics``;
    - every ``checkpoint_metrics`` entry must be in ``metrics`` (each is
      transitively a registered metric, already checked for ``metrics``);
    - ``checkpoint_metrics`` must have no duplicates — two identical-monitor
      checkpoint callbacks collide on Lightning's ``state_key`` when a full
      checkpoint is saved.
    """
    available = set(spec.metrics)
    primary = spec.monitor_metric or "pearson_r"
    if primary not in available:
        raise ValueError(
            f"monitor stream {spec.name!r}: monitor_metric={primary!r} is not "
            f"in its metrics={spec.metrics!r}; the monitored key "
            f"{spec.name}_{primary} would never be logged. Add it to `metrics`."
        )
    ckpt_metrics = spec.checkpoint_metrics
    if ckpt_metrics is None:
        return
    dupes = sorted({m for m in ckpt_metrics if ckpt_metrics.count(m) > 1})
    if dupes:
        raise ValueError(
            f"monitor stream {spec.name!r}: duplicate checkpoint_metrics "
            f"{dupes!r}; each metric maps to a single best_<metric> checkpoint."
        )
    missing = [m for m in ckpt_metrics if m not in available]
    if missing:
        raise ValueError(
            f"monitor stream {spec.name!r}: checkpoint_metrics {missing!r} not "
            f"in its metrics={spec.metrics!r}; each must be a computed metric so "
            f"its `{spec.name}_<metric>` key is logged for selection."
        )


def _validate_carve_stream(spec, scheme, cfg, get_val_strategy_class) -> None:
    """Carve strategy registered + scheme-compatible (B6)."""
    strat = spec.strategy or {}
    name = strat.get("name")
    if not name:
        raise ValueError(
            f"carve stream {spec.name!r} is missing a `strategy.name`."
        )
    try:
        cls = get_val_strategy_class(name)
    except KeyError as e:
        raise ValueError(str(e)) from e
    if getattr(cls, "requires_fold", False):
        from cv.schemes import get_scheme

        if not scheme or not get_scheme(scheme).requires_fold:
            raise ValueError(
                f"carve stream {spec.name!r} uses the 'fold' strategy, which "
                f"requires a fold-bearing scheme (cv_2_1 / cv_0_00); active "
                f"scheme is {scheme!r}."
            )
        test_fold = OmegaConf.select(cfg, "cv_spec.fold", default=None)
        if strat.get("fold") == test_fold:
            raise ValueError(
                f"carve stream {spec.name!r}: val fold {strat.get('fold')!r} "
                f"must differ from the test fold {test_fold!r} (the test-fold "
                f"varieties are PREDICT, not in the train pool — the carve "
                f"would be empty)."
            )


def _validate_observe_stream(spec, scheme, get_observe_selector_class) -> None:
    """Observe selector registered + scheme-compatible."""
    from utils.data.eval_streams import observe_selector_names

    sel = spec.selector or {}
    name = sel.get("name")
    if not name:
        raise ValueError(
            f"observe stream {spec.name!r} is missing a `selector.name`."
        )
    try:
        cls = get_observe_selector_class(name)
    except KeyError as e:
        raise ValueError(
            f"unknown observe selector {name!r} — not registered; "
            f"available: {observe_selector_names()}."
        ) from e
    if not cls.is_compatible_with(scheme):
        raise ValueError(
            f"observe stream {spec.name!r} (selector {name!r}) is only valid "
            f"under scheme(s) {sorted(cls.compatible_schemes)}; active scheme "
            f"is {scheme!r}."
        )


def _check_fpca_basis_guardrail(
    cfg: DictConfig, *, has_carve: bool
) -> None:
    """B1 guardrail: a ``carve`` stream + a ``fit_mask``-consuming processor.

    The OOD val carve would leak through an unsupervised FPCA basis fit (val
    rows enter the basis), and the ``vi_fpca`` cache key does not yet encode
    the val fit-set. Until the deferred TODO lands, this is a hard config
    error. Today only ``vi_fpca`` consumes the row-level ``fit_mask``.

    ``has_carve`` gates the whole check: an observe-only bundle (e.g. the
    ``random_kfold`` eval streams) has no carve, so there is nothing to
    leak and FPCA features are fine.
    """
    # TODO(eval-streams): OOD val + FPCA-basis features — lift this guardrail
    # by extending the vi_fpca cache key to encode the val fit-set and
    # excluding val rows from calibration.
    if not has_carve:
        return
    proc_cfg = OmegaConf.select(cfg, "dataset.processing", default=None)
    if proc_cfg is None:
        return
    for name in ("vi_fpca",):
        if OmegaConf.select(proc_cfg, f"{name}.enabled", default=False):
            raise ValueError(
                f"eval_streams declares a `carve` stream while the "
                f"fit_mask-consuming processor {name!r} is enabled. The OOD "
                f"validation carve would leak through the FPCA basis (val rows "
                f"enter the basis fit), and the {name} cache key does not yet "
                f"encode the val fit-set. Extend that cache key and exclude "
                f"val rows from calibration before combining the two."
            )


def build_eval_stream_runtimes(experiment) -> list[EvalStreamRuntime]:
    """Resolve the dataset's eval-stream specs into ``EvalStreamRuntime``s.

    Each stream gets a full :class:`PhaseConfig` built from the shared
    ``phase_configs.eval`` template (forward/metric wrappers + ``mode: eval``)
    with the stream's *own* ``metrics`` list filling ``metric_fn``. The list
    order is the canonical ``dataloader_idx`` order. Empty when no streams are
    configured (predict-mode / baselines).
    """
    specs = list(getattr(experiment.dataset, "eval_stream_specs", []) or [])
    if not specs:
        return []
    eval_template = getattr(experiment.phase_configs, "eval", None)
    if eval_template is None:
        raise ValueError(
            "eval_streams are configured but no `phase_configs.eval` template "
            "is defined. Add a `phase_configs.eval` block (forward/metric "
            "wrappers + `metric_kwargs.mode: eval`) to your model YAML."
        )
    runtimes: list[EvalStreamRuntime] = []
    for spec in specs:
        config = PhaseConfig(
            forward_wrapper=eval_template.forward_wrapper,
            metric_wrapper=eval_template.metric_wrapper,
            metric_fn=build_metric(spec.metrics),
            forward_kwargs=dict(eval_template.forward_kwargs),
            metric_kwargs=dict(eval_template.metric_kwargs),
        )
        runtimes.append(
            EvalStreamRuntime(
                name=spec.name,
                config=config,
                monitor=spec.monitor,
                in_sample=spec.in_sample,
            )
        )
    return runtimes


def _metric_mode(metric: str, override: str | None = None) -> str:
    """Optimization mode for ``metric``: explicit ``override`` else the registry.

    Precedence: an explicit ``monitor_mode`` override (back-compat / escape
    hatch) wins; otherwise the metric's own ``direction`` ClassVar (the single
    source of truth — ``rmse`` is ``min``, correlations / ``loglik`` are
    ``max``); falling back to ``"max"`` only for a metric the registry cannot
    place (so an unknown custom metric stays usable, as before).
    """
    if override:
        return override
    try:
        return metric_direction(metric)
    except (KeyError, ValueError):
        return "max"


def _derive_monitor(experiment) -> tuple[str, str] | None:
    """The PRIMARY ``(monitor_key, monitor_mode)`` derived from the monitor stream.

    ``f"{stream}_{metric}"`` (e.g. ``val_pearson_r``) with the stream's
    ``monitor_metric`` (default ``pearson_r``) and the mode from the metric
    registry (``monitor_mode`` overrides it). This single metric drives early
    stopping and the plateau LR scheduler — both inherently single-objective.
    ``None`` when there is no monitor stream (predict-mode / baselines / empty
    eval_streams) — no metric-monitoring callback is then attached.
    """
    specs = list(getattr(experiment.dataset, "eval_stream_specs", []) or [])
    monitors = [s for s in specs if s.monitor]
    if not monitors:
        return None
    m = monitors[0]
    metric = m.monitor_metric or "pearson_r"
    return f"{m.name}_{metric}", _metric_mode(metric, m.monitor_mode)


@dataclass
class CheckpointTarget:
    """One best-checkpoint target: a metric, its logged key, and its mode."""

    metric: str  # registry metric name, e.g. "rmse"
    key: str  # the trainer.callback_metrics key, e.g. "val_rmse"
    mode: str  # "max" | "min"


def _derive_checkpoint_targets(experiment) -> list[CheckpointTarget]:
    """Best-checkpoint targets derived from the monitor stream.

    One target per metric in the monitor stream's ``checkpoint_metrics``,
    defaulting to just the primary ``monitor_metric`` — i.e. the legacy
    single-best behaviour when the field is omitted. Each target gets an
    independent ``best_<metric>.ckpt`` (and, under freeze_on_plateau, a
    ``best_frozen_<metric>.ckpt`` that freezes on ITS OWN plateau). Modes come
    from the metric registry; the primary additionally honours an explicit
    ``monitor_mode``. Empty when there is no monitor stream.
    """
    specs = list(getattr(experiment.dataset, "eval_stream_specs", []) or [])
    monitors = [s for s in specs if s.monitor]
    if not monitors:
        return []
    m = monitors[0]
    primary = m.monitor_metric or "pearson_r"
    metrics = list(m.checkpoint_metrics) if m.checkpoint_metrics else [primary]
    return [
        CheckpointTarget(
            metric=metric,
            key=f"{m.name}_{metric}",
            mode=_metric_mode(metric, m.monitor_mode if metric == primary else None),
        )
        for metric in metrics
    ]


def resolve_scheduler_monitor(scheduler_config: dict, experiment) -> dict:
    """Sync a plateau LR scheduler's ``monitor`` with the derived monitor key.

    A ``ReduceLROnPlateau``-style scheduler carries a ``monitor`` in its
    ``scheduler_config``; it must watch the SAME key as early stopping /
    best-checkpoint, which are derived in Python from the monitor eval stream
    (B7) rather than hardcoded in YAML. This overrides the YAML's literal
    ``monitor:`` string with the derived key so a non-default eval-streams
    bundle (a different monitor stream name or metric) cannot silently desync
    the LR scheduler from model selection.

    Returns a (shallow) copy with ``monitor`` replaced when (a) the config
    declares a ``monitor`` (a plateau scheduler — step-based schedulers have
    none) and (b) a monitor stream exists. Otherwise the config is returned
    unchanged.
    """
    if "monitor" not in scheduler_config:
        return scheduler_config
    derived = _derive_monitor(experiment)
    if derived is None:
        return scheduler_config
    return {**scheduler_config, "monitor": derived[0]}


def initialize_callbacks(
    experiment: DictConfig,
) -> list[pl.callbacks.Callback]:
    """Creates a list of callbacks based on experiment configuration.

    Sets up checkpointing callbacks from ``misc.checkpointing``
    (``resume_from``, ``save_last``, ``best.enabled`` plus
    ``save_weights_only`` / ``enable_version_counter``, and the
    ``best.freeze_on_plateau`` block incl. ``extra_profiles``). The
    monitored key and mode are NOT read from the config — they are
    derived from the monitor eval stream via ``_derive_monitor`` /
    ``_derive_checkpoint_targets`` (one best_<metric>.ckpt per entry in
    the monitor stream's ``checkpoint_metrics``).

    Also includes the optional hardware performance monitoring callback.
    """
    callbacks: list[pl.callbacks.Callback] = []

    # ensure the flag always exists
    experiment.misc.default_lightning_logger = False

    checkpointing_config = getattr(experiment.misc, "checkpointing", {})
    checkpoint_dirpath = getattr(
        experiment.misc,
        "checkpoint_dirpath",
        os.path.join("artifacts", "checkpoints"),
    )
    save_weights_only = checkpointing_config.get("save_weights_only", True)
    version = checkpointing_config.get("enable_version_counter", False)

    if checkpointing_config.get("save_last", False):
        last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dirpath,
            filename="last",
            save_top_k=1,
            every_n_epochs=1,
            enable_version_counter=version,
            verbose=False,
            save_weights_only=save_weights_only,
        )
        callbacks.append(last_checkpoint_callback)

    # Monitor key/mode are DERIVED in Python from the single monitor eval
    # stream (B7) — e.g. ``val_pearson_r`` / ``max`` — not hardcoded in YAML
    # (no ${...} resolver exists). When there is no monitor stream
    # (predict-mode / baselines / empty eval_streams) no metric-monitoring
    # best-checkpoint / early-stopping callback is attached; the validation
    # loop is simply off.
    monitor = _derive_monitor(experiment)

    best_config = checkpointing_config.get("best", {}) or {}
    if best_config.get("enabled", False) and monitor is not None:
        # One best checkpoint per checkpoint-target metric (default: just the
        # primary monitor_metric → legacy single-best behaviour). Each metric is
        # selected independently and written to best_<metric>.ckpt; at eval each
        # produces its own suffixed metrics_best_<metric>.json /
        # predictions_best_<metric>.csv (no bare metrics.json — see
        # find_eval_checkpoints). Distinct monitor keys keep the callbacks'
        # Lightning state_keys distinct (the gate forbids duplicate metrics).
        targets = _derive_checkpoint_targets(experiment)
        freeze_config = best_config.get("freeze_on_plateau", {}) or {}
        freeze_enabled = freeze_config.get("enabled", False)

        # Freeze profiles: the base freeze_on_plateau fields define the DEFAULT
        # frozen set (best_frozen_<metric>); each `extra_profiles` entry adds a
        # NAMED set (best_frozen_<name>_<metric>) with its own thresholds, so
        # several plateau regimes can be compared offline from one run. Each
        # tuple is (label_name, patience, min_delta, min_delta_by_metric);
        # patience falls back to the base value when a profile omits it.
        base_patience = freeze_config.get("patience", 50)
        base_min_delta = freeze_config.get("min_delta", 0.001)
        base_by_metric = freeze_config.get("min_delta_by_metric") or {}
        freeze_profiles = [("", base_patience, base_min_delta, base_by_metric)]
        for prof in freeze_config.get("extra_profiles") or []:
            freeze_profiles.append(
                (
                    prof.get("name", ""),
                    prof.get("patience") or base_patience,
                    prof.get("min_delta", base_min_delta),
                    prof.get("min_delta_by_metric") or {},
                )
            )

        for tgt in targets:
            # The global best over ALL epochs for this metric.
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=checkpoint_dirpath,
                    filename=f"best_{tgt.metric}",
                    monitor=tgt.key,
                    mode=tgt.mode,
                    save_top_k=1,
                    enable_version_counter=version,
                    save_weights_only=save_weights_only,
                )
            )

            # `freeze_on_plateau` (opt-in): ADDITIONAL per-metric best(s) that
            # freeze once THIS metric plateaus, while training runs on to the full
            # epoch budget (the curve still logs the post-plateau trajectory).
            # Written to best_frozen[_<name>]_<metric>.ckpt so the global
            # best_<metric>.ckpt above is untouched — compare offline. Off by
            # default → only best_<metric> + last.
            if not freeze_enabled:
                continue
            for pname, ppatience, pmin_delta, pby_metric in freeze_profiles:
                label = (
                    f"best_frozen_{pname}_{tgt.metric}"
                    if pname
                    else f"best_frozen_{tgt.metric}"
                )
                frozen_checkpoint_callback = FreezableBestCheckpoint(
                    dirpath=checkpoint_dirpath,
                    filename=label,
                    monitor=tgt.key,
                    mode=tgt.mode,
                    save_top_k=1,
                    enable_version_counter=version,
                    save_weights_only=save_weights_only,
                )
                callbacks.append(frozen_checkpoint_callback)
                # Append the watcher AFTER its checkpoint so, on the freeze
                # epoch, the checkpoint records that epoch's best before freezing.
                callbacks.append(
                    PlateauFreezeCallback(
                        checkpoint=frozen_checkpoint_callback,
                        monitor=tgt.key,
                        mode=tgt.mode,
                        patience=ppatience,
                        # Per-metric min_delta (e.g. finer for rmse's ~2.0 scale);
                        # unlisted metrics fall back to the profile's min_delta.
                        min_delta=(pby_metric or {}).get(tgt.metric, pmin_delta),
                        label=label,
                    )
                )

        if freeze_enabled and getattr(
            experiment.misc, "early_stopping", {}
        ).get("enabled", False):
            pl.utilities.rank_zero_warn(
                "Both early_stopping and checkpointing.best.freeze_on_plateau "
                "are enabled; early stopping will halt training first, so the "
                "freeze (which is meant to keep training going) is moot. "
                "Disable early_stopping to observe the post-plateau trajectory."
            )

    # EarlyStopping
    es_config = getattr(experiment.misc, "early_stopping", {})
    if es_config.get("enabled", False) and monitor is not None:
        monitor_key, monitor_mode = monitor
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=monitor_key,
                patience=es_config.get("patience", 150),
                mode=monitor_mode,
                min_delta=es_config.get("min_delta", 0.001),
                verbose=True,
            )
        )

    # Hardware performance monitoring
    if getattr(experiment.misc, "hardware_callback", False):
        callbacks.append(LogPerformanceCallback())

    # Per-epoch training curve logging — records every eval stream's metrics
    # (B3), overwrite-by-default with an opt-in version counter.
    if getattr(experiment.misc, "save_training_curve", False) and hasattr(
        experiment.misc, "metrics_dirpath"
    ):
        stream_names = [
            s.name
            for s in getattr(experiment.dataset, "eval_stream_specs", []) or []
        ]
        tc_cfg = getattr(experiment.misc, "training_curve", {}) or {}
        version_curve = (
            tc_cfg.get("enable_version_counter", False)
            if hasattr(tc_cfg, "get")
            else getattr(tc_cfg, "enable_version_counter", False)
        )
        callbacks.append(
            TrainingCurveCallback(
                experiment.misc.metrics_dirpath,
                stream_names,
                enable_version_counter=bool(version_curve),
            )
        )

    # Weight averaging (EMA) — evaluates an EMA of the weights on the eval
    # streams each epoch, so the training curve reflects the averaged model.
    # Opt-in via misc.weight_averaging.enabled; off by default.
    wa_config = getattr(experiment.misc, "weight_averaging", {}) or {}
    if wa_config.get("enabled", False):
        kind = wa_config.get("kind", "ema")
        if kind != "ema":
            raise ValueError(
                f"misc.weight_averaging.kind={kind!r} unsupported (only 'ema')."
            )
        callbacks.append(
            WeightAveragingCallback(decay=wa_config.get("decay", 0.99))
        )

    if callbacks:
        experiment.misc.default_lightning_logger = True

    return callbacks


def initialize_logger(cfg: DictConfig) -> pl.loggers.Logger | bool:
    """Initializes the W&B logger from the raw Hydra config.

    Takes ``cfg`` (pre-instantiation) rather than the ``experiment`` _AttrDict
    so training/eval can hoist logger creation above ``initialize_experiment``
    and capture instantiation stdout/warnings in the W&B run.
    """
    # Initialize wandb only when explicitly enabled for metrics logging
    if getattr(cfg.misc, "wandb_logging_enabled", False):
        wandb_options = {
            "project": cfg.misc.project,
            "name": cfg.misc.name,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "log_model": False,
        }

        # any extra W&B settings
        if settings := getattr(cfg.misc, "wandb_settings", None):
            wandb_options["settings"] = settings

        # CV fold grouping: group all folds of the same (model, seed) together
        if group := getattr(cfg.misc, "wandb_group", None):
            from utils.experiment.utils import sanitize_wandb_group

            wandb_options["group"] = sanitize_wandb_group(group)

        # CV fold tags: e.g. ["loo", "neural_process", "fold=DEH1", "seed=42"]
        if tags := getattr(cfg.misc, "wandb_tags", None):
            from utils.experiment.utils import sanitize_wandb_tags

            wandb_options["tags"] = sanitize_wandb_tags(tags)

        return pl.loggers.WandbLogger(**wandb_options)

    # Whether to use lightning default local logger for logs without wandb
    return cfg.misc.default_lightning_logger


def _seed_worker(worker_id: int) -> None:
    """Re-seed numpy's and stdlib ``random``'s GLOBAL RNGs per DataLoader worker.

    PyTorch (>=1.9) auto-seeds each worker's ``torch`` RNG uniquely
    (``base_seed + worker_id``), but it does NOT touch numpy's legacy global RNG
    or the stdlib ``random`` module — on ``fork`` those are COPIED, so every
    worker would share identical state and a ``Dataset.__getitem__`` / collate
    that called ``np.random.*`` (or ``random.*``) would emit IDENTICAL draws in
    every worker (the classic duplicate-batch bug). Deriving a unique per-worker
    seed from ``torch.initial_seed()`` (already per-worker) makes any such
    randomness both fork-safe (distinct per worker) and reproducible under
    ``misc.seed``.

    The current standard path uses no numpy-global randomness (augmentation uses
    ``np.random.default_rng()``, which self-seeds from entropy per call, and the
    TaskSampler seeds its own ``default_rng`` explicitly), so this is a no-op
    today — it is defensive insurance so raising ``num_workers>1`` later, or
    adding numpy-global randomness to a Dataset/collate, stays correct.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = False,
    collate_fn: Callable | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Creates a DataLoader with common configurations.

    ``drop_last`` defaults to ``False`` so the final partial batch is
    always kept — dropping it would silently discard train samples and
    (on the eval path) lose predictions.

    A ``worker_init_fn`` (:func:`_seed_worker`) re-seeds the numpy/stdlib global
    RNGs per worker so ``num_workers>1`` is fork-safe against any future
    numpy-global randomness in the Dataset/collate (see its docstring).
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )


def log_hardware_info() -> None:
    """Print available hardware information to stdout (captured by W&B Logs tab)."""
    print("\n" + "=" * 50)
    print("HARDWARE INFORMATION")
    print("=" * 50)

    # System information
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    # CPU information
    cpu_count = psutil.cpu_count(logical=False)
    logical_cpu_count = psutil.cpu_count(logical=True)
    print(f"CPU Cores: {cpu_count} physical, {logical_cpu_count} logical")

    # Memory information
    memory = psutil.virtual_memory()
    print(f"Total RAM: {round(memory.total / (1024**3), 1)} GB")
    print(f"Available RAM: {round(memory.available / (1024**3), 1)} GB")

    # GPU information
    if torch.cuda.is_available():
        print("CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {round(props.total_memory / (1024**3), 1)} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("CUDA Available: No")

    # MPS (Apple Silicon) information
    if torch.backends.mps.is_available():
        print("MPS Available: Yes (Apple Silicon)")
    else:
        print("MPS Available: No")

    print("=" * 50)


def log_training_config(
    trainer: pl.Trainer,
    epochs: int,
    *,
    train_dataloader: torch.utils.data.DataLoader | None = None,
    val_dataloader: torch.utils.data.DataLoader | None = None,
) -> None:
    """Print training configuration to stdout (captured by W&B Logs tab)."""
    print("Training Configuration:")

    def _batches(dl):
        if dl is None:
            return "split does not exist"
        n = len(dl)
        return "0 (empty split)" if n == 0 else str(n)

    print(f"  Accelerator: {trainer.accelerator.__class__.__name__}")
    print(f"  Devices: {trainer.num_devices}")
    print(f"  Max Epochs: {epochs}")
    print(f"  Train Batches per Epoch: {_batches(train_dataloader)}")
    print(f"  Val Batches per Epoch: {_batches(val_dataloader)}")
    print("=" * 50 + "\n")
