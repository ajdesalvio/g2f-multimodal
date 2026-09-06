"""Structured schema for `misc.*`.

`conf/misc/base.yaml` overrides field-by-field for any value that
differs from these dataclass defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FreezeProfileConfig:
    # An ADDITIONAL frozen-checkpoint set with its own plateau thresholds,
    # written to best_frozen_<name>_<metric>.ckpt (the name is folded into the
    # label so eval scores it to metrics_best_frozen_<name>_<metric>.json).
    name: str = "alt"
    patience: Optional[int] = None  # None -> inherit FreezeOnPlateauConfig.patience
    min_delta: float = 0.001
    min_delta_by_metric: Dict[str, float] = field(default_factory=dict)


@dataclass
class FreezeOnPlateauConfig:
    # Opt-in: once the derived monitor has plateaued (same patience / min_delta
    # test as early stopping), stop updating best.ckpt while training continues
    # to the full epoch budget. Off by default → best.ckpt is the global best.
    enabled: bool = False
    patience: int = 50
    min_delta: float = 0.001
    # Per-metric overrides of `min_delta` (metric name -> delta); any metric not
    # listed falls back to `min_delta`. Lets a metric on a different scale (e.g.
    # rmse ~2.0) use a finer plateau threshold than pearson_r (~0-1).
    min_delta_by_metric: Dict[str, float] = field(default_factory=dict)
    # Extra frozen-checkpoint sets, each with its own thresholds. The base fields
    # above are the default set (best_frozen_<metric>); each entry here adds a
    # best_frozen_<name>_<metric> set that freezes on its own plateau test.
    extra_profiles: List[FreezeProfileConfig] = field(default_factory=list)


@dataclass
class BestCheckpointConfig:
    enabled: bool = False
    monitor: Optional[str] = None
    mode: Optional[str] = None
    freeze_on_plateau: FreezeOnPlateauConfig = field(
        default_factory=FreezeOnPlateauConfig
    )


@dataclass
class CheckpointingConfig:
    resume_from: Optional[str] = None
    save_last: bool = True
    best: BestCheckpointConfig = field(default_factory=BestCheckpointConfig)
    save_weights_only: bool = True
    enable_version_counter: bool = False


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    # `monitor` / `mode` are now DERIVED in Python from the monitor eval
    # stream (B7); these fields are retained for back-compat but ignored by
    # `initialize_callbacks`.
    monitor: str = "val_pearson_r"
    patience: int = 150
    mode: str = "max"
    min_delta: float = 0.001


@dataclass
class WeightAveragingConfig:
    # When enabled, an EMA of the weights is evaluated on the eval streams each
    # epoch (the training curve reflects the AVERAGED model). Decoupled from
    # optimization; intended to pair with a constant LR. Off by default — no
    # effect on existing runs. `kind` is currently only "ema".
    enabled: bool = False
    kind: str = "ema"
    decay: float = 0.99


@dataclass
class TrainingCurveConfig:
    # Overwrite-by-default (mirrors checkpointing.enable_version_counter): a
    # rerun of the same (model, seed) replaces training_curve.jsonl. Opt into
    # versioned files (training_curve-v1.jsonl, …) to preserve attempts.
    enable_version_counter: bool = False


@dataclass
class MiscConfig:
    experiment_name: str = "g2f-multimodal"
    # `fold_env` holds the compound `ENV.YEAR` string (e.g. `DEH1.2020`).
    # Default "" (not None) — the `cond` resolver treats "" as falsy to
    # drop the `-fold=...` suffix from `model_run_name`.
    fold_env: str = ""
    # Per-encoder time/channel pieces are interpolated inline into the peer
    # tokens in `model_name` (e.g. `vi.t=${vi_time}.c=${vi_chan}${vi_aug}`), so
    # the run name / slug no longer append separate subset/axis tags.
    model_run_name: str = (
        '${model_name}${oc.select:hp_tag,""}'
        '-seed=${.seed}${cond:${.fold_env},"-fold=${.fold_env}"}'
        '${oc.select:eval_tag,""}${oc.select:wa_tag,""}'
    )
    # Architecture + hp tag, without the seed / fold suffix — the
    # aggregation-stable identity written into `metrics.json`
    # `meta.model_slug`. The per-encoder time/channel/axis identity is already
    # baked into `model_name`. See `conf/misc/base.yaml` for the rationale.
    model_slug: str = (
        '${model_name}${oc.select:hp_tag,""}'
        '${oc.select:eval_tag,""}${oc.select:wa_tag,""}'
    )
    wandb_logging_enabled: bool = True
    # Separate toggle for the eval.py / run_eval pass (the DL scoring run).
    # Default False: evaluation never logs to W&B unless explicitly enabled,
    # independent of wandb_logging_enabled (which governs the training run).
    wandb_eval_logging_enabled: bool = False
    wandb_user: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_run_id: Optional[str] = None
    project: str = "${.data}-${scheme_label:${oc.select:cv_spec.scheme,env_year_loo}}"
    name: str = "${.model_run_name}"
    eval_name: str = "${.name}_eval"
    data: str = "G2F"
    # Cross-group root anchor `${model_name}` is resolved lazily when an
    # experiment group supplies it.
    artifacts_dir: str = "artifacts/${model_name}/seed=${.seed}"
    checkpoint_dirpath: str = "${.artifacts_dir}/checkpoints"
    metrics_dirpath: str = "${.artifacts_dir}/metrics"
    seed: int = 0
    epochs: int = 2500
    gradient_clip_val: float = 0.1
    check_val_every_n_epoch: int = 1
    enable_progress_bar: bool = True
    log_interval: int = 10
    hardware_callback: bool = True
    default_lightning_logger: bool = False
    save_training_curve: bool = True
    training_curve: TrainingCurveConfig = field(default_factory=TrainingCurveConfig)
    weight_averaging: WeightAveragingConfig = field(
        default_factory=WeightAveragingConfig
    )
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
