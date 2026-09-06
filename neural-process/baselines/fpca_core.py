"""Shared core for the FPCA + regression baseline.

Provides pure functions taking a resolved Hydra ``DictConfig`` (or
plain dict post-``_convert_="all"``) and dispatches train/predict.
The ``fpca_train.py`` and ``fpca_predict.py`` modules are
``@hydra.main`` shims that delegate here.
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from utils.data.dataset import G2FDataset
from utils.data.processing.rank_check import check_rank_consistency

from .regressors import build_regressor


# ── Helpers ─────────────────────────────────────────────────────


def _dump_processor_intermediates(keep_dir: str, ctx, dataset) -> None:
    """Write per-processor pre-kernel state for cross-pipeline diagnostics.

    Gated by ``BGLR_DUMP_INTERMEDIATES=1`` and piggybacks on the existing
    ``BGLR_KEEP_BLOCKS_DIR`` dump. For each relevant processor that's
    been fit, emits the matrices BGLR's K_blocks are derived from:

    - ``PhenomicFeatureBuilder``:
        ``phenomic_VI_fpc_raw.npy``   — (N, D) raw vi_fpc_scores
        ``phenomic_VI_all.npy``       — (N, D) per-env z-scored matrix
        ``phenomic_VI_md_idx.npy``    — (N,) metadata_df row index per K row
        ``phenomic_meta.json``        — scaling + fit_scope + shape
    - ``WeatherKernelProcessor``:
        ``weather_env_score_raw.npy``    — (n_envs, D) raw per-env scores
        ``weather_env_score_scaled.npy`` — (n_envs, D) after z-score
        ``weather_K_env.npy``            — (n_envs, n_envs) env-level K_W
        ``weather_meta.json``            — env_order + scaling + fit_scope

    Row order for phenomic matrices: concatenation of
    ``ctx.split_indices['train', 'val', 'test']`` in iter order — exactly
    what ``PhenomicFeatureBuilder.fit`` stacks into ``train_scores_scaled``
    under ``fit_scope='all'``. The diff script aligns Python rows to
    the R reference's ``order`` (Env-then-Pedigree) by Pedigree.Env string match.
    """
    procs: list = []
    if getattr(dataset, "processor", None) is not None:
        procs = dataset.processor._processors  # noqa: SLF001
    proc_map = {n: p for n, _, p in procs}

    pheno = proc_map.get("phenomic")
    if pheno is not None and getattr(pheno, "_state", None):
        scalings = list(pheno._state.keys())
        if len(scalings) != 1:
            print(
                f"[FPCA] dump_intermediates: phenomic has "
                f"{len(scalings)} scalings; expected exactly 1 for weather-GxE."
            )
        else:
            scaling = scalings[0]
            state = pheno._state[scaling]
            np.save(
                os.path.join(keep_dir, "phenomic_VI_all.npy"),
                state["train_scores_scaled"],
            )
            chunks: list[np.ndarray] = []
            md_idx: list[int] = []
            for split in ("train", "val", "test"):
                idxs = ctx.split_indices.get(split, [])
                if not idxs:
                    continue
                chunks.append(
                    ctx.stack_derived_feature("vi_fpc_scores", split)
                )
                md_idx.extend(idxs)
            np.save(
                os.path.join(keep_dir, "phenomic_VI_fpc_raw.npy"),
                np.concatenate(chunks, axis=0),
            )
            np.save(
                os.path.join(keep_dir, "phenomic_VI_md_idx.npy"),
                np.asarray(md_idx, dtype=np.int64),
            )
            with open(
                os.path.join(keep_dir, "phenomic_meta.json"), "w",
            ) as f:
                json.dump(
                    {
                        "scaling": scaling,
                        "fit_scope": pheno.fit_scope,
                        "kernel": pheno.kernel,
                        "center_kernel": bool(pheno.center_kernel),
                        "n_obs": int(state["train_scores_scaled"].shape[0]),
                        "n_fpcs": int(state["train_scores_scaled"].shape[1]),
                    },
                    f,
                    indent=2,
                )
            print(
                "[FPCA] dump_intermediates: wrote phenomic VI_all + "
                "raw FPC scores"
            )

    wk = proc_map.get("weather_kernel")
    if wk is not None and getattr(wk, "_state", None):
        scalings = list(wk._state.keys())
        if len(scalings) != 1:
            print(
                f"[FPCA] dump_intermediates: weather_kernel has "
                f"{len(scalings)} scalings; expected exactly 1 for weather-GxE."
            )
        else:
            scaling = scalings[0]
            state = wk._state[scaling]
            np.save(
                os.path.join(keep_dir, "weather_env_score_raw.npy"),
                wk._all_env_score_matrix,
            )
            np.save(
                os.path.join(keep_dir, "weather_env_score_scaled.npy"),
                wk._scaled_fit_matrix(scaling),
            )
            np.save(
                os.path.join(keep_dir, "weather_K_env.npy"),
                state["K_centered"],
            )
            with open(
                os.path.join(keep_dir, "weather_meta.json"), "w",
            ) as f:
                json.dump(
                    {
                        "scaling": scaling,
                        "fit_scope": wk.fit_scope,
                        "center_kernel": bool(wk.center_kernel),
                        "all_env_order": list(wk._all_env_order),
                        "fit_env_order": list(wk._fit_env_order),
                    },
                    f,
                    indent=2,
                )
            print(
                "[FPCA] dump_intermediates: wrote weather env_score "
                "raw + scaled + K_env"
            )


# Label-based scoring is shared verbatim with the DL eval path —
# see cv/scoring.py. Re-exported here under the historical private names so
# existing call sites (``train`` below) keep importing them from
# ``baselines.fpca_core``. The single home for the label==position alignment
# convention is cv.scoring.
from cv.scoring import (  # noqa: E402
    aligned_full_predictions as _aligned_full_predictions,
    block_array as _block_array,
    compute_metrics as _compute_metrics,
    label_array as _label_array,
    score_by_label as _score_by_label,
    write_predictions as _write_predictions,
)


def _to_plain(cfg: Any) -> dict:
    """Resolve a DictConfig to a plain dict; pass plain dicts through."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def _build_dataset(config_dict: dict) -> G2FDataset:
    """Instantiate G2FDataset with baseline-specific overrides.

    - Normalization disabled for all three fields (regressors handle scaling).
    - No eval streams: baselines have no early stopping, so all non-test data
      is used for fitting (no val carve). This differs from the DL pipeline
      (which configures a `carve` eval stream) and is documented in
      ``baselines/README.md``.
    """
    ds_cfg = config_dict.get("dataset", {}) or {}
    processing_cfg = ds_cfg.get("processing")
    # cv_spec (D9) lives at config root; when present it drives a
    # role-based split and test_filter_columns is ignored.
    cv_spec = config_dict.get("cv_spec")

    return G2FDataset(
        data_dir=ds_cfg.get(
            "data_dir", "./data/Pedigrees_Wide_Format_BLUEs"
        ),
        validate=ds_cfg.get("validate", True),
        case_sensitive=ds_cfg.get("case_sensitive", False),
        delete_invalid=ds_cfg.get("delete_invalid", True),
        to_tensor=True,
        random_state=ds_cfg.get("random_state", 0),
        test_ratio=ds_cfg.get("test_ratio", 0.1),
        eval_streams=None,  # baselines don't carve a val split — see docstring
        test_filter_columns=ds_cfg.get("test_filter_columns", None),
        normalize={
            "dap": False,
            "channels": False,
            "weather_concat_values": False,
            "yield_value": False,
        },
        smoke_n=ds_cfg.get("smoke_n", None),
        processing=processing_cfg,
        cv_spec=cv_spec,
    )


def _collect_xy(
    split_data: list[dict],
    feature_keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``derived_features[key]`` across samples and hstack in key order.

    Float64 promotion is required — sklearn's LinearRegression silently
    accepts float32 X but produces a numerically poor fit for
    ill-conditioned designs (observed as a coefficient-norm collapse on
    the FPCA + OLS baseline).
    """
    if not split_data:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    y = np.asarray(
        [s["yield_value"].item() for s in split_data], dtype=np.float64
    )

    parts = []
    for key in feature_keys:
        stacked = np.stack(
            [s["derived_features"][key].numpy() for s in split_data]
        )
        parts.append(stacked)
    X = np.hstack(parts).astype(np.float64)
    return X, y


def _resolve_feature_keys(
    dataset: G2FDataset, reg_cfg: dict
) -> list[str]:
    """Pick which ``derived_features`` keys to use."""
    if not dataset._train:
        raise RuntimeError("Train split is empty — cannot resolve feature keys.")

    sample_feats = dataset._train[0].get("derived_features", {}) or {}
    if not sample_feats:
        raise RuntimeError(
            "No derived_features on train samples.  Enable at least one "
            "processor under dataset.processing (e.g. vi_fpca.enabled=true)."
        )

    configured = reg_cfg.get("feature_keys")
    if configured:
        missing = [k for k in configured if k not in sample_feats]
        if missing:
            raise KeyError(
                f"feature_keys {missing} not present on samples. "
                f"Available: {list(sample_feats.keys())}"
            )
        return list(configured)
    return list(sample_feats.keys())


# ── Train ───────────────────────────────────────────────────────


def train(cfg: DictConfig | dict) -> dict:
    """Train a regressor on pre-computed derived features.

    Returns the metrics dict that is also persisted to disk at
    ``misc.metrics_dirpath/metrics.json``.
    """
    config_dict = _to_plain(cfg)

    reg_cfg = config_dict.get("regressor", {"name": "ols"}) or {"name": "ols"}
    misc_cfg = config_dict.get("misc", {}) or {}
    seed = misc_cfg.get("seed", 0)
    output_dir = misc_cfg.get("artifacts_dir", "artifacts/fpca-baseline/seed=0")
    metrics_dir = misc_cfg.get("metrics_dirpath", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Mutual-exclusion pre-check on kernel source configs before we
    # even touch the dataset. This mirrors the DL path's
    # _validate_config rank check (which FPCA does not traverse since
    # it has its own setup flow). Sub-check 5a (mutual exclusion) runs here with no
    # processor argument; sub-check 5b (hard-fix vs rank) re-runs
    # after the processor fit below with a real feature_dims dict.
    check_rank_consistency(config_dict, processor=None)

    print("[FPCA] Loading dataset (feature processing runs inside G2FDataset)...")
    dataset = _build_dataset(config_dict)
    print(
        f"[FPCA] Splits: train={len(dataset._train)}, test={len(dataset._test)}"
    )

    # Sub-check 5b: now that the processor has fit, re-run
    # the rank check with a real feature_dims dict. A hard-fix
    # n_components > effective_rank raises with fold-named context
    # before we spend any more time on _collect_xy / regressor fit.
    if getattr(dataset, "processor", None) is not None:
        check_rank_consistency(config_dict, dataset.processor)

    feature_keys = _resolve_feature_keys(dataset, reg_cfg)
    print(f"[FPCA] Using feature keys: {feature_keys}")

    # Propagate `misc.seed` into the regressor cfg as `seed` so the
    # stochastic regressor factories in `baselines/regressors.py`
    # (`gpr`, `bglr`, `gblup`) receive a seed that tracks
    # the run-level `misc.seed` override. Before this, every stochastic
    # regressor silently used `random_state=0` regardless of the
    # `misc.seed=N` CLI argument — a latent bug surfaced during the
    # post-merge Category B fix. `setdefault` lets an explicit
    # `regressor.seed=<value>` in the model fragment or CLI override
    # still win over `misc.seed`. For BGLR this seed flows into
    # `manifest.json["seed"]` and drives `set.seed()` in
    # `bglr_compute.R` BEFORE ETA construction.
    reg_cfg.setdefault("seed", seed)
    # The factory may need feature_keys for cross-checks (e.g. BGLR
    # validates env_brr_key in feature_keys at config time).
    reg_cfg.setdefault("feature_keys", feature_keys)
    model = build_regressor(reg_cfg)

    bglr_meta: dict | None = None

    if getattr(model, "is_transductive", False):
        from utils.data.processing.base import OrchestratorContext

        env_key = reg_cfg.get("env_brr_key")
        y_train = np.asarray(
            [s["yield_value"].item() for s in dataset._train],
            dtype=np.float64,
        )
        y_test = np.asarray(
            [s["yield_value"].item() for s in dataset._test],
            dtype=np.float64,
        )
        train_idx = list(dataset.split_indices.get("train", []))
        test_idx = list(dataset.split_indices.get("test", []))
        sample_indices = np.array(train_idx + test_idx, dtype=np.int64)
        n_train = len(y_train)
        n_test = len(y_test)

        # Build a shared context exposing the source-name → producer
        # registry so each producer's `raw_kernel(...)` works.
        ctx = OrchestratorContext(
            dataset=dataset,
            proc_config={},
            cache_dir=None,
            feature_to_processor=(
                dataset.processor.feature_to_processor
                if dataset.processor is not None else {}
            ),
        )

        # Build the raw obs-level kernel for each non-env feature_key
        # via its producing processor's raw_kernel.
        K_blocks: dict[str, np.ndarray] = {}
        for key in feature_keys:
            if key == env_key:
                continue
            producer = ctx.feature_to_processor.get(key)
            if producer is None:
                raise KeyError(
                    f"BGLR transductive path: feature_key {key!r} has "
                    f"no registered producer. Check that the upstream "
                    f"processor is enabled and declares this source name."
                )
            K_blocks[key] = producer.raw_kernel(ctx, key, sample_indices)

        # Env block: keep as a one-hot design matrix (handed to BRR,
        # not a kernel). Built directly from the producer's vocab so
        # it spans the requested row order without going through
        # derived_features.
        env_block = None
        if env_key:
            env_producer = ctx.feature_to_processor.get(env_key)
            if env_producer is None:
                raise KeyError(
                    f"BGLR transductive path: env_brr_key={env_key!r} "
                    f"has no registered producer."
                )
            env_src = next(
                s for s in env_producer.sources if s["name"] == env_key
            )
            column = env_src["column"]
            from utils.data.processing.metadata_features import _ENV_YEAR_KEY
            if column == _ENV_YEAR_KEY:
                series = ctx.env_year()
            else:
                series = ctx.metadata_df[column].astype(str)
            values = series.loc[sample_indices].tolist()
            vocab = env_producer._vocab[env_key]
            env_block = env_producer._onehot(
                values, vocab, env_key,
            ).astype(np.float64)

        n_blocks = len(K_blocks)
        env_msg = (
            f", + env BRR ({env_block.shape[1]} cols)"
            if env_block is not None else ""
        )
        print(
            f"[FPCA] Fitting {reg_cfg.get('name', 'bglr')} "
            f"(transductive): {n_blocks} RKHS kernel blocks, "
            f"n_train={n_train}, n_test={n_test}{env_msg}"
        )

        # Optional dump of K_blocks + side data right before the BGLR call.
        # Used for diffing Python's BGLR inputs against the R reference
        # pipeline (Pass 2).
        keep_dir = os.environ.get("BGLR_KEEP_BLOCKS_DIR")
        if keep_dir:
            os.makedirs(keep_dir, exist_ok=True)
            np.save(os.path.join(keep_dir, "sample_indices.npy"),
                    sample_indices)
            np.save(os.path.join(keep_dir, "y_train.npy"), y_train)
            np.save(os.path.join(keep_dir, "y_test.npy"), y_test)
            if env_block is not None:
                np.save(os.path.join(keep_dir, "env_block.npy"), env_block)
            for key, K in K_blocks.items():
                np.save(os.path.join(keep_dir, f"K_{key}.npy"), K)
            meta_cols = ["Pedigree", "Env"]
            if "Year" in ctx.metadata_df.columns:
                meta_cols.append("Year")
            meta_rows = ctx.metadata_df.iloc[sample_indices][
                meta_cols
            ].reset_index(drop=True)
            # The R reference's Pedigree.Env convention is <ped>.<env_loc>.<year>,
            # matching its Yield CSV where Env = "<location>.<year>".
            # Python's metadata_df splits location + year; rebuild
            # the R reference's format here so the cross-pipeline diff can align
            # rows by string match.
            if "Year" in meta_cols:
                meta_rows["Pedigree.Env"] = (
                    meta_rows["Pedigree"].astype(str) + "."
                    + meta_rows["Env"].astype(str) + "."
                    + meta_rows["Year"].astype(str)
                )
            else:
                meta_rows["Pedigree.Env"] = (
                    meta_rows["Pedigree"].astype(str) + "."
                    + meta_rows["Env"].astype(str)
                )
            meta_rows.to_csv(
                os.path.join(keep_dir, "sample_meta.csv"), index=False,
            )
            dump_meta = {
                "feature_keys": list(K_blocks.keys()),
                "env_key": env_key,
                "n_train": int(n_train),
                "n_test": int(n_test),
                "env_block_cols": (
                    int(env_block.shape[1])
                    if env_block is not None else 0
                ),
            }
            with open(os.path.join(keep_dir, "dump_meta.json"), "w") as f:
                json.dump(dump_meta, f, indent=2)
            print(
                f"[FPCA] Dumped K_blocks + side data to {keep_dir}"
            )
            if os.environ.get("BGLR_DUMP_INTERMEDIATES") == "1":
                _dump_processor_intermediates(keep_dir, ctx, dataset)
            if os.environ.get("BGLR_DUMP_AND_EXIT") == "1":
                print("[FPCA] BGLR_DUMP_AND_EXIT=1 — skipping BGLR call")
                return {}

        y_pred_train, y_pred_test, bglr_meta = model.fit_predict_blocks(
            K_blocks, y_train, n_test, env_block=env_block,
        )
        # Name the meta artifact + the meta_dict sub-key after the
        # regressor so every transductive method (bglr, gblup, …) gets
        # its own slot. Defaulting to "bglr" keeps the BGLR outputs
        # (bglr_meta.json, meta.bglr) byte-identical to before.
        transductive_name = reg_cfg.get("name", "bglr")
        meta_path = os.path.join(
            output_dir, f"{transductive_name}_meta.json"
        )
        with open(meta_path, "w") as f:
            json.dump(bglr_meta, f, indent=2)
        print(f"[FPCA] Saved {transductive_name} meta to {meta_path}")
    else:
        X_train, y_train = _collect_xy(dataset._train, feature_keys)
        X_test, y_test = _collect_xy(dataset._test, feature_keys)
        print(
            f"[FPCA] Fitting {reg_cfg.get('name', 'ols')}: "
            f"X_train={X_train.shape}, y_train={y_train.shape}"
        )
        model.fit(X_train, y_train)
        model_path = os.path.join(output_dir, "regression_model.joblib")
        joblib.dump(model, model_path)
        print(f"[FPCA] Saved regression model to {model_path}")
        y_pred_train = model.predict(X_train) if len(y_train) else None
        y_pred_test = model.predict(X_test) if len(y_test) else None

    # Evaluate by cv_label (D2): scatter train+test predictions back to
    # metadata_df positions, join the per-row cv_label, and score each
    # quadrant — including the in-sample CV2/CV0 tested-line metrics over
    # FIT rows — as a cor/RMSE over its label-masked slice. env_year_loo is
    # the degenerate one-label ("test") case.
    labels = _label_array(dataset)
    y_true_all, y_pred_all = _aligned_full_predictions(
        dataset,
        np.asarray(y_train, dtype=np.float64),
        None if y_pred_train is None else np.asarray(y_pred_train, np.float64),
        np.asarray(y_test, dtype=np.float64),
        None if y_pred_test is None else np.asarray(y_pred_test, np.float64),
    )
    by_metric = _score_by_label(
        labels, y_true_all, y_pred_all, blocks=_block_array(dataset)
    )
    for label, m in by_metric.items():
        print(
            f"[FPCA] {label}: N={m['n']}, RMSE={m['rmse']:.4f}, "
            f"Pearson r={m['pearson_r']:.4f}, "
            f"Spearman r={m['spearman_r']:.4f}"
        )

    processor_meta: dict = {}
    if dataset.processor is not None:
        processor_meta = {
            "enabled_processors": dataset.processor.enabled_names,
            "feature_dims": dataset.processor.feature_dims,
            "effective_y_dim": dataset.processor.effective_y_dim,
        }

    _bare_model_name = config_dict.get(
        "model_name", f"fpca_{reg_cfg.get('name', 'ols')}"
    )
    meta_dict: dict = {
        "model_name": _bare_model_name,
        # Architecture + subset tags (no seed/fold) — the
        # aggregation-stable identity. Falls back to the bare name for
        # configs composed without the misc.model_slug interpolation.
        "model_slug": misc_cfg.get("model_slug", _bare_model_name),
        "seed": seed,
        "regressor": reg_cfg,
        "feature_keys": feature_keys,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "processing": processor_meta,
    }
    if bglr_meta is not None:
        # Keyed by regressor name (defaulting to "bglr") so meta.bglr is
        # unchanged for BGLR and meta.gblup carries the REML summary for
        # GBLUP. n_iter/burn_in are BGLR-only (None for gblup); varE and
        # term_var are shared — for gblup, varE is the REML residual
        # variance and term_var holds the per-kernel variance components.
        meta_dict[reg_cfg.get("name", "bglr")] = {
            "n_iter": bglr_meta.get("n_iter"),
            "burn_in": bglr_meta.get("burn_in"),
            "varE": bglr_meta.get("varE"),
            # Per-term posterior variance. Field is `var` regardless
            # of the term's BGLR model type — RKHS terms back it
            # with `fit$ETA[[k]]$varU`, BRR terms with
            # `fit$ETA[[k]]$varB`. The R driver normalizes both into
            # a single `var` key per term.
            "term_var": {
                t["name"]: t["var"]
                for t in bglr_meta.get("terms", [])
            },
        }
    output = {"meta": meta_dict, "by_metric": by_metric}

    # Write predictions.csv BEFORE metrics.json. Submit scripts use
    # `[[ -f metrics.json ]]` as the idempotency skip marker, so metrics.json
    # must be the LAST artifact written — otherwise an interrupt between the
    # two writes leaves the fold marked "done" but missing predictions.csv,
    # and the rerun skips it, leaving the fold without per-row predictions.
    # Predictions are a downstream-aggregator convenience, so a failure here
    # is still warn-only — but it must not gate metrics.json existence.
    try:
        _write_predictions(
            metrics_dir, dataset, labels, y_true_all, y_pred_all,
        )
    except Exception as e:
        print(f"[FPCA] WARNING: failed to write predictions.csv: {e}")

    # metrics.json is the load-bearing artifact / skip marker: write it
    # atomically (tmp + os.replace) so a kill mid-dump cannot leave a
    # truncated file that both blocks the rerun and fails JSON parsing.
    metrics_path = os.path.join(metrics_dir, "metrics.json")
    tmp_path = metrics_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_path, metrics_path)
    print(f"[FPCA] Saved metrics to {metrics_path}")

    # W&B logging — flatten the per-label metrics to ``<label>/<metric>``
    # scalars on run.summary, matching the DL pipeline
    # (see utils/experiment/utils.py::_log_to_wandb).
    if misc_cfg.get("wandb_logging_enabled", False):
        from utils.experiment.utils import sanitize_wandb_tags

        run = wandb.init(
            project=misc_cfg.get("project"),
            name=misc_cfg.get("name"),
            group=misc_cfg.get("wandb_group"),
            tags=sanitize_wandb_tags(misc_cfg.get("wandb_tags", [])),
            config=config_dict,
        )
        flat = {
            f"{label}/{k}": v
            for label, m in by_metric.items()
            for k, v in m.items()
        }
        if flat:
            try:
                run.summary.update(flat)
                print(f"Logged {len(flat)} metrics to W&B summary")
            except Exception as e:
                print(f"Failed to log results to W&B: {e}")
        wandb.finish()

    return output


# ── Predict (stub) ─────────────────────────────────────────────


def predict(cfg: DictConfig | dict) -> None:
    """Predict entry point (stub).

    Full predict-mode support for FPCA baselines is unimplemented.
    The DL path uses ``eval.py`` with ``FeatureProcessor.load_and_transform``;
    the FPCA equivalent would mirror that flow.
    """
    raise NotImplementedError(
        "predict mode is not yet implemented in the unified FPCA baseline. "
        "Use eval.py for DL models, or extend this stub via "
        "FeatureProcessor.load_and_transform() + the cached regression model."
    )
