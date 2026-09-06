import json
import os
import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import lightning.pytorch as pl
import torch
import torch.nn as nn

from set_func.predictions import Prediction, cat_predictions

from ..data.base import BaseBatch
from ..data.tasks import ContextProvider, NPTaskBatch

from .forward_wrappers import ModelForwardWrapper, get_forward_wrapper
from .metric_wrappers import MetricCallWrapper, get_metric_call_wrapper
from .metrics import Metric


class PhaseConfig:
    """Configuration for a single forward+metric pass within a phase (train or test)."""

    def __init__(
        self,
        forward_wrapper: ModelForwardWrapper | None = None,
        metric_wrapper: MetricCallWrapper | None = None,
        metric_fn: Metric | None = None,
        forward_kwargs: dict | None = None,
        metric_kwargs: dict | None = None,
        name: str | None = None,
    ):
        self.forward_wrapper = forward_wrapper
        self.metric_wrapper = metric_wrapper
        self.metric_fn = metric_fn
        self.forward_kwargs = forward_kwargs or {}
        self.metric_kwargs = metric_kwargs or {}
        self.name = name


@dataclass
class EvalStreamRuntime:
    """One in-training evaluation stream as the ``LitWrapper`` sees it.

    A named, read-only dataset evaluated every val epoch, carrying its own
    resolved :class:`PhaseConfig` (the shared ``eval`` template's
    forward/metric wrappers + this stream's own ``metric_fn``) and a
    ``monitor`` flag. The list order is the canonical ``dataloader_idx``
    order ``run.py`` and the wrapper agree on. Exactly one stream is the
    monitor (always a ``carve``); every other is observe-only — its metrics
    are logged but structurally barred from driving selection (the monitor
    key is derived in ``setup.py`` and can only name the monitor stream).
    """

    name: str
    config: PhaseConfig
    monitor: bool = False
    #: Deliberately in-sample (reconstruction) stream — exempt from the
    #: context∩query disjointness guard. See ``EvalStreamSpec.in_sample``.
    in_sample: bool = False


def _is_distribution_validation_error(exc: ValueError) -> bool:
    """True if ``exc`` is ``torch.distributions``' argument-validation error.

    Raised as ``ValueError("Expected parameter <name> ... of distribution
    <D>(...) to satisfy the constraint <C>(), but found invalid values")`` when
    a NaN/inf reaches a distribution parameter. Matched on the stable part of
    that message so unrelated ValueErrors still propagate.
    """
    msg = exc.args[0] if exc.args and isinstance(exc.args[0], str) else ""
    return "to satisfy the constraint" in msg and "Expected parameter" in msg


class LitWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for Neural Process models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        train_config: PhaseConfig | list[PhaseConfig] | None = None,
        test_config: PhaseConfig | list[PhaseConfig] | None = None,
        eval_streams: "list[EvalStreamRuntime] | None" = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_config: dict | None = None,
        save_hyperparams: bool = True,
    ) -> None:
        """Initialize the Lightning wrapper.

        Args:
            model: Neural Process model to wrap.
            optimizer: Optimizer for model training.
            train_config: Configuration(s) for training phase. Can be a single PhaseConfig or list.
            test_config: Configuration(s) for the predict/test hooks. ``predict_step`` /
                ``on_test_epoch_end`` read ``test_config[0].forward_wrapper``; ``run_eval``
                loads the checkpoint with the shared ``eval`` template here (B12). Retained
                (not removed) by the eval-streams refactor. Also the source of the train-set
                eval metrics (``_train_eval_configs``, B8).
            eval_streams: Ordered list of :class:`EvalStreamRuntime` — the in-training
                evaluation streams (carve + observe), one per val dataloader, in the
                canonical ``dataloader_idx`` order. Replaces the former single ``val_config``.
                Each carries its own resolved ``PhaseConfig`` and a ``monitor`` flag.
            scheduler: Optional LR scheduler instance.
            lr_scheduler_config: Lightning scheduler config dict (interval, frequency, monitor).
                Defaults to {"interval": "step", "frequency": 1}.
            save_hyperparams: Whether to save hyperparameters for tracking.
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr_scheduler_config = lr_scheduler_config or {"interval": "step", "frequency": 1}
        self.train_config = self._normalize_config(train_config)
        self.test_config = self._normalize_config(test_config)
        self.eval_streams: list[EvalStreamRuntime] = list(eval_streams or [])

        # Training batches dropped by the non-finite guards (see
        # ``training_step`` / ``configure_gradient_clipping``). Non-zero means
        # the run survived an instability it would previously have died on —
        # worth checking against the training curve before trusting the result.
        self.num_skipped_batches = 0

        # Per-phase epoch buffers of (per-batch Prediction, target) pairs,
        # one entry per PhaseConfig in that phase. The buffered predictions are
        # flattened, detached, CPU Predictions (the full model output — not just
        # the mean), so the epoch reduction can run ANY metric, including the
        # likelihood-based ones (``loglik``) that need the predictive density,
        # not only the mean-based RMSE/correlations. Epoch metrics are computed
        # ONCE over the pooled buffer (see _eval_epoch_end) — never a
        # per-batch weighted mean — so every metric is batch-size invariant and
        # unbiased. Keyed by phase: "train", "test", and one namespace per eval
        # stream (its ``name``), so routing several val dataloaders through the
        # val loop never pools predictions across streams (B2).
        self._epoch_buffers: dict[str, list[dict[str, list[torch.Tensor]]]] = {}

        # Save hyperparameters for tracking
        if save_hyperparams:
            # context_provider holds the eval context = the whole train pool
            # (~7-8k samples), so pickling it into hparams bloats every checkpoint
            # by GBs. It is rebuilt from cfg and re-passed to load_from_checkpoint
            # (run.py), so the saved copy is never used — ignore it (like model /
            # eval_streams, which are also reconstructed at load).
            self.save_hyperparameters(ignore=[
                "model", "optimizer", "train_config", "test_config",
                "eval_streams", "scheduler", "context_provider",
            ])

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move custom batch objects to the target device."""
        if isinstance(batch, BaseBatch):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _normalize_config(
        self, config: PhaseConfig | list[PhaseConfig] | None
    ) -> list[PhaseConfig]:
        """Normalize config input to always be a list.

        Args:
            config: Configuration(s) for a phase.

        Returns:
            List of PhaseConfig objects.
        """
        if config is None:
            return []
        elif isinstance(config, PhaseConfig):
            return [config]
        else:
            return config

    def _get_wrapper(
        self,
        get_wrapper_fn: Callable,
        config: PhaseConfig,
        batch: BaseBatch,
        attr_name: str,
    ) -> ModelForwardWrapper | MetricCallWrapper:
        """Get an appropriate wrapper for the model and batch.

        Args:
            get_wrapper_fn: Function to get the wrapper.
            config: Configuration for the current phase.
            batch: Input batch data.
            attr_name: Name of the attribute to retrieve/set.

        Returns:
            The appropriate wrapper for the given phase and batch.
        """
        wrapper = getattr(config, attr_name)

        # Check if config has a wrapper
        if wrapper is None:
            # If no wrapper in config, try to get one from the registry
            # Cache the wrapper in config for future use
            wrapper = get_wrapper_fn(self.model, batch)
            setattr(config, attr_name, wrapper)

        if wrapper is None:
            raise ValueError(
                f"Could not find or create {attr_name} for the given model and batch"
            )

        return wrapper

    def _train_eval_configs(self) -> list[PhaseConfig]:
        """The eval metric set (RMSE/Pearson/Spearman) scored on the *train*
        split, sourced from the shared ``eval`` template carried in
        ``test_config`` (B8).

        The eval metrics are phase-agnostic regression metrics, so the train
        phase reuses the one ``eval`` template (forward/metric wrappers + its
        default metric list) rather than duplicating them in every train
        ``PhaseConfig``. ``run_training`` passes ``phase_configs.eval`` as
        ``test_config`` precisely so this path is fed. Empty ⇒ train logs
        only the loss.
        """
        return self.test_config

    def _record_eval_one(
        self,
        phase: str,
        ci: int,
        config: PhaseConfig,
        output,
        batch: BaseBatch,
        batch_size: int,
    ) -> None:
        """Log this config's per-batch ``*_step`` metrics and buffer raw preds.

        Shared by val/test (``_eval_step``, which forwards first) and train
        (``training_step``, which reuses the loss forward output — no extra
        forward pass).

        The per-batch (step) value uses the **same**
        ``metric_fn(prediction, target)`` computation as the epoch reduction
        (:meth:`_eval_epoch_end`), just over this batch — so step and epoch are
        the identical metric *definition* (only the data extent differs) and
        both are shape-robust (the Prediction and target are flattened to 1-D,
        matching ``predict_step``'s convention). The buffered
        ``(prediction, target)`` are reduced once per epoch; correlations are
        non-associative, so per-batch values are never averaged into the epoch
        number. ``{phase}_{name}_step`` is distinct from the epoch headline key
        ``{phase}_{name}`` so they never collide.
        """
        bufs = self._epoch_buffers[phase]
        prediction = output.flatten().detach_cpu()
        true = batch.s.detach().reshape(-1).cpu()

        step_metrics = config.metric_fn(
            prediction, true, **config.metric_kwargs
        )
        prefix = config.name + "_" if config.name else ""
        for name, value in step_metrics.items():
            if value is not None and isinstance(value, (torch.Tensor, int, float)):
                self.log(
                    f"{phase}_{prefix}{name}_step",
                    value,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )

        bufs[ci]["pred"].append(prediction)
        bufs[ci]["true"].append(true)

    def _ensure_buffers(self, phase: str, n: int) -> None:
        """Ensure ``_epoch_buffers[phase]`` has one ``{pred,true}`` slot per config.

        ``pred`` accumulates per-batch :class:`Prediction` objects (flattened,
        detached, CPU); ``true`` the matching 1-D target tensors.
        """
        bufs = self._epoch_buffers.setdefault(phase, [])
        while len(bufs) < n:
            bufs.append({"pred": [], "true": []})

    def training_step(
        self,
        batch: BaseBatch,
        batch_idx: int,
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Training step: backprop the loss; log step+epoch train eval metrics.

        One forward pass serves both: the loss (for ``backward``) and the
        train RMSE/correlation metrics, which are recorded from the *same*
        output (no extra forward). The loss epoch value uses Lightning's
        default mean-of-batch-means (standard for a loss); the eval metrics go
        through the pooled-buffer path so ``train_pearson_r``/``train_rmse``
        are batch-size invariant, exactly like val/test. Train eval metrics
        reflect training-mode forward passes with weights changing across the
        epoch — the conventional meaning of a training-set metric.
        """
        batch_size = self._get_batch_size(batch)
        eval_configs = self._train_eval_configs()
        if eval_configs:
            self._ensure_buffers("train", len(eval_configs))

        loss_value: torch.Tensor | None = None
        first_output = None
        try:
            for config in self.train_config:
                fw = self._get_wrapper(
                    get_forward_wrapper, config, batch, "forward_wrapper"
                )
                output = fw(self.model, batch, **config.forward_kwargs)
                if first_output is None:
                    first_output = output
                mw = self._get_wrapper(
                    get_metric_call_wrapper, config, batch, "metric_wrapper"
                )
                metrics = mw(config.metric_fn, output, batch, **config.metric_kwargs)
                self._log_results(
                    "train", metrics, on_step=True, on_epoch=True,
                    prog_bar=True, batch_size=batch_size,
                )
                if loss_value is None and "loss" in metrics:
                    loss_value = metrics["loss"]
        except ValueError as exc:
            # A non-finite distribution parameter (e.g. NaN `loc` reaching
            # `Normal`) raises out of the forward, killing the run. That is
            # recoverable: drop this batch. Only distribution-validation errors
            # are swallowed — anything else is a real bug and must propagate.
            if not _is_distribution_validation_error(exc):
                raise
            self._skip_train_batch(batch_idx, f"non-finite forward ({exc.args[0][:80]})")
            return None

        # A finite forward can still yield a non-finite loss (e.g. an inf
        # log-prob). Backpropagating it poisons every parameter, so skip.
        if loss_value is not None and not torch.isfinite(loss_value).all():
            self._skip_train_batch(batch_idx, "non-finite loss")
            return None

        # Train eval metrics from the first forward output — no extra forward.
        # Under no_grad: these never participate in backprop.
        if eval_configs and first_output is not None:
            with torch.no_grad():
                for ci, ec in enumerate(eval_configs):
                    self._record_eval_one(
                        "train", ci, ec, first_output, batch, batch_size
                    )

        return loss_value

    def _skip_train_batch(self, batch_idx: int, reason: str) -> None:
        """Record + report a dropped training batch (see the two guards above)."""
        self.num_skipped_batches += 1
        print(
            f"[nan-guard] skipped train batch {batch_idx} "
            f"(epoch {self.current_epoch}): {reason}. "
            f"total skipped this run: {self.num_skipped_batches}",
            flush=True,
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradients, but skip the update outright if any are non-finite.

        ``clip_grad_norm_`` computes ``clip_coef = max_norm / total_norm``; a
        single ``inf`` gradient makes ``total_norm`` non-finite, so
        ``clip_coef`` collapses to 0 and ``inf * 0 = NaN`` — which is then
        written into EVERY parameter, not just the offending one. The next
        forward pass therefore produces a NaN prediction and the run dies.

        Detect that case *before* clipping and detach the gradients instead:
        ``p.grad = None`` makes the optimizer skip those parameters entirely,
        turning a run-ending crash into one dropped step.
        """
        params = [p for p in self.parameters() if p.grad is not None]
        if params and not all(torch.isfinite(p.grad).all() for p in params):
            for p in params:
                p.grad = None
            self._skip_train_batch(
                self.trainer.global_step, "non-finite gradient (update skipped)"
            )
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def on_train_epoch_end(self) -> None:
        """Compute train eval metrics ONCE over the pooled epoch predictions."""
        self._eval_epoch_end("train", self._train_eval_configs())

    def test_step(
        self,
        batch: BaseBatch,
        batch_idx: int,
    ) -> None:
        """Test step: log per-batch (step) metrics, buffer raw preds for the epoch."""
        self._eval_step("test", self.test_config, batch, batch_idx)

    def predict_step(
        self,
        batch: BaseBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return per-batch ``(y_pred, y_true[, y_logprob])`` for pooling.

        Uses the test_config's first ``forward_wrapper`` to compute the model's
        :class:`Prediction` and reads its ``.mean`` (the scalar yield
        prediction). True targets come from ``batch.s``. When the head is
        distributional, the per-sample predictive log-density ``y_logprob`` is
        also returned, so ``evaluate_model`` can score per-``cv_label``
        log-likelihood alongside RMSE/correlation; a point head omits it (its
        ``log_prob`` would raise) and the test loglik is simply absent.
        ``trainer.predict`` returns these as a list of per-batch dicts;
        ``evaluate_model`` concatenates them.
        """
        if not self.test_config:
            raise RuntimeError(
                "predict_step requires a configured test_config to "
                "select the forward wrapper."
            )
        config = self.test_config[0]
        fw = self._get_wrapper(
            get_forward_wrapper, config, batch, "forward_wrapper"
        )
        output = fw(self.model, batch, **config.forward_kwargs)
        y_pred = output.mean.detach().cpu().reshape(-1)
        y_true = batch.s.detach().cpu().reshape(-1)
        result = {"y_pred": y_pred, "y_true": y_true}
        if output.is_distributional:
            flat = output.flatten()
            target = batch.s.detach().reshape(-1)
            result["y_logprob"] = (
                flat.log_prob(target).detach().cpu().reshape(-1)
            )
        return result

    def on_test_epoch_end(self) -> None:
        """Compute test metrics ONCE over the pooled epoch predictions."""
        self._eval_epoch_end("test", self.test_config)

    def validation_step(
        self,
        batch: BaseBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Validation step for one eval stream's batch.

        Lightning passes ``dataloader_idx`` when several val dataloaders are
        attached; it indexes ``self.eval_streams`` (the canonical order). The
        batch is forwarded once and routed to that stream's own buffer
        namespace (keyed by stream name), so streams never pool predictions
        together (B2). With a single val dataloader ``dataloader_idx`` is 0.
        """
        if not self.eval_streams:
            return
        stream = self.eval_streams[dataloader_idx]
        self._ensure_buffers(stream.name, 1)
        batch_size = self._get_batch_size(batch)
        fw = self._get_wrapper(
            get_forward_wrapper, stream.config, batch, "forward_wrapper"
        )
        output = fw(self.model, batch, **stream.config.forward_kwargs)
        self._record_eval_one(
            stream.name, 0, stream.config, output, batch, batch_size
        )

    def on_validation_epoch_end(self) -> None:
        """Compute each eval stream's metrics ONCE over its pooled epoch buffer.

        The monitor stream's ``<stream>_<metric>`` (e.g. ``val_pearson_r``) is
        the headline epoch value the early-stopping / LR-scheduler / checkpoint
        monitors read — the correlation/RMSE over the *pooled* stream, not a
        mean of per-batch values, so it is unaffected by the val batch size.
        Every observe stream (``test_*``, ``cv0_*``, …) is reduced the same way
        but is structurally barred from being the monitor (setup.py).
        """
        for stream in self.eval_streams:
            self._eval_epoch_end(stream.name, [stream.config])

    def _eval_step(
        self,
        phase: str,
        configs: list[PhaseConfig],
        batch: BaseBatch,
        batch_idx: int,
    ) -> None:
        """Run forward; log per-batch ``*_step`` metrics; buffer raw preds.

        Forwards once per config, then delegates step-logging + buffering to
        :meth:`_record_eval_one` (shared with the train path). The epoch
        reduction happens in :meth:`_eval_epoch_end`.
        """
        self._ensure_buffers(phase, len(configs))
        batch_size = self._get_batch_size(batch)
        for ci, config in enumerate(configs):
            fw = self._get_wrapper(
                get_forward_wrapper, config, batch, "forward_wrapper"
            )
            output = fw(self.model, batch, **config.forward_kwargs)
            self._record_eval_one(phase, ci, config, output, batch, batch_size)

    def _eval_epoch_end(
        self,
        phase: str,
        configs: list[PhaseConfig],
    ) -> None:
        """Compute each phase metric ONCE over the pooled epoch buffer.

        Reruns the phase's configured ``metric_fn`` on the pooled prediction —
        the per-batch :class:`Prediction` objects concatenated along the sample
        axis (:func:`cat_predictions`) — paired with the concatenated targets,
        so the epoch value is the exact metric over the whole split, identical
        no matter how it was batched (including ``loglik``, whose pooled mean
        log-density is exact only because the full per-sample distribution is
        buffered, not just the mean). The epoch key omits the ``_step`` suffix;
        it is the value monitors read.
        """
        bufs = self._epoch_buffers.get(phase, [])
        for ci, config in enumerate(configs):
            if ci >= len(bufs) or not bufs[ci]["pred"]:
                continue
            pooled = cat_predictions(bufs[ci]["pred"])
            true = torch.cat(bufs[ci]["true"])
            epoch_metrics = config.metric_fn(
                pooled, true, **config.metric_kwargs
            )
            prefix = config.name + "_" if config.name else ""
            # The epoch value is a single pooled scalar; pass the pooled size
            # explicitly so Lightning never tries to infer a batch_size from
            # the custom batch collection (which warns).
            pooled_size = int(true.shape[0]) if true.ndim > 0 else 1
            for name, value in epoch_metrics.items():
                if value is not None and isinstance(value, (torch.Tensor, int, float)):
                    self.log(
                        f"{phase}_{prefix}{name}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        add_dataloader_idx=False,
                        batch_size=pooled_size,
                    )
        # Reset for the next epoch.
        self._epoch_buffers[phase] = []

    def _get_batch_size(self, batch: BaseBatch) -> int:
        """Extract batch size from batch.

        Args:
            batch: Input batch data.

        Returns:
            Batch size as integer.
        """
        # Try to get batch size from common attributes
        for attr in ["vi_x", "vi_y", "s", "vi_pad_mask"]:
            if hasattr(batch, attr):
                tensor = getattr(batch, attr)
                if isinstance(tensor, torch.Tensor):
                    return tensor.size(0)

        # If no common attributes found, raise error
        raise ValueError("Could not extract batch size from batch")

    def _log_results(
        self,
        stage: str,
        metrics: dict[str, torch.Tensor],
        **log_kwargs,
    ) -> None:
        """Log a metrics dict, prefixing each key with the stage name.

        Args:
            stage: Stage prefix for metric names (e.g. 'train', 'test').
            metrics: Metric name → value mapping. Non-numeric values are skipped.
            **log_kwargs: Forwarded verbatim to self.log (e.g. on_step, on_epoch, batch_size).
        """
        for k, v in metrics.items():
            if v is not None and isinstance(v, (torch.Tensor, int, float)):
                self.log(f"{stage}_{k}", v, **log_kwargs)

    def configure_optimizers(self):
        """Configure optimizers (and optional LR scheduler) for training."""
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized for training!")
        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": self.scheduler, **self.lr_scheduler_config},
        }


class NPLitWrapper(LitWrapper):
    """Lightning wrapper for the Transformer Neural Process.

    The defining asymmetry: **train** batches are :class:`NPTaskBatch`es that
    carry their own context, but **val/test** batches are query-only
    ``G2FBatch``es. This wrapper owns the eval-time context: a single
    :class:`~utils.data.tasks.ContextProvider` materializes the context
    ``G2FBatch`` once per eval phase, and it is injected into each phase config's
    ``forward_kwargs`` so the inherited ``validation_step`` / ``predict_step`` /
    ``test_step`` call path stays unchanged — :func:`np_forward` reads the
    ``context`` kwarg and pairs it with each query batch. The context is
    re-encoded per batch (no ``kv`` cache in this step).
    """

    def __init__(self, *args, context_provider: ContextProvider | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_provider = context_provider

    def _out_of_sample_query_rows(
        self, streams: "list[EvalStreamRuntime]"
    ) -> list[dict] | None:
        """Union of rows from streams whose metric is meant to be out-of-sample.

        A stream marked ``in_sample`` (e.g. ``cv2``, a deliberate reconstruction
        probe whose rows legitimately sit in the eval context) is excluded. The
        returned rows are handed to :meth:`ContextProvider.build` as
        ``query_rows`` so the ``context ∩ query = ∅`` guard fires for every
        held-out stream — catching an ``eval_context`` that would silently
        degrade an out-of-sample metric to reconstruction. ``None`` when there
        is nothing to check (no provider / no out-of-sample streams).
        """
        if self.context_provider is None:
            return None
        dataset = self.context_provider.dataset
        rows: list[dict] = []
        for s in streams:
            if getattr(s, "in_sample", False):
                continue
            split = dataset.get_split(s.name)
            rows.extend(split[i] for i in range(len(split)))
        return rows or None

    def _stash_context(
        self,
        configs: list[PhaseConfig],
        *,
        query_rows: list[dict] | None = None,
    ) -> None:
        """Build the context once and inject it into ``configs``' forward_kwargs.

        ``query_rows`` (the out-of-sample streams' rows) is passed straight to
        ``ContextProvider.build``, which asserts ``context ∩ query = ∅`` — so a
        held-out query row that leaked into the context raises here.
        """
        if self.context_provider is None or not configs:
            return
        context = self.context_provider.build(query_rows=query_rows).to(self.device)
        for config in configs:
            config.forward_kwargs = {**config.forward_kwargs, "context": context}

    def on_validation_start(self) -> None:
        self._stash_context(
            [s.config for s in self.eval_streams],
            query_rows=self._out_of_sample_query_rows(self.eval_streams),
        )

    def on_test_start(self) -> None:
        self._stash_context(
            self.test_config,
            query_rows=self._out_of_sample_query_rows(self.eval_streams),
        )

    def on_predict_start(self) -> None:
        self._stash_context(
            self.test_config,
            query_rows=self._out_of_sample_query_rows(self.eval_streams),
        )

    def _get_batch_size(self, batch: BaseBatch) -> int:
        # An NPTaskBatch's "batch size" is the number of tasks B; read it from
        # the query targets (B, nq, 1). Query-only eval G2FBatches fall through
        # to the base extractor (which reads `s` -> Nq).
        if isinstance(batch, NPTaskBatch):
            return batch.query.s.shape[0]
        return super()._get_batch_size(batch)


class TrainingCurveCallback(pl.Callback):
    """Write one JSON line per validation epoch to ``training_curve.jsonl``.

    Records every eval stream's epoch metrics from ``trainer.callback_metrics``
    after each validation epoch — every ``{stream}_`` prefix from the resolved
    eval streams plus ``train_`` (B3). Each row is one epoch; columns are the
    namespaced ``{stream}_{metric}`` keys. Because stream names are unique
    (gate invariant) and a stream's metric names are unique, every
    (stream × metric) series is a distinct, unambiguous column
    — so ``test_*`` / ``cv0_*`` / ``val_iid_*`` are recorded alongside
    ``val_*`` instead of being silently dropped.

    Overwrite-by-default, mirroring the checkpoint convention
    (``checkpointing.enable_version_counter: false``): the file is truncated at
    ``on_train_start`` so a rerun of the same (model, seed) replaces the prior
    curve, exactly as the checkpoint does. Set
    ``misc.training_curve.enable_version_counter: true`` to preserve attempts as
    ``training_curve.jsonl`` / ``training_curve-v1.jsonl`` / … (the same lazy
    smallest-unused-integer scheme Lightning's checkpoint counter uses).
    """

    def __init__(
        self,
        metrics_dirpath: str,
        stream_names: "list[str] | None" = None,
        *,
        enable_version_counter: bool = False,
    ) -> None:
        super().__init__()
        self.metrics_dirpath = metrics_dirpath
        # Recorded key prefixes: one per eval stream + the train metrics.
        names = list(stream_names) if stream_names else ["val"]
        self._prefixes = tuple(f"{n}_" for n in (*names, "train"))
        self._enable_version_counter = enable_version_counter
        self._path = os.path.join(metrics_dirpath, "training_curve.jsonl")

    def _resolve_path(self) -> str:
        """The output path. With the version counter on, return the smallest
        unused ``training_curve[-vN].jsonl`` so prior attempts are preserved;
        otherwise the stable, overwritten ``training_curve.jsonl``."""
        if not self._enable_version_counter:
            return self._path
        base = os.path.join(self.metrics_dirpath, "training_curve")
        if not os.path.exists(f"{base}.jsonl"):
            return f"{base}.jsonl"
        v = 1
        while os.path.exists(f"{base}-v{v}.jsonl"):
            v += 1
        return f"{base}-v{v}.jsonl"

    @pl.utilities.rank_zero_only
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        os.makedirs(self.metrics_dirpath, exist_ok=True)
        self._path = self._resolve_path()
        # Overwrite-by-default truncates; the version-counter path resolves to
        # a fresh filename, so this just creates it.
        open(self._path, "w").close()

    @pl.utilities.rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Written at train-epoch end (not validation-epoch end): by this hook
        # Lightning has finalized ``callback_metrics`` for BOTH the train
        # metrics and every eval stream that ran this epoch, so each row
        # carries the full {stream}_{metric} + train_* set. (Reading at
        # on_validation_epoch_end races the logger-connector finalization and
        # silently drops the just-computed val/test keys.)
        row: dict = {"epoch": trainer.current_epoch}
        for k, v in trainer.callback_metrics.items():
            if k.startswith(self._prefixes) and not k.endswith("_step"):
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
        if len(row) > 1:  # at least one stream metric besides epoch
            with open(self._path, "a") as f:
                f.write(json.dumps(row) + "\n")


class WeightAveragingCallback(pl.Callback):
    """Evaluate an exponential moving average (EMA) of the weights each epoch.

    Maintains a shadow copy of ``pl_module.model``'s ``state_dict`` (parameters
    AND buffers), updated after every training batch as
    ``shadow = decay·shadow + (1-decay)·weights``. Around each validation loop
    the live weights are swapped for the shadow and restored afterwards, so
    every eval stream — and therefore ``training_curve.jsonl`` — reflects the
    AVERAGED model's per-epoch evolution. Comparing this curve to the raw-weight
    twin (a run with averaging off) shows whether averaging stabilises training.

    Decoupled from optimisation: the optimizer always steps the raw weights;
    only evaluation sees the average. Pairs with a constant LR — the average
    smooths the noise a flat schedule leaves in. Non-float buffers (e.g.
    BatchNorm ``num_batches_tracked``) are copied, not averaged; float buffers
    (BN running stats) are averaged so an averaged model still evaluates
    correctly. The set encoders here are LayerNorm-only (no running buffers).
    """

    def __init__(self, decay: float = 0.99) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"weight_averaging.decay must be in (0, 1); got {decay}")
        self.decay = float(decay)
        self._shadow: "dict[str, torch.Tensor] | None" = None
        self._backup: "dict[str, torch.Tensor] | None" = None

    @staticmethod
    def _inner(pl_module: pl.LightningModule) -> torch.nn.Module:
        return pl_module.model

    @torch.no_grad()
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Seed the shadow from the current weights once. Sanity-check validation
        # runs before this hook with _shadow=None, so it harmlessly uses the raw
        # weights (the swap hooks below no-op until the shadow exists).
        if self._shadow is None:
            self._shadow = {
                k: v.detach().clone()
                for k, v in self._inner(pl_module).state_dict().items()
            }

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        if self._shadow is None:
            return
        for k, v in self._inner(pl_module).state_dict().items():
            s = self._shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)  # integer buffers (e.g. num_batches_tracked)

    @torch.no_grad()
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Bracket the WHOLE validation loop with on_validation_start/end (not the
        # epoch-end hook) so the restore is independent of LightningModule vs
        # callback epoch-end ordering: every eval stream is forwarded under the
        # EMA weights, and the raw weights are back before the next train batch.
        if self._shadow is None:
            return
        model = self._inner(pl_module)
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self._shadow)

    @torch.no_grad()
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._backup is None:
            return
        self._inner(pl_module).load_state_dict(self._backup)
        self._backup = None

    @torch.no_grad()
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Rewrite ONLY the last/averaged checkpoint to hold the EMA weights, so
        a downstream eval that reloads ``last.ckpt`` predicts with the AVERAGED
        model — consistent with the EMA-evaluated training curve.

        The per-metric ``best_<metric>.ckpt`` files are deliberately left
        untouched: they are the true best-of-that-metric snapshots, and
        overwriting them with the FINAL-epoch EMA would make every
        ``best_*.ckpt`` identical to each other and to ``last.ckpt``, defeating
        the per-metric checkpoint feature.

        Done here, not via the swap hooks or ``on_save_checkpoint``:
        ``ModelCheckpoint`` captures the raw weights (its save fires outside the
        validation EMA-swap window), and with ``save_weights_only=True`` Lightning
        skips the callback ``on_save_checkpoint`` path entirely. Rewriting the
        file once at fit-end is independent of save timing and weights-only. The
        rewrite uses the FINAL EMA (end of training) — the intended last-epoch
        deliverable.
        """
        if self._shadow is None:
            return
        # Only the last-epoch checkpoint writer(s) receive the averaged weights;
        # best_model_path is never rewritten (see docstring).
        paths = set()
        for cb in getattr(trainer, "checkpoint_callbacks", []) or []:
            p = getattr(cb, "last_model_path", "")
            if p and os.path.exists(p):
                paths.add(p)
        prefixed = {f"model.{k}": v.detach().cpu().clone() for k, v in self._shadow.items()}
        for p in paths:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict")
            if sd is None:
                continue
            for mk, v in prefixed.items():
                if mk in sd:
                    sd[mk] = v
            torch.save(ckpt, p)


class FreezableBestCheckpoint(pl.callbacks.ModelCheckpoint):
    """A best-``ModelCheckpoint`` whose saving can be *frozen* mid-run.

    Identical to a normal best checkpoint until :meth:`freeze` is called;
    afterwards it stops writing entirely, so ``best.ckpt`` is locked to the best
    weights seen up to the freeze point. Pairs with :class:`PlateauFreezeCallback`,
    which freezes it once the monitored metric has plateaued — letting training
    run on to the full epoch budget (to observe the metric trajectory) while
    ``best.ckpt`` no longer chases late, possibly-noisy improvements. The
    ``last.ckpt`` writer is a *separate*, unfrozen ``ModelCheckpoint`` and keeps
    advancing every epoch.

    Saving is gated at the two epoch-level hooks ``ModelCheckpoint`` uses for an
    epoch-monitored best (validation end / train-epoch end); the best checkpoint
    here is never step-based, so those cover every save.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frozen: bool = False

    @property
    def state_key(self) -> str:
        # Fold the filename into the state_key so multiple frozen sets watching
        # the SAME metric (e.g. best_frozen_rmse vs best_frozen_d01_rmse under
        # freeze_on_plateau.extra_profiles) get DISTINCT keys. ModelCheckpoint's
        # default state_key is monitor/mode-based only, so same-metric instances
        # would collide — Lightning forbids two stateful callbacks of one type
        # sharing a state_key (RuntimeError at Trainer init).
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            filename=self.filename,
        )

    def freeze(self) -> None:
        """Stop saving from now on; ``best.ckpt`` keeps its current contents."""
        self.frozen = True

    def on_validation_end(self, trainer, pl_module) -> None:
        if self.frozen:
            return
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.frozen:
            return
        super().on_train_epoch_end(trainer, pl_module)

    # Persist the freeze flag across resume. A no-op under save_weights_only
    # (Lightning stores no callback state then), but correct with full checkpoints.
    def state_dict(self) -> dict:
        state = super().state_dict()
        state["frozen"] = self.frozen
        return state

    def load_state_dict(self, state_dict) -> None:
        self.frozen = bool(state_dict.get("frozen", False))
        super().load_state_dict(state_dict)


class PlateauFreezeCallback(pl.Callback):
    """Freeze a :class:`FreezableBestCheckpoint` once the monitor plateaus.

    Applies the same plateau test as
    :class:`~lightning.pytorch.callbacks.EarlyStopping` — ``patience`` validation
    epochs with no ``min_delta`` improvement in the monitored metric — but
    instead of setting ``trainer.should_stop`` it calls ``checkpoint.freeze()``.
    Training and per-epoch metric logging therefore continue to the full epoch
    budget; only the best-checkpoint *selection* stops. Intended for runs with
    early stopping OFF that still want a "converged" ``best.ckpt`` alongside the
    full post-plateau training curve.

    Its running best (with ``min_delta``) governs only *when* to freeze; the
    weights actually saved are the checkpoint's own running best up to that
    epoch, so ``best.ckpt`` is the best model seen before the plateau set in.
    """

    def __init__(
        self,
        checkpoint: FreezableBestCheckpoint,
        monitor: str,
        mode: str = "max",
        patience: int = 50,
        min_delta: float = 0.001,
        label: str = "best_frozen",
    ) -> None:
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(
                f"PlateauFreezeCallback mode must be 'min' or 'max'; got {mode!r}"
            )
        self.checkpoint = checkpoint
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = abs(float(min_delta))
        # Identifies which frozen checkpoint this watcher governs in the freeze
        # log line (with one watcher per checkpoint metric, "best_frozen" alone
        # would be ambiguous). Cosmetic — not part of the persisted state.
        self.label = label
        self.wait = 0
        self.best = float("-inf") if mode == "max" else float("inf")
        self.frozen_epoch: int | None = None

    @property
    def state_key(self) -> str:
        # `freeze_on_plateau` creates ONE PlateauFreezeCallback per
        # checkpoint-target metric (e.g. pearson_r / rmse / loglik). Lightning
        # forbids two stateful callbacks of the same type sharing a state_key,
        # so fold the per-metric label (best_frozen_<metric>, unique per target)
        # into the key. Without this, multi-metric checkpoint_metrics under
        # freeze raises "Found more than one stateful callback of type
        # PlateauFreezeCallback" at Trainer init.
        return self._generate_state_key(label=self.label)

    def _improved(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best + self.min_delta
        return current < self.best - self.min_delta

    def on_validation_end(self, trainer, pl_module) -> None:
        # Skip Lightning's pre-train sanity-check val pass, and do nothing once
        # already frozen (the callback is effectively "turned off").
        if trainer.sanity_checking or self.checkpoint.frozen:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        current = float(metric)
        if self._improved(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.checkpoint.freeze()
                self.frozen_epoch = int(trainer.current_epoch)
                pl.utilities.rank_zero_info(
                    f"[plateau-freeze] '{self.monitor}' did not improve by "
                    f">{self.min_delta} for {self.patience} val epochs; froze "
                    f"{self.label}.ckpt at epoch {self.frozen_epoch}. Training "
                    "continues to the full epoch budget (curve still logged)."
                )

    def state_dict(self) -> dict:
        return {
            "wait": self.wait,
            "best": self.best,
            "frozen_epoch": self.frozen_epoch,
        }

    def load_state_dict(self, state_dict) -> None:
        self.wait = int(state_dict.get("wait", 0))
        self.best = float(state_dict.get("best", self.best))
        self.frozen_epoch = state_dict.get("frozen_epoch", None)


class LogPerformanceCallback(pl.Callback):
    # Tracks forward/backward time separately, accumulates per-batch GPU
    # memory samples, and emits both live `hardware/*` step metrics and
    # end-of-phase summary rows (mean ± std over all observed batches).

    def __init__(self) -> None:
        super().__init__()

        self.start_time = 0.0
        self.last_batch_end_time = 0.0
        self.update_count = 0.0
        self.backward_start_time = 0.0
        self.between_step_time = 0.0

        # Forward pass timing for different phases
        self.train_forward_start_time = 0.0
        self.validation_forward_start_time = 0.0
        self.test_forward_start_time = 0.0

        # Accumulators for end-of-training/test mean/std summary
        # Note: val accumulators skip sanity-check batches (see on_validation_batch_end)
        self.all_train_forward_times: list[float] = []
        self.all_val_forward_times: list[float] = []
        self.all_test_forward_times: list[float] = []
        self.all_train_gpu_memory_GB: list[float] = []
        self.all_val_gpu_memory_GB: list[float] = []
        self.all_test_gpu_memory_GB: list[float] = []

        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(
                0
            ).total_memory  # Get total GPU memory

    @pl.utilities.rank_zero_only
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @pl.utilities.rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        pl_module.log(
            "hardware/performance_between_step_time",
            time.time() - self.between_step_time,
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )
        self.train_forward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: torch.Tensor,
    ):
        super().on_before_backward(trainer, pl_module, loss)
        forward_time = time.time() - self.train_forward_start_time
        self.all_train_forward_times.append(forward_time)
        pl_module.log(
            "hardware/performance_train_forward_time",
            forward_time,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=1,
        )

        # Log GPU memory after forward pass
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            gpu_mem_gb = memory_allocated / 1e9
            self.all_train_gpu_memory_GB.append(gpu_mem_gb)
            pl_module.log(
                "hardware/gpu_memory_used_after_train_forward_GB",
                gpu_mem_gb,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )
            pl_module.log(
                "hardware/gpu_memory_total_after_forward_GB",
                self.total_memory / 1e9,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )

        self.backward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        super().on_after_backward(trainer, pl_module)
        backward_time = time.time() - self.backward_start_time
        pl_module.log(
            "hardware/performance_backward_time",
            backward_time,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=1,
        )

        # Log GPU memory after gradient calculation
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()

            pl_module.log(
                "hardware/gpu_memory_used_after_gradients_GB",
                memory_allocated / 1e9,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )
            pl_module.log(
                "hardware/gpu_memory_total_after_gradients_GB",
                self.total_memory / 1e9,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )

    @pl.utilities.rank_zero_only
    def on_train_epoch_start(self, *args, **kwargs) -> None:
        super().on_train_epoch_start(*args, **kwargs)
        self.update_count = 0.0
        self.start_time = time.time()
        self.last_batch_end_time = time.time()
        self.between_step_time = time.time()

    @pl.utilities.rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.update_count += 1

        # Calculate total elapsed time
        total_elapsed_time = time.time() - self.start_time
        last_elapsed_time = time.time() - self.last_batch_end_time
        self.last_batch_end_time = time.time()

        # Calculate updates per second
        average_updates_per_second = self.update_count / total_elapsed_time
        last_updates_per_second = 1 / last_elapsed_time

        # Log updates per second to wandb using pl_module.log
        pl_module.log(
            "hardware/performance_average_updates_per_second",
            average_updates_per_second,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=1,
        )
        pl_module.log(
            "hardware/performance_last_updates_per_second",
            last_updates_per_second,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=1,
        )

        # Reset peak memory tracking for next step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.between_step_time = time.time()

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        t = torch.tensor(values)
        # correction=0: population std (correct for observed batches; also avoids
        # NaN when len==1 that Bessel's correction would produce)
        return t.mean().item(), t.std(correction=0).item()

    @staticmethod
    def _log_summary(
        summary: dict[str, float],
        trainer: pl.Trainer,
        header: str,
    ) -> None:
        print(f"\n=== {header} ===")
        for k, v in summary.items():
            print(f"  {k}: {v:.6f}")
        print("=" * (len(header) + 8))

        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            experiment = trainer.logger.experiment
            if hasattr(experiment, "summary"):
                for k, v in summary.items():
                    experiment.summary[k] = v

    @pl.utilities.rank_zero_only
    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        super().on_train_end(trainer, pl_module)

        summary: dict[str, float] = {}

        if self.all_train_forward_times:
            mean, std = self._mean_std(self.all_train_forward_times)
            summary["hardware/train_forward_time_mean_s"] = mean
            summary["hardware/train_forward_time_std_s"] = std

        if self.all_val_forward_times:
            mean, std = self._mean_std(self.all_val_forward_times)
            summary["hardware/val_forward_time_mean_s"] = mean
            summary["hardware/val_forward_time_std_s"] = std

        if self.all_train_gpu_memory_GB:
            mean, std = self._mean_std(self.all_train_gpu_memory_GB)
            summary["hardware/train_gpu_memory_mean_GB"] = mean
            summary["hardware/train_gpu_memory_std_GB"] = std

        if self.all_val_gpu_memory_GB:
            mean, std = self._mean_std(self.all_val_gpu_memory_GB)
            summary["hardware/val_gpu_memory_mean_GB"] = mean
            summary["hardware/val_gpu_memory_std_GB"] = std

        if summary:
            self._log_summary(summary, trainer, "Performance Summary (mean ± std over all batches)")

    @pl.utilities.rank_zero_only
    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_validation_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
        self.validation_forward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        forward_time = time.time() - self.validation_forward_start_time

        # Skip accumulation and logging during Lightning's sanity check
        if trainer.sanity_checking:
            return

        self.all_val_forward_times.append(forward_time)
        pl_module.log(
            "hardware/performance_validation_forward_time",
            forward_time,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=1,
        )

        # Log GPU memory after validation forward pass
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            gpu_mem_gb = memory_allocated / 1e9
            self.all_val_gpu_memory_GB.append(gpu_mem_gb)
            pl_module.log(
                "hardware/gpu_memory_used_after_val_forward_GB",
                gpu_mem_gb,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )

    @pl.utilities.rank_zero_only
    def on_test_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_test_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
        self.test_forward_start_time = time.time()

    @pl.utilities.rank_zero_only
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: pl.utilities.types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        forward_time = time.time() - self.test_forward_start_time
        self.all_test_forward_times.append(forward_time)

        # Only log to pl_module when a real logger is present.
        # With logger=False (eval.py), Lightning still accumulates pl_module.log
        # values into the trainer.test() return dict, which would pollute
        # results["metrics"] with hardware keys.
        if trainer.logger is not None:
            pl_module.log(
                "hardware/performance_test_forward_time",
                forward_time,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )

        # Log GPU memory after test forward pass
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            gpu_mem_gb = memory_allocated / 1e9
            self.all_test_gpu_memory_GB.append(gpu_mem_gb)
            if trainer.logger is not None:
                pl_module.log(
                    "hardware/gpu_memory_used_after_test_forward_GB",
                    gpu_mem_gb,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=False,
                    batch_size=1,
                )

    def get_test_summary(self) -> dict[str, float]:
        """Return mean/std of test forward time and GPU memory over all batches."""
        summary: dict[str, float] = {}
        if self.all_test_forward_times:
            mean, std = self._mean_std(self.all_test_forward_times)
            summary["test_forward_time_mean_s"] = mean
            summary["test_forward_time_std_s"] = std
        if self.all_test_gpu_memory_GB:
            mean, std = self._mean_std(self.all_test_gpu_memory_GB)
            summary["test_gpu_memory_mean_GB"] = mean
            summary["test_gpu_memory_std_GB"] = std
        return summary

    @pl.utilities.rank_zero_only
    def on_test_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        super().on_test_end(trainer, pl_module)

        summary: dict[str, float] = {}

        if self.all_test_forward_times:
            mean, std = self._mean_std(self.all_test_forward_times)
            summary["hardware/test_forward_time_mean_s"] = mean
            summary["hardware/test_forward_time_std_s"] = std

        if self.all_test_gpu_memory_GB:
            mean, std = self._mean_std(self.all_test_gpu_memory_GB)
            summary["hardware/test_gpu_memory_mean_GB"] = mean
            summary["hardware/test_gpu_memory_std_GB"] = std

        # Only print/log when running under a real logger (i.e. during training).
        # In eval.py the trainer uses logger=False; eval.py prints the summary itself.
        if summary and trainer.logger is not None:
            self._log_summary(summary, trainer, "Test Performance Summary (mean ± std over all batches)")
