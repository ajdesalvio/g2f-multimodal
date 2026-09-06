from typing import Any, ClassVar, List

import torch
import torch.nn as nn

from set_func.predictions import Prediction


class Metric:
    """Unified class for metric computation
    (both for loss and evaluation)."""

    # Set by `@register_metric(name)` on registered concrete subclasses;
    # None on the base and on undecorated/composite metrics.
    metric_name: ClassVar[str | None] = None

    # Optimization direction for model selection: "max" when a higher value is
    # better (correlations, log-likelihood), "min" when lower is better (rmse).
    # The single source of truth for best-checkpoint min/max (read via
    # `metric_direction`), so the direction is declared once here, not repeated
    # in every eval-streams YAML. None on the base / composite metrics.
    # NB: distinct from the `mode` *call* argument ("loss"/"eval") below.
    direction: ClassVar[str | None] = None

    def __init__(
        self,
        metrics: "Metric" | List["Metric"] | None = None,
    ) -> None:
        """Initialize metric function.

        Args:
            metrics: Optional metrics to combine (for composite metrics).
        """
        self.metrics = None
        if metrics is not None:
            if isinstance(metrics, Metric):
                self.metrics = (metrics,)
            else:
                self.metrics = tuple(metrics)

    @staticmethod
    def _is_loss(mode: str) -> bool:
        """Check if mode indicates loss computation."""
        return mode.lower() in ("loss", "as_loss")

    def __call__(
        self,
        prediction: Prediction,
        targets: torch.Tensor,
        mode: str = "eval",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute the metric.

        Args:
            prediction: Model output Prediction (point estimate, optionally with a density).
            targets: Target values.
            mode: Output mode ('loss' or 'eval').

        Returns:
            Dictionary with metric values.
        """
        if self.metrics is not None:
            results = {}
            for metric_fn in self.metrics:
                results.update(metric_fn(prediction, targets, mode, **kwargs))
            return results

        return self._compute(prediction, targets, mode, **kwargs)

    def _compute(
        self,
        prediction: Prediction,
        targets: torch.Tensor,
        mode: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute specific metric implementation
        (to be overridden by subclasses)."""
        return {}


# ── Metric registry (B11) ───────────────────────────────────────────
#
# A small name → Metric-class lookup mirroring the processor registry
# idiom (`@register_processor` / `get_processor_class`,
# `utils/data/processing/registry.py`). Metrics were previously
# instantiated only via Hydra `_target_`, so a terse config list like
# `metrics: [rmse, pearson_r]` (the eval-stream schema) and the VALIDATE
# gate's "metric implemented" check had nothing to resolve against. The
# registered name matches the metric's emitted key (e.g. `rmse`,
# `pearson_r`) so `build_metric([...])` and the logged metric key stay in
# lockstep.

_METRIC_REGISTRY: dict[str, type["Metric"]] = {}


def register_metric(name: str):
    """Class decorator registering a `Metric` subclass under `name`.

    Parameterized (unlike `@register_processor`, which reads `cls.name`)
    because the existing `Metric` subclasses carry no `name` ClassVar and
    the registry key must equal the metric's emitted dict key. Raises
    `ValueError` on a duplicate name (catches accidental re-registration
    on module reload).
    """

    def _decorate(cls: type["Metric"]) -> type["Metric"]:
        if name in _METRIC_REGISTRY:
            existing = _METRIC_REGISTRY[name]
            raise ValueError(
                f"Metric name {name!r} already registered to "
                f"{existing.__module__}.{existing.__name__}; cannot "
                f"re-register {cls.__module__}.{cls.__name__}."
            )
        cls.metric_name = name
        _METRIC_REGISTRY[name] = cls
        return cls

    return _decorate


def get_metric_class(name: str) -> type["Metric"]:
    """Look up a registered metric class by name.

    Raises `KeyError` with an "available names" hint on a miss — the
    error surfaced at the VALIDATE gate for an unknown metric name.
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"No metric registered under {name!r}. "
            f"Available: {sorted(_METRIC_REGISTRY)}."
        )
    return _METRIC_REGISTRY[name]


def metric_names() -> list[str]:
    """All registered metric names, alphabetically sorted."""
    return sorted(_METRIC_REGISTRY)


def metric_direction(name: str) -> str:
    """The optimization direction (``"max"``/``"min"``) of a registered metric.

    Reads the metric class's ``direction`` ClassVar — the single source of
    truth for whether a higher or lower value is "better" when selecting a
    best checkpoint (so min/max is declared once on the metric, not repeated
    per eval-streams YAML). Raises ``KeyError`` (via :func:`get_metric_class`)
    on an unknown name and ``ValueError`` if the metric declares no direction.
    """
    cls = get_metric_class(name)
    if cls.direction not in ("max", "min"):
        raise ValueError(
            f"metric {name!r} declares no optimization direction "
            f"(direction={cls.direction!r}); set `direction = 'max' | 'min'` "
            f"on {cls.__name__} to use it for best-checkpoint selection."
        )
    return cls.direction


def build_metric(names: list[str]) -> "Metric":
    """Build a composite `Metric` from a list of registry names.

    Returns a `Metric` whose `metrics` tuple holds one fresh instance per
    name, so calling it emits every named metric's key in one pass — the
    `metric_fn` an eval stream's `PhaseConfig` consumes. Raises
    `ValueError` on an empty list (a stream must name at least one metric;
    "requesting the subset alone is not sufficient") and `KeyError` (via
    `get_metric_class`) on an unknown name.
    """
    if not names:
        raise ValueError(
            "build_metric requires a non-empty list of metric names."
        )
    return Metric(metrics=[get_metric_class(n)() for n in names])


@register_metric("rmse")
class RMSE(Metric):
    """Root Mean Squared Error."""

    direction: ClassVar[str] = "min"

    def _compute(
        self,
        prediction: Prediction,
        targets: torch.Tensor,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        mse = nn.functional.mse_loss(
            prediction.mean, targets, reduction="mean"
        )
        rmse = torch.sqrt(mse)

        if self._is_loss(mode):
            return {"loss": rmse}
        return {"rmse": rmse}


@register_metric("loglik")
class LogLikelihood(Metric):
    """Mean predictive log-likelihood.

    Requires a distributional :class:`~set_func.predictions.Prediction`: it
    calls ``prediction.log_prob`` directly. On a point estimator (e.g. a
    :class:`~set_func.predictions.PointPrediction`) that call raises — naming
    ``loglik`` for a point head is a configuration error, surfaced rather than
    masked. In ``loss`` mode it returns the negative log-likelihood (the NLL
    training objective); in ``eval`` mode the (positive) mean log-likelihood
    under the key ``loglik``.
    """

    direction: ClassVar[str] = "max"

    def _compute(
        self,
        prediction: Prediction,
        targets: torch.Tensor,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        loglik = prediction.log_prob(targets).mean()

        if self._is_loss(mode):
            return {"loss": -loglik}
        return {"loglik": loglik}


@register_metric("pearson_r")
class PearsonCorrelation(Metric):
    """Pearson correlation coefficient between predictions and targets."""

    direction: ClassVar[str] = "max"

    def _compute(
        self,
        prediction: Prediction,
        targets: torch.Tensor,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        predictions = prediction.mean

        preds_flat = predictions.flatten()
        targets_flat = targets.flatten()

        pred_centered = preds_flat - preds_flat.mean()
        target_centered = targets_flat - targets_flat.mean()

        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (pred_centered**2).sum() * (target_centered**2).sum()
        )

        if denominator > 0:
            pearson_corr = numerator / denominator
        else:
            pearson_corr = torch.tensor(0.0, device=preds_flat.device)

        if self._is_loss(mode):
            return {"loss": -pearson_corr}
        return {"pearson_r": pearson_corr}


@register_metric("spearman_r")
class SpearmanCorrelation(Metric):
    """Spearman rank correlation coefficient between predictions and targets.

    Computes Pearson correlation on the ranks of the inputs. Ties are broken
    by original index order (ordinal ranking) — acceptable for continuous
    regression targets like crop yield where ties are vanishingly rare.
    """

    direction: ClassVar[str] = "max"

    def _compute(
        self,
        prediction: Prediction,
        targets: torch.Tensor,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        predictions = prediction.mean

        preds_flat = predictions.flatten()
        targets_flat = targets.flatten()

        # Pre-check for constant inputs: ordinal ranks of a constant
        # vector are [0, 1, ..., N-1] (ties broken by position), not a
        # constant — so we'd get a spurious correlation without this.
        if (
            preds_flat.numel() < 2
            or preds_flat.std(unbiased=False) == 0
            or targets_flat.std(unbiased=False) == 0
        ):
            spearman_corr = torch.tensor(0.0, device=preds_flat.device)
            if self._is_loss(mode):
                return {"loss": -spearman_corr}
            return {"spearman_r": spearman_corr}

        pred_ranks = preds_flat.argsort().argsort().float()
        target_ranks = targets_flat.argsort().argsort().float()

        pred_centered = pred_ranks - pred_ranks.mean()
        target_centered = target_ranks - target_ranks.mean()

        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (pred_centered**2).sum() * (target_centered**2).sum()
        )

        if denominator > 0:
            spearman_corr = numerator / denominator
        else:
            spearman_corr = torch.tensor(0.0, device=preds_flat.device)

        if self._is_loss(mode):
            return {"loss": -spearman_corr}
        return {"spearman_r": spearman_corr}
