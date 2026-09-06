from collections.abc import Callable
from typing import Any, Type

import torch.nn as nn

from set_func.predictions import Prediction
from set_func.models import TransformerNeuralProcess

from ..data import BaseBatch, G2FBatch, NPTaskBatch
from .metrics import Metric
from .registry.base import (
    BaseWrapperRegistry,
    register_class_wrapper,
    register_instance_wrapper,
)


# Type definition for the wrapper function
MetricCallWrapper = Callable[
    [Metric, Prediction, BaseBatch, dict[str, Any]], Any
]

# Global registry instance
registry = BaseWrapperRegistry[MetricCallWrapper]()


def register_metric_call_wrapper(
    model_cls: Type[nn.Module] | tuple[Type[nn.Module], ...],
    batch_cls: Type[BaseBatch] | tuple[Type[BaseBatch], ...],
) -> Callable[[MetricCallWrapper], MetricCallWrapper]:
    """Decorator to register a metric call wrapper for specific model and batch types.

    Args:
        model_cls: Model class or tuple of model classes.
        batch_cls: Batch class or tuple of batch classes.

    Returns:
        Decorator function.
    """
    return register_class_wrapper(registry, model_cls, batch_cls)


def register_instance_metric_wrapper(
    model: nn.Module,
) -> Callable[[MetricCallWrapper], MetricCallWrapper]:
    """Decorator to register a metric call wrapper for a specific model instance.

    Args:
        model: Model instance.

    Returns:
        Decorator function.
    """
    return register_instance_wrapper(registry, model)


def get_metric_call_wrapper(
    model: nn.Module, batch: BaseBatch
) -> MetricCallWrapper | None:
    """Get the appropriate metric call wrapper for the given model and batch.

    Args:
        model: Model instance.
        batch: Batch instance.

    Returns:
        The appropriate wrapper function or None if not found.
    """
    return registry.get_wrapper(model, batch)


@register_metric_call_wrapper(
    TransformerNeuralProcess,
    (NPTaskBatch, G2FBatch),
)
def np_metric_wrapper(
    metric: Metric, prediction: Prediction, batch: BaseBatch, **kwargs: Any
) -> Any:
    """Metric wrapper for the Transformer Neural Process.

    The target is the query yields ``batch.s`` — ``[B, nq, 1]`` for an
    :class:`NPTaskBatch` (training loss, broadcasts elementwise against the
    decoder output) and ``[Nq]`` for a query-only :class:`G2FBatch` (eval).
    """
    return metric(prediction=prediction, targets=batch.s, **kwargs)
