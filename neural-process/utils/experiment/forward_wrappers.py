from collections.abc import Callable
from typing import Any, Type

import einops
import torch.nn as nn

from set_func.predictions import Prediction
from set_func.models import TransformerNeuralProcess

from ..data import BaseBatch, G2FBatch, NPTaskBatch
from ..data.base import ViewBatch
from .registry.base import (
    BaseWrapperRegistry,
    register_class_wrapper,
    register_instance_wrapper,
)


# Type definition for the wrapper function
ModelForwardWrapper = Callable[[nn.Module, BaseBatch, dict], Prediction]

# Global registry instance
registry = BaseWrapperRegistry[ModelForwardWrapper]()


def register_forward_wrapper(
    model_cls: Type[nn.Module] | tuple[Type[nn.Module], ...],
    batch_cls: Type[BaseBatch] | tuple[Type[BaseBatch], ...],
) -> Callable[[ModelForwardWrapper], ModelForwardWrapper]:
    """Decorator to register a forward wrapper for specific model and batch types.

    Args:
        model_cls: Model class or tuple of model classes.
        batch_cls: Batch class or tuple of batch classes.

    Returns:
        Decorator function.
    """
    return register_class_wrapper(registry, model_cls, batch_cls)


def register_instance_forward_wrapper(
    model: nn.Module,
) -> Callable[[ModelForwardWrapper], ModelForwardWrapper]:
    """Decorator to register a forward wrapper for a specific model instance.

    Args:
        model: Model instance.

    Returns:
        Decorator function.
    """
    return register_instance_wrapper(registry, model)


def get_forward_wrapper(
    model: nn.Module, batch: BaseBatch
) -> ModelForwardWrapper | None:
    """Get the appropriate forward wrapper for the given model and batch.

    Args:
        model: Model instance.
        batch: Batch instance.

    Returns:
        Forward wrapper function or None if not found.
    """
    return registry.get_wrapper(model, batch)


def _lift_to_task_dim(batch: G2FBatch) -> G2FBatch:
    """Lift a flat ``[N, …]`` ``G2FBatch`` to a single-task ``[1, N, …]`` one."""

    def lift(t):
        return einops.rearrange(t, "n ... -> 1 n ...")

    views: dict[str, ViewBatch] = {}
    for name, v in batch.views.items():
        pad_mask = lift(v.pad_mask)  # [1, N, T]
        vb = ViewBatch(
            coords=lift(v.coords),
            channels=lift(v.channels),
            pad_mask=pad_mask,
            coord_names=v.coord_names,
            channel_names=v.channel_names,
        )
        # Guard the ViewBatch T==1 squeeze.
        vb.pad_mask = pad_mask
        views[name] = vb

    derived = {k: lift(t) for k, t in batch.derived_features.items()}
    return G2FBatch(views=views, s=lift(batch.s), derived_features=derived)


@register_forward_wrapper(
    TransformerNeuralProcess,
    (NPTaskBatch, G2FBatch),
)
def np_forward(
    model: TransformerNeuralProcess,
    batch: BaseBatch,
    *,
    context: G2FBatch | None = None,
    **kwargs: Any,
) -> Prediction:
    """Forward wrapper for the Transformer Neural Process.

    Two batch shapes reach this:

    - **Train** — ``batch`` is already an :class:`NPTaskBatch` (the
      ``TaskSampler`` + ``np_collate`` emit one). Forwarded directly.
    - **Val/test** — ``batch`` is a query-only :class:`G2FBatch` of ``Nq``
      samples. ``context`` (a flat ``G2FBatch`` of ``Nc`` samples, supplied by
      :class:`~utils.experiment.lightning_wrapper.NPLitWrapper` via
      ``forward_kwargs``) is lifted to ``[1, Nc, …]``, the query to ``[1, Nq, …]``,
      and both are paired as an ``NPTaskBatch`` (``B=1``) before ``model.forward``.

    So the task dim is always the leading tensor axis; the context is never a
    loose model kwarg.
    """
    if isinstance(batch, NPTaskBatch):
        return model(batch)

    if context is None:
        raise ValueError(
            "np_forward received a query-only G2FBatch but no `context`; the "
            "NPLitWrapper must inject the eval context into forward_kwargs."
        )
    # Pair the eval context with this query batch on the query's device.
    if context.s.device != batch.s.device:
        context = context.to(batch.s.device)
    npbatch = NPTaskBatch(
        context=_lift_to_task_dim(context),
        query=_lift_to_task_dim(batch),
    )
    return model(npbatch)
