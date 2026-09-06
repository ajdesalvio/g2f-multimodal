"""Warp transform — warped functional data analysis (v1: monotone reparam).

Re-expresses a curve on a warped time domain: the coordinate becomes the
``to`` axis (e.g. AGDD), the channel *values* are unchanged, and rows are
re-sorted ascending on the new coordinate. A single FPCA over the warped time
then realises warped-FDA.

The ``to`` axis must be present in the assembled :class:`ViewArrays` — as a
coordinate (e.g. view ``coords:[dap, agdd]``) or as a channel. ``assemble_view``
is responsible for selecting it in. The output is single-axis (``D == 1``,
``coord_names == [to]``); if ``to`` was a channel it is consumed (removed from
``channels``), so the VI channels are preserved unchanged.

``from`` is validated to be present (it is the original parametrisation) but
is otherwise dropped — v1 is a pure reparametrisation, not a resampling.
A richer warp (resampling onto a common grid) is a future transform.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import AxisTransform, ViewArrays
from .registry import register_axis_transform


@register_axis_transform
class WarpTransform(AxisTransform):
    """Reparametrise the curve from the ``from`` axis onto the ``to`` axis.

    Parameters (from ``transform_args``)
    ------------------------------------
    from : str
        The current parametrising coordinate (must be a view coord).
    to : str
        The target coordinate to warp onto (a view coord or channel).
    """

    name = "warp"

    def __init__(self, **kwargs) -> None:
        # ``from`` is a Python keyword, so accept it via kwargs (also allow
        # the ``from_`` spelling for programmatic callers).
        from_axis = kwargs.pop("from", kwargs.pop("from_", None))
        to_axis = kwargs.pop("to", None)
        if kwargs:
            raise TypeError(
                f"warp: unexpected transform_args {sorted(kwargs)}; "
                f"expected only 'from' and 'to'."
            )
        if not from_axis or not to_axis:
            raise ValueError("warp requires both 'from' and 'to' axis names.")
        self.from_axis = from_axis
        self.to_axis = to_axis

    def apply(self, arrays: ViewArrays) -> ViewArrays:
        if self.from_axis not in arrays.coord_names:
            raise KeyError(
                f"warp: 'from' axis {self.from_axis!r} not in view coords "
                f"{arrays.coord_names}."
            )

        # Locate the target axis: prefer coords, fall back to channels.
        new_coord_col: np.ndarray
        channels = arrays.channels.numpy()
        channel_names = list(arrays.channel_names)
        mask = None if arrays.mask is None else arrays.mask.numpy()

        if self.to_axis in arrays.coord_names:
            new_coord_col = arrays.coords[
                :, arrays.coord_index(self.to_axis)
            ].numpy()
        elif self.to_axis in channel_names:
            ci = channel_names.index(self.to_axis)
            new_coord_col = channels[:, ci].copy()
            # Consume the channel so it is not double-counted as a signal.
            keep = [j for j in range(channels.shape[1]) if j != ci]
            channels = channels[:, keep]
            channel_names = [channel_names[j] for j in keep]
            if mask is not None:
                mask = mask[:, keep]
        else:
            raise KeyError(
                f"warp: 'to' axis {self.to_axis!r} not in view coords "
                f"{arrays.coord_names} or channels {arrays.channel_names}. "
                f"Select it into the view (e.g. coords:[{self.from_axis}, "
                f"{self.to_axis}])."
            )

        # Monotone reparam: sort ascending on the new coordinate.
        order = np.argsort(new_coord_col, kind="stable")
        new_coords = new_coord_col[order].astype(np.float64)[:, None]
        out = ViewArrays(
            coords=torch.from_numpy(new_coords),
            channels=torch.from_numpy(channels[order]),
            coord_names=[self.to_axis],
            channel_names=channel_names,
            mask=None if mask is None else torch.from_numpy(mask[order]),
        )
        return out
