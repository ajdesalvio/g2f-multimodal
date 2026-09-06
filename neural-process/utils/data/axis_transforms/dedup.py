"""Curve dedup transform — the R reference's collapse, at the per-sample curve level.

Rows whose ``key`` coordinate is equal (within ``round_decimals``) are
collapsed into one: channels are aggregated (``agg``; default mean), the
**first** row's coordinates are kept for don't-care columns, and the per-cell
mask is combined so a collapsed cell counts as observed iff *any* contributing
cell was observed.

This is **distinct** from the env-level axis dedup already built in
``processing/axis_source._dedup_axis``: that one cleans the AGDD
*axis* (averages daily weather, then re-``cumsum(GDD)``) and feeds the weather
path / the VI→AGDD lookup. This transform is a per-sample **curve** op that
rarely fires for VI (whose measurement DAPs map to a unique subset of the
daily AGDD grid) but is the configurable hook for the axis-dedup scenarios. It
does **not** re-cumsum — the coordinate is taken as-is from the assembled view.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import AxisTransform, ViewArrays
from .registry import register_axis_transform

_DEFAULT_ROUND_DECIMALS = 6
_SUPPORTED_AGG = ("mean", "first")


@register_axis_transform
class DedupTransform(AxisTransform):
    """Collapse rows sharing a (rounded) ``key`` coordinate.

    Parameters (from ``transform_args``)
    ------------------------------------
    key : str, optional
        Name of the coordinate to group on. Defaults to the primary
        coordinate ``coords[:, 0]``. Must be one of the view's coords.
    agg : {"mean", "first"}, optional
        How to aggregate channels within a group. ``"mean"`` (default) is
        the R reference's averaging; ``"first"`` keeps the first row.
    round_decimals : int, optional
        Rounding applied to the key before grouping (float-tie safety).
    """

    name = "dedup"

    def __init__(
        self,
        key: str | None = None,
        agg: str = "mean",
        round_decimals: int = _DEFAULT_ROUND_DECIMALS,
    ) -> None:
        if agg not in _SUPPORTED_AGG:
            raise ValueError(
                f"dedup.agg must be one of {_SUPPORTED_AGG}, got {agg!r}."
            )
        self.key = key
        self.agg = agg
        self.round_decimals = int(round_decimals)

    def apply(self, arrays: ViewArrays) -> ViewArrays:
        key_idx = 0 if self.key is None else arrays.coord_index(self.key)

        coords = arrays.coords.numpy()
        channels = arrays.channels.numpy()
        mask = None if arrays.mask is None else arrays.mask.numpy()

        # Sort by the key so equal-key rows form contiguous runs, then group
        # on changes in the rounded key. Stable to keep first-row semantics.
        key_col = coords[:, key_idx]
        order = np.argsort(key_col, kind="stable")
        coords = coords[order]
        channels = channels[order]
        if mask is not None:
            mask = mask[order]

        rounded = np.round(coords[:, key_idx], self.round_decimals)
        starts = np.concatenate(([0], np.flatnonzero(np.diff(rounded)) + 1))
        ends = np.concatenate((starts[1:], [len(rounded)]))

        n_groups = len(starts)
        out_coords = np.empty((n_groups, coords.shape[1]), dtype=coords.dtype)
        out_channels = np.empty(
            (n_groups, channels.shape[1]), dtype=channels.dtype
        )
        out_mask = (
            None
            if mask is None
            else np.empty((n_groups, mask.shape[1]), dtype=mask.dtype)
        )

        for g, (s, e) in enumerate(zip(starts, ends)):
            out_coords[g] = coords[s]  # first() for don't-care coords
            grp = channels[s:e]
            if self.agg == "first" or (e - s) == 1:
                out_channels[g] = grp[0]
                if mask is not None and out_mask is not None:
                    out_mask[g] = mask[s]
                continue
            if mask is None or out_mask is None:
                out_channels[g] = grp.mean(axis=0)
            else:
                # Average only over observed cells (mask True = missing). A
                # collapsed cell is observed iff any contributor was.
                valid = ~mask[s:e]
                count = valid.sum(axis=0)
                summed = np.where(valid, grp, 0.0).sum(axis=0)
                with np.errstate(invalid="ignore"):
                    mean_valid = summed / np.where(count > 0, count, 1)
                # Where no cell was observed, fall back to the plain mean so
                # the value is defined; the mask flags it as missing.
                out_channels[g] = np.where(
                    count > 0, mean_valid, grp.mean(axis=0)
                )
                out_mask[g] = count == 0

        return ViewArrays(
            coords=torch.from_numpy(out_coords),
            channels=torch.from_numpy(out_channels),
            coord_names=list(arrays.coord_names),
            channel_names=list(arrays.channel_names),
            mask=None if out_mask is None else torch.from_numpy(out_mask),
        )
