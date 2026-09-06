"""Axis-transform base types: the :class:`ViewArrays` carrier and the
:class:`AxisTransform` ABC.

A *view* (see ``utils/data/views``) resolves a sample into a coords/channels
pair; an :class:`AxisTransform` then reshapes that pair along the time axis
(sort, dedup, warp …). Every transform is a leaf operation on a single
sample's arrays — no batching, no fitted state — so it is equally usable from
the DL collate path and the FPCA R-bridge.

``ViewArrays`` deliberately carries the per-cell ``mask`` (the reader's
``vi_nan_mask``, ``True`` = originally-missing/interpolated) so that every
transform reorders / collapses / slices the mask **in lockstep** with
``channels``. FPCA's ``missing_values='skip'`` reads this mask
after assembly, so a transform that moved channels without moving the mask
would silently misalign which cells are treated as real observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass
class ViewArrays:
    """One sample's assembled, post-transform view.

    Attributes
    ----------
    coords : Tensor ``(T, D)``
        The metric / index axes. ``D == 1`` for a single time axis (DAP /
        GDD / AGDD); ``D > 1`` for a multi-axis DL view. The first column
        ``coords[:, 0]`` is the canonical ordering / FPCA time axis.
    channels : Tensor ``(T, C)``
        Signal values measured along ``coords``.
    coord_names : list[str]
        Names of the ``D`` coordinate columns (e.g. ``["dap"]``).
    channel_names : list[str]
        Names of the ``C`` channel columns (e.g. VI names).
    mask : Tensor ``(T, C)`` or ``None``
        Per-cell missing mask aligned to ``channels`` (``True`` = the cell
        was originally NaN / interpolated). ``None`` when the sample carries
        no mask. Transforms move it alongside ``channels``.
    """

    coords: torch.Tensor
    channels: torch.Tensor
    coord_names: list[str]
    channel_names: list[str]
    mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            raise ValueError(
                f"ViewArrays.coords must be 2-D (T, D); got shape "
                f"{tuple(self.coords.shape)}."
            )
        if self.channels.ndim != 2:
            raise ValueError(
                f"ViewArrays.channels must be 2-D (T, C); got shape "
                f"{tuple(self.channels.shape)}."
            )
        if self.coords.shape[0] != self.channels.shape[0]:
            raise ValueError(
                f"ViewArrays coords/channels disagree on T: "
                f"{self.coords.shape[0]} vs {self.channels.shape[0]}."
            )
        if len(self.coord_names) != self.coords.shape[1]:
            raise ValueError(
                f"ViewArrays has {len(self.coord_names)} coord_names but "
                f"coords width is {self.coords.shape[1]}."
            )
        if len(self.channel_names) != self.channels.shape[1]:
            raise ValueError(
                f"ViewArrays has {len(self.channel_names)} channel_names but "
                f"channels width is {self.channels.shape[1]}."
            )
        if self.mask is not None and tuple(self.mask.shape) != tuple(
            self.channels.shape
        ):
            raise ValueError(
                f"ViewArrays.mask shape {tuple(self.mask.shape)} must match "
                f"channels shape {tuple(self.channels.shape)}."
            )

    @property
    def T(self) -> int:  # noqa: N802 — matches torch's row-count convention
        return self.coords.shape[0]

    @property
    def D(self) -> int:  # noqa: N802
        return self.coords.shape[1]

    @property
    def C(self) -> int:  # noqa: N802
        return self.channels.shape[1]

    def coord_index(self, name: str) -> int:
        """Column index of coordinate ``name`` (raises if absent)."""
        try:
            return self.coord_names.index(name)
        except ValueError:
            raise KeyError(
                f"coord {name!r} not in coord_names {self.coord_names}."
            ) from None

    def reindex(self, order: torch.Tensor) -> "ViewArrays":
        """Return a new :class:`ViewArrays` with rows permuted by ``order``."""
        return ViewArrays(
            coords=self.coords[order],
            channels=self.channels[order],
            coord_names=list(self.coord_names),
            channel_names=list(self.channel_names),
            mask=None if self.mask is None else self.mask[order],
        )


class AxisTransform(ABC):
    """Base class for a single-sample axis transform.

    Subclasses declare a ``name`` ClassVar (used by the registry) and
    implement :meth:`apply`. They are constructed from the view's
    ``transform_args`` dict, e.g. ``DedupTransform(key="agdd",
    round_decimals=6)``. Construction validates args eagerly so a bad config
    fails at view-build time, not deep inside a curve emit.
    """

    name: ClassVar[str]

    @abstractmethod
    def apply(self, arrays: ViewArrays) -> ViewArrays:
        """Return a transformed copy of ``arrays`` (never mutates input)."""
        raise NotImplementedError
