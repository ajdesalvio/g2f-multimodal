"""Base batch dataclasses for structured model inputs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class BaseBatch(ABC):
    """Base class for batch structures."""

    @abstractmethod
    def to(self, device: torch.device) -> "BaseBatch":
        """Move all tensor attributes to the specified device."""
        ...


@dataclass
class ViewBatch:
    """One named "view" of a set-structured modality.

    A view is the model-facing pairing of an *index/metric axis* (``coords``)
    with the *signal channels* measured along it (``channels``). Which raw
    quantity plays the coordinate (DAP / GDD / AGDD …) and which are channels
    is decided upstream (see ``utils/data/views``); the encoder only ever sees
    this resolved pair, so the same set encoder serves any axis choice.

    Attributes:
        coords: Index/metric axis values (B, T_max, D). ``D`` is the number
            of coordinate axes (1 for a single time axis; >1 for multi-axis).
        channels: Signal values at each set element (B, T_max, C).
        pad_mask: Padding mask (B, T_max), True for padding positions.
        coord_names: Names of the ``D`` coordinate axes (e.g. ``["dap"]``).
        channel_names: Names of the ``C`` channels (e.g. the VI names).
    """

    coords: torch.Tensor
    channels: torch.Tensor
    pad_mask: torch.Tensor
    coord_names: list[str] | None = None
    channel_names: list[str] | None = None

    def __post_init__(self) -> None:
        # Normalize pad_mask to (B, T); callers may pass (B, T, 1).
        if self.pad_mask.ndim != 2:
            self.pad_mask = self.pad_mask.squeeze(-1)

    def to(self, device: torch.device) -> "ViewBatch":
        """Move tensor attributes to the specified device."""
        self.coords = self.coords.to(device)
        self.channels = self.channels.to(device)
        self.pad_mask = self.pad_mask.to(device)
        return self


@dataclass
class G2FBatch(BaseBatch):
    """Batch structure for G2F data.

    Set-structured modalities are carried as named :class:`ViewBatch` entries
    under ``views`` (e.g. ``views["main"]`` for the VI curves,
    ``views["weather"]`` for a weather set encoder). Each set-encoder peer
    binds to one view by name. Fixed-size derived features (genomic, phenomic,
    kernel scores) stay in ``derived_features`` and feed MLP peers.

    Declared as a real dataclass so Lightning's ``_extract_batch_size`` utility
    can walk ``dataclasses.fields(batch)`` and infer the batch dimension from a
    tensor (``s``) without an explicit ``batch_size`` hint.

    Attributes:
        views: Named set-structured views (``{name: ViewBatch}``). At least
            ``"main"`` is always present.
        s: Scalar outputs (B, D_s).
        derived_features: Dict of named fixed-size feature tensors (B, D_feat).
            Access via ``batch.derived_features["key"]``.
    """

    views: dict[str, ViewBatch]
    s: torch.Tensor
    derived_features: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Preserve old `derived_features or {}` semantics when caller passes None
        if self.derived_features is None:
            self.derived_features = {}

    def to(self, device: torch.device) -> "G2FBatch":
        """Move all tensor attributes to the specified device."""
        self.views = {name: v.to(device) for name, v in self.views.items()}
        self.s = self.s.to(device)
        self.derived_features = {
            k: v.to(device) for k, v in self.derived_features.items()
        }
        return self
