from abc import ABC, abstractmethod

import torch
from torch import nn

from ...utils.aggregate import Aggregator


class BaseSetEncoder(nn.Module, ABC):
    """
    Abstract base class for set encoders that process input pairs (x, y).

    Follows a structured encoding pipeline:
    1. Optionally encode x and y separately
    2. Concatenate the encoded x and y
    3. Optionally embed the concatenation
    4. Apply set-level encoding (e.g., a transformer over the set elements)
    5. Aggregate the set into a single representation
    6. Optionally apply final encoding to the aggregated output

    Subclasses must implement the forward method to define how these components
    interact, as different encoder types may have different calling conventions
    for their latent_encoder and aggregator modules.
    """

    def __init__(
        self,
        latent_encoder: nn.Module,
        aggregator: Aggregator,
        out_dim: int,
        x_encoder: nn.Module | None = None,
        y_encoder: nn.Module | None = None,
        xy_encoder: nn.Module | None = None,
        output_encoder: nn.Module | None = None,
    ):
        """
        Initializes the set encoder with a structured pipeline.

        Args:
            latent_encoder (nn.Module): Encodes the latent representation at the set level
                (e.g., a Transformer encoder, or a per-element MLP). This operates on the
                full set before aggregation.
            aggregator (Aggregator): Aggregation module to pool set elements into a single
                representation.
            out_dim (int): Dimensionality of the encoder's output `(B, out_dim)`. Required
                so downstream consumers (e.g., the `SampleTokenizer`'s peer
                encoders) can validate decoder `in_dim` without having to
                infer output size from heterogeneous latent/output modules (a transformer
                preserves its input dim and exposes only `embed_dim` on internal layers,
                while an `output_encoder` MLP has its own `out_dim` — there is no
                single attribute to read generically).
            x_encoder (Optional[nn.Module]): Encodes x features before concatenation (default: Identity).
            y_encoder (Optional[nn.Module]): Encodes y features before concatenation (default: Identity).
            xy_encoder (Optional[nn.Module]): Encodes concatenated (x, y) features (default: Identity).
            output_encoder (Optional[nn.Module]): Encodes the final representation after
                aggregation (default: Identity).
        """
        super().__init__()
        self.x_encoder = x_encoder or nn.Identity()
        self.y_encoder = y_encoder or nn.Identity()
        self.xy_encoder = xy_encoder or nn.Identity()
        self.latent_encoder = latent_encoder
        self.aggregator = aggregator
        self.output_encoder = output_encoder or nn.Identity()
        self.out_dim = out_dim

    @abstractmethod
    def forward(
        self,
        coords: torch.Tensor,
        channels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes a set view (coords, channels) into a scalar representation.

        Args:
            coords (torch.Tensor): Index/metric axis, shape (batch, set_size, D).
            channels (torch.Tensor): Signal values, shape (batch, set_size, C).
            mask (Optional[torch.Tensor]): Binary mask of shape (batch, set_size) where
                True indicates padding elements. Subclasses should transform this mask as needed
                for their internal modules (e.g., to attention mask format for transformers).

        Returns:
            torch.Tensor: Aggregated scalar representation of shape (batch, latent_dim).
        """
        pass
