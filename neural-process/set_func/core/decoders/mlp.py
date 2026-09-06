import torch
from torch import nn

from .base import BaseDecoder


class MLPDecoder(BaseDecoder):
    """MLP-based decoder that transforms a single latent representations into output predictions."""

    def __init__(self, z_decoder: nn.Module):
        """
        Initializes the MLP set decoder.

        Args:
            z_decoder (nn.Module): Neural network module to process the latent representation.
        """
        super().__init__()
        self.z_decoder = z_decoder

    @property
    def in_dim(self) -> int:
        return self.z_decoder.in_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent representation into an output tensor.

        Args:
            z (torch.Tensor): Latent representation of shape (batch, ..., latent_dim).

        Returns:
            torch.Tensor: Decoded output of shape (batch, ..., output_dim).
        """
        return self.z_decoder(z)
