from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for decoders."""

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent representation into an output tensor.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            torch.Tensor: Decoded output.
        """
        pass
