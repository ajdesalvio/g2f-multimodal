import torch

from ...utils.helpers import get_clones
from ..attention_layers import MultiHeadAttentionLayer
from .base import BaseTransformerEncoder


class TransformerEncoder(BaseTransformerEncoder):
    """
    ACNPs/TNPs Transformer Encoder that applies self-attention to all inputs.
    """

    def __init__(
        self,
        layer: MultiHeadAttentionLayer,
        num_layers: int,
    ):
        """
        Initialize the transformer encoder.

        Args:
            layer (MultiHeadAttentionLayer): Attention layer to be cloned.
            num_layers (int): Number of transformer layers.
        """
        super().__init__(num_layers=num_layers)
        self.layers = get_clones(layer, num_layers)

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            z = layer(zq=z, zk=z, zv=z, mask=mask)

        return z
