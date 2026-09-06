import einops
import torch
import torch.nn as nn

from ...utils.helpers import get_clones
from ..attention_layers import MultiHeadAttentionLayer
from ..transformers.base import BaseTransformerEncoder


class PerceiverEncoder(BaseTransformerEncoder):
    """Pure-bottleneck (Perceiver) pseudo-token TNP encoder.

    ``num_pseudo`` (= K) learnable pseudo tokens form a bottleneck through which
    the query reads the context. Same ``nn.Parameter(randn(K, d))`` pseudo tokens
    broadcast per task as in :class:`ISTransformerEncoder`, but the **context is
    never updated** — cost ``O(Nc*K)``.

    Per layer, in order:

    1. ``z_pt = context_to_pseudo(z_pt, zc, zc)`` — pseudo tokens read context.
    2. ``z_pt = pseudo_to_pseudo(z_pt, z_pt, z_pt)`` — pseudo-token self-attention.
    3. ``zq = pseudo_to_query(zq, z_pt, z_pt)`` — query reads pseudo tokens.

    Tasks are the batch dim and never interact.
    """

    def __init__(
        self,
        layer: MultiHeadAttentionLayer,
        num_layers: int,
        num_pseudo: int,
    ):
        super().__init__(num_layers=num_layers)
        self.num_pseudo = num_pseudo
        self.z_pt = nn.Parameter(torch.randn(num_pseudo, layer.embed_dim))
        self.context_to_pseudo = get_clones(layer, num_layers)
        self.pseudo_to_pseudo = get_clones(layer, num_layers)
        self.pseudo_to_query = get_clones(layer, num_layers)

    def forward(self, zc: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        """Refine queries via a K-token bottleneck.

        Args:
            zc: Context tokens ``[B, nc, d]``.
            zq: Query tokens ``[B, nq, d]``.

        Returns:
            Refined query tokens ``[B, nq, d]``.
        """
        batch_size = zc.shape[0]
        z_pt = einops.repeat(self.z_pt, "k d -> b k d", b=batch_size)
        for i in range(self.num_layers):
            z_pt = self.context_to_pseudo[i](zq=z_pt, zk=zc, zv=zc)
            z_pt = self.pseudo_to_pseudo[i](zq=z_pt, zk=z_pt, zv=z_pt)
            zq = self.pseudo_to_query[i](zq=zq, zk=z_pt, zv=z_pt)
        return zq
