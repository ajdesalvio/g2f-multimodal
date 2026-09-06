import einops
import torch
import torch.nn as nn

from ...utils.helpers import get_clones
from ..attention_layers import MultiHeadAttentionLayer
from ..transformers.base import BaseTransformerEncoder


class ISTransformerEncoder(BaseTransformerEncoder):
    """Induced-set (pseudo-token) TNP encoder — memory fallback.

    ``num_pseudo`` (= K) learnable pseudo tokens mediate context<->query,
    dropping the context cost from ``O(Nc^2)`` to ``O(Nc*K)``. The pseudo tokens
    are a learnable ``nn.Parameter(randn(K, d))`` broadcast per task with
    ``einops.repeat`` (NOT a ``PMAAggregator``).

    Per layer, in order:

    1. ``z_pt = context_to_pseudo(z_pt, zc, zc)`` — pseudo tokens read context.
    2. ``zc = pseudo_to_context(zc, z_pt, z_pt)`` — context reads pseudo tokens.
       **Skipped in the last layer** (no point refreshing the context when no
       further query read follows), so ``pseudo_to_context`` holds
       ``num_layers - 1`` clones.
    3. ``zq = pseudo_to_query(zq, z_pt, z_pt)`` — query reads pseudo tokens.

    Unlike :class:`PerceiverEncoder`, the context **is** updated (induced-set).
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
        self.pseudo_to_context = get_clones(layer, num_layers - 1)
        self.pseudo_to_query = get_clones(layer, num_layers)

    def forward(self, zc: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        """Refine queries via K pseudo tokens.

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
            if i < self.num_layers - 1:
                zc = self.pseudo_to_context[i](zq=zc, zk=z_pt, zv=z_pt)
            zq = self.pseudo_to_query[i](zq=zq, zk=z_pt, zv=z_pt)
        return zq
