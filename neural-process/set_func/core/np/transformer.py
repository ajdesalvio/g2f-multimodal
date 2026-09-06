import torch

from ...utils.helpers import get_clones
from ..attention_layers import MultiHeadAttentionLayer
from ..transformers.base import BaseTransformerEncoder


class EfficientQueryTransformerEncoder(BaseTransformerEncoder):
    """Full-attention TNP-D ("efficient-query") encoder — the default.

    Context and query are refined **alternately, layer by layer** (not
    "encode the context through all ``L`` layers, then decode the query
    once"). Each of the ``num_layers`` layers does, in order:

    1. ``zc = context_to_context(zc, zc, zc)`` — context self-attention, so the
       context representation is deepened one layer.
    2. ``zq = context_to_query(zq, zc, zc)`` — the query cross-attends to the
       **just-updated** context, re-reading the current-depth context.

    The two stacks are always **separate** ``get_clones`` instances
    (parameters are never shared between them).
    Queries never attend to each other and never feed back into the context, so
    the query side is cheap and the context side is ``O(Nc^2)``.

    Tasks are the batch dimension and never interact: ``zc`` is ``[B, nc, d]``
    and ``zq`` is ``[B, nq, d]`` where ``B`` = tasks per batch, and all attention
    runs per batch element. No block-diagonal task mask is needed.
    """

    def __init__(self, layer: MultiHeadAttentionLayer, num_layers: int):
        super().__init__(num_layers=num_layers)
        self.context_to_context = get_clones(layer, num_layers)
        self.context_to_query = get_clones(layer, num_layers)

    def forward(self, zc: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        """Refine queries against the context.

        Args:
            zc: Context tokens ``[B, nc, d]`` (``y`` observed, density 0).
            zq: Query tokens ``[B, nq, d]`` (``y`` missing, density 1).

        Returns:
            Refined query tokens ``[B, nq, d]``.
        """
        for c2c, c2q in zip(self.context_to_context, self.context_to_query):
            zc = c2c(zq=zc, zk=zc, zv=zc)
            zq = c2q(zq=zq, zk=zc, zv=zc)
        return zq
