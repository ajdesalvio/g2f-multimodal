"""Transformer Neural Process (TNP) cross-sample encoders.

Each encoder is a :class:`~set_func.core.transformers.base.BaseTransformerEncoder`
with ``forward(zc, zq) -> zq_out`` and is selected by a single config switch
Task isolation is structural via the batch dim — no
block-diagonal mask.
"""

from .istransformer import ISTransformerEncoder
from .perceiver import PerceiverEncoder
from .transformer import EfficientQueryTransformerEncoder

__all__ = [
    "EfficientQueryTransformerEncoder",
    "ISTransformerEncoder",
    "PerceiverEncoder",
]
