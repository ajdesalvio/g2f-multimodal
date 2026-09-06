"""Identity transform: stable-sort rows by the primary coordinate.

Column *selection* happens upstream in ``assemble_view``; the identity
transform only canonicalises ordering (ascending ``coords[:, 0]``) with no
resampling. fdapace and the DL encoders are order-invariant, but a stable
sort keeps cache keys / debugging deterministic and is the natural no-op base
case all other transforms can be compared against.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import AxisTransform, ViewArrays
from .registry import register_axis_transform


@register_axis_transform
class IdentityTransform(AxisTransform):
    name = "identity"

    def __init__(self) -> None:
        pass

    def apply(self, arrays: ViewArrays) -> ViewArrays:
        key = arrays.coords[:, 0].numpy()
        # Stable so equal-coordinate rows keep their input order.
        order = np.argsort(key, kind="stable")
        return arrays.reindex(torch.from_numpy(order))
