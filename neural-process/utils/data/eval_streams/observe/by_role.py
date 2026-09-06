"""``role`` — select all rows of one ``Role`` (``test`` = the PREDICT rows).

The plan's former ``split: test`` source, now one observe selector among
others. Compatible with any scheme (``compatible_schemes = None``).
"""

from __future__ import annotations

import numpy as np

from cv.schemes.base import Role

from .base import ObserveSelector
from .registry import register_observe_selector

_VALID_ROLES = frozenset(r.value for r in Role)


@register_observe_selector
class ByRole(ObserveSelector):
    name = "role"
    compatible_schemes = None  # any scheme

    def __init__(self, role: str = Role.PREDICT.value) -> None:
        role = str(role)
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role selector: {role!r} is not a valid Role; expected "
                f"one of {sorted(_VALID_ROLES)}."
            )
        self.role = role

    def select(self, meta_df) -> np.ndarray:
        return np.where(meta_df["role"].to_numpy() == self.role)[0]
