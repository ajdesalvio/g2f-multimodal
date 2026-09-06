"""``random`` — IID flat-fraction carve (the legacy ``val_ratio`` role).

A fresh, cleanly-seeded carve: statistically equivalent to the old
``random.shuffle`` split, **not** bit-equal (documented non-parity, same
stance as ``native`` vs ``r_csv`` fold sources). Draws from the *seen*
distribution — use an OOD strategy to mirror the test regime.
"""

from __future__ import annotations

import numpy as np

from .base import ValidationStrategy
from .registry import register_val_strategy


@register_val_strategy("random")
class RandomValidationStrategy(ValidationStrategy):
    def __init__(self, frac: float) -> None:
        if not 0.0 < float(frac) < 1.0:
            raise ValueError(
                f"random strategy 'frac' must be in (0, 1), got {frac!r}."
            )
        self.frac = float(frac)

    def select(self, meta_df, train_positions, *, rng):
        # Sort first so a fixed val_seed is reproducible regardless of the
        # order train_positions arrives in.
        pool = np.sort(np.asarray(train_positions, dtype=int))
        # int() truncation mirrors the legacy carve's count formula
        # (splitter.assign_roles: n_val = int(len * val_ratio)).
        n_val = int(len(pool) * self.frac)
        if n_val == 0:
            return np.array([], dtype=int)
        chosen = rng.choice(pool, size=n_val, replace=False)
        return np.sort(chosen)
