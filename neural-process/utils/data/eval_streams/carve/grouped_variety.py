"""``grouped_variety`` — OOD carve holding out whole maternal lines.

Mirrors the ``cv_2_1`` regime: validation contains *unseen genotypes*
(no maternal line is split across train/val). Takes an exact group spec
(``n_groups`` or an explicit ``groups`` list) — a fraction would land
inexactly since varieties have unequal row counts.
"""

from __future__ import annotations

import numpy as np

from cv.schemes.identifiers import female_series, normalize_id

from .base import ValidationStrategy
from .registry import register_val_strategy


@register_val_strategy("grouped_variety")
class GroupedVarietyValidationStrategy(ValidationStrategy):
    def __init__(
        self,
        n_groups: int | None = None,
        groups: list[str] | None = None,
    ) -> None:
        if (n_groups is None) == (groups is None):
            raise ValueError(
                "grouped_variety requires exactly one of 'n_groups' / "
                f"'groups' (got n_groups={n_groups!r}, groups={groups!r})."
            )
        if n_groups is not None and int(n_groups) < 1:
            raise ValueError(f"n_groups must be >= 1, got {n_groups!r}.")
        self.n_groups = int(n_groups) if n_groups is not None else None
        self.groups = (
            [normalize_id(g) for g in groups] if groups is not None else None
        )

    def select(self, meta_df, train_positions, *, rng):
        pool = np.sort(np.asarray(train_positions, dtype=int))
        # female_series is indexed like meta_df (RangeIndex) → positional.
        fem_arr = female_series(meta_df).to_numpy()
        pool_females = fem_arr[pool]
        present = sorted(set(pool_females.tolist()))  # canonical order

        if self.groups is not None:
            missing = [g for g in self.groups if g not in present]
            if missing:
                raise ValueError(
                    f"grouped_variety: female(s) {missing} not present in "
                    f"the train pool; cannot carve them as validation."
                )
            chosen = set(self.groups)
        else:
            if self.n_groups > len(present):
                raise ValueError(
                    f"grouped_variety: requested n_groups={self.n_groups} "
                    f"but only {len(present)} maternal line(s) in the train "
                    f"pool."
                )
            chosen = set(
                rng.choice(
                    np.array(present), size=self.n_groups, replace=False
                ).tolist()
            )

        mask = np.fromiter(
            (f in chosen for f in pool_females), dtype=bool, count=len(pool)
        )
        return pool[mask]
