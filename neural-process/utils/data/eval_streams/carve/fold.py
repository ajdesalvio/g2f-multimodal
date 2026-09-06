"""``fold`` — OOD carve reusing the run's committed fold partition.

Designates fold ``k`` of the **training** varieties as validation, reusing
the same ``fold_map: dict[Female, int]`` the CV scheme built from
``female_folds.csv``. Only valid under a fold-bearing scheme (``cv_2_1`` /
``cv_0_00``); the chosen val fold must differ from the test fold (the
test-fold varieties are already PREDICT, not in the train pool, so a val
fold equal to it would carve nothing). ``requires_fold`` lets the gate
reject it under a no-fold scheme and lets ``get_val_strategy`` inject the
``fold_map``.
"""

from __future__ import annotations

import numpy as np

from cv.schemes.identifiers import female_series, normalize_id

from .base import ValidationStrategy
from .registry import register_val_strategy


@register_val_strategy("fold")
class FoldValidationStrategy(ValidationStrategy):
    requires_fold = True

    def __init__(
        self,
        fold: int,
        fold_map: dict[str, int] | None = None,
    ) -> None:
        self.fold = int(fold)
        self.fold_map = (
            {normalize_id(k): int(v) for k, v in dict(fold_map).items()}
            if fold_map is not None
            else None
        )

    def select(self, meta_df, train_positions, *, rng):
        if self.fold_map is None:
            raise ValueError(
                "fold strategy requires a fold_map; it is injected by "
                "get_val_strategy from the run's CV scheme. A no-fold "
                "scheme (env_year_loo) has none — the VALIDATE gate "
                "rejects 'fold' there."
            )
        pool = np.sort(np.asarray(train_positions, dtype=int))
        fem_arr = female_series(meta_df).to_numpy()
        pool_females = fem_arr[pool]
        mask = np.fromiter(
            (self.fold_map.get(f) == self.fold for f in pool_females),
            dtype=bool,
            count=len(pool),
        )
        return pool[mask]
