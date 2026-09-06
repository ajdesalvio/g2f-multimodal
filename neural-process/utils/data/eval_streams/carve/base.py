"""``ValidationStrategy`` ABC — the carve family's contract.

A *carve* strategy selects rows to pull out of the training pool as the
DL early-stopping holdout (the val set). It **reduces** the train set and
its rows' ``cv_label`` is cleared by the splitter (preserving the
"never score a val row" guarantee). A carve is the only stream family the
monitor may name.

Determinism rule: every strategy first **sorts** its candidate units
(rows / varieties / envs) into a canonical order, *then* samples from the
``rng`` — so a fixed ``val_seed`` is reproducible regardless of dict/set
iteration order (the same discipline ``NativeFoldSource.build`` uses).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class ValidationStrategy(ABC):
    """Select a subset of the training pool to carve as validation."""

    #: Registry key, set by ``@register_val_strategy(name)``.
    name: ClassVar[str]
    #: True ⇒ the strategy needs a fold-bearing scheme (``cv_2_1`` /
    #: ``cv_0_00``) and an injected ``fold_map`` (B6). The VALIDATE gate
    #: rejects it under a no-fold scheme; ``get_val_strategy`` injects the
    #: run's ``fold_map`` for these.
    requires_fold: ClassVar[bool] = False

    @abstractmethod
    def select(
        self,
        meta_df: "pd.DataFrame",
        train_positions: "np.ndarray",
        *,
        rng: "np.random.Generator",
    ) -> "np.ndarray":
        """Return the val positions — a subset of ``train_positions``.

        Args:
            meta_df: coverage-filtered, RangeIndex metadata. Positional
                access (``.to_numpy()[pos]``) aligns with ``train_positions``.
            train_positions: positions eligible for the carve
                (``FIT ∪ OBSERVE``), in the ``meta_df`` position space.
            rng: a ``Generator`` seeded from ``dataset.val_seed`` — **not**
                ``misc.seed``. Sort candidate units before sampling.

        Returns a sorted ``np.ndarray`` of positions (a subset of
        ``train_positions``); may be empty.
        """
        ...
