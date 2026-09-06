"""``env_year_loo`` — the existing leave-one-env-out scheme, refactored.

The degenerate case of the unified abstraction:
it masks an entire environment column with **no** row-fold structure,
blending CV0 and CV00 into one ``"test"`` label. There are no
``OBSERVE`` rows, so ``fit_mask`` is ``None`` and engine behaviour is
byte-identical to today (the weather-GxE LOEO regression gate, D3 / Verification
3).

    PREDICT (y=NA) = Env==heldout                → label "test"
    FIT            = Env!=heldout

``fold_map`` is unused (no female folding). Weather regime is ``LOEO``
(the held-out column has no train-split rows), exactly as the existing
pipeline already does it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import CVScheme, Role, _empty_label_array
from .identifiers import env_year_series, normalize_id

if TYPE_CHECKING:
    import pandas as pd


class EnvYearLOOScheme(CVScheme):
    name = "env_year_loo"
    requires_heldout_env = True
    requires_fold = False

    def _assign_arrays(
        self,
        meta_df: "pd.DataFrame",
        fold_map: dict[str, int],
        *,
        fold: int | None,
        heldout_env: str | None,
        row_folds: "np.ndarray | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(meta_df)
        env = env_year_series(meta_df).to_numpy()
        env_match = env == normalize_id(heldout_env)

        roles = np.empty(n, dtype="<U7")
        roles[:] = Role.FIT.value
        roles[env_match] = Role.PREDICT.value

        labels = _empty_label_array(n)
        labels[env_match] = "test"
        return roles, labels
