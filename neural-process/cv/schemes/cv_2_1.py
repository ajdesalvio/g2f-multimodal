"""``CV_2_1`` split group — no environment removed, one female-fold held out.

One BGLR fit → two quadrants (Background table, transcribed from R's
``mask_yields`` + ``evaluate_metrics``, ``k = fold``):

    PREDICT (y=NA) = common & Fold==k          (all envs)  → label CV1
    observed       = everything else
    basis fit set  = common & Fold!=k          (all envs)  → label CV2 (in-sample)

Role mapping (``fold_map`` membership ⇔ common):

    f = fold_map.get(female)            # NaN/None ⇒ non-common
      f is None        → OBSERVE, label None      (non-common background)
      f == fold        → PREDICT, label "CV1"     (new genotype, masked)
      f != fold        → FIT,     label "CV2"     (retained, scored in-sample)

Note CV2 is an **in-sample** metric scored over the retained FIT rows —
exactly R's asymmetry (``evaluate_metrics`` scores ``Fold!=k`` rows whose
``y`` was never masked). Weather regime is ``ALL`` (every env keeps
FIT/OBSERVE rows → all envs in the train split; D4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import CVScheme, Role, _empty_label_array, _fold_of_rows

if TYPE_CHECKING:
    import pandas as pd


class CV2_1Scheme(CVScheme):
    name = "cv_2_1"
    requires_heldout_env = False
    requires_fold = True

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
        fold_of = _fold_of_rows(meta_df, fold_map)  # NaN where non-common

        common = ~np.isnan(fold_of)
        in_fold = common & (fold_of == fold)

        roles = np.empty(n, dtype="<U7")
        roles[:] = Role.OBSERVE.value            # non-common default
        roles[common & ~in_fold] = Role.FIT.value
        roles[in_fold] = Role.PREDICT.value

        labels = _empty_label_array(n)
        labels[common & ~in_fold] = "CV2"        # retained, in-sample
        labels[in_fold] = "CV1"                  # masked, new genotype
        return roles, labels
