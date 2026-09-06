"""``CV_0_00`` split group — one environment removed **plus** a female-fold.

One BGLR fit → two quadrants, scored on the held-out env column only
(Background table, transcribed from R, ``k = fold``):

    PREDICT (y=NA) = (common & Fold==k) OR (Env==heldout)
    observed       = everything else
    basis fit set  = common & Fold!=k & Env!=heldout
    scored subset  = Env==heldout only:
        CV0  = Env==heldout & common & Fold!=k     (masked via env)
        CV00 = Env==heldout & common & Fold==k     (masked via env+fold)

Role mapping (R ``mask_yields`` / ``train_ids`` / ``evaluate_metrics``):

    masked  = (common & Fold==k) | (Env==heldout)        # mask_yields
    in_fit  = common & Fold!=k & Env!=heldout            # train_ids
      in_fit                              → FIT,     label None
      masked & common & env & Fold!=k     → PREDICT, label "CV0"
      masked & common & env & Fold==k     → PREDICT, label "CV00"
      masked otherwise                    → PREDICT, label None  (unscored)
      not masked, not in_fit              → OBSERVE, label None  (non-common, off-env)

Masked-but-unscored rows (``Fold==k`` in non-held-out envs; non-common in
the held-out env) are still masked so held-out genotypes are unseen
everywhere. Weather regime is ``LOEO``: the held-out column has zero
train-split rows, so weather projects it (D4) with no config change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import CVScheme, Role, _empty_label_array, _fold_of_rows
from .identifiers import env_year_series, normalize_id

if TYPE_CHECKING:
    import pandas as pd


class CV0_00Scheme(CVScheme):
    name = "cv_0_00"
    requires_heldout_env = True
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
        fold_of = _fold_of_rows(meta_df, fold_map)        # NaN where non-common
        env = env_year_series(meta_df).to_numpy()
        heldout = normalize_id(heldout_env)

        common = ~np.isnan(fold_of)
        in_fold = common & (fold_of == fold)
        env_match = env == heldout

        masked = (common & in_fold) | env_match           # R mask_yields
        in_fit = common & ~in_fold & ~env_match           # R train_ids

        roles = np.empty(n, dtype="<U7")
        roles[:] = Role.OBSERVE.value                     # not-masked & not-fit
        roles[in_fit] = Role.FIT.value
        roles[masked] = Role.PREDICT.value

        labels = _empty_label_array(n)
        labels[common & env_match & ~in_fold] = "CV0"     # tested lines, new env
        labels[common & env_match & in_fold] = "CV00"     # untested lines, new env
        return roles, labels
