"""``random_kfold`` — plain per-row random k-fold over ALL rows.

The in-distribution baseline scheme: unlike the structured G×E schemes
(``cv_2_1`` / ``cv_0_00`` hold out maternal-line folds and/or whole
environments; ``env_year_loo`` holds out an environment column), this one
draws the test fold as a **uniform random sample of every row**. No
common-female grouping, no environment masking — so the held-out fold is
drawn from the *same* distribution the model trains on. Use it to probe a
model's raw capacity, free of the out-of-distribution penalty the other
schemes impose.

    PREDICT (y=NA) = row_folds == fold          → label "test"
    FIT            = row_folds != fold

The per-row fold partition is a seeded balanced k-fold over the metadata
row positions (``random_row_fold_array``), threaded in by
``resolve_cv_assignment`` as ``row_folds`` (this is a ``fold_unit == "row"``
scheme, so it does NOT consume a maternal-line ``fold_map``). Like
``env_year_loo`` there are no ``OBSERVE`` rows, so ``fit_mask`` is ``None``
and engine behaviour is byte-identical to the established single-label
path. A single ``"test"`` label means the existing label-generic scorer
and the role-based ``test`` observe stream cover it with no new selectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import CVScheme, Role, _empty_label_array

if TYPE_CHECKING:
    import pandas as pd


class RandomKFoldScheme(CVScheme):
    name = "random_kfold"
    requires_heldout_env = False
    requires_fold = True
    fold_unit = "row"

    def _assign_arrays(
        self,
        meta_df: "pd.DataFrame",
        fold_map: dict[str, int],
        *,
        fold: int | None,
        heldout_env: str | None,
        row_folds: "np.ndarray | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if row_folds is None:
            raise ValueError(
                "random_kfold requires a per-row 'row_folds' array; it is "
                "supplied by resolve_cv_assignment for fold_unit='row' "
                "schemes. Call assign_table(..., row_folds=...)."
            )
        n = len(meta_df)
        if len(row_folds) != n:
            raise ValueError(
                f"random_kfold: row_folds has length {len(row_folds)} but "
                f"meta_df has {n} rows; they must align positionally."
            )

        in_fold = np.asarray(row_folds) == fold

        roles = np.empty(n, dtype="<U7")
        roles[:] = Role.FIT.value
        roles[in_fold] = Role.PREDICT.value

        labels = _empty_label_array(n)
        labels[in_fold] = "test"
        return roles, labels
