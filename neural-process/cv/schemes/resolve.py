"""Bridge: a resolved ``cv_spec`` + a ``metadata_df`` → a ``SchemeAssignment``.

This is the single entry point the dataset build calls (Step 2). It ties
together the common-female determination (D6), the fold backend (D5), and
the scheme's role/label rules (D1) so callers never wire those three by
hand.

The ``cv_spec`` shape (D9):

    {
      "scheme": "cv_0_00",            # cv_2_1 | cv_0_00 | env_year_loo | random_kfold
      "cv_seed": 1,                   # R Seed_Num — drives fold assignment
      "fold": 3,                      # required by cv_2_1 / cv_0_00
      "heldout_env": "DEH1.2020",     # required by cv_0_00 / env_year_loo
      "fold_source": "r_csv",         # r_csv | native
      "fold_csv": "/path/female_folds.csv",   # for r_csv
      "k_folds": 5,                   # for native
      "fold_validation": "strict",    # strict | subset (r_csv only)
    }
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from .base import _fold_of_rows
from .common_females import common_female_set
from .fold_source import FoldValidationError, build_fold_source
from .registry import get_scheme

if TYPE_CHECKING:
    import pandas as pd

    from .base import SchemeAssignment


def resolve_cv_assignment(
    metadata_df: "pd.DataFrame",
    cv_spec: dict[str, Any],
    *,
    common_universe_df: "pd.DataFrame",
) -> "SchemeAssignment":
    """Resolve a ``cv_spec`` against ``metadata_df`` into a role/label table.

    ``metadata_df`` must be the coverage-filtered frame with a contiguous
    ``RangeIndex`` (the genotyped universe = R's ``order``), so the role
    arrays align positionally with the dataset rows the splitter will
    partition.

    ``common_universe_df`` is the frame the **common-female set** (and its
    ``total_envs`` denominator) is derived from. This is the single source
    of truth for "common", and it must be the *full phenotype* frame —
    before the genotype-coverage filter — to match R's ``Metadata`` script
    (D6), independent of ``fold_source``. It is required: there is no
    genotyped-universe fallback, so a coverage drop can never silently
    redefine "common". Membership is applied to ``metadata_df`` rows via
    ``fold_map``: a female common over the full pheno but absent from the
    genotyped frame contributes no rows; one common over the full pheno yet
    present here stays foldable even if a coverage drop lowered its env
    incidence.
    """
    if "scheme" not in cv_spec:
        raise KeyError("cv_spec is missing the required 'scheme' key.")
    scheme = get_scheme(cv_spec["scheme"])

    # Row-unit schemes (random_kfold) partition the metadata ROWS directly
    # via a seeded balanced k-fold — no common-female set, no maternal-line
    # fold_map. Build the per-row fold array and thread it in as row_folds;
    # it doubles as the SchemeAssignment.folds (the D8 predictions Fold col).
    if getattr(scheme, "fold_unit", "female") == "row":
        if "cv_seed" not in cv_spec:
            raise KeyError(
                f"cv_spec for scheme {scheme.name!r} requires 'cv_seed' "
                "(it seeds the per-row fold partition)."
            )
        from .fold_source import random_row_fold_array

        row_folds = random_row_fold_array(
            len(metadata_df),
            cv_seed=cv_spec["cv_seed"],
            k_folds=cv_spec.get("k_folds", 5),
        )
        fold = cv_spec.get("fold")
        if fold is not None and not (row_folds == int(fold)).any():
            # The truncated R-shape partition can leave trailing folds
            # empty; without this, the run would score zero rows and write
            # an empty-by_metric metrics file with no error.
            raise FoldValidationError(
                f"{scheme.name}: fold={fold} selected zero PREDICT rows — "
                f"the k-fold partition of n={len(metadata_df)} rows into "
                f"k_folds={cv_spec.get('k_folds', 5)} left it empty. "
                "Lower k_folds or request a non-empty fold."
            )
        assignment = scheme.assign_table(
            metadata_df,
            {},
            fold=cv_spec.get("fold"),
            heldout_env=None,
            row_folds=row_folds,
        )
        return dataclasses.replace(assignment, folds=row_folds.astype(float))

    fold_map: dict[str, int] = {}
    if scheme.requires_fold:
        if "cv_seed" not in cv_spec:
            raise KeyError(
                f"cv_spec for scheme {scheme.name!r} requires 'cv_seed'."
            )
        common = common_female_set(common_universe_df)
        source = build_fold_source(
            fold_source=cv_spec.get("fold_source", "native"),
            cv_seed=cv_spec["cv_seed"],
            k_folds=cv_spec.get("k_folds", 5),
            fold_csv=cv_spec.get("fold_csv"),
            fold_validation=cv_spec.get("fold_validation", "strict"),
        )
        fold_map = source.build(common)
        fold = cv_spec.get("fold")
        if fold is not None and int(fold) not in set(fold_map.values()):
            # Same guard as the row-unit branch: an empty (or out-of-range)
            # fold would silently produce zero CV1/CV0 rows downstream.
            raise FoldValidationError(
                f"{scheme.name}: fold={fold} has no members in the fold "
                f"partition (present folds: "
                f"{sorted(set(fold_map.values()))}). The k-fold partition "
                f"of {len(fold_map)} common females left it empty — lower "
                "k_folds or request a non-empty fold."
            )

    assignment = scheme.assign_table(
        metadata_df,
        fold_map,
        fold=cv_spec.get("fold"),
        heldout_env=cv_spec.get("heldout_env"),
    )
    # Attach the per-row fold (NaN for non-common / fold-less schemes) so
    # the D8 predictions.csv can carry a ``Fold`` column without re-deriving
    # the fold_map downstream.
    folds = _fold_of_rows(metadata_df, fold_map)
    return dataclasses.replace(assignment, folds=folds)
