"""Core CV-scheme abstractions: ``Role``, ``RowAssignment``, ``CVScheme``.

A scheme emits, for one split job, a per-row ``(role, cv_label)`` table
that is the **single source of truth** for masking and scoring (D1). The
two are emitted together so they can never drift — the failure mode R
avoids by colocating ``mask_yields`` and ``evaluate_metrics``.

Three roles (D1):

==========  ==================  ==================  =====================
Role        ``y`` seen by       In FPCA basis fit   ``CV_2_1`` example
            model (unmasked)    (vi_fpca fit-mask)
==========  ==================  ==================  =====================
FIT         yes                 yes                 common, ``Fold!=k``
OBSERVE     yes                 no (projected)      non-common females
PREDICT     no (masked)         no (projected)      common, ``Fold==k``
==========  ==================  ==================  =====================

``role`` drives masking (``PREDICT`` → ``y=NA``) and the VI-FPCA basis
fit (``FIT`` → in basis); ``cv_label`` drives scoring (D2). Masked rows
that carry no label (e.g. ``Fold==k`` in a non-held-out env under
``cv_0_00``) are predicted but **unscored** — they must still be masked
so held-out genotypes are unseen everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class Role(str, Enum):
    """Per-row role. ``str`` mixin so values compare/serialize as plain strings."""

    FIT = "FIT"
    OBSERVE = "OBSERVE"
    PREDICT = "PREDICT"


@dataclass(frozen=True)
class RowAssignment:
    """One row's scheme assignment (the D1 record)."""

    metadata_idx: int
    role: Role
    cv_label: str | None


@dataclass(frozen=True)
class SchemeAssignment:
    """Vectorized role/label assignment over ``metadata_df`` rows.

    ``roles`` and ``cv_labels`` are positional arrays aligned to
    ``metadata_df``'s ``RangeIndex`` (the row-alignment invariant the
    dataset-build assertion guards). The engine consumes the arrays
    directly; ``to_rows()`` produces the equivalent
    ``list[RowAssignment]`` form for row-wise consumers.
    """

    roles: np.ndarray        # dtype '<U7' (Role values), shape (N,)
    cv_labels: np.ndarray    # dtype object, shape (N,); None where unscored
    scheme: str
    fold: int | None
    heldout_env: str | None
    #: Per-row fold number (float; ``NaN`` for non-common / fold-less
    #: schemes). Populated by :func:`resolve_cv_assignment` for the D8
    #: ``predictions.csv`` ``Fold`` column; ``None`` when unset.
    folds: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.roles.shape != self.cv_labels.shape:
            raise ValueError(
                f"roles {self.roles.shape} and cv_labels "
                f"{self.cv_labels.shape} must have the same shape"
            )
        if self.folds is not None and self.folds.shape != self.roles.shape:
            raise ValueError(
                f"folds {self.folds.shape} must match roles "
                f"{self.roles.shape}"
            )

    @property
    def n(self) -> int:
        return int(self.roles.shape[0])

    def role_mask(self, role: Role) -> np.ndarray:
        return self.roles == role.value

    def train_mask(self) -> np.ndarray:
        """Rows the model observes (``y`` unmasked) = FIT ∪ OBSERVE."""
        return self.roles != Role.PREDICT.value

    def predict_mask(self) -> np.ndarray:
        """Rows masked to ``y=NA`` and predicted = PREDICT."""
        return self.roles == Role.PREDICT.value

    def fit_mask(self) -> np.ndarray | None:
        """Explicit VI-FPCA basis-fit set (FIT rows), or ``None``.

        Returns ``None`` when there are **no** ``OBSERVE`` rows, i.e. when
        ``FIT`` equals the train split (FIT ∪ OBSERVE). In that case the
        engine must behave byte-identically to today (``fit_mask is None``
        regression gate, D3) — there is no fit/train distinction to make.
        Otherwise returns the boolean FIT mask so ``vi_fpca`` fits the
        basis on FIT rows only and projects the in-split ``OBSERVE`` rows.
        """
        if not self.role_mask(Role.OBSERVE).any():
            return None
        return self.role_mask(Role.FIT)

    def labels_present(self) -> list[str]:
        """Distinct non-``None`` ``cv_label`` values, in first-seen order."""
        seen: dict[str, None] = {}
        for lab in self.cv_labels:
            if lab is not None and lab not in seen:
                seen[lab] = None
        return list(seen)

    def to_rows(self) -> list[RowAssignment]:
        """Expand to the ``list[RowAssignment]`` contract (D1)."""
        return [
            RowAssignment(int(i), Role(self.roles[i]), self.cv_labels[i])
            for i in range(self.n)
        ]


class CVScheme(ABC):
    """A cross-validation scheme: ``(meta_df, fold_map, fold, heldout_env)``
    → per-row role/label table.

    Subclasses encode the role/label rules from the scheme table in
    ``docs/cv_schemes.md`` verbatim by implementing ``_assign_arrays``. ``assign`` (list
    form) and ``assign_table`` (vectorized form) are provided by the base.
    """

    name: ClassVar[str]
    #: Whether ``heldout_env`` is required (``cv_0_00``, ``env_year_loo``).
    requires_heldout_env: ClassVar[bool] = False
    #: Whether ``fold``/``fold_map`` partition rows (``cv_2_1``, ``cv_0_00``).
    requires_fold: ClassVar[bool] = False
    #: The unit the fold partition is keyed on. ``"female"`` (the default)
    #: ⇒ a maternal-line ``fold_map`` from the common-female set (the R
    #: parity path). ``"row"`` ⇒ a per-row partition over the metadata rows
    #: (``random_kfold``); ``resolve_cv_assignment`` then builds a ``row_folds``
    #: array (not a female ``fold_map``) and passes it to ``assign_table``.
    fold_unit: ClassVar[str] = "female"

    def assign_table(
        self,
        meta_df: "pd.DataFrame",
        fold_map: dict[str, int],
        *,
        fold: int | None = None,
        heldout_env: str | None = None,
        row_folds: "np.ndarray | None" = None,
    ) -> SchemeAssignment:
        """Vectorized assignment over ``meta_df`` rows.

        ``row_folds`` is the per-row fold array supplied for row-unit
        schemes (``fold_unit == "row"``); ``None`` for the female-keyed
        schemes, which read folds from ``fold_map`` instead.
        """
        self._validate_args(meta_df, fold=fold, heldout_env=heldout_env)
        roles, labels = self._assign_arrays(
            meta_df, fold_map, fold=fold, heldout_env=heldout_env,
            row_folds=row_folds,
        )
        return SchemeAssignment(
            roles=roles,
            cv_labels=labels,
            scheme=self.name,
            fold=fold,
            heldout_env=heldout_env,
        )

    def assign(
        self,
        meta_df: "pd.DataFrame",
        fold_map: dict[str, int],
        *,
        fold: int | None = None,
        heldout_env: str | None = None,
    ) -> list[RowAssignment]:
        """The plan's documented contract: a ``list[RowAssignment]``."""
        return self.assign_table(
            meta_df, fold_map, fold=fold, heldout_env=heldout_env
        ).to_rows()

    def _validate_args(
        self,
        meta_df: "pd.DataFrame",
        *,
        fold: int | None,
        heldout_env: str | None,
    ) -> None:
        import pandas as pd

        if not meta_df.index.equals(pd.RangeIndex(len(meta_df))):
            raise ValueError(
                f"{self.name}: meta_df must carry a contiguous RangeIndex "
                "(label == position); the role arrays are positional."
            )
        if self.requires_fold and fold is None:
            raise ValueError(f"{self.name}: scheme requires a 'fold' argument.")
        if self.requires_heldout_env and heldout_env is None:
            raise ValueError(
                f"{self.name}: scheme requires a 'heldout_env' argument."
            )

    @abstractmethod
    def _assign_arrays(
        self,
        meta_df: "pd.DataFrame",
        fold_map: dict[str, int],
        *,
        fold: int | None,
        heldout_env: str | None,
        row_folds: "np.ndarray | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(roles, cv_labels)`` positional arrays of length ``len(meta_df)``.

        ``row_folds`` is only populated for row-unit schemes
        (``fold_unit == "row"``); female-keyed schemes ignore it and read
        folds from ``fold_map``.
        """
        raise NotImplementedError


def _empty_label_array(n: int) -> np.ndarray:
    """Object array of ``None`` (so labels can be ``str`` or ``None``)."""
    return np.full(n, None, dtype=object)


def _fold_of_rows(meta_df: "pd.DataFrame", fold_map: dict[str, int]) -> np.ndarray:
    """Per-row fold number (``float``; ``NaN`` for non-common = not in fold_map).

    ``NaN`` is R's ``is.na(Fold)`` sentinel for non-common rows.
    """
    from .identifiers import female_series

    fem = female_series(meta_df)
    return fem.map(lambda f: fold_map.get(f, np.nan)).to_numpy(dtype=float)
