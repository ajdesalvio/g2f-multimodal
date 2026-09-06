"""Fold backends: ``native`` (seeded grouped k-fold) and ``r_csv`` (import
the R reference's ``female_folds.csv``).

Both produce a ``fold_map: dict[Female, int]``; a female is **foldable
(common)** iff it has an entry (D5). The common-female *set* is
deterministic (D5/D6), so it is reproducible from the data alone — only
the fold *assignment* needs R.

- ``NativeFoldSource`` — common set → seeded grouped k-fold. **Documented
  non-parity**: NumPy's RNG ≠ R's ``set.seed``/``sample``, so native
  folds differ from ``r_csv`` and per-split numbers won't match the R reference.
  The FIT/OBSERVE/PREDICT *structure* is faithful (it depends only on the
  deterministic common set), so the scheme stays valid on a different,
  statistically-equivalent partition.
- ``RCsvFoldSource`` — read ``female_folds.csv`` (``Seed_Num, Female,
  Fold``), filter ``Seed_Num``, run the D5 validation contract (normalize
  → set-equality → coverage), and **hard-fail** on data/CSV
  disagreement. It never silently falls back to ``native`` and never
  silently mislabels FIT/OBSERVE/PREDICT.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from .identifiers import normalize_id

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class FoldValidationError(ValueError):
    """Raised when ``r_csv`` folds disagree with the data (D5 contract)."""


def _r_shape_fold_base(n: int, k_folds: int) -> "np.ndarray":
    """R-shape fold base: ``rep(seq_len(k), each=ceil(n/k), length.out=n)``.

    NOT balanced: the truncation can starve trailing folds (e.g. ``n=8,
    k=5 -> sizes [2, 2, 2, 2, 0]``; ``n=21, k=5 -> [5, 5, 5, 5, 1]``).
    The shape is kept bit-identical to the R recipe so seeded partitions
    stay reproducible; a partition that leaves folds empty is logged here,
    and *requesting* an empty fold is a hard error in
    ``resolve_cv_assignment``.
    """
    base = np.resize(
        np.repeat(np.arange(1, k_folds + 1), math.ceil(n / k_folds)), n
    )
    sizes = np.bincount(base, minlength=k_folds + 1)[1:]
    if (sizes == 0).any():
        empty = [int(f) for f in np.flatnonzero(sizes == 0) + 1]
        logger.warning(
            "k-fold partition of n=%d into k_folds=%d leaves fold(s) %s "
            "empty (sizes %s); requesting an empty fold will fail with "
            "FoldValidationError.",
            n,
            k_folds,
            empty,
            sizes.tolist(),
        )
    return base


def random_row_fold_array(
    n_rows: int, *, cv_seed: int, k_folds: int = 5
) -> "np.ndarray":
    """Per-ROW seeded k-fold over ``n_rows`` metadata rows.

    The row-unit analogue of :meth:`NativeFoldSource.build` (same R-shape
    ``np.resize(np.repeat(...))`` base + seeded ``permutation`` — see
    :func:`_r_shape_fold_base` for why it is *not* balanced), but the
    partition unit is the metadata **row position**, not the maternal line.
    This is the plain IID k-fold used by ``random_kfold``: every row is
    independently assigned a fold in ``1..k_folds`` — no common-female /
    environment grouping — so the held-out fold is a random sample of the
    whole dataset.

    Returns a length-``n_rows`` integer array of fold numbers (``1..k``),
    aligned positionally to the (coverage-filtered, RangeIndex) metadata
    frame. Reproducible from ``cv_seed`` alone; **not** bit-equal to R.
    """
    if k_folds < 1:
        raise ValueError(f"k_folds must be >= 1, got {k_folds}")
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    if n_rows == 0:
        return np.array([], dtype=int)
    base = _r_shape_fold_base(n_rows, k_folds)
    rng = np.random.default_rng(int(cv_seed))
    return rng.permutation(base)


class NativeFoldSource:
    """Seeded grouped k-fold over the deterministic common-female set.

    Mirrors the *shape* of ``DAP_CV_Metadata_V1.R::make_fold_map``
    (``rep(seq_len(k), each=ceil(n/k), length.out=n)`` then a permutation)
    but with NumPy's RNG — so the partition is reproducible within Python
    yet **not** bit-equal to R. Use ``r_csv`` for R parity.
    """

    kind = "native"

    def __init__(self, cv_seed: int, k_folds: int = 5):
        if k_folds < 1:
            raise ValueError(f"k_folds must be >= 1, got {k_folds}")
        self.cv_seed = int(cv_seed)
        self.k_folds = int(k_folds)

    def build(self, common_females: set[str]) -> dict[str, int]:
        females = sorted(normalize_id(f) for f in common_females)
        n = len(females)
        if n == 0:
            raise FoldValidationError(
                "NativeFoldSource: the common-female set is empty; no rows "
                "could ever be folded into FIT/PREDICT."
            )
        base = _r_shape_fold_base(n, self.k_folds)
        rng = np.random.default_rng(self.cv_seed)
        folds = rng.permutation(base)
        return {fem: int(folds[i]) for i, fem in enumerate(females)}


class RCsvFoldSource:
    """Import the R reference's ``female_folds.csv`` for bit-exact fold parity.

    Validation contract (D5), run in ``build``:

    1. Normalize ``Female`` on both sides (lowercase) before comparing.
       Near-zero overlap ⇒ case/format bug → hard error.
    2. Set-equality between the CSV's folded females (for this ``cv_seed``)
       and the data's deterministic common set. ``strict`` requires exact
       equality; ``subset`` allows ``data_common ⊆ csv`` (with a warning),
       but ``data_common ⊄ csv`` is always fatal.
    3. Coverage: the requested ``cv_seed`` must exist in the CSV.

    The env-universe check (D5 #3) is enforced upstream by recomputing the
    common set on the data's ``Env.Year`` universe (``common_females.py``);
    a different env set changes what "common" means and surfaces here as a
    set-equality failure.
    """

    kind = "r_csv"

    def __init__(
        self,
        csv_path: str,
        cv_seed: int,
        fold_validation: Literal["strict", "subset"] = "strict",
    ):
        if fold_validation not in ("strict", "subset"):
            raise ValueError(
                f"fold_validation must be 'strict' or 'subset', got "
                f"{fold_validation!r}"
            )
        self.csv_path = csv_path
        self.cv_seed = int(cv_seed)
        self.fold_validation = fold_validation

    def _read_seed_rows(self) -> "pd.DataFrame":
        import pandas as pd

        df = pd.read_csv(self.csv_path)
        cols = {c.lower(): c for c in df.columns}
        for req in ("seed_num", "female", "fold"):
            if req not in cols:
                raise FoldValidationError(
                    f"RCsvFoldSource: {self.csv_path} is missing required "
                    f"column {req!r}. Found columns: {list(df.columns)}."
                )
        seed_col, fem_col, fold_col = cols["seed_num"], cols["female"], cols["fold"]

        seed_rows = df[df[seed_col].astype(int) == self.cv_seed]
        if seed_rows.empty:
            available = sorted(df[seed_col].astype(int).unique())
            raise FoldValidationError(
                f"RCsvFoldSource: cv_seed={self.cv_seed} not found in "
                f"{self.csv_path}. Available Seed_Num values: {available}."
            )
        return seed_rows.rename(
            columns={fem_col: "Female", fold_col: "Fold"}
        )[["Female", "Fold"]]

    def build(self, common_females: set[str]) -> dict[str, int]:
        seed_rows = self._read_seed_rows()

        fold_map: dict[str, int] = {}
        for fem, fold in zip(seed_rows["Female"], seed_rows["Fold"]):
            fold_map[normalize_id(fem)] = int(fold)

        data_common = {normalize_id(f) for f in common_females}
        csv_females = set(fold_map)

        # (1) Overlap sanity — catches case/format bugs before set-equality.
        if data_common and csv_females:
            overlap = len(csv_females & data_common)
            frac = overlap / max(len(csv_females), len(data_common))
            if frac < 0.1:
                raise FoldValidationError(
                    "RCsvFoldSource: near-zero overlap between CSV females "
                    f"({len(csv_females)}) and the data's common set "
                    f"({len(data_common)}); only {overlap} match. This is "
                    "almost certainly an identifier case/format mismatch. "
                    f"CSV sample: {sorted(csv_females)[:3]}; "
                    f"data sample: {sorted(data_common)[:3]}."
                )

        # (2) Set-equality / subset.
        in_csv_not_common = csv_females - data_common
        common_not_in_csv = data_common - csv_females
        if common_not_in_csv:
            raise FoldValidationError(
                "RCsvFoldSource: data common-female set is not covered by "
                f"the CSV for cv_seed={self.cv_seed}. "
                f"{len(common_not_in_csv)} common female(s) missing from CSV "
                f"(e.g. {sorted(common_not_in_csv)[:5]}). This indicates a "
                "stale CSV or a wrong dataset version."
            )
        if in_csv_not_common:
            if self.fold_validation == "strict":
                raise FoldValidationError(
                    "RCsvFoldSource: strict validation — CSV has "
                    f"{len(in_csv_not_common)} female(s) not common in the "
                    f"data (e.g. {sorted(in_csv_not_common)[:5]}). Set "
                    "fold_validation='subset' to run R's folds on a "
                    "female-subset (e.g. smoke tests)."
                )
            warnings.warn(
                "RCsvFoldSource: subset validation — CSV has "
                f"{len(in_csv_not_common)} female(s) not common in the data; "
                "running R's folds on a female-subset. Per-split numbers "
                "remain comparable only over the shared common set.",
                stacklevel=2,
            )
            # Keep only folds for females actually common in the data, so
            # "foldable ⇔ in fold_map" stays provably safe.
            fold_map = {f: k for f, k in fold_map.items() if f in data_common}

        return fold_map


def build_fold_source(
    fold_source: str,
    cv_seed: int,
    *,
    k_folds: int = 5,
    fold_csv: str | None = None,
    fold_validation: Literal["strict", "subset"] = "strict",
) -> NativeFoldSource | RCsvFoldSource:
    """Construct the configured fold backend from ``cv_spec`` fields (D9)."""
    if fold_source == "native":
        return NativeFoldSource(cv_seed=cv_seed, k_folds=k_folds)
    if fold_source == "r_csv":
        if not fold_csv:
            raise ValueError(
                "fold_source='r_csv' requires 'fold_csv' (path to "
                "female_folds.csv)."
            )
        return RCsvFoldSource(
            csv_path=fold_csv, cv_seed=cv_seed, fold_validation=fold_validation
        )
    raise ValueError(
        f"Unknown fold_source {fold_source!r}; expected 'native' or 'r_csv'."
    )
