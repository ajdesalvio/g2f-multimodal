"""Dataset splitting utilities for train/validation/test partitioning."""

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

if TYPE_CHECKING:
    from cv.schemes import SchemeAssignment
    from utils.data.eval_streams import ValidationStrategy


class DatasetSplitter:
    """Splits a dataset into train, validation, and test sets based on metadata filters or ratios."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_dict: dict[str, Any],
        random_state: int | None = None,
    ):
        """
        Initializes the DatasetSplitter with metadata and data.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata about the dataset.
            data_dict (dict[str, Any]): Dictionary containing file-to-data mappings.
            random_state (int): Controls the shuffling applied to the data before applying the split.
                        Pass an int for reproducible output across multiple function calls
        """
        self.metadata_df = metadata_df
        self.data_dict = data_dict
        self.random_state = random_state

    def _shuffle(self, x: list[Any]) -> list[Any]:
        """
        Shuffles a list deterministically or randomly based on the configuration.

        Args:
            items (List[Any]): List to shuffle.

        Returns:
            List[Any]: Shuffled list.
        """
        if self.random_state is not None:
            # Save global RNG states, seed deterministically, then restore.
            torch_state = torch.get_rng_state().clone()
            np_state = np.random.get_state()
            random_state = random.getstate()
            pl.seed_everything(self.random_state)
            random.shuffle(x)
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)
            random.setstate(random_state)
        else:
            random.shuffle(x)
        return x

    def _include_files(
        self,
        metadata_df: pd.DataFrame,
        match_column: str,
        match_values: Any | list[Any],
        expand_columns: str | list[str] | None = None,
    ) -> set[str]:
        """
        Selects files based on specified match_values in a given column and optionally includes related entries.

        This function extracts filenames from `metadata_df` where `match_column` matches any value
        in `match_values`. Additionally, if `expand_columns` is provided, it includes files where
        the specified columns share match_values with the initially matched rows.

        Args:
            metadata_df (pd.DataFrame):
                A DataFrame containing metadata, including a "File" column.
            match_column (str):
                The column used to filter the DataFrame (e.g., "Env", "Year", "Pedigree").
            match_values (Any | list[Any]):
                A list of match_values to match in `match_column`. Files corresponding to these match_values
                will be included.
            expand_columns (str | list[str] | None, optional):
                One or more columns used to find additional related entries. If specified,
                the function includes all files that share match_values in `expand_columns` with
                the initially matched rows. Defaults to None.

        Returns:
            set[str]: A set of filenames that match the specified criteria.

        Raises:
            ValueError: If `match_column` is not found in `metadata_df`.
            ValueError: If the "File" column is missing from `metadata_df`.

        Example:
            Given the following `metadata_df`:

            | File      | Env  | Year | Pedigree          |
            |---------- |------|------|-------------------|
            | file1.csv | A    | 2021 | W10004_0086/PHP02 |
            | file2.csv | B    | 2021 | W10004_0086/PHP02 |
            | file3.csv | A    | 2022 | X20010_1234/PHP07 |
            | file4.csv | B    | 2022 | Y30020_5678/PHP09 |

            To include all files where `Pedigree == "W10004_0086/PHP02"`, run:

            ```python
            _include_files(metadata_df, match_column="Pedigree", match_values=["W10004_0086/PHP02"])
            ```
            This returns `{"file1.csv", "file2.csv"}`.

            If we also want to include files sharing the same `Env` as the matched entries:

            ```python
            _include_files(metadata_df, match_column="Pedigree", match_values=["W10004_0086/PHP02"], expand_columns="Env")
            ```
            This includes all files where `Env` is either "A" or "B", resulting in:

            `{"file1.csv", "file2.csv", "file3.csv", "file4.csv"}`.

        """
        # Ensure required columns exist

        if match_column not in metadata_df.columns:
            raise ValueError(
                f"Column '{match_column}' not found in metadata_df!"
            )

        if not match_values:
            print("[INFO] No match_values provided for inclusion.")
            return set()

        if not isinstance(match_values, list):
            match_values = [match_values]

        included_files = set()

        # Match files for the specified match_values
        matching_files = metadata_df[
            metadata_df[match_column].isin(match_values)
        ]
        included_files.update(matching_files["File"])

        print(
            f"[INFO] Matched {len(matching_files)} "
            f"files for '{match_column}' with values: {match_values}"
        )

        # Expand selection based on related columns
        if expand_columns:
            expand_columns = (
                [expand_columns]
                if isinstance(expand_columns, str)
                else expand_columns
            )

            for col in expand_columns:
                if col not in metadata_df.columns:
                    print(
                        f"[WARNING] Expand column '{col}' "
                        f"not found in metadata. Skipping..."
                    )
                    continue

                expand_col_values = matching_files[col].dropna().unique()
                if len(expand_col_values) == 0:
                    continue

                matching_expand_col_files = set(
                    metadata_df[metadata_df[col].isin(expand_col_values)][
                        "File"
                    ]
                )
                included_files.update(matching_expand_col_files)

        print(f"[INFO] Total files included: {len(included_files)}")
        return included_files

    def _split_metadata_into_two(
        self,
        metadata_df: pd.DataFrame,
        filter_columns: list[dict] | None = None,
        split_ratio: float | None = None,
    ) -> tuple[set[str], pd.DataFrame]:
        """
        Splits the dataset into a subset and the remaining data.

        Args:
            metadata_df (pd.DataFrame): Input DataFrame containing metadata.
            filter_columns (list[dict] | None): Column names and target values used for splitting.
                If provided, it is a list of dictionaries of the format:
                [
                    {
                        "col_name": values (Any | list[Any]),
                        "_expand_": col_name (str) or list[col_name (str)]
                    }
                ]
            split_ratio (float | None): Proportion of data to include in the split subset.

        Returns:
            tuple[set[str], pd.DataFrame]: Files in the split subset and remaining data as a DataFrame.
        """
        # Case 1: Use filter_columns to determine split
        if filter_columns:
            if split_ratio is not None and split_ratio > 0:
                print(
                    f"[WARNING] Both filter_columns and split_ratio={split_ratio} "
                    f"were provided. Using filter_columns; split_ratio is ignored."
                )

            candidate_files = set(metadata_df["File"])
            for filter_info in filter_columns:
                expand_columns = filter_info.get("_expand_")
                key_value_pairs = [
                    (k, v)
                    for k, v in filter_info.items()
                    if k != "_expand_"
                ]

                # AND across every (key, value) pair inside a single filter
                # dict: a file qualifies only if it matches ALL keys. So
                # {"Env": "DEH1", "Year": 2020} selects the DEH1-and-2020 rows,
                # not just DEH1 (the old `next(...)` kept only the first key and
                # silently dropped the rest). The multi-dict form
                # [{Env: DEH1}, {Year: 2020}] is the same conjunction expressed
                # across dicts and still works, since candidate_files is
                # intersected per dict below.
                dict_files: set[str] | None = None
                for match_column, match_values in key_value_pairs:
                    if not match_values:
                        continue
                    matched = self._include_files(
                        metadata_df=metadata_df,
                        match_column=match_column,
                        match_values=match_values,
                        expand_columns=expand_columns,
                    )
                    dict_files = (
                        matched
                        if dict_files is None
                        else dict_files & matched
                    )

                if dict_files is not None:
                    candidate_files &= dict_files
            split_files = candidate_files

        # Case 2: Use split_ratio to determine split
        else:
            if split_ratio is None or not 0 <= split_ratio <= 1:
                raise ValueError(
                    "Must specify either filter_columns or a valid split_ratio "
                    f"(0 ≤ split_ratio ≤ 1)."
                )

            all_files = self._shuffle(list(metadata_df["File"]))
            split_count = int(len(all_files) * split_ratio)
            split_files = set(all_files[:split_count])

        # Compute remaining files
        all_files_set = set(metadata_df["File"])
        remaining_files = all_files_set - split_files
        remaining_metadata_df = metadata_df[
            metadata_df["File"].isin(remaining_files)
        ]

        # Sanity checks
        if split_files & remaining_files:
            raise AssertionError("Split and remaining files are not disjoint!")

        if not remaining_files:
            print("[WARNING] No remaining files after split.")

        print(
            f"[INFO] Split {len(split_files)} files, "
            f"leaving {len(remaining_files)} remaining."
        )
        return split_files, remaining_metadata_df

    def split_metadata(
        self,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        test_filter_columns: list[dict] | None = None,
        val_filter_columns: list[dict] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the metadata into train, validation, and test sets.

        Args:
            test_ratio (float): Proportion of data to allocate to the test set.
            val_ratio (float): Proportion of data to allocate to the validation set.
            test_filter_columns (list[dict] | None): Criteria for selecting the test set.
            val_filter_columns (list[dict] | None): Criteria for selecting the validation set.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for train, validation, and test splits.
        """
        if not 0 <= test_ratio <= 1:
            raise ValueError("Test ratio must be between 0 and 1.")

        if not 0 <= val_ratio <= 1:
            raise ValueError("Validation ratio must be between 0 and 1.")

        if test_ratio + val_ratio > 1:
            raise ValueError(
                "The sum of test_ratio and val_ratio cannot exceed 1."
            )

        print("[INFO] Starting metadata split...")

        # Step 1: Split test set
        print("[INFO] Splitting test set...")
        test_files, remaining_metadata_df = self._split_metadata_into_two(
            metadata_df=self.metadata_df,
            filter_columns=test_filter_columns,
            split_ratio=test_ratio,
        )

        if not test_files:
            print(
                "[WARNING] Test split is empty — no test_filter_columns provided "
                "and test_ratio is 0. The test set will have 0 samples."
            )

        if remaining_metadata_df.empty:
            print(
                "[WARNING] Test split resulted in no remaining data for training or validation. "
                "Consider reducing test_ratio or revising test_filter_columns."
            )
            empty_df = self.metadata_df.head(0)  # Preserve column names
            return empty_df, empty_df, self.metadata_df

        # Step 2: Adjust validation ratio for remaining data
        adjusted_val_ratio = (
            val_ratio / (1 - test_ratio) if val_ratio > 0 else 0
        )
        # logging.info(f"Adjusted validation ratio for remaining data: {adjusted_val_ratio:.3f}")

        # Step 3: Split validation set
        print("[INFO] Splitting validation set...")
        val_files, train_metadata_df = self._split_metadata_into_two(
            metadata_df=remaining_metadata_df,
            filter_columns=val_filter_columns,
            split_ratio=adjusted_val_ratio,
        )

        if not val_files:
            print(
                "[WARNING] Validation split is empty — no val_filter_columns provided "
                "and val_ratio is 0. The validation set will have 0 samples."
            )

        if train_metadata_df.empty:
            print(
                "[WARNING] Validation split resulted in no remaining data for training. "
                "Consider reducing val_ratio or revising val_filter_columns."
            )
            empty_df = self.metadata_df.head(0)  # Preserve column names
            return (
                empty_df,
                self.metadata_df[self.metadata_df["File"].isin(val_files)],
                self.metadata_df[self.metadata_df["File"].isin(test_files)],
            )

        # Get train files
        train_files = set(train_metadata_df["File"])

        # Final log and return
        print(
            f"[INFO] Data split complete: {len(train_files)} train, "
            f"{len(val_files)} validation, {len(test_files)} test files."
        )
        return (
            self.metadata_df[self.metadata_df["File"].isin(train_files)],
            self.metadata_df[self.metadata_df["File"].isin(val_files)],
            self.metadata_df[self.metadata_df["File"].isin(test_files)],
        )

    def split_data(
        self, **kwargs
    ) -> tuple[list[dict], list[dict], list[dict], dict[str, list[int]]]:
        """
        Splits the actual data into train, validation, and test lists based on metadata splits.

        Returns:
            Tuple of (train_data, val_data, test_data, split_indices) where
            split_indices maps split names to lists of integer row positions
            in the original ``metadata_df``.
        """
        train_meta, val_meta, test_meta = self.split_metadata(**kwargs)

        def _extract_data(meta):
            return [self.data_dict[file] for file in meta["File"]]

        split_indices = {
            "train": list(train_meta.index),
            "val": list(val_meta.index),
            "test": list(test_meta.index),
        }

        return (
            _extract_data(train_meta),
            _extract_data(val_meta),
            _extract_data(test_meta),
            split_indices,
        )

    # ── Explicit role-based split (CV schemes) ──────────────────────

    def assign_roles(
        self,
        assignment: "SchemeAssignment",
        *,
        val_strategies: "list[ValidationStrategy] | None" = None,
        val_seed: int = 0,
    ) -> tuple[list[dict], list[dict], list[dict], dict[str, list[int]]]:
        """Build train/val/test splits from a per-row role table (D1).

        Unlike the filter/ratio path, the partition is dictated by an
        explicit ``(role, cv_label)`` assignment from a ``CVScheme``:

        - ``train = FIT ∪ OBSERVE`` (the rows the model observes, ``y``
          unmasked), minus any DL ``val`` carve-out.
        - ``test  = PREDICT`` (rows masked to ``y=NA`` and scored by label).
        - ``val`` is a DL-only early-stopping carve drawn from the observed
          set (D7), orthogonal to ``cv_label``. It is selected by the
          supplied ``val_strategies`` (``utils.data.eval_streams.carve``)
          rather than a flat ratio; ``val_strategies=None``/``[]`` (BGLR)
          leaves it empty. Carved rows have their ``cv_label`` cleared so
          the scorer never scores an early-stopping holdout.

        The carve is driven by a dedicated ``val_seed`` — a single local
        ``np.random.default_rng(val_seed)`` handed to every strategy,
        decoupled from ``misc.seed`` and from ``random_state`` (no
        global-RNG ``_shuffle`` dance). Multiple strategies select
        sequentially from the shrinking train pool and are asserted
        disjoint.

        Side effects: stashes ``self.fit_mask`` (the FIT-role mask, or
        ``None`` when there are no OBSERVE rows — the D3 regression gate)
        and ``self.cv_labels`` (per-``metadata_df``-row label array) for
        :class:`G2FDataset` to read. ``split_indices`` are stored as
        ``metadata_df`` positions, consistent with ``split_data`` (label ==
        position under the RangeIndex invariant).
        """
        n = len(self.metadata_df)
        if assignment.n != n:
            raise ValueError(
                f"assign_roles: assignment covers {assignment.n} rows but "
                f"metadata_df has {n}. The assignment must be built from "
                "the same (coverage-filtered) metadata_df."
            )
        if not self.metadata_df.index.equals(pd.RangeIndex(n)):
            raise AssertionError(
                "assign_roles requires metadata_df with a contiguous "
                "RangeIndex (split_indices are positional)."
            )

        train_positions = np.where(assignment.train_mask())[0]
        test_positions = np.where(assignment.predict_mask())[0]
        cv_labels = assignment.cv_labels.copy()

        # DL val carve(s) from the observed (train) set — orthogonal to
        # label. Each strategy selects from the *remaining* (post earlier
        # carves) pool, so the carves are disjoint by construction; the
        # ⊆-remaining assertion makes that explicit.
        val_positions = np.array([], dtype=int)
        if val_strategies:
            rng = np.random.default_rng(val_seed)
            carved: list[int] = []
            remaining = train_positions
            for strat in val_strategies:
                sel = np.asarray(
                    strat.select(self.metadata_df, remaining, rng=rng),
                    dtype=int,
                )
                remaining_set = set(remaining.tolist())
                outside = [int(p) for p in sel if int(p) not in remaining_set]
                if outside:
                    raise AssertionError(
                        f"val strategy {strat.name!r} selected positions "
                        f"outside the remaining train pool: {outside[:5]}."
                    )
                sel_set = set(sel.tolist())
                carved.extend(sel.tolist())
                remaining = np.array(
                    [p for p in remaining if p not in sel_set], dtype=int
                )
            val_positions = np.array(sorted(carved), dtype=int)
            train_positions = remaining
            # A val row is an early-stopping holdout, never scored (D7).
            for p in val_positions:
                cv_labels[p] = None

        files = self.metadata_df["File"].to_numpy()

        def _extract(positions: np.ndarray) -> list[dict]:
            return [self.data_dict[files[p]] for p in positions]

        split_indices = {
            "train": train_positions.tolist(),
            "val": val_positions.tolist(),
            "test": test_positions.tolist(),
        }

        self.fit_mask = assignment.fit_mask()
        self.cv_labels = cv_labels

        return (
            _extract(train_positions),
            _extract(val_positions),
            _extract(test_positions),
            split_indices,
        )
