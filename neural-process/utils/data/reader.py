"""Data reader for parsing, validating, and extracting G2F vegetation index CSV files."""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .helpers import (
    vi_nonuniform_interpolate,
    apply_case_conversions,
    form_string_matching,
)


class DataReader:
    """Reads G2F CSV files, validates content against filename metadata, interpolates missing VI values, and extracts tensors."""

    def __init__(
        self,
        data_dir: str,
        validate: bool = True,
        # Whether to perform case-sensitive comparisons when
        # validating metadata against data content
        case_sensitive: bool = False,
        delete_invalid: bool = True,
        data_df_columns: list = [
            "Pedigree",
            "Vegetation.Index",
            "Year",
            "Env",
            ("VI.BLUE.", "prefix"),
            "Yield.t.ha.BLUE",
        ],
        metadata_df_columns: list[str] = ["Pedigree", "Year", "Env", "File"],
        data_df_fields_to_convert: dict | None = None,
        metadata_df_fields_to_convert: dict | None = None,
        to_tensor: bool = True,
        cache_dir: str | None = None,
    ):
        """
        Initialize the DataReader.

        Args:
            data_dir: Path to the directory containing CSV data files.
            validate: Whether to validate data content against filename metadata.
            case_sensitive: Whether to use case-sensitive comparisons during validation.
            delete_invalid: Whether to delete files that fail validation.
            data_df_columns: Expected column names (or prefix tuples) in data CSVs.
            metadata_df_columns: Expected column names in the parsed metadata DataFrame.
            data_df_fields_to_convert: Case conversion rules for data DataFrames.
            metadata_df_fields_to_convert: Case conversion rules for the metadata DataFrame.
            to_tensor: Whether to convert extracted arrays to PyTorch tensors.
            cache_dir: Directory for caching processed data. None (default)
                places the cache next to ``data_dir`` — i.e. in
                ``{parent_of_data_dir}/.cache/`` — so multiple sibling CSV
                directories under the same dataset root share one cache
                and the cache does not sit inside the raw-data folder.
                Set to False to disable caching entirely.

        Per-VI subsetting lives on ``VIFPCAProcessor.vi_subset`` as a
        post-load slice — the reader always loads every vegetation index
        present in the CSV, and downstream consumers slice the cached
        full-fit FPC scores. This keeps Design B intact end-to-end.
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"Invalid directory: {data_dir}")
        self.data_dir = data_dir
        self.validate = validate
        self.case_sensitive = case_sensitive
        self.delete_invalid = delete_invalid
        self.data_df_columns = data_df_columns
        self.metadata_df_columns = metadata_df_columns
        self.data_df_fields_to_convert = data_df_fields_to_convert
        self.metadata_df_fields_to_convert = metadata_df_fields_to_convert
        self.to_tensor = to_tensor
        # Canonical VI ordering captured at load time. Populated by
        # ``load()`` (or restored from the .pt cache); used by downstream
        # consumers to map VI names → tensor column indices.
        self.vi_names_: list[str] | None = None
        # Resolve cache directory
        if cache_dir is None:
            parent = os.path.dirname(os.path.normpath(data_dir))
            self.cache_dir = os.path.join(parent, ".cache")
        elif cache_dir is False:
            self.cache_dir = None
        else:
            self.cache_dir = cache_dir

    def _compute_cache_key(self) -> str:
        """Compute a deterministic hash of all parameters that affect load() output."""
        # Collect file state: sorted filenames + modification times
        csv_files = sorted(f for f in os.listdir(self.data_dir) if f.endswith(".csv"))
        file_state = {
            f: os.path.getmtime(os.path.join(self.data_dir, f)) for f in csv_files
        }

        config = {
            # Sample-dict schema version. Bumped when the per-sample dict
            # layout changes (here: days_after_plant/vegetation_index_values
            # -> dap/channels + channel_names) so stale `.pt` caches written
            # under the old layout are not reused.
            "schema": "coords_channels_v1",
            "data_dir": os.path.abspath(self.data_dir),
            "validate": self.validate,
            "case_sensitive": self.case_sensitive,
            "delete_invalid": self.delete_invalid,
            "to_tensor": self.to_tensor,
            "data_df_columns": [
                list(c) if isinstance(c, tuple) else c for c in self.data_df_columns
            ],
            "metadata_df_columns": self.metadata_df_columns,
            "data_df_fields_to_convert": self.data_df_fields_to_convert,
            "metadata_df_fields_to_convert": self.metadata_df_fields_to_convert,
            "file_state": file_state,
        }
        raw = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _parse_metadata(
        self,
        pattern: str = r"(?P<Env>[^.]+)\.(?P<Year>[^.]+)\.(?P<Inbred>[^.]+)\.(?P<Tester>[^.]+)\.csv",
    ) -> pd.DataFrame:
        """
        Parses metadata from file names in the data directory.

        Args:
            pattern (str): Regex pattern for parsing file names.

        Returns:
            pd.DataFrame: DataFrame with metadata (file, Env, Year as int, Pedigree).
        """
        regex = re.compile(pattern)
        all_files = [
            f for f in os.listdir(self.data_dir) if f.endswith(".csv")
        ]
        metadata = []

        for file in tqdm(all_files, desc="Parsing metadata"):
            match = regex.match(file)
            if match:
                data = match.groupdict()
                data["File"] = file
                data["Year"] = int(data["Year"])  # Convert Year to integer
                data["Pedigree"] = f"{data['Inbred']}/{data['Tester']}"
                metadata.append(data)
            else:
                print(f"[WARNING] File name does not match pattern: {file}")

        if not metadata:
            raise ValueError(
                "No valid files found in the specified directory."
            )

        return pd.DataFrame(metadata)

    def read_file(
        self, filename: str, base_dir: str | None = None, **kwargs
    ) -> pd.DataFrame:
        """
        Reads a file specified in the metadata
        """
        # Construct the file path
        file_path = os.path.join(base_dir, filename) if base_dir else filename

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file format
        file_extension = os.path.splitext(file_path)[1].lower()

        # Supported file readers
        file_readers = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".json": pd.read_json,
        }

        if file_extension not in file_readers:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Read the file
        try:
            return file_readers[file_extension](file_path, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to read the file: {file_path}. Error: {e}"
            )

    def _map_to_lower_case(
        self,
        dataframe: pd.DataFrame,
        dataframe_columns: list[str],
        target_fields_to_convert: dict | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Converts DataFrame rows, columns, and values to lowercase
            if case-insensitive processing is enabled.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.
            dataframe_columns (list[str]): Expected column names to match.
            target_fields_to_convert (dict | None): Field conversion rules for columns/rows/values.

        Returns:
            tuple[pd.DataFrame, dict]:
                - Processed DataFrame with case conversions applied.
                - Dictionary mapping original column names to processed column names.
        """
        # Apply case conversions to DataFrame
        dataframe = apply_case_conversions(dataframe, target_fields_to_convert)

        # Match expected columns to actual columns
        column_match = form_string_matching(
            dataframe_columns, dataframe.columns.to_list(), self.case_sensitive
        )

        return dataframe, column_match

    def _validate_fn(
        self,
        data_df: pd.DataFrame,
        data_column_match: dict,
        metadata_df: pd.DataFrame,
        metadata_column_match: dict,
    ):
        """
        Validates the consistency of data_df against metadata_df.

        Args:
            data_df (pd.DataFrame): DataFrame containing the data.
            data_column_match (dict): Mapping of expected column names to actual column names in data_df.
            metadata_df (pd.DataFrame): Metadata DataFrame with expected values.
            metadata_column_match (dict): Mapping of expected column names to actual column names in metadata_df.

        Raises:
            ValueError: If any validation fails.
        """

        # Helper function: Validate a single column for uniqueness
        def _validate_single_value(column, column_name, filename):
            if column.nunique() == 0:
                raise ValueError(
                    f"No value found in {column_name} for {filename}."
                )
            elif column.nunique() > 1:
                raise ValueError(
                    f"Inconsistent values in {column_name} for {filename}."
                )
            return column.iloc[0]

        # Helper function: Validate a match between data and metadata
        def _validate_match(data_value, metadata_value, column_name, filename):
            if data_value != metadata_value:
                raise ValueError(
                    f"{column_name} mismatch between data ({data_value}) "
                    f"and metadata ({metadata_value}) in {filename}."
                )

        # Extract filename
        filename = (
            metadata_df[metadata_column_match["File"]]
            if "File" in metadata_column_match
            else "Unknown_File"
        )

        try:
            # Validate Yield.t.ha.BLUE
            yield_col_name = data_column_match["Yield.t.ha.BLUE"]
            yield_value = _validate_single_value(
                data_df[yield_col_name], "Yield.t.ha.BLUE", filename
            )

            # Validate Pedigree
            pedigree_col_name = data_column_match["Pedigree"]
            pedigree_value = _validate_single_value(
                data_df[pedigree_col_name], "Pedigree", filename
            )
            metadata_pedigree = metadata_df[
                metadata_column_match["Pedigree"]
            ].iloc[0]
            _validate_match(
                pedigree_value, metadata_pedigree, "Pedigree", filename
            )

            # Validate Year
            year_col_name = data_column_match["Year"]
            year_value = _validate_single_value(
                data_df[year_col_name], "Year", filename
            )
            metadata_year = metadata_df[metadata_column_match["Year"]].iloc[0]
            _validate_match(year_value, metadata_year, "Year", filename)

            # Validate Env
            env_col_name = data_column_match["Env"]
            env_value = _validate_single_value(
                data_df[env_col_name], "Env", filename
            )
            expected_env = f"{metadata_df[metadata_column_match['Env']].iloc[0]}.{metadata_year}"
            _validate_match(env_value, expected_env, "Env", filename)

            # Validate Vegetation.Index
            vegetation_index_col_name = data_column_match["Vegetation.Index"]
            vegetation_index_set_new = set(
                data_df[vegetation_index_col_name].unique()
            )
            vegetation_index_set = getattr(self, "vegetation_index_set", None)
            if (
                vegetation_index_set is not None
                and vegetation_index_set_new != vegetation_index_set
            ):
                raise ValueError(
                    f"Inconsistent Vegetation.Index in {filename}."
                    f"Expected: {vegetation_index_set}, "
                    f"Found: {vegetation_index_set_new}."
                )
            # Store the validated vegetation index set for subsequent validations
            self.vegetation_index_set = vegetation_index_set_new

        except KeyError as e:
            raise ValueError(f"Missing column: {e}")
        except ValueError as ve:
            raise
        except Exception as e:
            raise

    def _get_vi_columns(
        self, data_df: pd.DataFrame, data_column_match: dict
    ) -> list[str]:
        """
        Extracts vegetation index columns from the DataFrame.

        Args:
            data_df (pd.DataFrame): Input data DataFrame.
            data_column_match (dict): Column name mappings for data_df.

        Returns:
            list[str]: List of vegetation index columns.
        """
        vi_prefix = data_column_match.get("VI.BLUE.", "")
        vi_columns = [
            col for col in data_df.columns if col.startswith(vi_prefix)
        ]

        if not vi_columns:
            print(
                f"[WARNING] No vegetation index columns found with prefix '{vi_prefix}'."
            )

        return vi_columns

    def _handle_missing_values(
        self,
        data_df: pd.DataFrame,
        data_column_match: dict,
        metadata_df: pd.DataFrame,
        metadata_column_match: dict,
        vi_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handles missing values in vegetation index columns using interpolation,
        and returns a boolean mask marking the originally-missing cells.

        The mask is captured *before* interpolation so downstream consumers
        can distinguish real observations from synthetic interpolants. The
        VI FPCA processor uses this to optionally exclude interpolated
        cells from fdapace fits (the R reference's scheme).

        Args:
            data_df (pd.DataFrame): Input data DataFrame.
            data_column_match (dict): Column name mappings for data_df.
            metadata_df (pd.DataFrame): Metadata DataFrame.
            metadata_column_match (dict): Column name mappings for metadata_df.
            vi_columns (list[str]): List of vegetation index columns.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                ``(data_df, vi_nan_mask_df)`` where ``vi_nan_mask_df`` is a
                boolean DataFrame with the same shape as ``data_df[vi_columns]``
                — True wherever the original cell was NaN.

        Raises:
            ValueError: If missing values cannot be interpolated.
        """
        # Extract filename for error reporting
        filename = (
            metadata_df[metadata_column_match["File"]]
            if "File" in metadata_column_match
            else "Unknown_File"
        )

        # Snapshot the original NaN positions before interpolation overwrites them.
        vi_nan_mask_df = data_df[vi_columns].isnull().copy()

        # Check for missing values in vegetation index columns
        if vi_nan_mask_df.to_numpy().any():
            # logging.warning(f"Missing values detected in {filename}. Performing interpolation...")
            # Perform interpolation
            data_df = vi_nonuniform_interpolate(
                data_df,
                vi_columns,
                vi_label=data_column_match["Vegetation.Index"],
            )

            # Recheck for remaining missing values
            if data_df[vi_columns].isnull().any().any():
                raise ValueError(
                    f"Unable to interpolate all missing values in {filename}."
                )

            # logging.info(f"Missing values successfully handled for {filename}.")

        return data_df, vi_nan_mask_df

    def _extract_data(
        self,
        data_df: pd.DataFrame,
        data_column_match: dict,
        vegetation_index_columns: list[str],
        vi_nan_mask_df: pd.DataFrame,
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        list[str],
    ]:
        """
        Extracts days after planting, vegetation index values, the per-cell
        NaN mask, and yield.

        Args:
            data_df (pd.DataFrame): Input data DataFrame (post-interpolation).
            data_column_match (dict): Column name mappings for data_df.
            vegetation_index_columns (list[str]): List of vegetation index columns.
            vi_nan_mask_df (pd.DataFrame): Boolean DataFrame (same shape as
                ``data_df[vegetation_index_columns]``) marking the cells that
                were NaN before interpolation. Captured by
                :meth:`_handle_missing_values`.

        Returns:
            ``(dap, channels, vi_nan_mask, yield_value, vi_names)``
            where ``vi_nan_mask`` has shape ``[T, n_vis]`` and is True wherever
            the original cell was NaN, and ``vi_names`` is the canonical
            (sorted) list of VI names corresponding to each column of
            ``channels``.
        """
        # Extract yield value
        yield_value = data_df[data_column_match["Yield.t.ha.BLUE"]].iloc[0]

        # Sort rows by Vegetation.Index value so column j of the per-sample
        # tensor always corresponds to the j-th alphabetic VI name across
        # every file. The validator already enforces a consistent VI set
        # across files, so this gives a cross-file canonical ordering that
        # downstream `vi_subset` slicing can rely on.
        vi_col = data_column_match["Vegetation.Index"]
        sort_idx = data_df[vi_col].argsort(kind="mergesort").to_numpy()
        data_df = data_df.iloc[sort_idx].reset_index(drop=True)
        # Apply the same row permutation to the mask so the (vi, dap) cell
        # at row r still corresponds to row r of the value DataFrame.
        vi_nan_mask_df = vi_nan_mask_df.iloc[sort_idx].reset_index(drop=True)
        vi_names = data_df[vi_col].astype(str).tolist()

        # Sort DAP columns by numeric suffix
        sorted_vi_columns = sorted(
            vegetation_index_columns,
            key=lambda x: (
                int(x.split(".")[-1])
                if x.split(".")[-1].isdigit()
                else float("inf")
            ),
        )

        dap = []
        channels = []
        vi_nan_mask_cols = []
        for col in sorted_vi_columns:
            try:
                dap.append(int(col.split(".")[-1]))
                channels.append(data_df[col])
                vi_nan_mask_cols.append(vi_nan_mask_df[col])
            except ValueError:
                print(
                    f"[WARNING] Failed to parse time step from column: {col}"
                )

        dap = np.array(dap)[..., None].astype(
            np.float32
        )
        channels = np.stack(channels).astype(
            np.float32
        )
        vi_nan_mask = np.stack(vi_nan_mask_cols).astype(bool)
        yield_value = np.array([yield_value]).astype(np.float32)

        if self.to_tensor:
            dap = torch.from_numpy(dap)
            channels = torch.from_numpy(channels)
            vi_nan_mask = torch.from_numpy(vi_nan_mask)
            yield_value = torch.from_numpy(yield_value)

        return (
            dap,
            channels,
            vi_nan_mask,
            yield_value,
            vi_names,
        )

    def load(self):
        """
        Process all files listed in the metadata, validate their content and metadata,
        handle missing values, and extract relevant data.

        Returns:
            tuple: ``(data_dict, metadata_df)`` — ``data_dict`` maps each
                filename to a per-sample dict with ``dap``, ``channels``,
                ``channel_names``, ``vi_nan_mask``, and ``yield_value``;
                ``metadata_df`` is the aligned per-sample metadata frame.
        """
        # --- cache check ---
        cache_path = None
        if self.cache_dir is not None:
            cache_key = self._compute_cache_key()
            cache_path = Path(self.cache_dir) / f"g2f_data_{cache_key}.pt"
            if cache_path.exists():
                print(f"[INFO] Loading cached data from {cache_path}")
                cached = torch.load(cache_path, weights_only=False)
                # Restore canonical VI ordering captured at cache-write time.
                self.vi_names_ = list(cached["vi_names"])
                return cached["data_dict"], cached["metadata_df"]
            else:
                print(f"[INFO] No cache found. Processing data from scratch...")
        else:
            print("[INFO] Caching is disabled. Processing data from scratch...")

        # Parse and preprocess metadata
        metadata_df = self._parse_metadata()
        metadata_df, metadata_column_match = self._map_to_lower_case(
            metadata_df,
            self.metadata_df_columns,
            self.metadata_df_fields_to_convert,
        )

        invalid_indices = []
        data_dict = {}

        for row in tqdm(
            metadata_df.itertuples(index=True),
            total=len(metadata_df),
            desc="Reading and validating files",
        ):

            index = row.Index  # Extract index manually
            filename = getattr(
                row, metadata_column_match["File"]
            )  # Extract filename from metadata
            metadata_row_df = pd.DataFrame(
                [row._asdict()]
            )  # Convert named tuple to DataFrame

            try:
                file_path = os.path.join(self.data_dir, filename)

                # Read and preprocess data file
                data_df = self.read_file(filename, base_dir=self.data_dir)
                data_df, data_column_match = self._map_to_lower_case(
                    data_df,
                    self.data_df_columns,
                    self.data_df_fields_to_convert,
                )

                # Validate file content against metadata
                if self.validate:
                    self._validate_fn(
                        data_df=data_df,
                        data_column_match=data_column_match,
                        metadata_df=metadata_row_df,  # Convert Series to DataFrame
                        metadata_column_match=metadata_column_match,
                    )

                # Handle missing values (also captures pre-interpolation mask)
                vi_columns = self._get_vi_columns(data_df, data_column_match)
                data_df, vi_nan_mask_df = self._handle_missing_values(
                    data_df,
                    data_column_match,
                    metadata_row_df,
                    metadata_column_match,
                    vi_columns,
                )

                # Extract relevant data
                (
                    dap,
                    channels,
                    vi_nan_mask,
                    yield_value,
                    vi_names,
                ) = self._extract_data(
                    data_df, data_column_match, vi_columns, vi_nan_mask_df
                )

                # Capture canonical VI ordering on first valid file; the
                # _validate_fn already ensures the VI set is consistent
                # across files, so any subsequent inconsistency in order
                # is a bug we want to surface loudly.
                if self.vi_names_ is None:
                    self.vi_names_ = list(vi_names)
                elif vi_names != self.vi_names_:
                    raise ValueError(
                        f"Inconsistent VI ordering in {filename!r}: "
                        f"expected {self.vi_names_}, got {vi_names}"
                    )

                # Update the metadata-content dictionary. `vi_nan_mask`
                # records the originally-missing cells so the VI FPCA
                # processor can optionally exclude interpolated cells from
                # its fdapace fits (the R reference's scheme); the DL pipeline
                # ignores it and consumes the dense interpolated values.
                data_dict[filename] = {
                    "dap": dap,
                    "channels": channels,
                    "channel_names": list(vi_names),
                    "vi_nan_mask": vi_nan_mask,
                    "yield_value": yield_value,
                }
                # logging.info(f"Successfully processed file: {filename}")

            except FileNotFoundError:
                print(f"[ERROR] File not found: {filename}")
                invalid_indices.append(index)
            except ValueError as ve:
                print(f"[ERROR] Validation failed for file {filename}: {ve}")
                invalid_indices.append(index)
            except Exception as e:
                print(
                    f"[ERROR] Unexpected error while processing file {filename}: {e}"
                )
                invalid_indices.append(index)

            # Optionally delete invalid files
            if index in invalid_indices and self.delete_invalid:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # logging.info(f"Deleted invalid file: {file_path}")

        # Drop invalid rows from metadata
        if invalid_indices:
            metadata_df = metadata_df.drop(index=invalid_indices).reset_index(
                drop=True
            )
            print(
                f"[INFO] Dropped {len(invalid_indices)} invalid entries from metadata."
            )

        # --- cache write ---
        if cache_path is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            # Parallel-safe write: this cache key is split-independent, so
            # concurrent SLURM jobs (per-fold parallel submission) can race to
            # populate a cold cache. Write to a PID-tagged temp file then
            # atomically rename, so a concurrent reader never sees a
            # partially-written archive (torch.save uses a zip container; a
            # truncated file fails to load). Mirrors the processor-cache
            # pattern documented in processing/README.md.
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            torch.save(
                {
                    "data_dict": data_dict,
                    "metadata_df": metadata_df,
                    "vi_names": list(self.vi_names_ or []),
                },
                tmp_path,
            )
            try:
                os.replace(tmp_path, cache_path)
            except FileNotFoundError:
                pass
            print(f"[INFO] Cached processed data to {cache_path}")

        return data_dict, metadata_df
