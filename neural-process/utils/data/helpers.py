"""Helper functions for DataFrame case conversion, string matching, and VI interpolation."""

import warnings
from typing import Any
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def vi_nonuniform_interpolate(
    df: pd.DataFrame, vi_columns: list[str], vi_label: Any
) -> pd.DataFrame:
    """
    Performs interpolation on VI.BLUE.k columns for rows with missing values,
    considering only the data from the same vegetation index.

    Args:
        df (pd.DataFrame): Input DataFrame with VI.BLUE.k columns.
        vi_columns (List[str]): List of VI.BLUE.k column names.
        vi_label:

    Returns:
        pd.DataFrame: DataFrame with interpolated VI.BLUE.k values.
    """
    interpolated_df = df.copy()

    # Identify rows with missing values in VI columns
    missing_rows_mask = interpolated_df[vi_columns].isnull().any(axis=1)
    rows_with_missing = interpolated_df[missing_rows_mask]

    # Extract time steps from column names
    try:
        time_steps = [int(col.split(".")[-1]) for col in vi_columns]
    except ValueError:
        raise ValueError(
            "Column names in vi_columns must contain time steps as the last element."
        )

    # Perform interpolation for rows with missing values
    for idx, row in rows_with_missing.iterrows():
        veg_index = row[vi_label]

        # Known time steps and values for this vegetation index
        valid_mask = row[vi_columns].notnull()
        if (
            not valid_mask.any()
        ):  # Raise an error if no valid data for interpolation
            raise ValueError(
                f"Row {idx} with vegetation index '{veg_index}' "
                f"has no valid data for interpolation."
            )
        valid_time_steps = np.array(time_steps)[valid_mask]
        valid_values = row[vi_columns][valid_mask].values

        # Interpolation function. Out-of-range cells (before the first / after
        # the last observed timepoint) are held constant at the nearest
        # observed endpoint rather than linearly extrapolated — a fabricated
        # linear trend beyond the observed range would be fed to the encoder as
        # if it were measured. `fill_value=(low, high)` with bounds_error=False
        # applies valid_values[0] below the range and valid_values[-1] above.
        interp_func = interp1d(
            valid_time_steps,
            valid_values,
            kind="linear",
            bounds_error=False,
            fill_value=(valid_values[0], valid_values[-1]),
        )

        # Interpolate missing time steps
        interpolated_values = interp_func(time_steps)
        interpolated_row = row[vi_columns].copy()
        interpolated_row[~valid_mask] = interpolated_values[~valid_mask]

        # Update the DataFrame with interpolated values
        interpolated_df.loc[idx, vi_columns] = interpolated_row

    return interpolated_df


def validate_inputs(df: pd.DataFrame, target: str) -> None:
    """
    Validate the inputs to the convert_df_to_lower function.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        target (str): The part of the DataFrame to transform; must be in
            {'cols', 'rows', 'vals'}.

    Raises:
        ValueError: If df is not a DataFrame or target is invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input 'df' must be a pandas DataFrame.")

    valid_targets = {"cols", "rows", "vals"}
    if target not in valid_targets:
        raise ValueError(
            f"Invalid target: '{target}'. Target must be one of {valid_targets}."
        )


def normalize_label_list(
    label_list: list | None = None, case_sensitive: bool = False
) -> list | None:
    """
    Normalize a list of labels by optionally converting strings to lowercase.

    Args:
        label_list (list | None): The list of labels to normalize.
        case_sensitive (bool): If False, convert all string labels to lowercase.

    Returns:
        list | None: The normalized list of labels, or None if
        the input was None or empty.
    """
    if not label_list:
        return None

    if not case_sensitive:
        return [
            lbl.lower() if isinstance(lbl, str) else lbl for lbl in label_list
        ]
    return label_list


def build_new_labels(
    labels: Iterable,
    include_labels: list | None = None,
    exclude_labels: list | None = None,
    case_sensitive: bool = False,
) -> list:
    """
    Build a new list of labels from the original labels by converting
    some or all of them to lowercase.

    The logic is as follows:
      - If include_labels is not None, only labels in include_labels are converted.
      - Else if exclude_labels is not None, all labels except those in exclude_labels are converted.
      - Otherwise, all labels are converted.
      - Labels that are not strings are never converted.

    Args:
        labels (Iterable): The original labels (e.g., df.columns or df.index).
        include_labels (list | None): Labels to be converted. Overrides exclude_labels.
        exclude_labels (list | None): Labels not to be converted.
        case_sensitive (bool): If True, the comparison with include/exclude is case-sensitive.

    Returns:
        list: A list of transformed labels.
    """
    new_labels = []

    for lbl in labels:
        # Convert to a comparable version if not case_sensitive
        lbl_comp = (
            lbl.lower() if not case_sensitive and isinstance(lbl, str) else lbl
        )

        # Decide if we should convert this label
        if include_labels is not None:
            should_convert = lbl_comp in include_labels
        elif exclude_labels is not None:
            should_convert = lbl_comp not in exclude_labels
        else:
            should_convert = True

        # Actually convert to lowercase if it's a string
        if should_convert and isinstance(lbl, str):
            new_labels.append(lbl.lower())
        else:
            new_labels.append(lbl)

    return new_labels


def map_values_to_lower_(
    df: pd.DataFrame,
    include_labels: list | None = None,
    exclude_labels: list | None = None,
    case_sensitive: bool = False,
) -> None:
    """
    In-place: convert string values in specified columns to lowercase.

    The logic is as follows:
      - Identify all columns with object or category dtype.
      - If include_labels is provided, only columns present in include_labels are converted.
      - Otherwise, if exclude_labels is provided, convert all columns except those.
      - Otherwise, all string-like columns are converted.

    Args:
        df (pd.DataFrame): The DataFrame to modify in-place.
        include_labels (list | None): Columns to convert. Overrides exclude_labels.
        exclude_labels (list | None): Columns not to convert.
        case_sensitive (bool): Whether to match column labels case-sensitively.

    Returns:
        None
    """
    # Identify string-like columns
    str_cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Build a mapping {lowercase_col_name: original_col_name} if not case_sensitive
    if not case_sensitive:
        col_map = {
            col.lower(): col for col in str_cat_cols if isinstance(col, str)
        }
    else:
        # If case-sensitive is True, keep a direct map
        col_map = {col: col for col in str_cat_cols}

    # Determine columns to convert
    if include_labels is not None:
        # Only convert columns that appear in col_map & in include_labels
        to_convert = [col_map[lbl] for lbl in include_labels if lbl in col_map]
    elif exclude_labels is not None:
        # Convert columns that are in col_map and NOT in exclude_labels
        # Note: for not case_sensitive, the key we check is already lowercased
        to_convert = [
            orig for lbl, orig in col_map.items() if lbl not in exclude_labels
        ]
    else:
        # Convert all string-like columns
        to_convert = list(str_cat_cols)

    if not to_convert:
        warnings.warn("No valid columns to convert.", UserWarning)

    # Convert each cell in the selected columns to lowercase if it's a string
    df[to_convert] = df[to_convert].map(
        lambda x: x.lower() if isinstance(x, str) else x
    )


def convert_df_to_lower(
    df: pd.DataFrame,
    target: str,
    include_labels: list | None = None,
    exclude_labels: list | None = None,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Converts parts of a DataFrame to lowercase. Depending on `target`,
    this conversion applies to:
      - 'cols': DataFrame columns labels
      - 'rows': DataFrame row labels
      - 'vals': String values in specified columns

    Args:
        df (pd.DataFrame): The DataFrame to transform.
        target (str): Which part of the DataFrame to convert. Must be one of
            {'cols', 'rows', 'vals'}.
        include_labels (list | None): Labels to include for conversion.
            Overrides exclude_labels.
        exclude_labels (list | None): Labels not to transform.
        case_sensitive (bool): Whether to consider upper/lower case when matching labels
            from include_labels/exclude_labels.

    Returns:
        pd.DataFrame: A new DataFrame with the requested conversions.
    """
    # 1. Validate the input
    validate_inputs(df, target)

    # 2. Warn if both include and exclude are present
    if include_labels and exclude_labels:
        warnings.warn(
            "Both include_labels and exclude_labels were provided. "
            "exclude_labels will be ignored.",
            UserWarning,
        )

    # 3. Make a copy of the DataFrame to avoid mutating the original
    df_copy = df.copy()

    # 4. Normalize label lists
    include_labels = normalize_label_list(include_labels, case_sensitive)
    exclude_labels = normalize_label_list(exclude_labels, case_sensitive)

    # 5. Perform the requested conversion
    if target == "cols":
        df_copy.columns = build_new_labels(
            df_copy.columns, include_labels, exclude_labels, case_sensitive
        )
    elif target == "rows":
        df_copy.index = build_new_labels(
            df_copy.index, include_labels, exclude_labels, case_sensitive
        )
    elif target == "vals":
        map_values_to_lower_(
            df_copy, include_labels, exclude_labels, case_sensitive
        )
    else:
        # Should never happen because of earlier validation
        raise RuntimeError(
            "Unexpected target value encountered. Check validate_inputs."
        )

    return df_copy


def apply_case_conversions(
    df: pd.DataFrame, target_fields: dict | None = None
) -> pd.DataFrame:
    """
    Applies case conversions to a DataFrame's columns, rows, or values based on specified rules.

    Args:
        df (pd.DataFrame): The DataFrame to transform.
        target_fields (dict | None): A dictionary where keys are targets ('cols', 'rows', 'vals') and
                                        values are dictionaries specifying transformation rules:
                                        - 'include_labels': List of labels to include.
                                        - 'exclude_labels': List of labels to exclude.
                                        - 'case_sensitive': Whether search for target_fields is case-sensitive (default: False).

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    if target_fields is None:
        return df

    for field in ["cols", "rows", "vals"]:
        mapping_info = target_fields.get(field)
        if mapping_info:
            df = convert_df_to_lower(
                df=df,
                target=field,
                include_labels=mapping_info.get("include_labels"),
                exclude_labels=mapping_info.get("exclude_labels"),
                case_sensitive=mapping_info.get("case_sensitive", False),
            )

    return df


def find_matching_token(
    query: Any, target_list: list[Any], case_sensitive: bool = False
) -> tuple[int | None, Any | None]:
    """
    Finds the first matching element for a query in the target list based on the specified morpheme.

    Args:
        query (Any): The query to match. Can be a string or a tuple/list in the format (token, morpheme).
        target_list (list[Any]): The list of target elements to search in.
        case_sensitive (bool): Use case-sensitive comparison.

    Returns:
        Any | None: The query token.
        Any | None: The matched token for query in target_list or None if no match is found.

    Raises:
        ValueError: If the query format is invalid or if an unsupported morpheme is provided.
    """
    if isinstance(query, (tuple, list)):
        if len(query) != 2:
            raise ValueError(
                "Query must be a tuple or list with exactly two elements: (token, morpheme)."
            )
        query_token, query_morpheme = query
        if not isinstance(query_token, str) or not isinstance(
            query_morpheme, str
        ):
            raise ValueError(
                f"Both query_token and query_morpheme must be strings. "
                f"Got query_token type: {type(query_token)}, "
                f"query_morpheme type: {type(query_morpheme)}."
            )
    else:
        query_token, query_morpheme = query, "root"

    if not isinstance(query_token, str):
        return (
            query_token,
            query_token,
        )  # Return the query as-is if it's not a string

    token_to_match = query_token.lower() if not case_sensitive else query_token

    for target in target_list:
        if isinstance(target, str):
            target_to_match = target.lower() if not case_sensitive else target
            if (
                query_morpheme.lower() == "root"
                and token_to_match == target_to_match
            ):
                return query_token, target
            elif (
                query_morpheme.lower() == "prefix"
                and target_to_match.startswith(token_to_match)
            ):
                return query_token, target[: len(token_to_match)]
            elif (
                query_morpheme.lower() == "suffix"
                and target_to_match.endswith(token_to_match)
            ):
                return query_token, target[-len(token_to_match) :]

    return query_token, None  # No match found


def form_string_matching(
    query_list: list[Any], target_list: list[Any], case_sensitive: bool = False
) -> dict[Any, Any]:
    """
    Maps queries to their corresponding matches in the target list.

    Args:
        query_list (list[Any]): The list of queries to process.
        target_list (list[Any]): The list of target elements to search in.
        case_sensitive (bool): Case-sensitive comparison.

    Returns:
        dict: A dictionary mapping each query to its matched target token
            (or ``None`` when unmatched).

    Warnings:
        Issues a warning for each query that does not have a corresponding match or is invalid.
    """
    query_to_target_map = {}

    for query in query_list:
        try:
            query_token, matched_token = find_matching_token(
                query, target_list, case_sensitive
            )
            if matched_token is None:
                warnings.warn(
                    f"No match found for query: {query}. "
                    f"Ensure the query and target list are correct.",
                    UserWarning,
                )
                query_to_target_map[query_token] = (
                    None  # Explicitly map unmatched queries to None
                )
            else:
                query_to_target_map[query_token] = matched_token
        except ValueError as error:
            warnings.warn(
                f"Failed to process query {query}: {error}", UserWarning
            )
            query_to_target_map[query_token] = (
                None  # Explicitly map invalid queries to None
            )

    return query_to_target_map
