"""Canonical, normalized identifiers for the CV-scheme layer.

Every identifier the scheme layer compares or joins on — ``Female``,
``Env.Year``, ``Pedigree.Env`` — is lowercased here so strings sourced
from the R reference's ``female_folds.csv`` (mixed case) join correctly against the
dataset's ``metadata_df`` regardless of the case the reader happened to
preserve. The dataset reader matches *columns* case-insensitively under
``case_sensitive=False`` but does not lowercase *values* unless
``metadata_df_fields_to_convert`` is configured, so the scheme layer owns
value normalization to keep the fold join provably case-safe (D5, plan
``identifiers.py`` bullet).

Mirrors the R reference:
- ``female_of`` ⇔ ``strsplit(Pedigree, "/")[[1]][1]`` (DAP_CV_Metadata_V1.R).
- ``env_year`` ⇔ R's ``Env`` column, which encodes location×year; here it
  is rebuilt from the dataset's separate ``Env`` + ``Year`` columns,
  matching ``OrchestratorContext.env_year`` (utils/data/processing/base.py).
- ``pedigree_env`` ⇔ R's ``Pedigree.Env`` (``paste(Pedigree, Env, ".")``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def normalize_id(value: object) -> str:
    """Canonical form for any scheme-level identifier: stripped + lowercased."""
    return str(value).strip().lower()


def female_of(pedigree: object) -> str:
    """Maternal-line key = the pedigree substring before the first ``/``.

    ``B14A/H95`` → ``b14a``. A pedigree with no ``/`` maps to itself
    (lowercased), matching R's ``strsplit`` returning the whole string.
    """
    return normalize_id(str(pedigree).split("/")[0])


def env_year_series(meta_df: "pd.DataFrame") -> "pd.Series":
    """Normalized ``Env.Year`` identifier series, indexed like ``meta_df``.

    Equivalent to ``OrchestratorContext.env_year`` but lowercased for the
    scheme-layer join. This is the *environment* key the common-female
    determination and ``cv_0_00`` held-out-env match run on.
    """
    return (
        meta_df["Env"].map(normalize_id)
        + "."
        + meta_df["Year"].map(normalize_id)
    )


def female_series(meta_df: "pd.DataFrame") -> "pd.Series":
    """Normalized ``Female`` (maternal-line) series, indexed like ``meta_df``."""
    return meta_df["Pedigree"].map(female_of)


def pedigree_env_series(meta_df: "pd.DataFrame") -> "pd.Series":
    """Normalized ``Pedigree.Env`` series (R's row key), indexed like ``meta_df``."""
    return meta_df["Pedigree"].map(normalize_id) + "." + env_year_series(meta_df)
