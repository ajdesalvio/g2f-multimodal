"""Unified CV-scheme abstraction (CV1 / CV2 / CV0 / CV00 + env_year_loo).

Ports the R reference's four-quadrant G×E cross-validation taxonomy
into a single ``CVScheme`` abstraction that emits a per-row
``(role, cv_label)`` table — the single source of truth for masking
(role) and scoring (cv_label), mirroring R's ``mask_yields`` +
``evaluate_metrics`` in one place.
"""

from __future__ import annotations

from .base import CVScheme, Role, RowAssignment, SchemeAssignment
from .common_females import (
    common_female_set,
    env_universe,
    sorted_common_females,
)
from .cv_0_00 import CV0_00Scheme
from .cv_2_1 import CV2_1Scheme
from .env_year_loo import EnvYearLOOScheme
from .fold_source import (
    FoldValidationError,
    NativeFoldSource,
    RCsvFoldSource,
    build_fold_source,
    random_row_fold_array,
)
from .random_kfold import RandomKFoldScheme
from .registry import available_schemes, get_scheme
from .resolve import resolve_cv_assignment

__all__ = [
    "CVScheme",
    "Role",
    "RowAssignment",
    "SchemeAssignment",
    "common_female_set",
    "sorted_common_females",
    "env_universe",
    "CV0_00Scheme",
    "CV2_1Scheme",
    "EnvYearLOOScheme",
    "RandomKFoldScheme",
    "FoldValidationError",
    "NativeFoldSource",
    "RCsvFoldSource",
    "build_fold_source",
    "random_row_fold_array",
    "available_schemes",
    "get_scheme",
    "resolve_cv_assignment",
]
