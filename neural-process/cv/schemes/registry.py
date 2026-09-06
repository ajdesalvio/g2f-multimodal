"""Scheme name → ``CVScheme`` lookup."""

from __future__ import annotations

from .base import CVScheme
from .cv_0_00 import CV0_00Scheme
from .cv_2_1 import CV2_1Scheme
from .env_year_loo import EnvYearLOOScheme
from .random_kfold import RandomKFoldScheme

_REGISTRY: dict[str, type[CVScheme]] = {
    cls.name: cls
    for cls in (EnvYearLOOScheme, CV2_1Scheme, CV0_00Scheme, RandomKFoldScheme)
}


def get_scheme(name: str) -> CVScheme:
    """Instantiate the scheme registered under ``name``."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f"Unknown CV scheme {name!r}; registered: {sorted(_REGISTRY)}."
        ) from None


def available_schemes() -> list[str]:
    return sorted(_REGISTRY)
