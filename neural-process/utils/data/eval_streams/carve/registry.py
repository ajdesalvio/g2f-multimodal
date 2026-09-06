"""Carve-strategy registry — ``@register_val_strategy`` + ``get_val_strategy``.

Mirrors the processor registry's decorator idiom
(``utils/data/processing/registry.py``). Parameterized like
``@register_metric`` (the ABC carries no pre-set ``name``); the decorator
sets ``cls.name``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .base import ValidationStrategy


_VAL_REGISTRY: dict[str, type["ValidationStrategy"]] = {}


def register_val_strategy(name: str):
    """Class decorator registering a ``ValidationStrategy`` under ``name``.

    Raises ``ValueError`` on a duplicate name (catches accidental
    re-registration on module reload).
    """

    def _decorate(
        cls: type["ValidationStrategy"],
    ) -> type["ValidationStrategy"]:
        if name in _VAL_REGISTRY:
            existing = _VAL_REGISTRY[name]
            raise ValueError(
                f"Validation strategy {name!r} already registered to "
                f"{existing.__module__}.{existing.__name__}; cannot "
                f"re-register {cls.__module__}.{cls.__name__}."
            )
        cls.name = name
        _VAL_REGISTRY[name] = cls
        return cls

    return _decorate


def get_val_strategy_class(name: str) -> type["ValidationStrategy"]:
    """Look up a registered strategy class by name (``KeyError`` on miss)."""
    if name not in _VAL_REGISTRY:
        raise KeyError(
            f"No validation strategy registered under {name!r}. "
            f"Available: {sorted(_VAL_REGISTRY)}."
        )
    return _VAL_REGISTRY[name]


def val_strategy_names() -> list[str]:
    """All registered strategy names, alphabetically sorted."""
    return sorted(_VAL_REGISTRY)


def get_val_strategy(
    cfg: Mapping[str, Any],
    *,
    fold_map: Mapping[str, int] | None = None,
) -> "ValidationStrategy":
    """Build a strategy from a config mapping (e.g. ``{name, frac}``).

    Pops ``name``, looks up the class, and constructs it with the
    remaining keys as kwargs. For a ``requires_fold`` strategy the run's
    ``fold_map`` is injected (unless the config already carries one).
    Raises ``ValueError`` if ``name`` is absent and ``KeyError`` (via
    ``get_val_strategy_class``) if it is unregistered.
    """
    params = {k: v for k, v in dict(cfg).items() if k != "name"}
    name = dict(cfg).get("name")
    if not name:
        raise ValueError(
            f"Validation strategy config is missing 'name': {dict(cfg)!r}."
        )
    cls = get_val_strategy_class(name)
    if cls.requires_fold and "fold_map" not in params:
        params["fold_map"] = fold_map
    return cls(**params)
