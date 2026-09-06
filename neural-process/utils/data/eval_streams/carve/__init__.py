"""Carve validation strategies + their registry.

Importing this package registers every built-in strategy (``random``,
``grouped_variety``, ``holdout_env``, ``fold``) via the decorator
side-effect, so ``get_val_strategy`` / ``val_strategy_names`` see them.
"""

from __future__ import annotations

from .base import ValidationStrategy
from .registry import (
    get_val_strategy,
    get_val_strategy_class,
    register_val_strategy,
    val_strategy_names,
)

# Side-effect imports: populate the registry. Order is irrelevant.
from . import fold, grouped_variety, holdout_env, random  # noqa: E402,F401

__all__ = [
    "ValidationStrategy",
    "get_val_strategy",
    "get_val_strategy_class",
    "register_val_strategy",
    "val_strategy_names",
]
