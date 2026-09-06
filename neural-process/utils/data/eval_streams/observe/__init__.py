"""Observe selectors + their registry.

Importing this package registers every built-in selector (``role``,
``cv0`` / ``cv00`` / ``cv1`` / ``cv2``) via the decorator side-effect.
"""

from __future__ import annotations

from .base import ObserveSelector
from .registry import (
    get_observe_selector,
    get_observe_selector_class,
    observe_selector_names,
    register_observe_selector,
)

# Side-effect imports: populate the registry. Order is irrelevant.
from . import by_label, by_role  # noqa: E402,F401

__all__ = [
    "ObserveSelector",
    "get_observe_selector",
    "get_observe_selector_class",
    "observe_selector_names",
    "register_observe_selector",
]
