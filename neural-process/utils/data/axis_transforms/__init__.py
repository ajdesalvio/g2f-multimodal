"""Axis-transform package.

Importing this package registers every built-in transform (identity / dedup /
warp) via the ``@register_axis_transform`` decorator side-effects, mirroring
the ``processing`` package's import-to-register pattern. View assembly resolves
transforms by name through :func:`get_transform`.
"""

from __future__ import annotations

from .base import AxisTransform, ViewArrays
from .registry import (
    get_transform,
    get_transform_class,
    register_axis_transform,
    reset_registry,
    transform_names,
)

# Import side-effects: register the built-in transforms.
from .identity import IdentityTransform  # noqa: E402
from .dedup import DedupTransform  # noqa: E402
from .warp import WarpTransform  # noqa: E402

__all__ = [
    "AxisTransform",
    "ViewArrays",
    "get_transform",
    "get_transform_class",
    "register_axis_transform",
    "reset_registry",
    "transform_names",
    "IdentityTransform",
    "DedupTransform",
    "WarpTransform",
]
