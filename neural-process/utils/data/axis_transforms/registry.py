"""Axis-transform registry — mirrors ``utils/data/processing/registry.py``.

Each transform module decorates its concrete :class:`AxisTransform` subclass
with ``@register_axis_transform``. View assembly looks transforms up by name
(``get_transform``) and instantiates them with the view's ``transform_args``.

Adding a new transform requires only:

    @register_axis_transform
    class MyTransform(AxisTransform):
        name = "my_transform"
        ...

The decoration is the single registration point — no dispatch table to edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AxisTransform


_REGISTRY: dict[str, type["AxisTransform"]] = {}


# ── Registration ────────────────────────────────────────────────────


def register_axis_transform(
    cls: type["AxisTransform"],
) -> type["AxisTransform"]:
    """Class decorator that registers a transform under ``cls.name``.

    Validates that the class declares a non-empty ``name: ClassVar[str]``;
    raises ``AttributeError`` otherwise and ``ValueError`` on a duplicate
    name (catches accidental double-registration on module reload).
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise AttributeError(
            f"{cls.__name__} must declare `name: ClassVar[str]` with a "
            f"non-empty value before @register_axis_transform."
        )
    if name in _REGISTRY:
        existing = _REGISTRY[name]
        raise ValueError(
            f"Axis transform {name!r} already registered to "
            f"{existing.__module__}.{existing.__name__}; cannot re-register "
            f"{cls.__module__}.{cls.__name__}."
        )
    _REGISTRY[name] = cls
    return cls


# ── Lookup ──────────────────────────────────────────────────────────


def get_transform_class(name: str) -> type["AxisTransform"]:
    """Look up a registered transform class by name (raises ``KeyError``)."""
    if name not in _REGISTRY:
        raise KeyError(
            f"No axis transform registered under {name!r}. "
            f"Available: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name]


def get_transform(name: str, **transform_args) -> "AxisTransform":
    """Instantiate the registered transform ``name`` with ``transform_args``.

    The one-call convenience the view-assembly path uses:
    ``get_transform(view.transform, **view.transform_args).apply(arrays)``.
    """
    return get_transform_class(name)(**transform_args)


def transform_names() -> list[str]:
    """All registered transform names, alphabetically sorted."""
    return sorted(_REGISTRY)


# ── Tooling ───────────────────────────────────────────────────────


def reset_registry() -> None:
    """Clear all registrations (debug/tooling only; production must not call this)."""
    _REGISTRY.clear()
