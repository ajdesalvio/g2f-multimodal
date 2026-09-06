"""Processor registry — replaces hardcoded name-based dispatch.

Each processor module decorates its concrete `BaseProcessor` subclass
with `@register_processor`. Cross-cutting checks (the orchestrator,
`rank_check`, `setup._validate_config`,
`dataset._filter_to_modality_coverage`) query the registry by capability
flags rather than hardcoding processor names.

Adding a new processor in the future requires:

    @register_processor
    class MyProcessor(BaseProcessor):
        name = "my_proc"
        priority = 8
        ...

That single decoration replaces the previous tax of editing eight
separate locations across the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseProcessor


_REGISTRY: dict[str, type["BaseProcessor"]] = {}


# ── Registration ────────────────────────────────────────────────────


def register_processor(
    cls: type["BaseProcessor"],
) -> type["BaseProcessor"]:
    """Class decorator that registers a processor under `cls.name`.

    Validates that the class declares the two mandatory ClassVars
    (`name: str` and `priority: int`); raises `AttributeError` otherwise.
    Raises `ValueError` if the name is already registered (catches
    accidental duplicate decoration when modules are reloaded).
    """
    name = getattr(cls, "name", None)
    priority = getattr(cls, "priority", None)

    if not isinstance(name, str) or not name:
        raise AttributeError(
            f"{cls.__name__} must declare `name: ClassVar[str]` "
            f"with a non-empty value before @register_processor."
        )
    if not isinstance(priority, int):
        raise AttributeError(
            f"{cls.__name__} must declare `priority: ClassVar[int]` "
            f"before @register_processor."
        )
    if name in _REGISTRY:
        existing = _REGISTRY[name]
        raise ValueError(
            f"Processor name {name!r} already registered to "
            f"{existing.__module__}.{existing.__name__}; cannot "
            f"re-register {cls.__module__}.{cls.__name__}."
        )

    _REGISTRY[name] = cls
    return cls


# ── Lookup ──────────────────────────────────────────────────────────


def get_processor_class(name: str) -> type["BaseProcessor"]:
    """Look up a registered processor class by name.

    Raises `KeyError` with a helpful "available names" hint when the
    lookup misses.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"No processor registered under {name!r}. "
            f"Available: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name]


def all_processor_classes() -> list[type["BaseProcessor"]]:
    """Every registered processor class (no order guarantee)."""
    return list(_REGISTRY.values())


def processor_names() -> list[str]:
    """All registered processor names, alphabetically sorted."""
    return sorted(_REGISTRY)


# ── Capability queries ──────────────────────────────────────────────


def names_in_priority_order() -> list[str]:
    """Registered processor names sorted by `cls.priority` ascending.

    The orchestrator's main dispatch loop iterates this list.
    """
    return [
        c.name
        for c in sorted(_REGISTRY.values(), key=lambda c: c.priority)
    ]


def names_with_eigen_sources() -> list[str]:
    """Names of processors whose class declares `has_eigen_sources=True`.

    Used by the D1 rank-consistency check (`rank_check.py`) — replaces
    the hardcoded `_KERNEL_PROCESSORS_WITH_SOURCES` tuple.
    """
    return [c.name for c in _REGISTRY.values() if c.has_eigen_sources]


def batch_field_owners() -> dict[str, str]:
    """Map of batch-field name → owning processor name.

    Aggregated from `cls.produces_batch_fields` across the registry, so
    consumers can check that any batch field they read has its backing
    processor enabled.

    Raises `RuntimeError` if two processors claim the same field name —
    that's a configuration error in the processor classes themselves.
    """
    out: dict[str, str] = {}
    for cls in _REGISTRY.values():
        for field_name in cls.produces_batch_fields:
            if field_name in out:
                raise RuntimeError(
                    f"Batch field {field_name!r} claimed by both "
                    f"{out[field_name]!r} and {cls.name!r}; only one "
                    f"processor may produce a given batch field."
                )
            out[field_name] = cls.name
    return out


def view_owners() -> dict[str, str]:
    """Map of view name → owning processor name.

    Aggregated from `cls.produces_views` across the registry. Used by
    `G2FDataset._filter_views_to_enabled_processors` to check that any
    non-``"main"`` view has its backing processor enabled
    (e.g. the ``weather`` view requires ``raw_weather``).
    ``"main"`` (the VI view) is populated by the collate path, not a
    processor, so it never appears here.

    Raises `RuntimeError` if two processors claim the same view name.
    """
    out: dict[str, str] = {}
    for cls in _REGISTRY.values():
        for view_name in cls.produces_views:
            if view_name in out:
                raise RuntimeError(
                    f"View {view_name!r} claimed by both "
                    f"{out[view_name]!r} and {cls.name!r}; only one "
                    f"processor may produce a given view."
                )
            out[view_name] = cls.name
    return out


# ── Tooling ───────────────────────────────────────────────────────


def reset_registry() -> None:
    """Clear all registrations.

    Debug/tooling utility — production code MUST NOT call this. Anything
    registering temporary processors should snapshot and restore the
    registry around its use.
    """
    _REGISTRY.clear()
