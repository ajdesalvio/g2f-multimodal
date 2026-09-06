"""Observe-selector registry — ``@register_observe_selector`` + lookup.

Bare decorator reading ``cls.name`` (mirrors ``@register_processor``;
the four scenario subsets declare ``name`` as a ClassVar).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .base import ObserveSelector


_OBSERVE_REGISTRY: dict[str, type["ObserveSelector"]] = {}


def register_observe_selector(
    cls: type["ObserveSelector"],
) -> type["ObserveSelector"]:
    """Class decorator registering an ``ObserveSelector`` under ``cls.name``.

    Raises ``AttributeError`` if the class declares no non-empty ``name``,
    and ``ValueError`` on a duplicate name.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise AttributeError(
            f"{cls.__name__} must declare `name: ClassVar[str]` with a "
            f"non-empty value before @register_observe_selector."
        )
    if name in _OBSERVE_REGISTRY:
        existing = _OBSERVE_REGISTRY[name]
        raise ValueError(
            f"Observe selector {name!r} already registered to "
            f"{existing.__module__}.{existing.__name__}; cannot "
            f"re-register {cls.__module__}.{cls.__name__}."
        )
    _OBSERVE_REGISTRY[name] = cls
    return cls


def get_observe_selector_class(name: str) -> type["ObserveSelector"]:
    """Look up a registered selector class by name (``KeyError`` on miss)."""
    if name not in _OBSERVE_REGISTRY:
        raise KeyError(
            f"No observe selector registered under {name!r}. "
            f"Available: {sorted(_OBSERVE_REGISTRY)}."
        )
    return _OBSERVE_REGISTRY[name]


def observe_selector_names() -> list[str]:
    """All registered selector names, alphabetically sorted."""
    return sorted(_OBSERVE_REGISTRY)


def get_observe_selector(cfg: Mapping[str, Any]) -> "ObserveSelector":
    """Build a selector from a config mapping (e.g. ``{name: cv0}`` or
    ``{name: role, role: PREDICT}``). Pops ``name``, passes the rest as
    kwargs. ``ValueError`` if ``name`` is absent; ``KeyError`` if it is
    unregistered."""
    params = {k: v for k, v in dict(cfg).items() if k != "name"}
    name = dict(cfg).get("name")
    if not name:
        raise ValueError(
            f"Observe selector config is missing 'name': {dict(cfg)!r}."
        )
    cls = get_observe_selector_class(name)
    return cls(**params)
