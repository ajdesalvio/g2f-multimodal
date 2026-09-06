"""D1 rank-consistency check shared by setup.py and baselines/fpca_core.py.

Implements the two-knob (``n_components`` / ``n_components_max``)
contract and the check 5 mutual-exclusion / rank semantics.

This module lives under ``utils/data/processing`` (not
``utils/experiment``) because it is consumed by both the DL path
(``utils.experiment.setup._validate_config``) and the FPCA path
(``baselines.fpca_core.train``), and the FPCA path must not import
from ``utils.experiment`` (circular).
"""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def _kernel_processors_with_sources() -> tuple[str, ...]:
    """Names of registered processors with eigen sources.

    Replaces the previously-hardcoded
    ``_KERNEL_PROCESSORS_WITH_SOURCES = ("genomic", "phenomic", "enviromic")``
    tuple. Any new processor that declares
    ``has_eigen_sources: ClassVar[bool] = True`` is automatically picked
    up by the D1 rank-consistency check without editing this module.

    The query is wrapped in a function (rather than a module-level call)
    so that processor modules importing `rank_check` don't trigger the
    registry lookup at import time — registration order matters less
    that way.
    """
    from .registry import names_with_eigen_sources

    return tuple(names_with_eigen_sources())


def check_rank_consistency(
    config: DictConfig | dict,
    processor: Any,
) -> None:
    """Enforce the D1 two-knob contract on every kernel eigen source.

    Two sub-checks:

    5a. **Mutual exclusion** (fold-independent). Any source declaring
        both ``n_components`` and ``n_components_max`` raises
        ``ValueError`` — config shape error, not data-dependent.

    5b. **Hard-fix vs effective rank** (fold-named). Any source with
        ``n_components: K`` compared against
        ``processor.feature_dims[name]``. If ``K > effective_rank``,
        raises ``ValueError`` with the fold id from
        ``cfg.misc.fold_env``, source name, K, and rank.

    The soft-ceiling path (``n_components_max``) is intentionally NOT
    checked here — clamping is legal, and
    ``feature_dims`` already reports the clamped width.

    Sub-check 5a reads only the cfg and never touches
    ``processor.feature_dims``, so it fires cleanly even if a later
    feature_dims access would raise. Sub-check 5b is skipped when
    ``processor`` is ``None`` or has no ``feature_dims`` — the
    submit-side pre-flight uses this to run mutual-exclusion without
    a fit processor.

    ``config`` may be a ``DictConfig`` or a plain dict (after
    ``OmegaConf.to_container`` / D3's ``_convert_="all"``). Plain
    dicts are wrapped for uniform ``OmegaConf.select`` access.
    """
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)

    fold_id = (
        OmegaConf.select(config, "misc.fold_env", default="") or "<unknown>"
    )

    # Collect (block_name, source_dict) pairs across every registered
    # kernel processor (has_eigen_sources=True). Each source dict has
    # name / type / n_components / n_components_max fields.
    source_pairs: list[tuple[str, Any]] = []
    for block in _kernel_processors_with_sources():
        sources = OmegaConf.select(
            config, f"dataset.processing.{block}.sources", default=None,
        )
        if not sources:
            continue
        for src in sources:
            source_pairs.append((block, src))

    # 5a. Mutual exclusion — config-only, no feature_dims access.
    for block, src in source_pairs:
        name = _get(src, "name", "<anonymous>")
        k = _get(src, "n_components", None)
        k_max = _get(src, "n_components_max", None)
        if k is not None and k_max is not None:
            raise ValueError(
                f"dataset.processing.{block}.sources source={name!r}: "
                f"n_components and n_components_max are mutually "
                f"exclusive; set at most one."
            )

    # 5b. Hard-fix vs effective rank. Only reads feature_dims for
    # sources that actually declare n_components — kmax sources and
    # default-full sources can legitimately fail the rank check
    # (kmax clamps, default-full always fits), so we skip them.
    feature_dims = getattr(processor, "feature_dims", None)
    if feature_dims is None:
        return
    for block, src in source_pairs:
        k = _get(src, "n_components", None)
        if k is None:
            continue
        name = _get(src, "name", "<anonymous>")
        effective = feature_dims.get(name)
        if effective is None:
            continue  # source disabled / not exposed by the processor
        if k > effective:
            raise ValueError(
                f"fold={fold_id} "
                f"dataset.processing.{block}.sources source={name!r}: "
                f"n_components={k} (hard fix) exceeds effective rank "
                f"{effective}. Use n_components_max={k} for adaptive "
                f"clamping, or reduce n_components to <= {effective}."
            )


def _get(src: Any, key: str, default: Any) -> Any:
    """Uniform getter over DictConfig / dict source entries."""
    if hasattr(src, "get"):
        return src.get(key, default)
    return default
