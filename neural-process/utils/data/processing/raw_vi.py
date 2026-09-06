"""Raw vegetation-index subsetting for the DL set encoder.

``RawVISubsetProcessor`` slices the per-sample ``channels``
tensor down to a chosen subset of vegetation indices (e.g. NGRDI only),
in place. This is the DL analog of ``VIFPCAProcessor.vi_subset`` — but
where the FPCA path slices *cached FPC scores* post-load, this processor
slices the *raw VI curves* that feed the VI set encoder.

It is intentionally a separate, opt-in processor so the FPCA pipeline's
"fit FPCA on every VI, cache the full score matrix, slice post-load"
behaviour stays byte-identical: ``raw_vi`` is never enabled in FPCA
experiments, and ``vi_fpca`` is never enabled when ``raw_vi`` is — the
two never co-run.

Stateless and in-place (mirrors ``WeatherConcatProcessor``): no fitted
state to persist, ``transform`` mutates sample dicts and returns ``None``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import ClassVar

import numpy as np

from .base import BaseProcessor, OrchestratorContext
from .registry import register_processor

logger = logging.getLogger(__name__)


@register_processor
class RawVISubsetProcessor(BaseProcessor):
    """Slice ``channels`` to a VI subset, in place.

    Parameters
    ----------
    vi_subset : list[str] | None
        Vegetation indices to keep, by name. ``None`` (default) is a
        pass-through no-op — every VI column is retained in canonical
        order. The names are resolved against ``ctx.vi_names`` (the
        canonical per-column ordering captured by the reader), so the
        slice picks exactly the requested columns regardless of input
        order.
    """

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "raw_vi"
    # Must run before weather_concat (priority 9), which appends a weather
    # tail to the same tensor — slice native VI columns first. After all
    # derived-feature processors, which never read channels.
    priority: ClassVar[int] = 8
    # Mutates the existing vi_y tensor in place — no new batch field.
    produces_batch_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(self, vi_subset: list[str] | None = None):
        self.vi_subset = list(vi_subset) if vi_subset is not None else None
        # Resolved lazily (first fit/transform) from the full VI ordering,
        # then cached so every split slices the same columns and
        # dataset.vi_names is mutated exactly once.
        self._subset_indices: list[int] | None = None
        self._subset_names: list[str] | None = None

    @classmethod
    def from_config(cls, config: dict) -> "RawVISubsetProcessor":
        """Hydra-config factory used by the orchestrator."""
        return cls(vi_subset=config.get("vi_subset"))

    # ── Subset resolution ───────────────────────────────────────────

    def _ensure_resolved(self, ctx: OrchestratorContext) -> None:
        """Resolve the subset indices once, against the *full* VI ordering.

        Captures ``ctx.vi_names()`` before mutating ``dataset.vi_names`` so
        a later split's resolution can't see the already-sliced list.
        No-op when ``vi_subset`` is ``None`` (pass-through).
        """
        if self.vi_subset is None or self._subset_indices is not None:
            return

        full_names = list(ctx.vi_names())
        idx_by_name = {n: i for i, n in enumerate(full_names)}
        unknown = [v for v in self.vi_subset if v not in idx_by_name]
        if unknown:
            raise ValueError(
                f"raw_vi vi_subset names not found in dataset: "
                f"{sorted(unknown)}. Available: {full_names}"
            )
        # Preserve canonical (full-list) ordering of the kept columns.
        keep = set(self.vi_subset)
        self._subset_indices = [
            i for i, n in enumerate(full_names) if n in keep
        ]
        self._subset_names = [full_names[i] for i in self._subset_indices]
        # Keep dataset bookkeeping in sync with the sliced tensor. Only
        # vi_fpca reads dataset.vi_names, and it never co-runs with raw_vi.
        ctx.dataset.vi_names = list(self._subset_names)

    # ── BaseProcessor lifecycle ─────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Stateless — just resolve the subset indices."""
        self._ensure_resolved(ctx)

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """In-place: slice each sample's ``channels`` columns.

        Returns ``None`` (in-place signal — the orchestrator skips the
        derived-feature attach step, same as weather_concat/raw_weather).
        """
        if self.vi_subset is None:
            return None  # pass-through no-op

        self._ensure_resolved(ctx)
        idx = self._subset_indices
        for sample in ctx.split_data(split):
            sample["channels"] = sample["channels"][:, idx]
            # Keep channel_names consistent with the sliced channels.
            if "channel_names" in sample:
                sample["channel_names"] = list(self._subset_names)
        return None

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Content key over the requested subset (stateless processor)."""
        h = hashlib.sha256()
        h.update(b"raw_vi_v1|subset=")
        h.update(json.dumps(self.vi_subset, sort_keys=True).encode())
        return h.hexdigest()
