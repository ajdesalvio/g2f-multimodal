"""``EvalStreamSpec`` — the parsed shape of one ``dataset.eval_streams`` entry.

One config entry such as::

    - name: val
      source: carve
      strategy: { name: grouped_variety, n_groups: 20 }
      monitor: true
      monitor_metric: pearson_r
      monitor_mode: max
      metrics: [rmse, pearson_r, spearman_r]

becomes one :class:`EvalStreamSpec`. The spec is the single source of truth
threaded through three layers:

- the **dataset** reads ``source`` / ``strategy`` / ``selector`` to build the
  per-stream row list (the carve reduces train; observe selectors read the
  post-carve assignment);
- **run.py** reads ``name`` / ``metrics`` to build one resolved ``PhaseConfig``
  per stream (atop the shared ``eval`` template) and one val dataloader, in
  list order (the canonical ``dataloader_idx`` order);
- **setup.py** reads ``monitor`` / ``monitor_metric`` / ``monitor_mode`` to
  derive the early-stop / best-checkpoint key, and validates the whole list at
  the VALIDATE gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _to_plain(obj: Any) -> Any:
    """Best-effort convert an OmegaConf node to a plain dict/list.

    Accepts already-plain mappings/lists unchanged so the parser is usable
    from both the Hydra path (DictConfig) and plain dicts.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    return obj


@dataclass
class EvalStreamSpec:
    """Parsed, validated-shape representation of one eval stream entry."""

    name: str
    source: str  # "carve" | "observe"
    monitor: bool = False
    monitor_metric: str | None = None
    monitor_mode: str | None = None
    #: Metrics that each get an independent best (and, under freeze_on_plateau,
    #: best_frozen) checkpoint — written as ``best_<metric>.ckpt``. Only read on
    #: the monitor stream. ``None`` ⇒ default to ``[monitor_metric]`` (exactly
    #: the legacy single-best behaviour). Must be a subset of ``metrics``.
    checkpoint_metrics: list[str] | None = None
    metrics: list[str] = field(default_factory=list)
    #: carve strategy config (``{name, ...}``); None for observe streams.
    strategy: dict | None = None
    #: observe selector config (``{name, ...}``); None for carve streams.
    selector: dict | None = None
    #: When True, this stream's rows are DELIBERATELY in the eval context (a
    #: reconstruction / in-sample probe, e.g. ``cv2``), so the
    #: ``context ∩ query = ∅`` guard is skipped for it. Default False: the
    #: stream's metric is a held-out / out-of-sample estimate and MUST be
    #: disjoint from the context, or it silently degrades to reconstruction.
    in_sample: bool = False

    @property
    def is_carve(self) -> bool:
        return self.source == "carve"

    @property
    def is_observe(self) -> bool:
        return self.source == "observe"


def parse_eval_stream(entry: Mapping[str, Any]) -> EvalStreamSpec:
    """Build one :class:`EvalStreamSpec` from a config mapping.

    Shape-only parse: it raises on a missing ``name`` / ``source`` and on an
    unknown ``source``, but the richer stream↔scheme / metric-registry checks
    live in the VALIDATE gate (setup.py). ``strategy`` / ``selector`` are kept
    as plain dicts to hand to the carve / observe registries verbatim.
    """
    entry = _to_plain(entry) or {}
    name = entry.get("name")
    if not name:
        raise ValueError(f"eval stream entry is missing 'name': {dict(entry)!r}.")
    source = entry.get("source")
    if source not in ("carve", "observe"):
        raise ValueError(
            f"eval stream {name!r}: 'source' must be 'carve' or 'observe', "
            f"got {source!r}."
        )
    metrics = list(entry.get("metrics") or [])
    return EvalStreamSpec(
        name=str(name),
        source=str(source),
        monitor=bool(entry.get("monitor", False)),
        monitor_metric=entry.get("monitor_metric"),
        monitor_mode=entry.get("monitor_mode"),
        checkpoint_metrics=(
            [str(m) for m in entry["checkpoint_metrics"]]
            if entry.get("checkpoint_metrics") is not None
            else None
        ),
        metrics=[str(m) for m in metrics],
        strategy=(
            dict(entry["strategy"]) if entry.get("strategy") is not None else None
        ),
        selector=(
            dict(entry["selector"]) if entry.get("selector") is not None else None
        ),
        in_sample=bool(entry.get("in_sample", False)),
    )


def parse_eval_streams(cfg: Any) -> list[EvalStreamSpec]:
    """Parse a ``dataset.eval_streams`` list into ordered specs (``[]`` if None)."""
    cfg = _to_plain(cfg)
    if not cfg:
        return []
    return [parse_eval_stream(entry) for entry in cfg]
