"""Evaluation-stream building blocks (DL eval-streams plan).

Two sibling families live here:

- ``carve`` — validation strategies that pull rows out of the training
  pool (the OOD / IID val carve). Monitor-eligible.
- ``observe`` — read-only selectors over the already-assigned rows
  (``test`` / ``cv0`` / ``cv1`` / …). Never the monitor.

Both mirror the processor-registry decorator idiom
(``utils/data/processing/registry.py``) and operate in the one position
space the dataset commits to: the coverage-filtered, RangeIndex
``metadata_df``.
"""

from __future__ import annotations

from .carve import (
    ValidationStrategy,
    get_val_strategy,
    register_val_strategy,
    val_strategy_names,
)
from .observe import (
    ObserveSelector,
    get_observe_selector,
    observe_selector_names,
    register_observe_selector,
)
from .spec import EvalStreamSpec, parse_eval_stream, parse_eval_streams

__all__ = [
    "EvalStreamSpec",
    "ObserveSelector",
    "ValidationStrategy",
    "get_observe_selector",
    "get_val_strategy",
    "observe_selector_names",
    "parse_eval_stream",
    "parse_eval_streams",
    "register_observe_selector",
    "register_val_strategy",
    "val_strategy_names",
]
