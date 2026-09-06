"""``by_label`` — the scenario subset selectors (``cv0`` / ``cv00`` /
``cv1`` / ``cv2``).

Each selects the rows carrying one ``cv_label``, reading the SAME
post-carve ``cv_label`` column the offline scorer partitions on
(``cv/scoring.score_by_label``). The four are **explicit registrations**,
each scheme-tagged — *not* one generic "filter by any label" selector —
so the training loop never infers which subsets to track; the user
declares them and the gate validates scheme compatibility.

Note ``CV2`` selects the in-sample retained FIT rows (``cv_2_1.py``): a
``cv2`` observe stream evaluates on rows the model trains on. That is
intentional (it mirrors the offline in-sample CV2 metric) and is not
leakage into *selection* — observe streams cannot monitor.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .base import ObserveSelector
from .registry import register_observe_selector


class ByLabel(ObserveSelector):
    """Select rows carrying ``cls.label`` in the post-carve ``cv_label``
    column. A thin shared base (DRY); concrete subclasses below stay
    explicit registrations."""

    label: ClassVar[str]

    def select(self, meta_df) -> np.ndarray:
        return np.where(meta_df["cv_label"].to_numpy() == self.label)[0]


@register_observe_selector
class CV0Selector(ByLabel):
    name = "cv0"
    label = "CV0"
    compatible_schemes = frozenset({"cv_0_00"})


@register_observe_selector
class CV00Selector(ByLabel):
    name = "cv00"
    label = "CV00"
    compatible_schemes = frozenset({"cv_0_00"})


@register_observe_selector
class CV1Selector(ByLabel):
    name = "cv1"
    label = "CV1"
    compatible_schemes = frozenset({"cv_2_1"})


@register_observe_selector
class CV2Selector(ByLabel):
    name = "cv2"
    label = "CV2"
    compatible_schemes = frozenset({"cv_2_1"})
