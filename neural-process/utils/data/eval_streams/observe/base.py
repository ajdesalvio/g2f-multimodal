"""``ObserveSelector`` ABC — the observe family's contract.

An *observe* selector is a read-only selection over the *already-assigned*
rows. It carves nothing, reduces nothing, and is **structurally barred
from being the monitor** (the VALIDATE gate enforces this). Selectors read
the ``role`` / ``cv_label`` columns the dataset carries onto
``metadata_df`` (``dataset.py``, post-``assign_roles``) — the SAME
post-carve ``cv_label`` arrays the offline scorer uses
(``cv/scoring.score_by_label``): single source of truth, no re-derivation.

Taking ``meta_df`` (not the raw ``SchemeAssignment``) is deliberate — the
assignment predates the coverage reindex, so its positions would not align
with ``split_indices``; ``metadata_df`` is in the coverage-filtered,
RangeIndex position space the dataset commits to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class ObserveSelector(ABC):
    """Select positions for one read-only (observe) eval stream."""

    #: Registry key, declared on each concrete subclass.
    name: ClassVar[str]
    #: Scheme names this selector is valid under; ``None`` ⇒ any scheme.
    #: The VALIDATE gate rejects an incompatible (selector, scheme) pair.
    compatible_schemes: ClassVar[frozenset[str] | None] = None

    @abstractmethod
    def select(self, meta_df: "pd.DataFrame") -> "np.ndarray":
        """Return positions for this stream in the coverage-filtered,
        RangeIndex ``metadata_df`` (aligned with ``split_indices``)."""
        ...

    @classmethod
    def is_compatible_with(cls, scheme_name: str) -> bool:
        """True if this selector may run under ``scheme_name``."""
        return (
            cls.compatible_schemes is None
            or scheme_name in cls.compatible_schemes
        )
