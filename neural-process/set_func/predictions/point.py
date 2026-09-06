"""Deterministic point-estimate prediction (no predictive density)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import Prediction


class PointPrediction(Prediction):
    """A deterministic point estimate — a predicted value with no density.

    :attr:`mean` returns the predicted value; :meth:`log_prob` raises (inherited
    from :class:`Prediction`), so a likelihood metric requested on a point
    estimator fails loudly. This is the clean replacement for the old idiom of
    wrapping a point prediction in a fixed-variance Gaussian: the model's intent
    (point estimate, not a calibrated distribution) is now explicit in the type.
    """

    is_distributional = False

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    @property
    def mean(self) -> torch.Tensor:
        return self.value

    def flatten(self) -> "PointPrediction":
        return PointPrediction(self.value.reshape(-1))

    def detach_cpu(self) -> "PointPrediction":
        return PointPrediction(self.value.detach().cpu())

    @classmethod
    def cat(cls, parts: Sequence[Prediction]) -> "PointPrediction":
        return cls(torch.cat([p.value for p in parts]))  # type: ignore[attr-defined]
