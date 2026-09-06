"""Gaussian predictive distribution."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributions import Normal

from .base import Prediction


class GaussianPrediction(Prediction):
    """A ``Normal`` predictive distribution.

    Wraps a :class:`torch.distributions.Normal`; :attr:`mean` and
    :meth:`log_prob` delegate to it. :meth:`cat` rebuilds a ``Normal`` from the
    stacked ``(loc, scale)``, so pooled NLL over a whole eval split is the exact
    mean log density — batch-size invariant, exactly like the pooled RMSE /
    correlation already are.
    """

    is_distributional = True

    def __init__(self, distribution: Normal) -> None:
        self.distribution = distribution

    @property
    def mean(self) -> torch.Tensor:
        return self.distribution.mean

    def log_prob(self, targets: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(targets)

    def flatten(self) -> "GaussianPrediction":
        return GaussianPrediction(
            Normal(
                self.distribution.loc.reshape(-1),
                self.distribution.scale.reshape(-1),
            )
        )

    def detach_cpu(self) -> "GaussianPrediction":
        return GaussianPrediction(
            Normal(
                self.distribution.loc.detach().cpu(),
                self.distribution.scale.detach().cpu(),
            )
        )

    @classmethod
    def cat(cls, parts: Sequence[Prediction]) -> "GaussianPrediction":
        loc = torch.cat([p.distribution.loc for p in parts])  # type: ignore[attr-defined]
        scale = torch.cat([p.distribution.scale for p in parts])  # type: ignore[attr-defined]
        return cls(Normal(loc, scale))
