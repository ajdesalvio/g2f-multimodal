"""The model output-head contract.

A *head* maps a decoder's raw output tensor to a :class:`Prediction`. Two
families implement it:

- **distributional heads** — the likelihoods in :mod:`set_func.likelihoods`
  (e.g. :class:`~set_func.likelihoods.HeteroscedasticNormalLikelihood`), which
  return a distributional :class:`Prediction` carrying a density; and
- **point heads** — :class:`~set_func.heads.PointHead`, which returns a
  :class:`~set_func.predictions.PointPrediction` (a point estimate, no density).

The model is agnostic to which it holds: it calls ``head(decoder_output)`` and
returns the resulting :class:`Prediction`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from set_func.predictions import Prediction


class BaseHead(nn.Module, ABC):
    """Maps a decoder output tensor to a :class:`Prediction`."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Prediction:
        """Turn the decoder output ``x`` into a :class:`Prediction`."""
