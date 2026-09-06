"""Point-estimate head."""

from __future__ import annotations

import torch

from set_func.predictions import PointPrediction

from .base import BaseHead


class PointHead(BaseHead):
    """Wrap a decoder's raw output as a :class:`PointPrediction`.

    The clean alternative to faking a point estimate as a fixed-variance
    Gaussian: the decoder emits ``s_dim`` values (the prediction itself, no
    scale parameter) and this head returns a point prediction with no density.
    Models using it train on a point loss (e.g. RMSE) and are scored by the
    mean-based metrics (RMSE / Pearson / Spearman); a likelihood metric such as
    ``loglik`` correctly errors on them.
    """

    def forward(self, x: torch.Tensor) -> PointPrediction:
        return PointPrediction(x)
