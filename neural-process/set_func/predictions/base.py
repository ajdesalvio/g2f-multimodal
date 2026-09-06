"""The :class:`Prediction` output contract shared by every model head.

A model's ``forward`` returns a :class:`Prediction`. It always exposes a point
estimate via :attr:`~Prediction.mean`; a *distributional* prediction
additionally implements :meth:`~Prediction.log_prob`. The framework never
assumes a closed-form density — adding a new predictive distribution means
writing a new :class:`Prediction` subclass that supplies its own
:meth:`log_prob` (closed form, or a sample-based estimate the author chooses)
and :meth:`cat`. Code's job ends at *calling* ``log_prob``; deciding whether a
density exists and how to compute or estimate it is the head author's job.

This is the abstraction that replaces "wrap a point estimate in a Gaussian with
constant variance": a genuine point estimator returns a :class:`PointPrediction`
(no ``log_prob``), and likelihood-based metrics (e.g. ``loglik``) error loudly
on it instead of silently scoring a fabricated distribution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class Prediction(ABC):
    """A model's predictive output — a point estimate, optionally with density.

    Concrete subclasses (:class:`GaussianPrediction`,
    :class:`PointPrediction`) wrap whatever parameters they need. The eval
    pooler treats them opaquely: it :meth:`flatten`\\ s and
    :meth:`detach_cpu`\\ s each per-batch prediction, buffers them, then
    :meth:`cat`\\ s the batch parts into one prediction over the whole split so
    every metric — mean-based or likelihood-based — is computed once, exactly,
    and batch-size invariantly.
    """

    #: Whether this prediction carries a density (implements :meth:`log_prob`).
    #: Point predictions set this ``False`` so callers (e.g. the test-path
    #: ``predict_step``) can skip likelihood scoring without catching an
    #: exception. Likelihood-based *metrics*, by contrast, do call
    #: :meth:`log_prob` and surface its error — asking for ``loglik`` on a
    #: point head is a configuration mistake, not something to paper over.
    is_distributional: bool = True

    @property
    @abstractmethod
    def mean(self) -> torch.Tensor:
        """The point estimate. Always available, for every prediction type."""

    def log_prob(self, targets: torch.Tensor) -> torch.Tensor:
        """Per-element log density of ``targets`` under this prediction.

        Only distributional predictions implement this. The base raises so a
        point estimator fails loudly (and informatively) when a likelihood
        metric is requested, rather than substituting a fabricated density.
        """
        raise NotImplementedError(
            f"{type(self).__name__} implements no log_prob: it is a point "
            "estimate with no predictive density. A likelihood-based metric "
            "(e.g. 'loglik') or an NLL loss requires a distributional head. "
            "Either use a distributional head, or, if your distribution has no "
            "closed-form density, implement log_prob with a sample-based "
            "estimate on a dedicated Prediction subclass."
        )

    @abstractmethod
    def flatten(self) -> "Prediction":
        """A copy with every leaf tensor reshaped to 1-D along the sample axis.

        The pooler buffers one flattened prediction per batch and
        :meth:`cat`\\ s them along that single axis, so a metric sees the same
        flat ``mean`` / ``log_prob`` extent as its flattened targets — matching
        ``predict_step``'s long-standing reshape-to-1-D convention.
        """

    @abstractmethod
    def detach_cpu(self) -> "Prediction":
        """A detached, CPU-resident copy, for cross-epoch buffering."""

    @classmethod
    @abstractmethod
    def cat(cls, parts: Sequence["Prediction"]) -> "Prediction":
        """Concatenate same-type (flattened) parts along the sample axis."""


def cat_predictions(parts: Sequence[Prediction]) -> Prediction:
    """Concatenate a non-empty sequence of same-type predictions.

    Dispatches to the concrete subclass's :meth:`Prediction.cat`. All parts
    must already be flattened (the pooler buffers them that way).
    """
    parts = list(parts)
    if not parts:
        raise ValueError("cat_predictions requires a non-empty sequence.")
    return type(parts[0]).cat(parts)
