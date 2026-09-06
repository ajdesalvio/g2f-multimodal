"""Per-feature score scalers for processor inputs (FPC scores, etc.).

Scalers operate on 2D arrays ``X`` of shape ``(n_samples, n_features)``
and optionally a list of per-sample env labels. They support ``fit``
(learn scaling parameters), ``transform`` (apply learned parameters to
new data), ``state_dict`` (serialize for caching), and ``load_state``
(restore from cache).

The registry ``_SCALING_REGISTRY`` is the **single source of truth**
mapping config names to slug codes and scaler classes. Any new scaling
scheme adds one entry here; nothing else encodes the mapping.

Slug grammar
------------
The slug code is the lowercase ``<scope><method>`` form, used as an
`.s=` facet on a peer token (e.g. ``vi.…s=wez``,
``vi.…s=gz``). Scope tokens: ``g`` = global, ``we`` = within-env. Method
letters: ``z`` = z-score; future ``m`` = max-norm, ``u`` = unit-norm.

See ``baselines/README.md`` and ``utils/data/processing/README.md`` for
the full registry table.
"""

from __future__ import annotations

import warnings

import numpy as np


def _safe_std_ddof1(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """``X.std(axis, ddof=1)`` matching R's ``sd()`` semantics.

    NumPy emits ``RuntimeWarning: Degrees of freedom <= 0`` and a
    follow-on ``invalid value encountered in divide`` (0/0) when the
    sliced length is 1, and returns nan; R's ``sd()`` returns NA in
    the same case. Suppress both warnings here — the upstream clamp
    ``sigma[~np.isfinite(sigma)] = 1.0`` is the equivalent of the R reference's
    ``safe_scale_matrix`` defensive branch.
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0",
            category=RuntimeWarning,
        )
        return X.std(axis=axis, ddof=1)


class _BaseScaler:
    """Abstract base — subclasses learn and apply a per-feature scaling.

    All scalers accept ``envs`` for API symmetry; env-agnostic scalers
    ignore it. ``fit`` returns ``None``; ``transform`` returns a new
    array of the same shape as the input.
    """

    name: str = ""  # config name, e.g. "global_zscore"
    slug_code: str = ""  # slug infix, e.g. "gz"; empty string for "none"

    def fit(
        self,
        X: np.ndarray,
        envs: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    def transform(
        self,
        X: np.ndarray,
        envs: list[str] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a dict of arrays for npz serialization. Empty for stateless."""
        return {}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore from a state dict produced by ``state_dict``."""
        return None

    @property
    def is_fitted(self) -> bool:
        return True


class NoOpScaler(_BaseScaler):
    """Identity scaler — passes input through unchanged."""

    name = "none"
    slug_code = ""

    def fit(self, X, envs=None):
        return None

    def transform(self, X, envs=None):
        return np.asarray(X, dtype=np.float64)


class GlobalZScoreScaler(_BaseScaler):
    """Z-score using global mean/std computed at fit time.

    Stats are intended to come from the train split; predict-mode
    transforms then apply those stored stats to val/test.
    """

    name = "global_zscore"
    slug_code = "gz"

    def __init__(self):
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X, envs=None):
        X = np.asarray(X, dtype=np.float64)
        self._mean = X.mean(axis=0)
        # ddof=1 matches R's sd() (sample SD, divides by n-1) — the R reference's
        # safe_scale_matrix → scale() uses sd() under the hood.
        std = _safe_std_ddof1(X, axis=0)
        # Match the R reference's safe_scale_matrix: clamp std=0 / non-finite to 1
        std[~np.isfinite(std) | (std == 0)] = 1.0
        self._std = std

    def transform(self, X, envs=None):
        if self._mean is None:
            raise RuntimeError("GlobalZScoreScaler.transform called before fit.")
        X = np.asarray(X, dtype=np.float64)
        out = (X - self._mean) / self._std
        out[~np.isfinite(out)] = 0.0
        return out

    def state_dict(self):
        return {"mean": self._mean, "std": self._std}

    def load_state(self, state):
        self._mean = state["mean"]
        self._std = state["std"]

    @property
    def is_fitted(self):
        return self._mean is not None


class WithinEnvZScoreScaler(_BaseScaler):
    """Per-env z-score — each env's column mean/std derived from its own samples.

    ``fit`` builds a per-env stats table; ``transform`` looks up the
    sample's env and applies that env's stats. Mirrors the R reference's
    ``safe_scale_matrix`` applied per-env (transductive: an env's stats
    are computed from all samples in that env regardless of split).

    A sample whose env was not seen at fit time raises ``KeyError``.
    """

    name = "within_env_zscore"
    slug_code = "wez"

    def __init__(self):
        self._envs_order: list[str] | None = None
        self._mean_by_env: np.ndarray | None = None  # (n_envs, D)
        self._std_by_env: np.ndarray | None = None  # (n_envs, D)
        self._env_to_idx: dict[str, int] | None = None

    def fit(self, X, envs=None):
        if envs is None:
            raise ValueError(
                "WithinEnvZScoreScaler.fit requires per-sample env labels."
            )
        X = np.asarray(X, dtype=np.float64)
        if len(envs) != X.shape[0]:
            raise ValueError(
                f"envs length ({len(envs)}) does not match X rows "
                f"({X.shape[0]})."
            )
        unique_envs = sorted(set(envs))
        envs_arr = np.asarray(envs)
        means = []
        stds = []
        for env in unique_envs:
            mask = envs_arr == env
            X_env = X[mask]
            mu = X_env.mean(axis=0)
            # ddof=1 matches R's sd() — the R reference's per-env safe_scale_matrix
            # call uses sd() under the hood. Single-row envs collapse to
            # nan under ddof=1, which the clamp below maps to 1.0
            # (consistent with the previous ddof=0 zero-variance branch).
            sigma = _safe_std_ddof1(X_env, axis=0)
            sigma[~np.isfinite(sigma) | (sigma == 0)] = 1.0
            means.append(mu)
            stds.append(sigma)
        self._envs_order = unique_envs
        self._mean_by_env = np.stack(means)
        self._std_by_env = np.stack(stds)
        self._env_to_idx = {e: i for i, e in enumerate(unique_envs)}

    def transform(self, X, envs=None):
        if self._mean_by_env is None:
            raise RuntimeError(
                "WithinEnvZScoreScaler.transform called before fit."
            )
        if envs is None:
            raise ValueError(
                "WithinEnvZScoreScaler.transform requires per-sample env labels."
            )
        X = np.asarray(X, dtype=np.float64)
        if len(envs) != X.shape[0]:
            raise ValueError(
                f"envs length ({len(envs)}) does not match X rows "
                f"({X.shape[0]})."
            )
        try:
            idx = np.array([self._env_to_idx[e] for e in envs])
        except KeyError as exc:
            raise KeyError(
                f"Env not seen at fit time: {exc.args[0]!r}. "
                f"WithinEnvZScoreScaler requires fit to cover every env "
                f"that will appear at transform time (use the union of "
                f"all dataset splits at fit time)."
            ) from None
        means = self._mean_by_env[idx]
        stds = self._std_by_env[idx]
        out = (X - means) / stds
        out[~np.isfinite(out)] = 0.0
        return out

    def state_dict(self):
        return {
            "envs_order": np.array(self._envs_order, dtype=object),
            "mean_by_env": self._mean_by_env,
            "std_by_env": self._std_by_env,
        }

    def load_state(self, state):
        self._envs_order = [str(e) for e in state["envs_order"]]
        self._mean_by_env = state["mean_by_env"]
        self._std_by_env = state["std_by_env"]
        self._env_to_idx = {e: i for i, e in enumerate(self._envs_order)}

    @property
    def is_fitted(self):
        return self._mean_by_env is not None


# ── Registry ─────────────────────────────────────────────────────
#
# Single source of truth: config name → (slug code, scaler class).
# Adding a new scheme = add one entry here.

_SCALING_REGISTRY: dict[str, tuple[str, type[_BaseScaler]]] = {
    "none": ("", NoOpScaler),
    "global_zscore": ("gz", GlobalZScoreScaler),
    "within_env_zscore": ("wez", WithinEnvZScoreScaler),
}


def build_scaler(scaling: str) -> _BaseScaler:
    """Instantiate a scaler by config name.

    Raises ``ValueError`` if ``scaling`` is not in ``_SCALING_REGISTRY``.
    """
    if scaling not in _SCALING_REGISTRY:
        raise ValueError(
            f"Unknown scaling {scaling!r}. Valid options: "
            f"{sorted(_SCALING_REGISTRY)}."
        )
    _, cls = _SCALING_REGISTRY[scaling]
    return cls()


def slug_code_for(scaling: str) -> str:
    """Return the slug-grammar code for a scaling name (or '' for none)."""
    if scaling not in _SCALING_REGISTRY:
        raise ValueError(
            f"Unknown scaling {scaling!r}. Valid options: "
            f"{sorted(_SCALING_REGISTRY)}."
        )
    code, _ = _SCALING_REGISTRY[scaling]
    return code


def requires_envs(scaling: str) -> bool:
    """True iff the scaling needs per-sample env labels at fit/transform."""
    return scaling in {"within_env_zscore"}
