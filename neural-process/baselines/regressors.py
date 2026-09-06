"""
Regressor registry for FPCA baselines.

Each regressor is a factory function decorated with @register(name).
It receives a config dict (from the ``regressor`` YAML section) and
returns a sklearn-compatible estimator (must have .fit() and .predict()).

To add a new regressor, define a factory function and decorate it.
"""

from __future__ import annotations

from typing import Any

REGISTRY: dict[str, callable] = {}


def register(name: str):
    """Decorator that registers a regressor factory under *name*."""
    def decorator(fn):
        REGISTRY[name] = fn
        return fn
    return decorator


_SUPPORTED_LOSSES = {"mse"}


def build_regressor(cfg: dict) -> Any:
    """
    Instantiate a regressor from a config dict.

    *cfg* should have at least a ``name`` key.  All other keys are
    passed through to the factory as keyword arguments.

    Example cfg::

        {"name": "ols", "loss_name": "mse"}
    """
    cfg = dict(cfg)  # shallow copy so pop doesn't mutate caller
    name = cfg.pop("name")
    loss_name = cfg.pop("loss_name", "mse")

    if loss_name not in _SUPPORTED_LOSSES:
        raise ValueError(
            f"Unknown loss_name '{loss_name}'. "
            f"Available: {sorted(_SUPPORTED_LOSSES)}"
        )
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown regressor '{name}'. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name](cfg)


# ── Regressor factories ─────────────────────────────────────────────


@register("ols")
def _build_ols(cfg):
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


@register("gpr")
def _build_gpr(cfg):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel, DotProduct, RBF, WhiteKernel,
    )

    kernel_name = cfg.get("gpr_kernel", "dot_product")
    if kernel_name == "dot_product":
        kernel = ConstantKernel() * DotProduct() + WhiteKernel()
    elif kernel_name == "rbf":
        kernel = ConstantKernel() * RBF() + WhiteKernel()
    else:
        raise ValueError(f"Unknown gpr_kernel: {kernel_name}")

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=cfg.get("alpha", 0.0),
        n_restarts_optimizer=cfg.get("n_restarts_optimizer", 1),
        normalize_y=cfg.get("normalize_y", True),
        random_state=cfg.get("seed", 0),
    )


@register("bglr")
def _build_bglr(cfg):
    from .bglr_regressor import BGLRRegressor

    env_brr_key = cfg.get("env_brr_key")
    feature_keys = cfg.get("feature_keys") or []
    if env_brr_key is not None and env_brr_key not in feature_keys:
        raise ValueError(
            f"BGLR config error: env_brr_key={env_brr_key!r} is set "
            f"but {env_brr_key!r} is not in feature_keys="
            f"{list(feature_keys)!r}. The env block must be listed "
            f"in feature_keys so the upstream metadata_features "
            f"processor produces it; the BGLR branch in fpca_core "
            f"then pops it out and routes it to a dedicated BRR ETA "
            f"term."
        )

    return BGLRRegressor(
        n_iter=cfg.get("n_iter", 10000),
        burn_in=cfg.get("burn_in", 1000),
        seed=cfg.get("seed", 0),
        env_brr_key=env_brr_key,
    )


@register("gblup")
def _build_gblup(cfg):
    from .gblup_regressor import GBLUPRegressor

    # Same env-block contract as bglr: env_brr_key must be declared in
    # feature_keys so the upstream metadata_features processor produces
    # it; the transductive branch in fpca_core then routes it to the
    # dedicated env_block argument (folded into K_env = Z_e Z_eᵀ).
    env_brr_key = cfg.get("env_brr_key")
    feature_keys = cfg.get("feature_keys") or []
    if env_brr_key is not None and env_brr_key not in feature_keys:
        raise ValueError(
            f"GBLUP config error: env_brr_key={env_brr_key!r} is set "
            f"but {env_brr_key!r} is not in feature_keys="
            f"{list(feature_keys)!r}. The env block must be listed "
            f"in feature_keys so the upstream metadata_features "
            f"processor produces it; the transductive branch in "
            f"fpca_core then pops it out and routes it to a dedicated "
            f"env kernel term."
        )

    return GBLUPRegressor(
        reml_max_iter=cfg.get("reml_max_iter", 40),
        tol=cfg.get("tol", 1e-4),
        seed=cfg.get("seed", 0),
        env_brr_key=env_brr_key,
    )
