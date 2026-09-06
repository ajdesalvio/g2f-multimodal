"""``holdout_env`` — OOD carve holding out whole ``Env.Year`` column(s).

Mirrors the ``cv_0_00`` / LOEO regime: validation is one or more *unseen
environments* drawn from the train pool. Takes an exact spec (``n_envs``
or an explicit ``envs`` list); the held-out **test** env is already
PREDICT (not in the train pool), so naming it is a config error.
"""

from __future__ import annotations

import numpy as np

from cv.schemes.identifiers import env_year_series, normalize_id

from .base import ValidationStrategy
from .registry import register_val_strategy


@register_val_strategy("holdout_env")
class HoldoutEnvValidationStrategy(ValidationStrategy):
    def __init__(
        self,
        n_envs: int | None = None,
        envs: list[str] | None = None,
    ) -> None:
        if (n_envs is None) == (envs is None):
            raise ValueError(
                "holdout_env requires exactly one of 'n_envs' / 'envs' "
                f"(got n_envs={n_envs!r}, envs={envs!r})."
            )
        if n_envs is not None and int(n_envs) < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs!r}.")
        self.n_envs = int(n_envs) if n_envs is not None else None
        self.envs = (
            [normalize_id(e) for e in envs] if envs is not None else None
        )

    def select(self, meta_df, train_positions, *, rng):
        pool = np.sort(np.asarray(train_positions, dtype=int))
        env_arr = env_year_series(meta_df).to_numpy()
        pool_envs = env_arr[pool]
        present = sorted(set(pool_envs.tolist()))  # canonical order

        if self.envs is not None:
            missing = [e for e in self.envs if e not in present]
            if missing:
                raise ValueError(
                    f"holdout_env: Env.Year {missing} not in the train pool "
                    f"(the held-out test env is PREDICT, not train — it "
                    f"cannot be a val carve). Present: {present}."
                )
            chosen = set(self.envs)
        else:
            if self.n_envs > len(present):
                raise ValueError(
                    f"holdout_env: requested n_envs={self.n_envs} but only "
                    f"{len(present)} Env.Year column(s) in the train pool."
                )
            chosen = set(
                rng.choice(
                    np.array(present), size=self.n_envs, replace=False
                ).tolist()
            )

        mask = np.fromiter(
            (e in chosen for e in pool_envs), dtype=bool, count=len(pool)
        )
        return pool[mask]
