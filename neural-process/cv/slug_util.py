"""Single source of truth for run identity slugs.

`resolve_slug` composes the experiment and returns `cfg.misc.model_slug`,
which the misc base schema defines as `${model_name}${hp_tag}`. Under the
unified per-encoder grammar each set encoder / FPCA block is one peer token
that carries its own time axis and channel set inline — e.g.
`vi.t=${vi_time}.c=${vi_chan}${vi_aug}` and the weather analog
`wthr.t=...c=...` — so the axis / feature-subset identity lives *inside*
`model_name` rather than in separate trailing tags. The axis group sets the
`*_time` / `*_aug` pieces; the `vi_subset` / `weather_subset` groups set the
`*_chan` pieces (see conf/config.yaml for the defaults). The
artifact-directory slug, the W&B run name, and `meta.model_slug` in
`metrics.json` all derive from this one interpolation.

Feature-subset and axis selection are owned entirely by the experiment
composer's `defaults:` list (`vi_subset` / `weather_subset` / `axis`). The
submit scripts do not pass overrides — that path previously masked an
R-reproduction misconfiguration (silent fallback to the 37-VI
`full` subset). Ad-hoc subset/axis experiments require a dedicated
experiment composer.
"""
from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import conf  # noqa: F401  -- side-effect: register OmegaConf resolvers

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


def fold_dir_token(
    cv_seed: int | None = None,
    fold: int | None = None,
    heldout_env: str | None = None,
) -> str:
    """The per-split artifact-dir token for a CV scheme (D10).

    Single source of truth for the fold-identity path component that
    isolates each split's artifacts under ``artifacts/cv/<scheme>/``:

    - ``cv_2_1``      → ``Seed01.Fold3``                (cv_seed + fold)
    - ``cv_0_00``     → ``Seed01.Fold3.DEH1.2020``      (+ heldout_env)
    - ``env_year_loo``→ ``DEH1.2020``                   (heldout_env only)

    Seeds/folds are zero-padded to two digits so lexical sort matches
    numeric order (``Seed02`` < ``Seed10``). ``heldout_env`` is appended
    verbatim (already an ``Env.Year`` string). The submit arrays
    build/parse this token, so keep it here.
    """
    parts: list[str] = []
    if cv_seed is not None:
        parts.append(f"Seed{int(cv_seed):02d}")
    if fold is not None:
        parts.append(f"Fold{int(fold)}")
    if heldout_env:
        parts.append(str(heldout_env))
    if not parts:
        raise ValueError(
            "fold_dir_token: at least one of cv_seed / fold / heldout_env "
            "must be set."
        )
    return ".".join(parts)


def resolve_slug(experiment_path: str) -> str:
    """Compose ``experiment_path`` and return ``misc.model_slug``.

    Parameters
    ----------
    experiment_path : str
        Hydra experiment name relative to ``conf/experiment/``, e.g.
        ``"fpca/fpca_..._bglr_mse"`` or ``"dl/istnp_vi__vienc-tfm__aug-vi__nll"``. All run identity
        (feature subsets, time axis, ...) is baked into the composer's
        ``defaults`` — e.g. an AGDD variant pulls ``/axis: shared_t-agdd``, so its
        per-encoder time tokens (``vi.t=agdd`` / ``wthr.t=agdd``) are already
        in the slug. No CLI overrides.

    Returns
    -------
    str
        The resolved ``misc.model_slug`` interpolation.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONF_DIR), version_base="1.3"
    ):
        cfg = compose(
            config_name="config",
            overrides=[f"+experiment={experiment_path}"],
        )
    slug = str(
        OmegaConf.to_container(
            OmegaConf.create({"v": cfg.misc.model_slug}), resolve=True
        )["v"]
    )
    assert "${" not in slug, f"Unresolved interpolation in slug: {slug!r}"
    return slug
