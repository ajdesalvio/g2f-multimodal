"""OmegaConf custom resolver registration for the Hydra config tree.

Imported unconditionally by ``conf/__init__.py`` so registration fires
the first time anything imports the ``conf`` package: every entry
point imports ``conf`` before any
``@hydra.main`` or ``hydra.compose`` call.

Resolvers:

- ``eval`` — arithmetic in YAML interpolations.
- ``cond`` — two YAML call sites: the ``-fold=...`` suffix in
  ``model_run_name`` (conf/misc/base.yaml) and ``wa_tag``
  (conf/config.yaml). Truthiness contract is **Python-truthy** —
  empty string and 0 are falsy, any non-empty string is truthy. Do
  NOT replace with a strict ``bool(condition)`` parse: callers rely
  on Python truthiness of the ``fold_env`` string.
- ``scheme_label`` — maps a ``cv_spec.scheme`` token to its W&B project
  label (one project per scheme: ``G2F-<label>``). Identity for every
  scheme except ``env_year_loo`` → ``ey_loo`` (condensed). Drives
  ``misc.project``; the ``G2F-`` prefix stays in ``conf/misc/base.yaml``.
- ``opt_tag`` — conditional optimizer/LR fragment for ``hp_tag``: empty for the
  legacy defaults (AdamW, lr=5e-5) so existing slugs/artifacts/W&B runs are
  preserved; emits ``_opt=<name>`` / ``_lr=<value>`` only when changed.
"""
from __future__ import annotations

from omegaconf import OmegaConf


def _cond(condition, true_val, false_val=""):
    return true_val if condition else false_val


# cv_spec.scheme → W&B project label. Identity unless listed here; the only
# condensed token is the verbose env_year_loo. Keep keys in sync with the
# `scheme:` values in conf/cv_spec/*.yaml.
_SCHEME_PROJECT_LABELS = {"env_year_loo": "ey_loo"}


def _scheme_label(scheme):
    return _SCHEME_PROJECT_LABELS.get(str(scheme), str(scheme))


# Legacy optimizer defaults — runs using exactly these keep their historical slug
# (no _opt=/_lr= token), so existing artifacts / W&B runs are never orphaned.
_OPT_DEFAULT_NAME = "AdamW"
_LR_DEFAULT = 5.0e-5


def _opt_tag(target, lr):
    """Conditional optimizer/LR fragment for `hp_tag`.

    Empty for the legacy defaults (AdamW, lr=5e-5) so previously-trained models
    keep their slug; emits a distinguishing token ONLY when changed:
    ``_opt=<name>`` when the optimizer differs from AdamW, ``_lr=<value>`` when
    the LR differs from 5e-5. (weight_decay stays unconditional in hp_tag — legacy
    runs already carry ``_wd=0.01`` — so changing it is already collision-safe.)
    """
    parts = []
    name = str(target).rsplit(".", 1)[-1]
    if name != _OPT_DEFAULT_NAME:
        parts.append(f"_opt={name.lower()}")
    if float(lr) != _LR_DEFAULT:
        parts.append(f"_lr={float(lr):g}")
    return "".join(parts)


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    if not OmegaConf.has_resolver("cond"):
        OmegaConf.register_new_resolver("cond", _cond)
    if not OmegaConf.has_resolver("scheme_label"):
        OmegaConf.register_new_resolver("scheme_label", _scheme_label)
    if not OmegaConf.has_resolver("opt_tag"):
        OmegaConf.register_new_resolver("opt_tag", _opt_tag)


register_resolvers()
