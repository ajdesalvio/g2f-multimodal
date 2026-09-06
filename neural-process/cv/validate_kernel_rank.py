"""Submit-side pre-flight for the rank-consistency contract.

Given a list of experiment config names, composes each one via
Hydra and runs the mutual-exclusion sub-check from
``utils.data.processing.rank_check.check_rank_consistency`` with no
fitted processor. Exits non-zero if any config declares both
``n_components`` and ``n_components_max`` on a kernel eigen source,
printing the full offender list.

Scope and limitations
---------------------
This is the minimal-useful pre-flight: it catches **config-shape**
errors (mutual exclusion) on every config, fold-independent. It does
NOT currently catch hard-fix-exceeds-rank errors on warm-cache folds
— that path needs the fitted processor state. Hard-fix
overflow is still caught at runtime by the per-job check in
``utils.experiment.setup._validate_config`` / the
``baselines.fpca_core`` validator, with fold-named error messages.

Rationale: until a production rank CSV (cv/rank_scan.py) lands and the
``gae_k{K}`` hard-fix variants are generated, there are no hard-fix
configs in production anyway. The ambitious warm-cache path can be
added in a follow-up PR that pairs with the ``gae_k{K}`` config
generation.

Usage
-----
    python -m cv.validate_kernel_rank \\
        --configs istnp_vi_ga_gd__grm-rows__vienc-tfm__aug-vi__nll \\
                  fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse \\
        [--folds DEH1.2020 MNH1.2021 ...]

The ``--folds`` argument is accepted for CLI-compatibility with the
submit-side shell scripts but is currently unused (mutual exclusion
is fold-independent). When the warm-cache path is added, folds will
drive per-fold cache key resolution.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

# conf/__init__.py registers the MiscConfig structured schema. Without
# this import, hydra.compose fails with MissingConfigException on
# 'misc/base'.
import conf  # noqa: F401
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from utils.data.processing.rank_check import check_rank_consistency


DL_EXPERIMENT_GROUP = "dl"
FPCA_EXPERIMENT_GROUP = "fpca"


def _compose_experiment(experiment_name: str) -> DictConfig:
    """Compose a config by searching both dl and fpca experiment groups.

    Matches the resolution behaviour of ``cv.slug_util.resolve_slug``.
    Tries DL first, falls back to FPCA.
    """
    for group in (DL_EXPERIMENT_GROUP, FPCA_EXPERIMENT_GROUP):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        try:
            with initialize(version_base=None, config_path="../conf"):
                return compose(
                    config_name="config",
                    overrides=[f"+experiment={group}/{experiment_name}"],
                )
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(
        f"Could not compose experiment {experiment_name!r} under "
        f"either dl or fpca. Check the config file exists under "
        f"conf/experiment/{{dl,fpca}}/."
    )


def validate_configs(
    config_names: Sequence[str],
) -> list[tuple[str, str]]:
    """Run the config-shape pre-check over every config.

    Returns a list of ``(config_name, error_message)`` tuples for any
    config that fails. Empty list means everything passed.
    """
    offenders: list[tuple[str, str]] = []
    for name in config_names:
        try:
            cfg = _compose_experiment(name)
        except Exception as e:  # noqa: BLE001
            offenders.append((name, f"compose failed: {e}"))
            continue
        try:
            check_rank_consistency(cfg, processor=None)
        except ValueError as e:
            offenders.append((name, str(e)))
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", required=True, nargs="+",
        help="Experiment config names (without .yaml). Names are "
             "searched under conf/experiment/{dl,fpca}/.",
    )
    parser.add_argument(
        "--folds", nargs="*", default=None,
        help="Fold identifiers (Env.Year). Currently unused — "
             "accepted for future warm-cache hard-fix check.",
    )
    args = parser.parse_args()

    offenders = validate_configs(args.configs)

    if not offenders:
        print(
            f"[validate_kernel_rank] {len(args.configs)} config(s) "
            f"passed the D1 mutual-exclusion check."
        )
        return 0

    print(
        f"[validate_kernel_rank] {len(offenders)}/{len(args.configs)} "
        f"config(s) failed the D1 mutual-exclusion check:",
        file=sys.stderr,
    )
    for name, msg in offenders:
        print(f"  - {name}: {msg}", file=sys.stderr)
    print(
        "[validate_kernel_rank] Fix the config(s) above before "
        "resubmitting. Mutual exclusion is a config shape error — "
        "set at most one of n_components / n_components_max per "
        "kernel eigen source.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
