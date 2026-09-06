"""Deterministic common-female set — females present in *every* environment.

Mirrors ``DAP_CV_Metadata_V1.R:97-104``:

    common_females <- pheno %>%
      distinct(Env, Female) %>%
      filter(!is.na(Female)) %>%
      add_count(Female, name = "n_envs") %>%
      filter(n_envs == total_envs) %>%
      pull(Female) %>% unique() %>% sort()

The set is **RNG-free** (D5): it depends only on the data's
``(Female, Env.Year)`` incidence, so it is reproducible from the data
alone. Only the *fold assignment* over this set needs R's RNG.

D6 — the set is computed over the dataset's ``Env.Year`` universe (the
unique environments actually present in the frame passed in). To match
R's ``Metadata`` script the dataset build passes the **full** phenotype
frame (pre genotype-coverage filter) as the common-female universe, so a
coverage drop can't silently demote a female from "common"
(``resolve_cv_assignment(common_universe_df=...)``). Membership is then
applied to the coverage-filtered (genotyped) rows the splitter
partitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .identifiers import env_year_series, female_series

if TYPE_CHECKING:
    import pandas as pd


def env_universe(meta_df: "pd.DataFrame") -> set[str]:
    """Distinct normalized ``Env.Year`` values in ``meta_df``."""
    return set(env_year_series(meta_df).unique())


def common_female_set(meta_df: "pd.DataFrame") -> set[str]:
    """Return the set of normalized females present in **all** environments.

    A female counts as common iff the number of distinct ``Env.Year``
    values it appears in equals the total number of distinct environments
    in ``meta_df``.

    Rows with a missing ``Pedigree`` are excluded from the female-incidence
    count, matching R's ``filter(!is.na(Female))`` (``Metadata_V1.R:99``):
    they have no real maternal line, so they must not seed a spurious
    ``"nan"`` female. This matters now that the universe is the full
    pre-coverage frame, which may carry such rows. ``total_envs`` is still
    the full environment count (R derives it from the env list, not the
    NA-filtered rows), so an env present only via null-pedigree rows still
    counts toward the "appears in every env" bar.
    """
    import pandas as pd

    fem = female_series(meta_df)
    env = env_year_series(meta_df)
    total_envs = env.nunique()
    if total_envs == 0:
        return set()
    # Drop null-Pedigree rows before counting (female_of() stringifies NaN
    # to "nan", so guard on the raw Pedigree, not the derived female).
    valid = meta_df["Pedigree"].notna()
    pairs = pd.DataFrame(
        {"female": fem[valid], "env": env[valid]}
    ).drop_duplicates()
    n_envs_per_female = pairs.groupby("female")["env"].nunique()
    return set(n_envs_per_female.index[n_envs_per_female == total_envs])


def sorted_common_females(meta_df: "pd.DataFrame") -> list[str]:
    """Common females as a sorted list (matches R's ``sort()`` ordering).

    Native fold assignment consumes this order; sorting makes the
    partition reproducible across runs given a fixed RNG seed.
    """
    return sorted(common_female_set(meta_df))
