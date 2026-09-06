#!/usr/bin/env python
"""Generate a pinned female→fold map (``female_folds.csv``) for the CV schemes.

Why this exists
---------------
For a **fair, paired** comparison of DL and FPCA/BGLR models under the
``cv_2_1`` / ``cv_0_00`` schemes, both pipelines must hold out the *same*
genotypes in the same fold for every seed. The common-female *set* is
already deterministic (RNG-free; ``cv/schemes/common_females.py``), but the
fold *assignment* over that set is seeded. Materializing the assignment once
into a committed CSV and reading it back via ``cv_spec.fold_source=r_csv``
pins the partition across pipelines *and* across reruns, and ships as a
reproducible supplementary artifact.

The assignment is byte-identical to what ``cv_spec.fold_source=native``
produces for the same seed — this script reuses :class:`NativeFoldSource`
verbatim — so ``r_csv`` and ``native`` agree fold-for-fold; the CSV merely
*freezes* the native partition into a shareable file. Columns match
:class:`cv.schemes.fold_source.RCsvFoldSource`'s contract exactly:
``Seed_Num, Female, Fold``.

The common-female universe is computed over the **full pre-coverage**
phenotype frame — the same ``common_universe_df`` the dataset build passes to
``resolve_cv_assignment`` (``dataset.py``: ``full_metadata_df`` captured
*before* the genotype-coverage filter). Building ``G2FDataset`` with
``processing=None`` applies no coverage filter, so ``dataset.metadata_df``
here equals that full frame — which guarantees the emitted females are
exactly the set ``RCsvFoldSource``'s strict validation recomputes at runtime.

Usage
-----
    python scripts/cv/make_female_folds.py \
        --data-dir ./dataset-files/g2f/Pedigrees_Wide_Format_BLUEs \
        --seeds 1-10 --k-folds 5 \
        --out cv/data/female_folds.csv

Then point the pipelines at it (both DL and FPCA jobs read this env var):
    export G2F_FEMALE_FOLDS_CSV=$PWD/cv/data/female_folds.csv
    # ... +cv_spec=cv_2_1 cv_spec.fold_source=r_csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Make the repo root importable when run as a plain script (sys.path[0] is the
# script's own dir, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_seeds(spec: str) -> list[int]:
    """Parse ``"1-10"`` (inclusive range) or ``"1,2,5"`` (explicit list)."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _load_common_universe(data_dir: str) -> "pd.DataFrame":
    """Return the full pre-coverage phenotype frame (R's Metadata universe).

    Built with ``cv_spec=None`` + ``processing=None`` so no coverage filter
    runs and ``metadata_df`` is the reader's full frame — exactly the
    ``common_universe_df`` the runtime feeds to ``common_female_set``.
    """
    from utils.data.dataset import G2FDataset

    ds = G2FDataset(
        data_dir=data_dir,
        validate=False,
        case_sensitive=False,
        delete_invalid=True,
        # to_tensor=True so the build's train-statistics pass (torch.cat over
        # the train tensors) succeeds; we only read metadata_df afterward.
        to_tensor=True,
        test_ratio=0,
        eval_streams=None,
        normalize={
            "dap": False,
            "channels": False,
            "weather_concat_values": False,
            "yield_value": False,
        },
        processing=None,
        cv_spec=None,
    )
    return ds.metadata_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a pinned female_folds.csv (Seed_Num, Female, Fold).",
    )
    parser.add_argument(
        "--data-dir",
        default="./dataset-files/g2f/Pedigrees_Wide_Format_BLUEs",
        help="G2FDataset data_dir (the wide-format BLUEs tree).",
    )
    parser.add_argument(
        "--seeds",
        default="1-10",
        help="Seeds as an inclusive range '1-10' or a list '1,2,5'.",
    )
    parser.add_argument(
        "--k-folds", type=int, default=5, help="Number of folds (default 5)."
    )
    parser.add_argument(
        "--out",
        default="cv/data/female_folds.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    from cv.schemes.common_females import sorted_common_females
    from cv.schemes.fold_source import NativeFoldSource

    seeds = _parse_seeds(args.seeds)
    print(f"[make_female_folds] seeds={seeds} k_folds={args.k_folds}")
    print(f"[make_female_folds] loading common-female universe from {args.data_dir}")

    meta = _load_common_universe(args.data_dir)
    commons = sorted_common_females(meta)  # normalized (lowercased), sorted
    if not commons:
        raise SystemExit(
            "[make_female_folds] ERROR: no common females found — check data_dir."
        )
    print(
        f"[make_female_folds] {len(commons)} common females over "
        f"{meta['Env'].map(str).nunique()} env codes "
        f"({_env_year_count(meta)} Env.Year environments)"
    )

    rows: list[tuple[int, str, int]] = []
    common_set = set(commons)
    for seed in seeds:
        fold_map = NativeFoldSource(cv_seed=seed, k_folds=args.k_folds).build(
            common_set
        )
        # Emit in the deterministic sorted-female order for a stable diff.
        sizes: dict[int, int] = {}
        for fem in commons:
            fold = fold_map[fem]
            rows.append((seed, fem, fold))
            sizes[fold] = sizes.get(fold, 0) + 1
        print(
            f"[make_female_folds] seed={seed:2d} fold sizes "
            f"{ {k: sizes[k] for k in sorted(sizes)} }"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Seed_Num", "Female", "Fold"])
        writer.writerows(rows)

    print(
        f"[make_female_folds] wrote {len(rows)} rows "
        f"({len(seeds)} seeds × {len(commons)} females) → {out_path}"
    )
    print(
        "[make_female_folds] NOTE: '*.csv' is gitignored — commit with:\n"
        f"    git add -f {out_path}"
    )


def _env_year_count(meta) -> int:
    from cv.schemes.identifiers import env_year_series

    return int(env_year_series(meta).nunique())


if __name__ == "__main__":
    main()
