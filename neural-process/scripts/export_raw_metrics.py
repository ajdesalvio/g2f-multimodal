#!/usr/bin/env python3
"""Export raw per-split metrics from the TNP full-scale CV campaign.

Reads the ``metrics_<checkpoint>.json`` files written by ``eval.py`` under
``artifacts/cv/cv_0_00_2_1/neural_process/`` and emits two layers:

  1. ``tidy_all_metrics.csv``  — one row per
     (model, cv_scheme, cv_label, fold, seed, checkpoint). Lossless.
  2. ``wide/<model>__<metric>__<cv_label>.csv`` — rows are (fold, checkpoint),
     columns are ``seed_1``..``seed_5``.

Metrics come straight from the JSONs rather than ``summary.csv`` because
``cv.aggregate`` fixes its fieldnames at cv/aggregate.py:443-449 and drops
``loglik``.

Seeds are matched cv_seed:model_seed pairs (seed_N == pair N:N), so the single
``seed`` index identifies the pair.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import os
# Env-overridable so a new campaign (e.g. a different fold table) can be exported
# next to, not over, an earlier one.
ROOT = Path(os.environ.get("TNP_EXPORT_ROOT", REPO / "artifacts" / "cv" / "cv_0_00_2_1" / "neural_process")).resolve()
OUT = Path(os.environ.get("TNP_EXPORT_OUT", REPO / "exports" / "tnp_full_scale")).resolve()

SCHEMES = ("cv_0_00", "cv_2_1")
# Only the reported metrics. spearman_r / loglik are deliberately not exported:
# they are still available in the source metrics_*.json if ever needed. Note
# this is independent of the *checkpoints* — the loglik-selected checkpoints
# (best_loglik, best_frozen_loglik, best_frozen_d01_loglik) are all still here,
# because the stopping rule need not match the reported metric.
METRICS = ("rmse", "pearson_r")
# carried in the tidy file only. r_w / rho_w are Tiezzi within-block
# correlations (block = environment); their ``*_n_blocks`` counts travel with
# them because the metric degenerates to the plain correlation at n_blocks == 1
# (which is every cv_0_00 split, since those hold out a single environment).
EXTRA = ("r_w", "r_w_n_blocks", "rho_w", "rho_w_n_blocks", "n")

# Weighted correlations also get wide files, but only for CV labels where they
# actually pool more than one environment — see ``_informative_weighted``.
WEIGHTED = ("r_w", "rho_w")
NBLOCKS = {"r_w": "r_w_n_blocks", "rho_w": "rho_w_n_blocks"}

TOKEN_RE = re.compile(r"^Seed(\d+)\.(.+)$")


def model_name(slug: str) -> str:
    """Map a config slug to its modality label."""
    has_weather = "wthr." in slug
    has_geno = "ga.rows" in slug
    if has_weather and has_geno:
        return "istnp-full"
    if has_geno:
        return "weather-free"
    return "vi-only"


def collect() -> list[dict]:
    rows: list[dict] = []
    for scheme in SCHEMES:
        scheme_dir = ROOT / scheme
        if not scheme_dir.is_dir():
            sys.exit(f"missing scheme dir: {scheme_dir}")
        for token_dir in sorted(scheme_dir.iterdir()):
            if not token_dir.is_dir():
                continue
            m = TOKEN_RE.match(token_dir.name)
            if not m:
                sys.exit(f"unparseable token: {token_dir.name}")
            token_seed, fold = int(m.group(1)), m.group(2)

            for slug_dir in sorted(token_dir.iterdir()):
                if not slug_dir.is_dir():
                    continue
                model = model_name(slug_dir.name)

                for seed_dir in sorted(slug_dir.glob("seed=*")):
                    seed = int(seed_dir.name.split("=", 1)[1])
                    # cv_seed lives in both the token and the leaf; they must agree
                    if seed != token_seed:
                        sys.exit(f"seed mismatch: {seed_dir} vs token {token_seed}")
                    if not (seed_dir / "curve.done").exists():
                        sys.exit(f"incomplete split (no curve.done): {seed_dir}")

                    for mf in sorted(seed_dir.glob("metrics_*.json")):
                        checkpoint = mf.name[len("metrics_") : -len(".json")]
                        with open(mf) as fh:
                            payload = json.load(fh)
                        for cv_label, vals in payload["by_metric"].items():
                            row = {
                                "model": model,
                                "cv_scheme": scheme,
                                "cv_label": cv_label,
                                "fold": fold,
                                "seed": seed,
                                "checkpoint": checkpoint,
                                "model_slug": slug_dir.name,
                            }
                            for key in METRICS + EXTRA:
                                row[key] = vals.get(key)
                            rows.append(row)
    return rows


def write_tidy(rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tidy_all_metrics.csv"
    fieldnames = [
        "model", "cv_scheme", "cv_label", "fold", "seed", "checkpoint",
        *METRICS, *EXTRA, "model_slug",
    ]
    ordered = sorted(
        rows,
        key=lambda r: (r["model"], r["cv_scheme"], r["cv_label"],
                       r["fold"], r["checkpoint"], r["seed"]),
    )
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ordered)
    return path


def _informative_weighted(rows: list[dict]) -> set[tuple[str, str]]:
    """(cv_label, weighted_metric) pairs worth writing a wide file for.

    A weighted correlation pools per-environment correlations, so it only
    carries information beyond the plain metric when it spans more than one
    environment. At ``n_blocks == 1`` it is *identical* to ``pearson_r`` /
    ``spearman_r`` — true of every cv_0_00 split, which holds out a single
    environment — and a file for it would be a byte-for-byte duplicate.
    """
    labels = {r["cv_label"] for r in rows}
    return {
        (label, metric)
        for metric in WEIGHTED
        for label in labels
        if all(r[NBLOCKS[metric]] > 1 for r in rows if r["cv_label"] == label)
    }


def write_wide(rows: list[dict]) -> tuple[int, list[str], set]:
    wide_dir = OUT / "wide"
    wide_dir.mkdir(parents=True, exist_ok=True)

    seeds = sorted({r["seed"] for r in rows})
    seed_cols = [f"seed_{s}" for s in seeds]
    informative = _informative_weighted(rows)

    # (model, metric, cv_label) -> (fold, checkpoint) -> {seed: value}
    table: dict[tuple, dict[tuple, dict]] = defaultdict(lambda: defaultdict(dict))
    scheme_of: dict[tuple, str] = {}
    for r in rows:
        emit = list(METRICS) + [
            m for m in WEIGHTED if (r["cv_label"], m) in informative
        ]
        for metric in emit:
            key = (r["model"], metric, r["cv_label"])
            scheme_of[key] = r["cv_scheme"]
            table[key][(r["fold"], r["checkpoint"])][r["seed"]] = r[metric]

    warnings: list[str] = []
    for (model, metric, cv_label), cells in sorted(table.items()):
        path = wide_dir / f"{model}__{metric}__{cv_label}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["fold", "checkpoint", *seed_cols])
            for (fold, checkpoint) in sorted(cells):
                per_seed = cells[(fold, checkpoint)]
                missing = [s for s in seeds if per_seed.get(s) is None]
                if missing:
                    warnings.append(
                        f"{path.name}: {fold}/{checkpoint} missing seeds {missing}"
                    )
                w.writerow([fold, checkpoint, *[per_seed.get(s) for s in seeds]])
    return len(table), warnings, informative


def main() -> None:
    rows = collect()
    tidy = write_tidy(rows)
    n_files, warnings, informative = write_wide(rows)

    n_models = len({r["model"] for r in rows})
    labels = sorted({r["cv_label"] for r in rows})
    skipped = sorted(
        f"{lab}/{met}"
        for met in WEIGHTED
        for lab in labels
        if (lab, met) not in informative
    )
    print(f"tidy rows       : {len(rows)}")
    print(f"models          : {n_models}")
    print(f"cv labels       : {labels}")
    print(f"tidy file       : {tidy.relative_to(REPO)}")
    print(f"wide files      : {n_files}")
    print(f"weighted wide   : written for {sorted(informative)}")
    print(f"                  skipped (n_blocks==1, duplicate of plain): {skipped}")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for line in warnings[:20]:
            print(f"  {line}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")
    else:
        print("completeness    : every (fold, checkpoint) cell has all 5 seeds")


if __name__ == "__main__":
    main()
