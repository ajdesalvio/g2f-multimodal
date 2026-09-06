"""Per-fold kernel eigen-rank scanner.

For every LOO fold, fit the per-processor kernel state on the training
split and record the effective rank (post-noise-filter eigen count) of
each kernel eigen source. The output CSV is the reference for choosing hard-fix
``n_components`` values that are legal on every fold.

Execution location
------------------
This script MUST run on Grace (or any host with the full production
genomic/phenomic/enviromic caches in ``dataset-files/g2f/.cache/``).
Local developer machines typically carry only a handful of smoke-test
cache entries — running locally would silently produce ranks computed
against a partial dosage matrix and corrupt the hard-fix
sweep grid. The script checks for a ``.grace-scan`` sentinel file at
the repo root and refuses to run without it unless ``--i-know-what-im-doing``
is passed.

Usage
-----
    # On Grace, from the repo root:
    touch .grace-scan  # one-time: asserts this host is the production env
    python -m cv.rank_scan \\
        --folds DEH1.2020 IAH4.2021 MIH1.2020 MNH1.2020 MNH1.2021 \\
                MOH1.2020 NEH1.2021 TXH1.2020 TXH1.2021 TXH2.2020 \\
                TXH2.2021 TXH3.2020 TXH3.2021 WIH1.2020 WIH1.2021 \\
                WIH2.2020 WIH2.2021 WIH3.2020 WIH3.2021 \\
        --output artifacts/analysis/per_fold_rank.csv

The CSV header records the full dosage matrix checksum, so downstream
consumers can verify a CSV was generated against the current production
dosage matrix.

Output schema
-------------
Header row::

    # dosage_sha256=<hex> n_pedigrees=<int> generated=<iso8601>
    fold_id,n_train_pedigrees,n_train_envs,r_add,r_dom,r_phe,r_env

``r_*`` columns hold the post-filter effective rank for each source;
absent sources (processor disabled, or source not in the rank probe
fragment) are left blank.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import os
import sys
from typing import Iterable

# conf/__init__.py registers the MiscConfig structured schema via
# ConfigStore. Without this import, compose() below fails with
# MissingConfigException: Could not load 'base_misc'.
import conf  # noqa: F401
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict

from utils.data.dataset import G2FDataset


_SOURCE_COLUMNS = [
    ("genomic_add", "r_add"),
    ("genomic_dom", "r_dom"),
    ("phenomic_relmat", "r_phe"),
    ("enviromic_linear", "r_env"),
]


def _dosage_checksum(genomic_csv_path: str) -> tuple[str, int]:
    """Return (sha256_hex, n_pedigrees) for the production dosage matrix.

    We hash the raw CSV bytes rather than the decoded matrix — faster,
    and the decoding path is deterministic given the same bytes.
    """
    h = hashlib.sha256()
    n_rows = 0
    with open(genomic_csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n_rows += chunk.count(b"\n")
    # n_rows counts the header line too; subtract one for the pedigree count.
    return h.hexdigest(), max(0, n_rows - 1)


def _build_fold_dataset(fold_env: str, fold_year: int) -> G2FDataset:
    """Compose the rank-probe experiment with a per-fold test filter.

    We use ``hydra.initialize`` with a relative path so the script is
    runnable from anywhere under the repo root, and we explicitly pass
    ``--config-path=../conf`` via the context manager.
    """
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=_rank_probe",
                f"dataset.test_filter_columns=[{{Env: {fold_env}}}, {{Year: {fold_year}}}]",
            ],
        )

    with open_dict(cfg):
        if "_target_" in cfg.dataset:
            del cfg.dataset["_target_"]
    ds_kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
    return G2FDataset(**ds_kwargs)


def _scan_folds(folds: Iterable[str]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for fold_id in folds:
        fold_env, fold_year_s = fold_id.split(".")
        fold_year = int(fold_year_s)
        print(f"[rank_scan] fitting processors for fold {fold_id} ...", flush=True)
        dataset = _build_fold_dataset(fold_env, fold_year)
        dims = dataset.processor.feature_dims if dataset.processor else {}

        train_meta = dataset.metadata_df.iloc[dataset.split_indices["train"]]
        row: dict[str, int | str] = {
            "fold_id": fold_id,
            "n_train_pedigrees": int(train_meta["Pedigree"].nunique()),
            "n_train_envs": int(train_meta["Env"].nunique()),
        }
        for src_name, col in _SOURCE_COLUMNS:
            row[col] = int(dims[src_name]) if src_name in dims else ""
        print(f"[rank_scan]   {row}", flush=True)
        rows.append(row)
    return rows


def _write_csv(
    output_path: str,
    rows: list[dict[str, int | str]],
    dosage_sha256: str,
    n_pedigrees: int,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    columns = ["fold_id", "n_train_pedigrees", "n_train_envs"] + [
        col for _, col in _SOURCE_COLUMNS
    ]
    generated = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with open(output_path, "w", newline="") as f:
        f.write(
            f"# dosage_sha256={dosage_sha256} "
            f"n_pedigrees={n_pedigrees} generated={generated}\n"
        )
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[rank_scan] wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--genomic-csv",
        default="./dataset-files/g2f/G2F_2020_2021_Genomic_Data.csv",
        help="Path to the production dosage matrix for checksum recording.",
    )
    parser.add_argument(
        "--i-know-what-im-doing",
        action="store_true",
        help="Bypass the .grace-scan sentinel guard. Only use for local "
             "dry-runs where you understand the output is not authoritative.",
    )
    args = parser.parse_args()

    if not os.path.isfile(".grace-scan") and not args.i_know_what_im_doing:
        print(
            "[rank_scan] refusing to run: no .grace-scan sentinel found.\n"
            "            This script is authoritative only when run against\n"
            "            the production processor cache (Grace). See the\n"
            "            module docstring for the full protocol.",
            file=sys.stderr,
        )
        return 2

    if not os.path.isfile(args.genomic_csv):
        print(
            f"[rank_scan] genomic CSV not found: {args.genomic_csv}",
            file=sys.stderr,
        )
        return 2

    dosage_sha256, n_pedigrees = _dosage_checksum(args.genomic_csv)
    print(f"[rank_scan] dosage_sha256={dosage_sha256} n_pedigrees={n_pedigrees}")

    rows = _scan_folds(args.folds)
    _write_csv(args.output, rows, dosage_sha256, n_pedigrees)
    return 0


if __name__ == "__main__":
    sys.exit(main())
