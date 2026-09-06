"""
LOO fold utilities: validation and metadata extraction.

Subcommands:

    python -m cv.folds validate \\
        --data-dir ./dataset-files/g2f/Pedigrees_Wide_Format_BLUEs \\
        --folds DEH1.2020 IAH4.2021 WIH1.2020

    python -m cv.folds metadata \\
        --config <path/to/dataset.yaml> \\
        --folds DEH1.2020 IAH4.2021 \\
        --output artifacts/cv/env_year_loo/fold_metadata.json

    After the Hydra migration, the dataset config lives in
    conf/data/g2f.yaml; supply that path directly or any YAML with a
    matching ``dataset:`` block.
"""

import argparse
import difflib
import json
import os
import re
import sys


_FILENAME_PATTERN = re.compile(
    r"(?P<Env>[^.]+)\.(?P<Year>[^.]+)\.(?P<Inbred>[^.]+)\.(?P<Tester>[^.]+)\.csv"
)


def list_folds(data_dir: str) -> set[str]:
    """Return the set of unique 'Env.Year' fold identifiers present in data_dir filenames."""
    folds = set()
    try:
        files = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    for fname in files:
        m = _FILENAME_PATTERN.match(fname)
        if m:
            folds.add(f"{m.group('Env')}.{m.group('Year')}")
    return folds


def validate(data_dir: str, requested: list[str]) -> bool:
    """
    Check that every fold identifier in `requested` is present in `data_dir`.

    Fold identifiers are 'Env.Year' strings (e.g. 'DEH1.2020').
    Returns True if all found; prints diagnostics and returns False otherwise.
    """
    available = list_folds(data_dir)
    missing = [f for f in requested if f not in available]

    if not missing:
        print(f"[folds] All {len(requested)} fold(s) found in {data_dir}.")
        return True

    print(f"[folds] ERROR: {len(missing)} fold(s) not found in {data_dir}:")
    for fold in missing:
        hint = difflib.get_close_matches(fold, available, n=1, cutoff=0.6)
        suffix = f"  (did you mean: {hint[0]}?)" if hint else ""
        print(f"  - {fold}{suffix}")

    print(f"\n[folds] Available folds ({len(available)}):")
    for fold in sorted(available):
        print(f"  {fold}")
    return False


def _compute_fold_stats(
    metadata_df,
    split_indices: dict[str, list[int]],
) -> dict[str, int]:
    """Compute per-fold statistics from metadata_df and split_indices.

    Returns a dict with counts for each split (n_samples, unique pedigrees,
    unique environments).
    """
    stats = {}
    for split_name in ("train", "val", "test"):
        idxs = split_indices.get(split_name, [])
        if idxs:
            split_meta = metadata_df.iloc[idxs]
            stats[f"n_{split_name}_samples"] = len(idxs)
            stats[f"n_{split_name}_unique_pedigrees"] = int(
                split_meta["Pedigree"].nunique()
            )
            stats[f"n_{split_name}_unique_envs"] = int(
                split_meta["Env"].nunique()
            )
        else:
            stats[f"n_{split_name}_samples"] = 0
            stats[f"n_{split_name}_unique_pedigrees"] = 0
            stats[f"n_{split_name}_unique_envs"] = 0
    return stats


def compute_fold_metadata(
    config_paths: list[str],
    folds: list[str],
) -> dict[str, dict[str, int]]:
    """Compute per-fold statistics for a list of (Env.Year) folds.

    Loads the dataset config from YAML files, then for each fold constructs
    the DataReader + DatasetSplitter and collects split statistics
    (processing is forced off, so no coverage filter runs).

    No feature processing is run — only metadata-level operations.

    Parameters
    ----------
    config_paths : list[str]
        YAML config files to merge (e.g., conf/data/g2f.yaml).
    folds : list[str]
        Fold identifiers in ``Env.Year`` format.

    Returns
    -------
    dict[str, dict[str, int]]
        Mapping from fold_id to stats dict.
    """
    from omegaconf import OmegaConf

    from utils.data.dataset import G2FDataset

    # Merge configs
    configs = [OmegaConf.load(p) for p in config_paths]
    base_config = OmegaConf.merge(*configs)
    ds_cfg = base_config.dataset

    all_stats = {}
    for fold_id in folds:
        fold_env, fold_year = fold_id.split(".")

        # Build fold-specific test filter
        fold_ds_cfg = OmegaConf.merge(
            ds_cfg,
            OmegaConf.create({
                "test_filter_columns": [{"Env": fold_env}, {"Year": int(fold_year)}],
                # Disable feature processing — metadata only
                "processing": None,
                # Disable smoke subsampling
                "smoke_n": None,
            }),
        )

        # Construct dataset (steps 1-5: load, split, coverage filter, no processing)
        dataset = G2FDataset(**OmegaConf.to_container(fold_ds_cfg, resolve=True))

        all_stats[fold_id] = _compute_fold_stats(
            dataset.metadata_df, dataset.split_indices
        )
        print(f"[metadata] {fold_id}: {all_stats[fold_id]}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="LOO fold utilities: validation and metadata extraction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── validate ──
    val_parser = subparsers.add_parser(
        "validate", help="Check that all requested Env.Year folds exist in the data"
    )
    val_parser.add_argument(
        "--data-dir", required=True,
        help="Path to G2F data directory containing per-pedigree CSV files"
    )
    val_parser.add_argument(
        "--folds", required=True, nargs="+",
        help="Fold identifiers to validate, in Env.Year format (e.g. DEH1.2020 IAH4.2021)"
    )

    # ── metadata ──
    meta_parser = subparsers.add_parser(
        "metadata",
        help="Compute per-fold split statistics (sample counts, unique pedigrees/envs)",
    )
    meta_parser.add_argument(
        "--config", required=True, nargs="+",
        help="YAML config file(s) to merge (e.g. conf/data/g2f.yaml)",
    )
    meta_parser.add_argument(
        "--folds", required=True, nargs="+",
        help="Fold identifiers in Env.Year format",
    )
    meta_parser.add_argument(
        "--output", required=True,
        help="Path to write fold_metadata.json",
    )

    args = parser.parse_args()

    if args.command == "validate":
        ok = validate(args.data_dir, args.folds)
        sys.exit(0 if ok else 1)
    elif args.command == "metadata":
        stats = compute_fold_metadata(args.config, args.folds)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[metadata] Wrote {args.output}")


if __name__ == "__main__":
    main()
