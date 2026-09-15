"""Validate repository inventories and SHA-256 checksums using the Python standard library."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def is_candidate(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return (
        path.is_file()
        and ".git" not in parts
        and "__pycache__" not in parts
        and path.suffix.lower() != ".pyc"
    )


def check_rows(
    label: str,
    rows: list[dict[str, str]],
    path_column: str,
    hash_column: str,
    size_column: str,
    expected_paths: set[str],
) -> None:
    recorded = [row[path_column] for row in rows]
    duplicates = sorted({path for path in recorded if recorded.count(path) > 1})
    if duplicates:
        ERRORS.append(f"{label} has duplicate paths: {', '.join(duplicates)}")

    recorded_set = set(recorded)
    missing = sorted(expected_paths - recorded_set)
    extra = sorted(recorded_set - expected_paths)
    if missing:
        ERRORS.append(f"{label} omits: {', '.join(missing)}")
    if extra:
        ERRORS.append(f"{label} records absent/unexpected files: {', '.join(extra)}")

    for row in rows:
        path = ROOT / row[path_column]
        if not path.is_file():
            continue
        actual_size = str(path.stat().st_size)
        actual_hash = sha256(path)
        if row[size_column] != actual_size:
            ERRORS.append(f"{label} has a stale size for {row[path_column]}")
        if row[hash_column].upper() != actual_hash:
            ERRORS.append(f"{label} has a stale SHA-256 for {row[path_column]}")


def repository_files() -> set[str]:
    excluded = {"FILE_MANIFEST.csv", "config/paths.yml"}
    return {
        relative(path)
        for path in ROOT.rglob("*")
        if is_candidate(path)
        and relative(path) not in excluded
    }


def code_files(include_source_map: bool) -> set[str]:
    files = {
        relative(path)
        for directory in ("R", "scripts")
        for path in (ROOT / directory).rglob("*")
        if is_candidate(path)
    }
    if not include_source_map:
        files.discard("scripts/SOURCE_MAP.csv")
    return files


def result_artifacts() -> set[str]:
    return {
        relative(path)
        for path in (ROOT / "results").rglob("*")
        if is_candidate(path)
        and path.name not in {"README.md", "RESULT_MANIFEST.csv"}
    }


def main() -> int:
    file_rows = read_csv("FILE_MANIFEST.csv")
    check_rows(
        "FILE_MANIFEST.csv",
        file_rows,
        "relative_path",
        "sha256",
        "size_bytes",
        repository_files(),
    )

    source_rows = read_csv("scripts/SOURCE_MAP.csv")
    check_rows(
        "scripts/SOURCE_MAP.csv",
        source_rows,
        "staging_path",
        "current_sha256",
        "size_bytes",
        code_files(include_source_map=False),
    )

    inventory_rows = read_csv("docs/SCRIPT_INVENTORY.csv")
    check_rows(
        "docs/SCRIPT_INVENTORY.csv",
        inventory_rows,
        "relative_path",
        "sha256",
        "size_bytes",
        code_files(include_source_map=True),
    )

    result_rows = read_csv("results/RESULT_MANIFEST.csv")
    check_rows(
        "results/RESULT_MANIFEST.csv",
        result_rows,
        "relative_path",
        "sha256",
        "size_bytes",
        result_artifacts(),
    )
    for row in result_rows:
        if row["verified_exact_copy"].lower() == "true":
            if row["source_sha256"].upper() != row["sha256"].upper():
                ERRORS.append(
                    "results/RESULT_MANIFEST.csv has an invalid exact-copy flag for "
                    f"{row['relative_path']}"
                )

    for row in read_csv("data/manifest.csv"):
        if row["redistribution_status"] != "included_in_github":
            continue
        path = ROOT / row["relative_path"]
        if not path.is_file():
            ERRORS.append(f"data/manifest.csv cannot find {row['relative_path']}")
            continue
        if row["size_bytes"] != str(path.stat().st_size):
            ERRORS.append(f"data/manifest.csv has a stale size for {row['relative_path']}")
        if row["sha256"].upper() != sha256(path):
            ERRORS.append(f"data/manifest.csv has a stale SHA-256 for {row['relative_path']}")

    if ERRORS:
        print("Manifest validation failed:", file=sys.stderr)
        for error in ERRORS:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Validated repository, source, script, result, and included-data manifests "
        "with SHA-256."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
