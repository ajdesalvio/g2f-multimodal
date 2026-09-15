"""Fast, dependency-free checks of the manuscript release (no model fitting).

Run from any directory: python scripts/validate_release.py
Hashes and inventory coverage are checked separately by validate_manifests.py.
"""

from __future__ import annotations

import csv
import math
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent if ROOT.name == "functional-analysis" else ROOT
ERRORS: list[str] = []
WARNINGS: list[str] = []
CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        ERRORS.append(message)


def table(relative: str, expected: int, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{relative}: missing columns {sorted(missing)}")
        rows = list(reader)
    check(len(rows) == expected, f"{relative}: expected {expected} rows, found {len(rows)}")
    return rows


def unique(rows: list[dict[str, str]], columns: tuple[str, ...], label: str) -> None:
    keys = [tuple(row[column] for column in columns) for row in rows]
    check(len(set(keys)) == len(keys), f"{label}: duplicate keys {columns}")
    check(all(all(value.strip() for value in key) for key in keys), f"{label}: blank key values")


def numbers(rows: list[dict[str, str]], columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        bad = []
        for index, row in enumerate(rows, start=2):
            try:
                value = float(row[column])
                valid = math.isfinite(value)
                if "RMSE" in column or column.endswith("_SD"):
                    valid = valid and value >= 0
                elif column in {"Correlation", "Cor", "Within_Environment_r", "Pooled_Pearson"}:
                    valid = valid and -1 <= value <= 1
            except (ValueError, TypeError):
                valid = False
            if not valid:
                bad.append(index)
        check(not bad, f"{label}: invalid {column} at CSV lines {bad[:5]}")


def predictions() -> None:
    folder = "results/prediction/final_summaries/"
    key = ("Method", "Model", "Time_Domain", "CV")
    rows = table(folder + "G2F_Final_Prediction_Summary.csv", 152,
                 key + ("Checkpoint", "Correlation", "RMSE", "Seeds", "Environments"))
    unique(rows, key, "Primary predictions")
    numbers(rows, ("Correlation", "RMSE"), "Primary predictions")
    check(Counter(row["Method"] for row in rows) == {"BGLR": 140, "TNP": 12},
          "Primary predictions: expected 140 BGLR and 12 TNP rows")
    loeo = [row for row in rows if row["CV"] == "LOEO"]
    check(len(loeo) == 28 and all(row["Method"] == "BGLR" for row in loeo),
          "Primary predictions: expected 28 BGLR LOEO rows")
    check(Counter(row["Time_Domain"] for row in loeo) == {"DAP": 14, "AGDD": 14},
          "LOEO: expected 14 models in each domain")
    tnp = [row for row in rows if row["Method"] == "TNP"]
    check(all(row["Checkpoint"] == "best_frozen_pearson_r" and row["Time_Domain"] == "DAP"
              for row in tnp), "Primary TNP: checkpoint/domain mismatch")
    check(all(row["Environments"] == "19" for row in rows),
          "Primary predictions: each summary must span 19 environments")
    cv = [row for row in rows if row["CV"] != "LOEO"]
    check(all(row["CV"] in {"CV0", "CV00", "CV1", "CV2"} for row in cv),
          "Primary predictions: unexpected CV label")
    check(all(row["Seeds"] == ("10" if row["Method"] == "BGLR" else "5") for row in cv),
          "Primary predictions: expected 10 BGLR seeds and 5 TNP seeds")
    numbers(cv, ("Correlation_SD", "RMSE_SD"), "Primary CV predictions")

    seed = table(folder + "G2F_Final_Prediction_Seed_Summary.csv", 1180,
                 key + ("Seed", "Correlation", "RMSE"))
    unique(seed, key + ("Seed",), "Seed summaries")
    numbers(seed, ("Correlation", "RMSE"), "Seed summaries")
    check(Counter(row["Method"] for row in seed) == {"BGLR": 1120, "TNP": 60},
          "Seed summaries: unexpected method counts")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in seed:
        groups.setdefault(tuple(row[column] for column in key), []).append(row)
    for row in cv:
        identity = tuple(row[column] for column in key)
        group = groups.get(identity, [])
        expected_seeds = {str(i) for i in range(1, int(row["Seeds"]) + 1)}
        check({item["Seed"] for item in group} == expected_seeds,
              f"Seed summaries: incomplete seed coverage for {identity}")
        if group:
            for metric in ("Correlation", "RMSE"):
                average = sum(float(item[metric]) for item in group) / len(group)
                check(math.isclose(average, float(row[metric]), rel_tol=1e-12, abs_tol=1e-12),
                      f"Primary {metric} does not match seed means for {identity}")

    sensitivity = table(folder + "G2F_CV1_CV2_Pooled_Pearson_Sensitivity.csv", 590,
                        key + ("Seed", "Within_Environment_r", "Pooled_Pearson"))
    unique(sensitivity, key + ("Seed",), "Pearson sensitivity")
    numbers(sensitivity, ("Within_Environment_r", "Pooled_Pearson"), "Pearson sensitivity")
    check(all(row["CV"] in {"CV1", "CV2"} for row in sensitivity),
          "Pearson sensitivity contains a non-CV1/CV2 row")
    checkpoints = table(folder + "G2F_TNP_Checkpoint_Sensitivity.csv", 120,
                        key + ("checkpoint", "Correlation", "RMSE"))
    unique(checkpoints, key + ("checkpoint",), "Checkpoint sensitivity")
    numbers(checkpoints, ("Correlation", "RMSE"), "Checkpoint sensitivity")
    check(len({row["checkpoint"] for row in checkpoints}) == 10,
          "Checkpoint sensitivity: expected 10 checkpoint rules")
    raw = table("results/prediction/tnp/tidy_all_metrics.csv", 30000,
                ("model", "cv_label", "fold", "seed", "checkpoint"))
    unique(raw, ("model", "cv_label", "fold", "seed", "checkpoint"), "Raw TNP metrics")


def qtl() -> None:
    key = ("lodcolumn", "chr", "pos", "Env.Tester", "FPCA_type")
    full = table("results/qtl/QTL_Results_Combined_Revised_Intervals.csv", 206,
                 key + ("lod", "ci_lo", "ci_hi"))
    subset = table("results/qtl/publication/QTL_Results_NGRDI_FPC_167_Peaks.csv", 167,
                   key + ("lod", "ci_lo", "ci_hi"))
    unique(full, key, "Full revised QTL table")
    unique(subset, key, "Publication QTL table")
    check(Counter(row["FPCA_type"] for row in subset) == {"DAP": 71, "AGDD": 96},
          "Publication QTL table: expected 71 DAP and 96 AGDD peaks")
    check(all(re.fullmatch(r"FPC\d+_NGRDI", row["lodcolumn"]) for row in subset),
          "Publication QTL table contains a non-NGRDI-FPC peak")
    lookup = {tuple(row[column] for column in key): row for row in full}
    for row in subset:
        identity = tuple(row[column] for column in key)
        original = lookup.get(identity)
        check(original is not None, f"Publication QTL peak absent from full table: {identity}")
        if original:
            for column in ("lod", "ci_lo", "ci_hi"):
                check(math.isclose(float(row[column]), float(original[column]),
                                   rel_tol=1e-12, abs_tol=1e-8),
                      f"Publication QTL {column} differs from full table: {identity}")


def cohorts_and_weather() -> None:
    folder = "data/supplementary/"
    environments = table(folder + "Env_Names_G2F_2020_2021.csv", 19, ("Env",))
    unique(environments, ("Env",), "Environments")
    envs = {row["Env"] for row in environments}
    matched = table(folder + "Pedigree_Overlap_Phenomic_Phenotypic_BLUEs.csv", 10109,
                    ("Pedigree.Env", "Pedigree"))
    unique(matched, ("Pedigree.Env",), "Matched phenotype/phenomic cohort")
    hybrids = {row["Pedigree"] for row in matched}
    check(len(hybrids) == 1180, "Matched cohort: expected 1,180 unique hybrids")
    matched_envs = []
    for row in matched:
        prefix = row["Pedigree"] + "."
        matched_envs.append(row["Pedigree.Env"][len(prefix):] if row["Pedigree.Env"].startswith(prefix) else "")
    check(set(matched_envs) == envs, "Matched cohort: pedigree/environment keys do not match 19 environments")
    genomic = table(folder + "Pedigree_Overlap_Genomic_Phenomic.csv", 1180, ("Pedigree",))
    unique(genomic, ("Pedigree",), "Genomic/phenomic overlap")
    check({row["Pedigree"] for row in genomic} == hybrids, "Genomic and matched phenotype hybrid sets differ")
    females = table(folder + "Common_Females.csv", 223, ("Female",))
    unique(females, ("Female",), "Canonical maternal lines")

    weather = table("results/figures/source_data/Yield_FPC_Correlations_All_DAP.csv", 151,
                    ("FPC_ID", "Cor", "Domain", "Environments", "Yield_Records"))
    unique(weather, ("FPC_ID",), "Weather correlations")
    numbers(weather, ("Cor",), "Weather correlations")
    check(all(row["Domain"] == "DAP" and row["Environments"] == "19" and
              row["Yield_Records"] == "10109" for row in weather),
          "Weather correlations must use all-DAP scores and 10,109 records in 19 environments")
    lookup = {row["FPC_ID"]: row for row in weather}
    check("FPC2_T2M_MIN" in lookup and math.isclose(float(lookup["FPC2_T2M_MIN"]["Cor"]),
          -0.637662111202421, rel_tol=1e-12, abs_tol=1e-12),
          "Weather release sentinel differs from the corrected matched-cohort result")


def qtl_archive_inventory() -> None:
    """Validate the portable inventory, without requiring the external 32.80 GB archive."""
    rows = table("data/QTL_RDS_MANIFEST.csv", 58,
                 ("analysis", "env_tester", "source_path", "size_bytes", "sha256",
                  "expected_results_path", "checked_date"))
    unique(rows, ("analysis", "env_tester"), "QTL scan archive")
    unique(rows, ("sha256",), "QTL scan checksums")
    expected = set((ROOT / "scripts/04_qtl/hprc/env_testers.txt").read_text(
        encoding="utf-8-sig").splitlines())
    expected.discard("")
    check(len(expected) == 29, "QTL task configuration must contain 29 environment/tester keys")
    check(Counter(row["analysis"] for row in rows) == {"dap": 29, "agdd": 29},
          "QTL scan archive: expected 29 DAP and 29 AGDD objects")
    for analysis in ("dap", "agdd"):
        check({row["env_tester"] for row in rows if row["analysis"] == analysis} == expected,
              f"QTL scan archive: {analysis} task coverage differs from configured tasks")
    for row in rows:
        label = f"{row['analysis']}/{row['env_tester']}"
        check(re.fullmatch(r"[0-9A-F]{64}", row["sha256"]) is not None,
              f"QTL scan archive: invalid SHA-256 for {label}")
        check(row["size_bytes"].isdigit() and int(row["size_bytes"]) > 0,
              f"QTL scan archive: invalid byte count for {label}")
        filename = row["env_tester"] + "QTL.Outputs.rds"
        check(row["expected_results_path"] == f"qtl/{row['analysis']}/output/{filename}",
              f"QTL scan archive: unexpected working-results path for {label}")
        folder = {"dap": "Standard_FPCA", "agdd": "AGDD_FPCA"}.get(row["analysis"], "")
        check(row["source_path"] == f"5.2_HPRC_TIME_CAPSULE/{folder}/{filename}",
              f"QTL scan archive: unexpected source path for {label}")


def public_file(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    return (path.is_file() and not any(part in {".git", ".codex", ".agents", "__pycache__"}
            for part in relative.parts) and path != ROOT / "config/paths.yml"
            and path.suffix != ".pyc")


def release_paths():
    """Own workflow plus shared metadata, without auditing the sibling implementation."""
    yield from ROOT.rglob("*")
    if REPO_ROOT != ROOT:
        yield REPO_ROOT / "README.md"
        yield REPO_ROOT / "LICENSING.md"
        yield REPO_ROOT / "CITATION.cff"
        yield from (REPO_ROOT / ".github").rglob("*")


def layout_and_imagery() -> None:
    check((REPO_ROOT / "neural-process/README.md").is_file(), "Sibling TNP README is missing")
    check((REPO_ROOT / "CITATION.cff").read_bytes() == (ROOT / "CITATION.cff").read_bytes(),
          "Shared and functional-analysis citation files differ")
    # This small, value-only workbook is read without Excel or optional Python packages.
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(ROOT / "results/drone/D2S_STAC_Links.xlsx") as archive:
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = book.findall("s:sheets/s:sheet", ns)
        check(len(sheets) == 1 and sheets[0].get("name") == "D2S_STAC_Links",
              "Unexpected imagery workbook layout")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(node.itertext()) for node in
                       ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", ns)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        check(not sheet.findall(".//s:f", ns), "Imagery workbook unexpectedly contains formulas")
        rows = []
        for row in sheet.findall("s:sheetData/s:row", ns):
            cells = {}
            for cell in row.findall("s:c", ns):
                column = re.sub(r"\d", "", cell.get("r", ""))
                value = cell.findtext("s:v", "", ns)
                if cell.get("t") == "s":
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(n.text or "" for n in cell.findall(".//s:t", ns))
                cells[column] = value
            if any(cells.values()):
                rows.append([cells.get(column, "") for column in ("A", "B", "C")])
    check(rows[0] == ["Env", "STAC Link", "DOI"], "Imagery workbook headers changed")
    rows = rows[1:]
    check(len(rows) == 16, "Expected 16 geospatial collections")
    check(len({row[0] for row in rows}) == 16, "Duplicate imagery collection identifiers")
    check(len({url for row in rows for url in row[1:]}) == 32, "Duplicate imagery URLs")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    mapped_envs = set()
    for env, stac, doi in rows:
        check(stac.startswith("https://stac.d2s.org/collections/") and
              doi.startswith("https://doi.org/10.6084/"), f"Unexpected imagery URL for {env}")
        expected = f"| {env} | [Open collection]({stac}) | [{doi.removeprefix('https://doi.org/')}]({doi}) |"
        check(expected in readme, f"README collection row differs from workbook: {env}")
        if env.startswith("TXH123."):
            mapped_envs.update(f"TXH{i}.{env.split('.')[1]}" for i in (1, 2, 3))
        else:
            mapped_envs.add(re.sub(r"[.]C5[ab]$", "", env))
    environments = table("data/supplementary/Env_Names_G2F_2020_2021.csv", 19, ("Env",))
    check(mapped_envs == {row["Env"] for row in environments},
          "Imagery collections do not cover exactly the 19 analysis environments")


def documentation_and_sizes() -> None:
    # Standard inline links plus reference definitions. Fragments are ignored;
    # checking their rendering requires a Markdown renderer, not filesystem QA.
    inline = re.compile(r"\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")
    reference = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE)
    for path in release_paths():
        if not public_file(path):
            continue
        if path.stat().st_size >= 50 * 1024 * 1024:
            WARNINGS.append(f"Large public file (>=50 MiB): {path.relative_to(REPO_ROOT).as_posix()}")
        if path.suffix.lower() != ".md":
            continue
        content = path.read_text(encoding="utf-8-sig")
        content = re.sub(r"(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", content)
        for target in inline.findall(content) + reference.findall(content):
            target = target.strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith(("#", "//")):
                continue
            link = unquote(parsed.path)
            if not link:
                continue
            resolved = (REPO_ROOT / link.lstrip("/")) if link.startswith("/") else (path.parent / link)
            try:
                resolved.resolve().relative_to(REPO_ROOT)
            except ValueError:
                check(False, f"Link escapes repository in {path.relative_to(REPO_ROOT).as_posix()}: {target}")
                continue
            check(resolved.exists(), f"Broken local link in {path.relative_to(REPO_ROOT).as_posix()}: {target}")


def obvious_private_material() -> None:
    # Deliberately narrow: scientific CSVs/provenance are not treated as code,
    # generic configuration examples are allowed, and no matched secret value
    # is ever printed. This supplements, not replaces, a real secret scanner.
    suffixes = {".r", ".py", ".sh", ".md", ".yml", ".yaml", ".json", ".cff"}
    account_path = re.compile(
        r"(?:[A-Za-z]:[/\\]Users[/\\]|/scratch/user/|/home/)([A-Za-z0-9][A-Za-z0-9_.-]*)"
    )
    examples = {"USER", "USERNAME", "YOUR_USER", "YOUR_USERNAME", "EXAMPLE"}
    secret = re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
        r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----)"
    )
    for path in release_paths():
        if not public_file(path) or path.suffix.lower() not in suffixes:
            continue
        content = path.read_text(encoding="utf-8-sig")
        personal = [match for match in account_path.finditer(content)
                    if match.group(1).upper() not in examples]
        label = path.relative_to(REPO_ROOT).as_posix()
        check(not personal, f"Potential personal account path in {label}")
        check(secret.search(content) is None, f"Potential credential/private key in {label}")


def main() -> int:
    for name, operation in (("prediction tables", predictions), ("QTL tables", qtl),
                            ("QTL archive inventory", qtl_archive_inventory),
                            ("cohorts and weather", cohorts_and_weather),
                            ("two-workflow layout and geospatial links", layout_and_imagery),
                            ("Markdown links and file sizes", documentation_and_sizes),
                            ("obvious private material", obvious_private_material)):
        before = len(ERRORS)
        try:
            operation()
        except (OSError, ValueError, KeyError, TypeError) as error:
            ERRORS.append(f"{name}: {error}")
        print(f"{'PASS' if len(ERRORS) == before else 'FAIL'}: {name}")
    for warning in WARNINGS:
        print(f"WARNING: {warning}")
    for error in ERRORS:
        print(f"ERROR: {error}")
    print(f"{CHECKS} checks; {len(ERRORS)} errors; {len(WARNINGS)} warnings.")
    print("Scope: deposited tables, internal consistency, local links, and size warnings; "
          "not model reruns, external-link checks, spreadsheet formulas, or comprehensive secret scanning.")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
