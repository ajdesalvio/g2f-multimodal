"""Query the two exploratory QTL intervals; no assembly or causal-gene filter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from intermine.webservice import Service


# The effective endpoint in Query_MaizeMine_Specific_Interval_V3.py. That source
# assigned the v1.5 URL first, then replaced it with this URL. Keep it explicit.
SERVICE_URL = "http://maizemine.rnet.missouri.edu:8080/maizemine/service"
INTERVALS = {
    "chr7": (126_763_800, 138_089_853),
    "chr3": (112_559_472, 184_794_196),
}


def query_genes_with_go_in_interval(
    service: Service, chromosome: str, start_bp: int, end_bp: int
) -> pd.DataFrame:
    """Return genes overlapping a physical interval, with one row per GO term."""
    query = service.new_query("Gene")
    query.add_view(
        "primaryIdentifier",
        "symbol",
        "name",
        "description",
        "chromosome.primaryIdentifier",
        "chromosomeLocation.start",
        "chromosomeLocation.end",
        "goAnnotation.ontologyTerm.identifier",
        "goAnnotation.ontologyTerm.name",
        "goAnnotation.ontologyTerm.namespace",
    )
    query.add_constraint("chromosome.primaryIdentifier", "=", chromosome, code="A")
    query.add_constraint("chromosomeLocation.start", "<=", end_bp, code="B")
    query.add_constraint("chromosomeLocation.end", ">=", start_bp, code="C")
    rows = [dict(row.items()) for row in query.rows()]
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.columns = [column.replace("Gene.", "") for column in result.columns]
    return result.sort_values(["chromosomeLocation.start", "primaryIdentifier"])


def collapse_go(long_table: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated GO annotations to one row per gene."""
    if long_table.empty:
        return long_table.copy()
    table = long_table.copy()
    go_columns = [
        "goAnnotation.ontologyTerm.identifier",
        "goAnnotation.ontologyTerm.name",
        "goAnnotation.ontologyTerm.namespace",
    ]
    for column in go_columns:
        if column not in table:
            table[column] = pd.NA
    table["GO_triplet"] = table[go_columns].fillna("").agg(" | ".join, axis=1)
    table.loc[table[go_columns[0]].fillna("").eq(""), "GO_triplet"] = ""
    return (
        table.groupby("primaryIdentifier", as_index=False)
        .agg(
            symbol=("symbol", "first"),
            name=("name", "first"),
            description=("description", "first"),
            chr=("chromosome.primaryIdentifier", "first"),
            start=("chromosomeLocation.start", "first"),
            end=("chromosomeLocation.end", "first"),
            n_go=(go_columns[0], lambda values: values.notna().sum()),
            go_terms=(
                "GO_triplet",
                lambda values: "; ".join(sorted({value for value in values if value})),
            ),
        )
        .sort_values(["start", "primaryIdentifier"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosome", choices=["all", *INTERVALS], default="all")
    parser.add_argument("--start-bp", type=int, default=None)
    parser.add_argument("--end-bp", type=int, default=None)
    parser.add_argument("--service-url", default=SERVICE_URL)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    custom = args.start_bp is not None or args.end_bp is not None
    if custom and (args.start_bp is None or args.end_bp is None or args.chromosome == "all"):
        raise ValueError("Custom bounds require one chromosome and both --start-bp and --end-bp")
    if custom and (args.start_bp < 1 or args.start_bp > args.end_bp):
        raise ValueError("Bounds must satisfy 1 <= --start-bp <= --end-bp")
    default_results = Path(os.environ.get("G2F_RESULTS_DIR", "results"))
    output_dir = args.output_dir or Path(
        os.environ.get("G2F_QTL_REFINEMENT_DIR", default_results / "qtl" / "refinement")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    service = Service(args.service_url)
    chromosomes = list(INTERVALS) if args.chromosome == "all" else [args.chromosome]
    for chromosome in chromosomes:
        start_bp, end_bp = (args.start_bp, args.end_bp) if custom else INTERVALS[chromosome]
        long_table = query_genes_with_go_in_interval(service, chromosome, start_bp, end_bp)
        collapsed_table = collapse_go(long_table)
        interval_stub = f"{chromosome}_{start_bp}_{end_bp}_genes_GO"
        long_table.to_csv(output_dir / f"{interval_stub}_long.csv", index=False)
        collapsed_table.to_csv(output_dir / f"{interval_stub}_collapsed.csv", index=False)
        print(f"Saved {len(collapsed_table)} gene identifiers for {chromosome}:{start_bp}-{end_bp}")
    print("Exploratory, multi-assembly annotation only; coordinates are not assembly-harmonized.")


if __name__ == "__main__":
    main()
