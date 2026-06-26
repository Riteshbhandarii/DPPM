#!/usr/bin/env python3
"""
Purpose:
Estimate potential identity leakage that can remain when splitting only by product_id.

Inputs:
- datasets/cleaned/clean_master_dataset.csv by default

Outputs:
- Markdown report summarizing identities shared by multiple product_id groups
- CSV artifacts with repeated-identity tables and product-id-count distributions

Assumptions:
- This is a structural dataset diagnostic only. It does not train models, create splits,
  or modify thesis methodology.
- A potential leakage surface exists when the same canonical identity is represented by
  multiple product_id values, because product-id-only splitting can place comparable
  identities in both train and test.
- Canonicalization is imported from analyze_grouping_strategies.py so this report uses
  the same identity keys as the earlier grouping diagnostics.

How to run:
python3 scripts/analyze_product_id_identity_leakage.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analyze_grouping_strategies import (
    GroupingStrategy,
    canonicalize_series,
    markdown_table,
)


IDENTITY_STRATEGIES = [
    GroupingStrategy(
        name="canonical(part_name, brand, model)",
        slug="part_name_brand_model",
        columns=("part_name", "brand", "model"),
    ),
    GroupingStrategy(
        name="canonical(part_name, brand, model, oem_number)",
        slug="part_name_brand_model_oem_number",
        columns=("part_name", "brand", "model", "oem_number"),
    ),
    GroupingStrategy(
        name="canonical(part_name, brand, model, year_start, year_end)",
        slug="part_name_brand_model_year_start_year_end",
        columns=("part_name", "brand", "model", "year_start", "year_end"),
    ),
]

KEY_SEPARATOR = "\x1f"


@dataclass(frozen=True)
class StrategyOutput:
    strategy: GroupingStrategy
    identity_table: pd.DataFrame
    repeated_identities: pd.DataFrame
    distribution: pd.DataFrame
    summary: dict[str, object]
    repeated_path: Path
    distribution_path: Path
    largest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate identity leakage potential under product-id-only splitting."
    )
    parser.add_argument(
        "--input-path",
        default="datasets/cleaned/clean_master_dataset.csv",
        help="Cleaned master dataset to analyze.",
    )
    parser.add_argument(
        "--output-path",
        default="results/product_id_identity_leakage.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="results/product_id_identity_leakage",
        help="Directory for repeated-identity and distribution CSV artifacts.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of largest shared identities to show in the markdown report.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame) -> None:
    required_columns = sorted(
        {"product_id", *{column for strategy in IDENTITY_STRATEGIES for column in strategy.columns}}
    )
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Required columns are missing from the input dataset: {missing_columns}")


def canonical_frame(frame: pd.DataFrame, strategy: GroupingStrategy) -> pd.DataFrame:
    columns = ("product_id", *strategy.columns)
    return pd.DataFrame(
        {column: canonicalize_series(frame[column]) for column in columns},
        index=frame.index,
    )


def identity_key(frame: pd.DataFrame, strategy: GroupingStrategy) -> pd.Series:
    return frame[list(strategy.columns)].astype("string").agg(KEY_SEPARATOR.join, axis=1)


def product_id_sample(values: pd.Series, limit: int = 10) -> str:
    ordered = sorted(set(values.astype(str)))
    sample = ordered[:limit]
    suffix = "" if len(ordered) <= limit else f" ... (+{len(ordered) - limit} more)"
    return ", ".join(sample) + suffix


def build_identity_table(frame: pd.DataFrame, strategy: GroupingStrategy) -> pd.DataFrame:
    canonical = canonical_frame(frame, strategy)
    working = canonical.copy()
    if "price" in frame.columns:
        working["price"] = pd.to_numeric(frame["price"], errors="coerce")

    aggregations: dict[str, tuple[str, str | object]] = {
        "row_count": ("product_id", "size"),
        "product_id_count": ("product_id", "nunique"),
        "product_id_sample": ("product_id", product_id_sample),
    }
    if "price" in working.columns:
        aggregations.update(
            {
                "median_price": ("price", "median"),
                "min_price": ("price", "min"),
                "max_price": ("price", "max"),
            }
        )

    identity_table = (
        working.groupby(list(strategy.columns), dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    identity_table["rows_per_product_id"] = (
        identity_table["row_count"] / identity_table["product_id_count"]
    ).round(2)
    return identity_table.sort_values(
        ["product_id_count", "row_count", *strategy.columns],
        ascending=[False, False, *([True] * len(strategy.columns))],
    ).reset_index(drop=True)


def repeated_distribution(repeated_identities: pd.DataFrame) -> pd.DataFrame:
    if repeated_identities.empty:
        return pd.DataFrame(
            columns=[
                "product_ids_per_identity",
                "identity_count",
                "identity_pct",
                "product_id_count",
                "row_count",
            ]
        )

    distribution = (
        repeated_identities.groupby("product_id_count", dropna=False)
        .agg(identity_count=("product_id_count", "size"), row_count=("row_count", "sum"))
        .reset_index()
        .rename(columns={"product_id_count": "product_ids_per_identity"})
        .sort_values("product_ids_per_identity")
    )
    total_repeated = distribution["identity_count"].sum()
    distribution["identity_pct"] = (
        distribution["identity_count"] / total_repeated * 100
    ).round(2)
    distribution["product_id_count"] = (
        distribution["product_ids_per_identity"] * distribution["identity_count"]
    )
    return distribution[
        [
            "product_ids_per_identity",
            "identity_count",
            "identity_pct",
            "product_id_count",
            "row_count",
        ]
    ]


def summarize_strategy(
    frame: pd.DataFrame,
    strategy: GroupingStrategy,
    artifact_dir: Path,
) -> StrategyOutput:
    identity_table = build_identity_table(frame, strategy)
    repeated = identity_table[identity_table["product_id_count"] > 1].copy()
    distribution = repeated_distribution(repeated)

    canonical = canonical_frame(frame, strategy)
    repeated_keys = set(identity_key(repeated, strategy)) if not repeated.empty else set()
    row_identity_keys = identity_key(canonical, strategy)
    repeated_row_mask = row_identity_keys.isin(repeated_keys)

    total_product_ids = int(canonical["product_id"].nunique())
    total_rows = int(len(frame))
    product_ids_in_repeated = int(
        canonical.loc[repeated_row_mask, "product_id"].nunique()
    )
    rows_in_repeated = int(repeated["row_count"].sum()) if not repeated.empty else 0

    summary = {
        "identity_strategy": strategy.name,
        "total_identities": int(len(identity_table)),
        "identities_represented_by_multiple_product_ids": int(len(repeated)),
        "repeated_identity_pct": round((len(repeated) / len(identity_table)) * 100, 2)
        if len(identity_table)
        else 0.0,
        "total_product_ids": total_product_ids,
        "product_ids_in_repeated_identities": product_ids_in_repeated,
        "product_ids_in_repeated_identity_pct": round(
            (product_ids_in_repeated / total_product_ids) * 100,
            2,
        )
        if total_product_ids
        else 0.0,
        "rows_in_repeated_identities": rows_in_repeated,
        "rows_in_repeated_identity_pct": round((rows_in_repeated / total_rows) * 100, 2)
        if total_rows
        else 0.0,
        "largest_product_ids_per_identity": int(repeated["product_id_count"].max())
        if not repeated.empty
        else 0,
        "median_product_ids_per_repeated_identity": float(repeated["product_id_count"].median())
        if not repeated.empty
        else 0.0,
    }

    repeated_path = artifact_dir / f"{strategy.slug}_repeated_identities.csv"
    distribution_path = artifact_dir / f"{strategy.slug}_product_id_distribution.csv"
    largest_path = artifact_dir / f"{strategy.slug}_largest_shared_identities.csv"
    repeated.to_csv(repeated_path, index=False)
    distribution.to_csv(distribution_path, index=False)
    repeated.head(50).to_csv(largest_path, index=False)

    return StrategyOutput(
        strategy=strategy,
        identity_table=identity_table,
        repeated_identities=repeated,
        distribution=distribution,
        summary=summary,
        repeated_path=repeated_path,
        distribution_path=distribution_path,
        largest_path=largest_path,
    )


def format_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "strategy": summary["identity_strategy"],
        "total_identities": f"{summary['total_identities']:,}",
        "multi_product_id_identities": (
            f"{summary['identities_represented_by_multiple_product_ids']:,} "
            f"({summary['repeated_identity_pct']:.2f}%)"
        ),
        "product_ids_in_multi_identities": (
            f"{summary['product_ids_in_repeated_identities']:,} "
            f"({summary['product_ids_in_repeated_identity_pct']:.2f}%)"
        ),
        "rows_in_multi_identities": (
            f"{summary['rows_in_repeated_identities']:,} "
            f"({summary['rows_in_repeated_identity_pct']:.2f}%)"
        ),
        "largest_product_ids_per_identity": summary["largest_product_ids_per_identity"],
        "median_product_ids_per_repeated_identity": (
            f"{summary['median_product_ids_per_repeated_identity']:.1f}"
        ),
    }


def top_largest_table(output: StrategyOutput, top_n: int) -> pd.DataFrame:
    columns = [
        *output.strategy.columns,
        "product_id_count",
        "row_count",
        "rows_per_product_id",
        "product_id_sample",
    ]
    price_columns = [column for column in ["median_price", "min_price", "max_price"] if column in output.repeated_identities]
    columns.extend(price_columns)
    return output.repeated_identities[columns].head(top_n)


def write_report(
    input_path: Path,
    output_path: Path,
    row_count: int,
    outputs: list[StrategyOutput],
    top_n: int,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_table = pd.DataFrame([format_summary(output.summary) for output in outputs])

    lines = [
        "# Product-ID-Only Identity Leakage Estimate",
        "",
        f"Generated: {generated_at}",
        "",
        f"Input dataset: `{input_path.as_posix()}`",
        f"Rows analyzed: {row_count:,}",
        "",
        "This report estimates potential optimistic leakage when train/test splitting is performed "
        "only by `product_id`. A product-id-only split prevents the same listing ID from crossing "
        "splits, but it can still allow comparable part identities to appear in both train and test "
        "when the same identity is represented by multiple product IDs.",
        "",
        "Canonicalization matches the earlier grouping reports: Unicode normalization, lowercase, "
        "trimmed whitespace, collapsed internal whitespace, missing-value sentinel, and no fuzzy matching.",
        "",
        "## Summary",
        "",
        markdown_table(summary_table),
        "",
        "## Interpretation",
        "",
        "- `multi_product_id_identities` are identities that could cross train/test boundaries under product-id-only splitting.",
        "- `product_ids_in_multi_identities` estimates how much of the product-id population is exposed to this risk.",
        "- More restrictive identity keys reduce the measured leakage surface, but may also hide leakage if the added field is noisy or fragments true comparable identities.",
        "",
    ]

    for output in outputs:
        lines.extend(
            [
                f"## {output.strategy.name}",
                "",
                f"Full repeated-identity table: `{output.repeated_path.as_posix()}`",
                f"Product-id-count distribution: `{output.distribution_path.as_posix()}`",
                f"Largest shared identities CSV: `{output.largest_path.as_posix()}`",
                "",
                "### Distribution Of Repeated Identities",
                "",
                markdown_table(output.distribution),
                "",
                f"### Largest Shared Identities Top {top_n}",
                "",
                markdown_table(top_largest_table(output, top_n)),
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    validate_columns(frame)

    outputs = [
        summarize_strategy(frame, strategy, artifact_dir)
        for strategy in IDENTITY_STRATEGIES
    ]

    pd.DataFrame([output.summary for output in outputs]).to_csv(
        artifact_dir / "identity_leakage_summary.csv",
        index=False,
    )
    write_report(
        input_path=input_path,
        output_path=output_path,
        row_count=len(frame),
        outputs=outputs,
        top_n=args.top_n,
    )

    print(f"Wrote product-id identity leakage report to: {output_path}")
    print(f"Wrote CSV artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
