#!/usr/bin/env python3
"""
Purpose:
Measure target-price variability within candidate identity/grouping definitions.

Inputs:
- datasets/cleaned/clean_master_dataset.csv by default

Outputs:
- Markdown report summarizing within-group price dispersion
- CSV artifacts with full per-group variability tables and high/low variance examples

Assumptions:
- This is a structural dataset diagnostic only. It does not train models, create splits,
  or modify thesis methodology.
- Singleton groups are reported separately and excluded from average/median variance
  summaries because their within-group variance cannot be estimated from one observation.
- Canonicalization is imported from analyze_grouping_strategies.py so this report uses
  the same grouping keys as the earlier diagnostics.

How to run:
python3 scripts/analyze_group_price_variability.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_grouping_strategies import (
    GroupingStrategy,
    canonicalize_series,
    markdown_table,
)


GROUPING_STRATEGIES = [
    GroupingStrategy(
        name="product_id",
        slug="product_id",
        columns=("product_id",),
    ),
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


@dataclass(frozen=True)
class VariabilityOutput:
    strategy: GroupingStrategy
    group_stats: pd.DataFrame
    summary: dict[str, object]
    full_path: Path
    highest_path: Path
    lowest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze within-group target-price variability for candidate identities."
    )
    parser.add_argument(
        "--input-path",
        default="datasets/cleaned/clean_master_dataset.csv",
        help="Cleaned master dataset to analyze.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target price column. Defaults to price if present, otherwise listing_price.",
    )
    parser.add_argument(
        "--output-path",
        default="results/group_price_variability.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="results/group_price_variability",
        help="Directory for per-group variability CSV artifacts.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of highest/lowest variance identities to show in the markdown report.",
    )
    return parser.parse_args()


def resolve_target_column(frame: pd.DataFrame, requested_column: str | None) -> str:
    if requested_column:
        if requested_column not in frame.columns:
            raise KeyError(f"Requested target column is missing: {requested_column}")
        return requested_column

    for column in ("price", "listing_price"):
        if column in frame.columns:
            return column
    raise KeyError("Could not find a target column. Expected either 'price' or 'listing_price'.")


def validate_columns(frame: pd.DataFrame, target_column: str) -> None:
    required_columns = sorted(
        {
            target_column,
            *{column for strategy in GROUPING_STRATEGIES for column in strategy.columns},
        }
    )
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Required columns are missing from the input dataset: {missing_columns}")


def canonical_group_frame(frame: pd.DataFrame, strategy: GroupingStrategy) -> pd.DataFrame:
    return pd.DataFrame(
        {column: canonicalize_series(frame[column]) for column in strategy.columns},
        index=frame.index,
    )


def coefficient_of_variation(std_price: pd.Series, mean_price: pd.Series) -> pd.Series:
    mean = mean_price.replace(0, np.nan)
    return std_price / mean


def group_price_stats(
    frame: pd.DataFrame,
    strategy: GroupingStrategy,
    target_column: str,
) -> pd.DataFrame:
    canonical = canonical_group_frame(frame, strategy)
    working = canonical.copy()
    working["target_price"] = pd.to_numeric(frame[target_column], errors="coerce")
    working = working.dropna(subset=["target_price"])

    rows = []
    grouped = working.groupby(list(strategy.columns), dropna=False, observed=False)
    for key_values, group in grouped:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        prices = group["target_price"].astype(float)
        listing_count = int(len(prices))

        # Singletons have no observable within-group dispersion. Keeping variance
        # missing avoids making over-strict grouping definitions look artificially good.
        if listing_count >= 2:
            std_price = float(prices.std(ddof=0))
            price_variance = float(prices.var(ddof=0))
            cv = float(std_price / prices.mean()) if prices.mean() != 0 else np.nan
        else:
            std_price = np.nan
            price_variance = np.nan
            cv = np.nan

        rows.append(
            {
                **dict(zip(strategy.columns, key_values, strict=True)),
                "listing_count": listing_count,
                "mean_price": float(prices.mean()),
                "median_price": float(prices.median()),
                "std_price": std_price,
                "price_variance": price_variance,
                "coefficient_of_variation": cv,
                "min_price": float(prices.min()),
                "max_price": float(prices.max()),
                "price_range": float(prices.max() - prices.min()),
            }
        )

    group_stats = pd.DataFrame(rows)
    return group_stats.sort_values(
        ["price_variance", "listing_count", *strategy.columns],
        ascending=[False, False, *([True] * len(strategy.columns))],
        na_position="last",
    ).reset_index(drop=True)


def summarize_variability(group_stats: pd.DataFrame, strategy: GroupingStrategy) -> dict[str, object]:
    non_singleton = group_stats[group_stats["listing_count"] >= 2]
    singleton_count = int((group_stats["listing_count"] == 1).sum())
    group_count = int(len(group_stats))

    return {
        "strategy": strategy.name,
        "group_count": group_count,
        "singleton_groups": singleton_count,
        "singleton_group_pct": (singleton_count / group_count * 100) if group_count else 0.0,
        "non_singleton_groups": int(len(non_singleton)),
        "average_group_size": float(group_stats["listing_count"].mean()),
        "median_group_size": float(group_stats["listing_count"].median()),
        "average_within_group_variance": float(non_singleton["price_variance"].mean())
        if not non_singleton.empty
        else np.nan,
        "median_within_group_variance": float(non_singleton["price_variance"].median())
        if not non_singleton.empty
        else np.nan,
        "average_coefficient_of_variation": float(
            non_singleton["coefficient_of_variation"].mean()
        )
        if not non_singleton.empty
        else np.nan,
        "median_coefficient_of_variation": float(
            non_singleton["coefficient_of_variation"].median()
        )
        if not non_singleton.empty
        else np.nan,
        "maximum_within_group_variance": float(non_singleton["price_variance"].max())
        if not non_singleton.empty
        else np.nan,
        "maximum_coefficient_of_variation": float(
            non_singleton["coefficient_of_variation"].max()
        )
        if not non_singleton.empty
        else np.nan,
    }


def round_report_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in [
        "mean_price",
        "median_price",
        "std_price",
        "price_variance",
        "coefficient_of_variation",
        "min_price",
        "max_price",
        "price_range",
    ]:
        if column in output.columns:
            output[column] = output[column].round(4)
    return output


def format_group_table_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    two_decimal_columns = [
        "mean_price",
        "median_price",
        "std_price",
        "min_price",
        "max_price",
        "price_range",
    ]
    four_decimal_columns = ["price_variance", "coefficient_of_variation"]

    for column in two_decimal_columns:
        if column in output.columns:
            output[column] = output[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
            )
    for column in four_decimal_columns:
        if column in output.columns:
            output[column] = output[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    return output


def highest_variance(group_stats: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return round_report_columns(
        group_stats[group_stats["listing_count"] >= 2]
        .sort_values(["price_variance", "listing_count"], ascending=[False, False])
        .head(top_n)
    )


def lowest_variance(group_stats: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return round_report_columns(
        group_stats[group_stats["listing_count"] >= 2]
        .sort_values(["price_variance", "listing_count"], ascending=[True, False])
        .head(top_n)
    )


def format_summary_row(summary: dict[str, object]) -> dict[str, object]:
    return {
        "strategy": summary["strategy"],
        "groups": f"{summary['group_count']:,}",
        "singletons": f"{summary['singleton_groups']:,} ({summary['singleton_group_pct']:.2f}%)",
        "non_singleton_groups": f"{summary['non_singleton_groups']:,}",
        "avg_size": f"{summary['average_group_size']:.2f}",
        "median_size": f"{summary['median_group_size']:.2f}",
        "avg_variance": f"{summary['average_within_group_variance']:.2f}",
        "median_variance": f"{summary['median_within_group_variance']:.2f}",
        "avg_cv": f"{summary['average_coefficient_of_variation']:.4f}",
        "median_cv": f"{summary['median_coefficient_of_variation']:.4f}",
        "max_variance": f"{summary['maximum_within_group_variance']:.2f}",
        "max_cv": f"{summary['maximum_coefficient_of_variation']:.4f}",
    }


def analyze_strategy(
    frame: pd.DataFrame,
    strategy: GroupingStrategy,
    target_column: str,
    artifact_dir: Path,
    top_n: int,
) -> VariabilityOutput:
    group_stats = group_price_stats(frame, strategy, target_column)
    summary = summarize_variability(group_stats, strategy)
    highest = highest_variance(group_stats, top_n)
    lowest = lowest_variance(group_stats, top_n)

    full_path = artifact_dir / f"{strategy.slug}_price_variability.csv"
    highest_path = artifact_dir / f"{strategy.slug}_highest_variance.csv"
    lowest_path = artifact_dir / f"{strategy.slug}_lowest_variance.csv"

    round_report_columns(group_stats).to_csv(full_path, index=False)
    highest.to_csv(highest_path, index=False)
    lowest.to_csv(lowest_path, index=False)

    return VariabilityOutput(
        strategy=strategy,
        group_stats=group_stats,
        summary=summary,
        full_path=full_path,
        highest_path=highest_path,
        lowest_path=lowest_path,
    )


def write_report(
    input_path: Path,
    output_path: Path,
    target_column: str,
    row_count: int,
    outputs: list[VariabilityOutput],
    top_n: int,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_table = pd.DataFrame([format_summary_row(output.summary) for output in outputs])

    lines = [
        "# Group-Level Target Price Variability",
        "",
        f"Generated: {generated_at}",
        "",
        f"Input dataset: `{input_path.as_posix()}`",
        f"Rows analyzed: {row_count:,}",
        f"Target column: `{target_column}`",
        "",
        "This report measures whether candidate grouping definitions collect observations with "
        "similar target prices. High within-group variance suggests a grouping rule may be too coarse. "
        "A high singleton rate suggests a grouping rule may be too strict for evaluation or modeling "
        "diagnostics.",
        "",
        "Variance, standard deviation, and coefficient of variation are computed only for groups with "
        "at least two observations. Singleton groups are reported separately rather than assigned zero "
        "variance.",
        "",
        "## Summary",
        "",
        markdown_table(summary_table),
        "",
    ]

    for output in outputs:
        lines.extend(
            [
                f"## {output.strategy.name}",
                "",
                f"Full per-group table: `{output.full_path.as_posix()}`",
                f"Highest variance CSV: `{output.highest_path.as_posix()}`",
                f"Lowest variance CSV: `{output.lowest_path.as_posix()}`",
                "",
                f"### Highest Variance Identities Top {top_n}",
                "",
                markdown_table(
                    format_group_table_for_markdown(highest_variance(output.group_stats, top_n))
                ),
                "",
                f"### Lowest Variance Identities Top {top_n}",
                "",
                "Singleton groups are excluded from this table.",
                "",
                markdown_table(
                    format_group_table_for_markdown(lowest_variance(output.group_stats, top_n))
                ),
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
    target_column = resolve_target_column(frame, args.target_column)
    validate_columns(frame, target_column)

    outputs = [
        analyze_strategy(frame, strategy, target_column, artifact_dir, args.top_n)
        for strategy in GROUPING_STRATEGIES
    ]
    pd.DataFrame([output.summary for output in outputs]).to_csv(
        artifact_dir / "group_price_variability_summary.csv",
        index=False,
    )
    write_report(
        input_path=input_path,
        output_path=output_path,
        target_column=target_column,
        row_count=len(frame),
        outputs=outputs,
        top_n=args.top_n,
    )

    print(f"Wrote group price variability report to: {output_path}")
    print(f"Wrote CSV artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
