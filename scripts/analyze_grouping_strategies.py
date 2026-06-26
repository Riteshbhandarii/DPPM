#!/usr/bin/env python3
"""
Purpose:
Analyze candidate row-grouping strategies before choosing a thesis evaluation split protocol.

Inputs:
- datasets/cleaned/clean_master_dataset.csv by default

Outputs:
- Markdown report summarizing group-size statistics
- CSV artifacts with exact group-size histograms and top largest groups

Assumptions:
- This is a structural dataset diagnostic only. It does not train models, create splits,
  or modify thesis methodology.
- Canonical grouping values use Unicode normalization, lowercase text, trimmed and
  collapsed whitespace, and a documented missing-value sentinel.
- No fuzzy matching is performed.

How to run:
python3 scripts/analyze_grouping_strategies.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import numbers
from pathlib import Path
import re
import unicodedata

import pandas as pd


MISSING_SENTINEL = "__missing__"
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class GroupingStrategy:
    name: str
    slug: str
    columns: tuple[str, ...]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze candidate grouping strategies without training models."
    )
    parser.add_argument(
        "--input-path",
        default="datasets/cleaned/clean_master_dataset.csv",
        help="Cleaned master dataset to analyze.",
    )
    parser.add_argument(
        "--output-path",
        default="results/grouping_strategy_analysis.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="results/grouping_strategy_analysis",
        help="Directory for exact histogram and top-group CSV artifacts.",
    )
    return parser.parse_args()


def canonical_text(value: object) -> str:
    if pd.isna(value):
        return MISSING_SENTINEL

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            text = str(int(numeric_value))
        else:
            text = str(value)
    else:
        text = str(value)

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = WHITESPACE_RE.sub(" ", text.strip())
    return text if text else MISSING_SENTINEL


def canonicalize_series(series: pd.Series) -> pd.Series:
    return series.map(canonical_text).astype("string")


def markdown_escape(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:.2f}"
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(markdown_escape(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def validate_columns(frame: pd.DataFrame) -> None:
    required_columns = sorted(
        {column for strategy in GROUPING_STRATEGIES for column in strategy.columns}
    )
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Required grouping columns are missing from the input dataset: {missing_columns}")


def build_canonical_frame(frame: pd.DataFrame, strategy: GroupingStrategy) -> pd.DataFrame:
    return pd.DataFrame(
        {column: canonicalize_series(frame[column]) for column in strategy.columns},
        index=frame.index,
    )


def group_size_table(frame: pd.DataFrame, strategy: GroupingStrategy) -> pd.DataFrame:
    canonical_frame = build_canonical_frame(frame, strategy)
    grouped = (
        canonical_frame.groupby(list(strategy.columns), dropna=False)
        .size()
        .reset_index(name="group_size")
        .sort_values(
            ["group_size", *strategy.columns],
            ascending=[False, *([True] * len(strategy.columns))],
        )
        .reset_index(drop=True)
    )
    return grouped


def summary_statistics(group_sizes: pd.Series) -> dict[str, float | int]:
    group_count = int(len(group_sizes))
    singleton_count = int((group_sizes == 1).sum())
    return {
        "unique_groups": group_count,
        "average_group_size": float(group_sizes.mean()),
        "median_group_size": float(group_sizes.median()),
        "maximum_group_size": int(group_sizes.max()),
        "minimum_group_size": int(group_sizes.min()),
        "singleton_groups": singleton_count,
        "singleton_group_pct": float((singleton_count / group_count) * 100) if group_count else 0.0,
    }


def exact_histogram(group_sizes: pd.Series) -> pd.DataFrame:
    histogram = (
        group_sizes.value_counts()
        .rename_axis("group_size")
        .reset_index(name="group_count")
        .sort_values("group_size")
        .reset_index(drop=True)
    )
    histogram["row_count"] = histogram["group_size"] * histogram["group_count"]
    histogram["group_pct"] = (
        (histogram["group_count"] / histogram["group_count"].sum()) * 100
    ).round(2)
    return histogram


def format_summary_table(stats: dict[str, float | int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "unique groups", "value": f"{stats['unique_groups']:,}"},
            {"metric": "average group size", "value": f"{stats['average_group_size']:.2f}"},
            {"metric": "median group size", "value": f"{stats['median_group_size']:.2f}"},
            {"metric": "maximum group size", "value": f"{stats['maximum_group_size']:,}"},
            {"metric": "minimum group size", "value": f"{stats['minimum_group_size']:,}"},
            {"metric": "singleton groups", "value": f"{stats['singleton_groups']:,}"},
            {"metric": "singleton group percentage", "value": f"{stats['singleton_group_pct']:.2f}%"},
        ]
    )


def comparison_row(strategy: GroupingStrategy, stats: dict[str, float | int]) -> dict[str, object]:
    return {
        "strategy": strategy.name,
        "unique_groups": f"{stats['unique_groups']:,}",
        "avg_size": f"{stats['average_group_size']:.2f}",
        "median_size": f"{stats['median_group_size']:.2f}",
        "max_size": f"{stats['maximum_group_size']:,}",
        "min_size": f"{stats['minimum_group_size']:,}",
        "singleton_groups": f"{stats['singleton_groups']:,}",
        "singleton_pct": f"{stats['singleton_group_pct']:.2f}%",
    }


def missing_value_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = sorted({column for strategy in GROUPING_STRATEGIES for column in strategy.columns})
    rows = []
    for column in columns:
        missing_count = int(frame[column].isna().sum())
        rows.append(
            {
                "column": column,
                "missing_rows": f"{missing_count:,}",
                "missing_pct": f"{(missing_count / len(frame)) * 100:.2f}%",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    frame: pd.DataFrame,
    input_path: Path,
    output_path: Path,
    strategy_outputs: list[dict[str, object]],
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    input_display = input_path.as_posix()

    lines = [
        "# Grouping Strategy Structural Analysis",
        "",
        f"Generated: {generated_at}",
        "",
        f"Input dataset: `{input_display}`",
        f"Rows analyzed: {len(frame):,}",
        "",
        "This report compares candidate grouping keys for future train/test split design. "
        "It is a structural diagnostic only: no machine learning models are trained and no split files are created.",
        "",
        "Canonicalization applied to grouping values:",
        "",
        "- Unicode normalization: `NFKC`",
        "- lowercase text",
        "- trim leading and trailing whitespace",
        "- collapse repeated internal whitespace",
        f"- replace missing or empty values with `{MISSING_SENTINEL}`",
        "- no fuzzy matching",
        "",
        "## Missing Values In Grouping Columns",
        "",
        markdown_table(missing_value_table(frame)),
        "",
        "## Strategy Comparison",
        "",
        markdown_table(pd.DataFrame([output["comparison_row"] for output in strategy_outputs])),
        "",
    ]

    for output in strategy_outputs:
        strategy = output["strategy"]
        assert isinstance(strategy, GroupingStrategy)
        summary = output["summary"]
        histogram = output["histogram"]
        top_groups = output["top_groups"]
        histogram_path = output["histogram_path"]
        top_groups_path = output["top_groups_path"]

        assert isinstance(summary, pd.DataFrame)
        assert isinstance(histogram, pd.DataFrame)
        assert isinstance(top_groups, pd.DataFrame)
        assert isinstance(histogram_path, Path)
        assert isinstance(top_groups_path, Path)

        lines.extend(
            [
                f"## {strategy.name}",
                "",
                f"Grouping columns: `{', '.join(strategy.columns)}`",
                "",
                "### Summary Statistics",
                "",
                markdown_table(summary),
                "",
                "### Group Size Histogram",
                "",
                f"The exact histogram is also saved at `{histogram_path.as_posix()}`.",
                "",
                markdown_table(histogram),
                "",
                "### Top 20 Largest Groups",
                "",
                f"The same top-group table is saved at `{top_groups_path.as_posix()}`.",
                "",
                markdown_table(top_groups),
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

    frame = pd.read_csv(input_path)
    validate_columns(frame)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    strategy_outputs = []
    for strategy in GROUPING_STRATEGIES:
        groups = group_size_table(frame, strategy)
        sizes = groups["group_size"]
        stats = summary_statistics(sizes)
        histogram = exact_histogram(sizes)
        top_groups = groups.head(20).copy()

        histogram_path = artifact_dir / f"{strategy.slug}_histogram.csv"
        top_groups_path = artifact_dir / f"{strategy.slug}_top_20_groups.csv"
        histogram.to_csv(histogram_path, index=False)
        top_groups.to_csv(top_groups_path, index=False)

        strategy_outputs.append(
            {
                "strategy": strategy,
                "comparison_row": comparison_row(strategy, stats),
                "summary": format_summary_table(stats),
                "histogram": histogram,
                "top_groups": top_groups,
                "histogram_path": histogram_path,
                "top_groups_path": top_groups_path,
            }
        )

    write_report(frame, input_path, output_path, strategy_outputs)
    print(f"Wrote grouping strategy report to: {output_path}")
    print(f"Wrote CSV artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
