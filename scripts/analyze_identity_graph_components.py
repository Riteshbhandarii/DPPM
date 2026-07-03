#!/usr/bin/env python3
"""
Purpose:
Analyze connected components formed by product_id links and candidate identity links.

Inputs:
- datasets/cleaned/clean_master_dataset.csv by default

Outputs:
- Markdown report summarizing component-size statistics
- CSV artifacts with component histograms and top component summaries
- Row-level CSV exports assigning every input row to a graph component

Assumptions:
- This is a structural dataset diagnostic only. It does not train models, create splits,
  or modify thesis methodology.
- Each row is a graph node.
- Rows are connected when they share the same product_id or the same canonical identity.
- Canonicalization is imported from analyze_grouping_strategies.py so this graph uses
  the same identity keys as the earlier grouping diagnostics.

How to run:
python3 scripts/analyze_identity_graph_components.py
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


GRAPH_STRATEGIES = [
    GroupingStrategy(
        name="product_id + canonical(part_name, brand, model)",
        slug="product_id_part_name_brand_model",
        columns=("part_name", "brand", "model"),
    ),
    GroupingStrategy(
        name="product_id + canonical(part_name, brand, model, year_start, year_end)",
        slug="product_id_part_name_brand_model_year_start_year_end",
        columns=("part_name", "brand", "model", "year_start", "year_end"),
    ),
]

KEY_SEPARATOR = "\x1f"
DISPLAY_SEPARATOR = " | "


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


@dataclass(frozen=True)
class GraphOutput:
    strategy: GroupingStrategy
    row_assignments: pd.DataFrame
    component_summary: pd.DataFrame
    histogram: pd.DataFrame
    stats: dict[str, object]
    row_path: Path
    component_path: Path
    histogram_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze connected components from product_id and identity links."
    )
    parser.add_argument(
        "--input-path",
        default="datasets/cleaned/clean_master_dataset.csv",
        help="Cleaned master dataset to analyze.",
    )
    parser.add_argument(
        "--output-path",
        default="results/identity_graph_components.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="results/identity_graph_components",
        help="Directory for component CSV artifacts.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of largest connected components to show in the markdown report.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame) -> None:
    required_columns = sorted(
        {"product_id", *{column for strategy in GRAPH_STRATEGIES for column in strategy.columns}}
    )
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Required columns are missing from the input dataset: {missing_columns}")


def canonical_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: canonicalize_series(frame[column]) for column in columns},
        index=frame.index,
    )


def combined_key(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    separator: str = KEY_SEPARATOR,
) -> pd.Series:
    return frame[list(columns)].astype("string").agg(separator.join, axis=1)


def union_by_key(union_find: UnionFind, keys: pd.Series) -> int:
    first_seen: dict[str, int] = {}
    edge_count = 0
    for row_position, key in enumerate(keys.astype(str)):
        first_row = first_seen.get(key)
        if first_row is None:
            first_seen[key] = row_position
        else:
            union_find.union(first_row, row_position)
            edge_count += 1
    return edge_count


def component_ids(union_find: UnionFind, prefix: str, row_count: int) -> pd.DataFrame:
    roots = pd.Series([union_find.find(index) for index in range(row_count)], name="root")
    component_sizes = roots.value_counts().rename_axis("root").reset_index(name="component_size")
    min_positions = roots.reset_index().groupby("root")["index"].min().rename("min_row_index")
    component_sizes = component_sizes.join(min_positions, on="root")
    component_sizes = component_sizes.sort_values(
        ["component_size", "min_row_index"],
        ascending=[False, True],
    ).reset_index(drop=True)
    component_sizes["component_id"] = [
        f"{prefix}_component_{index:06d}" for index in range(1, len(component_sizes) + 1)
    ]
    return roots.to_frame().merge(
        component_sizes[["root", "component_id", "component_size"]],
        on="root",
        how="left",
    )[["component_id", "component_size"]]


def product_id_sample(values: pd.Series, limit: int = 12) -> str:
    ordered = sorted(set(values.astype(str)))
    sample = ordered[:limit]
    suffix = "" if len(ordered) <= limit else f" ... (+{len(ordered) - limit} more)"
    return ", ".join(sample) + suffix


def identity_sample(values: pd.Series, limit: int = 8) -> str:
    ordered = sorted(set(values.astype(str)))
    sample = ordered[:limit]
    suffix = "" if len(ordered) <= limit else f" ... (+{len(ordered) - limit} more)"
    return " || ".join(sample) + suffix


def summarize_components(row_assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        row_assignments.groupby("component_id", dropna=False)
        .agg(
            component_size=("component_id", "size"),
            product_id_count=("canonical_product_id", "nunique"),
            identity_count=("identity_key", "nunique"),
            product_id_sample=("canonical_product_id", product_id_sample),
            identity_sample=("identity_key", identity_sample),
            min_row_index=("row_index", "min"),
        )
        .reset_index()
        .sort_values(
            ["component_size", "product_id_count", "identity_count", "min_row_index"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )


def component_histogram(component_summary: pd.DataFrame) -> pd.DataFrame:
    histogram = (
        component_summary["component_size"]
        .value_counts()
        .rename_axis("component_size")
        .reset_index(name="component_count")
        .sort_values("component_size")
        .reset_index(drop=True)
    )
    histogram["row_count"] = histogram["component_size"] * histogram["component_count"]
    histogram["component_pct"] = (
        histogram["component_count"] / histogram["component_count"].sum() * 100
    ).round(2)
    return histogram


def component_stats(component_summary: pd.DataFrame) -> dict[str, object]:
    sizes = component_summary["component_size"]
    return {
        "connected_components": int(len(component_summary)),
        "average_component_size": float(sizes.mean()),
        "median_component_size": float(sizes.median()),
        "maximum_component_size": int(sizes.max()),
        "minimum_component_size": int(sizes.min()),
        "singleton_components": int((sizes == 1).sum()),
        "singleton_component_pct": float((sizes == 1).mean() * 100),
    }


def analyze_graph(
    frame: pd.DataFrame,
    strategy: GroupingStrategy,
    artifact_dir: Path,
) -> GraphOutput:
    row_count = len(frame)
    union_find = UnionFind(row_count)

    product_id = canonicalize_series(frame["product_id"])
    identity_values = canonical_frame(frame, strategy.columns)
    identity = combined_key(identity_values, strategy.columns)
    identity_label = combined_key(identity_values, strategy.columns, DISPLAY_SEPARATOR)

    product_id_edge_count = union_by_key(union_find, product_id)
    identity_edge_count = union_by_key(union_find, identity)
    component_assignment = component_ids(union_find, strategy.slug, row_count)

    metadata = pd.DataFrame(
        {
            "component_id": component_assignment["component_id"],
            "component_size": component_assignment["component_size"],
            "canonical_product_id": product_id.to_numpy(),
            "identity_key": identity_label.to_numpy(),
            **{
                f"canonical_{column}": identity_values[column].to_numpy()
                for column in strategy.columns
            },
        }
    )
    row_assignments = pd.concat(
        [
            frame.reset_index(names="row_index")[["row_index"]],
            metadata,
            frame.reset_index(drop=True),
        ],
        axis=1,
    )

    component_summary = summarize_components(row_assignments)
    histogram = component_histogram(component_summary)
    stats = component_stats(component_summary)
    stats["product_id_union_links_used"] = int(product_id_edge_count)
    stats["identity_union_links_used"] = int(identity_edge_count)

    row_path = artifact_dir / f"{strategy.slug}_row_components.csv"
    component_path = artifact_dir / f"{strategy.slug}_component_summary.csv"
    histogram_path = artifact_dir / f"{strategy.slug}_component_size_histogram.csv"

    row_assignments.to_csv(row_path, index=False)
    component_summary.to_csv(component_path, index=False)
    histogram.to_csv(histogram_path, index=False)

    return GraphOutput(
        strategy=strategy,
        row_assignments=row_assignments,
        component_summary=component_summary,
        histogram=histogram,
        stats=stats,
        row_path=row_path,
        component_path=component_path,
        histogram_path=histogram_path,
    )


def format_stats(output: GraphOutput) -> dict[str, object]:
    stats = output.stats
    return {
        "graph": output.strategy.name,
        "connected_components": f"{stats['connected_components']:,}",
        "average_component_size": f"{stats['average_component_size']:.2f}",
        "median_component_size": f"{stats['median_component_size']:.2f}",
        "maximum_component_size": f"{stats['maximum_component_size']:,}",
        "minimum_component_size": f"{stats['minimum_component_size']:,}",
        "singleton_components": (
            f"{stats['singleton_components']:,} ({stats['singleton_component_pct']:.2f}%)"
        ),
    }


def write_report(
    input_path: Path,
    output_path: Path,
    row_count: int,
    outputs: list[GraphOutput],
    top_n: int,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Identity Graph Connected Component Analysis",
        "",
        f"Generated: {generated_at}",
        "",
        f"Input dataset: `{input_path.as_posix()}`",
        f"Rows analyzed: {row_count:,}",
        "",
        "Each row is treated as a node. Edges are created when two rows share the same canonical "
        "`product_id` or the same candidate identity. This report analyzes graph structure only; "
        "it does not create train/test splits.",
        "",
        "A component can contain more than one identity label when product_id links bridge small "
        "canonical identity differences. This is useful for leakage analysis because the connected "
        "component is the unit that would need to stay together to block both repeated-listing and "
        "identity-sharing paths.",
        "",
        "Canonicalization matches the earlier grouping reports: Unicode normalization, lowercase, "
        "trimmed whitespace, collapsed internal whitespace, missing-value sentinel, and no fuzzy matching.",
        "",
        "## Summary",
        "",
        markdown_table(pd.DataFrame([format_stats(output) for output in outputs])),
        "",
    ]

    for output in outputs:
        lines.extend(
            [
                f"## {output.strategy.name}",
                "",
                f"Row-level component export: `{output.row_path.as_posix()}`",
                f"Component summary CSV: `{output.component_path.as_posix()}`",
                f"Component-size histogram CSV: `{output.histogram_path.as_posix()}`",
                "",
                "### Component Size Histogram",
                "",
                markdown_table(output.histogram),
                "",
                f"### Top {top_n} Largest Connected Components",
                "",
                markdown_table(output.component_summary.head(top_n)),
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
        analyze_graph(frame, strategy, artifact_dir)
        for strategy in GRAPH_STRATEGIES
    ]
    pd.DataFrame([output.stats | {"graph": output.strategy.name} for output in outputs]).to_csv(
        artifact_dir / "graph_component_summary.csv",
        index=False,
    )
    write_report(
        input_path=input_path,
        output_path=output_path,
        row_count=len(frame),
        outputs=outputs,
        top_n=args.top_n,
    )

    print(f"Wrote identity graph component report to: {output_path}")
    print(f"Wrote CSV artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
