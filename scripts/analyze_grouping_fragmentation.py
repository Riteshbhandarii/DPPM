#!/usr/bin/env python3
"""
Purpose:
Compare whether candidate grouping strategies merge observations or fragment existing groups.

Inputs:
- datasets/cleaned/clean_master_dataset.csv by default

Outputs:
- Markdown report summarizing fragmentation and merging behavior
- CSV artifacts with transition summaries, split-group summaries, and representative examples

Assumptions:
- This is a structural dataset diagnostic only. It does not train models, create splits,
  or modify thesis methodology.
- Canonicalization is imported from analyze_grouping_strategies.py so this report uses
  the same grouping keys as the initial grouping-size analysis.
- Product-id groups and canonical identity groups are not nested, so transitions involving
  product_id are interpreted as crosswalk diagnostics rather than pure refinements.
- OEM and compatibility-year effects are isolated by comparing each refinement against
  canonical(part_name, brand, model).

How to run:
python3 scripts/analyze_grouping_fragmentation.py
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


BASE_STRATEGY = GroupingStrategy(
    name="canonical(part_name, brand, model)",
    slug="part_name_brand_model",
    columns=("part_name", "brand", "model"),
)
PRODUCT_ID_STRATEGY = GroupingStrategy(
    name="product_id",
    slug="product_id",
    columns=("product_id",),
)
OEM_STRATEGY = GroupingStrategy(
    name="canonical(part_name, brand, model, oem_number)",
    slug="part_name_brand_model_oem_number",
    columns=("part_name", "brand", "model", "oem_number"),
)
YEAR_STRATEGY = GroupingStrategy(
    name="canonical(part_name, brand, model, year_start, year_end)",
    slug="part_name_brand_model_year_start_year_end",
    columns=("part_name", "brand", "model", "year_start", "year_end"),
)

KEY_SEPARATOR = "\x1f"


@dataclass(frozen=True)
class Transition:
    name: str
    slug: str
    previous: GroupingStrategy
    current: GroupingStrategy
    interpretation: str


ORDERED_TRANSITIONS = [
    Transition(
        name="product_id -> canonical(part_name, brand, model)",
        slug="product_id_to_part_name_brand_model",
        previous=PRODUCT_ID_STRATEGY,
        current=BASE_STRATEGY,
        interpretation=(
            "Crosswalk between repeated listing IDs and broad part identity. "
            "This is not a nested refinement."
        ),
    ),
    Transition(
        name="canonical(part_name, brand, model) -> + oem_number",
        slug="part_name_brand_model_to_oem_number",
        previous=BASE_STRATEGY,
        current=OEM_STRATEGY,
        interpretation="Nested refinement that isolates the effect of adding OEM.",
    ),
    Transition(
        name="+ oem_number -> + compatibility years",
        slug="oem_number_to_compatibility_years",
        previous=OEM_STRATEGY,
        current=YEAR_STRATEGY,
        interpretation=(
            "Ordered candidate-to-candidate comparison. This both removes OEM and adds "
            "compatibility years, so it should not be interpreted as the isolated effect of years."
        ),
    ),
]

FIELD_ADDITION_TRANSITIONS = [
    Transition(
        name="canonical(part_name, brand, model) -> + oem_number",
        slug="base_to_oem_number",
        previous=BASE_STRATEGY,
        current=OEM_STRATEGY,
        interpretation="Isolated OEM effect.",
    ),
    Transition(
        name="canonical(part_name, brand, model) -> + compatibility years",
        slug="base_to_compatibility_years",
        previous=BASE_STRATEGY,
        current=YEAR_STRATEGY,
        interpretation="Isolated compatibility-year effect.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how grouping strategies merge or fragment observations."
    )
    parser.add_argument(
        "--input-path",
        default="datasets/cleaned/clean_master_dataset.csv",
        help="Cleaned master dataset to analyze.",
    )
    parser.add_argument(
        "--output-path",
        default="results/grouping_strategy_fragmentation.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="results/grouping_strategy_fragmentation",
        help="Directory for CSV artifacts.",
    )
    parser.add_argument(
        "--large-group-threshold",
        type=int,
        default=20,
        help="Minimum previous-group size counted as a large group.",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=20,
        help="Number of representative examples to report for OEM and year refinements.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame, strategies: list[GroupingStrategy]) -> None:
    required_columns = sorted({column for strategy in strategies for column in strategy.columns})
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Required grouping columns are missing from the input dataset: {missing_columns}")


def canonical_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: canonicalize_series(frame[column]) for column in columns},
        index=frame.index,
    )


def key_series(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    canonical = canonical_columns(frame, columns)
    return canonical.astype("string").agg(KEY_SEPARATOR.join, axis=1)


def transition_crosswalk(frame: pd.DataFrame, transition: Transition) -> pd.DataFrame:
    previous_key = key_series(frame, transition.previous.columns)
    current_key = key_series(frame, transition.current.columns)
    crosswalk = (
        pd.DataFrame({"previous_key": previous_key, "current_key": current_key})
        .groupby(["previous_key", "current_key"], dropna=False)
        .size()
        .reset_index(name="overlap_rows")
    )

    previous_sizes = previous_key.value_counts().rename("previous_size")
    current_sizes = current_key.value_counts().rename("current_size")
    crosswalk = crosswalk.join(previous_sizes, on="previous_key")
    crosswalk = crosswalk.join(current_sizes, on="current_key")
    return crosswalk


def summarize_transition(
    frame: pd.DataFrame,
    transition: Transition,
    large_group_threshold: int,
) -> dict[str, object]:
    crosswalk = transition_crosswalk(frame, transition)
    current_children_per_previous = crosswalk.groupby("previous_key")["current_key"].nunique()
    previous_parents_per_current = crosswalk.groupby("current_key")["previous_key"].nunique()
    previous_sizes = crosswalk.drop_duplicates("previous_key").set_index("previous_key")[
        "previous_size"
    ]

    split_previous_keys = current_children_per_previous[current_children_per_previous > 1].index
    large_previous_keys = previous_sizes[previous_sizes >= large_group_threshold].index
    split_large_previous_keys = split_previous_keys.intersection(large_previous_keys)

    new_singleton_current_groups = crosswalk.loc[
        (crosswalk["current_size"] == 1) & (crosswalk["previous_size"] > 1),
        "current_key",
    ].nunique()
    smaller_fragments = int((crosswalk["overlap_rows"] < crosswalk["previous_size"]).sum())

    return {
        "comparison": transition.name,
        "interpretation": transition.interpretation,
        "previous_groups": int(crosswalk["previous_key"].nunique()),
        "current_groups": int(crosswalk["current_key"].nunique()),
        "previous_groups_split_or_smaller": int(len(split_previous_keys)),
        "resulting_smaller_fragments": smaller_fragments,
        "new_singleton_groups": int(new_singleton_current_groups),
        f"large_previous_groups_split_n_ge_{large_group_threshold}": int(
            len(split_large_previous_keys)
        ),
        "current_groups_merging_previous_groups": int((previous_parents_per_current > 1).sum()),
        "rows_in_split_previous_groups": int(previous_sizes.loc[split_previous_keys].sum())
        if len(split_previous_keys)
        else 0,
    }


def split_group_summary(
    frame: pd.DataFrame,
    transition: Transition,
    added_columns: tuple[str, ...],
) -> pd.DataFrame:
    all_columns = tuple(dict.fromkeys((*transition.previous.columns, *added_columns)))
    canonical = canonical_columns(frame, all_columns)
    previous_key = canonical[list(transition.previous.columns)].agg(KEY_SEPARATOR.join, axis=1)
    current_key = canonical[list(transition.current.columns)].agg(KEY_SEPARATOR.join, axis=1)

    working = canonical.copy()
    working["previous_key"] = previous_key
    working["current_key"] = current_key
    working["previous_size"] = working.groupby("previous_key")["previous_key"].transform("size")
    working["current_size"] = working.groupby("current_key")["current_key"].transform("size")
    working["refined_group_count"] = working.groupby("previous_key")["current_key"].transform(
        "nunique"
    )

    child_summary = (
        working.drop_duplicates("current_key")
        .groupby("previous_key", dropna=False)
        .agg(
            previous_size=("previous_size", "first"),
            refined_group_count=("current_key", "nunique"),
            largest_refined_group=("current_size", "max"),
            smallest_refined_group=("current_size", "min"),
            singleton_refined_groups=("current_size", lambda values: int((values == 1).sum())),
        )
        .reset_index()
    )
    split_previous = child_summary[child_summary["refined_group_count"] > 1].copy()

    previous_values = (
        canonical[list(transition.previous.columns)]
        .assign(previous_key=previous_key)
        .drop_duplicates("previous_key")
    )
    split_previous = split_previous.merge(previous_values, on="previous_key", how="left")
    ordered_columns = [
        *transition.previous.columns,
        "previous_size",
        "refined_group_count",
        "largest_refined_group",
        "smallest_refined_group",
        "singleton_refined_groups",
    ]
    return split_previous[ordered_columns].sort_values(
        ["previous_size", "refined_group_count", "singleton_refined_groups"],
        ascending=[False, False, False],
    )


def representative_examples(
    frame: pd.DataFrame,
    transition: Transition,
    added_columns: tuple[str, ...],
    example_count: int,
    max_examples_per_parent: int = 3,
) -> pd.DataFrame:
    all_columns = tuple(
        dict.fromkeys(
            (
                *transition.previous.columns,
                *transition.current.columns,
                "product_id",
                "price",
                "year_start",
                "year_end",
                "oem_number",
            )
        )
    )
    available_columns = tuple(column for column in all_columns if column in frame.columns)
    canonical = canonical_columns(frame, available_columns)

    previous_key = canonical[list(transition.previous.columns)].agg(KEY_SEPARATOR.join, axis=1)
    current_key = canonical[list(transition.current.columns)].agg(KEY_SEPARATOR.join, axis=1)
    working = frame.reset_index(names="row_index").copy()
    for column in available_columns:
        working[f"canonical_{column}"] = canonical[column].to_numpy()

    working["previous_key"] = previous_key.to_numpy()
    working["current_key"] = current_key.to_numpy()
    working["previous_group_size"] = working.groupby("previous_key")["previous_key"].transform(
        "size"
    )
    working["refined_group_size"] = working.groupby("current_key")["current_key"].transform("size")
    working["refined_groups_within_previous"] = working.groupby("previous_key")[
        "current_key"
    ].transform("nunique")
    working["singleton_created"] = (
        (working["refined_group_size"] == 1) & (working["previous_group_size"] > 1)
    )

    child_representatives = (
        working[working["refined_groups_within_previous"] > 1]
        .sort_values(
            [
                "previous_group_size",
                "refined_groups_within_previous",
                "refined_group_size",
                "row_index",
            ],
            ascending=[False, False, False, True],
        )
        .drop_duplicates("current_key")
    )

    rows = []
    parent_order = (
        child_representatives.drop_duplicates("previous_key")
        .sort_values(
            ["previous_group_size", "refined_groups_within_previous"],
            ascending=[False, False],
        )["previous_key"]
        .tolist()
    )
    for parent_key in parent_order:
        children = child_representatives[child_representatives["previous_key"] == parent_key]
        largest = children.sort_values(
            ["refined_group_size", "row_index"], ascending=[False, True]
        )
        smallest = children.sort_values(
            ["refined_group_size", "row_index"], ascending=[True, True]
        )
        selected = pd.concat([largest.head(2), smallest.head(1)]).drop_duplicates("current_key")
        rows.extend(selected.head(max_examples_per_parent).to_dict("records"))
        if len(rows) >= example_count:
            break

    examples = pd.DataFrame(rows).head(example_count)
    base_columns = [
        "row_index",
        "product_id",
        "part_name",
        "brand",
        "model",
        *added_columns,
        "year_start",
        "year_end",
        "oem_number",
        "price",
        "previous_group_size",
        "refined_group_size",
        "refined_groups_within_previous",
        "singleton_created",
    ]
    output_columns = []
    for column in base_columns:
        if column in examples.columns and column not in output_columns:
            output_columns.append(column)
    return examples[output_columns]


def top_rows(frame: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    return frame.head(limit).reset_index(drop=True)


def write_report(
    input_path: Path,
    output_path: Path,
    artifact_dir: Path,
    row_count: int,
    large_group_threshold: int,
    ordered_summary: pd.DataFrame,
    field_summary: pd.DataFrame,
    oem_split_groups: pd.DataFrame,
    year_split_groups: pd.DataFrame,
    oem_examples: pd.DataFrame,
    year_examples: pd.DataFrame,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    oem_effect = field_summary[field_summary["comparison"].str.contains("oem_number")].iloc[0]
    year_effect = field_summary[
        field_summary["comparison"].str.contains("compatibility years")
    ].iloc[0]
    lines = [
        "# Grouping Strategy Fragmentation Analysis",
        "",
        f"Generated: {generated_at}",
        "",
        f"Input dataset: `{input_path.as_posix()}`",
        f"Rows analyzed: {row_count:,}",
        "",
        "This report compares whether candidate grouping strategies merge observations or fragment "
        "existing groups. It is a structural diagnostic only: no machine learning models are trained "
        "and no split files are created.",
        "",
        "Definitions used in the tables:",
        "",
        "- `previous_groups_split_or_smaller`: previous groups whose rows are distributed across more than one current group.",
        "- `resulting_smaller_fragments`: previous/current overlap fragments that are smaller than their previous group.",
        "- `new_singleton_groups`: current singleton groups created from a previous group that had more than one row.",
        f"- `large_previous_groups_split_n_ge_{large_group_threshold}`: previous groups with at least {large_group_threshold} rows that split.",
        "- `current_groups_merging_previous_groups`: current groups containing rows from more than one previous group.",
        "",
        "Important interpretation note: `product_id` and canonical identity groups are not nested. "
        "The product-id comparison is therefore a crosswalk diagnostic. OEM and compatibility-year "
        "effects are interpreted using isolated comparisons against `canonical(part_name, brand, model)`.",
        "",
        "## Main Findings",
        "",
        f"- Adding OEM increases group count from {int(oem_effect['previous_groups']):,} to "
        f"{int(oem_effect['current_groups']):,}, splits "
        f"{int(oem_effect['previous_groups_split_or_smaller']):,} base groups, and creates "
        f"{int(oem_effect['new_singleton_groups']):,} new singleton groups.",
        f"- Adding compatibility years increases group count from {int(year_effect['previous_groups']):,} to "
        f"{int(year_effect['current_groups']):,}, splits "
        f"{int(year_effect['previous_groups_split_or_smaller']):,} base groups, and creates "
        f"{int(year_effect['new_singleton_groups']):,} new singleton groups.",
        "- In this dataset, OEM fragments the broad part identity more aggressively than compatibility years. "
        "That does not automatically make OEM invalid, but it raises a stronger risk that OEM is acting as a "
        "noisy fragmentation key rather than a consistently meaningful identity boundary.",
        "",
        "## Ordered Candidate Transitions",
        "",
        markdown_table(ordered_summary),
        "",
        "## Isolated Field-Addition Effects",
        "",
        markdown_table(field_summary),
        "",
        "## Largest Base Groups Split By OEM",
        "",
        f"Full CSV: `{(artifact_dir / 'base_to_oem_number_split_groups.csv').as_posix()}`",
        "",
        markdown_table(top_rows(oem_split_groups)),
        "",
        "## Largest Base Groups Split By Compatibility Years",
        "",
        f"Full CSV: `{(artifact_dir / 'base_to_compatibility_years_split_groups.csv').as_posix()}`",
        "",
        markdown_table(top_rows(year_split_groups)),
        "",
        "## Representative Rows Where Adding OEM Changes Grouping",
        "",
        f"Full CSV: `{(artifact_dir / 'oem_fragmentation_examples.csv').as_posix()}`",
        "",
        markdown_table(oem_examples),
        "",
        "## Representative Rows Where Adding Compatibility Years Changes Grouping",
        "",
        f"Full CSV: `{(artifact_dir / 'compatibility_year_fragmentation_examples.csv').as_posix()}`",
        "",
        markdown_table(year_examples),
        "",
        "## Thesis Interpretation",
        "",
        "OEM and compatibility years both split broad part-identity groups, but the split statistics "
        "should be interpreted as evidence about grouping behavior rather than as proof that either "
        "field is a valid identity boundary. A field that creates many smaller or singleton groups may "
        "either capture meaningful compatibility distinctions or fragment otherwise comparable parts.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    validate_columns(frame, [PRODUCT_ID_STRATEGY, BASE_STRATEGY, OEM_STRATEGY, YEAR_STRATEGY])

    ordered_summary = pd.DataFrame(
        [
            summarize_transition(frame, transition, args.large_group_threshold)
            for transition in ORDERED_TRANSITIONS
        ]
    )
    field_summary = pd.DataFrame(
        [
            summarize_transition(frame, transition, args.large_group_threshold)
            for transition in FIELD_ADDITION_TRANSITIONS
        ]
    )

    oem_transition = FIELD_ADDITION_TRANSITIONS[0]
    year_transition = FIELD_ADDITION_TRANSITIONS[1]
    oem_split_groups = split_group_summary(frame, oem_transition, ("oem_number",))
    year_split_groups = split_group_summary(frame, year_transition, ("year_start", "year_end"))
    oem_examples = representative_examples(
        frame,
        oem_transition,
        ("oem_number",),
        example_count=args.example_count,
    )
    year_examples = representative_examples(
        frame,
        year_transition,
        ("year_start", "year_end"),
        example_count=args.example_count,
    )

    ordered_summary.to_csv(artifact_dir / "ordered_transition_summary.csv", index=False)
    field_summary.to_csv(artifact_dir / "isolated_field_addition_summary.csv", index=False)
    oem_split_groups.to_csv(artifact_dir / "base_to_oem_number_split_groups.csv", index=False)
    year_split_groups.to_csv(
        artifact_dir / "base_to_compatibility_years_split_groups.csv",
        index=False,
    )
    oem_examples.to_csv(artifact_dir / "oem_fragmentation_examples.csv", index=False)
    year_examples.to_csv(
        artifact_dir / "compatibility_year_fragmentation_examples.csv",
        index=False,
    )

    write_report(
        input_path=input_path,
        output_path=output_path,
        artifact_dir=artifact_dir,
        row_count=len(frame),
        large_group_threshold=args.large_group_threshold,
        ordered_summary=ordered_summary,
        field_summary=field_summary,
        oem_split_groups=oem_split_groups,
        year_split_groups=year_split_groups,
        oem_examples=oem_examples,
        year_examples=year_examples,
    )

    print(f"Wrote grouping fragmentation report to: {output_path}")
    print(f"Wrote CSV artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
