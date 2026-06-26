#!/usr/bin/env python3
"""
Purpose:
Diagnose whether the proposed final thesis connected-component split can produce
balanced train/validation/test partitions.

Inputs:
- datasets/cleaned/clean_master_dataset.csv by default

Outputs:
- results/candidate_component_split_balance.md
- CSV diagnostic artifacts under results/candidate_component_split_balance/

Assumptions:
- This script is diagnostic only. It does not train models and does not create final
  train.csv, validation.csv, or test.csv files.
- Rows are connected when they share product_id or the canonical part identity:
  part_name + brand + model + year_start + year_end.
- Candidate split assignment is simulated at connected-component level.
- Canonicalization uses Unicode normalization, casefolding, trimmed/collapsed whitespace,
  and the documented missing-value sentinel. No fuzzy matching is performed.

How to run:
python3 scripts/analyze_candidate_component_split_balance.py
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

import numpy as np
import pandas as pd


MISSING_SENTINEL = "__missing__"
WHITESPACE_RE = re.compile(r"\s+")
KEY_SEPARATOR = "\x1f"
DISPLAY_SEPARATOR = " | "
IDENTITY_COLUMNS = ("part_name", "brand", "model", "year_start", "year_end")
SPLITS = ("train", "validation", "test")
TARGET_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


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
class SimulationResult:
    seed: int
    row_summary: pd.DataFrame
    component_summary: pd.DataFrame
    price_summary: pd.DataFrame
    leakage_summary: pd.DataFrame
    score_row_balance: float
    score_price_balance: float
    score_combined: float
    max_row_pct_deviation: float
    assignments: pd.DataFrame
    rows: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose balance of proposed connected-component train/validation/test splits."
    )
    parser.add_argument(
        "--input-path",
        default="datasets/cleaned/clean_master_dataset.csv",
        help="Cleaned master dataset to analyze.",
    )
    parser.add_argument(
        "--output-path",
        default="results/candidate_component_split_balance.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="results/candidate_component_split_balance",
        help="Directory for diagnostic CSV artifacts.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target price column. Defaults to price if present, otherwise listing_price.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100,
        help="Number of random seeds to simulate. Must be at least 100 for thesis diagnostics.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First random seed to evaluate.",
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
    text = text.casefold()
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
        return f"{numeric_value:.4f}"
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


def resolve_target_column(frame: pd.DataFrame, requested_column: str | None) -> str:
    if requested_column:
        if requested_column not in frame.columns:
            raise KeyError(f"Requested target column is missing: {requested_column}")
        return requested_column

    for column in ("price", "listing_price"):
        if column in frame.columns:
            return column
    raise KeyError("Could not find a target column. Expected 'price' or 'listing_price'.")


def validate_columns(frame: pd.DataFrame, target_column: str) -> None:
    required = {"product_id", target_column, *IDENTITY_COLUMNS}
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise KeyError(f"Required columns are missing from the input dataset: {missing}")


def key_from_columns(frame: pd.DataFrame, columns: tuple[str, ...], separator: str) -> pd.Series:
    return frame[list(columns)].astype("string").agg(separator.join, axis=1)


def union_by_key(union_find: UnionFind, keys: pd.Series) -> None:
    first_seen: dict[str, int] = {}
    for row_position, key in enumerate(keys.astype(str)):
        first_row = first_seen.get(key)
        if first_row is None:
            first_seen[key] = row_position
        else:
            union_find.union(first_row, row_position)


def build_component_frame(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    output = frame.reset_index(names="row_index").copy()
    output["canonical_product_id"] = canonicalize_series(frame["product_id"]).to_numpy()

    for column in IDENTITY_COLUMNS:
        output[f"canonical_{column}"] = canonicalize_series(frame[column]).to_numpy()

    canonical_identity_columns = tuple(f"canonical_{column}" for column in IDENTITY_COLUMNS)
    output["identity_key"] = key_from_columns(output, canonical_identity_columns, KEY_SEPARATOR)
    output["identity_label"] = key_from_columns(output, canonical_identity_columns, DISPLAY_SEPARATOR)
    output["target_price"] = pd.to_numeric(frame[target_column], errors="coerce")

    union_find = UnionFind(len(output))
    union_by_key(union_find, output["canonical_product_id"])
    union_by_key(union_find, output["identity_key"])

    roots = pd.Series([union_find.find(index) for index in range(len(output))], name="root")
    root_sizes = roots.value_counts().rename_axis("root").reset_index(name="component_size")
    root_min_rows = roots.reset_index().groupby("root")["index"].min().rename("min_row_index")
    root_sizes = root_sizes.join(root_min_rows, on="root")
    root_sizes = root_sizes.sort_values(
        ["component_size", "min_row_index"],
        ascending=[False, True],
    ).reset_index(drop=True)
    root_sizes["component_id"] = [
        f"candidate_component_{index:06d}" for index in range(1, len(root_sizes) + 1)
    ]

    component_lookup = roots.to_frame().merge(
        root_sizes[["root", "component_id", "component_size"]],
        on="root",
        how="left",
    )
    output.insert(1, "component_id", component_lookup["component_id"])
    output.insert(2, "component_size", component_lookup["component_size"])
    return output


def component_summary(component_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        component_frame.groupby("component_id", dropna=False)
        .agg(
            component_size=("component_id", "size"),
            product_id_count=("canonical_product_id", "nunique"),
            identity_key_count=("identity_key", "nunique"),
            target_mean=("target_price", "mean"),
            min_row_index=("row_index", "min"),
        )
        .reset_index()
        .sort_values(
            ["component_size", "product_id_count", "identity_key_count", "min_row_index"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )


def assign_components(
    components: pd.DataFrame,
    seed: int,
    total_rows: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = components.copy()
    shuffled["_random"] = rng.random(len(shuffled))
    shuffled = shuffled.sort_values("_random").reset_index(drop=True)

    target_rows = {split: TARGET_PROPORTIONS[split] * total_rows for split in SPLITS}
    current_rows = {split: 0 for split in SPLITS}
    assignment_rows = []

    for _, component in shuffled.iterrows():
        component_size = int(component["component_size"])

        def candidate_score(split_name: str) -> tuple[float, float, int]:
            proposed = current_rows.copy()
            proposed[split_name] += component_size
            row_deviation = sum(
                abs(proposed[name] - target_rows[name]) / total_rows for name in SPLITS
            )
            target_pressure = proposed[split_name] / target_rows[split_name]
            return row_deviation, target_pressure, proposed[split_name]

        split = min(SPLITS, key=candidate_score)
        current_rows[split] += component_size
        assignment_rows.append(
            {
                "component_id": component["component_id"],
                "split": split,
                "component_size": component_size,
            }
        )

    return pd.DataFrame(assignment_rows)


def price_summary(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("split", dropna=False)["target_price"]
    summary = grouped.agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        p25=lambda values: values.quantile(0.25),
        p75=lambda values: values.quantile(0.75),
    ).reset_index()
    return order_split_rows(summary)


def split_row_summary(rows: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(rows)
    summary = (
        rows.groupby("split", dropna=False)
        .agg(
            row_count=("split", "size"),
            component_count=("component_id", "nunique"),
            product_id_count=("canonical_product_id", "nunique"),
            identity_key_count=("identity_key", "nunique"),
        )
        .reset_index()
    )
    summary["row_pct"] = summary["row_count"] / total_rows * 100
    summary["target_pct"] = summary["split"].map(lambda split: TARGET_PROPORTIONS[split] * 100)
    summary["pct_point_deviation"] = summary["row_pct"] - summary["target_pct"]
    return order_split_rows(summary)


def split_component_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return order_split_rows(
        rows.groupby("split", dropna=False)
        .agg(
            component_count=("component_id", "nunique"),
            min_component_size=("component_size", "min"),
            median_component_size=("component_size", "median"),
            max_component_size=("component_size", "max"),
        )
        .reset_index()
    )


def order_split_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_split_order"] = ordered["split"].map({split: index for index, split in enumerate(SPLITS)})
    return ordered.sort_values("_split_order").drop(columns="_split_order").reset_index(drop=True)


def leakage_check(rows: pd.DataFrame, column: str, label: str) -> dict[str, object]:
    split_counts = rows.groupby(column, dropna=False)["split"].nunique()
    leaked_values = split_counts[split_counts > 1]
    return {
        "check": label,
        "status": "PASS" if leaked_values.empty else "FAIL",
        "leaked_value_count": int(len(leaked_values)),
    }


def run_leakage_checks(rows: pd.DataFrame) -> pd.DataFrame:
    checks = [
        leakage_check(rows, "canonical_product_id", "product_id appears in only one split"),
        leakage_check(rows, "identity_key", "identity key appears in only one split"),
        leakage_check(rows, "component_id", "connected component appears in only one split"),
    ]
    summary = pd.DataFrame(checks)
    if not (summary["status"] == "PASS").all():
        raise AssertionError(f"Leakage check failed:\n{summary.to_string(index=False)}")
    return summary


def distribution_table(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in rows.columns:
        return pd.DataFrame()

    total_by_split = rows.groupby("split", dropna=False).size().rename("split_rows")
    counts = (
        rows.groupby(["split", column], dropna=False)
        .size()
        .reset_index(name="row_count")
        .join(total_by_split, on="split")
    )
    counts["row_pct_within_split"] = counts["row_count"] / counts["split_rows"] * 100
    pivot_count = counts.pivot(index=column, columns="split", values="row_count").fillna(0)
    pivot_pct = counts.pivot(index=column, columns="split", values="row_pct_within_split").fillna(0)

    for split in SPLITS:
        if split not in pivot_count.columns:
            pivot_count[split] = 0
        if split not in pivot_pct.columns:
            pivot_pct[split] = 0.0

    output = pd.DataFrame({column: pivot_count.index})
    for split in SPLITS:
        output[f"{split}_count"] = pivot_count[split].to_numpy(dtype=int)
        output[f"{split}_pct"] = pivot_pct[split].to_numpy(dtype=float)
    output["total_count"] = output[[f"{split}_count" for split in SPLITS]].sum(axis=1)
    output["max_pct_gap"] = (
        output[[f"{split}_pct" for split in SPLITS]].max(axis=1)
        - output[[f"{split}_pct" for split in SPLITS]].min(axis=1)
    )
    return output.sort_values(["total_count", "max_pct_gap"], ascending=[False, False]).reset_index(
        drop=True
    )


def price_balance_score(split_prices: pd.DataFrame, global_mean: float, global_median: float) -> float:
    score = 0.0
    denominator_mean = abs(global_mean) if global_mean else 1.0
    denominator_median = abs(global_median) if global_median else 1.0
    for _, row in split_prices.iterrows():
        score += abs(float(row["mean"]) - global_mean) / denominator_mean
        score += abs(float(row["median"]) - global_median) / denominator_median
    return float(score)


def simulate_seed(
    component_frame: pd.DataFrame,
    components: pd.DataFrame,
    seed: int,
    total_rows: int,
    global_mean: float,
    global_median: float,
) -> SimulationResult:
    assignments = assign_components(components, seed, total_rows)
    rows = component_frame.merge(assignments[["component_id", "split"]], on="component_id", how="left")
    if rows["split"].isna().any():
        raise AssertionError("Some rows did not receive a diagnostic split assignment.")

    leakage_summary = run_leakage_checks(rows)
    row_summary = split_row_summary(rows)
    comp_summary = split_component_summary(rows)
    prices = price_summary(rows)

    max_row_pct_deviation = float(row_summary["pct_point_deviation"].abs().max())
    score_row_balance = float(
        sum(abs(float(row["row_pct"]) - float(row["target_pct"])) for _, row in row_summary.iterrows())
    )
    score_price = price_balance_score(prices, global_mean, global_median)
    score_combined = score_row_balance + score_price

    return SimulationResult(
        seed=seed,
        row_summary=row_summary,
        component_summary=comp_summary,
        price_summary=prices,
        leakage_summary=leakage_summary,
        score_row_balance=score_row_balance,
        score_price_balance=score_price,
        score_combined=score_combined,
        max_row_pct_deviation=max_row_pct_deviation,
        assignments=assignments,
        rows=rows,
    )


def simulation_metric_rows(results: list[SimulationResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        summary = result.row_summary.set_index("split")
        price = result.price_summary.set_index("split")
        row = {
            "seed": result.seed,
            "row_balance_score": result.score_row_balance,
            "price_balance_score": result.score_price_balance,
            "combined_score": result.score_combined,
            "max_row_pct_deviation": result.max_row_pct_deviation,
        }
        for split in SPLITS:
            row[f"{split}_rows"] = int(summary.loc[split, "row_count"])
            row[f"{split}_row_pct"] = float(summary.loc[split, "row_pct"])
            row[f"{split}_components"] = int(summary.loc[split, "component_count"])
            row[f"{split}_price_mean"] = float(price.loc[split, "mean"])
            row[f"{split}_price_median"] = float(price.loc[split, "median"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["combined_score", "row_balance_score", "seed"])


def all_price_distribution_rows(results: list[SimulationResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for row in result.price_summary.to_dict("records"):
            rows.append({"seed": result.seed, **row})
    return pd.DataFrame(rows)


def all_leakage_check_rows(results: list[SimulationResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for row in result.leakage_summary.to_dict("records"):
            rows.append({"seed": result.seed, **row})
    return pd.DataFrame(rows)


def all_distribution_rows(results: list[SimulationResult], columns: list[str]) -> pd.DataFrame:
    rows = []
    for result in results:
        for column in columns:
            table = distribution_table(result.rows, column)
            if table.empty:
                continue
            table = table.copy()
            table.insert(0, "seed", result.seed)
            table.insert(1, "distribution_column", column)
            table = table.rename(columns={column: "value"})
            rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def split_size_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        values = metrics[f"{split}_rows"]
        pct_values = metrics[f"{split}_row_pct"]
        rows.append(
            {
                "split": split,
                "target_pct": TARGET_PROPORTIONS[split] * 100,
                "average_rows": values.mean(),
                "min_rows": values.min(),
                "max_rows": values.max(),
                "std_rows": values.std(ddof=0),
                "average_pct": pct_values.mean(),
                "min_pct": pct_values.min(),
                "max_pct": pct_values.max(),
                "std_pct": pct_values.std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def best_reasonable_seed(
    metrics: pd.DataFrame,
    best_row_seed: int,
    best_price_seed: int,
) -> tuple[int, str]:
    reasonable = metrics[
        (metrics["max_row_pct_deviation"] <= 2.0)
        & (metrics["price_balance_score"] <= metrics["price_balance_score"].quantile(0.25))
    ]
    if not reasonable.empty:
        selected = int(reasonable.sort_values(["combined_score", "seed"]).iloc[0]["seed"])
        return selected, (
            "Selected because row proportions are within 2 percentage points of target and "
            "price balance is in the best quartile among simulated seeds."
        )

    selected = int(metrics.sort_values(["combined_score", "seed"]).iloc[0]["seed"])
    if selected == best_row_seed == best_price_seed:
        reason = "Selected because it is best on both row-count balance and price-distribution balance."
    elif selected == best_row_seed:
        reason = "Selected because it has the strongest row-count balance and acceptable price balance."
    elif selected == best_price_seed:
        reason = "Selected because it has the strongest price-distribution balance and acceptable row balance."
    else:
        reason = "Selected as the best combined compromise between row-count and price balance."
    return selected, reason


def format_numeric_table(frame: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].round(decimals)
    return output


def top_distribution_table(rows: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    table = distribution_table(rows, column)
    if table.empty:
        return table
    return format_numeric_table(table.head(top_n), decimals=2)


def write_report(
    output_path: Path,
    input_path: Path,
    artifact_dir: Path,
    component_frame: pd.DataFrame,
    components: pd.DataFrame,
    metrics: pd.DataFrame,
    size_summary: pd.DataFrame,
    best_row: SimulationResult,
    best_price: SimulationResult,
    recommended: SimulationResult,
    recommended_reason: str,
    target_column: str,
    distribution_paths: dict[str, Path],
    all_price_path: Path,
    all_leakage_path: Path,
    all_distribution_path: Path,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    worst = metrics.sort_values(["max_row_pct_deviation", "row_balance_score"], ascending=False).iloc[0]
    all_leakage = pd.read_csv(all_leakage_path)
    leakage_across_simulations = (
        all_leakage
        .groupby("check")
        .agg(
            simulated_seeds=("seed", "nunique"),
            passed_seeds=("status", lambda values: int((values == "PASS").sum())),
            failed_seeds=("status", lambda values: int((values == "FAIL").sum())),
            max_leaked_value_count=("leaked_value_count", "max"),
        )
        .reset_index()
    )

    overview = pd.DataFrame(
        [
            {"metric": "rows", "value": f"{len(component_frame):,}"},
            {"metric": "connected components", "value": f"{len(components):,}"},
            {
                "metric": "largest component rows",
                "value": f"{int(components['component_size'].max()):,}",
            },
            {"metric": "simulated seeds", "value": f"{len(metrics):,}"},
            {"metric": "best row-balance seed", "value": int(best_row.seed)},
            {"metric": "best price-balance seed", "value": int(best_price.seed)},
            {"metric": "recommended diagnostic seed", "value": int(recommended.seed)},
            {
                "metric": "worst row imbalance observed",
                "value": f"{worst['max_row_pct_deviation']:.2f} percentage points "
                f"(seed {int(worst['seed'])})",
            },
        ]
    )

    seed_comparison = pd.DataFrame(
        [
            {
                "selection": "best row-count balance",
                "seed": best_row.seed,
                "row_balance_score": best_row.score_row_balance,
                "price_balance_score": best_row.score_price_balance,
                "max_row_pct_deviation": best_row.max_row_pct_deviation,
            },
            {
                "selection": "best price-distribution balance",
                "seed": best_price.seed,
                "row_balance_score": best_price.score_row_balance,
                "price_balance_score": best_price.score_price_balance,
                "max_row_pct_deviation": best_price.max_row_pct_deviation,
            },
            {
                "selection": "recommended diagnostic seed",
                "seed": recommended.seed,
                "row_balance_score": recommended.score_row_balance,
                "price_balance_score": recommended.score_price_balance,
                "max_row_pct_deviation": recommended.max_row_pct_deviation,
            },
        ]
    )

    lines = [
        "# Candidate Connected-Component Split Balance Diagnostic",
        "",
        f"Generated: {generated_at}",
        "",
        f"Input dataset: `{input_path.as_posix()}`",
        f"Target column: `{target_column}`",
        "",
        "## Candidate Protocol",
        "",
        "The proposed final thesis evaluation protocol keeps connected components intact. "
        "Each row is a node. Rows are connected if they share `product_id` or the canonical identity "
        "`part_name + brand + model + year_start + year_end`. Simulated train/validation/test "
        "assignments are made at component level using 70% / 15% / 15% row targets.",
        "",
        "Canonicalization: Unicode `NFKC`, `casefold`, trimmed whitespace, collapsed repeated "
        f"whitespace, missing values replaced with `{MISSING_SENTINEL}`, and no fuzzy matching.",
        "",
        "This script is diagnostic only. It does not train models and does not save final split files.",
        "",
        "## Overview",
        "",
        markdown_table(overview),
        "",
        "## Summary Across Simulations",
        "",
        markdown_table(format_numeric_table(size_summary, decimals=2)),
        "",
        "Simulation metrics CSV: "
        f"`{(artifact_dir / 'simulation_metrics.csv').as_posix()}`",
        "",
        "Full per-seed diagnostic CSVs:",
        "",
        f"- Price distributions: `{all_price_path.as_posix()}`",
        f"- Leakage checks: `{all_leakage_path.as_posix()}`",
        f"- Brand/model/category/year distributions: `{all_distribution_path.as_posix()}`",
        "",
        "## Best Diagnostic Seed",
        "",
        "Scoring note: `row_balance_score` is the sum of absolute percentage-point deviations "
        "from the 70/15/15 row targets. `price_balance_score` sums each split's relative mean "
        "and median price deviation from the full dataset. Lower values are better.",
        "",
        markdown_table(format_numeric_table(seed_comparison, decimals=4)),
        "",
        f"Recommended seed: `{recommended.seed}`. {recommended_reason}",
        "",
        "The recommended assignment is saved only as a diagnostic artifact, not as final split files:",
        "",
        f"- `{(artifact_dir / 'diagnostic_recommended_seed_component_assignments.csv').as_posix()}`",
        f"- `{(artifact_dir / 'diagnostic_recommended_seed_row_assignments.csv').as_posix()}`",
        "",
        "## Detailed Split Summary For Recommended Seed",
        "",
        markdown_table(format_numeric_table(recommended.row_summary, decimals=2)),
        "",
        "## Component Summary For Recommended Seed",
        "",
        markdown_table(format_numeric_table(recommended.component_summary, decimals=2)),
        "",
        "## Leakage Check Results",
        "",
        markdown_table(leakage_across_simulations),
        "",
        "Strong assertions were run for every simulated seed. A failure would stop the script.",
        "",
        "## Price Distribution For Recommended Seed",
        "",
        markdown_table(format_numeric_table(recommended.price_summary, decimals=2)),
        "",
        "## Distribution Comparison Tables",
        "",
    ]

    for column, path in distribution_paths.items():
        display = column
        lines.extend(
            [
                f"### {display}",
                "",
                f"Full table: `{path.as_posix()}`",
                "",
                markdown_table(top_distribution_table(recommended.rows, column, top_n=20)),
                "",
            ]
        )

    usable = (
        recommended.max_row_pct_deviation <= 2.0
        and (recommended.leakage_summary["status"] == "PASS").all()
    )
    usability_note = (
        "The proposed protocol appears usable for final model evaluation from a leakage-control "
        "and row-balance perspective. Price and subgroup distributions should still be reviewed "
        "before freezing the final split."
        if usable
        else "The proposed protocol needs review before freezing the final split because the "
        "recommended diagnostic assignment did not meet the row-balance threshold used here."
    )

    lines.extend(
        [
            "## Usability Notes",
            "",
            usability_note,
            "",
            "The connected-component design is stricter than product-id-only splitting because it keeps "
            "repeated listings and comparable part identities together. This should reduce optimistic "
            "identity leakage, but it also constrains split balance because large components cannot be "
            "divided across train, validation, and test.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_seeds < 100:
        raise ValueError("--num-seeds must be at least 100 for this diagnostic.")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    target_column = resolve_target_column(frame, args.target_column)
    validate_columns(frame, target_column)

    component_frame = build_component_frame(frame, target_column)
    components = component_summary(component_frame)
    component_frame.to_csv(artifact_dir / "diagnostic_component_rows.csv", index=False)
    components.to_csv(artifact_dir / "diagnostic_component_summary.csv", index=False)

    total_rows = len(component_frame)
    global_mean = float(component_frame["target_price"].mean())
    global_median = float(component_frame["target_price"].median())
    seeds = range(args.seed_start, args.seed_start + args.num_seeds)

    results = [
        simulate_seed(component_frame, components, seed, total_rows, global_mean, global_median)
        for seed in seeds
    ]
    metrics = simulation_metric_rows(results)
    metrics.to_csv(artifact_dir / "simulation_metrics.csv", index=False)
    size_summary = split_size_summary(metrics)
    size_summary.to_csv(artifact_dir / "split_size_summary_across_simulations.csv", index=False)

    all_price_path = artifact_dir / "simulation_price_distributions.csv"
    all_leakage_path = artifact_dir / "simulation_leakage_checks.csv"
    all_distribution_path = artifact_dir / "simulation_distribution_diagnostics.csv"
    all_price_distribution_rows(results).to_csv(all_price_path, index=False)
    all_leakage_check_rows(results).to_csv(all_leakage_path, index=False)

    best_row_seed = int(metrics.sort_values(["row_balance_score", "seed"]).iloc[0]["seed"])
    best_price_seed = int(metrics.sort_values(["price_balance_score", "seed"]).iloc[0]["seed"])
    recommended_seed, recommended_reason = best_reasonable_seed(
        metrics,
        best_row_seed=best_row_seed,
        best_price_seed=best_price_seed,
    )
    result_by_seed = {result.seed: result for result in results}
    best_row = result_by_seed[best_row_seed]
    best_price = result_by_seed[best_price_seed]
    recommended = result_by_seed[recommended_seed]

    recommended.assignments.sort_values("component_id").to_csv(
        artifact_dir / "diagnostic_recommended_seed_component_assignments.csv",
        index=False,
    )
    recommended.rows.sort_values("row_index").to_csv(
        artifact_dir / "diagnostic_recommended_seed_row_assignments.csv",
        index=False,
    )
    recommended.row_summary.to_csv(
        artifact_dir / "recommended_seed_split_row_summary.csv",
        index=False,
    )
    recommended.component_summary.to_csv(
        artifact_dir / "recommended_seed_split_component_summary.csv",
        index=False,
    )
    recommended.price_summary.to_csv(
        artifact_dir / "recommended_seed_price_distribution.csv",
        index=False,
    )
    recommended.leakage_summary.to_csv(
        artifact_dir / "recommended_seed_leakage_checks.csv",
        index=False,
    )

    distribution_columns = [
        column
        for column in [
            "brand",
            "model",
            "category",
            "subcategory",
            "year_start",
            "year_end",
        ]
        if column in recommended.rows.columns
    ]
    all_distribution_rows(results, distribution_columns).to_csv(all_distribution_path, index=False)

    distribution_paths = {}
    for column in distribution_columns:
        table = distribution_table(recommended.rows, column)
        path = artifact_dir / f"recommended_seed_{column}_distribution.csv"
        format_numeric_table(table, decimals=4).to_csv(path, index=False)
        distribution_paths[column] = path

    write_report(
        output_path=output_path,
        input_path=input_path,
        artifact_dir=artifact_dir,
        component_frame=component_frame,
        components=components,
        metrics=metrics,
        size_summary=size_summary,
        best_row=best_row,
        best_price=best_price,
        recommended=recommended,
        recommended_reason=recommended_reason,
        target_column=target_column,
        distribution_paths=distribution_paths,
        all_price_path=all_price_path,
        all_leakage_path=all_leakage_path,
        all_distribution_path=all_distribution_path,
    )

    print(f"Wrote candidate split balance report to: {output_path}")
    print(f"Wrote diagnostic CSV artifacts to: {artifact_dir}")
    print(f"Recommended diagnostic seed: {recommended.seed}")


if __name__ == "__main__":
    main()
