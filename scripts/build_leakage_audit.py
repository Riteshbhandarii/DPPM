"""
Purpose:
Build the feature leakage audit: one row per candidate column, saying where it
comes from, whether it would exist when a new listing is priced, how risky it
is, and whether the frozen models used it.

The table is derived, not typed by hand. Group membership and exclusions are
read from the feature constants in `src/tree_modeling.py`, which are the rules
the tuning actually applied, and model membership from the frozen tuning
summaries. The risk tier and its basis come from the rules below, each of which
names the record it rests on. Where no record explains a choice, the table
says so instead of inventing a reason.

Inputs:
- datasets/cleaned/clean_master_dataset.csv   every column the models could see
- src/tree_modeling.py                        feature groups and exclusion sets
- artifacts/strict_model_tuning/{random_forest,ridge}/best_tuning_summary.json

Outputs:
- artifacts/leakage_audit/feature_leakage_audit.csv

Assumptions:
- Read-only. Nothing is refitted and no split or frozen artifact is touched.
- "Available at prediction time" means: known when a dismantler lists a part
  that has never been listed before.

How to run:
    .venv/bin/python scripts/build_leakage_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import tree_modeling as fs  # noqa: E402  (needs the repo root on the path)

MASTER = ROOT / "datasets/cleaned/clean_master_dataset.csv"
TUNING = ROOT / "artifacts/strict_model_tuning"
OUT = ROOT / "artifacts/leakage_audit/feature_leakage_audit.csv"

FULL_HISTORY = set(fs.LISTING_DYNAMICS_FEATURES) | {
    "last_seen_day_offset",
    "listing_midpoint_day_offset",
}
HISTORY_SO_FAR = {
    "observations_so_far",
    "days_since_first_seen_so_far",
    "price_changed_flag_so_far",
    "price_change_count_so_far",
    "absolute_price_change_so_far",
    "relative_price_change_pct_so_far",
}
REGISTRY = set(
    fs.TRAFICOM_FEATURES + fs.REGISTRY_LIFECYCLE_CANDIDATES + fs.TRAFICOM_EXTENDED_CANDIDATES
)
# AGENTS.md "Safe Features" names these, or the compatibility year range they
# are computed from.
AGENTS_SAFE = {
    "brand", "model", "category", "subcategory", "quality_grade", "mileage",
    "year_start", "year_end", "year_span", "year_mid",
}


def classify(column: str) -> tuple[str, str, str, str]:
    """Return (source, available at prediction time, risk, basis) for a column."""
    if column == fs.TARGET_COLUMN:
        return "marketplace listing", "no", "forbidden", "AGENTS.md Forbidden Features: target price"
    if column == fs.GROUP_COLUMN:
        return "marketplace identifier", "no", "forbidden", (
            "AGENTS.md Forbidden Features: product_id; used only as a split grouping key "
            "(docs/evaluation/01_PROTOCOL_DECISION.md)"
        )
    if column in FULL_HISTORY:
        return "listing history, whole scrape window", "no", "high", (
            "src/tree_modeling.py COMMON_LEAKAGE_RISK_FEATURES: full-history variables "
            "that would leak future information; AGENTS.md Leakage-Sensitive Features"
        )
    if column in HISTORY_SO_FAR:
        return "listing history up to the snapshot", "only for parts already listed", "medium", (
            "AGENTS.md Leakage-Sensitive Features (listing and price history); the four "
            "price_*_so_far columns are in COMMON_LEAKAGE_RISK_FEATURES, the two count/day "
            "columns are in no exclusion set and no record explains why"
        )
    if column in fs.DATE_COLUMNS or column == "first_seen_day_offset":
        return "scrape timing", "no", "medium", (
            "AGENTS.md Leakage-Sensitive Features (first_seen_date, last_seen_date, scrape "
            "timing); excluded by RECOMMENDED_EXCLUDED_FEATURES / RANDOM_FOREST_LEAKAGE_ONLY "
            "and the *_without_listing_dates / *_without_date_offsets variants"
        )
    if column == "oem_number":
        return "marketplace listing", "yes", "medium", (
            "AGENTS.md Leakage-Sensitive Features (OEM numbers); docs/DESIGN_DECISIONS.md "
            "2026-06-26: OEM values are noisy and reused; the frozen winner is the "
            "*_without_oem_number variant"
        )
    if column in fs.RECOMMENDED_EXCLUDED_FEATURES:
        return "pipeline key or flag", "yes", "none", (
            "src/tree_modeling.py RECOMMENDED_EXCLUDED_FEATURES: identifier, redundant "
            "key or unsuitable metadata"
        )
    if column in REGISTRY:
        return "Traficom registry aggregate", "yes", "none", (
            "AGENTS.md Context Features; aggregated from the national registry, "
            "independent of listing prices. Constant within each vehicle model, so "
            "they carry no information beyond `model` (artifacts/registry_ablation/)"
        )
    if column in AGENTS_SAFE:
        return "marketplace listing", "yes", "none", "AGENTS.md Safe Features"
    if column == "part_name":
        return "marketplace listing", "yes", "none", (
            "part taxonomy shown on the listing; no separate leakage record"
        )
    if column == "repair_status":
        return "registry join flag", "yes", "none", (
            "set by the registry join (notebooks/02_integration/03_final_dataset_merging.ipynb); "
            "one value on every row, so it carries no information"
        )
    if column in {"mileage_missing_flag", "brand_is_known_model_family"}:
        return "derived listing flag", "yes", "none", "no recorded decision"
    return "unclassified", "unknown", "unknown", "no recorded decision"


def main() -> None:
    master = pd.read_csv(MASTER, low_memory=False)
    rf = set(json.loads((TUNING / "random_forest/best_tuning_summary.json").read_text())["feature_names"])
    ridge = set(json.loads((TUNING / "ridge/best_tuning_summary.json").read_text())["feature_names"])

    # Candidates named in the code but never written to the dataset are listed
    # too, so the audit shows they could not have reached a model.
    candidates = list(dict.fromkeys(
        list(master.columns)
        + fs.LISTING_DATE_OFFSET_FEATURES
        + sorted(fs.COMMON_LEAKAGE_RISK_FEATURES)
    ))

    rows = []
    for column in candidates:
        source, available, risk, basis = classify(column)
        rows.append({
            "feature": column,
            "source": source,
            "available_at_prediction": available,
            "leakage_risk": risk,
            "in_dataset": column in master.columns,
            "distinct_values": int(master[column].nunique()) if column in master.columns else 0,
            "used_by_rf_winner": column in rf,
            "used_by_ridge_finalist": column in ridge,
            "basis": basis,
        })

    table = pd.DataFrame(rows)
    unknown = table[table.leakage_risk == "unknown"]
    if len(unknown):
        raise SystemExit(f"unclassified columns, add a rule: {list(unknown.feature)}")
    # The frozen winner must not contain anything this audit calls forbidden or high.
    bad = table[table.used_by_rf_winner & table.leakage_risk.isin(["forbidden", "high"])]
    if len(bad):
        raise SystemExit(f"frozen winner uses high-risk features: {list(bad.feature)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.groupby(["leakage_risk", "used_by_rf_winner", "used_by_ridge_finalist"]).size().to_string())
    print(f"\n{len(table)} candidates, RF winner uses {int(table.used_by_rf_winner.sum())}, "
          f"Ridge finalist {int(table.used_by_ridge_finalist.sum())}")
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
