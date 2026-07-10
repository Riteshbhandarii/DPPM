"""SHAP explanation of the frozen strict Random Forest (#62).

Descriptive only. This explains model behaviour, never market causality, and is never
grounds to retune: the holdout has been run once and the guard is consumed
(`docs/DESIGN_DECISIONS.md`, 2026-07-10).

The model is refit on `train_strict + validation_strict` exactly as the holdout did, then
explained on the test rows so the attributions describe the predictions that were actually
reported. Only test *features* are read; the test target is used solely to slice rows into
price bands for reporting.

Three outputs, mirroring the questions the holdout left open:

1. Global attributions - does the taxonomy dominate, and does the Traficom registry stack
   earn its place among the 61 features?
2. Attributions restricted to the 500-1000 EUR band, the only segment where the model
   significantly beats the subcategory-median heuristic.
3. Local explanations for listings above 1000 EUR, where the model over-predicts by a
   median of ~898 EUR.

Usage:
    uv run python scripts/run_strict_shap.py [--sample-size N] [--seed 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from final_rf_shap_utils import (  # noqa: E402
    aggregate_shap_to_raw_features,
    dense_array,
    grouped_feature_importance,
    raw_feature_importance,
)
from src.tree_modeling import fit_random_forest, load_training_data  # noqa: E402

SPLITS = ROOT / "datasets/splits_strict"
SUMMARY_PATH = ROOT / "artifacts/strict_model_tuning/random_forest/best_tuning_summary.json"
HOLDOUT_DIR = ROOT / "artifacts/strict_final_holdout"
OUTPUT_DIR = ROOT / "artifacts/strict_final_shap"
TARGET = "price"

WIN_BAND = (500.0, 1000.0)  # the only segment where the model beats the heuristic
TAIL_THRESHOLD = 1000.0  # where the model over-predicts


def align_test_frame(test_df: pd.DataFrame, reference_first_seen_date) -> pd.DataFrame:
    """Recompute date offsets on the test frame against the fit frame's reference date."""
    for column in ("first_seen_date", "last_seen_date", "scrape_date"):
        if column in test_df.columns:
            test_df[column] = pd.to_datetime(test_df[column], errors="coerce")
    if reference_first_seen_date is not None and "first_seen_date" in test_df.columns:
        reference = pd.Timestamp(reference_first_seen_date)
        test_df["first_seen_day_offset"] = (test_df["first_seen_date"] - reference).dt.days
        test_df["last_seen_day_offset"] = (test_df["last_seen_date"] - reference).dt.days
        test_df["listing_midpoint_day_offset"] = (
            test_df["first_seen_day_offset"] + test_df["last_seen_day_offset"]
        ) / 2
    return test_df


def importance_for(raw_shap: pd.DataFrame, mask: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = raw_shap if mask is None else raw_shap.loc[mask]
    per_feature = raw_feature_importance(subset)
    return per_feature, grouped_feature_importance(per_feature)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Explain a random subsample of test rows (0 = all 1,696).",
    )
    parser.add_argument("--seed", type=int, default=32)
    args = parser.parse_args()

    if not (HOLDOUT_DIR / "final_holdout_metrics.json").exists():
        raise SystemExit("Holdout artifacts missing — run the holdout before explaining it.")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    features = list(summary["feature_names"])
    config = dict(summary["config"])
    print(f"Frozen winner: random_forest | {summary['feature_variant']} | {summary['config_name']}")
    print(f"  target_mode={config['target_mode']}  features={len(features)}")

    prepared = load_training_data(SPLITS / "train_strict.csv", SPLITS / "validation_strict.csv")
    fit_df = pd.concat([prepared.train_df, prepared.validation_df], ignore_index=True)
    test_df = align_test_frame(pd.read_csv(SPLITS / "test_strict.csv"), prepared.reference_first_seen_date)
    print(f"  fit rows={len(fit_df)}  test rows={len(test_df)}")

    if config["target_mode"] != "raw":
        raise SystemExit(
            f"target_mode={config['target_mode']}: SHAP values would be on the log scale, "
            "not euros. Handle the back-transform before trusting the numbers."
        )

    pipeline = fit_random_forest(fit_df[features].copy(), fit_df[TARGET].copy(), config)
    preprocessor, regressor = pipeline[:-1], pipeline[-1]

    explain_df = test_df
    if args.sample_size and args.sample_size < len(test_df):
        explain_df = test_df.sample(args.sample_size, random_state=args.seed).sort_index()
    print(f"  explaining {len(explain_df)} test rows")

    transformed = dense_array(preprocessor.transform(explain_df[features].copy()))
    transformed_names = list(preprocessor.get_feature_names_out())

    print("  computing TreeExplainer SHAP values (this is the slow part)...")
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(transformed, check_additivity=False)
    raw_shap = aggregate_shap_to_raw_features(shap_values, transformed_names, features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y = explain_df[TARGET].to_numpy(dtype=float)

    # --- 1. global -------------------------------------------------------------------
    per_feature, grouped = importance_for(raw_shap)
    per_feature.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)
    grouped.to_csv(OUTPUT_DIR / "shap_group_importance.csv", index=False)

    print("\n=== global attribution by feature group ===")
    print(grouped[["feature_group", "feature_count", "mean_abs_shap", "importance_share"]].to_string(index=False))
    print("\n=== top 12 features ===")
    print(per_feature.head(12)[["feature", "feature_group", "mean_abs_shap", "importance_share"]].to_string(index=False))

    # --- 2. the band the model actually wins ------------------------------------------
    win_mask = (y > WIN_BAND[0]) & (y <= WIN_BAND[1])
    band_report: dict[str, object] = {"n": int(win_mask.sum())}
    if win_mask.sum() >= 10:
        band_feature, band_grouped = importance_for(raw_shap, win_mask)
        band_feature.to_csv(OUTPUT_DIR / "shap_feature_importance_500_1000.csv", index=False)
        band_grouped.to_csv(OUTPUT_DIR / "shap_group_importance_500_1000.csv", index=False)
        band_report["group_shares"] = dict(
            zip(band_grouped["feature_group"], band_grouped["importance_share"].round(4))
        )
        print(f"\n=== 500-1000 EUR band (n={int(win_mask.sum())}) — where the model beats the heuristic ===")
        print(band_grouped[["feature_group", "mean_abs_shap", "importance_share"]].to_string(index=False))
        print("  top features:", ", ".join(band_feature.head(5)["feature"]))

    # --- 3. the tail the model over-predicts ------------------------------------------
    tail_mask = y > TAIL_THRESHOLD
    tail_report: dict[str, object] = {"n": int(tail_mask.sum())}
    if tail_mask.sum() >= 5:
        tail_feature, tail_grouped = importance_for(raw_shap, tail_mask)
        tail_feature.to_csv(OUTPUT_DIR / "shap_feature_importance_above_1000.csv", index=False)
        tail_grouped.to_csv(OUTPUT_DIR / "shap_group_importance_above_1000.csv", index=False)
        # signed push: which groups drive predictions UP on the tail rows?
        signed = (
            tail_feature.assign(group=tail_feature["feature_group"])
            .groupby("group", as_index=False)["mean_shap"]
            .sum()
            .sort_values("mean_shap", ascending=False)
        )
        signed.to_csv(OUTPUT_DIR / "shap_signed_push_above_1000.csv", index=False)
        tail_report["signed_push_eur"] = dict(zip(signed["group"], signed["mean_shap"].round(2)))
        print(f"\n=== above {TAIL_THRESHOLD:.0f} EUR (n={int(tail_mask.sum())}) — where the model over-predicts ===")
        print("  signed SHAP push in EUR (positive = pushes the prediction up):")
        print(signed.to_string(index=False))
        print(f"  explainer base value (expected prediction): {float(np.mean(explainer.expected_value)):.2f} EUR")

    # --- provenance --------------------------------------------------------------------
    taxonomy_share = float(
        grouped.loc[grouped["feature_group"] == "part_taxonomy", "importance_share"].sum()
    )
    traficom_share = float(
        grouped.loc[
            grouped["feature_group"].isin(["traficom_model_context", "traficom_brand_context"]),
            "importance_share",
        ].sum()
    )
    traficom_count = int(
        grouped.loc[
            grouped["feature_group"].isin(["traficom_model_context", "traficom_brand_context"]),
            "feature_count",
        ].sum()
    )

    report = {
        "generated_for": "#62 SHAP explainability, descriptive only",
        "model": "random_forest (frozen holdout winner)",
        "feature_variant": summary["feature_variant"],
        "config_name": summary["config_name"],
        "fit_scope": "train_strict + validation_strict (9,625 rows), identical to the holdout refit",
        "explained_rows": int(len(explain_df)),
        "explained_split": "test_strict.csv (features only; target used only to slice price bands)",
        "expected_value_eur": float(np.mean(explainer.expected_value)),
        "part_taxonomy_importance_share": round(taxonomy_share, 4),
        "traficom_importance_share": round(traficom_share, 4),
        "traficom_feature_count": traficom_count,
        "group_shares": dict(zip(grouped["feature_group"], grouped["importance_share"].round(4))),
        "band_500_1000": band_report,
        "above_1000": tail_report,
    }
    (OUTPUT_DIR / "shap_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== headline ===")
    print(f"  part_taxonomy carries {taxonomy_share:.1%} of total attribution")
    print(f"  the {traficom_count} Traficom registry features carry {traficom_share:.1%} between them")
    print(f"\nwritten: {OUTPUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
