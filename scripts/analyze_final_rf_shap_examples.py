#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_rf_shap_utils import (  # noqa: E402
    DEFAULT_GROUP_COLUMNS,
    DEFAULT_SUMMARY_PATH,
    aggregate_shap_to_raw_features,
    dense_array,
    feature_group,
    fit_final_rf,
    load_final_rf_inputs,
)
from src.tree_modeling import TARGET_COLUMN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local SHAP examples for final strict RF.")
    parser.add_argument("--data-path", action="append", default=["datasets/splits/train_grouped.csv"])
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--behavior-path", default="artifacts/final_model_behavior/oof_predictions.csv")
    parser.add_argument("--output-dir", default="artifacts/final_model_shap")
    parser.add_argument("--examples-per-type", type=int, default=3)
    parser.add_argument("--background-size", type=int, default=150)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--quick-estimators", type=int, default=None)
    parser.add_argument("--group-columns", nargs="+", default=DEFAULT_GROUP_COLUMNS)
    return parser.parse_args()


def select_examples(behavior: pd.DataFrame, examples_per_type: int) -> pd.DataFrame:
    behavior = behavior.copy()
    behavior["example_source"] = "unassigned"

    selectors = [
        ("accurate_low_error", behavior.sort_values(["absolute_error", TARGET_COLUMN], ascending=[True, True])),
        ("expensive_high_error", behavior.sort_values(["absolute_error", TARGET_COLUMN], ascending=[False, False])),
        ("largest_underprediction", behavior.sort_values("error", ascending=False)),
        ("largest_overprediction", behavior.sort_values("error", ascending=True)),
        ("cheap_item", behavior.sort_values(TARGET_COLUMN, ascending=True)),
        ("expensive_item", behavior.sort_values(TARGET_COLUMN, ascending=False)),
    ]

    selected = []
    seen_products = set()
    for label, candidate_frame in selectors:
        taken = 0
        for _, row in candidate_frame.iterrows():
            product_id = row.get("product_id")
            key = product_id if pd.notna(product_id) else row.name
            if key in seen_products:
                continue
            output = row.copy()
            output["example_source"] = label
            selected.append(output)
            seen_products.add(key)
            taken += 1
            if taken >= examples_per_type:
                break
    return pd.DataFrame(selected).reset_index(drop=True)


def group_local_effects(raw_shap_row: pd.Series) -> pd.DataFrame:
    rows = []
    for feature, shap_value in raw_shap_row.items():
        rows.append(
            {
                "feature_group": feature_group(feature),
                "feature": feature,
                "shap_value": float(shap_value),
                "abs_shap_value": abs(float(shap_value)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, features, config, summary = load_final_rf_inputs(
        args.data_path,
        summary_path=args.summary_path,
        group_columns=args.group_columns,
    )
    if args.quick_estimators is not None:
        config["model_params"]["n_estimators"] = args.quick_estimators

    behavior = pd.read_csv(args.behavior_path)
    examples = select_examples(behavior, args.examples_per_type)
    example_product_ids = examples["product_id"].tolist()
    example_frame = frame[frame["product_id"].isin(example_product_ids)].copy()
    example_frame = examples[["product_id", "example_source"]].merge(
        example_frame,
        on="product_id",
        how="left",
    )

    model = fit_final_rf(frame, features, config)
    preprocessor = model.named_steps["preprocessor"]
    forest = model.named_steps["model"]
    background_frame = frame.sample(n=min(args.background_size, len(frame)), random_state=43)
    transformed = dense_array(preprocessor.transform(example_frame[features].copy()))
    background_transformed = dense_array(preprocessor.transform(background_frame[features].copy()))
    transformed_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(
        forest,
        data=background_transformed,
        feature_perturbation="interventional",
    )
    shap_values = np.asarray(explainer.shap_values(transformed, check_additivity=False), dtype=float)
    raw_shap = aggregate_shap_to_raw_features(shap_values, transformed_names, features)
    predictions = model.predict(example_frame[features].copy())
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    local_rows = []
    group_rows = []
    for row_idx, (_, row) in enumerate(example_frame.iterrows()):
        local_effects = group_local_effects(raw_shap.iloc[row_idx])
        local_effects["product_id"] = row["product_id"]
        local_effects["example_source"] = row["example_source"]
        local_effects["feature_value"] = [
            row.get(feature) for feature in local_effects["feature"].tolist()
        ]
        local_rows.append(
            local_effects.sort_values("abs_shap_value", ascending=False)
            .head(args.top_n)
            .reset_index(drop=True)
        )

        grouped = (
            local_effects.groupby("feature_group", as_index=False)
            .agg(shap_value=("shap_value", "sum"), abs_shap_value=("abs_shap_value", "sum"))
            .sort_values("abs_shap_value", ascending=False)
        )
        grouped["product_id"] = row["product_id"]
        grouped["example_source"] = row["example_source"]
        group_rows.append(grouped)

    example_summary = example_frame[
        [
            "example_source",
            "product_id",
            "part_name",
            "brand",
            "model",
            "quality_grade",
            "mileage",
            "year_mid",
            TARGET_COLUMN,
        ]
    ].copy()
    example_summary["predicted_price"] = predictions
    example_summary["base_value"] = base_value
    example_summary["net_shap_effect"] = raw_shap.sum(axis=1).to_numpy()
    example_summary["reconstruction_error"] = (
        example_summary["base_value"] + example_summary["net_shap_effect"] - predictions
    ).abs()

    example_summary.to_csv(output_dir / "local_examples_summary.csv", index=False)
    pd.concat(local_rows, ignore_index=True).to_csv(output_dir / "local_top_feature_effects.csv", index=False)
    pd.concat(group_rows, ignore_index=True).to_csv(output_dir / "local_grouped_effects.csv", index=False)
    (output_dir / "local_examples_metadata.json").write_text(
        json.dumps(
            {
                "model_type": summary.get("model_type"),
                "feature_variant": summary.get("feature_variant"),
                "config_name": summary.get("config_name"),
                "base_value": base_value,
                "examples": len(example_summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved final RF local SHAP examples to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
