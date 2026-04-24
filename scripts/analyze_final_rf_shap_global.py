#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
    fit_final_rf,
    grouped_feature_importance,
    load_final_rf_inputs,
    raw_feature_importance,
)
from src.tree_modeling import TARGET_COLUMN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global SHAP analysis for the final strict RF.")
    parser.add_argument("--data-path", action="append", default=["datasets/splits/train_grouped.csv"])
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", default="artifacts/final_model_shap")
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--background-size", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--quick-estimators", type=int, default=None)
    parser.add_argument("--group-columns", nargs="+", default=DEFAULT_GROUP_COLUMNS)
    return parser.parse_args()


def plot_bar(df: pd.DataFrame, label_col: str, value_col: str, path: Path, title: str) -> None:
    plot_df = df.sort_values(value_col, ascending=True)
    plt.figure(figsize=(9, max(4, 0.32 * len(plot_df))))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.xlabel("Mean absolute SHAP value (EUR)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_dependence(sample_frame: pd.DataFrame, raw_shap: pd.DataFrame, feature: str, path: Path) -> None:
    if feature not in sample_frame.columns or feature not in raw_shap.columns:
        return
    values = pd.to_numeric(sample_frame[feature], errors="coerce")
    mask = values.notna()
    if int(mask.sum()) < 20:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(values[mask], raw_shap.loc[mask, feature], s=10, alpha=0.35)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel(feature)
    plt.ylabel("SHAP value (EUR)")
    plt.title(f"SHAP dependence: {feature}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def category_effects(sample_frame: pd.DataFrame, raw_shap: pd.DataFrame, feature: str, min_rows: int = 15):
    if feature not in sample_frame.columns or feature not in raw_shap.columns:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            feature: sample_frame[feature].astype("string").fillna("__missing__"),
            "shap_value": raw_shap[feature],
        }
    )
    result = (
        frame.groupby(feature, as_index=False)
        .agg(rows=("shap_value", "size"), mean_shap=("shap_value", "mean"), mean_abs_shap=("shap_value", lambda x: np.mean(np.abs(x))))
    )
    result = result[result["rows"] >= min_rows]
    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    frame, features, config, summary = load_final_rf_inputs(
        args.data_path,
        summary_path=args.summary_path,
        group_columns=args.group_columns,
    )
    if args.quick_estimators is not None:
        config["model_params"]["n_estimators"] = args.quick_estimators

    sample_frame = frame.sample(
        n=min(args.sample_size, len(frame)),
        random_state=args.random_state,
    ).reset_index(drop=True)
    model = fit_final_rf(frame, features, config)
    preprocessor = model.named_steps["preprocessor"]
    forest = model.named_steps["model"]

    background_frame = frame.sample(
        n=min(args.background_size, len(frame)),
        random_state=args.random_state + 1,
    )
    transformed = dense_array(preprocessor.transform(sample_frame[features].copy()))
    background_transformed = dense_array(preprocessor.transform(background_frame[features].copy()))
    transformed_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(
        forest,
        data=background_transformed,
        feature_perturbation="interventional",
    )
    shap_values = np.asarray(explainer.shap_values(transformed, check_additivity=False), dtype=float)
    raw_shap = aggregate_shap_to_raw_features(shap_values, transformed_names, features)

    feature_importance = raw_feature_importance(raw_shap)
    group_importance = grouped_feature_importance(feature_importance)
    traficom_importance = feature_importance[
        feature_importance["feature_group"].isin(
            ["traficom_model_context", "traficom_brand_context"]
        )
    ].reset_index(drop=True)

    predictions = model.predict(sample_frame[features].copy())
    metadata = {
        "model_type": summary.get("model_type"),
        "feature_variant": summary.get("feature_variant"),
        "config_name": summary.get("config_name"),
        "sample_size": int(len(sample_frame)),
        "background_size": int(len(background_frame)),
        "feature_count": len(features),
        "transformed_feature_count": int(transformed.shape[1]),
        "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        "mean_prediction": float(np.mean(predictions)),
        "mean_actual_price": float(sample_frame[TARGET_COLUMN].mean()),
    }

    feature_importance.to_csv(output_dir / "global_feature_importance.csv", index=False)
    group_importance.to_csv(output_dir / "grouped_feature_importance.csv", index=False)
    traficom_importance.to_csv(output_dir / "traficom_feature_importance.csv", index=False)
    sample_frame[["product_id", TARGET_COLUMN, "part_name", "brand", "model"]].assign(
        predicted_price=predictions
    ).to_csv(output_dir / "shap_sample_rows.csv", index=False)
    (output_dir / "shap_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plot_bar(
        feature_importance.head(args.top_n),
        "feature",
        "mean_abs_shap",
        plot_dir / "top_raw_features.png",
        "Final RF SHAP: top raw features",
    )
    plot_bar(
        group_importance,
        "feature_group",
        "mean_abs_shap",
        plot_dir / "feature_group_importance.png",
        "Final RF SHAP: feature group importance",
    )
    if not traficom_importance.empty:
        plot_bar(
            traficom_importance.head(args.top_n),
            "feature",
            "mean_abs_shap",
            plot_dir / "top_traficom_features.png",
            "Final RF SHAP: top Traficom features",
        )

    for feature in [
        "mileage",
        "year_mid",
        "model_total_registered",
        "model_median_mileage",
        "brand_total_registered",
        "observations_so_far",
    ]:
        plot_dependence(sample_frame, raw_shap, feature, plot_dir / f"dependence_{feature}.png")

    for feature in ["part_name", "brand", "model", "quality_grade"]:
        category_effects(sample_frame, raw_shap, feature).to_csv(
            output_dir / f"category_effects_{feature}.csv",
            index=False,
        )

    print(f"Saved final RF global SHAP analysis to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
