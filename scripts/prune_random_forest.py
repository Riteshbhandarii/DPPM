#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_modeling import (  # noqa: E402
    TARGET_COLUMN,
    build_random_forest_pipeline,
    convert_predictions_to_eur,
    prepare_target,
)


WINNING_FEATURES = [
    "part_name",
    "quality_grade",
    "oem_number",
    "mileage",
    "brand",
    "model",
    "category",
    "subcategory",
    "year_start",
    "year_end",
    "year_span",
    "year_mid",
    "repair_status",
    "brand_is_known_model_family",
    "model_total_registered",
    "model_median_vehicle_age",
    "model_mean_vehicle_age",
    "model_median_mileage",
    "model_mean_mileage",
    "model_median_engine_cc",
    "model_median_power_kw",
    "model_median_mass_kg",
    "model_share_of_market",
    "model_share_within_brand",
    "model_share_over_10y",
    "model_share_over_200k_km",
    "model_automatic_share",
    "model_petrol_share",
    "model_diesel_share",
    "model_ev_share",
    "model_hybrid_share",
    "model_turbo_share",
    "model_firstreg_total_2014_2026",
    "model_firstreg_year_span",
    "model_firstreg_peak_year",
    "model_firstreg_peak_count",
    "model_firstreg_recent_share",
    "model_firstreg_old_share",
    "model_firstreg_weighted_year",
    "brand_total_registered",
    "brand_median_vehicle_age",
    "brand_mean_vehicle_age",
    "brand_median_mileage",
    "brand_mean_mileage",
    "brand_median_engine_cc",
    "brand_median_power_kw",
    "brand_median_mass_kg",
    "brand_share_of_market",
    "brand_share_over_10y",
    "brand_share_over_200k_km",
    "brand_automatic_share",
    "brand_petrol_share",
    "brand_diesel_share",
    "brand_ev_share",
    "brand_hybrid_share",
    "brand_turbo_share",
    "brand_firstreg_total_2014_2026",
    "brand_firstreg_year_span",
    "brand_firstreg_peak_year",
    "brand_firstreg_peak_count",
    "brand_firstreg_recent_share",
    "brand_firstreg_old_share",
    "brand_firstreg_weighted_year",
    "mileage_missing_flag",
    "observations_so_far",
    "days_since_first_seen_so_far",
]

BASELINE_VALIDATION_MAE = 18.2409
TARGET_MODE = "raw"
ONEHOT_MIN_FREQUENCY = 5
MODEL_PARAMS = {
    "n_estimators": 400,
    "min_samples_leaf": 1,
    "max_features": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune low-importance features from the winning random forest."
    )
    parser.add_argument("--train-path", default="datasets/splits/train_grouped.csv")
    parser.add_argument("--validation-path", default="datasets/splits/validation_grouped.csv")
    parser.add_argument("--output-dir", default="artifacts/random_forest_final")
    return parser.parse_args()


def fit_and_score(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    features: list[str],
):
    X_train = train_df[features].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_validation = validation_df[features].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()

    pipeline = build_random_forest_pipeline(
        X_train_current=X_train,
        model_params=MODEL_PARAMS,
        onehot_min_frequency=ONEHOT_MIN_FREQUENCY,
    )
    pipeline.fit(X_train, prepare_target(y_train, TARGET_MODE))
    predictions = convert_predictions_to_eur(
        pipeline.predict(X_validation),
        TARGET_MODE,
        y_train,
    )
    mae = float(mean_absolute_error(y_validation, predictions))
    return pipeline, mae


def aggregate_importances(pipeline, original_features: list[str]) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()
    raw_importances = model.feature_importances_

    rows = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "importance": raw_importances,
        }
    )
    rows["feature"] = rows["transformed_feature"].map(
        lambda name: map_transformed_feature_to_original(name, original_features)
    )

    ranked = (
        rows.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked[["rank", "feature", "importance"]]


def map_transformed_feature_to_original(name: str, original_features: list[str]) -> str:
    if name.startswith("num__"):
        return name.removeprefix("num__")
    if name.startswith("cat__"):
        encoded_name = name.removeprefix("cat__")
        for feature in sorted(original_features, key=len, reverse=True):
            if encoded_name == feature or encoded_name.startswith(f"{feature}_"):
                return feature
    raise ValueError(f"Could not map transformed feature back to original column: {name}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    validation_df = pd.read_csv(args.validation_path)

    missing_features = [feature for feature in WINNING_FEATURES if feature not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing winning features from train split: {missing_features}")

    print("Random forest feature pruning")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Baseline validation MAE: {BASELINE_VALIDATION_MAE:.4f}")
    print(f"Original feature count: {len(WINNING_FEATURES)}")

    original_pipeline, original_mae = fit_and_score(
        train_df=train_df,
        validation_df=validation_df,
        features=WINNING_FEATURES,
    )
    print(f"Recomputed original validation MAE: {original_mae:.4f}")

    ranked_importances = aggregate_importances(original_pipeline, WINNING_FEATURES)
    importance_path = output_dir / "feature_importance_builtin.csv"
    ranked_importances.to_csv(importance_path, index=False)
    print(f"Saved built-in feature importances to: {importance_path}")

    results = [
        {
            "prune_fraction": 0.0,
            "removed_feature_count": 0,
            "kept_feature_count": len(WINNING_FEATURES),
            "validation_MAE": original_mae,
            "removed_features": "",
        }
    ]

    best_pruned_result = None
    for prune_fraction in [0.10, 0.20, 0.30]:
        remove_count = max(1, int(round(len(WINNING_FEATURES) * prune_fraction)))
        removed_features = ranked_importances.tail(remove_count)["feature"].tolist()
        kept_features = [
            feature for feature in WINNING_FEATURES if feature not in set(removed_features)
        ]

        _, pruned_mae = fit_and_score(
            train_df=train_df,
            validation_df=validation_df,
            features=kept_features,
        )
        delta = pruned_mae - BASELINE_VALIDATION_MAE
        print(
            f"Pruned bottom {int(prune_fraction * 100)}%: "
            f"{len(kept_features)} features, validation MAE={pruned_mae:.4f}, "
            f"delta vs baseline={delta:+.4f}"
        )

        result = {
            "prune_fraction": prune_fraction,
            "removed_feature_count": remove_count,
            "kept_feature_count": len(kept_features),
            "validation_MAE": pruned_mae,
            "removed_features": ", ".join(removed_features),
        }
        results.append(result)

        if pruned_mae < BASELINE_VALIDATION_MAE:
            if best_pruned_result is None or pruned_mae < best_pruned_result["validation_MAE"]:
                best_pruned_result = {
                    **result,
                    "kept_features": kept_features,
                }

    results_path = output_dir / "feature_pruning_results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Saved pruning results to: {results_path}")

    if best_pruned_result is None:
        print("KEEP ORIGINAL 66 FEATURES")
        return

    pruned_features_path = output_dir / "pruned_features.txt"
    pruned_features_path.write_text(
        "\n".join(best_pruned_result["kept_features"]) + "\n",
        encoding="utf-8",
    )
    print(
        "PRUNING HELPS — "
        f"best MAE: {best_pruned_result['validation_MAE']:.4f} "
        f"with {best_pruned_result['kept_feature_count']} features"
    )
    print(f"Saved winning pruned feature list to: {pruned_features_path}")


if __name__ == "__main__":
    main()
