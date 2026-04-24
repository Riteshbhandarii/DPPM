from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.part_identity_evaluation import add_part_identity_group, load_split_frames
from src.tree_modeling import TARGET_COLUMN, fit_random_forest


DEFAULT_GROUP_COLUMNS = ["part_name", "brand", "model", "oem_number"]
DEFAULT_SUMMARY_PATH = "artifacts/random_forest_tuning_strict/best_tuning_summary.json"

PART_TAXONOMY = {"part_name", "category", "subcategory"}
VEHICLE_IDENTITY = {"brand", "model"}
CONDITION = {"quality_grade", "repair_status"}
VEHICLE_AGE_USAGE = {
    "mileage",
    "mileage_missing_flag",
    "year_start",
    "year_end",
    "year_span",
    "year_mid",
}
LISTING_HISTORY = {
    "observations_so_far",
    "days_since_first_seen_so_far",
    "first_seen_day_offset",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_final_rf_inputs(
    data_paths: list[str | Path],
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    group_columns: list[str] | None = None,
    drop_features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, Any]]:
    """Load final strict RF summary and the modeling frame used for SHAP."""

    summary = load_json(summary_path)
    features = list(summary["feature_names"])
    if drop_features:
        drop_set = set(drop_features)
        features = [feature for feature in features if feature not in drop_set]
    config = dict(summary["config"])
    config["model_params"] = dict(config["model_params"])

    frame = load_split_frames(data_paths)
    frame, _ = add_part_identity_group(
        frame,
        columns=group_columns or DEFAULT_GROUP_COLUMNS,
    )
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise KeyError(f"Final RF features missing from data: {missing}")
    return frame, features, config, summary


def fit_final_rf(frame: pd.DataFrame, features: list[str], config: dict[str, Any]):
    """Fit the final RF pipeline on the provided frame."""

    return fit_random_forest(
        frame[features].copy(),
        frame[TARGET_COLUMN].copy(),
        config,
    )


def dense_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def transformed_to_raw_feature(transformed_name: str, raw_features: list[str]) -> str:
    """Map ColumnTransformer output names back to original feature names."""

    if transformed_name.startswith("num__"):
        return transformed_name.removeprefix("num__")
    if transformed_name.startswith("cat__"):
        remainder = transformed_name.removeprefix("cat__")
        matches = [
            feature
            for feature in raw_features
            if remainder == feature or remainder.startswith(feature + "_")
        ]
        if matches:
            return max(matches, key=len)
        return remainder
    return transformed_name


def feature_group(feature: str) -> str:
    """Assign raw model features to thesis-readable explanation groups."""

    if feature in PART_TAXONOMY:
        return "part_taxonomy"
    if feature in VEHICLE_IDENTITY:
        return "vehicle_identity"
    if feature in CONDITION:
        return "part_condition"
    if feature in VEHICLE_AGE_USAGE:
        return "vehicle_age_usage"
    if feature.startswith("model_"):
        return "traficom_model_context"
    if feature.startswith("brand_"):
        return "traficom_brand_context"
    if feature in LISTING_HISTORY:
        return "listing_history"
    return "other"


def aggregate_shap_to_raw_features(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    raw_features: list[str],
) -> pd.DataFrame:
    """Aggregate transformed SHAP columns back to raw model features."""

    raw_to_indices: dict[str, list[int]] = {}
    for idx, transformed_name in enumerate(transformed_feature_names):
        raw_name = transformed_to_raw_feature(transformed_name, raw_features)
        raw_to_indices.setdefault(raw_name, []).append(idx)

    grouped = {
        raw_name: shap_values[:, indices].sum(axis=1)
        for raw_name, indices in raw_to_indices.items()
    }
    return pd.DataFrame(grouped)


def raw_feature_importance(raw_shap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in raw_shap.columns:
        values = raw_shap[feature].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group(feature),
                "mean_abs_shap": float(np.mean(np.abs(values))),
                "mean_shap": float(np.mean(values)),
                "median_abs_shap": float(np.median(np.abs(values))),
            }
        )
    result = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    total = float(result["mean_abs_shap"].sum())
    result["importance_share"] = result["mean_abs_shap"] / total if total else 0.0
    return result.reset_index(drop=True)


def grouped_feature_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        feature_importance.groupby("feature_group", as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "sum"),
            mean_shap=("mean_shap", "sum"),
            feature_count=("feature", "count"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["mean_abs_shap"].sum())
    grouped["importance_share"] = grouped["mean_abs_shap"] / total if total else 0.0
    return grouped


def raw_shap_long_table(
    sample_frame: pd.DataFrame,
    raw_shap: pd.DataFrame,
    row_id_column: str = "sample_row_id",
) -> pd.DataFrame:
    """Build a compact long table for local/group analysis."""

    rows = []
    for row_idx, (_, input_row) in enumerate(sample_frame.iterrows()):
        for feature in raw_shap.columns:
            shap_value = float(raw_shap.iloc[row_idx][feature])
            rows.append(
                {
                    row_id_column: row_idx,
                    "feature": feature,
                    "feature_group": feature_group(feature),
                    "feature_value": input_row.get(feature),
                    "shap_value": shap_value,
                    "abs_shap_value": abs(shap_value),
                }
            )
    return pd.DataFrame(rows)
