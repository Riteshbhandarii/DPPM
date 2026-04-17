from __future__ import annotations

"""Shared data preparation and tuning helpers for the tree-model experiments."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - handled by CLI/runtime checks
    XGBRegressor = None


# Core dataset column names reused across both model families.
TARGET_COLUMN = "price"
GROUP_COLUMN = "product_id"
DATE_COLUMNS = ["first_seen_date", "last_seen_date", "scrape_date"]
LISTING_DATE_OFFSET_FEATURES = [
    "first_seen_day_offset",
    "last_seen_day_offset",
    "listing_midpoint_day_offset",
]

# Baseline listing features shared by the notebook experiments.
BASELINE_FEATURES = [
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
]

# Core Traficom aggregation features.
TRAFICOM_FEATURES = [
    "model_total_registered",
    "model_median_vehicle_age",
    "model_mean_vehicle_age",
    "model_median_mileage",
    "model_mean_mileage",
    "model_median_engine_cc",
    "model_median_power_kw",
    "model_median_mass_kg",
    "brand_total_registered",
    "brand_median_vehicle_age",
    "brand_mean_vehicle_age",
    "brand_median_mileage",
    "brand_mean_mileage",
]

# Registration-history candidates added when they exist in the split files.
REGISTRY_LIFECYCLE_CANDIDATES = [
    "model_firstreg_total_2014_2026",
    "model_firstreg_recent_share",
    "model_firstreg_old_share",
    "model_firstreg_weighted_year",
    "model_firstreg_peak_year",
    "model_firstreg_peak_count",
    "model_firstreg_year_span",
    "brand_firstreg_total_2014_2026",
    "brand_firstreg_recent_share",
    "brand_firstreg_old_share",
    "brand_firstreg_weighted_year",
    "brand_firstreg_peak_year",
    "brand_firstreg_peak_count",
    "brand_firstreg_year_span",
]

# Additional Traficom-derived features that were treated as optional in the notebooks.
TRAFICOM_EXTENDED_CANDIDATES = [
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
]

# Listing-behavior variables that need leakage checks before model selection.
LISTING_DYNAMICS_FEATURES = [
    "times_observed",
    "observed_span_days",
    "price_changed_flag",
    "price_change_count",
    "absolute_price_change",
    "relative_price_change_pct",
]

# Columns excluded because they are identifiers, redundant keys, or unsuitable metadata.
RECOMMENDED_EXCLUDED_FEATURES = {
    "product_id",
    "scrape_date",
    "brand_merge_key",
    "model_merge_key",
    "model_family_clean",
    "model_looks_like_part_taxonomy",
}

# Raw date fields removed for XGBoost because numeric offsets are used instead.
XGBOOST_SPECIFIC_EXCLUDED_FEATURES = {
    "first_seen_date",
    "last_seen_date",
}

# Full-history variables that would leak future information into training.
COMMON_LEAKAGE_RISK_FEATURES = {
    "times_observed",
    "observed_span_days",
    "price_changed_flag",
    "price_change_count",
    "absolute_price_change",
    "relative_price_change_pct",
    "price_changed_flag_so_far",
    "price_change_count_so_far",
    "absolute_price_change_so_far",
    "relative_price_change_pct_so_far",
    "last_seen_day_offset",
    "listing_midpoint_day_offset",
}

# Random forest excludes the raw end-date field as an extra leakage guard.
RANDOM_FOREST_LEAKAGE_ONLY = {
    "last_seen_date",
}

# Candidate XGBoost configurations copied from the tuned notebook finalists.
XGBOOST_CONFIGS = {
    "log_absoluteerror_balanced": {
        "target_mode": "log",
        "model_params": {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 2200,
            "learning_rate": 0.025,
            "max_depth": 5,
            "min_child_weight": 5,
            "gamma": 0.05,
            "subsample": 0.80,
            "colsample_bytree": 0.70,
            "colsample_bylevel": 0.80,
            "reg_alpha": 0.20,
            "reg_lambda": 3.00,
            "max_bin": 256,
            "max_cat_to_onehot": 10,
            "max_cat_threshold": 64,
        },
    },
    "log_absoluteerror_regularized": {
        "target_mode": "log",
        "model_params": {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 2600,
            "learning_rate": 0.020,
            "max_depth": 4,
            "min_child_weight": 8,
            "gamma": 0.10,
            "subsample": 0.75,
            "colsample_bytree": 0.65,
            "colsample_bylevel": 0.80,
            "reg_alpha": 0.40,
            "reg_lambda": 4.50,
            "max_bin": 256,
            "max_cat_to_onehot": 10,
            "max_cat_threshold": 64,
        },
    },
    "log_sqerror_balanced": {
        "target_mode": "log",
        "model_params": {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "n_estimators": 1800,
            "learning_rate": 0.030,
            "max_depth": 5,
            "min_child_weight": 5,
            "gamma": 0.08,
            "subsample": 0.80,
            "colsample_bytree": 0.70,
            "colsample_bylevel": 0.80,
            "reg_alpha": 0.20,
            "reg_lambda": 3.25,
            "max_bin": 256,
            "max_cat_to_onehot": 10,
            "max_cat_threshold": 64,
        },
    },
    "raw_absoluteerror_balanced": {
        "target_mode": "raw",
        "model_params": {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 1800,
            "learning_rate": 0.025,
            "max_depth": 5,
            "min_child_weight": 5,
            "gamma": 0.05,
            "subsample": 0.80,
            "colsample_bytree": 0.70,
            "colsample_bylevel": 0.80,
            "reg_alpha": 0.20,
            "reg_lambda": 3.00,
            "max_bin": 256,
            "max_cat_to_onehot": 10,
            "max_cat_threshold": 64,
        },
    },
    "raw_sqerror_reference": {
        "target_mode": "raw",
        "model_params": {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "n_estimators": 1800,
            "learning_rate": 0.030,
            "max_depth": 5,
            "min_child_weight": 5,
            "gamma": 0.08,
            "subsample": 0.80,
            "colsample_bytree": 0.70,
            "colsample_bylevel": 0.80,
            "reg_alpha": 0.20,
            "reg_lambda": 3.25,
            "max_bin": 256,
            "max_cat_to_onehot": 10,
            "max_cat_threshold": 64,
        },
    },
}

# Candidate random-forest configurations copied from the tuned notebook finalists.
RANDOM_FOREST_CONFIGS = {
    "log_reference": {
        "target_mode": "log",
        "onehot_min_frequency": 5,
        "model_params": {
            "n_estimators": 300,
            "min_samples_leaf": 2,
        },
    },
    "log_half_features_leaf_1": {
        "target_mode": "log",
        "onehot_min_frequency": 5,
        "model_params": {
            "n_estimators": 400,
            "min_samples_leaf": 1,
            "max_features": 0.5,
        },
    },
    "log_half_features_leaf_2": {
        "target_mode": "log",
        "onehot_min_frequency": 5,
        "model_params": {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "max_features": 0.5,
        },
    },
    "raw_half_features_leaf_1": {
        "target_mode": "raw",
        "onehot_min_frequency": 5,
        "model_params": {
            "n_estimators": 400,
            "min_samples_leaf": 1,
            "max_features": 0.5,
        },
    },
    "raw_half_features_leaf_2": {
        "target_mode": "raw",
        "onehot_min_frequency": 5,
        "model_params": {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "max_features": 0.5,
        },
    },
}


@dataclass
class PreparedData:
    """Grouped train and validation splits with derived date-offset metadata."""

    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    reference_first_seen_date: str | None


def load_training_data(train_path: str | Path, validation_path: str | Path) -> PreparedData:
    """Load grouped split files and derive the listing-date offsets used in tuning."""

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    for dataset_df in (train_df, validation_df):
        for date_column in DATE_COLUMNS:
            if date_column in dataset_df.columns:
                dataset_df[date_column] = pd.to_datetime(
                    dataset_df[date_column],
                    errors="coerce",
                )

    reference_first_seen_date = None
    first_seen_candidates = []
    for dataset_df in (train_df, validation_df):
        if "first_seen_date" in dataset_df.columns:
            first_seen_candidates.append(dataset_df["first_seen_date"].min())

    if first_seen_candidates:
        reference_first_seen_timestamp = min(first_seen_candidates)
        reference_first_seen_date = reference_first_seen_timestamp.strftime("%Y-%m-%d")
        for dataset_df in (train_df, validation_df):
            dataset_df["first_seen_day_offset"] = (
                dataset_df["first_seen_date"] - reference_first_seen_timestamp
            ).dt.days
            dataset_df["last_seen_day_offset"] = (
                dataset_df["last_seen_date"] - reference_first_seen_timestamp
            ).dt.days
            dataset_df["listing_midpoint_day_offset"] = (
                dataset_df["first_seen_day_offset"] + dataset_df["last_seen_day_offset"]
            ) / 2

    return PreparedData(
        train_df=train_df,
        validation_df=validation_df,
        reference_first_seen_date=reference_first_seen_date,
    )


def prepare_target(y_series: pd.Series, target_mode: str) -> pd.Series:
    """Transform the training target according to the selected objective scale."""

    if target_mode == "log":
        return np.log(y_series)
    if target_mode == "raw":
        return y_series.copy()
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def convert_predictions_to_eur(
    predictions: np.ndarray,
    target_mode: str,
    y_train_reference: pd.Series,
) -> np.ndarray:
    """Map model outputs back to euro prices with basic numeric safety checks."""

    predictions = np.asarray(predictions, dtype=float)
    if target_mode == "log":
        upper_price_bound = float(y_train_reference.max()) * 10
        clipped_predictions = np.nan_to_num(
            predictions,
            nan=0.0,
            posinf=np.log(upper_price_bound),
            neginf=np.log(1e-3),
        )
        clipped_predictions = np.clip(
            clipped_predictions,
            a_min=np.log(1e-3),
            a_max=np.log(upper_price_bound),
        )
        return np.exp(clipped_predictions)
    if target_mode == "raw":
        return np.clip(
            np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0),
            a_min=0.0,
            a_max=None,
        )
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the validation metrics used throughout the notebook comparisons."""

    return {
        "validation_MAE": float(mean_absolute_error(y_true, y_pred)),
        "validation_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "validation_R2": float(r2_score(y_true, y_pred)),
    }


def make_config_signature(config: dict[str, Any]) -> str:
    """Build a stable JSON signature so repeated parameter sets can be removed safely."""

    return json.dumps(config, sort_keys=True)


def sample_log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    """Sample a positive floating-point value on a log scale."""

    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def generate_xgboost_search_configs(
    random_trials: int,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    """Create a wider XGBoost search space while keeping the notebook finalists as anchors."""

    rng = np.random.default_rng(random_seed)
    configs = {name: dict(config) for name, config in XGBOOST_CONFIGS.items()}
    seen_signatures = {make_config_signature(config) for config in configs.values()}

    objectives = {
        "raw": ["reg:squarederror", "reg:absoluteerror"],
        "log": ["reg:squarederror", "reg:absoluteerror"],
    }
    max_bins = [128, 192, 256, 384, 512]
    cat_to_onehot_choices = [4, 8, 10, 12, 16]
    cat_threshold_choices = [32, 48, 64, 96, 128]

    trial_id = 1
    while len(configs) < len(XGBOOST_CONFIGS) + random_trials:
        target_mode = str(rng.choice(["raw", "log"]))
        objective = str(rng.choice(objectives[target_mode]))
        config = {
            "target_mode": target_mode,
            "model_params": {
                "objective": objective,
                "eval_metric": "mae",
                "n_estimators": int(rng.integers(1200, 3201)),
                "learning_rate": round(sample_log_uniform(rng, 0.012, 0.06), 6),
                "max_depth": int(rng.integers(3, 9)),
                "min_child_weight": int(rng.integers(1, 13)),
                "gamma": round(float(rng.uniform(0.0, 0.25)), 6),
                "subsample": round(float(rng.uniform(0.65, 0.95)), 6),
                "colsample_bytree": round(float(rng.uniform(0.55, 0.95)), 6),
                "colsample_bylevel": round(float(rng.uniform(0.60, 1.00)), 6),
                "reg_alpha": round(sample_log_uniform(rng, 0.001, 1.0), 6),
                "reg_lambda": round(sample_log_uniform(rng, 1.0, 8.0), 6),
                "max_bin": int(rng.choice(max_bins)),
                "max_cat_to_onehot": int(rng.choice(cat_to_onehot_choices)),
                "max_cat_threshold": int(rng.choice(cat_threshold_choices)),
            },
        }

        signature = make_config_signature(config)
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        configs[f"random_search_{trial_id:03d}"] = config
        trial_id += 1

    return configs


def generate_random_forest_search_configs(
    random_trials: int,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    """Create a wider random-forest search space while keeping the notebook finalists as anchors."""

    rng = np.random.default_rng(random_seed)
    configs = {name: dict(config) for name, config in RANDOM_FOREST_CONFIGS.items()}
    seen_signatures = {make_config_signature(config) for config in configs.values()}
    onehot_frequency_choices = [2, 3, 5, 8, 12]
    max_features_choices: list[str | float] = ["sqrt", "log2", 0.35, 0.5, 0.7, 0.9]
    max_samples_choices: list[float | None] = [None, 0.6, 0.75, 0.9]
    max_depth_choices: list[int | None] = [None, 12, 18, 24, 32]

    trial_id = 1
    while len(configs) < len(RANDOM_FOREST_CONFIGS) + random_trials:
        target_mode = str(rng.choice(["raw", "log"]))
        bootstrap = bool(rng.integers(0, 2))
        max_samples = None
        if bootstrap:
            max_samples = max_samples_choices[int(rng.integers(0, len(max_samples_choices)))]
        config = {
            "target_mode": target_mode,
            "onehot_min_frequency": int(rng.choice(onehot_frequency_choices)),
            "model_params": {
                "n_estimators": int(rng.integers(300, 1401)),
                "min_samples_leaf": int(rng.integers(1, 7)),
                "min_samples_split": int(rng.integers(2, 13)),
                "max_features": rng.choice(max_features_choices).item()
                if hasattr(rng.choice(max_features_choices), "item")
                else rng.choice(max_features_choices),
                "bootstrap": bootstrap,
                "max_depth": rng.choice(max_depth_choices).item()
                if hasattr(rng.choice(max_depth_choices), "item")
                else rng.choice(max_depth_choices),
                "max_samples": max_samples,
            },
        }

        signature = make_config_signature(config)
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        configs[f"random_search_{trial_id:03d}"] = config
        trial_id += 1

    return configs


def build_feature_catalog(train_df: pd.DataFrame, model_kind: str) -> dict[str, Any]:
    """Recreate the trusted feature variants used by each model family."""

    registry_lifecycle_features = [
        column for column in REGISTRY_LIFECYCLE_CANDIDATES if column in train_df.columns
    ]
    traficom_extended_features = [
        column for column in TRAFICOM_EXTENDED_CANDIDATES if column in train_df.columns
    ]
    manual_all_feature_groups = list(
        dict.fromkeys(
            BASELINE_FEATURES
            + TRAFICOM_FEATURES
            + registry_lifecycle_features
            + traficom_extended_features
            + LISTING_DYNAMICS_FEATURES
            + LISTING_DATE_OFFSET_FEATURES
        )
    )

    if model_kind == "xgboost":
        model_specific_exclusions = XGBOOST_SPECIFIC_EXCLUDED_FEATURES
        leakage_features = COMMON_LEAKAGE_RISK_FEATURES
    elif model_kind == "random_forest":
        model_specific_exclusions = set(DATE_COLUMNS)
        leakage_features = COMMON_LEAKAGE_RISK_FEATURES | RANDOM_FOREST_LEAKAGE_ONLY
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    base_exclusions = RECOMMENDED_EXCLUDED_FEATURES | model_specific_exclusions
    trusted_exclusions = base_exclusions | leakage_features

    recommended_model_features = [
        column
        for column in train_df.columns
        if column != TARGET_COLUMN and column not in base_exclusions
    ]
    recommended_model_features_without_date_offsets = [
        column
        for column in recommended_model_features
        if column not in LISTING_DATE_OFFSET_FEATURES
    ]
    recommended_model_features_leakage_safe = [
        column
        for column in train_df.columns
        if column != TARGET_COLUMN and column not in trusted_exclusions
    ]
    recommended_model_features_leakage_safe_without_date_offsets = [
        column
        for column in recommended_model_features_leakage_safe
        if column not in LISTING_DATE_OFFSET_FEATURES
    ]

    manual_all_model_features = [
        column for column in manual_all_feature_groups if column not in base_exclusions
    ]
    manual_all_model_features_leakage_safe = [
        column
        for column in manual_all_model_features
        if column not in leakage_features
    ]

    if model_kind == "xgboost":
        feature_sets = {
            "trusted_registry_lifecycle_stack": list(
                dict.fromkeys(BASELINE_FEATURES + TRAFICOM_FEATURES + registry_lifecycle_features)
            ),
            "trusted_registry_lifecycle_stack_without_oem_number": [
                column
                for column in dict.fromkeys(
                    BASELINE_FEATURES + TRAFICOM_FEATURES + registry_lifecycle_features
                )
                if column != "oem_number"
            ],
            "trusted_extended_traficom_stack": list(
                dict.fromkeys(
                    BASELINE_FEATURES
                    + TRAFICOM_FEATURES
                    + registry_lifecycle_features
                    + traficom_extended_features
                )
            ),
            "trusted_manual_all_features_usable_by_xgboost": (
                manual_all_model_features_leakage_safe
            ),
            "trusted_recommended_features": recommended_model_features_leakage_safe,
            "trusted_recommended_features_without_date_offsets": (
                recommended_model_features_leakage_safe_without_date_offsets
            ),
            "trusted_recommended_features_without_date_offsets_without_oem_number": [
                column
                for column in recommended_model_features_leakage_safe_without_date_offsets
                if column != "oem_number"
            ],
        }
    else:
        feature_sets = {
            "trusted_extended_traficom_stack": list(
                dict.fromkeys(
                    BASELINE_FEATURES
                    + TRAFICOM_FEATURES
                    + registry_lifecycle_features
                    + traficom_extended_features
                )
            ),
            "trusted_manual_all_features_set": manual_all_model_features_leakage_safe,
            "trusted_recommended_features": recommended_model_features_leakage_safe,
            "trusted_recommended_features_without_listing_dates": [
                column
                for column in recommended_model_features_leakage_safe_without_date_offsets
                if column not in {"first_seen_date", "last_seen_date"}
            ],
        }

    available_feature_sets = {}
    seen_signatures = set()
    for name, features in feature_sets.items():
        available = [column for column in dict.fromkeys(features) if column in train_df.columns]
        signature = tuple(available)
        if not available or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        available_feature_sets[name] = available

    return {
        "feature_sets": available_feature_sets,
        "recommended_model_features": recommended_model_features,
        "recommended_model_features_without_date_offsets": (
            recommended_model_features_without_date_offsets
        ),
        "recommended_model_features_leakage_safe": recommended_model_features_leakage_safe,
        "recommended_model_features_leakage_safe_without_date_offsets": (
            recommended_model_features_leakage_safe_without_date_offsets
        ),
    }


def build_random_forest_pipeline(
    X_train_current: pd.DataFrame,
    model_params: dict[str, Any],
    onehot_min_frequency: int,
) -> Pipeline:
    """Build the preprocessing and estimator pipeline for random forest tuning."""

    numeric_features = X_train_current.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X_train_current.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=onehot_min_frequency),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    final_model_params = {
        "random_state": 42,
        "n_jobs": -1,
    }
    final_model_params.update(model_params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(**final_model_params)),
        ]
    )


def align_xgboost_frames(
    X_train_current: pd.DataFrame,
    X_validation_current: pd.DataFrame,
    category_levels: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Align dtypes and categorical levels so XGBoost sees matching train and validation frames."""

    X_train_prepared = X_train_current.copy()
    X_validation_prepared = X_validation_current.copy()

    datetime_columns = X_train_prepared.select_dtypes(
        include=["datetime64[ns]", "datetimetz"]
    ).columns.tolist()
    if datetime_columns:
        X_train_prepared = X_train_prepared.drop(columns=datetime_columns)
        X_validation_prepared = X_validation_prepared.drop(
            columns=datetime_columns,
            errors="ignore",
        )

    bool_columns = X_train_prepared.select_dtypes(include=["bool"]).columns.tolist()
    for column in bool_columns:
        X_train_prepared[column] = X_train_prepared[column].astype(int)
        X_validation_prepared[column] = X_validation_prepared[column].astype(int)

    categorical_columns = X_train_prepared.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    derived_category_levels = {} if category_levels is None else dict(category_levels)

    for column in categorical_columns:
        train_as_string = X_train_prepared[column].astype("string")
        validation_as_string = X_validation_prepared[column].astype("string")

        categories = derived_category_levels.get(column)
        if categories is None:
            categories_index = pd.Index(train_as_string.dropna().unique())
            if len(categories_index) == 0:
                categories_index = pd.Index(["__missing__"])
            categories = categories_index.astype(str).tolist()
            derived_category_levels[column] = categories

        X_train_prepared[column] = pd.Categorical(train_as_string, categories=categories)
        X_validation_prepared[column] = pd.Categorical(
            validation_as_string,
            categories=categories,
        )

    return X_train_prepared, X_validation_prepared, derived_category_levels


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> Pipeline:
    """Fit one random-forest candidate on the provided feature frame."""

    pipeline = build_random_forest_pipeline(
        X_train_current=X_train,
        model_params=config["model_params"],
        onehot_min_frequency=config["onehot_min_frequency"],
    )
    pipeline.fit(X_train, prepare_target(y_train, config["target_mode"]))
    return pipeline


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    config: dict[str, Any],
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Fit one XGBoost candidate and return the model together with categorical metadata."""

    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Install xgboost before running XGBoost tuning.")

    X_train_prepared, X_validation_prepared, category_levels = align_xgboost_frames(
        X_train,
        X_validation,
    )

    model_params = {
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 120,
        "device": device,
    }
    model_params.update(config["model_params"])

    model = XGBRegressor(**model_params)
    model.fit(
        X_train_prepared,
        prepare_target(y_train, config["target_mode"]),
        eval_set=[
            (
                X_validation_prepared,
                prepare_target(y_validation, config["target_mode"]),
            )
        ],
        verbose=False,
    )

    metadata = {
        "category_levels": category_levels,
        "best_iteration": getattr(model, "best_iteration", None),
    }
    return model, metadata


def screen_xgboost_candidates(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    configs: dict[str, dict[str, Any]],
    xgboost_device: str,
    top_k_finalists: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Score a broad XGBoost search space on the fixed validation split before grouped CV."""

    y_train = train_df[TARGET_COLUMN].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()
    total_candidates = len(feature_sets) * len(configs)
    current_candidate = 0
    screening_rows = []

    for feature_variant_name, feature_list in feature_sets.items():
        X_train = train_df[feature_list].copy()
        X_validation = validation_df[feature_list].copy()

        for config_name, config in configs.items():
            current_candidate += 1
            print(
                f"[screen {current_candidate}/{total_candidates}] "
                f"{feature_variant_name} | {config_name}"
            )
            model, metadata = fit_xgboost(
                X_train,
                y_train,
                X_validation,
                y_validation,
                config,
                device=xgboost_device,
            )
            _, X_validation_prepared, _ = align_xgboost_frames(X_train, X_validation)
            raw_predictions = model.predict(X_validation_prepared)
            validation_predictions = convert_predictions_to_eur(
                raw_predictions,
                config["target_mode"],
                y_train_reference=y_train,
            )
            screening_rows.append(
                {
                    "model_type": "xgboost",
                    "feature_variant": feature_variant_name,
                    "config_name": config_name,
                    "target_mode": config["target_mode"],
                    "feature_count": len(feature_list),
                    "best_iteration": metadata.get("best_iteration"),
                    "validation_MAE": float(mean_absolute_error(y_validation, validation_predictions)),
                    "validation_RMSE": float(
                        np.sqrt(mean_squared_error(y_validation, validation_predictions))
                    ),
                    "validation_R2": float(r2_score(y_validation, validation_predictions)),
                    "feature_names": feature_list,
                    "config": config,
                }
            )

    screening_results_df = pd.DataFrame(screening_rows).sort_values(
        ["validation_MAE", "validation_RMSE", "feature_count"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    finalists = []
    seen_combinations = set()
    best_per_feature_variant = (
        screening_results_df.sort_values(["validation_MAE", "validation_RMSE"])
        .groupby("feature_variant", as_index=False)
        .head(1)
    )

    for _, row in best_per_feature_variant.iterrows():
        combination_key = (row["feature_variant"], row["config_name"])
        if combination_key in seen_combinations:
            continue
        finalists.append(row.to_dict())
        seen_combinations.add(combination_key)

    for _, row in screening_results_df.iterrows():
        if len(finalists) >= top_k_finalists:
            break
        combination_key = (row["feature_variant"], row["config_name"])
        if combination_key in seen_combinations:
            continue
        finalists.append(row.to_dict())
        seen_combinations.add(combination_key)

    finalists = finalists[:top_k_finalists]
    return screening_results_df, finalists


def screen_random_forest_candidates(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    configs: dict[str, dict[str, Any]],
    top_k_finalists: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Score a broad random-forest search space on the fixed validation split before grouped CV."""

    y_train = train_df[TARGET_COLUMN].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()
    total_candidates = len(feature_sets) * len(configs)
    current_candidate = 0
    screening_rows = []

    for feature_variant_name, feature_list in feature_sets.items():
        X_train = train_df[feature_list].copy()
        X_validation = validation_df[feature_list].copy()

        for config_name, config in configs.items():
            current_candidate += 1
            print(
                f"[screen {current_candidate}/{total_candidates}] "
                f"{feature_variant_name} | {config_name}"
            )
            model = fit_random_forest(X_train, y_train, config)
            raw_predictions = model.predict(X_validation)
            validation_predictions = convert_predictions_to_eur(
                raw_predictions,
                config["target_mode"],
                y_train_reference=y_train,
            )
            screening_rows.append(
                {
                    "model_type": "random_forest",
                    "feature_variant": feature_variant_name,
                    "config_name": config_name,
                    "target_mode": config["target_mode"],
                    "feature_count": len(feature_list),
                    "validation_MAE": float(mean_absolute_error(y_validation, validation_predictions)),
                    "validation_RMSE": float(
                        np.sqrt(mean_squared_error(y_validation, validation_predictions))
                    ),
                    "validation_R2": float(r2_score(y_validation, validation_predictions)),
                    "feature_names": feature_list,
                    "config": config,
                }
            )

    screening_results_df = pd.DataFrame(screening_rows).sort_values(
        ["validation_MAE", "validation_RMSE", "feature_count"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    finalists = []
    seen_combinations = set()
    best_per_feature_variant = (
        screening_results_df.sort_values(["validation_MAE", "validation_RMSE"])
        .groupby("feature_variant", as_index=False)
        .head(1)
    )

    for _, row in best_per_feature_variant.iterrows():
        combination_key = (row["feature_variant"], row["config_name"])
        if combination_key in seen_combinations:
            continue
        finalists.append(row.to_dict())
        seen_combinations.add(combination_key)

    for _, row in screening_results_df.iterrows():
        if len(finalists) >= top_k_finalists:
            break
        combination_key = (row["feature_variant"], row["config_name"])
        if combination_key in seen_combinations:
            continue
        finalists.append(row.to_dict())
        seen_combinations.add(combination_key)

    finalists = finalists[:top_k_finalists]
    return screening_results_df, finalists


def evaluate_selected_xgboost_candidates(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    selected_candidates: list[dict[str, Any]],
    cv_splits: int,
    xgboost_device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run grouped cross-validation only on the best screened XGBoost candidates."""

    if not selected_candidates:
        raise RuntimeError("No XGBoost finalists were provided for grouped cross-validation.")

    y_train = train_df[TARGET_COLUMN].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()
    train_groups = train_df[GROUP_COLUMN]
    group_kfold = GroupKFold(n_splits=cv_splits)
    cv_rows = []

    for candidate_index, candidate in enumerate(selected_candidates, start=1):
        print(
            f"[cv {candidate_index}/{len(selected_candidates)}] "
            f"{candidate['feature_variant']} | {candidate['config_name']}"
        )
        feature_list = list(candidate["feature_names"])
        config = dict(candidate["config"])
        X_candidate_full = train_df[feature_list].copy()
        fold_metrics = []

        for fold_id, (fold_train_idx, fold_validation_idx) in enumerate(
            group_kfold.split(X_candidate_full, y_train, train_groups),
            start=1,
        ):
            print(
                f"  fold {fold_id}/{cv_splits} for "
                f"{candidate['feature_variant']} | {candidate['config_name']}"
            )
            X_fold_train = X_candidate_full.iloc[fold_train_idx].copy()
            X_fold_validation = X_candidate_full.iloc[fold_validation_idx].copy()
            y_fold_train = y_train.iloc[fold_train_idx].copy()
            y_fold_validation = y_train.iloc[fold_validation_idx].copy()

            model, metadata = fit_xgboost(
                X_fold_train,
                y_fold_train,
                X_fold_validation,
                y_fold_validation,
                config,
                device=xgboost_device,
            )
            _, X_fold_validation_prepared, _ = align_xgboost_frames(
                X_fold_train,
                X_fold_validation,
            )
            raw_predictions = model.predict(X_fold_validation_prepared)
            fold_predictions = convert_predictions_to_eur(
                raw_predictions,
                config["target_mode"],
                y_train_reference=y_fold_train,
            )
            fold_metrics.append(
                {
                    "fold": fold_id,
                    "validation_MAE": float(mean_absolute_error(y_fold_validation, fold_predictions)),
                    "validation_RMSE": float(
                        np.sqrt(mean_squared_error(y_fold_validation, fold_predictions))
                    ),
                    "validation_R2": float(r2_score(y_fold_validation, fold_predictions)),
                    "best_iteration": metadata.get("best_iteration"),
                }
            )

        fold_metrics_df = pd.DataFrame(fold_metrics)
        cv_rows.append(
            {
                "model_type": "xgboost",
                "feature_variant": candidate["feature_variant"],
                "config_name": candidate["config_name"],
                "target_mode": config["target_mode"],
                "feature_count": len(feature_list),
                "screen_validation_MAE": candidate["validation_MAE"],
                "screen_validation_RMSE": candidate["validation_RMSE"],
                "cv_mean_MAE": float(fold_metrics_df["validation_MAE"].mean()),
                "cv_std_MAE": float(fold_metrics_df["validation_MAE"].std(ddof=0)),
                "cv_mean_RMSE": float(fold_metrics_df["validation_RMSE"].mean()),
                "cv_mean_R2": float(fold_metrics_df["validation_R2"].mean()),
                "feature_names": feature_list,
                "config": config,
            }
        )

    cv_results_df = pd.DataFrame(cv_rows).sort_values(
        ["cv_mean_MAE", "cv_mean_RMSE", "screen_validation_MAE"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best_candidate = cv_results_df.iloc[0].to_dict()

    X_train_best = train_df[best_candidate["feature_names"]].copy()
    X_validation_best = validation_df[best_candidate["feature_names"]].copy()
    fitted_model, metadata = fit_xgboost(
        X_train_best,
        y_train,
        X_validation_best,
        y_validation,
        best_candidate["config"],
        device=xgboost_device,
    )
    _, X_validation_best_prepared, _ = align_xgboost_frames(X_train_best, X_validation_best)
    raw_predictions = fitted_model.predict(X_validation_best_prepared)
    validation_predictions = convert_predictions_to_eur(
        raw_predictions,
        best_candidate["config"]["target_mode"],
        y_train_reference=y_train,
    )

    final_summary = {
        **best_candidate,
        "best_iteration": metadata.get("best_iteration"),
        **regression_metrics(y_validation, validation_predictions),
    }
    return cv_results_df, final_summary


def evaluate_selected_random_forest_candidates(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    selected_candidates: list[dict[str, Any]],
    cv_splits: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run grouped cross-validation only on the best screened random-forest candidates."""

    if not selected_candidates:
        raise RuntimeError("No random-forest finalists were provided for grouped cross-validation.")

    y_train = train_df[TARGET_COLUMN].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()
    train_groups = train_df[GROUP_COLUMN]
    group_kfold = GroupKFold(n_splits=cv_splits)
    cv_rows = []

    for candidate_index, candidate in enumerate(selected_candidates, start=1):
        print(
            f"[cv {candidate_index}/{len(selected_candidates)}] "
            f"{candidate['feature_variant']} | {candidate['config_name']}"
        )
        feature_list = list(candidate["feature_names"])
        config = dict(candidate["config"])
        X_candidate_full = train_df[feature_list].copy()
        fold_metrics = []

        for fold_id, (fold_train_idx, fold_validation_idx) in enumerate(
            group_kfold.split(X_candidate_full, y_train, train_groups),
            start=1,
        ):
            print(
                f"  fold {fold_id}/{cv_splits} for "
                f"{candidate['feature_variant']} | {candidate['config_name']}"
            )
            X_fold_train = X_candidate_full.iloc[fold_train_idx].copy()
            X_fold_validation = X_candidate_full.iloc[fold_validation_idx].copy()
            y_fold_train = y_train.iloc[fold_train_idx].copy()
            y_fold_validation = y_train.iloc[fold_validation_idx].copy()

            model = fit_random_forest(X_fold_train, y_fold_train, config)
            raw_predictions = model.predict(X_fold_validation)
            fold_predictions = convert_predictions_to_eur(
                raw_predictions,
                config["target_mode"],
                y_train_reference=y_fold_train,
            )
            fold_metrics.append(
                {
                    "fold": fold_id,
                    "validation_MAE": float(mean_absolute_error(y_fold_validation, fold_predictions)),
                    "validation_RMSE": float(
                        np.sqrt(mean_squared_error(y_fold_validation, fold_predictions))
                    ),
                    "validation_R2": float(r2_score(y_fold_validation, fold_predictions)),
                }
            )

        fold_metrics_df = pd.DataFrame(fold_metrics)
        cv_rows.append(
            {
                "model_type": "random_forest",
                "feature_variant": candidate["feature_variant"],
                "config_name": candidate["config_name"],
                "target_mode": config["target_mode"],
                "feature_count": len(feature_list),
                "screen_validation_MAE": candidate["validation_MAE"],
                "screen_validation_RMSE": candidate["validation_RMSE"],
                "cv_mean_MAE": float(fold_metrics_df["validation_MAE"].mean()),
                "cv_std_MAE": float(fold_metrics_df["validation_MAE"].std(ddof=0)),
                "cv_mean_RMSE": float(fold_metrics_df["validation_RMSE"].mean()),
                "cv_mean_R2": float(fold_metrics_df["validation_R2"].mean()),
                "feature_names": feature_list,
                "config": config,
            }
        )

    cv_results_df = pd.DataFrame(cv_rows).sort_values(
        ["cv_mean_MAE", "cv_mean_RMSE", "screen_validation_MAE"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best_candidate = cv_results_df.iloc[0].to_dict()

    X_train_best = train_df[best_candidate["feature_names"]].copy()
    X_validation_best = validation_df[best_candidate["feature_names"]].copy()
    fitted_model = fit_random_forest(X_train_best, y_train, best_candidate["config"])
    raw_predictions = fitted_model.predict(X_validation_best)
    validation_predictions = convert_predictions_to_eur(
        raw_predictions,
        best_candidate["config"]["target_mode"],
        y_train_reference=y_train,
    )

    final_summary = {
        **best_candidate,
        **regression_metrics(y_validation, validation_predictions),
    }
    return cv_results_df, final_summary


def evaluate_model_candidates(
    model_type: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    configs: dict[str, dict[str, Any]],
    cv_splits: int,
    xgboost_device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run grouped cross-validation across every feature/config combination and keep the best result."""

    y_train = train_df[TARGET_COLUMN].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()
    train_groups = train_df[GROUP_COLUMN]
    group_kfold = GroupKFold(n_splits=cv_splits)

    cv_rows = []
    best_candidate: dict[str, Any] | None = None

    for feature_variant_name, feature_list in feature_sets.items():
        X_candidate_full = train_df[feature_list].copy()

        for config_name, config in configs.items():
            fold_metrics = []
            for fold_id, (fold_train_idx, fold_validation_idx) in enumerate(
                group_kfold.split(X_candidate_full, y_train, train_groups),
                start=1,
            ):
                X_fold_train = X_candidate_full.iloc[fold_train_idx].copy()
                X_fold_validation = X_candidate_full.iloc[fold_validation_idx].copy()
                y_fold_train = y_train.iloc[fold_train_idx].copy()
                y_fold_validation = y_train.iloc[fold_validation_idx].copy()

                if model_type == "random_forest":
                    model = fit_random_forest(X_fold_train, y_fold_train, config)
                    raw_predictions = model.predict(X_fold_validation)
                    metadata = {}
                else:
                    model, metadata = fit_xgboost(
                        X_fold_train,
                        y_fold_train,
                        X_fold_validation,
                        y_fold_validation,
                        config,
                        device=xgboost_device,
                    )
                    X_fold_train_prepared, X_fold_validation_prepared, _ = align_xgboost_frames(
                        X_fold_train,
                        X_fold_validation,
                    )
                    raw_predictions = model.predict(X_fold_validation_prepared)

                fold_predictions = convert_predictions_to_eur(
                    raw_predictions,
                    config["target_mode"],
                    y_train_reference=y_fold_train,
                )
                metrics = regression_metrics(y_fold_validation, fold_predictions)
                metrics["fold"] = fold_id
                metrics["best_iteration"] = metadata.get("best_iteration")
                fold_metrics.append(metrics)

            fold_metrics_df = pd.DataFrame(fold_metrics)
            row = {
                "model_type": model_type,
                "feature_variant": feature_variant_name,
                "config_name": config_name,
                "target_mode": config["target_mode"],
                "feature_count": len(feature_list),
                "cv_mean_MAE": float(fold_metrics_df["validation_MAE"].mean()),
                "cv_std_MAE": float(fold_metrics_df["validation_MAE"].std(ddof=0)),
                "cv_mean_RMSE": float(fold_metrics_df["validation_RMSE"].mean()),
                "cv_mean_R2": float(fold_metrics_df["validation_R2"].mean()),
            }
            cv_rows.append(row)
            if (
                best_candidate is None
                or row["cv_mean_MAE"] < best_candidate["cv_mean_MAE"]
                or (
                    row["cv_mean_MAE"] == best_candidate["cv_mean_MAE"]
                    and row["cv_mean_RMSE"] < best_candidate["cv_mean_RMSE"]
                )
            ):
                best_candidate = {
                    **row,
                    "feature_names": feature_list,
                    "config": config,
                }

    if best_candidate is None:
        raise RuntimeError(f"No valid candidates were found for model_type={model_type}.")

    X_train_best = train_df[best_candidate["feature_names"]].copy()
    X_validation_best = validation_df[best_candidate["feature_names"]].copy()

    if model_type == "random_forest":
        fitted_model = fit_random_forest(X_train_best, y_train, best_candidate["config"])
        raw_predictions = fitted_model.predict(X_validation_best)
    else:
        fitted_model, metadata = fit_xgboost(
            X_train_best,
            y_train,
            X_validation_best,
            y_validation,
            best_candidate["config"],
            device=xgboost_device,
        )
        _, X_validation_best_prepared, _ = align_xgboost_frames(X_train_best, X_validation_best)
        raw_predictions = fitted_model.predict(X_validation_best_prepared)
        best_candidate["best_iteration"] = metadata["best_iteration"]

    validation_predictions = convert_predictions_to_eur(
        raw_predictions,
        best_candidate["config"]["target_mode"],
        y_train_reference=y_train,
    )
    final_metrics = regression_metrics(y_validation, validation_predictions)

    final_summary = {
        **best_candidate,
        **final_metrics,
    }
    return pd.DataFrame(cv_rows), final_summary


def save_tuning_reports(
    output_dir: str | Path,
    model_reports: list[dict[str, Any]],
    cv_frames: list[pd.DataFrame],
) -> None:
    """Write the cross-validation table and the final tuning summary to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cv_results_df = pd.concat(cv_frames, ignore_index=True)
    cv_results_df.to_csv(output_path / "cv_results.csv", index=False)

    comparison_df = pd.DataFrame(model_reports).sort_values(
        ["validation_MAE", "validation_RMSE", "cv_mean_MAE"],
        ascending=[True, True, True],
    )
    comparison_df.to_csv(output_path / "model_comparison.csv", index=False)

    best_model_summary = comparison_df.iloc[0].to_dict()
    with (output_path / "best_tuning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(best_model_summary, handle, indent=2)
