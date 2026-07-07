"""Model comparison and tuning on the strict connected-component split (issues #36/#37).

Mirrors the original grouped-split workflow with the grouping unit upgraded from
``product_id`` to strict connected components (src/strict_protocol.py):

Stage 1 - model comparison (notebook 01): each of the four models (Ridge baseline,
Random Forest, XGBoost, CatBoost) is trained on the strict training split with its
known anchor configurations across the trusted feature variants and scored once on
the strict validation split - the same fixed-validation procedure the original
per-model training notebooks used. No random search at this stage.

Stage 2 - tuning (after the comparison): the top two models get the full search
(same config generators and seeds as the original workflow) ranked by
component-grouped cross-validation inside the strict training split. Boosted models
early-stop on an inner component-grouped carve, never on the scored fold/split.

Stage 3 - final holdout (notebook 02, run once): the single winner is refit on
train + validation and evaluated once on the untouched strict test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.strict_protocol import COMPONENT_GROUP_COLUMN, add_component_group
from src.tree_modeling import (
    TARGET_COLUMN,
    align_xgboost_frames,
    build_feature_catalog,
    convert_predictions_to_eur,
    fit_random_forest,
    fit_xgboost,
    generate_random_forest_refinement_configs,
    generate_random_forest_search_configs,
    generate_xgboost_refinement_configs,
    generate_xgboost_search_configs,
    load_training_data,
    prepare_target,
)

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - handled at fit time
    CatBoostRegressor = None


MODELS = ("ridge", "random_forest", "xgboost", "catboost")
MODEL_TO_CATALOG_KIND = {
    "ridge": "linear",
    "random_forest": "random_forest",
    "xgboost": "xgboost",
    "catboost": "catboost",
}
EARLY_STOPPING_MODELS = {"xgboost", "catboost"}
CV_RANK_COLUMNS = ["cv_mean_MAE", "cv_mean_RMSE", "cv_std_MAE", "feature_count"]
SCREEN_RANK_COLUMNS = ["validation_MAE", "validation_RMSE", "feature_count"]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def json_default(value: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python so summaries round-trip cleanly."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def euro_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
        f"{prefix}_median_AE": float(np.median(np.abs(errors))),
    }


# ---------------------------------------------------------------------------
# Config spaces. Anchor configs are the known configurations carried over from the
# original workflow (tree_modeling anchors for RF/XGBoost; the previously selected
# Ridge and CatBoost setups for the other two). The comparison stage uses anchors
# only; the tuning stage widens them with the original random-search generators.
# ---------------------------------------------------------------------------

def generate_ridge_search_configs() -> dict[str, dict[str, Any]]:
    """Log-target Ridge grid around the previously selected alpha=0.05 baseline."""

    configs: dict[str, dict[str, Any]] = {}
    for alpha in [0.01, 0.05, 0.1, 0.3, 1.0, 3.0]:
        name = f"log_ridge_alpha_{str(alpha).replace('.', '_')}"
        configs[name] = {
            "target_mode": "log",
            "onehot_min_frequency": 5,
            "model_params": {"alpha": alpha},
        }
    return configs


def generate_ridge_refinement_configs(base_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_alpha = float(base_config["model_params"]["alpha"])
    configs = {"refinement_anchor": base_config}
    for factor in [0.5, 0.75, 1.5, 2.0]:
        alpha = round(base_alpha * factor, 6)
        name = f"refinement_alpha_{str(alpha).replace('.', '_')}"
        configs[name] = {
            "target_mode": base_config["target_mode"],
            "onehot_min_frequency": base_config["onehot_min_frequency"],
            "model_params": {"alpha": alpha},
        }
    return configs


def generate_catboost_search_configs() -> dict[str, dict[str, Any]]:
    """Anchor from the previous CatBoost evaluation plus modest known variants."""

    return {
        "raw_rmse_depth7": {
            "target_mode": "raw",
            "model_params": {"loss_function": "RMSE", "iterations": 2000, "learning_rate": 0.035, "depth": 7},
        },
        "raw_rmse_depth6": {
            "target_mode": "raw",
            "model_params": {"loss_function": "RMSE", "iterations": 2000, "learning_rate": 0.05, "depth": 6},
        },
        "raw_mae_depth7": {
            "target_mode": "raw",
            "model_params": {"loss_function": "MAE", "iterations": 2000, "learning_rate": 0.035, "depth": 7},
        },
        "log_rmse_depth7": {
            "target_mode": "log",
            "model_params": {"loss_function": "RMSE", "iterations": 2000, "learning_rate": 0.035, "depth": 7},
        },
        "raw_rmse_depth8": {
            "target_mode": "raw",
            "model_params": {"loss_function": "RMSE", "iterations": 2500, "learning_rate": 0.03, "depth": 8},
        },
    }


def generate_catboost_refinement_configs(base_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_params = dict(base_config["model_params"])
    configs = {"refinement_anchor": base_config}
    for suffix, overrides in {
        "lr_down": {"learning_rate": round(base_params["learning_rate"] * 0.7, 6)},
        "lr_up": {"learning_rate": round(base_params["learning_rate"] * 1.3, 6)},
        "depth_down": {"depth": max(4, int(base_params["depth"]) - 1)},
        "depth_up": {"depth": min(10, int(base_params["depth"]) + 1)},
    }.items():
        configs[f"refinement_{suffix}"] = {
            "target_mode": base_config["target_mode"],
            "model_params": {**base_params, **overrides},
        }
    return configs


def anchor_configs_for(model: str) -> dict[str, dict[str, Any]]:
    """Known configurations only - no random search. Used by the comparison stage."""

    if model == "ridge":
        return generate_ridge_search_configs()
    if model == "random_forest":
        return generate_random_forest_search_configs(random_trials=0, random_seed=42)
    if model == "xgboost":
        return generate_xgboost_search_configs(random_trials=0, random_seed=42)
    if model == "catboost":
        return generate_catboost_search_configs()
    raise ValueError(f"Unsupported model: {model}")


def search_configs_for(model: str, random_trials: int, random_seed: int) -> dict[str, dict[str, Any]]:
    if model == "ridge":
        return generate_ridge_search_configs()
    if model == "random_forest":
        return generate_random_forest_search_configs(random_trials=random_trials, random_seed=random_seed)
    if model == "xgboost":
        return generate_xgboost_search_configs(random_trials=random_trials, random_seed=random_seed)
    if model == "catboost":
        return generate_catboost_search_configs()
    raise ValueError(f"Unsupported model: {model}")


def refinement_configs_for(
    model: str, base_config: dict[str, Any], refinement_trials: int, random_seed: int
) -> dict[str, dict[str, Any]]:
    if model == "ridge":
        return generate_ridge_refinement_configs(base_config)
    if model == "random_forest":
        return generate_random_forest_refinement_configs(
            base_config=base_config, refinement_trials=refinement_trials, random_seed=random_seed
        )
    if model == "xgboost":
        return generate_xgboost_refinement_configs(
            base_config=base_config, refinement_trials=refinement_trials, random_seed=random_seed
        )
    if model == "catboost":
        return generate_catboost_refinement_configs(base_config)
    raise ValueError(f"Unsupported model: {model}")


# ---------------------------------------------------------------------------
# Fit/predict per model
# ---------------------------------------------------------------------------

def build_ridge_pipeline(X_train: pd.DataFrame, config: dict[str, Any]) -> Pipeline:
    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=config["onehot_min_frequency"],
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clone(Ridge(**config["model_params"]))),
        ]
    )


def prepare_catboost_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared = frame.copy()
    datetime_columns = prepared.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    if datetime_columns:
        prepared = prepared.drop(columns=datetime_columns)
    for column in prepared.select_dtypes(include=["bool"]).columns:
        prepared[column] = prepared[column].astype(int)
    categorical_columns = prepared.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical_columns:
        prepared[column] = prepared[column].astype("string").fillna("__missing__").astype(str)
    return prepared, categorical_columns


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_early_stopping: pd.DataFrame,
    y_early_stopping: pd.Series,
    config: dict[str, Any],
) -> Any:
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed. Install catboost before CatBoost training.")

    X_train_prepared, cat_features = prepare_catboost_frame(X_train)
    X_early_stopping_prepared, _ = prepare_catboost_frame(X_early_stopping)
    X_early_stopping_prepared = X_early_stopping_prepared.reindex(columns=X_train_prepared.columns)

    model = CatBoostRegressor(
        **config["model_params"],
        random_seed=42,
        early_stopping_rounds=120,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        X_train_prepared,
        prepare_target(y_train, config["target_mode"]),
        cat_features=cat_features,
        eval_set=(
            X_early_stopping_prepared,
            prepare_target(y_early_stopping, config["target_mode"]),
        ),
    )
    return model


def make_early_stopping_carve(
    train_df: pd.DataFrame,
    group_column: str,
    random_state: int = 42,
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the (fold-)train frame into inner-train and early-stopping parts by component."""

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    inner_train_idx, early_stopping_idx = next(
        splitter.split(train_df, groups=train_df[group_column])
    )
    return train_df.iloc[inner_train_idx].copy(), train_df.iloc[early_stopping_idx].copy()


def fit_and_predict(
    model: str,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    group_column: str,
    early_stopping_seed: int = 42,
) -> np.ndarray:
    """Fit one candidate on ``train_df`` and return euro-scale predictions for ``predict_df``.

    XGBoost and CatBoost early-stop on an inner component-grouped carve of ``train_df``;
    ``predict_df`` is never shown to the fit.
    """

    y_train = train_df[TARGET_COLUMN]

    if model == "ridge":
        pipeline = build_ridge_pipeline(train_df[features].copy(), config)
        pipeline.fit(train_df[features].copy(), prepare_target(y_train, config["target_mode"]))
        raw_predictions = pipeline.predict(predict_df[features].copy())
        y_reference = y_train
    elif model == "random_forest":
        pipeline = fit_random_forest(train_df[features].copy(), y_train.copy(), config)
        raw_predictions = pipeline.predict(predict_df[features].copy())
        y_reference = y_train
    elif model == "xgboost":
        inner_train_df, early_stopping_df = make_early_stopping_carve(
            train_df, group_column, random_state=early_stopping_seed
        )
        booster, metadata = fit_xgboost(
            inner_train_df[features].copy(),
            inner_train_df[TARGET_COLUMN].copy(),
            early_stopping_df[features].copy(),
            early_stopping_df[TARGET_COLUMN].copy(),
            config,
            device="cpu",
        )
        _, predict_prepared, _ = align_xgboost_frames(
            inner_train_df[features].copy(),
            predict_df[features].copy(),
            category_levels=metadata.get("category_levels"),
        )
        raw_predictions = booster.predict(predict_prepared)
        y_reference = inner_train_df[TARGET_COLUMN]
    elif model == "catboost":
        inner_train_df, early_stopping_df = make_early_stopping_carve(
            train_df, group_column, random_state=early_stopping_seed
        )
        booster = fit_catboost(
            inner_train_df[features].copy(),
            inner_train_df[TARGET_COLUMN].copy(),
            early_stopping_df[features].copy(),
            early_stopping_df[TARGET_COLUMN].copy(),
            config,
        )
        train_prepared, _ = prepare_catboost_frame(inner_train_df[features].copy())
        predict_prepared, _ = prepare_catboost_frame(predict_df[features].copy())
        predict_prepared = predict_prepared.reindex(columns=train_prepared.columns)
        raw_predictions = booster.predict(predict_prepared)
        y_reference = inner_train_df[TARGET_COLUMN]
    else:
        raise ValueError(f"Unsupported model: {model}")

    return convert_predictions_to_eur(raw_predictions, config["target_mode"], y_train_reference=y_reference)


# ---------------------------------------------------------------------------
# Stage 1: model comparison (fixed strict validation, anchors only)
# ---------------------------------------------------------------------------

def model_feature_sets(model: str, train_df: pd.DataFrame) -> dict[str, list[str]]:
    catalog_frame = train_df.drop(columns=[COMPONENT_GROUP_COLUMN], errors="ignore")
    catalog = build_feature_catalog(catalog_frame, model_kind=MODEL_TO_CATALOG_KIND[model])
    return catalog["feature_sets"]


def screen_model(
    model: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    configs: dict[str, dict[str, Any]],
    group_column: str = COMPONENT_GROUP_COLUMN,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Fit each candidate on strict train, score once on strict validation."""

    y_validation = validation_df[TARGET_COLUMN]
    total = len(feature_sets) * len(configs)
    current = 0
    rows = []

    for feature_variant, features in feature_sets.items():
        for config_name, config in configs.items():
            current += 1
            progress(f"[{model} {current}/{total}] {feature_variant} | {config_name}")
            predictions = fit_and_predict(
                model, train_df, validation_df, features, config, group_column
            )
            rows.append(
                {
                    "model_type": model,
                    "feature_variant": feature_variant,
                    "config_name": config_name,
                    "target_mode": config["target_mode"],
                    "feature_count": len(features),
                    **euro_metrics(y_validation, predictions, prefix="validation"),
                    "feature_names": features,
                    "config": config,
                }
            )

    return pd.DataFrame(rows).sort_values(SCREEN_RANK_COLUMNS).reset_index(drop=True)


def run_model_comparison(
    model: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: str | Path,
    quick: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Stage 1 for one model: anchor configs x trusted feature variants on strict validation.

    Saves the per-candidate results and the model's best setup; returns the best summary.
    """

    feature_sets = model_feature_sets(model, train_df)
    configs = anchor_configs_for(model)
    if quick:
        feature_sets = dict(list(feature_sets.items())[:2])
        configs = cap_configs_for_quick_mode(configs)

    results_df = screen_model(model, train_df, validation_df, feature_sets, configs, progress=progress)

    best = results_df.iloc[0].to_dict()
    best["stage"] = "model_comparison_fixed_validation"
    best["train_rows"] = int(len(train_df))
    best["validation_rows"] = int(len(validation_df))
    best["quick_mode"] = bool(quick)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.drop(columns=["feature_names", "config"], errors="ignore").to_csv(
        output_path / "comparison_results.csv", index=False
    )
    write_summary_json(output_path / "best_comparison_summary.json", best)
    return best


def tuned_comparison_table(best_summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Rank the tuned models by component-grouped CV (the stage-2 decision table)."""

    rows = []
    for model, summary in best_summaries.items():
        rows.append(
            {
                "model_type": model,
                "feature_variant": summary.get("feature_variant"),
                "config_name": summary.get("config_name"),
                "target_mode": summary.get("target_mode"),
                "feature_count": summary.get("feature_count"),
                "cv_mean_MAE": summary.get("cv_mean_MAE"),
                "cv_std_MAE": summary.get("cv_std_MAE"),
                "cv_mean_RMSE": summary.get("cv_mean_RMSE"),
                "cv_mean_R2": summary.get("cv_mean_R2"),
                "cv_mean_median_AE": summary.get("cv_mean_median_AE"),
                "screen_validation_MAE": summary.get("screen_validation_MAE"),
            }
        )
    return pd.DataFrame(rows).sort_values(["cv_mean_MAE", "cv_mean_RMSE"]).reset_index(drop=True)


def model_comparison_table(best_summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, summary in best_summaries.items():
        rows.append(
            {
                "model_type": model,
                "feature_variant": summary.get("feature_variant"),
                "config_name": summary.get("config_name"),
                "target_mode": summary.get("target_mode"),
                "feature_count": summary.get("feature_count"),
                "validation_MAE": summary.get("validation_MAE"),
                "validation_RMSE": summary.get("validation_RMSE"),
                "validation_R2": summary.get("validation_R2"),
                "validation_median_AE": summary.get("validation_median_AE"),
            }
        )
    return pd.DataFrame(rows).sort_values(["validation_MAE", "validation_RMSE"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 2: tuning for the top models (component-grouped CV; used after comparison)
# ---------------------------------------------------------------------------

def select_finalists(screening_df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    """Keep the best candidate per feature variant first, then fill by overall rank."""

    finalists: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    best_per_variant = (
        screening_df.sort_values(["validation_MAE", "validation_RMSE"])
        .groupby("feature_variant", as_index=False)
        .head(1)
    )
    for _, row in best_per_variant.iterrows():
        key = (row["feature_variant"], row["config_name"])
        if key not in seen:
            finalists.append(row.to_dict())
            seen.add(key)

    for _, row in screening_df.iterrows():
        if len(finalists) >= top_k:
            break
        key = (row["feature_variant"], row["config_name"])
        if key not in seen:
            finalists.append(row.to_dict())
            seen.add(key)

    return finalists[:top_k]


def cv_candidates(
    model: str,
    train_df: pd.DataFrame,
    candidates: list[dict[str, Any]],
    cv_splits: int,
    stage: str,
    group_column: str = COMPONENT_GROUP_COLUMN,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Component-grouped CV inside strict train for the given candidates."""

    y_full = train_df[TARGET_COLUMN]
    groups = train_df[group_column]
    group_kfold = GroupKFold(n_splits=cv_splits)
    rows = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        features = list(candidate["feature_names"])
        config = dict(candidate["config"])
        progress(
            f"[cv {model} {stage} {candidate_index}/{len(candidates)}] "
            f"{candidate['feature_variant']} | {candidate['config_name']}"
        )
        fold_metrics = []
        for fold_id, (fold_train_idx, fold_validation_idx) in enumerate(
            group_kfold.split(train_df, y_full, groups), start=1
        ):
            fold_train_df = train_df.iloc[fold_train_idx]
            fold_validation_df = train_df.iloc[fold_validation_idx]
            predictions = fit_and_predict(
                model,
                fold_train_df,
                fold_validation_df,
                features,
                config,
                group_column,
                early_stopping_seed=42 + fold_id,
            )
            fold_metrics.append(
                {
                    "fold": fold_id,
                    **euro_metrics(
                        fold_validation_df[TARGET_COLUMN], predictions, prefix="validation"
                    ),
                }
            )

        fold_df = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "model_type": model,
                "stage": stage,
                "selection_mode": "strict_component_grouped_cv",
                "feature_variant": candidate["feature_variant"],
                "config_name": candidate["config_name"],
                "target_mode": config["target_mode"],
                "feature_count": len(features),
                "screen_validation_MAE": candidate.get("validation_MAE"),
                "cv_mean_MAE": float(fold_df["validation_MAE"].mean()),
                "cv_std_MAE": float(fold_df["validation_MAE"].std(ddof=0)),
                "cv_mean_RMSE": float(fold_df["validation_RMSE"].mean()),
                "cv_std_RMSE": float(fold_df["validation_RMSE"].std(ddof=0)),
                "cv_mean_R2": float(fold_df["validation_R2"].mean()),
                "cv_mean_median_AE": float(fold_df["validation_median_AE"].mean()),
                "cv_folds": int(len(fold_df)),
                "feature_names": features,
                "config": config,
            }
        )

    return pd.DataFrame(rows).sort_values(CV_RANK_COLUMNS).reset_index(drop=True)


def cap_configs_for_quick_mode(configs: dict[str, dict[str, Any]], max_configs: int = 2) -> dict[str, dict[str, Any]]:
    """Shrink a config space for smoke tests: few configs, tiny ensembles."""

    capped = {}
    for name, config in list(configs.items())[:max_configs]:
        params = dict(config["model_params"])
        for key in ("n_estimators", "iterations"):
            if key in params:
                params[key] = min(int(params[key]), 60)
        capped[name] = {**config, "model_params": params}
    return capped


def run_model_tuning(
    model: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: str | Path,
    random_trials: int = 24,
    refinement_trials: int = 16,
    top_k_finalists: int = 10,
    cv_splits: int = 4,
    random_seed: int = 42,
    quick: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Stage 2 for one model: screening -> finalist CV -> refinement CV. Saves artifacts."""

    feature_sets = model_feature_sets(model, train_df)
    configs = search_configs_for(model, random_trials=random_trials, random_seed=random_seed)
    if quick:
        feature_sets = dict(list(feature_sets.items())[:2])
        configs = cap_configs_for_quick_mode(configs)

    screening_df = screen_model(model, train_df, validation_df, feature_sets, configs, progress=progress)
    finalists = select_finalists(screening_df, top_k=(3 if quick else top_k_finalists))
    finalist_cv_df = cv_candidates(model, train_df, finalists, cv_splits, stage="finalist_cv", progress=progress)

    best_finalist = finalist_cv_df.iloc[0].to_dict()
    refinement = refinement_configs_for(
        model, dict(best_finalist["config"]), refinement_trials, random_seed + 1000
    )
    if quick:
        refinement = cap_configs_for_quick_mode(refinement)
    refinement_candidates = [
        {
            "feature_variant": best_finalist["feature_variant"],
            "config_name": config_name,
            "feature_names": list(best_finalist["feature_names"]),
            "config": config,
            "validation_MAE": None,
        }
        for config_name, config in refinement.items()
    ]
    refinement_cv_df = cv_candidates(
        model, train_df, refinement_candidates, cv_splits, stage="refinement_cv", progress=progress
    )

    combined_cv_df = (
        pd.concat([finalist_cv_df, refinement_cv_df], ignore_index=True)
        .sort_values(CV_RANK_COLUMNS)
        .reset_index(drop=True)
    )
    best_summary = combined_cv_df.iloc[0].to_dict()
    best_summary["cv_group_column"] = COMPONENT_GROUP_COLUMN
    best_summary["cv_splits"] = int(cv_splits)
    best_summary["quick_mode"] = bool(quick)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    drop_columns = ["feature_names", "config"]
    screening_df.drop(columns=drop_columns, errors="ignore").to_csv(
        output_path / "screening_results.csv", index=False
    )
    finalist_cv_df.drop(columns=drop_columns, errors="ignore").to_csv(
        output_path / "finalist_cv_results.csv", index=False
    )
    refinement_cv_df.drop(columns=drop_columns, errors="ignore").to_csv(
        output_path / "refinement_cv_results.csv", index=False
    )
    combined_cv_df.drop(columns=drop_columns, errors="ignore").to_csv(
        output_path / "cv_results_combined.csv", index=False
    )
    write_summary_json(output_path / "best_tuning_summary.json", best_summary)
    return best_summary


# ---------------------------------------------------------------------------
# Dummy anchors
# ---------------------------------------------------------------------------

def dummy_baselines(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    """Trivial anchors: global median and per-subcategory median price."""

    y_validation = validation_df[TARGET_COLUMN]
    rows = []

    global_median = float(train_df[TARGET_COLUMN].median())
    rows.append(
        {
            "model_type": "dummy_global_median",
            "description": "Predict the training-set median price for every listing.",
            **euro_metrics(y_validation, np.full(len(validation_df), global_median), prefix="validation"),
        }
    )

    if "subcategory" in train_df.columns:
        medians = train_df.groupby("subcategory")[TARGET_COLUMN].median()
        predictions = (
            validation_df["subcategory"].map(medians).fillna(global_median).to_numpy(dtype=float)
        )
        rows.append(
            {
                "model_type": "dummy_subcategory_median",
                "description": "Predict the training-set median price of the listing's subcategory.",
                **euro_metrics(y_validation, predictions, prefix="validation"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_strict_training_frames(
    train_path: str | Path = "datasets/splits_strict/train_strict.csv",
    validation_path: str | Path = "datasets/splits_strict/validation_strict.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load strict train/validation with shared date-offset reference and component groups."""

    prepared = load_training_data(train_path, validation_path)
    train_df = add_component_group(prepared.train_df)
    return train_df, prepared.validation_df
