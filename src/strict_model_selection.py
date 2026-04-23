from __future__ import annotations

"""Strict part-identity model-selection helpers."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.part_identity_evaluation import add_part_identity_group, load_split_frames
from src.tree_modeling import (
    TARGET_COLUMN,
    align_xgboost_frames,
    convert_predictions_to_eur,
    fit_random_forest,
    fit_xgboost,
)


DEFAULT_PART_IDENTITY_COLUMNS = ["part_name", "brand", "model", "oem_number"]
STRICT_GROUP_COLUMN = "part_identity_group"


def load_strict_tuning_frame(
    data_paths: list[str | Path],
    group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load one or more split files and attach the strict part-identity grouping."""

    frame = load_split_frames(data_paths)
    return add_part_identity_group(frame, columns=group_columns, group_column=STRICT_GROUP_COLUMN)


def strict_cv_metric_summary(fold_metrics_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate strict grouped-CV metrics into one candidate-level summary."""

    return {
        "cv_mean_MAE": float(fold_metrics_df["validation_MAE"].mean()),
        "cv_std_MAE": float(fold_metrics_df["validation_MAE"].std(ddof=0)),
        "cv_mean_RMSE": float(fold_metrics_df["validation_RMSE"].mean()),
        "cv_std_RMSE": float(fold_metrics_df["validation_RMSE"].std(ddof=0)),
        "cv_mean_R2": float(fold_metrics_df["validation_R2"].mean()),
        "cv_std_R2": float(fold_metrics_df["validation_R2"].std(ddof=0)),
        "cv_mean_median_AE": float(fold_metrics_df["median_AE"].mean()),
        "cv_std_median_AE": float(fold_metrics_df["median_AE"].std(ddof=0)),
        "cv_folds": int(len(fold_metrics_df)),
    }


def _strict_fold_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "validation_MAE": float(mean_absolute_error(y_true, y_pred)),
        "validation_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "validation_R2": float(r2_score(y_true, y_pred)),
        "median_AE": float(np.median(np.abs(errors))),
    }


def _make_xgboost_inner_split(
    train_frame: pd.DataFrame,
    group_column: str,
    random_state: int = 42,
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a train-only grouped split for XGBoost early stopping."""

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    inner_train_idx, early_stopping_idx = next(
        splitter.split(train_frame, groups=train_frame[group_column])
    )
    inner_train_df = train_frame.iloc[inner_train_idx].copy()
    early_stopping_df = train_frame.iloc[early_stopping_idx].copy()
    return inner_train_df, early_stopping_df


def evaluate_random_forest_candidates_strict(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    configs: dict[str, dict[str, Any]],
    cv_splits: int,
    top_k_finalists: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Rank random-forest candidates by strict part-identity grouped CV."""

    y_full = frame[TARGET_COLUMN].copy()
    groups = frame[STRICT_GROUP_COLUMN]
    group_kfold = GroupKFold(n_splits=cv_splits)
    total_candidates = len(feature_sets) * len(configs)
    current_candidate = 0
    candidate_rows = []

    for feature_variant_name, feature_list in feature_sets.items():
        X_candidate_full = frame[feature_list].copy()

        for config_name, config in configs.items():
            current_candidate += 1
            print(
                f"[strict rf {current_candidate}/{total_candidates}] "
                f"{feature_variant_name} | {config_name}"
            )
            fold_metrics = []
            for fold_id, (fold_train_idx, fold_validation_idx) in enumerate(
                group_kfold.split(X_candidate_full, y_full, groups),
                start=1,
            ):
                X_fold_train = X_candidate_full.iloc[fold_train_idx].copy()
                X_fold_validation = X_candidate_full.iloc[fold_validation_idx].copy()
                y_fold_train = y_full.iloc[fold_train_idx].copy()
                y_fold_validation = y_full.iloc[fold_validation_idx].copy()

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
                        **_strict_fold_metrics(y_fold_validation, fold_predictions),
                    }
                )

            fold_metrics_df = pd.DataFrame(fold_metrics)
            candidate_rows.append(
                {
                    "model_type": "random_forest",
                    "selection_mode": "strict_part_identity_grouped_cv",
                    "feature_variant": feature_variant_name,
                    "config_name": config_name,
                    "target_mode": config["target_mode"],
                    "feature_count": len(feature_list),
                    "feature_names": feature_list,
                    "config": config,
                    **strict_cv_metric_summary(fold_metrics_df),
                }
            )

    results_df = pd.DataFrame(candidate_rows).sort_values(
        ["cv_mean_MAE", "cv_mean_RMSE", "cv_std_MAE", "feature_count"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    finalists = results_df.head(top_k_finalists).to_dict(orient="records")
    return results_df, finalists


def evaluate_xgboost_candidates_strict(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    configs: dict[str, dict[str, Any]],
    cv_splits: int,
    xgboost_device: str,
    top_k_finalists: int,
    early_stopping_test_size: float = 0.1,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Rank XGBoost candidates by strict part-identity grouped CV."""

    y_full = frame[TARGET_COLUMN].copy()
    groups = frame[STRICT_GROUP_COLUMN]
    group_kfold = GroupKFold(n_splits=cv_splits)
    total_candidates = len(feature_sets) * len(configs)
    current_candidate = 0
    candidate_rows = []

    for feature_variant_name, feature_list in feature_sets.items():
        X_candidate_full = frame[feature_list].copy()

        for config_name, config in configs.items():
            current_candidate += 1
            print(
                f"[strict xgb {current_candidate}/{total_candidates}] "
                f"{feature_variant_name} | {config_name}"
            )
            fold_metrics = []
            best_iterations: list[int] = []

            for fold_id, (fold_train_idx, fold_validation_idx) in enumerate(
                group_kfold.split(X_candidate_full, y_full, groups),
                start=1,
            ):
                fold_train_frame = frame.iloc[fold_train_idx].copy()
                fold_validation_frame = frame.iloc[fold_validation_idx].copy()
                inner_train_frame, early_stopping_frame = _make_xgboost_inner_split(
                    fold_train_frame,
                    group_column=STRICT_GROUP_COLUMN,
                    random_state=42 + fold_id,
                    test_size=early_stopping_test_size,
                )

                model, metadata = fit_xgboost(
                    inner_train_frame[feature_list].copy(),
                    inner_train_frame[TARGET_COLUMN].copy(),
                    early_stopping_frame[feature_list].copy(),
                    early_stopping_frame[TARGET_COLUMN].copy(),
                    config,
                    device=xgboost_device,
                )
                _, validation_prepared, _ = align_xgboost_frames(
                    inner_train_frame[feature_list].copy(),
                    fold_validation_frame[feature_list].copy(),
                    category_levels=metadata.get("category_levels"),
                )
                raw_predictions = model.predict(validation_prepared)
                fold_predictions = convert_predictions_to_eur(
                    raw_predictions,
                    config["target_mode"],
                    y_train_reference=inner_train_frame[TARGET_COLUMN],
                )
                fold_metrics.append(
                    {
                        "fold": fold_id,
                        **_strict_fold_metrics(
                            fold_validation_frame[TARGET_COLUMN],
                            fold_predictions,
                        ),
                    }
                )
                if metadata.get("best_iteration") is not None:
                    best_iterations.append(int(metadata["best_iteration"]))

            fold_metrics_df = pd.DataFrame(fold_metrics)
            candidate_rows.append(
                {
                    "model_type": "xgboost",
                    "selection_mode": "strict_part_identity_grouped_cv",
                    "feature_variant": feature_variant_name,
                    "config_name": config_name,
                    "target_mode": config["target_mode"],
                    "feature_count": len(feature_list),
                    "feature_names": feature_list,
                    "config": config,
                    "best_iteration_mean": (
                        float(np.mean(best_iterations)) if best_iterations else None
                    ),
                    **strict_cv_metric_summary(fold_metrics_df),
                }
            )

    results_df = pd.DataFrame(candidate_rows).sort_values(
        ["cv_mean_MAE", "cv_mean_RMSE", "cv_std_MAE", "feature_count"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    finalists = results_df.head(top_k_finalists).to_dict(orient="records")
    return results_df, finalists


def save_strict_tuning_reports(
    output_dir: str | Path,
    broad_results_df: pd.DataFrame,
    refinement_results_df: pd.DataFrame,
    group_columns: list[str],
    cv_splits: int,
    source_paths: list[str | Path],
) -> dict[str, Any]:
    """Save broad and refinement strict-tuning results and return the best summary."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    broad_results_df.drop(columns=["feature_names", "config"], errors="ignore").to_csv(
        output_path / "screening_results.csv",
        index=False,
    )
    refinement_results_df.drop(columns=["feature_names", "config"], errors="ignore").to_csv(
        output_path / "refinement_results.csv",
        index=False,
    )

    combined_df = pd.concat([broad_results_df, refinement_results_df], ignore_index=True)
    combined_df = combined_df.sort_values(
        ["cv_mean_MAE", "cv_mean_RMSE", "cv_std_MAE", "feature_count"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    combined_df.drop(columns=["feature_names", "config"], errors="ignore").to_csv(
        output_path / "model_comparison.csv",
        index=False,
    )

    best_summary = combined_df.iloc[0].to_dict()
    best_summary["group_columns"] = list(group_columns)
    best_summary["cv_splits"] = int(cv_splits)
    best_summary["source_paths"] = [str(path) for path in source_paths]

    with (output_path / "best_tuning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2)

    return best_summary
