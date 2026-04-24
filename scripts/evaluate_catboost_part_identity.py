#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - runtime dependency
    CatBoostRegressor = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part_identity_evaluation import (  # noqa: E402
    add_part_identity_group,
    evaluate_grouped_cv,
    load_split_frames,
    split_sanity_checks,
    write_model_outputs,
)
from src.tree_modeling import TARGET_COLUMN, build_feature_catalog, convert_predictions_to_eur  # noqa: E402


CATBOOST_CONFIG = {
    "target_mode": "raw",
    "config_name": "raw_rmse_depth7",
    "model_params": {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "iterations": 2000,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 10,
        "random_strength": 1.0,
        "bagging_temperature": 0.75,
        "border_count": 254,
        "one_hot_max_size": 10,
        "rsm": 0.8,
        "boosting_type": "Plain",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the notebook-selected CatBoost setup with stricter part-identity grouped CV."
    )
    parser.add_argument("--data-path", action="append", default=["datasets/splits/train_grouped.csv"])
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument(
        "--feature-variant",
        default="trusted_recommended_features_without_date_offsets_without_oem_number",
        help="Feature variant from the CatBoost strict feature catalog.",
    )
    parser.add_argument("--output-dir", default="artifacts/part_identity_evaluation/catboost")
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=["part_name", "brand", "model", "oem_number"],
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def prepare_catboost_frame(frame):
    prepared = frame.copy()
    datetime_columns = prepared.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    if datetime_columns:
        prepared = prepared.drop(columns=datetime_columns)
    object_columns = prepared.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in object_columns:
        prepared[column] = prepared[column].astype("string")
    bool_columns = prepared.select_dtypes(include=["bool"]).columns.tolist()
    for column in bool_columns:
        prepared[column] = prepared[column].astype(int)
    return prepared


def catboost_cat_features(prepared_frame):
    return prepared_frame.select_dtypes(include=["string"]).columns.tolist()


def main() -> None:
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed. Install catboost before running this evaluation.")

    args = parse_args()
    config = {
        **CATBOOST_CONFIG,
        "model_params": dict(CATBOOST_CONFIG["model_params"]),
    }
    if args.quick:
        config["model_params"]["iterations"] = min(int(config["model_params"]["iterations"]), 300)

    frame = load_split_frames(args.data_path)
    frame, group_columns = add_part_identity_group(frame, args.group_columns)
    feature_catalog = build_feature_catalog(
        frame.drop(columns=["part_identity_group"]),
        model_kind="catboost",
    )
    try:
        features = list(feature_catalog["feature_sets"][args.feature_variant])
    except KeyError as exc:
        available = sorted(feature_catalog["feature_sets"])
        raise KeyError(
            f"Unknown CatBoost feature variant {args.feature_variant!r}. Available: {available}"
        ) from exc
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise KeyError(f"CatBoost features missing from data: {missing}")

    def fit_predict(train_frame, validation_frame, selected_features):
        X_train = prepare_catboost_frame(train_frame[selected_features].copy())
        X_validation = prepare_catboost_frame(validation_frame[selected_features].copy())
        cat_features = catboost_cat_features(X_train)
        model_params = {
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            **config["model_params"],
        }
        model = CatBoostRegressor(**model_params)
        model.fit(
            X_train,
            train_frame[TARGET_COLUMN].copy(),
            cat_features=cat_features,
            eval_set=(X_validation, validation_frame[TARGET_COLUMN].copy()),
            use_best_model=True,
            early_stopping_rounds=120,
        )
        raw_predictions = model.predict(X_validation)
        return convert_predictions_to_eur(
            np.asarray(raw_predictions, dtype=float),
            config["target_mode"],
            y_train_reference=train_frame[TARGET_COLUMN],
        )

    sanity_checks = split_sanity_checks(frame, features, "part_identity_group", args.cv_splits)
    fold_metrics, cv_summary = evaluate_grouped_cv(
        frame=frame,
        features=features,
        group_column="part_identity_group",
        cv_splits=args.cv_splits,
        fit_predict=fit_predict,
        model_name="catboost",
    )
    summary = {
        "model": "catboost",
        "selection_mode": "strict_part_identity_grouped_cv",
        "feature_source": "build_feature_catalog(model_kind='catboost')",
        "feature_variant": args.feature_variant,
        "config_name": config["config_name"],
        "target_mode": config["target_mode"],
        "config": config,
        "feature_count": len(features),
        "group_columns": group_columns,
        "group_count": int(frame["part_identity_group"].nunique()),
        "row_count": int(len(frame)),
        "cv_summary": cv_summary,
        "quick_mode": bool(args.quick),
    }
    write_model_outputs(args.output_dir, fold_metrics, summary, sanity_checks)
    print(f"Saved CatBoost part-identity evaluation to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
