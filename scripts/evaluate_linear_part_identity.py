#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part_identity_evaluation import (  # noqa: E402
    add_part_identity_group,
    evaluate_grouped_cv,
    load_json,
    load_split_frames,
    split_sanity_checks,
    write_model_outputs,
)
from src.tree_modeling import TARGET_COLUMN, convert_predictions_to_eur  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected Ridge baseline with stricter part-identity grouped CV."
    )
    parser.add_argument("--data-path", action="append", default=["datasets/splits/train_grouped.csv"])
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument(
        "--rf-summary-path",
        default="artifacts/random_forest_tuning/best_tuning_summary.json",
        help="Used only to reuse the trusted 66-column feature set from the model comparison table.",
    )
    parser.add_argument("--output-dir", default="artifacts/part_identity_evaluation/linear")
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=["part_name", "brand", "model", "oem_number"],
    )
    return parser.parse_args()


def build_linear_pipeline(X_train, estimator, onehot_min_frequency=5) -> Pipeline:
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=onehot_min_frequency)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clone(estimator))])


def main() -> None:
    args = parse_args()
    rf_summary = load_json(args.rf_summary_path)
    features = list(rf_summary["feature_names"])
    config = {
        "target_mode": "log",
        "estimator": "Ridge(alpha=0.1)",
        "onehot_min_frequency": 5,
    }

    frame = load_split_frames(args.data_path)
    frame, group_columns = add_part_identity_group(frame, args.group_columns)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise KeyError(f"Linear baseline features missing from data: {missing}")

    def fit_predict(train_frame, validation_frame, selected_features):
        model = build_linear_pipeline(
            train_frame[selected_features].copy(),
            estimator=Ridge(alpha=0.1),
            onehot_min_frequency=config["onehot_min_frequency"],
        )
        model.fit(train_frame[selected_features].copy(), np.log(train_frame[TARGET_COLUMN]))
        raw_predictions = model.predict(validation_frame[selected_features].copy())
        return convert_predictions_to_eur(
            raw_predictions,
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
        model_name="linear_ridge",
    )
    summary = {
        "model": "linear_ridge",
        "feature_source": args.rf_summary_path,
        "feature_variant": "trusted_recommended_features_without_listing_dates",
        "config": config,
        "feature_count": len(features),
        "group_columns": group_columns,
        "group_count": int(frame["part_identity_group"].nunique()),
        "row_count": int(len(frame)),
        "cv_summary": cv_summary,
    }
    write_model_outputs(args.output_dir, fold_metrics, summary, sanity_checks)
    print(f"Saved linear part-identity evaluation to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
