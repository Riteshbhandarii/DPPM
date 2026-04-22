#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
from src.tree_modeling import (  # noqa: E402
    TARGET_COLUMN,
    align_xgboost_frames,
    convert_predictions_to_eur,
    fit_xgboost,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected XGBoost model with stricter part-identity grouped CV."
    )
    parser.add_argument("--data-path", action="append", default=["datasets/splits/train_grouped.csv"])
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--xgboost-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--summary-path", default="artifacts/xgboost_tuning/best_tuning_summary.json")
    parser.add_argument("--output-dir", default="artifacts/part_identity_evaluation/xgboost")
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=["part_name", "brand", "model", "oem_number"],
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_summary = load_json(args.summary_path)
    features = list(model_summary["feature_names"])
    config = dict(model_summary["config"])
    if args.quick:
        config = {
            **config,
            "model_params": {
                **config["model_params"],
                "n_estimators": min(int(config["model_params"].get("n_estimators", 100)), 250),
            },
        }

    frame = load_split_frames(args.data_path)
    frame, group_columns = add_part_identity_group(frame, args.group_columns)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise KeyError(f"Selected XGBoost features missing from data: {missing}")

    def fit_predict(train_frame, validation_frame, selected_features):
        model, metadata = fit_xgboost(
            train_frame[selected_features].copy(),
            train_frame[TARGET_COLUMN].copy(),
            validation_frame[selected_features].copy(),
            validation_frame[TARGET_COLUMN].copy(),
            config,
            device=args.xgboost_device,
        )
        _, validation_prepared, _ = align_xgboost_frames(
            train_frame[selected_features].copy(),
            validation_frame[selected_features].copy(),
            category_levels=metadata.get("category_levels"),
        )
        raw_predictions = model.predict(validation_prepared)
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
        model_name="xgboost",
    )
    summary = {
        "model": "xgboost",
        "source_summary": args.summary_path,
        "feature_variant": model_summary.get("feature_variant"),
        "config_name": model_summary.get("config_name"),
        "target_mode": config["target_mode"],
        "feature_count": len(features),
        "group_columns": group_columns,
        "group_count": int(frame["part_identity_group"].nunique()),
        "row_count": int(len(frame)),
        "cv_summary": cv_summary,
        "quick_mode": bool(args.quick),
        "xgboost_device": args.xgboost_device,
    }
    write_model_outputs(args.output_dir, fold_metrics, summary, sanity_checks)
    print(f"Saved XGBoost part-identity evaluation to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
