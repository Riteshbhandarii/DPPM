#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_modeling import (
    RANDOM_FOREST_CONFIGS,
    build_feature_catalog,
    evaluate_model_candidates,
    load_training_data,
    save_tuning_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune random forest on the grouped train/validation splits."
    )
    parser.add_argument("--train-path", default="datasets/splits/train_grouped.csv")
    parser.add_argument("--validation-path", default="datasets/splits/validation_grouped.csv")
    parser.add_argument("--output-dir", default="artifacts/random_forest_tuning")
    parser.add_argument("--cv-splits", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_data = load_training_data(args.train_path, args.validation_path)
    feature_catalog = build_feature_catalog(prepared_data.train_df, model_kind="random_forest")

    cv_results_df, summary = evaluate_model_candidates(
        model_type="random_forest",
        train_df=prepared_data.train_df,
        validation_df=prepared_data.validation_df,
        feature_sets=feature_catalog["feature_sets"],
        configs=RANDOM_FOREST_CONFIGS,
        cv_splits=args.cv_splits,
        xgboost_device="cpu",
    )

    save_tuning_reports(
        output_dir=args.output_dir,
        model_reports=[summary],
        cv_frames=[cv_results_df],
    )

    print("Best random forest config")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"config", "feature_names"}
            },
            indent=2,
            default=str,
        )
    )
    print(f"Saved reports to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
