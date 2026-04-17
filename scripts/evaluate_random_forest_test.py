#!/usr/bin/env python3
from __future__ import annotations

# Standard library imports used by the CLI entrypoint.
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Add the repository root so the shared modeling module can be imported on Puhti.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Shared random-forest helpers used for the final held-out test evaluation.
from src.tree_modeling import (
    TARGET_COLUMN,
    convert_predictions_to_eur,
    fit_random_forest,
    regression_metrics,
)


def parse_args():
    """Define command-line arguments for the final random-forest test evaluation."""

    parser = argparse.ArgumentParser(
        description="Train the selected random forest on train+validation and evaluate on test."
    )
    parser.add_argument("--train-path", default="datasets/splits/train_grouped.csv")
    parser.add_argument("--validation-path", default="datasets/splits/validation_grouped.csv")
    parser.add_argument("--test-path", default="datasets/splits/test_grouped.csv")
    parser.add_argument(
        "--tuning-summary-path",
        default="artifacts/random_forest_tuning/best_tuning_summary.json",
    )
    parser.add_argument("--output-dir", default="artifacts/random_forest_test")
    return parser.parse_args()


def main():
    """Fit the selected random forest on train plus validation and score it on the held-out test split."""

    # Load the fixed best configuration selected during the tuning workflow.
    args = parse_args()
    summary_path = Path(args.tuning_summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if summary.get("model_type") != "random_forest":
        raise ValueError("The supplied tuning summary does not describe a random forest model.")

    feature_names = list(summary["feature_names"])
    config = dict(summary["config"])

    # Combine train and validation because model selection is already complete.
    train_df = pd.read_csv(args.train_path)
    validation_df = pd.read_csv(args.validation_path)
    test_df = pd.read_csv(args.test_path)
    fit_df = pd.concat([train_df, validation_df], ignore_index=True)

    X_fit = fit_df[feature_names].copy()
    y_fit = fit_df[TARGET_COLUMN].copy()
    X_test = test_df[feature_names].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    # Fit the selected model once on the full development data and evaluate it on the held-out test split.
    model = fit_random_forest(X_fit, y_fit, config)
    raw_predictions = model.predict(X_test)
    test_predictions = convert_predictions_to_eur(
        raw_predictions,
        config["target_mode"],
        y_train_reference=y_fit,
    )
    test_metrics = regression_metrics(y_test, test_predictions)

    final_summary = {
        "model_type": "random_forest",
        "evaluation_split": "test_grouped",
        "feature_variant": summary["feature_variant"],
        "config_name": summary["config_name"],
        "target_mode": config["target_mode"],
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "config": config,
        "fit_rows": int(len(fit_df)),
        "test_rows": int(len(test_df)),
        "test_MAE": test_metrics["validation_MAE"],
        "test_RMSE": test_metrics["validation_RMSE"],
        "test_R2": test_metrics["validation_R2"],
    }

    # Save the held-out test result in its own artifact folder.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    print("Random forest held-out test metrics")
    print(json.dumps(final_summary, indent=2, default=str))
    print(f"Saved test metrics to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
