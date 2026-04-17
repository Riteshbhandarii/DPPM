#!/usr/bin/env python3

# Standard library imports used by the CLI entrypoint.
import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

# Add the repository root so the shared modeling module can be imported on Puhti.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Shared random-forest helper used for the final export.
from src.tree_modeling import TARGET_COLUMN, fit_random_forest


def parse_args():
    """Define command-line arguments for the final random-forest export."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the selected random forest and save both the evaluated "
            "train+validation bundle and the full-data deployment bundle."
        )
    )
    parser.add_argument("--train-path", default="datasets/splits/train_grouped.csv")
    parser.add_argument("--validation-path", default="datasets/splits/validation_grouped.csv")
    parser.add_argument("--test-path", default="datasets/splits/test_grouped.csv")
    parser.add_argument(
        "--tuning-summary-path",
        default="artifacts/random_forest_tuning/best_tuning_summary.json",
    )
    parser.add_argument(
        "--test-metrics-path",
        default="artifacts/random_forest_test/test_metrics.json",
    )
    parser.add_argument("--output-dir", default="artifacts/random_forest_final")
    return parser.parse_args()


def save_bundle(output_dir, model, metadata, reference_rows):
    """Save one fitted model bundle with metadata and reference rows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    reference_rows.to_csv(output_dir / "reference_rows.csv", index=False)


def build_bundle_metadata(summary, config, feature_names, fit_df, split_label, test_metrics):
    """Create the metadata file saved next to the fitted model bundle."""

    metadata = {
        "model_type": "random_forest",
        "artifact_version": 1,
        "bundle_split": split_label,
        "feature_variant": summary["feature_variant"],
        "config_name": summary["config_name"],
        "target_mode": config["target_mode"],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "config": config,
        "fit_rows": int(len(fit_df)),
        "target_reference_max": float(fit_df[TARGET_COLUMN].max()),
        "trusted_validation_metrics": {
            "validation_MAE": summary["validation_MAE"],
            "validation_RMSE": summary["validation_RMSE"],
            "validation_R2": summary["validation_R2"],
        },
    }

    if test_metrics is not None:
        metadata["held_out_test_metrics"] = test_metrics

    return metadata


def main():
    """Fit and save the final random-forest bundles for evaluation and deployment."""

    args = parse_args()

    summary_path = Path(args.tuning_summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("model_type") != "random_forest":
        raise ValueError("The supplied tuning summary does not describe a random forest model.")

    test_metrics = None
    test_metrics_path = Path(args.test_metrics_path)
    if test_metrics_path.exists():
        test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))

    feature_names = list(summary["feature_names"])
    config = dict(summary["config"])

    train_df = pd.read_csv(args.train_path)
    validation_df = pd.read_csv(args.validation_path)
    test_df = pd.read_csv(args.test_path)

    # Bundle 1 matches the evaluated thesis setup trained on train plus validation only.
    development_df = pd.concat([train_df, validation_df], ignore_index=True)
    development_model = fit_random_forest(
        development_df[feature_names].copy(),
        development_df[TARGET_COLUMN].copy(),
        config,
    )
    development_metadata = build_bundle_metadata(
        summary=summary,
        config=config,
        feature_names=feature_names,
        fit_df=development_df,
        split_label="train_plus_validation",
        test_metrics=test_metrics,
    )

    # Bundle 2 uses all available grouped data after evaluation is already complete.
    full_data_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    full_data_model = fit_random_forest(
        full_data_df[feature_names].copy(),
        full_data_df[TARGET_COLUMN].copy(),
        config,
    )
    full_data_metadata = build_bundle_metadata(
        summary=summary,
        config=config,
        feature_names=feature_names,
        fit_df=full_data_df,
        split_label="train_plus_validation_plus_test",
        test_metrics=test_metrics,
    )

    output_dir = Path(args.output_dir)
    save_bundle(
        output_dir / "development_bundle",
        development_model,
        development_metadata,
        development_df[feature_names + [TARGET_COLUMN]].copy(),
    )
    save_bundle(
        output_dir / "full_data_bundle",
        full_data_model,
        full_data_metadata,
        full_data_df[feature_names + [TARGET_COLUMN]].copy(),
    )

    print("Saved random forest model bundles")
    print(
        json.dumps(
            {
                "development_bundle": str((output_dir / "development_bundle").resolve()),
                "full_data_bundle": str((output_dir / "full_data_bundle").resolve()),
                "feature_variant": summary["feature_variant"],
                "config_name": summary["config_name"],
                "feature_count": len(feature_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
