#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part_identity_evaluation import load_split_frames  # noqa: E402
from src.tree_modeling import TARGET_COLUMN, fit_random_forest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected random forest and save both a development bundle "
            "and a full-data deployment bundle."
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
        "--evaluation-summary-path",
        default=None,
        help=(
            "Optional JSON with evaluation metrics used for uncertainty metadata. "
            "Examples: artifacts/part_identity_evaluation/random_forest/summary.json "
            "or artifacts/final_model_behavior/overall_metrics.json."
        ),
    )
    parser.add_argument(
        "--test-metrics-path",
        default="artifacts/random_forest_test/test_metrics.json",
        help="Optional held-out metrics JSON. Ignored if the file does not exist.",
    )
    parser.add_argument("--output-dir", default="artifacts/random_forest_final")
    return parser.parse_args()


def load_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def load_frame(path: str | Path) -> pd.DataFrame:
    return load_split_frames([path])


def save_bundle(
    output_dir: Path,
    model,
    metadata: dict[str, Any],
    reference_rows: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    reference_rows.to_csv(output_dir / "reference_rows.csv", index=False)


def extract_validation_metrics(summary: dict[str, Any], evaluation: dict[str, Any] | None) -> dict[str, float]:
    if evaluation:
        if "cv_summary" in evaluation:
            cv = evaluation["cv_summary"]
            return {
                "validation_MAE": float(cv["mean_MAE"]),
                "validation_RMSE": float(cv["mean_RMSE"]),
                "validation_R2": float(cv["mean_R2"]),
            }
        if "metrics" in evaluation:
            metrics = evaluation["metrics"]
            return {
                "validation_MAE": float(metrics["MAE"]),
                "validation_RMSE": float(metrics["RMSE"]),
                "validation_R2": float(metrics["R2"]),
            }

    return {
        "validation_MAE": float(summary.get("cv_mean_MAE", 0.0)),
        "validation_RMSE": float(summary.get("cv_mean_RMSE", 0.0)),
        "validation_R2": float(summary.get("cv_mean_R2", 0.0)),
    }


def build_bundle_metadata(
    summary: dict[str, Any],
    config: dict[str, Any],
    feature_names: list[str],
    fit_df: pd.DataFrame,
    split_label: str,
    validation_metrics: dict[str, float],
    held_out_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model_type": "random_forest",
        "artifact_version": 1,
        "bundle_split": split_label,
        "selection_mode": summary.get("selection_mode"),
        "feature_variant": summary.get("feature_variant"),
        "config_name": summary.get("config_name"),
        "target_mode": config["target_mode"],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "config": config,
        "fit_rows": int(len(fit_df)),
        "target_reference_max": float(fit_df[TARGET_COLUMN].max()),
        "trusted_validation_metrics": validation_metrics,
        "group_columns": summary.get("group_columns", []),
    }
    if held_out_metrics is not None:
        metadata["held_out_test_metrics"] = held_out_metrics
    return metadata


def main() -> None:
    args = parse_args()

    summary = json.loads(Path(args.tuning_summary_path).read_text(encoding="utf-8"))
    if summary.get("model_type") != "random_forest":
        raise ValueError("The supplied tuning summary does not describe a random forest model.")

    evaluation_summary = load_json_if_exists(args.evaluation_summary_path)
    held_out_metrics = load_json_if_exists(args.test_metrics_path)

    feature_names = list(summary["feature_names"])
    config = dict(summary["config"])
    config["model_params"] = dict(config["model_params"])
    validation_metrics = extract_validation_metrics(summary, evaluation_summary)

    train_df = load_frame(args.train_path)
    validation_df = load_frame(args.validation_path)
    test_df = load_frame(args.test_path)

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
        validation_metrics=validation_metrics,
        held_out_metrics=held_out_metrics,
    )

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
        validation_metrics=validation_metrics,
        held_out_metrics=held_out_metrics,
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
                "feature_variant": summary.get("feature_variant"),
                "config_name": summary.get("config_name"),
                "feature_count": len(feature_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
