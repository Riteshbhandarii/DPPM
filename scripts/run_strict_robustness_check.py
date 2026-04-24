#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

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
    fit_random_forest,
    fit_xgboost,
)


DEFAULT_DROP_FEATURES = [
    "observations_so_far",
    "days_since_first_seen_so_far",
    "first_seen_day_offset",
]

DEFAULT_GROUP_COLUMNS = ["part_name", "brand", "model", "oem_number"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict grouped-CV robustness checks for the selected final models."
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost", "both"],
        default="both",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=["datasets/splits/train_grouped.csv"],
    )
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument(
        "--rf-summary-path",
        default="artifacts/random_forest_tuning_strict/best_tuning_summary.json",
    )
    parser.add_argument(
        "--xgb-summary-path",
        default="artifacts/xgboost_tuning_strict/best_tuning_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/robustness_checks/strict_models",
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=DEFAULT_GROUP_COLUMNS,
    )
    parser.add_argument(
        "--drop-feature",
        action="append",
        dest="drop_features",
        default=None,
        help="Feature to remove in the robustness variant. Repeatable.",
    )
    parser.add_argument("--xgboost-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def evaluate_random_forest_variant(
    frame: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    cv_splits: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    def fit_predict(train_frame, validation_frame, selected_features):
        model = fit_random_forest(
            train_frame[selected_features].copy(),
            train_frame[TARGET_COLUMN].copy(),
            config,
        )
        raw_predictions = model.predict(validation_frame[selected_features].copy())
        return convert_predictions_to_eur(
            raw_predictions,
            config["target_mode"],
            y_train_reference=train_frame[TARGET_COLUMN],
        )

    return evaluate_grouped_cv(
        frame=frame,
        features=features,
        group_column="part_identity_group",
        cv_splits=cv_splits,
        fit_predict=fit_predict,
        model_name="random_forest",
    )


def evaluate_xgboost_variant(
    frame: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    cv_splits: int,
    device: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    def fit_predict(train_frame, validation_frame, selected_features):
        model, metadata = fit_xgboost(
            train_frame[selected_features].copy(),
            train_frame[TARGET_COLUMN].copy(),
            validation_frame[selected_features].copy(),
            validation_frame[TARGET_COLUMN].copy(),
            config,
            device=device,
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

    return evaluate_grouped_cv(
        frame=frame,
        features=features,
        group_column="part_identity_group",
        cv_splits=cv_splits,
        fit_predict=fit_predict,
        model_name="xgboost",
    )


def run_variant(
    model_name: str,
    variant_name: str,
    output_dir: Path,
    frame: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    source_summary: str,
    feature_variant: str | None,
    config_name: str | None,
    group_columns: list[str],
    cv_splits: int,
    xgboost_device: str,
) -> dict[str, Any]:
    if model_name == "random_forest":
        fold_metrics, cv_summary = evaluate_random_forest_variant(
            frame=frame,
            features=features,
            config=config,
            cv_splits=cv_splits,
        )
    elif model_name == "xgboost":
        fold_metrics, cv_summary = evaluate_xgboost_variant(
            frame=frame,
            features=features,
            config=config,
            cv_splits=cv_splits,
            device=xgboost_device,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    sanity_checks = split_sanity_checks(frame, features, "part_identity_group", cv_splits)
    summary = {
        "model": model_name,
        "variant": variant_name,
        "source_summary": source_summary,
        "feature_variant": feature_variant,
        "config_name": config_name,
        "target_mode": config["target_mode"],
        "feature_count": len(features),
        "group_columns": group_columns,
        "group_count": int(frame["part_identity_group"].nunique()),
        "row_count": int(len(frame)),
        "cv_summary": cv_summary,
        "quick_mode": False,
    }
    if model_name == "xgboost":
        summary["xgboost_device"] = xgboost_device

    write_model_outputs(output_dir, fold_metrics, summary, sanity_checks)
    return summary


def comparison_rows(
    baseline_summary: dict[str, Any],
    ablated_summary: dict[str, Any],
    ablated_features: list[str],
    dropped_present: list[str],
) -> pd.DataFrame:
    baseline_cv = baseline_summary["cv_summary"]
    ablated_cv = ablated_summary["cv_summary"]
    rows = []
    for metric in ["mean_MAE", "mean_RMSE", "mean_R2", "mean_median_AE"]:
        rows.append(
            {
                "metric": metric,
                "baseline": baseline_cv[metric],
                "ablated": ablated_cv[metric],
                "delta_ablated_minus_baseline": ablated_cv[metric] - baseline_cv[metric],
            }
        )
    metadata = pd.DataFrame(
        [
            {
                "metric": "feature_count",
                "baseline": baseline_summary["feature_count"],
                "ablated": ablated_summary["feature_count"],
                "delta_ablated_minus_baseline": (
                    ablated_summary["feature_count"] - baseline_summary["feature_count"]
                ),
            },
            {
                "metric": "dropped_feature_count",
                "baseline": 0,
                "ablated": len(dropped_present),
                "delta_ablated_minus_baseline": len(dropped_present),
            },
        ]
    )
    result = pd.concat([pd.DataFrame(rows), metadata], ignore_index=True)
    result.attrs["ablated_features"] = ablated_features
    result.attrs["dropped_present"] = dropped_present
    return result


def run_model(
    model_name: str,
    summary_path: str,
    frame: pd.DataFrame,
    output_root: Path,
    cv_splits: int,
    group_columns: list[str],
    drop_features: list[str],
    xgboost_device: str,
    quick: bool,
) -> None:
    model_summary = load_json(summary_path)
    baseline_features = list(model_summary["feature_names"])
    config = dict(model_summary["config"])
    config["model_params"] = dict(config["model_params"])

    if quick:
        if model_name == "random_forest":
            config["model_params"]["n_estimators"] = min(
                int(config["model_params"].get("n_estimators", 100)),
                80,
            )
        else:
            config["model_params"]["n_estimators"] = min(
                int(config["model_params"].get("n_estimators", 100)),
                250,
            )

    drop_set = set(drop_features)
    ablated_features = [feature for feature in baseline_features if feature not in drop_set]
    dropped_present = [feature for feature in baseline_features if feature in drop_set]

    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = run_variant(
        model_name=model_name,
        variant_name="baseline",
        output_dir=model_dir / "baseline",
        frame=frame,
        features=baseline_features,
        config=config,
        source_summary=summary_path,
        feature_variant=model_summary.get("feature_variant"),
        config_name=model_summary.get("config_name"),
        group_columns=group_columns,
        cv_splits=cv_splits,
        xgboost_device=xgboost_device,
    )
    baseline_summary["quick_mode"] = bool(quick)

    ablated_summary = run_variant(
        model_name=model_name,
        variant_name="drop_listing_history_context",
        output_dir=model_dir / "drop_listing_history_context",
        frame=frame,
        features=ablated_features,
        config=config,
        source_summary=summary_path,
        feature_variant=model_summary.get("feature_variant"),
        config_name=model_summary.get("config_name"),
        group_columns=group_columns,
        cv_splits=cv_splits,
        xgboost_device=xgboost_device,
    )
    ablated_summary["quick_mode"] = bool(quick)

    comparison = comparison_rows(
        baseline_summary=baseline_summary,
        ablated_summary=ablated_summary,
        ablated_features=ablated_features,
        dropped_present=dropped_present,
    )
    comparison.to_csv(model_dir / "comparison.csv", index=False)

    comparison_summary = {
        "model": model_name,
        "source_summary": summary_path,
        "baseline_variant": "baseline",
        "robustness_variant": "drop_listing_history_context",
        "requested_drop_features": drop_features,
        "dropped_features_present_in_model": dropped_present,
        "baseline_feature_count": len(baseline_features),
        "robustness_feature_count": len(ablated_features),
        "quick_mode": bool(quick),
        "baseline_cv_summary": baseline_summary["cv_summary"],
        "robustness_cv_summary": ablated_summary["cv_summary"],
    }
    with (model_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison_summary, handle, indent=2)

    print(f"Saved robustness check for {model_name} to: {model_dir.resolve()}")


def main() -> None:
    args = parse_args()
    drop_features = args.drop_features or list(DEFAULT_DROP_FEATURES)

    frame = load_split_frames(args.data_path)
    frame, group_columns = add_part_identity_group(frame, args.group_columns)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = (
        ["random_forest", "xgboost"]
        if args.model == "both"
        else [args.model]
    )

    for model_name in model_names:
        summary_path = (
            args.rf_summary_path if model_name == "random_forest" else args.xgb_summary_path
        )
        run_model(
            model_name=model_name,
            summary_path=summary_path,
            frame=frame,
            output_root=output_root,
            cv_splits=args.cv_splits,
            group_columns=group_columns,
            drop_features=drop_features,
            xgboost_device=args.xgboost_device,
            quick=args.quick,
        )


if __name__ == "__main__":
    main()
