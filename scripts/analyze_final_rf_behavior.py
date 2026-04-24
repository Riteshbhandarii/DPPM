#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part_identity_evaluation import add_part_identity_group, load_split_frames  # noqa: E402
from src.tree_modeling import TARGET_COLUMN, convert_predictions_to_eur, fit_random_forest  # noqa: E402


DEFAULT_GROUP_COLUMNS = ["part_name", "brand", "model", "oem_number"]
DEFAULT_SUMMARY_PATH = "artifacts/random_forest_tuning_strict/best_tuning_summary.json"
DEFAULT_OUTPUT_DIR = "artifacts/final_model_behavior"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze final strict random-forest out-of-fold prediction behavior."
    )
    parser.add_argument("--data-path", action="append", default=["datasets/splits/train_grouped.csv"])
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--quick-estimators",
        type=int,
        default=None,
        help="Optional smoke-test override for RF n_estimators. Omit for final analysis.",
    )
    parser.add_argument("--group-columns", nargs="+", default=DEFAULT_GROUP_COLUMNS)
    return parser.parse_args()


def load_model_summary(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    absolute_errors = np.abs(errors)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "median_AE": float(np.median(absolute_errors)),
        "mean_price": float(np.mean(y_true)),
        "median_price": float(np.median(y_true)),
        "mean_absolute_percentage_error": float(
            np.mean(absolute_errors / np.clip(np.asarray(y_true, dtype=float), 1.0, None)) * 100
        ),
        "rows": int(len(y_true)),
    }


def build_oof_predictions(
    frame: pd.DataFrame,
    features: list[str],
    config: dict,
    cv_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupKFold(n_splits=cv_splits)
    groups = frame["part_identity_group"]
    y = frame[TARGET_COLUMN]
    prediction_rows = []
    fold_rows = []

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(frame[features], y, groups), start=1):
        train_frame = frame.iloc[train_idx].copy()
        validation_frame = frame.iloc[validation_idx].copy()

        model = fit_random_forest(
            train_frame[features].copy(),
            train_frame[TARGET_COLUMN].copy(),
            config,
        )
        raw_predictions = model.predict(validation_frame[features].copy())
        predictions = convert_predictions_to_eur(
            raw_predictions,
            config["target_mode"],
            y_train_reference=train_frame[TARGET_COLUMN],
        )

        fold_output = validation_frame.copy()
        fold_output["fold"] = fold
        fold_output["predicted_price"] = predictions
        fold_output["error"] = fold_output[TARGET_COLUMN] - fold_output["predicted_price"]
        fold_output["absolute_error"] = fold_output["error"].abs()
        fold_output["absolute_percentage_error"] = (
            fold_output["absolute_error"] / fold_output[TARGET_COLUMN].clip(lower=1.0) * 100
        )
        prediction_rows.append(fold_output)

        fold_metrics = metric_summary(fold_output[TARGET_COLUMN], fold_output["predicted_price"])
        fold_rows.append({"fold": fold, **fold_metrics})
        print(
            f"[final rf behavior] fold={fold} "
            f"MAE={fold_metrics['MAE']:.2f} RMSE={fold_metrics['RMSE']:.2f} "
            f"R2={fold_metrics['R2']:.4f} median_AE={fold_metrics['median_AE']:.2f}"
        )

    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(fold_rows)


def summarize_group(
    predictions: pd.DataFrame,
    group_column: str,
    min_group_size: int,
    top_n: int,
) -> pd.DataFrame:
    if group_column not in predictions.columns:
        return pd.DataFrame()
    grouped = (
        predictions.groupby(group_column, dropna=False, observed=False)
        .agg(
            rows=("absolute_error", "size"),
            mean_price=(TARGET_COLUMN, "mean"),
            median_price=(TARGET_COLUMN, "median"),
            mean_prediction=("predicted_price", "mean"),
            MAE=("absolute_error", "mean"),
            median_AE=("absolute_error", "median"),
            RMSE=("error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            mean_APE=("absolute_percentage_error", "mean"),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["rows"] >= min_group_size]
    return grouped.sort_values(["MAE", "rows"], ascending=[False, False]).head(top_n)


def add_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    output["price_bin"] = pd.qcut(output[TARGET_COLUMN], q=5, duplicates="drop")
    if "mileage" in output.columns:
        output["mileage_bin"] = pd.qcut(output["mileage"], q=5, duplicates="drop")
    if "year_mid" in output.columns:
        output["year_mid_bin"] = pd.qcut(output["year_mid"], q=5, duplicates="drop")
    return output


def write_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.scatter(predictions[TARGET_COLUMN], predictions["predicted_price"], s=8, alpha=0.35)
    upper = float(max(predictions[TARGET_COLUMN].max(), predictions["predicted_price"].max()))
    plt.plot([0, upper], [0, upper], color="black", linewidth=1)
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Final RF strict out-of-fold actual vs predicted")
    plt.tight_layout()
    plt.savefig(plot_dir / "actual_vs_predicted.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(predictions[TARGET_COLUMN], predictions["absolute_error"], s=8, alpha=0.35)
    plt.xlabel("Actual price")
    plt.ylabel("Absolute error")
    plt.title("Final RF strict absolute error by actual price")
    plt.tight_layout()
    plt.savefig(plot_dir / "absolute_error_by_price.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    predictions["absolute_error"].hist(bins=50)
    plt.xlabel("Absolute error")
    plt.ylabel("Rows")
    plt.title("Final RF strict absolute error distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "absolute_error_distribution.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_model_summary(args.summary_path)
    features = list(summary["feature_names"])
    config = dict(summary["config"])
    config["model_params"] = dict(config["model_params"])
    if args.quick_estimators is not None:
        config["model_params"]["n_estimators"] = args.quick_estimators

    frame = load_split_frames(args.data_path)
    frame, group_columns = add_part_identity_group(frame, args.group_columns)
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise KeyError(f"Final RF features missing from data: {missing}")

    predictions, fold_metrics = build_oof_predictions(
        frame=frame,
        features=features,
        config=config,
        cv_splits=args.cv_splits,
    )
    predictions = add_bins(predictions)

    overall = {
        "model_type": summary.get("model_type"),
        "feature_variant": summary.get("feature_variant"),
        "config_name": summary.get("config_name"),
        "target_mode": config["target_mode"],
        "feature_count": len(features),
        "cv_splits": args.cv_splits,
        "group_columns": group_columns,
        "metrics": metric_summary(predictions[TARGET_COLUMN], predictions["predicted_price"]),
    }

    keep_columns = [
        column
        for column in [
            "fold",
            "product_id",
            "part_identity_group",
            "part_name",
            "brand",
            "model",
            "category",
            "subcategory",
            "quality_grade",
            "mileage",
            "year_mid",
            TARGET_COLUMN,
            "predicted_price",
            "error",
            "absolute_error",
            "absolute_percentage_error",
            "price_bin",
            "mileage_bin",
            "year_mid_bin",
        ]
        if column in predictions.columns
    ]
    predictions[keep_columns].to_csv(output_dir / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    for column, filename in [
        ("price_bin", "error_by_price_bin.csv"),
        ("mileage_bin", "error_by_mileage_bin.csv"),
        ("year_mid_bin", "error_by_year_bin.csv"),
        ("part_name", "error_by_part_name.csv"),
        ("brand", "error_by_brand.csv"),
        ("category", "error_by_category.csv"),
        ("quality_grade", "error_by_quality_grade.csv"),
    ]:
        summarize_group(predictions, column, args.min_group_size, args.top_n).to_csv(
            output_dir / filename,
            index=False,
        )

    predictions.sort_values("absolute_error", ascending=False).head(args.top_n)[keep_columns].to_csv(
        output_dir / "worst_predictions.csv",
        index=False,
    )
    predictions.sort_values("error", ascending=False).head(args.top_n)[keep_columns].to_csv(
        output_dir / "largest_underpredictions.csv",
        index=False,
    )
    predictions.sort_values("error", ascending=True).head(args.top_n)[keep_columns].to_csv(
        output_dir / "largest_overpredictions.csv",
        index=False,
    )
    write_plots(predictions, output_dir)

    print(f"Saved final RF behavior analysis to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
