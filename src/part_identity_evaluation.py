from __future__ import annotations

"""Shared helpers for stricter part-identity grouped evaluation."""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.tree_modeling import DATE_COLUMNS, GROUP_COLUMN, TARGET_COLUMN


DEFAULT_PART_IDENTITY_COLUMNS = ["part_name", "brand", "model", "oem_number"]


def load_split_frames(paths: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    for column in DATE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    add_listing_date_offsets(frame)
    return frame


def add_listing_date_offsets(frame: pd.DataFrame) -> None:
    if not {"first_seen_date", "last_seen_date"}.issubset(frame.columns):
        return
    reference = frame["first_seen_date"].min()
    frame["first_seen_day_offset"] = (frame["first_seen_date"] - reference).dt.days
    frame["last_seen_day_offset"] = (frame["last_seen_date"] - reference).dt.days
    frame["listing_midpoint_day_offset"] = (
        frame["first_seen_day_offset"] + frame["last_seen_day_offset"]
    ) / 2


def normalize_group_value(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("__missing__")
        .str.strip()
        .str.lower()
        .replace("", "__missing__")
    )


def add_part_identity_group(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    group_column: str = "part_identity_group",
) -> tuple[pd.DataFrame, list[str]]:
    group_columns = list(columns or DEFAULT_PART_IDENTITY_COLUMNS)
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Part-identity columns missing from data: {missing}")

    output = frame.copy()
    normalized = [normalize_group_value(output[column]) for column in group_columns]
    group_values = normalized[0]
    for series in normalized[1:]:
        group_values = group_values.str.cat(series, sep="|")
    output[group_column] = group_values
    return output, group_columns


def metric_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "median_AE": float(np.median(np.abs(errors))),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true, ddof=0)),
        "prediction_std": float(np.std(y_pred, ddof=0)),
        "n_rows": int(len(y_true)),
    }


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in ["MAE", "RMSE", "R2", "median_AE"]:
        summary[f"mean_{metric}"] = float(fold_metrics[metric].mean())
        summary[f"std_{metric}"] = float(fold_metrics[metric].std(ddof=0))
        summary[f"min_{metric}"] = float(fold_metrics[metric].min())
        summary[f"max_{metric}"] = float(fold_metrics[metric].max())
    summary["folds"] = int(len(fold_metrics))
    return summary


def split_sanity_checks(
    frame: pd.DataFrame,
    features: list[str],
    group_column: str,
    cv_splits: int,
) -> dict[str, Any]:
    splitter = GroupKFold(n_splits=cv_splits)
    X = frame[features]
    y = frame[TARGET_COLUMN]
    groups = frame[group_column]
    duplicate_columns = [column for column in features + [TARGET_COLUMN] if column in frame.columns]
    fold_rows = []
    total_duplicate_fingerprints = 0
    total_group_overlap = 0

    for fold, (train_index, validation_index) in enumerate(splitter.split(X, y, groups), start=1):
        train_frame = frame.iloc[train_index]
        validation_frame = frame.iloc[validation_index]
        train_groups = set(train_frame[group_column])
        validation_groups = set(validation_frame[group_column])
        group_overlap = len(train_groups & validation_groups)

        train_fingerprints = set(
            pd.util.hash_pandas_object(train_frame[duplicate_columns], index=False)
        )
        validation_fingerprints = set(
            pd.util.hash_pandas_object(validation_frame[duplicate_columns], index=False)
        )
        duplicate_fingerprints = len(train_fingerprints & validation_fingerprints)

        total_group_overlap += group_overlap
        total_duplicate_fingerprints += duplicate_fingerprints
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_frame)),
                "validation_rows": int(len(validation_frame)),
                "train_groups": int(len(train_groups)),
                "validation_groups": int(len(validation_groups)),
                "group_overlap": int(group_overlap),
                "duplicate_modeling_rows_with_target": int(duplicate_fingerprints),
            }
        )

    return {
        "group_overlap": {
            "status": "PASS" if total_group_overlap == 0 else "FAIL",
            "total_overlap": int(total_group_overlap),
            "interpretation": (
                "Part-identity groups do not cross folds."
                if total_group_overlap == 0
                else "Part-identity groups cross folds."
            ),
        },
        "cross_fold_exact_duplicates": {
            "status": "PASS" if total_duplicate_fingerprints == 0 else "WARN",
            "total_duplicate_fingerprints": int(total_duplicate_fingerprints),
            "interpretation": (
                "No exact modeling-row plus target duplicates cross strict folds."
                if total_duplicate_fingerprints == 0
                else "Exact modeling-row plus target duplicates still cross strict folds."
            ),
        },
        "folds": fold_rows,
    }


def evaluate_grouped_cv(
    frame: pd.DataFrame,
    features: list[str],
    group_column: str,
    cv_splits: int,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, list[str]], np.ndarray],
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = GroupKFold(n_splits=cv_splits)
    X = frame[features]
    y = frame[TARGET_COLUMN]
    groups = frame[group_column]
    rows = []

    for fold, (train_index, validation_index) in enumerate(splitter.split(X, y, groups), start=1):
        train_frame = frame.iloc[train_index].copy()
        validation_frame = frame.iloc[validation_index].copy()
        predictions = fit_predict(train_frame, validation_frame, features)
        metrics = metric_dict(validation_frame[TARGET_COLUMN], predictions)
        row = {
            "model": model_name,
            "fold": fold,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "train_groups": int(train_frame[group_column].nunique()),
            "validation_groups": int(validation_frame[group_column].nunique()),
            **metrics,
        }
        rows.append(row)
        print(
            f"[{model_name}] fold={fold} "
            f"MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} "
            f"R2={metrics['R2']:.4f} median_AE={metrics['median_AE']:.2f}"
        )

    fold_metrics = pd.DataFrame(rows)
    return fold_metrics, summarize_fold_metrics(fold_metrics)


def write_model_outputs(
    output_dir: str | Path,
    fold_metrics: pd.DataFrame,
    summary: dict[str, Any],
    sanity_checks: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_path / "fold_metrics.csv", index=False)
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (output_path / "sanity_checks.json").write_text(
        json.dumps(sanity_checks, indent=2, default=str),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
