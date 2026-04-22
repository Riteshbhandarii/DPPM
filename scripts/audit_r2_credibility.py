#!/usr/bin/env python3
from __future__ import annotations

"""Adversarial audit for unusually high R2 in the grouped tree-model pipeline."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_modeling import (  # noqa: E402
    COMMON_LEAKAGE_RISK_FEATURES,
    GROUP_COLUMN,
    LISTING_DATE_OFFSET_FEATURES,
    LISTING_DYNAMICS_FEATURES,
    TARGET_COLUMN,
    TRAFICOM_EXTENDED_CANDIDATES,
    TRAFICOM_FEATURES,
    REGISTRY_LIFECYCLE_CANDIDATES,
    convert_predictions_to_eur,
    fit_random_forest,
    load_training_data,
)


DATE_LIKE_TOKENS = ("date", "day_offset", "year", "first_seen", "last_seen")
ID_LIKE_TOKENS = ("id", "oem", "number", "key")
PRICE_BINS = [0, 25, 50, 100, 250, 500, 1000, np.inf]
PRICE_BIN_LABELS = ["0-25", "25-50", "50-100", "100-250", "250-500", "500-1000", "1000+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether high grouped-validation R2 is robust to leakage checks and destructive tests."
    )
    parser.add_argument("--train-path", default="datasets/splits/train_grouped.csv")
    parser.add_argument("--validation-path", default="datasets/splits/validation_grouped.csv")
    parser.add_argument("--test-path", default="datasets/splits/test_grouped.csv")
    parser.add_argument(
        "--rf-summary-path",
        default="artifacts/random_forest_tuning/best_tuning_summary.json",
    )
    parser.add_argument(
        "--xgb-summary-path",
        default="artifacts/xgboost_tuning/best_tuning_summary.json",
    )
    parser.add_argument("--output-dir", default="artifacts/r2_audit")
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer trees for local smoke checks. Do not use quick output as thesis evidence.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def add_date_offsets_for_test(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    date_columns = ["first_seen_date", "last_seen_date", "scrape_date"]
    for frame in (train_df, validation_df, test_df):
        for column in date_columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
    candidates = [
        frame["first_seen_date"].min()
        for frame in (train_df, validation_df)
        if "first_seen_date" in frame.columns
    ]
    if not candidates:
        return
    reference = min(candidates)
    for frame in (test_df,):
        if {"first_seen_date", "last_seen_date"}.issubset(frame.columns):
            frame["first_seen_day_offset"] = (frame["first_seen_date"] - reference).dt.days
            frame["last_seen_day_offset"] = (frame["last_seen_date"] - reference).dt.days
            frame["listing_midpoint_day_offset"] = (
                frame["first_seen_day_offset"] + frame["last_seen_day_offset"]
            ) / 2


def shrink_config_for_quick(config: dict[str, Any]) -> dict[str, Any]:
    quick_config = json.loads(json.dumps(config))
    quick_config["model_params"]["n_estimators"] = min(
        int(quick_config["model_params"].get("n_estimators", 100)),
        80,
    )
    return quick_config


def metric_dict(y_true: pd.Series, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        f"{prefix}MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}R2": float(r2_score(y_true, y_pred)),
        f"{prefix}median_AE": float(np.median(np.abs(errors))),
        f"{prefix}target_mean": float(np.mean(y_true)),
        f"{prefix}target_std": float(np.std(y_true, ddof=0)),
        f"{prefix}prediction_std": float(np.std(y_pred, ddof=0)),
        f"{prefix}n_rows": int(len(y_true)),
    }


def fit_predict_rf(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    model = fit_random_forest(train_df[features].copy(), train_df[TARGET_COLUMN].copy(), config)
    raw_predictions = model.predict(eval_df[features].copy())
    predictions = convert_predictions_to_eur(
        raw_predictions,
        config["target_mode"],
        y_train_reference=train_df[TARGET_COLUMN],
    )
    return predictions, metric_dict(eval_df[TARGET_COLUMN], predictions)


def print_interpretation(label: str, status: str, detail: str) -> None:
    print(f"[{status}] {label}: {detail}")


def grouped_split_checks(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    split_frames = {"train": train_df, "validation": validation_df, "test": test_df}
    split_groups = {name: set(frame[GROUP_COLUMN].dropna()) for name, frame in split_frames.items()}

    overlaps = {
        "train_validation": sorted(split_groups["train"] & split_groups["validation"])[:20],
        "train_test": sorted(split_groups["train"] & split_groups["test"])[:20],
        "validation_test": sorted(split_groups["validation"] & split_groups["test"])[:20],
    }
    overlap_count = sum(len(values) for values in overlaps.values())
    checks["product_id_overlap"] = {
        "status": "PASS" if overlap_count == 0 else "FAIL",
        "overlap_examples": overlaps,
        "interpretation": "Grouped split integrity holds." if overlap_count == 0 else "Same product_id appears in multiple splits.",
    }

    duplicate_columns = [column for column in features + [TARGET_COLUMN] if column in train_df.columns]
    duplicate_counts = {}
    fingerprints = {}
    for name, frame in split_frames.items():
        fingerprints[name] = set(pd.util.hash_pandas_object(frame[duplicate_columns], index=False))
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        duplicate_counts[f"{left}_{right}"] = int(len(fingerprints[left] & fingerprints[right]))
    total_duplicates = sum(duplicate_counts.values())
    checks["cross_split_exact_duplicates"] = {
        "status": "PASS" if total_duplicates == 0 else "FAIL",
        "duplicate_counts": duplicate_counts,
        "interpretation": (
            "No exact modeling-row plus target duplicates cross split boundaries."
            if total_duplicates == 0
            else "Exact modeling-row plus target duplicates cross split boundaries."
        ),
    }

    target_feature_hits = [
        column
        for column in features
        if column == TARGET_COLUMN or column.lower() in {"price", "target", "y", "eur_price"}
    ]
    checks["target_in_features"] = {
        "status": "PASS" if not target_feature_hits else "FAIL",
        "columns": target_feature_hits,
        "interpretation": "No direct target column is selected." if not target_feature_hits else "Target-like column is selected.",
    }

    numeric_proxy_hits = []
    for column in features:
        if column not in train_df.columns or not pd.api.types.is_numeric_dtype(train_df[column]):
            continue
        corr = train_df[[column, TARGET_COLUMN]].corr(numeric_only=True).iloc[0, 1]
        if pd.notna(corr) and abs(float(corr)) >= 0.999:
            numeric_proxy_hits.append({"column": column, "abs_corr_with_target": abs(float(corr))})
    checks["deterministic_or_near_target_numeric_features"] = {
        "status": "PASS" if not numeric_proxy_hits else "FAIL",
        "columns": numeric_proxy_hits,
        "interpretation": (
            "No selected numeric feature is nearly identical to price."
            if not numeric_proxy_hits
            else "At least one selected numeric feature is nearly identical to price."
        ),
    }

    future_columns = sorted(set(features) & set(COMMON_LEAKAGE_RISK_FEATURES))
    selected_listing_dynamics = sorted(set(features) & set(LISTING_DYNAMICS_FEATURES))
    checks["future_or_listing_history_columns"] = {
        "status": "PASS" if not future_columns and not selected_listing_dynamics else "FAIL",
        "future_columns": future_columns,
        "listing_dynamics_columns": selected_listing_dynamics,
        "interpretation": (
            "Selected RF features exclude declared full-history leakage-risk columns."
            if not future_columns and not selected_listing_dynamics
            else "Selected RF features include declared future/listing-history leakage-risk columns."
        ),
    }

    date_offset_hits = sorted(set(features) & set(LISTING_DATE_OFFSET_FEATURES))
    checks["split_specific_preparation"] = {
        "status": "PASS" if not date_offset_hits else "WARN",
        "date_offset_features": date_offset_hits,
        "interpretation": (
            "Selected RF features do not use date offsets; imputers and encoders are fit inside train-only pipelines."
            if not date_offset_hits
            else "Selected features include date offsets whose reference date should be checked for split-only fitting."
        ),
    }
    return checks


def feature_groups(features: list[str], train_df: pd.DataFrame) -> dict[str, list[str]]:
    registry = set(TRAFICOM_FEATURES) | set(TRAFICOM_EXTENDED_CANDIDATES) | set(REGISTRY_LIFECYCLE_CANDIDATES)
    date_like = {column for column in features if any(token in column.lower() for token in DATE_LIKE_TOKENS)}
    listing_history = set(LISTING_DYNAMICS_FEATURES) | set(COMMON_LEAKAGE_RISK_FEATURES)
    id_like = {column for column in features if any(token in column.lower() for token in ID_LIKE_TOKENS)}
    categorical = set(train_df[features].select_dtypes(include=["object", "category"]).columns)
    numeric = set(train_df[features].select_dtypes(include=["number", "bool"]).columns)
    baseline = [
        column
        for column in [
            "part_name",
            "quality_grade",
            "oem_number",
            "mileage",
            "brand",
            "model",
            "category",
            "subcategory",
            "year_start",
            "year_end",
            "year_span",
            "year_mid",
            "repair_status",
            "brand_is_known_model_family",
            "mileage_missing_flag",
            "observations_so_far",
            "days_since_first_seen_so_far",
        ]
        if column in features
    ]
    return {
        "registry": [column for column in features if column in registry],
        "date_like": [column for column in features if column in date_like],
        "listing_history": [column for column in features if column in listing_history],
        "id_like": [column for column in features if column in id_like],
        "categorical": [column for column in features if column in categorical],
        "numeric": [column for column in features if column in numeric],
        "baseline": baseline,
    }


def run_ablations(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    groups = feature_groups(features, train_df)
    variants = {
        "full_selected_feature_set": features,
        "remove_date_like_columns": [column for column in features if column not in groups["date_like"]],
        "remove_listing_dynamics_history": [column for column in features if column not in groups["listing_history"]],
        "remove_registry_aggregate_columns": [column for column in features if column not in groups["registry"]],
        "remove_high_cardinality_id_like_text_proxies": [
            column for column in features if column not in groups["id_like"]
        ],
        "baseline_only_listing_features": groups["baseline"],
        "baseline_plus_registry_only": list(dict.fromkeys(groups["baseline"] + groups["registry"])),
        "registry_only": groups["registry"],
        "categorical_only": groups["categorical"],
        "numeric_only": groups["numeric"],
    }
    rows = []
    for name, variant_features in variants.items():
        if not variant_features:
            continue
        predictions, metrics = fit_predict_rf(train_df, validation_df, variant_features, config)
        row = {
            "variant": name,
            "feature_count": len(variant_features),
            **metrics,
        }
        rows.append(row)
        print_interpretation(
            f"ablation {name}",
            "INFO",
            f"R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, features={len(variant_features)}",
        )
    return pd.DataFrame(rows).sort_values(["R2", "MAE"], ascending=[False, True]).reset_index(drop=True)


def shuffled_frame(frame: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = rng.permutation(output[column].to_numpy())
    return output


def shuffle_within_groups(frame: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    output = frame.copy()
    if GROUP_COLUMN not in output.columns:
        return output
    for _, index in output.groupby(GROUP_COLUMN).groups.items():
        index_array = np.asarray(list(index))
        if len(index_array) <= 1:
            continue
        for column in columns:
            output.loc[index_array, column] = rng.permutation(output.loc[index_array, column].to_numpy())
    return output


def run_permutation_tests(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    groups = feature_groups(features, train_df)
    model = fit_random_forest(train_df[features].copy(), train_df[TARGET_COLUMN].copy(), config)

    def score_eval(frame: pd.DataFrame, label: str, changed: str) -> dict[str, Any]:
        raw_predictions = model.predict(frame[features].copy())
        predictions = convert_predictions_to_eur(raw_predictions, config["target_mode"], train_df[TARGET_COLUMN])
        metrics = metric_dict(validation_df[TARGET_COLUMN], predictions)
        print_interpretation(label, "INFO", f"R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}")
        return {"test": label, "perturbation": changed, **metrics}

    rows = [score_eval(validation_df, "unpermuted_validation", "none")]

    suspicious_features = list(
        dict.fromkeys(groups["id_like"] + groups["date_like"] + ["observations_so_far", "days_since_first_seen_so_far"])
    )
    for column in [column for column in suspicious_features if column in features][:12]:
        rows.append(score_eval(shuffled_frame(validation_df, [column], rng), f"shuffle_feature_{column}", column))

    grouped_shuffles = {
        "shuffle_all_registry_features": groups["registry"],
        "shuffle_all_text_categorical_features": groups["categorical"],
        "shuffle_all_numeric_features": groups["numeric"],
    }
    for label, columns in grouped_shuffles.items():
        if columns:
            rows.append(score_eval(shuffled_frame(validation_df, columns, rng), label, ",".join(columns)))

    if groups["numeric"]:
        rows.append(
            score_eval(
                shuffle_within_groups(validation_df, groups["numeric"], rng),
                "shuffle_numeric_within_product_id",
                ",".join(groups["numeric"]),
            )
        )

    shuffled_train = train_df.copy()
    shuffled_train[TARGET_COLUMN] = rng.permutation(shuffled_train[TARGET_COLUMN].to_numpy())
    shuffled_predictions, shuffled_metrics = fit_predict_rf(shuffled_train, validation_df, features, config)
    rows.append({"test": "shuffle_y_train_and_refit", "perturbation": "target", **shuffled_metrics})
    print_interpretation(
        "shuffle_y_train_and_refit",
        "INFO",
        f"R2={shuffled_metrics['R2']:.4f}, MAE={shuffled_metrics['MAE']:.2f}; destructive target shuffle should collapse.",
    )

    return pd.DataFrame(rows)


def dependency_and_cardinality_tables(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    encoded = pd.DataFrame(index=train_df.index)
    for column in features:
        series = train_df[column]
        row: dict[str, Any] = {
            "feature": column,
            "dtype": str(series.dtype),
            "unique_count": int(series.nunique(dropna=True)),
            "missing_fraction": float(series.isna().mean()),
        }
        if pd.api.types.is_numeric_dtype(series):
            encoded[column] = pd.to_numeric(series, errors="coerce").fillna(series.median())
            corr = train_df[[column, TARGET_COLUMN]].corr(numeric_only=True).iloc[0, 1]
            row["abs_corr_with_target"] = None if pd.isna(corr) else abs(float(corr))
        else:
            codes, _ = pd.factorize(series.astype("string"), sort=True)
            encoded[column] = codes
            medians = train_df.groupby(column, dropna=False)[TARGET_COLUMN].median()
            row["category_price_median_std"] = float(medians.std(ddof=0)) if len(medians) else 0.0
        rows.append(row)

    if not encoded.empty:
        sample_size = min(len(encoded), 5000)
        sampled = encoded.sample(n=sample_size, random_state=42) if len(encoded) > sample_size else encoded
        y_sampled = train_df.loc[sampled.index, TARGET_COLUMN]
        mi_values = mutual_info_regression(sampled, y_sampled, random_state=42)
        mi_map = dict(zip(sampled.columns, mi_values))
    else:
        mi_map = {}

    suspicion_rows = []
    for row in rows:
        column = row["feature"]
        unique_ratio = row["unique_count"] / max(len(train_df), 1)
        train_values = set(train_df[column].dropna().astype(str))
        validation_unseen = (
            validation_df[column].dropna().astype(str).map(lambda value: value not in train_values).mean()
            if column in validation_df.columns
            else np.nan
        )
        test_unseen = (
            test_df[column].dropna().astype(str).map(lambda value: value not in train_values).mean()
            if column in test_df.columns
            else np.nan
        )
        flags = []
        if unique_ratio > 0.50:
            flags.append("very_high_cardinality")
        if row.get("abs_corr_with_target") is not None and row["abs_corr_with_target"] >= 0.95:
            flags.append("very_high_target_correlation")
        if mi_map.get(column, 0.0) >= 1.0:
            flags.append("high_mutual_information")
        suspicion_rows.append(
            {
                **row,
                "mutual_information": float(mi_map.get(column, np.nan)),
                "unique_ratio": float(unique_ratio),
                "train_coverage": float(1.0 - row["missing_fraction"]),
                "validation_unseen_fraction": float(validation_unseen) if pd.notna(validation_unseen) else None,
                "test_unseen_fraction": float(test_unseen) if pd.notna(test_unseen) else None,
                "suspicion_flags": ";".join(flags),
            }
        )

    categorical_rows = [
        row
        for row in suspicion_rows
        if train_df[row["feature"]].dtype == "object" or str(train_df[row["feature"]].dtype) == "category"
    ]
    return pd.DataFrame(suspicion_rows), pd.DataFrame(categorical_rows)


def median_lookup_baselines(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_keys: list[list[str]],
) -> pd.DataFrame:
    rows = []
    global_median = float(train_df[TARGET_COLUMN].median())
    for eval_name, eval_df in [("validation", validation_df), ("test", test_df)]:
        for keys in candidate_keys:
            available_keys = [key for key in keys if key in train_df.columns and key in eval_df.columns]
            if len(available_keys) != len(keys):
                continue
            lookup = train_df.groupby(available_keys, dropna=False)[TARGET_COLUMN].median()
            predictions = []
            for _, row in eval_df[available_keys].iterrows():
                key = tuple(row[key] for key in available_keys)
                if len(available_keys) == 1:
                    key = key[0]
                predictions.append(float(lookup.get(key, global_median)))
            metrics = metric_dict(eval_df[TARGET_COLUMN], np.asarray(predictions))
            rows.append({"eval_split": eval_name, "baseline": "+".join(available_keys), **metrics})
    return pd.DataFrame(rows)


def grouped_cv_metrics(train_df: pd.DataFrame, features: list[str], config: dict[str, Any], cv_splits: int) -> pd.DataFrame:
    rows = []
    splitter = GroupKFold(n_splits=cv_splits)
    X = train_df[features].copy()
    y = train_df[TARGET_COLUMN].copy()
    groups = train_df[GROUP_COLUMN].copy()
    for fold, (train_index, validation_index) in enumerate(splitter.split(X, y, groups), start=1):
        fold_train = train_df.iloc[train_index].copy()
        fold_validation = train_df.iloc[validation_index].copy()
        predictions, metrics = fit_predict_rf(fold_train, fold_validation, features, config)
        rows.append({"fold": fold, **metrics})
        print_interpretation("grouped_cv_fold", "INFO", f"fold={fold}, R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}")
    return pd.DataFrame(rows)


def error_stratification(eval_df: pd.DataFrame, predictions: np.ndarray, split_name: str) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": eval_df[TARGET_COLUMN].to_numpy(), "predicted": predictions})
    frame["bin"] = pd.cut(frame["actual"], bins=PRICE_BINS, labels=PRICE_BIN_LABELS, right=False)
    rows = []
    for bin_label, group in frame.groupby("bin", observed=False):
        if group.empty:
            continue
        abs_errors = np.abs(group["actual"] - group["predicted"])
        safe_mape = np.nan
        safe = group["actual"] > 1.0
        if safe.any():
            safe_mape = float((abs_errors[safe] / group.loc[safe, "actual"]).mean() * 100)
        rows.append(
            {
                "split": split_name,
                "price_bin": str(bin_label),
                "count": int(len(group)),
                "mean_actual": float(group["actual"].mean()),
                "mean_predicted": float(group["predicted"].mean()),
                "MAE": float(abs_errors.mean()),
                "median_AE": float(abs_errors.median()),
                "RMSE": float(np.sqrt(np.mean((group["actual"] - group["predicted"]) ** 2))),
                "MAPE_percent": safe_mape,
            }
        )
    return pd.DataFrame(rows)


def make_verdict(
    sanity_checks: dict[str, Any],
    real_r2: float,
    shuffled_target_r2: float | None,
    strongest_tiny_r2: float | None,
    suspicious_feature_count: int,
) -> dict[str, str]:
    failed = [name for name, result in sanity_checks.items() if result["status"] == "FAIL"]
    warnings = [name for name, result in sanity_checks.items() if result["status"] == "WARN"]
    if failed:
        verdict = "Strong evidence metrics are inflated"
        reason = f"Failed sanity checks: {', '.join(failed)}."
    elif shuffled_target_r2 is not None and shuffled_target_r2 > max(0.25, real_r2 - 0.50):
        verdict = "Strong evidence metrics are inflated"
        reason = f"Shuffled-target refit retained high R2 ({shuffled_target_r2:.4f} vs real {real_r2:.4f})."
    elif suspicious_feature_count or (strongest_tiny_r2 is not None and strongest_tiny_r2 > 0.90) or warnings:
        verdict = "Some suspicious target-proxy / leakage-risk signals found"
        reason = (
            f"Warnings={warnings}; suspicious feature flags={suspicious_feature_count}; "
            f"strongest tiny feature-set R2={strongest_tiny_r2}."
        )
    else:
        verdict = "No clear evidence of leakage found"
        reason = f"Core checks passed and shuffled-target R2 collapsed from {real_r2:.4f} to {shuffled_target_r2:.4f}."
    return {"verdict": verdict, "justification": reason}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    prepared = load_training_data(args.train_path, args.validation_path)
    train_df = prepared.train_df
    validation_df = prepared.validation_df
    test_df = pd.read_csv(args.test_path)
    add_date_offsets_for_test(train_df, validation_df, test_df)

    rf_summary = load_json(args.rf_summary_path)
    xgb_summary = load_json(args.xgb_summary_path) if Path(args.xgb_summary_path).exists() else {}
    rf_features = list(rf_summary["feature_names"])
    rf_config = dict(rf_summary["config"])
    if args.quick:
        rf_config = shrink_config_for_quick(rf_config)

    print("R2 credibility audit for selected random forest")
    print(f"Train rows={len(train_df)}, validation rows={len(validation_df)}, test rows={len(test_df)}")
    print(f"RF feature set={rf_summary['feature_variant']} ({len(rf_features)} features), target_mode={rf_config['target_mode']}")
    if args.quick:
        print("[WARN] Quick mode uses fewer trees; outputs are suitable for smoke testing only.")

    validation_predictions, validation_metrics = fit_predict_rf(train_df, validation_df, rf_features, rf_config)
    fit_df = pd.concat([train_df, validation_df], ignore_index=True)
    test_predictions, test_metrics = fit_predict_rf(fit_df, test_df, rf_features, rf_config)
    print_interpretation(
        "reproduced_fixed_validation",
        "INFO",
        f"R2={validation_metrics['R2']:.4f}, MAE={validation_metrics['MAE']:.2f}, RMSE={validation_metrics['RMSE']:.2f}",
    )
    print_interpretation(
        "reproduced_held_out_test",
        "INFO",
        f"R2={test_metrics['R2']:.4f}, MAE={test_metrics['MAE']:.2f}, RMSE={test_metrics['RMSE']:.2f}",
    )

    sanity_checks = grouped_split_checks(train_df, validation_df, test_df, rf_features)
    for label, result in sanity_checks.items():
        print_interpretation(label, result["status"], result["interpretation"])

    ablation_df = run_ablations(train_df, validation_df, rf_features, rf_config)
    ablation_df.to_csv(output_dir / "ablation_results.csv", index=False)

    permutation_df = run_permutation_tests(train_df, validation_df, rf_features, rf_config, rng)
    permutation_df.to_csv(output_dir / "permutation_results.csv", index=False)

    suspicion_df, categorical_df = dependency_and_cardinality_tables(
        train_df,
        validation_df,
        test_df,
        rf_features,
    )
    suspicion_df.sort_values(["mutual_information", "unique_count"], ascending=[False, False]).to_csv(
        output_dir / "feature_suspicion_table.csv",
        index=False,
    )
    categorical_df.to_csv(output_dir / "categorical_cardinality_table.csv", index=False)

    nearest_neighbor_df = median_lookup_baselines(
        train_df,
        validation_df,
        test_df,
        [["part_name"], ["part_name", "brand", "model"], ["oem_number"]],
    )
    nearest_neighbor_df.to_csv(output_dir / "nearest_neighbor_baselines.csv", index=False)

    cv_df = grouped_cv_metrics(train_df, rf_features, rf_config, args.cv_splits)
    cv_df.to_csv(output_dir / "grouped_cv_fold_metrics.csv", index=False)
    cv_summary = {
        "mean_MAE": float(cv_df["MAE"].mean()),
        "std_MAE": float(cv_df["MAE"].std(ddof=0)),
        "min_MAE": float(cv_df["MAE"].min()),
        "max_MAE": float(cv_df["MAE"].max()),
        "mean_RMSE": float(cv_df["RMSE"].mean()),
        "mean_R2": float(cv_df["R2"].mean()),
        "std_R2": float(cv_df["R2"].std(ddof=0)),
        "min_R2": float(cv_df["R2"].min()),
        "max_R2": float(cv_df["R2"].max()),
    }

    stratification_df = pd.concat(
        [
            error_stratification(validation_df, validation_predictions, "validation"),
            error_stratification(test_df, test_predictions, "test"),
        ],
        ignore_index=True,
    )
    stratification_df.to_csv(output_dir / "error_stratification.csv", index=False)

    shuffled_target_rows = permutation_df[permutation_df["test"] == "shuffle_y_train_and_refit"]
    shuffled_target_r2 = (
        float(shuffled_target_rows["R2"].iloc[0]) if not shuffled_target_rows.empty else None
    )
    tiny_sets = ablation_df[ablation_df["feature_count"] <= 5]
    strongest_tiny_r2 = float(tiny_sets["R2"].max()) if not tiny_sets.empty else None
    suspicious_feature_count = int(suspicion_df["suspicion_flags"].astype(bool).sum())
    verdict = make_verdict(
        sanity_checks,
        validation_metrics["R2"],
        shuffled_target_r2,
        strongest_tiny_r2,
        suspicious_feature_count,
    )

    audit_summary = {
        "evaluation_pipeline": {
            "primary_model": {
                "model": "random_forest",
                "split_source_files": {
                    "train": args.train_path,
                    "validation": args.validation_path,
                    "test": args.test_path,
                    "group_assignment": "datasets/splits/group_split_assignment.csv",
                },
                "feature_variant": rf_summary["feature_variant"],
                "feature_names": rf_features,
                "target_mode": rf_summary["target_mode"],
                "metric_computation_location": "src/tree_modeling.py::regression_metrics and scripts/evaluate_random_forest_test.py",
                "predictions_converted_back_to_euros_before_metrics": True,
                "conversion_function": "src/tree_modeling.py::convert_predictions_to_eur",
            },
            "secondary_model": {
                "model": "xgboost",
                "split_source_files": {
                    "train": args.train_path,
                    "validation": args.validation_path,
                },
                "feature_variant": xgb_summary.get("feature_variant"),
                "feature_names": xgb_summary.get("feature_names", []),
                "target_mode": xgb_summary.get("target_mode"),
                "metric_computation_location": "src/tree_modeling.py::evaluate_selected_xgboost_candidates",
                "predictions_converted_back_to_euros_before_metrics": True,
                "conversion_function": "src/tree_modeling.py::convert_predictions_to_eur",
            },
        },
        "reproduced_metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
            "grouped_cv": cv_summary,
        },
        "interpretation": {
            "fixed_validation_vs_grouped_cv": (
                f"Fixed validation R2={validation_metrics['R2']:.4f}; grouped CV mean R2={cv_summary['mean_R2']:.4f} "
                f"with fold range {cv_summary['min_R2']:.4f}-{cv_summary['max_R2']:.4f}."
            ),
            "ablations": {
                "largest_r2_drop": (
                    ablation_df.assign(r2_drop=validation_metrics["R2"] - ablation_df["R2"])
                    .sort_values("r2_drop", ascending=False)
                    .head(3)[["variant", "R2", "MAE", "r2_drop"]]
                    .to_dict(orient="records")
                ),
                "barely_changed": (
                    ablation_df.assign(r2_drop=validation_metrics["R2"] - ablation_df["R2"])
                    .sort_values("r2_drop", ascending=True)
                    .head(3)[["variant", "R2", "MAE", "r2_drop"]]
                    .to_dict(orient="records")
                ),
            },
            "permutation": {
                "shuffled_target_R2": shuffled_target_r2,
                "note": "High R2 after destructive perturbations would be a red flag; collapse supports credibility.",
            },
            "final_verdict": verdict,
        },
        "quick_mode": bool(args.quick),
    }

    write_json(output_dir / "sanity_checks.json", sanity_checks)
    write_json(output_dir / "audit_summary.json", audit_summary)

    print("Final verdict")
    print(f"{verdict['verdict']}: {verdict['justification']}")
    print(f"Saved audit artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
