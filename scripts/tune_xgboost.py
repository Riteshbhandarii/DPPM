#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_modeling import (
    GROUP_COLUMN,
    build_feature_catalog,
    evaluate_selected_xgboost_candidates,
    generate_xgboost_refinement_configs,
    generate_xgboost_search_configs,
    load_training_data,
    save_tuning_reports,
    screen_xgboost_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune XGBoost on the grouped train/validation splits."
    )
    parser.add_argument("--train-path", default="datasets/splits/train_grouped.csv")
    parser.add_argument("--validation-path", default="datasets/splits/validation_grouped.csv")
    parser.add_argument("--output-dir", default="artifacts/xgboost_tuning")
    parser.add_argument("--xgboost-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument(
        "--random-trials",
        type=int,
        default=32,
        help="Number of additional random-search XGBoost configurations to sample.",
    )
    parser.add_argument(
        "--refinement-trials",
        type=int,
        default=20,
        help="Number of local refinement configurations sampled around the best screened result.",
    )
    parser.add_argument(
        "--top-k-finalists",
        type=int,
        default=10,
        help="Number of screened candidates promoted to grouped cross-validation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the wider XGBoost search.",
    )
    return parser.parse_args()


def make_inner_early_stopping_split(train_df):
    """Create a train-only grouped eval split so validation data is never used for early stopping."""

    if GROUP_COLUMN not in train_df.columns:
        raise KeyError(
            f"Expected grouped split column {GROUP_COLUMN!r} in train data for XGBoost early stopping."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    inner_train_idx, early_stopping_idx = next(
        splitter.split(train_df, groups=train_df[GROUP_COLUMN])
    )
    inner_train_df = train_df.iloc[inner_train_idx].copy()
    early_stopping_df = train_df.iloc[early_stopping_idx].copy()

    overlap = set(inner_train_df[GROUP_COLUMN]).intersection(early_stopping_df[GROUP_COLUMN])
    if overlap:
        raise RuntimeError("Inner XGBoost early-stopping split has overlapping listing groups.")

    print(
        "XGBoost early-stopping split:",
        f"inner_train_rows={len(inner_train_df)}",
        f"early_stopping_rows={len(early_stopping_df)}",
        f"group_column={GROUP_COLUMN}",
    )
    return inner_train_df, early_stopping_df


def main() -> None:
    args = parse_args()
    prepared_data = load_training_data(args.train_path, args.validation_path)
    feature_catalog = build_feature_catalog(prepared_data.train_df, model_kind="xgboost")
    inner_train_df, early_stopping_df = make_inner_early_stopping_split(prepared_data.train_df)

    # Stage one runs a broader search across all trusted XGBoost feature variants.
    search_configs = generate_xgboost_search_configs(
        random_trials=args.random_trials,
        random_seed=args.random_seed,
    )

    print(f"Generated {len(search_configs)} XGBoost configurations for broad screening.")
    screening_results_df, finalists = screen_xgboost_candidates(
        train_df=inner_train_df,
        validation_df=prepared_data.validation_df,
        early_stopping_df=early_stopping_df,
        feature_sets=feature_catalog["feature_sets"],
        configs=search_configs,
        xgboost_device=args.xgboost_device,
        top_k_finalists=args.top_k_finalists,
    )

    # Stage two narrows the search around the best screened candidate and its feature set.
    best_screened_candidate = screening_results_df.iloc[0].to_dict()
    print(
        "Best broad-screen candidate before refinement:",
        best_screened_candidate["feature_variant"],
        best_screened_candidate["config_name"],
        f"MAE={best_screened_candidate['validation_MAE']:.4f}",
    )
    refinement_feature_sets = {
        best_screened_candidate["feature_variant"]: list(best_screened_candidate["feature_names"])
    }
    refinement_configs = generate_xgboost_refinement_configs(
        base_config=best_screened_candidate["config"],
        refinement_trials=args.refinement_trials,
        random_seed=args.random_seed + 1000,
    )
    print(
        f"Generated {len(refinement_configs)} local refinement configurations "
        f"for feature variant {best_screened_candidate['feature_variant']}."
    )
    refinement_results_df, refinement_finalists = screen_xgboost_candidates(
        train_df=inner_train_df,
        validation_df=prepared_data.validation_df,
        early_stopping_df=early_stopping_df,
        feature_sets=refinement_feature_sets,
        configs=refinement_configs,
        xgboost_device=args.xgboost_device,
        top_k_finalists=args.top_k_finalists,
    )

    # Merge broad-search and refinement finalists before the grouped-CV selection pass.
    combined_finalists = []
    seen_finalists = set()
    for candidate in finalists + refinement_finalists:
        finalist_key = (candidate["feature_variant"], candidate["config_name"])
        if finalist_key in seen_finalists:
            continue
        combined_finalists.append(candidate)
        seen_finalists.add(finalist_key)

    print(f"Promoting {len(combined_finalists)} screened candidates to grouped cross-validation.")
    cv_results_df, summary = evaluate_selected_xgboost_candidates(
        train_df=prepared_data.train_df,
        validation_df=prepared_data.validation_df,
        selected_candidates=combined_finalists,
        cv_splits=args.cv_splits,
        xgboost_device=args.xgboost_device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save both screening stages separately so the tuning path is easy to inspect later.
    screening_results_df.drop(columns=["feature_names", "config"]).to_csv(
        output_dir / "screening_results.csv",
        index=False,
    )
    refinement_results_df.drop(columns=["feature_names", "config"]).to_csv(
        output_dir / "refinement_results.csv",
        index=False,
    )

    save_tuning_reports(
        output_dir=output_dir,
        model_reports=[summary],
        cv_frames=[cv_results_df.drop(columns=["feature_names", "config"])],
    )

    print("Best XGBoost config")
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
    print(f"Saved reports to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
