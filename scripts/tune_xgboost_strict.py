#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strict_model_selection import (  # noqa: E402
    DEFAULT_PART_IDENTITY_COLUMNS,
    evaluate_xgboost_candidates_strict,
    load_strict_tuning_frame,
    save_strict_tuning_reports,
)
from src.tree_modeling import (  # noqa: E402
    build_feature_catalog,
    generate_xgboost_refinement_configs,
    generate_xgboost_search_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune XGBoost using strict part-identity grouped CV."
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=["datasets/splits/train_grouped.csv"],
        help="One or more split files to combine into the strict tuning frame.",
    )
    parser.add_argument("--output-dir", default="artifacts/xgboost_tuning_strict")
    parser.add_argument("--xgboost-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=DEFAULT_PART_IDENTITY_COLUMNS,
    )
    parser.add_argument("--random-trials", type=int, default=32)
    parser.add_argument("--refinement-trials", type=int, default=20)
    parser.add_argument("--top-k-finalists", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-test-size", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, group_columns = load_strict_tuning_frame(args.data_path, args.group_columns)
    feature_catalog = build_feature_catalog(frame, model_kind="xgboost")

    search_configs = generate_xgboost_search_configs(
        random_trials=args.random_trials,
        random_seed=args.random_seed,
    )
    print(f"Generated {len(search_configs)} strict XGBoost configurations.")
    broad_results_df, finalists = evaluate_xgboost_candidates_strict(
        frame=frame,
        feature_sets=feature_catalog["feature_sets"],
        configs=search_configs,
        cv_splits=args.cv_splits,
        xgboost_device=args.xgboost_device,
        top_k_finalists=args.top_k_finalists,
        early_stopping_test_size=args.early_stopping_test_size,
    )

    best_screened_candidate = broad_results_df.iloc[0].to_dict()
    refinement_feature_sets = {
        best_screened_candidate["feature_variant"]: list(best_screened_candidate["feature_names"])
    }
    refinement_configs = generate_xgboost_refinement_configs(
        base_config=best_screened_candidate["config"],
        refinement_trials=args.refinement_trials,
        random_seed=args.random_seed + 1000,
    )
    print(
        f"Generated {len(refinement_configs)} strict refinement configurations "
        f"for feature variant {best_screened_candidate['feature_variant']}."
    )
    refinement_results_df, _ = evaluate_xgboost_candidates_strict(
        frame=frame,
        feature_sets=refinement_feature_sets,
        configs=refinement_configs,
        cv_splits=args.cv_splits,
        xgboost_device=args.xgboost_device,
        top_k_finalists=args.top_k_finalists,
        early_stopping_test_size=args.early_stopping_test_size,
    )

    summary = save_strict_tuning_reports(
        output_dir=args.output_dir,
        broad_results_df=broad_results_df,
        refinement_results_df=refinement_results_df,
        group_columns=group_columns,
        cv_splits=args.cv_splits,
        source_paths=args.data_path,
    )

    print("Best strict XGBoost config")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"config", "feature_names"}},
            indent=2,
            default=str,
        )
    )
    print(f"Saved reports to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
