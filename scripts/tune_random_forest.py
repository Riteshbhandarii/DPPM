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
    build_feature_catalog,
    evaluate_selected_random_forest_candidates,
    generate_random_forest_search_configs,
    load_training_data,
    save_tuning_reports,
    screen_random_forest_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune random forest on the grouped train/validation splits."
    )
    parser.add_argument("--train-path", default="datasets/splits/train_grouped.csv")
    parser.add_argument("--validation-path", default="datasets/splits/validation_grouped.csv")
    parser.add_argument("--output-dir", default="artifacts/random_forest_tuning")
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument(
        "--random-trials",
        type=int,
        default=24,
        help="Number of additional random-search random-forest configurations to sample.",
    )
    parser.add_argument(
        "--top-k-finalists",
        type=int,
        default=8,
        help="Number of screened candidates promoted to grouped cross-validation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the wider random-forest search.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_data = load_training_data(args.train_path, args.validation_path)
    feature_catalog = build_feature_catalog(prepared_data.train_df, model_kind="random_forest")
    search_configs = generate_random_forest_search_configs(
        random_trials=args.random_trials,
        random_seed=args.random_seed,
    )

    print(f"Generated {len(search_configs)} random-forest configurations for screening.")
    screening_results_df, finalists = screen_random_forest_candidates(
        train_df=prepared_data.train_df,
        validation_df=prepared_data.validation_df,
        feature_sets=feature_catalog["feature_sets"],
        configs=search_configs,
        top_k_finalists=args.top_k_finalists,
    )

    print(f"Promoting {len(finalists)} screened candidates to grouped cross-validation.")
    cv_results_df, summary = evaluate_selected_random_forest_candidates(
        train_df=prepared_data.train_df,
        validation_df=prepared_data.validation_df,
        selected_candidates=finalists,
        cv_splits=args.cv_splits,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screening_results_df.drop(columns=["feature_names", "config"]).to_csv(
        output_dir / "screening_results.csv",
        index=False,
    )

    save_tuning_reports(
        output_dir=output_dir,
        model_reports=[summary],
        cv_frames=[cv_results_df.drop(columns=["feature_names", "config"])],
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
    print(f"Saved reports to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
