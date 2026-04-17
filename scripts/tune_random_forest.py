#!/usr/bin/env python3
from __future__ import annotations

# Standard library imports used by the CLI entrypoint.
import argparse
import json
from pathlib import Path
import sys

# Add the repository root so the shared modeling module can be imported on Puhti.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Shared random-forest tuning helpers.
from src.tree_modeling import (
    build_feature_catalog,
    evaluate_selected_random_forest_candidates,
    generate_random_forest_refinement_configs,
    generate_random_forest_search_configs,
    load_training_data,
    save_tuning_reports,
    screen_random_forest_candidates,
)


def parse_args():
    """Define command-line arguments for the random-forest tuning run."""

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
        "--refinement-trials",
        type=int,
        default=16,
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
        help="Random seed for the wider random-forest search.",
    )
    return parser.parse_args()


def main():
    """Run the widened random-forest search and save the resulting reports."""

    # Load the grouped split files and rebuild the trusted feature variants.
    args = parse_args()
    prepared_data = load_training_data(args.train_path, args.validation_path)
    feature_catalog = build_feature_catalog(prepared_data.train_df, model_kind="random_forest")

    # Build the broader random-forest search space around the notebook anchor configs.
    search_configs = generate_random_forest_search_configs(
        random_trials=args.random_trials,
        random_seed=args.random_seed,
    )

    # Screen the full candidate pool on the fixed validation split first.
    print(f"Generated {len(search_configs)} random-forest configurations for screening.")
    screening_results_df, finalists = screen_random_forest_candidates(
        train_df=prepared_data.train_df,
        validation_df=prepared_data.validation_df,
        feature_sets=feature_catalog["feature_sets"],
        configs=search_configs,
        top_k_finalists=args.top_k_finalists,
    )

    # Narrow the next search around the best screened random-forest candidate.
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
    refinement_configs = generate_random_forest_refinement_configs(
        base_config=best_screened_candidate["config"],
        refinement_trials=args.refinement_trials,
        random_seed=args.random_seed + 1000,
    )
    print(
        f"Generated {len(refinement_configs)} local refinement configurations "
        f"for feature variant {best_screened_candidate['feature_variant']}."
    )
    refinement_results_df, refinement_finalists = screen_random_forest_candidates(
        train_df=prepared_data.train_df,
        validation_df=prepared_data.validation_df,
        feature_sets=refinement_feature_sets,
        configs=refinement_configs,
        top_k_finalists=args.top_k_finalists,
    )

    # Promote the combined broad-search and refinement finalists to grouped cross-validation.
    combined_finalists = []
    seen_finalists = set()
    for candidate in finalists + refinement_finalists:
        finalist_key = (candidate["feature_variant"], candidate["config_name"])
        if finalist_key in seen_finalists:
            continue
        combined_finalists.append(candidate)
        seen_finalists.add(finalist_key)

    print(f"Promoting {len(combined_finalists)} screened candidates to grouped cross-validation.")
    cv_results_df, summary = evaluate_selected_random_forest_candidates(
        train_df=prepared_data.train_df,
        validation_df=prepared_data.validation_df,
        selected_candidates=combined_finalists,
        cv_splits=args.cv_splits,
    )

    # Save the screening table and the grouped-CV reports to the output folder.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    # Print the best final summary so it is visible in the terminal or batch log.
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
