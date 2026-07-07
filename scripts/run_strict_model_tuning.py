"""Run stage-2 strict model tuning for the final thesis protocol.

Purpose:
Tune one selected model family under the strict connected-component protocol.

Inputs:
- datasets/splits_strict/train_strict.csv
- datasets/splits_strict/validation_strict.csv

Outputs:
- artifacts/strict_model_tuning/<model>/screening_results.csv
- artifacts/strict_model_tuning/<model>/finalist_cv_results.csv
- artifacts/strict_model_tuning/<model>/refinement_cv_results.csv
- artifacts/strict_model_tuning/<model>/cv_results_combined.csv
- artifacts/strict_model_tuning/<model>/best_tuning_summary.json
- artifacts/strict_model_tuning/<model>/run_metadata.json

Notes:
The strict test split is not loaded or used here. Final holdout evaluation belongs to
notebooks/05_strict_training/03_strict_final_holdout.ipynb after the winner is frozen.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strict_funnel import load_strict_training_frames, run_model_tuning, write_summary_json


MODELS = ("ridge", "random_forest", "xgboost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict stage-2 tuning for one model family."
    )
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to artifacts/strict_model_tuning/<model>.",
    )
    parser.add_argument("--random-trials", type=int, default=24)
    parser.add_argument("--refinement-trials", type=int, default=16)
    parser.add_argument("--top-k-finalists", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=None,
        help="Worker cap for RandomForestRegressor. Use for --model random_forest.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced smoke-test configs. Do not use for thesis results.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def assert_output_is_safe(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise SystemExit(
            f"Output directory already exists and is non-empty: {output_dir}\n"
            "Use --force only when intentionally rerunning this tuning stage."
        )


def main() -> None:
    args = parse_args()
    if args.rf_n_jobs is not None and args.model != "random_forest":
        raise SystemExit("--rf-n-jobs is only valid with --model random_forest.")

    output_dir = args.output_dir or ROOT / "artifacts" / "strict_model_tuning" / args.model
    output_dir = output_dir.resolve()
    assert_output_is_safe(output_dir, args.force)

    print("Strict model tuning")
    print("repository:", ROOT)
    print("model:", args.model)
    print("output_dir:", output_dir)
    print("quick:", args.quick)
    print("random_trials:", args.random_trials)
    print("refinement_trials:", args.refinement_trials)
    print("top_k_finalists:", args.top_k_finalists)
    print("cv_splits:", args.cv_splits)
    print("random_seed:", args.random_seed)
    print("rf_n_jobs:", args.rf_n_jobs)

    train_df, validation_df = load_strict_training_frames(
        train_path=ROOT / "datasets" / "splits_strict" / "train_strict.csv",
        validation_path=ROOT / "datasets" / "splits_strict" / "validation_strict.csv",
    )
    print("train_rows:", len(train_df))
    print("validation_rows:", len(validation_df))

    started_at = datetime.now(timezone.utc)
    summary = run_model_tuning(
        model=args.model,
        train_df=train_df,
        validation_df=validation_df,
        output_dir=output_dir,
        random_trials=args.random_trials,
        refinement_trials=args.refinement_trials,
        top_k_finalists=args.top_k_finalists,
        cv_splits=args.cv_splits,
        random_seed=args.random_seed,
        quick=args.quick,
        rf_n_jobs=args.rf_n_jobs,
    )
    finished_at = datetime.now(timezone.utc)

    metadata = {
        "started_at_utc": started_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "finished_at_utc": finished_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "model": args.model,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "random_trials": args.random_trials,
        "refinement_trials": args.refinement_trials,
        "top_k_finalists": args.top_k_finalists,
        "cv_splits": args.cv_splits,
        "random_seed": args.random_seed,
        "rf_n_jobs": args.rf_n_jobs,
        "quick": args.quick,
        "strict_test_split_used": False,
    }
    write_summary_json(output_dir / "run_metadata.json", metadata)

    print("Best tuning summary")
    for key in (
        "model_type",
        "stage",
        "feature_variant",
        "config_name",
        "target_mode",
        "feature_count",
        "screen_validation_MAE",
        "cv_mean_MAE",
        "cv_std_MAE",
        "cv_mean_RMSE",
        "cv_std_RMSE",
        "cv_mean_R2",
        "cv_mean_median_AE",
    ):
        print(f"{key}: {summary.get(key)}")
    print("config:", summary.get("config"))


if __name__ == "__main__":
    main()
