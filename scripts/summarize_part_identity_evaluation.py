#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize separate part-identity grouped evaluation outputs."
    )
    parser.add_argument("--input-dir", default="artifacts/part_identity_evaluation")
    parser.add_argument("--output-path", default="artifacts/part_identity_evaluation/model_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = []
    for summary_path in sorted(input_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cv_summary = summary["cv_summary"]
        rows.append(
            {
                "model": summary["model"],
                "feature_variant": summary.get("feature_variant"),
                "feature_count": summary.get("feature_count"),
                "group_columns": "+".join(summary.get("group_columns", [])),
                "group_count": summary.get("group_count"),
                "row_count": summary.get("row_count"),
                "mean_MAE": cv_summary["mean_MAE"],
                "std_MAE": cv_summary["std_MAE"],
                "min_MAE": cv_summary["min_MAE"],
                "max_MAE": cv_summary["max_MAE"],
                "mean_RMSE": cv_summary["mean_RMSE"],
                "mean_R2": cv_summary["mean_R2"],
                "mean_median_AE": cv_summary["mean_median_AE"],
            }
        )

    if not rows:
        raise RuntimeError(f"No model summaries found under {input_dir}.")

    comparison = pd.DataFrame(rows).sort_values(["mean_MAE", "mean_RMSE"]).reset_index(drop=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(comparison.to_string(index=False))
    print(f"Saved part-identity model comparison to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
