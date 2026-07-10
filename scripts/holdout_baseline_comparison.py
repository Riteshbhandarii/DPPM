"""Score the training-fitted dummy baselines on the strict holdout, segment by price.

Descriptive only: reuses the frozen Random Forest predictions from
`artifacts/strict_final_holdout/`. Fits nothing but the trivial anchors, and those
are fitted on train+validation exactly like the Random Forest was. No model is
selected, retuned, or re-predicted here.

Baseline definitions match `strict_funnel.dummy_baselines` (global median and
per-subcategory median, unseen subcategories falling back to the global median).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "datasets/splits_strict"
HOLDOUT = ROOT / "artifacts/strict_final_holdout"
TARGET = "price"

SEGMENT_EDGES = [0, 50, 100, 200, 500, 1000, np.inf]
SEGMENT_LABELS = ["<50", "50-100", "100-200", "200-500", "500-1k", "1k+"]


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    error = actual.to_numpy(dtype=float) - predicted
    absolute = np.abs(error)
    return {
        "MAE": float(absolute.mean()),
        "median_AE": float(np.median(absolute)),
        "RMSE": float(np.sqrt((error**2).mean())),
        "MdAPE_pct": float(np.median(absolute / actual.to_numpy(dtype=float)) * 100),
    }


def main() -> None:
    fit_df = pd.concat(
        [pd.read_csv(SPLITS / "train_strict.csv"), pd.read_csv(SPLITS / "validation_strict.csv")],
        ignore_index=True,
    )
    test_df = pd.read_csv(SPLITS / "test_strict.csv")
    rf = pd.read_csv(HOLDOUT / "final_holdout_predictions.csv")
    if len(rf) != len(test_df):
        raise SystemExit("prediction/test row mismatch — refusing to align silently")

    actual = test_df[TARGET]
    global_median = float(fit_df[TARGET].median())
    subcategory_medians = fit_df.groupby("subcategory")[TARGET].median()

    predictions = {
        "random_forest": rf["predicted_price"].to_numpy(dtype=float),
        "dummy_subcategory_median": (
            test_df["subcategory"].map(subcategory_medians).fillna(global_median).to_numpy(dtype=float)
        ),
        "dummy_global_median": np.full(len(test_df), global_median),
    }

    segment = pd.cut(actual, SEGMENT_EDGES, labels=SEGMENT_LABELS)
    report: dict[str, object] = {
        "fit_scope": "train_strict + validation_strict (matches the frozen RF refit)",
        "note": "Descriptive baseline comparison on the already-run holdout. No model selection.",
        "unseen_subcategories_in_test": int(
            (~test_df["subcategory"].isin(subcategory_medians.index)).sum()
        ),
        "overall": {name: metrics(actual, pred) for name, pred in predictions.items()},
        "by_segment": {},
    }

    for label in SEGMENT_LABELS:
        mask = (segment == label).to_numpy()
        if not mask.any():
            continue
        entry = {
            "n": int(mask.sum()),
            **{name: metrics(actual[mask], pred[mask]) for name, pred in predictions.items()},
        }
        entry["rf_vs_subcategory_median"] = {
            "MAE_improvement_pct": round(
                (1 - entry["random_forest"]["MAE"] / entry["dummy_subcategory_median"]["MAE"]) * 100, 2
            ),
            "median_AE_improvement_pct": round(
                (1 - entry["random_forest"]["median_AE"] / entry["dummy_subcategory_median"]["median_AE"])
                * 100,
                2,
            ),
        }
        report["by_segment"][label] = entry

    output = HOLDOUT / "holdout_baseline_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"unseen subcategories in test: {report['unseen_subcategories_in_test']}\n")
    print("=== overall (test split) ===")
    for name, m in report["overall"].items():
        print(f"  {name:26s} MAE {m['MAE']:8.2f}  medAE {m['median_AE']:7.2f}  MdAPE {m['MdAPE_pct']:6.2f}%")

    print("\n=== RF vs subcategory-median dummy, by price segment ===")
    header = f"{'seg':>8} {'n':>5} {'RF MAE':>9} {'dummy MAE':>10} {'MAE gain':>9} {'RF medAE':>9} {'dummy medAE':>12} {'medAE gain':>11}"
    print(header)
    for label, e in report["by_segment"].items():
        print(
            f"{label:>8} {e['n']:5d} {e['random_forest']['MAE']:9.2f} "
            f"{e['dummy_subcategory_median']['MAE']:10.2f} "
            f"{e['rf_vs_subcategory_median']['MAE_improvement_pct']:8.1f}% "
            f"{e['random_forest']['median_AE']:9.2f} "
            f"{e['dummy_subcategory_median']['median_AE']:12.2f} "
            f"{e['rf_vs_subcategory_median']['median_AE_improvement_pct']:10.1f}%"
        )
    print(f"\nwritten: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
