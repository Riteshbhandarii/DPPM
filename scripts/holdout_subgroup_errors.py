"""
Purpose:
Break the test-set error (connected-component split) down by brand and by part category, for the
frozen Random Forest and the subcategory-median baseline side by side. This is
the brand/category half of the subgroup analysis; the price-band half is in
`artifacts/strict_final_holdout/holdout_baseline_comparison.json`.

Inputs:
- datasets/splits_strict/{train,validation,test}_strict.csv
- artifacts/strict_final_holdout/final_holdout_predictions.csv

Outputs:
- artifacts/holdout_subgroup_errors/subgroup_errors.csv   one row per group
- artifacts/holdout_subgroup_errors/subgroup_errors.json  the same, plus run notes

Assumptions:
- Descriptive only. The saved holdout predictions are read, nothing is refitted
  or re-predicted, and no model is selected on these numbers. The holdout guard
  stays consumed.
- The prediction file carries no row id, so it is aligned to test_strict by row
  order. The script refuses to run unless the row counts match and every
  actual_price equals the test split's price in the same row.
- The baseline is the one in scripts/holdout_baseline_comparison.py: the
  per-subcategory median fitted on train+validation, unseen subcategories
  falling back to the global median.
- Brand and vehicle model are the same split here: three brands, one model
  each, so the brand table is also the per-vehicle table.

How to run:
    .venv/bin/python scripts/holdout_subgroup_errors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from holdout_baseline_comparison import HOLDOUT, SPLITS, TARGET, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/holdout_subgroup_errors"
GROUPINGS = ["brand", "category"]

# Same bootstrap settings as the rest of the holdout reporting.
BOOTSTRAP_RESAMPLES = 10_000
SEED = 32


def aligned_holdout() -> pd.DataFrame:
    fit_df = pd.concat(
        [pd.read_csv(SPLITS / "train_strict.csv"), pd.read_csv(SPLITS / "validation_strict.csv")],
        ignore_index=True,
    )
    test_df = pd.read_csv(SPLITS / "test_strict.csv")
    rf = pd.read_csv(HOLDOUT / "final_holdout_predictions.csv")

    if len(rf) != len(test_df) or not np.array_equal(
        rf["actual_price"].to_numpy(dtype=float), test_df[TARGET].to_numpy(dtype=float)
    ):
        raise SystemExit("saved predictions do not line up with test_strict row by row; refusing to guess")

    global_median = float(fit_df[TARGET].median())
    subcategory_medians = fit_df.groupby("subcategory")[TARGET].median()
    return test_df.assign(
        rf=rf["predicted_price"].to_numpy(dtype=float),
        dummy=test_df["subcategory"].map(subcategory_medians).fillna(global_median),
    )


def mae_difference_ci(actual, rf, dummy, rng) -> tuple[float, float]:
    """Paired bootstrap 95% CI for RF MAE minus baseline MAE (positive = RF worse)."""
    rf_error = np.abs(actual - rf)
    dummy_error = np.abs(actual - dummy)
    index = rng.integers(0, len(actual), size=(BOOTSTRAP_RESAMPLES, len(actual)))
    differences = rf_error[index].mean(axis=1) - dummy_error[index].mean(axis=1)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    holdout = aligned_holdout()
    rng = np.random.default_rng(SEED)
    rows = []

    for grouping in GROUPINGS:
        for group, part in holdout.groupby(grouping, sort=True):
            actual = part[TARGET].to_numpy(dtype=float)
            rf_metrics = metrics(part[TARGET], part["rf"].to_numpy(dtype=float))
            dummy_metrics = metrics(part[TARGET], part["dummy"].to_numpy(dtype=float))
            low, high = mae_difference_ci(actual, part["rf"].to_numpy(), part["dummy"].to_numpy(), rng)
            rows.append(
                {
                    "grouping": grouping,
                    "group": group,
                    "n": len(part),
                    "median_price": float(np.median(actual)),
                    # Price spread inside the group: a high value means the
                    # group mixes cheap and expensive parts, which is where a
                    # single subcategory label carries least information.
                    "price_cv": float(actual.std() / actual.mean()),
                    **{f"rf_{key}": value for key, value in rf_metrics.items()},
                    **{f"dummy_{key}": value for key, value in dummy_metrics.items()},
                    "mae_diff_rf_minus_dummy": rf_metrics["MAE"] - dummy_metrics["MAE"],
                    "mae_diff_ci_low": low,
                    "mae_diff_ci_high": high,
                    "rf_closer_share": float(
                        (np.abs(actual - part["rf"]) < np.abs(actual - part["dummy"])).mean()
                    ),
                }
            )

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "subgroup_errors.csv", index=False, float_format="%.4f")
    (OUT / "subgroup_errors.json").write_text(
        json.dumps(
            {
                "note": "Descriptive breakdown of the already-run connected-component test set. No refit, no selection.",
                "baseline": "per-subcategory median fitted on train+validation, global-median fallback",
                "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": SEED, "statistic": "RF MAE - baseline MAE"},
                "brand_equals_vehicle": "three brands, one model each",
                "groups": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    shown = table[
        ["grouping", "group", "n", "median_price", "price_cv", "rf_MAE", "dummy_MAE",
         "mae_diff_ci_low", "mae_diff_ci_high", "rf_median_AE", "dummy_median_AE", "rf_closer_share"]
    ]
    print(shown.to_string(index=False, float_format="{:,.2f}".format))
    print(f"\nwritten: {OUT.relative_to(ROOT)}/subgroup_errors.{{csv,json}}")


if __name__ == "__main__":
    main()
