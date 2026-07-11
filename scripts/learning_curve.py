"""Learning curve: is the model data-limited or signal-limited? (holdout-safe, CV only)

Answers issue #63 / the "maybe it just needs more data" hypothesis. Trains the frozen Random
Forest on increasing fractions of the strict TRAIN split and evaluates on fixed component-grouped
folds, overlaying the subcategory-median dummy refit at each size.

Design guarantees:
  * Subsample by whole CONNECTED COMPONENT, never by row -> preserves the strict part-identity
    grouping at every training size (random row subsampling would leak identity across sizes).
  * Evaluation folds are FIXED (the 4-fold component GroupKFold); only the training subset shrinks.
  * Frozen RF config recovered from the tuning artifact -> characterises THE model, no re-selection.
  * The dummy is REFIT on each subsample (unseen subcategories -> global-median fallback), so its
    curve degrades fairly; it is not given the full-data medians.
  * GUARDRAIL: at fraction = 1.0 there is no subsampling, so the RF endpoint MUST reproduce the
    frozen CV MAE 105.334. Asserted; abort if it does not (harness is then untrustworthy).
  * NEVER touches the frozen strict test split. Descriptive only.

Reading it (prepare for all outcomes, do not pre-write the conclusion):
  * RF plateau AND tracks the dummy -> NOT data-quantity-limited within the 3-model scope; more
    listings of the same 3 models won't close the gap to the heuristic. (Says nothing about more
    *models* / broader scope -> that stays future work.)
  * RF still rising at 100%, or pulling away from the dummy as data grows -> a real, reportable
    finding that complicates the administered-pricing story. Report honestly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.strict_funnel import (
    TARGET_COLUMN,
    fit_and_predict,
    load_strict_training_frames,
)
from src.strict_protocol import COMPONENT_GROUP_COLUMN

RF_SUMMARY = Path("artifacts/strict_model_tuning/random_forest/best_tuning_summary.json")
OUTPUT_DIR = Path("artifacts/learning_curve")
CV_SPLITS = 4
FRACTIONS = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
N_SEEDS = 4  # subsample draws per fraction (which components you draw matters); 1 at fraction 1.0
SEED_BASE = 1000
REPRODUCTION_TOL_EUR = 0.75


def dummy_predict(train_sub: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    """Subcategory-median heuristic refit on the subsample; global-median fallback for unseen."""
    global_median = float(train_sub[TARGET_COLUMN].median())
    medians = train_sub.groupby("subcategory")[TARGET_COLUMN].median()
    return eval_df["subcategory"].map(medians).fillna(global_median).to_numpy(dtype=float)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_pred - y_true)))


def main() -> None:
    summary = json.loads(RF_SUMMARY.read_text())
    features = list(summary["feature_names"])
    config = summary["config"]
    expected_full_mae = float(summary["cv_mean_MAE"])

    train_df, _validation_df = load_strict_training_frames()
    groups = train_df[COMPONENT_GROUP_COLUMN]
    gkf = GroupKFold(n_splits=CV_SPLITS)
    folds = list(gkf.split(train_df, train_df[TARGET_COLUMN], groups))
    print(f"Loaded strict TRAIN: {len(train_df):,} rows, {groups.nunique():,} components, "
          f"{CV_SPLITS} fixed eval folds\n")

    records = []  # one row per (fraction, fold, seed)
    for fraction in FRACTIONS:
        seeds = [0] if fraction >= 1.0 else list(range(N_SEEDS))
        for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
            fold_train = train_df.iloc[tr_idx]
            fold_val = train_df.iloc[va_idx]
            y_val = fold_val[TARGET_COLUMN].to_numpy(dtype=float)
            comps = fold_train[COMPONENT_GROUP_COLUMN].unique()
            n_target = max(1, round(fraction * len(comps)))
            for seed in seeds:
                rng = np.random.default_rng(SEED_BASE + fold_id * 100 + seed)
                picked = comps if fraction >= 1.0 else rng.permutation(comps)[:n_target]
                sub = fold_train[fold_train[COMPONENT_GROUP_COLUMN].isin(picked)]
                rf_pred = fit_and_predict(
                    "random_forest", sub, fold_val, features, config, COMPONENT_GROUP_COLUMN,
                )
                d_pred = dummy_predict(sub, fold_val)
                records.append({
                    "fraction": fraction, "fold": fold_id, "seed": seed,
                    "train_rows": int(len(sub)), "train_components": int(len(picked)),
                    "rf_mae": mae(y_val, rf_pred), "rf_medae": medae(y_val, rf_pred),
                    "dummy_mae": mae(y_val, d_pred),
                })
        done = pd.DataFrame([r for r in records if r["fraction"] == fraction])
        print(f"  fraction {fraction:.2f}: ~{int(done['train_rows'].mean()):>5,} rows | "
              f"RF MAE {done['rf_mae'].mean():7.2f} | dummy MAE {done['dummy_mae'].mean():7.2f}")

    df = pd.DataFrame(records)

    # ---- Guardrail: 100% endpoint must reproduce the frozen CV MAE ----
    full = df[df["fraction"] >= 1.0]
    endpoint_mae = float(full["rf_mae"].mean())
    reproduced = abs(endpoint_mae - expected_full_mae) <= REPRODUCTION_TOL_EUR
    print(f"\nreproduction check: 100% RF fold-avg MAE = {endpoint_mae:.3f} vs frozen "
          f"{expected_full_mae:.3f} -> {'OK' if reproduced else 'MISMATCH!'}")
    if not reproduced:
        raise SystemExit(
            f"ABORT: endpoint {endpoint_mae:.3f} != frozen {expected_full_mae:.3f}. "
            f"Component subsampling or fold construction is off; curve is untrustworthy."
        )

    # ---- Aggregate per fraction (mean + 10/90 band across folds x seeds) ----
    curve = []
    for fraction in FRACTIONS:
        g = df[df["fraction"] == fraction]
        curve.append({
            "fraction": fraction,
            "mean_train_rows": float(g["train_rows"].mean()),
            "mean_train_components": float(g["train_components"].mean()),
            "rf_mae_mean": float(g["rf_mae"].mean()),
            "rf_mae_p10": float(np.percentile(g["rf_mae"], 10)),
            "rf_mae_p90": float(np.percentile(g["rf_mae"], 90)),
            "rf_medae_mean": float(g["rf_medae"].mean()),
            "dummy_mae_mean": float(g["dummy_mae"].mean()),
            "dummy_mae_p10": float(np.percentile(g["dummy_mae"], 10)),
            "dummy_mae_p90": float(np.percentile(g["dummy_mae"], 90)),
            "rf_minus_dummy_mae": float(g["rf_mae"].mean() - g["dummy_mae"].mean()),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "learning_curve.json").write_text(json.dumps({
        "description": "Learning curve of the frozen RF vs the refit subcategory-median dummy, "
                       "strict component-grouped CV on the training split. Subsample unit = "
                       "connected component. Descriptive; frozen holdout never touched.",
        "frozen_cv_mean_MAE": expected_full_mae,
        "endpoint_reproduced": reproduced,
        "cv_splits": CV_SPLITS, "n_seeds": N_SEEDS,
        "curve": curve,
    }, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'learning_curve.json'}")

    # ---- Rough plot (functional, not publication-polished) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [c["mean_train_rows"] for c in curve]
        rf = [c["rf_mae_mean"] for c in curve]
        rf_lo = [c["rf_mae_p10"] for c in curve]
        rf_hi = [c["rf_mae_p90"] for c in curve]
        dm = [c["dummy_mae_mean"] for c in curve]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, rf, "o-", label="Random Forest (frozen)", color="#2563eb")
        ax.fill_between(x, rf_lo, rf_hi, alpha=0.15, color="#2563eb")
        ax.plot(x, dm, "s--", label="Subcategory-median dummy (refit)", color="#dc2626")
        ax.set_xlabel("Training rows (subsampled by component)")
        ax.set_ylabel("Component-grouped CV MAE (€)")
        ax.set_title("Learning curve — data-limited vs signal-limited")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150)
        print(f"Wrote {OUTPUT_DIR / 'learning_curve.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
