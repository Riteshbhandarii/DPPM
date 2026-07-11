"""Registry ablation: does the Traficom join pay for itself? (holdout-safe, CV only)

Confirmatory ablation behind the SHAP negative result (see docs / thesis note 2026-07-11):
the Traficom registry features are model/brand-level aggregates that collapse to ~3 constants
over the 3-model PoC scope and are collinear with the ``model``/``brand`` identity already in
the model. This script measures their *marginal* contribution by holding everything constant
and toggling only the registry feature block, for both the frozen Random Forest winner and the
frozen Ridge finalist, on the strict component-grouped CV inside the training split.

Design guarantees:
  * Reuses ``fit_and_predict`` + ``GroupKFold`` from ``src.strict_funnel`` -> folds, preprocessing
    and target transform (raw for RF, log-with-back-transform for Ridge) are byte-identical to the
    original selection run. Predictions come back on the euro scale.
  * Each model uses its OWN frozen config/feature-variant recovered from the tuning artifacts.
  * The "without" arm is the "with" arm MINUS exactly the registry columns (``model_*``/``brand_*``),
    derived by subtraction so the delta is attributable to nothing else.
  * GUARDRAIL: the with-registry arm must reproduce the frozen selection CV MAE (RF 105.334,
    Ridge 106.706) before any delta is trusted. If it does not, fold construction differs and the
    experiment is aborted.
  * NEVER touches the frozen strict test split. Descriptive only; cannot change model selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.strict_funnel import (
    TARGET_COLUMN,
    euro_metrics,
    fit_and_predict,
    load_strict_training_frames,
)
from src.strict_protocol import COMPONENT_GROUP_COLUMN

MODELS = ["random_forest", "ridge"]
TUNING_DIR = Path("artifacts/strict_model_tuning")
OUTPUT_DIR = Path("artifacts/registry_ablation")
CV_SPLITS = 4
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 32
REPRODUCTION_TOL_EUR = 0.75  # RF has bootstrap=False + fixed seeds; expect near-exact.


def is_registry_feature(name: str) -> bool:
    """Traficom aggregates are the only model_*/brand_* columns; identity is plain model/brand."""
    return name.startswith(("model_", "brand_"))


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_with: np.ndarray,
    pred_without: np.ndarray,
    b: int,
    seed: int,
) -> dict[str, float]:
    """Row-level paired bootstrap of (with - without) errors. Positive => registry arm worse.

    Matches the holdout_baseline_comparison methodology (mildly anti-conservative under the grouped
    structure; acceptable for a descriptive confirmatory check).
    """
    err_with = np.abs(pred_with - y_true)
    err_without = np.abs(pred_without - y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    d_mae = np.empty(b)
    d_medae = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        ew, eo = err_with[idx], err_without[idx]
        d_mae[i] = ew.mean() - eo.mean()
        d_medae[i] = np.median(ew) - np.median(eo)
    return {
        "delta_MAE": float(err_with.mean() - err_without.mean()),
        "delta_MAE_ci_low": float(np.percentile(d_mae, 2.5)),
        "delta_MAE_ci_high": float(np.percentile(d_mae, 97.5)),
        "delta_median_AE": float(np.median(err_with) - np.median(err_without)),
        "delta_median_AE_ci_low": float(np.percentile(d_medae, 2.5)),
        "delta_median_AE_ci_high": float(np.percentile(d_medae, 97.5)),
        "share_rows_registry_closer": float(np.mean(err_with < err_without)),
    }


def run_arm_cv(model, train_df, features, config):
    """Hand-rolled component CV mirroring cv_candidates, but collecting per-row OOF predictions.

    Returns pooled (y_true, preds) and the list of per-fold MAEs (whose mean matches cv_mean_MAE).
    """
    y_full = train_df[TARGET_COLUMN]
    groups = train_df[COMPONENT_GROUP_COLUMN]
    gkf = GroupKFold(n_splits=CV_SPLITS)
    y_parts, pred_parts, fold_maes = [], [], []
    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_full, groups), start=1):
        fold_tr = train_df.iloc[tr_idx]
        fold_va = train_df.iloc[va_idx]
        preds = fit_and_predict(
            model, fold_tr, fold_va, features, config, COMPONENT_GROUP_COLUMN,
            early_stopping_seed=42 + fold_id,
        )
        y_true = fold_va[TARGET_COLUMN].to_numpy(dtype=float)
        y_parts.append(y_true)
        pred_parts.append(np.asarray(preds, dtype=float))
        fold_maes.append(float(np.mean(np.abs(np.asarray(preds, dtype=float) - y_true))))
    return np.concatenate(y_parts), np.concatenate(pred_parts), fold_maes


def main() -> None:
    train_df, _validation_df = load_strict_training_frames()
    print(f"Loaded strict TRAIN split: {len(train_df):,} rows "
          f"({train_df[COMPONENT_GROUP_COLUMN].nunique():,} components)\n")

    results = {}
    for model in MODELS:
        summary = json.loads((TUNING_DIR / model / "best_tuning_summary.json").read_text())
        with_features = list(summary["feature_names"])
        config = summary["config"]
        expected_mae = float(summary["cv_mean_MAE"])
        registry = [f for f in with_features if is_registry_feature(f)]
        without_features = [f for f in with_features if not is_registry_feature(f)]

        print(f"===== {model} | {summary['feature_variant']} | {summary['config_name']} =====")
        print(f"  with:    {len(with_features)} features  (registry dropped in 'without': {len(registry)})")
        print(f"  without: {len(without_features)} features -> {without_features}")

        # Paired arms on identical folds.
        y_true, pred_with, fold_maes_with = run_arm_cv(model, train_df, with_features, config)
        _, pred_without, fold_maes_without = run_arm_cv(model, train_df, without_features, config)

        fold_avg_mae_with = float(np.mean(fold_maes_with))
        fold_avg_mae_without = float(np.mean(fold_maes_without))
        reproduced = abs(fold_avg_mae_with - expected_mae) <= REPRODUCTION_TOL_EUR

        print(f"  reproduction check: with-arm fold-avg MAE = {fold_avg_mae_with:.3f} "
              f"vs frozen {expected_mae:.3f}  -> {'OK' if reproduced else 'MISMATCH!'}")
        if not reproduced:
            raise SystemExit(
                f"ABORT: {model} with-arm did not reproduce the frozen CV MAE "
                f"({fold_avg_mae_with:.3f} vs {expected_mae:.3f}). Fold construction differs; "
                f"delta is not trustworthy."
            )

        boot = paired_bootstrap_delta(y_true, pred_with, pred_without, BOOTSTRAP_B, BOOTSTRAP_SEED)
        with_metrics = euro_metrics(pd.Series(y_true), pred_with, prefix="oof")
        without_metrics = euro_metrics(pd.Series(y_true), pred_without, prefix="oof")

        print(f"  fold-avg MAE  with={fold_avg_mae_with:.2f}  without={fold_avg_mae_without:.2f}  "
              f"(delta {fold_avg_mae_with - fold_avg_mae_without:+.2f})")
        print(f"  pooled paired dMAE = {boot['delta_MAE']:+.2f} EUR "
              f"[{boot['delta_MAE_ci_low']:+.2f}, {boot['delta_MAE_ci_high']:+.2f}]  "
              f"(positive => registry HELPS)")
        print(f"  pooled paired dMedAE = {boot['delta_median_AE']:+.2f} EUR "
              f"[{boot['delta_median_AE_ci_low']:+.2f}, {boot['delta_median_AE_ci_high']:+.2f}]\n")

        results[model] = {
            "feature_variant": summary["feature_variant"],
            "config_name": summary["config_name"],
            "target_mode": summary.get("target_mode"),
            "n_features_with": len(with_features),
            "n_features_without": len(without_features),
            "n_registry_features_dropped": len(registry),
            "registry_features_dropped": registry,
            "frozen_cv_mean_MAE": expected_mae,
            "reproduced_with_arm": reproduced,
            "fold_avg_MAE_with": fold_avg_mae_with,
            "fold_avg_MAE_without": fold_avg_mae_without,
            "fold_MAEs_with": fold_maes_with,
            "fold_MAEs_without": fold_maes_without,
            "pooled_oof_with": {k: v for k, v in with_metrics.items()},
            "pooled_oof_without": {k: v for k, v in without_metrics.items()},
            "paired_bootstrap": boot,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "registry_ablation.json"
    out_path.write_text(json.dumps(
        {
            "description": "Marginal contribution of the Traficom registry feature block, "
                           "strict component-grouped CV on the training split. Descriptive; "
                           "the frozen holdout is never touched.",
            "cv_splits": CV_SPLITS,
            "group_column": COMPONENT_GROUP_COLUMN,
            "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "type": "row-level paired"},
            "sign_convention": "delta = with_registry - without_registry; positive => registry helps",
            "models": results,
        },
        indent=2,
    ))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
