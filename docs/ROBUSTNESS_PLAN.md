# Robustness Plan

## Purpose

The robustness plan defines supporting analyses for the final thesis model. These analyses should diagnose stability and limitations, not replace the main strict evaluation.

## Required Robustness Analyses

### Compatibility-Family Evaluation

Evaluate whether model behavior is stable across broader compatibility or part-family groupings.

This is a robustness diagnostic, not a replacement for the main strict evaluation. The main evidence should remain the final strict split and strict holdout.

### SHAP Explainability — done 2026-07-10 (#62)

Run via `scripts/run_strict_shap.py` on the frozen strict winner, refit on train+validation exactly as the holdout did. Artifacts: `artifacts/strict_final_shap/`. Result and interpretation: `docs/STRICT_MODEL_COMPARISON.md` section 11.

| Expected output | Status |
| --- | --- |
| Global feature importance | Done — `shap_feature_importance.csv` |
| Feature-group importance | Done — `shap_group_importance.csv`; `part_taxonomy` 80.67 %, the 49 Traficom features 3.81 % between them |
| Segment-restricted attribution (500–1 000 €, above 1 000 €) | Done — the band where the model beats the heuristic, and the tail where it over-predicts |
| Local example explanations | Partly — the above-1 000 € tail is characterized by signed group push, not by individual listing walkthroughs |
| Dependence plots | **Not produced** |

Historical SHAP outputs (`artifacts/final_model_shap/`, `artifacts/final_model_shap_conservative/`, `artifacts/random_forest_shap/`, April 2026) predate the strict split and explain a different model (`trusted_recommended_features_without_oem_number` / `raw_half_features_leaf_1`). They are historical context, not thesis evidence, and were not regenerated.

### Feature Leakage Assessment

Assess final features for:

- Direct target leakage.
- Near-deterministic numeric target proxies.
- Full-history listing features that would not be available at prediction time.
- Split-specific preprocessing leakage.

Point-in-time listing-history features should be documented separately from full-span features.

### Subgroup Analysis

Report model performance by:

- Brand/model family.
- Category.
- Price band.
- Potentially mileage band, if useful and sample sizes are adequate.

Subgroup tables should include row counts so that sparse groups are not overinterpreted.

## Optional Robustness Analyses

### Temporal Evaluation

If feasible, evaluate whether later scrape dates are harder to predict than earlier ones. This should be treated as exploratory because the observation window is short.

### Uncertainty Calibration

If time permits, evaluate prediction intervals or uncertainty proxies. This is optional and should not be required for the core thesis result.

## Reporting Rule

Robustness analyses should be reported as supporting evidence. The final performance claim should be based on:

1. Final strict model selection
2. Final strict holdout evaluation — **done 2026-07-10, run once**
3. Documented leakage checks

Any robustness analysis proposed from here on must run on the training and validation splits only. The test split is spent, and a robustness result is never grounds to revisit the frozen model. Reference baselines (trivial anchors) may be reported on the holdout; candidate models may not. See `docs/DESIGN_DECISIONS.md` (2026-07-10).
