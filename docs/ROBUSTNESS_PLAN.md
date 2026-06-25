# Robustness Plan

## Purpose

The robustness plan defines supporting analyses for the final thesis model. These analyses should diagnose stability and limitations, not replace the main strict evaluation.

## Required Robustness Analyses

### Compatibility-Family Evaluation

Evaluate whether model behavior is stable across broader compatibility or part-family groupings.

This is a robustness diagnostic, not a replacement for the main strict evaluation. The main evidence should remain the final strict split and strict holdout.

### SHAP Explainability

Run SHAP analysis for the final selected model after final reruns.

Expected outputs:

- Global feature importance.
- Feature-group importance.
- Local example explanations.
- Dependence plots for key numeric and categorical features where appropriate.

Historical SHAP outputs may guide planning, but final thesis explanations should be regenerated for the final model.

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
2. Final strict holdout evaluation
3. Documented leakage checks
