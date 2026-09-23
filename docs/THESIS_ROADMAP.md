# DPPM Thesis Roadmap

## Project Overview

DPPM is a bachelor thesis project for spare-part price prediction. The project combines marketplace listing data with Traficom-derived vehicle registry context and evaluates machine-learning models as a proof-of-concept decision-support tool.

## Thesis Objective

Build and evaluate a reproducible proof-of-concept model for estimating used spare-part listing prices for selected passenger-car model families.

Reported results come from the documented strict pipeline, not from earlier exploratory or historical runs.

## Current Status

The cleaned modeling dataset has been characterized and is suitable for a bachelor thesis proof-of-concept, provided its limitations are documented.

The earlier `product_id` grouped split has been verified as a leakage-aware optimistic baseline for repeated listing observations. It is preserved as historical/contextual evidence and an operational benchmark, not as the final conservative thesis result.

The final strict evaluation protocol has now been selected. It uses connected components built from `product_id` and `canonical(part_name, brand, model, year_start, year_end)`. The full decision record is maintained in [docs/evaluation/01_PROTOCOL_DECISION.md](evaluation/01_PROTOCOL_DECISION.md).

## Completed Checks

- [x] Dataset characterized and assessed as suitable for proof-of-concept use.
- [x] Missingness checked for target and core identifiers.
- [x] Duplicate and repeated listing behavior checked.
- [x] Product-id grouped split verified to have zero `product_id` overlap across train, validation, and test.
- [x] Grouped baseline classified as an optimistic operational benchmark.
- [x] Existing strict identity logic inspected.
- [x] OEM number reliability concerns identified for final strict identity design.

## Current Phase

The final strict split is frozen (`datasets/splits_strict/`, seed 32). The stage-1 model comparison under it is complete (2026-07-07): all four models were compared with their known configurations on the fixed strict validation split. Ridge (92.05 EUR MAE) and Random Forest (92.93) advanced to stage 2; XGBoost (106.91) and CatBoost (168.82) were eliminated. Protocol and results: [docs/STRICT_MODEL_COMPARISON.md](STRICT_MODEL_COMPARISON.md).

Stage-2 tuning is complete. Under component-grouped cross-validation inside the strict training split, Random Forest won the primary MAE comparison (by 1.38 EUR) and Ridge remained the linear runner-up.

**Stage 3 is complete: the final strict holdout ran once on 2026-07-10 and the guard is consumed.** Random Forest scored MAE 69.46 EUR, median AE 29.37 EUR, RMSE 182.41 EUR, R2 0.9113. On the same rows a subcategory-median lookup scored MAE 66.15 EUR and median AE 15.32 EUR — the model ties the heuristic on MAE and is significantly worse on median AE, winning significantly only in the 500-1,000 EUR band. Full result, bootstrap intervals, and interpretation: [docs/STRICT_MODEL_COMPARISON.md](STRICT_MODEL_COMPARISON.md) sections 8-10; decision records in [docs/DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) (2026-07-10).

Modeling is finished. No model may be scored on the test split again.

**SHAP is done (2026-07-10, #62), descriptive only.** The frozen model spends **80.67 %** of its attribution on three taxonomy features, `subcategory` alone carrying 66.55 %. The 49 Traficom registry features carry **3.81 %** between them; `mileage` ranks 7th at 0.73 %. This is the mechanism behind the tie with the subcategory-median lookup. Full result: [docs/STRICT_MODEL_COMPARISON.md](STRICT_MODEL_COMPARISON.md) section 11.

## Build status

The build work is finished:

- Learning curve and registry ablation (2026-07-11, #63): `artifacts/learning_curve/`, `artifacts/registry_ablation/`.
- September 2026 live validation: listings chosen by hand and scored with the frozen model, on the three trained vehicles and on three vehicles the model never saw. `results/september_live_validation/`.
- Holdout error by brand and category (#41): `artifacts/holdout_subgroup_errors/`.
- Feature leakage audit (#42): `artifacts/leakage_audit/`, [LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md).

Headline numbers: [RESULTS.md](RESULTS.md).

## GitHub Issue Mapping

| Area | Issue | Status |
| --- | --- | --- |
| Strict identity rule and strict split design | #34 | Documented |
| Generate final strict split | #35 | Done — frozen artifacts in `datasets/splits_strict/` |
| Strict model selection rerun | #36 | Closed 2026-07-10 |
| Final strict holdout evaluation | #37 | Closed 2026-07-10 — run once, guard consumed |
| Preserve grouped baseline / transition narrative | #38 | Closed |
| Subgroup analysis | #41 | Price bands in `holdout_baseline_comparison.json`; brand/category in `artifacts/holdout_subgroup_errors/` |
| Feature leakage assessment | #42 | `artifacts/leakage_audit/`, [LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md) |
| Compatibility-family robustness | #46 | Closed as not planned 2026-07-08 — the connected-component split is the broader grouping |
| Notebooks to scripts | #47 | Not planned — the notebooks stay as the record of the data preparation |
| Final full rerun and artifact freeze | #50 | Closed 2026-06-26 |
| SHAP explainability (descriptive) | #62 | Closed 2026-07-10 — `artifacts/strict_final_shap/` |
| Learning curve | #63 | Closed — `artifacts/learning_curve/` |

## Progress Checklist

- [x] Dataset suitability investigation
- [x] Grouped baseline verification
- [x] Candidate identity and fragmentation diagnostics
- [x] Connected-component split balance diagnostics
- [x] Final strict evaluation protocol documented
- [x] Final strict split generated (seed 32)
- [x] Model selection rerun under the strict protocol
- [x] Final strict holdout evaluated (2026-07-10, run once)
- [x] SHAP on the frozen model
- [x] Learning curve and registry ablation
- [x] Subgroup error analysis
- [x] Feature leakage audit
- [x] September 2026 live validation

### Standing decisions

- Do not recollect the February dataset.
- The product-id grouped split stays as an optimistic benchmark only.
- The connected-component split is the evaluation protocol; its identity key is `canonical(part_name, brand, model, year_start, year_end)`.
- No model is scored on the connected-component test set again.
