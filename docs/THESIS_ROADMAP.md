# DPPM Thesis Roadmap

## Project Overview

DPPM is a bachelor thesis project for spare-part price prediction. The project combines marketplace listing data with Traficom-derived vehicle registry context and evaluates machine-learning models as a proof-of-concept decision-support tool.

The current documentation system separates:

1. Completed checks
2. Historical/contextual results
3. Current methodological decisions
4. Future work required for final thesis evidence

## Thesis Objective

Build and evaluate a reproducible proof-of-concept model for estimating used spare-part listing prices for selected passenger-car model families.

The final thesis evidence should come from the documented final pipeline and future final reruns, not from earlier exploratory or historical runs.

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

Modeling is finished. All remaining work is analysis and writing. No model may be scored on the test split again.

**SHAP is done (2026-07-10, #62), descriptive only.** The frozen model spends **80.67 %** of its attribution on three taxonomy features, `subcategory` alone carrying 66.55 %. The 49 Traficom registry features carry **3.81 %** between them; `mileage` ranks 7th at 0.73 %. This is the mechanism behind the tie with the subcategory-median lookup. Full result: [docs/STRICT_MODEL_COMPARISON.md](STRICT_MODEL_COMPARISON.md) section 11.

## Next Milestones

1. Learning curve on train/validation, holdout-safe, to establish whether the model is data-limited or signal-limited (#63). This determines whether "dataset too small" is a quantified finding or an unsupported hedge.
2. Subgroup analysis by brand and category (#41) — the price-band half is already done in `artifacts/strict_final_holdout/holdout_baseline_comparison.json`.
3. Formal feature leakage audit (#42) — appendix table.
4. Rewrite the contribution statement around the administered-pricing result and write the results and discussion chapters (#64; see `DESIGN_DECISIONS.md`, 2026-07-10).
5. Convert final notebook workflows into reproducible scripts (#47), then clean the repo surface (#40).

## GitHub Issue Mapping

The current roadmap maps to the open DPPM thesis issues below.

| Area | Issue | Status |
| --- | --- | --- |
| Strict identity rule and strict split design | #34 | Documented |
| Preserve grouped baseline / transition narrative | #38 | In progress |
| Generate final strict split | #35 | Done — frozen artifacts in `datasets/splits_strict/` |
| Strict model selection rerun | #36 | **Closed 2026-07-10** — stages 1 and 2 complete, winner frozen and documented |
| Final strict holdout evaluation | #37 | **Closed 2026-07-10** — run once, guard consumed, numbers final |
| SHAP explainability (descriptive) | #62 | **Closed 2026-07-10** — `artifacts/strict_final_shap/`, section 11 |
| Learning curve: data-limited or signal-limited | #63 | Not started — next |
| Subgroup analysis | #41 | Price-band half done via the holdout baseline comparison; brand/category remain |
| Feature leakage assessment | #42 | Not started |
| Contribution statement rewrite + results/discussion | #64 | Not started — the thesis writing |
| Compatibility-family robustness | #46 | **Closed as not planned 2026-07-08** — superseded; the connected-component split *is* the broader grouping |
| Reproducible final scripts | #47 | Partly done — `run_strict_shap.py` and `holdout_baseline_comparison.py` written as scripts. Any stage-3 script must stay guarded or read-only |
| Final full rerun and artifact freeze | #50 | Not started |

## Progress Checklist

### Completed

- [x] Dataset suitability investigation
- [x] Grouped baseline verification
- [x] Current strict identity implementation review
- [x] Documentation system created
- [x] Candidate identity and fragmentation diagnostics completed
- [x] Connected-component split balance diagnostics completed
- [x] Final strict evaluation protocol documented

### Current Decisions

- [x] Do not recollect the dataset unless a fundamental invalidating issue is found.
- [x] Preserve product-id grouped baseline as optimistic/contextual benchmark.
- [x] Use connected-component splitting as the final strict thesis protocol.
- [x] Generate the future final strict split directly from `clean_master_dataset.csv`.
- [x] Use `canonical(part_name, brand, model, year_start, year_end)` as the strict identity key.

### Future Work

- [x] Finalize strict identity rule.
- [x] Document strict split generation requirements.
- [x] Generate final strict split (frozen under `datasets/splits_strict/`, seed 32).
- [x] Rerun model selection.
- [x] Evaluate final strict holdout (2026-07-10, run once).
- [x] Run explainability (SHAP, descriptive) on the frozen model (2026-07-10, #62).
- [ ] Run the holdout-safe learning curve (#63).
- [ ] Rewrite the contribution statement around the administered-pricing finding.
- [ ] Prepare final thesis results tables.
