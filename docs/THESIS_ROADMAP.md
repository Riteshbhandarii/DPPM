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

The final strict split is frozen (`datasets/splits_strict/`, seed 32) and the stage-1 model comparison under it is complete (2026-07-07): all four models were compared with their known configurations on the fixed strict validation split. Ridge (92.05 EUR MAE) and Random Forest (92.93) advance to the tuning stage; XGBoost (106.91) and CatBoost (168.82) were eliminated. Protocol and results: [docs/STRICT_MODEL_COMPARISON.md](STRICT_MODEL_COMPARISON.md).

The immediate task is stage-2 tuning of the two finalists (full config search ranked by component-grouped CV inside the strict training split), followed by the single final holdout evaluation.

## Next Milestones

1. Generate a new strict connected-component split directly from `datasets/cleaned/clean_master_dataset.csv`.
2. Rerun model selection under the new strict split.
3. Evaluate the selected model on the final strict holdout.
4. Run compatibility-family robustness analysis.
5. Run SHAP explainability for the final model.
6. Run subgroup analysis by brand, category, and price band.
7. Prepare final thesis artifacts and tables.

## GitHub Issue Mapping

The current roadmap maps to the open DPPM thesis issues below.

| Area | Issue | Status |
| --- | --- | --- |
| Strict identity rule and strict split design | #34 | Documented |
| Preserve grouped baseline / transition narrative | #38 | In progress |
| Generate final strict split | #35 | Done — frozen artifacts in `datasets/splits_strict/` |
| Strict model selection rerun | #36 | Stage 1 done (4-model comparison, 2026-07-07) — finalist tuning next |
| Final strict holdout evaluation | #37 | Not started (guarded run-once notebook ready) |
| Compatibility-family robustness | #46 | Not started |
| Subgroup analysis | #41 | Not started |
| Feature leakage assessment | #42 | Not started |
| Reproducible final scripts | #47 | Not started |
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
- [ ] Rerun model selection.
- [ ] Evaluate final strict holdout.
- [ ] Run robustness and explainability analyses.
- [ ] Prepare final thesis results tables.
