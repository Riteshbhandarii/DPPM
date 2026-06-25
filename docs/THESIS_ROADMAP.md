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

The cleaned modeling dataset has been audited and is suitable for a bachelor thesis proof-of-concept, provided its limitations are documented.

The earlier `product_id` grouped split has been verified as a leakage-aware optimistic baseline for repeated listing observations. It is preserved as historical/contextual evidence and an operational benchmark, not as the final conservative thesis result.

## Completed Checks

- [x] Dataset characterized and assessed as suitable for proof-of-concept use.
- [x] Missingness checked for target and core identifiers.
- [x] Duplicate and repeated listing behavior checked.
- [x] Product-id grouped split verified to have zero `product_id` overlap across train, validation, and test.
- [x] Grouped baseline classified as an optimistic operational benchmark.
- [x] Existing strict identity logic inspected.
- [x] OEM number reliability concerns identified for final strict identity design.

## Current Phase

Document methodology and define the final strict evaluation protocol before generating new final splits or rerunning models.

The immediate methodological task is to finalize the strict identity rule used to prevent comparable-part leakage.

## Next Milestones

1. Decide final strict identity rule.
2. Generate a new strict split directly from `datasets/cleaned/clean_master_dataset.csv`.
3. Rerun model selection under the new strict split.
4. Evaluate the selected model on the final strict holdout.
5. Run compatibility-family robustness analysis.
6. Run SHAP explainability for the final model.
7. Run subgroup analysis by brand, category, and price band.
8. Prepare final thesis artifacts and tables.

## GitHub Issue Mapping

Only the strict identity issue number is currently known from the working context.

| Area | Issue | Status |
| --- | --- | --- |
| Strict identity rule and strict split design | #34 | In progress |
| Generate final strict split | TBD | Not started |
| Final model reruns | TBD | Not started |
| Final holdout evaluation | TBD | Not started |
| Compatibility-family robustness | TBD | Not started |
| SHAP and subgroup analysis | TBD | Not started |
| Final thesis tables and artifacts | TBD | Not started |

## Progress Checklist

### Completed

- [x] Dataset suitability investigation
- [x] Grouped baseline audit
- [x] Current strict identity implementation audit
- [x] Documentation system created

### Current Decisions

- [x] Do not recollect the dataset unless a fundamental invalidating issue is found.
- [x] Preserve product-id grouped baseline as optimistic/contextual benchmark.
- [x] Do not accept the old strict setup automatically as the final thesis protocol.
- [x] Generate the future final strict split directly from `clean_master_dataset.csv`.

### Future Work

- [ ] Finalize strict identity rule.
- [ ] Implement or document strict split generation.
- [ ] Generate final strict split.
- [ ] Rerun model selection.
- [ ] Evaluate final strict holdout.
- [ ] Run robustness and explainability analyses.
- [ ] Prepare final thesis results tables.
