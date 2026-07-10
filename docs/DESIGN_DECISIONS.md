# Design Decisions

This file records current thesis-methodology decisions. Entries are dated so future changes can be tracked.

## 2026-06-26 - Dataset Will Not Be Recollected

Decision: The dataset will not be recollected.

Reasoning: Scraping is no longer possible in the current project context. The current cleaned dataset is internally consistent enough for a bachelor thesis proof-of-concept, so the correct approach is to document limitations rather than restart data collection.

## 2026-06-26 - Dataset Will Be Characterized and Limitations Documented

Decision: The thesis will explicitly characterize the dataset and document its limitations.

Required limitations:

- One marketplace source.
- Asking prices rather than transaction prices.
- Short scrape window.
- Three model families only.
- Repeated listing snapshots.
- OEM reuse/noise.
- Taxonomy and text inconsistencies.

## 2026-06-26 - Product-ID Grouped Split Preserved as Optimistic Baseline

Decision: The existing product-id grouped split is preserved as an optimistic baseline.

Reasoning: The split has zero `product_id` overlap across train, validation, and test, so it is leakage-aware for repeated listing observations.

Boundary: This baseline is historical/contextual and operationally useful, but it is not final conservative thesis evidence.

## 2026-06-26 - Grouped Baseline Will Not Be Retrained by Default

Decision: The grouped baseline will not be retrained unless fresh verification logs are needed.

Reasoning: The baseline has already been verified for split integrity. Future effort should focus on the final strict protocol and final reruns.

## 2026-06-26 - Final Strict Split Comes From Clean Master Dataset

Decision: The final strict split will be generated directly from:

```text
datasets/cleaned/clean_master_dataset.csv
```

Reasoning: The final split should be derived from the canonical cleaned dataset, not from historical grouped split files.

## 2026-06-26 - Old Strict Setup Was Not Automatically Final

Decision: The old strict setup is historical/contextual and is not the selected final strict identity rule.

Historical setup:

```text
part_name + brand + model + oem_number
```

Reasoning: Later inspection found concerns about OEM reuse and identity fragmentation.

## 2026-06-26 - OEM Excluded From Final Strict Identity

Decision: `oem_number` is excluded from the final strict identity key.

Reasoning: The fragmentation diagnostics showed that adding OEM to `part_name + brand + model` split 348 base groups and created 309 new singleton groups. This was stronger fragmentation than the compatibility-year identity and raised the risk that OEM would act as a noisy fragmentation key rather than a reliable identity boundary.

Reference: [Evaluation protocol decision](evaluation/01_PROTOCOL_DECISION.md).

## 2026-06-26 - Final Strict Protocol Uses Connected Components

Decision: The final strict thesis evaluation protocol uses connected components built from:

```text
product_id
OR
canonical(part_name, brand, model, year_start, year_end)
```

Target proportions:

```text
train: 70%
validation: 15%
test: 15%
```

Diagnostic seed:

```text
32
```

Reasoning: The candidate split balance diagnostic found 886 connected components and showed that diagnostic seed 32 produced 70.05% / 14.97% / 14.98% row proportions with zero product-id, identity-key, or connected-component leakage.

Reference: [Strict protocol specification](evaluation/02_PROTOCOL_SPECIFICATION.md).

## 2026-06-26 - Compatibility-Family Evaluation Is Robustness Analysis

Decision: Compatibility-family evaluation will be used as robustness analysis.

Boundary: It does not replace the main strict evaluation. The main thesis evidence should come from the final strict split and strict holdout.

## 2026-06-26 - Old Runs Are Historical, Not Final Evidence

Decision: Old grouped and strict runs are historical/contextual.

Reasoning: The final thesis should rely on the new documented pipeline and future final reruns. Historical runs can explain the development path and motivate methodological decisions, but should not be presented as final results.

## 2026-07-07 - Stage-1 Model Comparison Procedure on the Strict Split

Decision: The four models (Ridge baseline, Random Forest, XGBoost, CatBoost) are compared on the frozen strict split using each model's known configurations from the original workflow, crossed with the trusted feature variants, fit on `train_strict` and scored once on `validation_strict`. No random-search tuning at this stage. The two best models by validation MAE proceed to tuning; the rule was fixed before results existed.

Reasoning: Keeping the procedure identical to the original fixed-validation comparison makes the strict-vs-original difference attributable to the evaluation design alone. Known configurations give every model its best-known setup (library defaults would bias the comparison arbitrarily). Full protocol and bias controls: docs/STRICT_MODEL_COMPARISON.md.

Result (2026-07-07): Ridge 92.05 EUR MAE, Random Forest 92.93, XGBoost 106.91, CatBoost 168.82; subcategory-median dummy 120.54. Ridge and Random Forest proceed to tuning.

## 2026-07-07 - Early Stopping Uses Inner Component-Grouped Carve

Decision: XGBoost and CatBoost early stopping always uses a 10% component-grouped carve of the (fold-)training data, never the validation split or CV fold being scored.

Reasoning: Early stopping on the scored data lets the evaluation set influence training, which biases results optimistically. The original grouped-CV code had this weakness; the strict pipeline corrects it.


## 2026-07-07 - Tuning Finalists: Ridge and Random Forest; XGBoost and CatBoost Eliminated

Decision: The strict-protocol tuning stage covers Ridge and Random Forest, per the pre-registered rule (two best models by stage-1 validation MAE). XGBoost and CatBoost receive no further tuning.

Reasoning: XGBoost (106.91 EUR MAE) trailed the untuned linear baseline (92.05) and has underperformed Random Forest under all three evaluation designs in this project (fixed validation, oem-identity strict CV, strict component split). The improvement needed to catch the leaders (~15%) exceeds gains realistically available from tuning already-tuned configurations while the leaders also improve. CatBoost (168.82) trails by ~80%, far beyond any plausible tuning gain, repeating its result under the earlier evaluation design. Tuning investment follows comparison performance under a consistent rule.

Escape hatch: if the finalist cross-validation results are inconclusive, the tuning stage may be widened with a documented amendment - but only before the final holdout runs.


## 2026-07-10 - Final Strict Holdout Was Executed Once; The Guard Is Consumed

Decision: The strict untouched test split (`datasets/splits_strict/test_strict.csv`, 1,696 rows) was evaluated exactly once, on 2026-07-10, against the frozen Random Forest winner. No further evaluation of any model on this split will be performed.

Provenance: run from `main` at `b530e59` (the merge of PR #58, which documented the Random Forest winner *before* the holdout, satisfying the pre-registered ordering). Winner: `random_forest`, feature variant `trusted_extended_traficom_stack_without_oem_number`, config `refinement_search_008`, raw target, 61 features, refit on `train_strict + validation_strict` (9,625 rows). Pre-flight verification: 23/23 tests passing, no pre-existing holdout artifacts, zero `product_id` overlap between fit and test, zero connected components spanning splits.

Result: MAE 69.46 EUR, median AE 29.37 EUR, RMSE 182.41 EUR, R2 0.9113. Bootstrap 95% CI (B=10,000, seed 32): MAE [61.70, 77.96], median AE [27.85, 31.92], R2 [0.8871, 0.9310]. Artifacts under `artifacts/strict_final_holdout/`.

Boundary: These numbers are final. The run-once guard in `notebooks/05_strict_training/03_strict_final_holdout.ipynb` has been consumed and the notebook will refuse to re-run while the artifacts exist. Deleting the artifacts to obtain a different number would invalidate the holdout claim. A rerun is legitimate only on a demonstrated pipeline bug, documented here first.


## 2026-07-10 - Trivial Baselines May Be Scored on the Holdout; Candidate Models May Not

Decision: The two trivial anchors (global median, per-subcategory median), fitted on `train_strict + validation_strict` exactly as the Random Forest was, were scored on the holdout for reference (`scripts/holdout_baseline_comparison.py`, results in `artifacts/strict_final_holdout/holdout_baseline_comparison.json`).

Reasoning: A reference baseline is part of *reporting* a holdout result, not a second attempt at *selecting* a model. Reporting a model's error without stating what a trivial rule achieves on the same data is uninformative. No model was fitted, chosen, retuned, or re-predicted by this comparison; the Random Forest predictions were read from the frozen artifact.

Boundary: This permission does not extend to candidate models. See the next entry.


## 2026-07-10 - Ridge Will Not Be Evaluated on the Holdout

Decision: Ridge, the stage-2 runner-up, will not be scored on the strict test split. The final claim rests on Random Forest alone.

Reasoning: Random Forest was frozen as the winner under the pre-registered primary metric (mean CV MAE) before the test split was touched. Evaluating Ridge afterwards - having now seen that Random Forest ties the subcategory-median dummy on MAE and loses to it on median AE, and knowing that Ridge cleared the dummy in cross-validation on both metrics - would be model selection on the test set. It would convert the holdout from an unbiased estimate into a shopped one, and would forfeit the pre-registration that is the methodological contribution of this thesis.

Consequence for the write-up: the thesis may claim "this tuned Random Forest does not beat a subcategory-median heuristic on typical listings". It may **not** claim "machine learning cannot beat the heuristic on this dataset" - the stage-2 cross-validation evidence, where Ridge beat the subcategory-median dummy by roughly 24% on MAE and 32% on median AE, actively contradicts the stronger claim. The distinction belongs in the limitations section, together with the observation that the winner was decided by a 1.38 EUR margin on mean CV MAE (105.33 vs 106.71) against a fold standard deviation of 24-34 EUR - a coin flip on a tail-dominated metric.


## 2026-07-10 - Contribution Claim on High-Value Inventory Is Withdrawn

Decision: The drafted contribution claim that "ML adds value over category heuristics specifically on high-value heterogeneous inventory (engines/gearboxes)" is withdrawn. It is not supported by the holdout.

Evidence: segment-wise paired bootstrap against the subcategory-median dummy shows the Random Forest is significantly *worse* below 100 EUR, indistinguishable between 100 and 500 EUR, significantly *better* only in the 500-1,000 EUR band (n=67), and indistinguishable above 1,000 EUR (n=75), where both approaches fail badly (MAE 718.94 vs 679.61 EUR).

Replacement claim: used spare-part asking prices in this marketplace are largely administered by subcategory convention; a tuned Random Forest reproduces that convention (prediction correlation with the dummy 0.9841) and adds modest incremental value (17.6% squared-error reduction over the dummy, RMSE 182.41 vs 200.94 EUR) concentrated in higher-priced, less conventional listings. The model therefore belongs in the dismantler workflow as a consistency check on high-value inventory, not as a pricing engine. This supports, rather than weakens, the pre-existing "market-consistency tool, not valuation tool" claim boundary.
