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

## 2026-06-26 - Old Strict Setup Is Not Automatically Final

Decision: The old strict setup is historical/contextual only until the final strict identity rule is decided.

Historical setup:

```text
part_name + brand + model + oem_number
```

Reasoning: Later inspection found concerns about OEM reuse and identity fragmentation.

## 2026-06-26 - OEM Usage in Strict Identity Remains Under Review

Decision: `oem_number` usage in the final strict identity key remains under review.

Current preferred candidate:

```text
canonical_part_name + canonical_brand + canonical_model
```

Reasoning: OEM values are complete but low-cardinality and reused across unrelated part identities. Including OEM can fragment groups and may make the strict split less conservative.

## 2026-06-26 - Compatibility-Family Evaluation Is Robustness Analysis

Decision: Compatibility-family evaluation will be used as robustness analysis.

Boundary: It does not replace the main strict evaluation. The main thesis evidence should come from the final strict split and strict holdout.

## 2026-06-26 - Old Runs Are Historical, Not Final Evidence

Decision: Old grouped and strict runs are historical/contextual.

Reasoning: The final thesis should rely on the new documented pipeline and future final reruns. Historical runs can explain the development path and motivate methodological decisions, but should not be presented as final results.
