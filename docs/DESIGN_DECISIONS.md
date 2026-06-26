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
