# Baseline Evaluation

## Purpose

This document records the status of the historical product-id grouped baseline. It is preserved as an optimistic operational benchmark, not as final conservative thesis evidence.

Final thesis evidence will come from a future documented strict split and final reruns.

## Grouped Split Definition

The grouped split uses:

```text
product_id
```

This prevents repeated observations of the same marketplace listing from crossing train, validation, and test splits.

## Split Files

| Split | File | Rows | Unique product IDs |
| --- | --- | ---: | ---: |
| Train | `datasets/splits/train_grouped.csv` | 7,954 | 1,833 |
| Validation | `datasets/splits/validation_grouped.csv` | 1,689 | 393 |
| Test | `datasets/splits/test_grouped.csv` | 1,678 | 393 |
| Assignment | `datasets/splits/group_split_assignment.csv` | 2,619 product IDs | 2,619 |

## Completed Verification

- [x] `product_id` overlap between train and validation is zero.
- [x] `product_id` overlap between train and test is zero.
- [x] `product_id` overlap between validation and test is zero.
- [x] `group_split_assignment.csv` matches the split files.

## Interpretation

The product-id grouped split is leakage-aware for repeated listing observations. It is useful as an optimistic benchmark for an operational setting where the model predicts prices for unseen listings but may still see highly comparable part identities during training.

## Known Limitation

Comparable part identities can still cross product-id splits. Different listings may represent the same or near-identical modeled part identity, even when `product_id` does not overlap.

This limitation means the grouped baseline is not final conservative evidence for generalization to unseen comparable part identities.

## Current Decision

- Preserve the grouped baseline as historical/contextual evidence.
- Treat it as an optimistic operational benchmark.
- Do not rerun it unless fresh verification logs are needed.
- Use it to motivate the stricter final evaluation protocol.

## Future Evidence Boundary

Historical grouped results may be mentioned in the thesis only as context. Final model claims should be based on the future strict split and final reruns.
