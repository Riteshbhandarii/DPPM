# Experiment History

## Purpose

This file records lightweight historical context. It should not be used as the source of final thesis evidence.

Final thesis evidence will be regenerated through the documented future pipeline.

## Dataset Characterization Summary

The cleaned modeling dataset has been assessed as suitable for a bachelor thesis proof-of-concept:

- 11,321 rows
- 85 columns
- Target variable: `price`
- 2,619 unique `product_id` values
- No missing target or core identifiers
- 1,156 missing mileage values, equal to 10.21%
- No exact duplicate rows
- No same `product_id` + same `scrape_date` duplicate groups
- Repeated listing snapshots are intentional

The main limitations are marketplace scope, asking-price labels, short scrape window, repeated snapshots, and identity/taxonomy noise.

## Grouped Baseline Verification Summary

The historical product-id grouped split has been verified:

- Train: 7,954 rows / 1,833 product IDs
- Validation: 1,689 rows / 393 product IDs
- Test: 1,678 rows / 393 product IDs
- Product-id overlap across train/validation/test: zero
- `group_split_assignment.csv` matches the split files

This split is preserved as an optimistic operational benchmark.

## Historical Grouped and Strict Results

Earlier grouped and strict model outputs exist in the repository. They may be useful for context and planning, but they are not final thesis evidence.

Historical interpretation:

- Product-id grouped evaluation estimated performance when repeated listings are separated but comparable part identities may still cross splits.
- Earlier strict evaluation attempted to hold out comparable part identities using `part_name + brand + model + oem_number`.
- Later inspection raised concerns about using `oem_number` in the final strict identity key.

## Evidence Boundary

Do not present old grouped or strict results as final thesis evidence.

Use historical results only to explain:

- Why strict evaluation is needed.
- Why the strict identity rule must be documented carefully.
- Why final reruns are required.

Final thesis claims should come from the future documented strict split and final reruns.
