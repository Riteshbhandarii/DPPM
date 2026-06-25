# Dataset Characteristics

## Source File

Primary cleaned modeling dataset:

```text
datasets/cleaned/clean_master_dataset.csv
```

This file is the intended source for future final strict split generation.

## Overview

| Property | Value |
| --- | ---: |
| Rows | 11,321 |
| Columns | 85 |
| Target variable | `price` |
| Unique `product_id` values | 2,619 |
| Scrape dates | 6 |
| Date range | 2026-02-03 to 2026-02-18 |

## Feature Groups

| Feature group | Count | Examples |
| --- | ---: | --- |
| Identifier and dates | 4 | `product_id`, `scrape_date`, `first_seen_date`, `last_seen_date` |
| Target | 1 | `price` |
| Part/listing categorical | 8 | `part_name`, `quality_grade`, `oem_number`, `brand`, `model`, `category`, `subcategory` |
| Vehicle age/usage | 6 | `mileage`, `year_start`, `year_end`, `year_span`, `year_mid` |
| Merge diagnostics | 5 | `brand_merge_key`, `model_merge_key`, `repair_status` |
| Model-level registry context | 25 | `model_total_registered`, first-registration and vehicle-profile aggregates |
| Brand-level registry context | 24 | `brand_total_registered`, first-registration and vehicle-profile aggregates |
| Point-in-time listing history | 6 | `observations_so_far`, `days_since_first_seen_so_far` |
| Full-span listing history | 6 | `times_observed`, `observed_span_days`, price-change summaries |

## Missing Data

Core thesis fields have no missing values.

| Column | Missing rows | Missing share |
| --- | ---: | ---: |
| `price` | 0 | 0.00% |
| `product_id` | 0 | 0.00% |
| `brand` | 0 | 0.00% |
| `model` | 0 | 0.00% |
| `category` | 0 | 0.00% |
| `subcategory` | 0 | 0.00% |
| `part_name` | 0 | 0.00% |
| `oem_number` | 0 | 0.00% |
| `mileage` | 1,156 | 10.21% |

`mileage` missingness is material but acceptable for modeling because it is explicitly flagged with `mileage_missing_flag`.

## Duplicate and Repeated Snapshot Behavior

| Check | Result |
| --- | ---: |
| Exact duplicate rows | 0 |
| Same `product_id` + same `scrape_date` duplicate groups | 0 |
| Product IDs appearing across multiple dates | 1,785 |
| Product IDs with price changes | 1,777 |

Repeated `product_id` observations are intentional. The dataset records repeated marketplace snapshots over six scrape dates. These rows should be treated as repeated listing observations rather than accidental duplicates.

## Coverage

### Model Families

| Brand | Model | Rows |
| --- | --- | ---: |
| Toyota | Corolla | 3,894 |
| VW | Golf | 3,790 |
| Skoda | Octavia | 3,637 |

### Part Coverage

| Property | Value |
| --- | ---: |
| Categories | 7 |
| Subcategories | 137 |
| Part names | 486 |
| OEM numbers | 54 |

## Price Distribution

| Statistic | Price |
| --- | ---: |
| Minimum | 5.90 |
| Median | 100.60 |
| Mean | 270.79 |
| 95th percentile | 947.20 |
| 99th percentile | 3,927.00 |
| Maximum | 4,641.00 |

The price distribution is strongly right-skewed. High-price observations are mainly large components such as engines and automatic gearboxes, so they appear plausible rather than impossible.

## Limitations

- One marketplace source.
- Asking prices, not completed transaction prices.
- Short observation window from 2026-02-03 to 2026-02-18.
- Three model families only.
- Repeated listing snapshots mean rows are not fully independent.
- OEM numbers are complete but appear reused/noisy and should be handled carefully.
- Part taxonomy and text fields contain formatting inconsistencies.
- The dataset reflects availability and seller behavior in the sampled marketplace period.

## Conclusion

The dataset is suitable for a bachelor thesis proof-of-concept. It is internally consistent enough for documented modeling work, but the thesis must explicitly acknowledge scope, marketplace, repeated-snapshot, and identity-leakage limitations.
