# Candidate Connected-Component Split Balance Diagnostic

Generated: 2026-06-26 16:58:33 UTC

Input dataset: `datasets/cleaned/clean_master_dataset.csv`
Target column: `price`

## Candidate Protocol

The proposed final thesis evaluation protocol keeps connected components intact. Each row is a node. Rows are connected if they share `product_id` or the canonical identity `part_name + brand + model + year_start + year_end`. Simulated train/validation/test assignments are made at component level using 70% / 15% / 15% row targets.

Canonicalization: Unicode `NFKC`, `casefold`, trimmed whitespace, collapsed repeated whitespace, missing values replaced with `__missing__`, and no fuzzy matching.

This script is diagnostic only. It does not train models and does not save final split files.

## Overview

| metric | value |
| --- | --- |
| rows | 11,321 |
| connected components | 886 |
| largest component rows | 71 |
| simulated seeds | 100 |
| best row-balance seed | 31 |
| best price-balance seed | 23 |
| recommended diagnostic seed | 32 |
| worst row imbalance observed | 0.31 percentage points (seed 53) |

## Summary Across Simulations

| split | target_pct | average_rows | min_rows | max_rows | std_rows | average_pct | min_pct | max_pct | std_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 70 | 7927.3800 | 7906 | 7952 | 7.9400 | 70.0200 | 69.8300 | 70.2400 | 0.0700 |
| validation | 15 | 1696.2100 | 1677 | 1709 | 5.8200 | 14.9800 | 14.8100 | 15.1000 | 0.0500 |
| test | 15 | 1697.4100 | 1680 | 1733 | 8.5000 | 14.9900 | 14.8400 | 15.3100 | 0.0800 |

Simulation metrics CSV: `results/candidate_component_split_balance/simulation_metrics.csv`

Full per-seed diagnostic CSVs:

- Price distributions: `results/candidate_component_split_balance/simulation_price_distributions.csv`
- Leakage checks: `results/candidate_component_split_balance/simulation_leakage_checks.csv`
- Brand/model/category/year distributions: `results/candidate_component_split_balance/simulation_distribution_diagnostics.csv`

## Best Diagnostic Seed

Scoring note: `row_balance_score` is the sum of absolute percentage-point deviations from the 70/15/15 row targets. `price_balance_score` sums each split's relative mean and median price deviation from the full dataset. Lower values are better.

| selection | seed | row_balance_score | price_balance_score | max_row_pct_deviation |
| --- | --- | --- | --- | --- |
| best row-count balance | 31 | 0.0053 | 1.3300 | 0.0026 |
| best price-distribution balance | 23 | 0.1793 | 0.2343 | 0.0897 |
| recommended diagnostic seed | 32 | 0.0936 | 0.2347 | 0.0468 |

Recommended seed: `32`. Selected because row proportions are within 2 percentage points of target and price balance is in the best quartile among simulated seeds.

The recommended assignment is saved only as a diagnostic artifact, not as final split files:

- `results/candidate_component_split_balance/diagnostic_recommended_seed_component_assignments.csv`
- `results/candidate_component_split_balance/diagnostic_recommended_seed_row_assignments.csv`

## Detailed Split Summary For Recommended Seed

| split | row_count | component_count | product_id_count | identity_key_count | row_pct | target_pct | pct_point_deviation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 7930 | 510 | 1802 | 599 | 70.0500 | 70 | 0.0500 |
| validation | 1695 | 178 | 408 | 209 | 14.9700 | 15 | -0.0300 |
| test | 1696 | 198 | 409 | 224 | 14.9800 | 15 | -0.0200 |

## Component Summary For Recommended Seed

| split | component_count | min_component_size | median_component_size | max_component_size |
| --- | --- | --- | --- | --- |
| train | 510 | 1 | 24 | 71 |
| validation | 178 | 1 | 14 | 70 |
| test | 198 | 1 | 12 | 48 |

## Leakage Check Results

| check | simulated_seeds | passed_seeds | failed_seeds | max_leaked_value_count |
| --- | --- | --- | --- | --- |
| connected component appears in only one split | 100 | 100 | 0 | 0 |
| identity key appears in only one split | 100 | 100 | 0 | 0 |
| product_id appears in only one split | 100 | 100 | 0 | 0 |

Strong assertions were run for every simulated seed. A failure would stop the script.

## Price Distribution For Recommended Seed

| split | mean | median | std | min | max | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 266.2800 | 100.7000 | 595.9300 | 5.9000 | 4641 | 53.4000 | 236 |
| validation | 294.0400 | 107.1000 | 633.9500 | 5.9000 | 4343.5000 | 56.1000 | 236 |
| test | 268.6400 | 94.7000 | 612.7900 | 5.9000 | 4641 | 55.3500 | 230.2200 |

## Distribution Comparison Tables

### brand

Full table: `results/candidate_component_split_balance/recommended_seed_brand_distribution.csv`

| brand | train_count | train_pct | validation_count | validation_pct | test_count | test_pct | total_count | max_pct_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toyota | 2761 | 34.8200 | 574 | 33.8600 | 559 | 32.9600 | 3894 | 1.8600 |
| vw | 2702 | 34.0700 | 524 | 30.9100 | 564 | 33.2500 | 3790 | 3.1600 |
| skoda | 2467 | 31.1100 | 597 | 35.2200 | 573 | 33.7900 | 3637 | 4.1100 |

### model

Full table: `results/candidate_component_split_balance/recommended_seed_model_distribution.csv`

| model | train_count | train_pct | validation_count | validation_pct | test_count | test_pct | total_count | max_pct_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| corolla | 2761 | 34.8200 | 574 | 33.8600 | 559 | 32.9600 | 3894 | 1.8600 |
| golf | 2702 | 34.0700 | 524 | 30.9100 | 564 | 33.2500 | 3790 | 3.1600 |
| octavia | 2467 | 31.1100 | 597 | 35.2200 | 573 | 33.7900 | 3637 | 4.1100 |

### category

Full table: `results/candidate_component_split_balance/recommended_seed_category_distribution.csv`

| category | train_count | train_pct | validation_count | validation_pct | test_count | test_pct | total_count | max_pct_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| electric / transmitter / databox / sensor | 1884 | 23.7600 | 439 | 25.9000 | 489 | 28.8300 | 2812 | 5.0700 |
| vehicle exterior / suspension | 1538 | 19.3900 | 97 | 5.7200 | 163 | 9.6100 | 1798 | 13.6700 |
| engine | 1138 | 14.3500 | 441 | 26.0200 | 201 | 11.8500 | 1780 | 14.1700 |
| gear box / drive axle / middle axle | 1032 | 13.0100 | 216 | 12.7400 | 182 | 10.7300 | 1430 | 2.2800 |
| fuel | 812 | 10.2400 | 238 | 14.0400 | 336 | 19.8100 | 1386 | 9.5700 |
| brakes | 877 | 11.0600 | 173 | 10.2100 | 203 | 11.9700 | 1253 | 1.7600 |
| airbag | 649 | 8.1800 | 91 | 5.3700 | 122 | 7.1900 | 862 | 2.8200 |

### subcategory

Full table: `results/candidate_component_split_balance/recommended_seed_subcategory_distribution.csv`

| subcategory | train_count | train_pct | validation_count | validation_pct | test_count | test_pct | total_count | max_pct_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 896 | 11.3000 | 85 | 5.0100 | 67 | 3.9500 | 1048 | 7.3500 |
| rear | 246 | 3.1000 | 33 | 1.9500 | 36 | 2.1200 | 315 | 1.1600 |
| right | 239 | 3.0100 | 15 | 0.8800 | 43 | 2.5400 | 297 | 2.1300 |
| left | 224 | 2.8200 | 3 | 0.1800 | 50 | 2.9500 | 277 | 2.7700 |
| right rear | 237 | 2.9900 | 0 | 0 | 18 | 1.0600 | 255 | 2.9900 |
| left rear | 192 | 2.4200 | 0 | 0 | 57 | 3.3600 | 249 | 3.3600 |
| left front | 164 | 2.0700 | 24 | 1.4200 | 12 | 0.7100 | 200 | 1.3600 |
| right front | 149 | 1.8800 | 18 | 1.0600 | 32 | 1.8900 | 199 | 0.8200 |
| either side | 108 | 1.3600 | 2 | 0.1200 | 18 | 1.0600 | 128 | 1.2400 |
| passenger airbag | 85 | 1.0700 | 16 | 0.9400 | 0 | 0 | 101 | 1.0700 |
| alternator | 46 | 0.5800 | 54 | 3.1900 | 0 | 0 | 100 | 3.1900 |
| automatic gear | 54 | 0.6800 | 0 | 0 | 46 | 2.7100 | 100 | 2.7100 |
| caliper bracket | 16 | 0.2000 | 46 | 2.7100 | 38 | 2.2400 | 100 | 2.5100 |
| knee | 58 | 0.7300 | 42 | 2.4800 | 0 | 0 | 100 | 2.4800 |
| abs hydraulic aggregate | 60 | 0.7600 | 0 | 0 | 40 | 2.3600 | 100 | 2.3600 |
| steering wheel airbag | 62 | 0.7800 | 2 | 0.1200 | 36 | 2.1200 | 100 | 2 |
| starter gasoline | 42 | 0.5300 | 40 | 2.3600 | 18 | 1.0600 | 100 | 1.8300 |
| tank lid | 43 | 0.5400 | 17 | 1 | 40 | 2.3600 | 100 | 1.8200 |
| motor bracket | 54 | 0.6800 | 38 | 2.2400 | 8 | 0.4700 | 100 | 1.7700 |
| starter diesel | 58 | 0.7300 | 36 | 2.1200 | 6 | 0.3500 | 100 | 1.7700 |

### year_start

Full table: `results/candidate_component_split_balance/recommended_seed_year_start_distribution.csv`

| year_start | train_count | train_pct | validation_count | validation_pct | test_count | test_pct | total_count | max_pct_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2013 | 2795 | 35.2500 | 585 | 34.5100 | 531 | 31.3100 | 3911 | 3.9400 |
| 2019 | 1174 | 14.8000 | 195 | 11.5000 | 193 | 11.3800 | 1562 | 3.4200 |
| 2005 | 742 | 9.3600 | 187 | 11.0300 | 109 | 6.4300 | 1038 | 4.6100 |
| 2002 | 768 | 9.6800 | 65 | 3.8300 | 156 | 9.2000 | 989 | 5.8500 |
| 1998 | 610 | 7.6900 | 127 | 7.4900 | 140 | 8.2500 | 877 | 0.7600 |
| 2004 | 472 | 5.9500 | 166 | 9.7900 | 102 | 6.0100 | 740 | 3.8400 |
| 2009 | 464 | 5.8500 | 84 | 4.9600 | 151 | 8.9000 | 699 | 3.9500 |
| 2008 | 328 | 4.1400 | 98 | 5.7800 | 59 | 3.4800 | 485 | 2.3000 |
| 2020 | 277 | 3.4900 | 66 | 3.8900 | 72 | 4.2500 | 415 | 0.7500 |
| 1993 | 124 | 1.5600 | 28 | 1.6500 | 49 | 2.8900 | 201 | 1.3300 |
| 1996 | 54 | 0.6800 | 30 | 1.7700 | 77 | 4.5400 | 161 | 3.8600 |
| 1988 | 85 | 1.0700 | 13 | 0.7700 | 4 | 0.2400 | 102 | 0.8400 |
| 1992 | 24 | 0.3000 | 24 | 1.4200 | 36 | 2.1200 | 84 | 1.8200 |
| 1983 | 1 | 0.0100 | 27 | 1.5900 | 5 | 0.2900 | 33 | 1.5800 |
| 1984 | 11 | 0.1400 | 0 | 0 | 12 | 0.7100 | 23 | 0.7100 |
| 1982 | 1 | 0.0100 | 0 | 0 | 0 | 0 | 1 | 0.0100 |

### year_end

Full table: `results/candidate_component_split_balance/recommended_seed_year_end_distribution.csv`

| year_end | train_count | train_pct | validation_count | validation_pct | test_count | test_pct | total_count | max_pct_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 2761 | 34.8200 | 530 | 31.2700 | 517 | 30.4800 | 3808 | 4.3300 |
| 2027 | 1451 | 18.3000 | 261 | 15.4000 | 265 | 15.6200 | 1977 | 2.9000 |
| 2013 | 1206 | 15.2100 | 271 | 15.9900 | 260 | 15.3300 | 1737 | 0.7800 |
| 2007 | 768 | 9.6800 | 65 | 3.8300 | 156 | 9.2000 | 989 | 5.8500 |
| 2009 | 472 | 5.9500 | 166 | 9.7900 | 102 | 6.0100 | 740 | 3.8400 |
| 2012 | 328 | 4.1400 | 98 | 5.7800 | 59 | 3.4800 | 485 | 2.3000 |
| 2006 | 364 | 4.5900 | 34 | 2.0100 | 61 | 3.6000 | 459 | 2.5800 |
| 2001 | 246 | 3.1000 | 93 | 5.4900 | 79 | 4.6600 | 418 | 2.3800 |
| 1997 | 124 | 1.5600 | 28 | 1.6500 | 49 | 2.8900 | 201 | 1.3300 |
| 2011 | 54 | 0.6800 | 30 | 1.7700 | 77 | 4.5400 | 161 | 3.8600 |
| 2018 | 34 | 0.4300 | 55 | 3.2400 | 14 | 0.8300 | 103 | 2.8200 |
| 1992 | 85 | 1.0700 | 13 | 0.7700 | 4 | 0.2400 | 102 | 0.8400 |
| 1999 | 24 | 0.3000 | 24 | 1.4200 | 36 | 2.1200 | 84 | 1.8200 |
| 1987 | 2 | 0.0300 | 27 | 1.5900 | 5 | 0.2900 | 34 | 1.5700 |
| 1991 | 11 | 0.1400 | 0 | 0 | 12 | 0.7100 | 23 | 0.7100 |

## Usability Notes

The proposed protocol appears usable for final model evaluation from a leakage-control and row-balance perspective. Price and subgroup distributions should still be reviewed before freezing the final split.

The connected-component design is stricter than product-id-only splitting because it keeps repeated listings and comparable part identities together. This should reduce optimistic identity leakage, but it also constrains split balance because large components cannot be divided across train, validation, and test.
