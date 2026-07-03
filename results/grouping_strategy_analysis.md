# Grouping Strategy Structural Analysis

Generated: 2026-06-26 16:32:24 UTC

Input dataset: `datasets/cleaned/clean_master_dataset.csv`
Rows analyzed: 11,321

This report compares candidate grouping keys for future train/test split design. It is a structural diagnostic only: no machine learning models are trained and no split files are created.

Canonicalization applied to grouping values:

- Unicode normalization: `NFKC`
- lowercase text
- trim leading and trailing whitespace
- collapse repeated internal whitespace
- replace missing or empty values with `__missing__`
- no fuzzy matching

## Missing Values In Grouping Columns

| column | missing_rows | missing_pct |
| --- | --- | --- |
| brand | 0 | 0.00% |
| model | 0 | 0.00% |
| oem_number | 0 | 0.00% |
| part_name | 0 | 0.00% |
| product_id | 0 | 0.00% |
| year_end | 0 | 0.00% |
| year_start | 0 | 0.00% |

## Strategy Comparison

| strategy | unique_groups | avg_size | median_size | max_size | min_size | singleton_groups | singleton_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| product_id | 2,619 | 4.32 | 6.00 | 6 | 1 | 834 | 31.84% |
| canonical(part_name, brand, model) | 593 | 19.09 | 18.00 | 87 | 1 | 96 | 16.19% |
| canonical(part_name, brand, model, oem_number) | 1,624 | 6.97 | 6.00 | 60 | 1 | 405 | 24.94% |
| canonical(part_name, brand, model, year_start, year_end) | 1,032 | 10.97 | 6.00 | 70 | 1 | 189 | 18.31% |

## product_id

Grouping columns: `product_id`

### Summary Statistics

| metric | value |
| --- | --- |
| unique groups | 2,619 |
| average group size | 4.32 |
| median group size | 6.00 |
| maximum group size | 6 |
| minimum group size | 1 |
| singleton groups | 834 |
| singleton group percentage | 31.84% |

### Group Size Histogram

The exact histogram is also saved at `results/grouping_strategy_analysis/product_id_histogram.csv`.

| group_size | group_count | row_count | group_pct |
| --- | --- | --- | --- |
| 1 | 834 | 834 | 31.84 |
| 2 | 10 | 20 | 0.38 |
| 3 | 30 | 90 | 1.15 |
| 4 | 15 | 60 | 0.57 |
| 5 | 63 | 315 | 2.41 |
| 6 | 1667 | 10002 | 63.65 |

### Top 20 Largest Groups

The same top-group table is saved at `results/grouping_strategy_analysis/product_id_top_20_groups.csv`.

| product_id | group_size |
| --- | --- |
| 53365106 | 6 |
| 53366348 | 6 |
| 53367213 | 6 |
| 53368149 | 6 |
| 53368241 | 6 |
| 53368418 | 6 |
| 53368462 | 6 |
| 53369770 | 6 |
| 53369814 | 6 |
| 53371576 | 6 |
| 53373961 | 6 |
| 53375244 | 6 |
| 53376928 | 6 |
| 53379281 | 6 |
| 53380327 | 6 |
| 53380920 | 6 |
| 53380963 | 6 |
| 53382600 | 6 |
| 53386676 | 6 |
| 53386727 | 6 |

## canonical(part_name, brand, model)

Grouping columns: `part_name, brand, model`

### Summary Statistics

| metric | value |
| --- | --- |
| unique groups | 593 |
| average group size | 19.09 |
| median group size | 18.00 |
| maximum group size | 87 |
| minimum group size | 1 |
| singleton groups | 96 |
| singleton group percentage | 16.19% |

### Group Size Histogram

The exact histogram is also saved at `results/grouping_strategy_analysis/part_name_brand_model_histogram.csv`.

| group_size | group_count | row_count | group_pct |
| --- | --- | --- | --- |
| 1 | 96 | 96 | 16.19 |
| 2 | 38 | 76 | 6.41 |
| 3 | 17 | 51 | 2.87 |
| 4 | 7 | 28 | 1.18 |
| 5 | 11 | 55 | 1.85 |
| 6 | 84 | 504 | 14.17 |
| 7 | 4 | 28 | 0.67 |
| 8 | 1 | 8 | 0.17 |
| 9 | 1 | 9 | 0.17 |
| 10 | 2 | 20 | 0.34 |
| 11 | 6 | 66 | 1.01 |
| 12 | 18 | 216 | 3.04 |
| 13 | 2 | 26 | 0.34 |
| 16 | 1 | 16 | 0.17 |
| 17 | 1 | 17 | 0.17 |
| 18 | 13 | 234 | 2.19 |
| 19 | 2 | 38 | 0.34 |
| 20 | 1 | 20 | 0.17 |
| 21 | 2 | 42 | 0.34 |
| 22 | 2 | 44 | 0.34 |
| 23 | 9 | 207 | 1.52 |
| 24 | 23 | 552 | 3.88 |
| 25 | 6 | 150 | 1.01 |
| 26 | 2 | 52 | 0.34 |
| 27 | 6 | 162 | 1.01 |
| 28 | 14 | 392 | 2.36 |
| 29 | 29 | 841 | 4.89 |
| 30 | 89 | 2670 | 15.01 |
| 31 | 1 | 31 | 0.17 |
| 32 | 2 | 64 | 0.34 |
| 33 | 4 | 132 | 0.67 |
| 34 | 1 | 34 | 0.17 |
| 35 | 1 | 35 | 0.17 |
| 36 | 8 | 288 | 1.35 |
| 37 | 4 | 148 | 0.67 |
| 38 | 10 | 380 | 1.69 |
| 39 | 15 | 585 | 2.53 |
| 40 | 14 | 560 | 2.36 |
| 41 | 2 | 82 | 0.34 |
| 42 | 9 | 378 | 1.52 |
| 44 | 1 | 44 | 0.17 |
| 45 | 1 | 45 | 0.17 |
| 46 | 2 | 92 | 0.34 |
| 47 | 2 | 94 | 0.34 |
| 48 | 4 | 192 | 0.67 |
| 50 | 2 | 100 | 0.34 |
| 51 | 1 | 51 | 0.17 |
| 52 | 1 | 52 | 0.17 |
| 53 | 2 | 106 | 0.34 |
| 54 | 2 | 108 | 0.34 |
| 55 | 3 | 165 | 0.51 |
| 57 | 1 | 57 | 0.17 |
| 59 | 2 | 118 | 0.34 |
| 60 | 2 | 120 | 0.34 |
| 61 | 1 | 61 | 0.17 |
| 63 | 1 | 63 | 0.17 |
| 65 | 1 | 65 | 0.17 |
| 66 | 1 | 66 | 0.17 |
| 70 | 1 | 70 | 0.17 |
| 73 | 1 | 73 | 0.17 |
| 77 | 1 | 77 | 0.17 |
| 78 | 1 | 78 | 0.17 |
| 87 | 1 | 87 | 0.17 |

### Top 20 Largest Groups

The same top-group table is saved at `results/grouping_strategy_analysis/part_name_brand_model_top_20_groups.csv`.

| part_name | brand | model | group_size |
| --- | --- | --- | --- |
| drive shaft -(left front) | toyota | corolla | 87 |
| suspension -(rear) | toyota | corolla | 78 |
| trailing link rear -(left) | toyota | corolla | 77 |
| shock absorbers rear -(rear) | toyota | corolla | 73 |
| trailing link rear -(right) | toyota | corolla | 70 |
| drive shaft -(right rear) | skoda | octavia | 66 |
| drive shaft -(right front) | toyota | corolla | 65 |
| curtain airbags -(right) | toyota | corolla | 63 |
| hub rear -(rear) | toyota | corolla | 61 |
| drive shaft - , e-(left front) | vw | golf | 60 |
| wheel bearing spindle shaft -(left rear) | toyota | corolla | 60 |
| strut rear -(left) | toyota | corolla | 59 |
| suspension - , e-(rear) | vw | golf | 59 |
| suspension -(rear) | skoda | octavia | 57 |
| curtain airbags -(left) | toyota | corolla | 55 |
| shock absorbers rear -(rear) | skoda | octavia | 55 |
| strut rear -(right) | toyota | corolla | 55 |
| curtain airbags -(right) | skoda | octavia | 54 |
| trailing link rear - , e-(left) | vw | golf | 54 |
| brake caliper -(right rear) | toyota | corolla | 53 |

## canonical(part_name, brand, model, oem_number)

Grouping columns: `part_name, brand, model, oem_number`

### Summary Statistics

| metric | value |
| --- | --- |
| unique groups | 1,624 |
| average group size | 6.97 |
| median group size | 6.00 |
| maximum group size | 60 |
| minimum group size | 1 |
| singleton groups | 405 |
| singleton group percentage | 24.94% |

### Group Size Histogram

The exact histogram is also saved at `results/grouping_strategy_analysis/part_name_brand_model_oem_number_histogram.csv`.

| group_size | group_count | row_count | group_pct |
| --- | --- | --- | --- |
| 1 | 405 | 405 | 24.94 |
| 2 | 69 | 138 | 4.25 |
| 3 | 38 | 114 | 2.34 |
| 4 | 21 | 84 | 1.29 |
| 5 | 96 | 480 | 5.91 |
| 6 | 558 | 3348 | 34.36 |
| 7 | 38 | 266 | 2.34 |
| 8 | 12 | 96 | 0.74 |
| 9 | 9 | 81 | 0.55 |
| 10 | 17 | 170 | 1.05 |
| 11 | 30 | 330 | 1.85 |
| 12 | 138 | 1656 | 8.50 |
| 13 | 12 | 156 | 0.74 |
| 14 | 13 | 182 | 0.80 |
| 15 | 8 | 120 | 0.49 |
| 16 | 4 | 64 | 0.25 |
| 17 | 17 | 289 | 1.05 |
| 18 | 49 | 882 | 3.02 |
| 19 | 6 | 114 | 0.37 |
| 20 | 7 | 140 | 0.43 |
| 21 | 5 | 105 | 0.31 |
| 22 | 1 | 22 | 0.06 |
| 23 | 9 | 207 | 0.55 |
| 24 | 20 | 480 | 1.23 |
| 26 | 3 | 78 | 0.18 |
| 27 | 1 | 27 | 0.06 |
| 28 | 2 | 56 | 0.12 |
| 29 | 5 | 145 | 0.31 |
| 30 | 12 | 360 | 0.74 |
| 32 | 4 | 128 | 0.25 |
| 34 | 1 | 34 | 0.06 |
| 35 | 2 | 70 | 0.12 |
| 36 | 3 | 108 | 0.18 |
| 37 | 1 | 37 | 0.06 |
| 38 | 2 | 76 | 0.12 |
| 39 | 1 | 39 | 0.06 |
| 40 | 1 | 40 | 0.06 |
| 42 | 2 | 84 | 0.12 |
| 50 | 1 | 50 | 0.06 |
| 60 | 1 | 60 | 0.06 |

### Top 20 Largest Groups

The same top-group table is saved at `results/grouping_strategy_analysis/part_name_brand_model_oem_number_top_20_groups.csv`.

| part_name | brand | model | oem_number | group_size |
| --- | --- | --- | --- | --- |
| drive shaft -(left front) | toyota | corolla | fi27837687a | 60 |
| hub rear -(rear) | toyota | corolla | fi15710056a | 50 |
| brake caliper - , e-(left front) | vw | golf | fi27837687a | 42 |
| drive shaft - , e-(left front) | vw | golf | fi09389104a | 42 |
| distributors vacuum regulator - | toyota | corolla | fi15710056a | 40 |
| curtain airbags -(right) | toyota | corolla | fi27837687a | 39 |
| curtain airbags -(left) | toyota | corolla | fi27837687a | 38 |
| fuel pump electric - | toyota | corolla | fi15710056a | 38 |
| sensor abs - | toyota | corolla | fi15710056a | 37 |
| brake caliper - , e-(right front) | vw | golf | fi27837687a | 36 |
| curtain airbags - , e-(left) | vw | golf | fi27837687a | 36 |
| hub rear - , e-(rear) | vw | golf | fi10331575a | 36 |
| trailing link rear -(left) | toyota | corolla | fi05028803a | 35 |
| wheel bearing spindle shaft -(left rear) | skoda | octavia | fi05351686a | 35 |
| brake caliper -(right front) | toyota | corolla | fi27837687a | 34 |
| air-flow sensor - | toyota | corolla | fi06376738a | 32 |
| brake caliper -(left front) | toyota | corolla | fi27837687a | 32 |
| parkeringshjälp frontsensor - | toyota | corolla | fi27837687a | 32 |
| shock absorbers rear -(rear) | toyota | corolla | fi06376738a | 32 |
| airbag front sensor -(left) | toyota | corolla | fi15710056a | 30 |

## canonical(part_name, brand, model, year_start, year_end)

Grouping columns: `part_name, brand, model, year_start, year_end`

### Summary Statistics

| metric | value |
| --- | --- |
| unique groups | 1,032 |
| average group size | 10.97 |
| median group size | 6.00 |
| maximum group size | 70 |
| minimum group size | 1 |
| singleton groups | 189 |
| singleton group percentage | 18.31% |

### Group Size Histogram

The exact histogram is also saved at `results/grouping_strategy_analysis/part_name_brand_model_year_start_year_end_histogram.csv`.

| group_size | group_count | row_count | group_pct |
| --- | --- | --- | --- |
| 1 | 189 | 189 | 18.31 |
| 2 | 48 | 96 | 4.65 |
| 3 | 22 | 66 | 2.13 |
| 4 | 15 | 60 | 1.45 |
| 5 | 45 | 225 | 4.36 |
| 6 | 248 | 1488 | 24.03 |
| 7 | 21 | 147 | 2.03 |
| 8 | 10 | 80 | 0.97 |
| 9 | 6 | 54 | 0.58 |
| 10 | 13 | 130 | 1.26 |
| 11 | 25 | 275 | 2.42 |
| 12 | 100 | 1200 | 9.69 |
| 13 | 11 | 143 | 1.07 |
| 14 | 9 | 126 | 0.87 |
| 15 | 6 | 90 | 0.58 |
| 16 | 7 | 112 | 0.68 |
| 17 | 18 | 306 | 1.74 |
| 18 | 63 | 1134 | 6.10 |
| 19 | 6 | 114 | 0.58 |
| 20 | 1 | 20 | 0.10 |
| 21 | 8 | 168 | 0.78 |
| 22 | 5 | 110 | 0.48 |
| 23 | 14 | 322 | 1.36 |
| 24 | 27 | 648 | 2.62 |
| 25 | 4 | 100 | 0.39 |
| 26 | 4 | 104 | 0.39 |
| 27 | 3 | 81 | 0.29 |
| 28 | 6 | 168 | 0.58 |
| 29 | 17 | 493 | 1.65 |
| 30 | 30 | 900 | 2.91 |
| 31 | 2 | 62 | 0.19 |
| 32 | 5 | 160 | 0.48 |
| 33 | 1 | 33 | 0.10 |
| 35 | 3 | 105 | 0.29 |
| 36 | 9 | 324 | 0.87 |
| 38 | 2 | 76 | 0.19 |
| 39 | 4 | 156 | 0.39 |
| 40 | 3 | 120 | 0.29 |
| 41 | 2 | 82 | 0.19 |
| 42 | 2 | 84 | 0.19 |
| 43 | 1 | 43 | 0.10 |
| 44 | 2 | 88 | 0.19 |
| 46 | 1 | 46 | 0.10 |
| 47 | 1 | 47 | 0.10 |
| 48 | 1 | 48 | 0.10 |
| 49 | 1 | 49 | 0.10 |
| 50 | 1 | 50 | 0.10 |
| 52 | 1 | 52 | 0.10 |
| 53 | 1 | 53 | 0.10 |
| 54 | 2 | 108 | 0.19 |
| 59 | 1 | 59 | 0.10 |
| 60 | 1 | 60 | 0.10 |
| 62 | 1 | 62 | 0.10 |
| 66 | 1 | 66 | 0.10 |
| 69 | 1 | 69 | 0.10 |
| 70 | 1 | 70 | 0.10 |

### Top 20 Largest Groups

The same top-group table is saved at `results/grouping_strategy_analysis/part_name_brand_model_year_start_year_end_top_20_groups.csv`.

| part_name | brand | model | year_start | year_end | group_size |
| --- | --- | --- | --- | --- | --- |
| drive shaft -(left front) | toyota | corolla | 2019 | 2027 | 70 |
| trailing link rear -(left) | toyota | corolla | 2019 | 2027 | 69 |
| drive shaft -(right rear) | skoda | octavia | 2013 | 2020 | 66 |
| trailing link rear -(right) | toyota | corolla | 2019 | 2027 | 62 |
| wheel bearing spindle shaft -(left rear) | toyota | corolla | 2019 | 2027 | 60 |
| suspension -(rear) | toyota | corolla | 2019 | 2027 | 59 |
| curtain airbags -(right) | skoda | octavia | 2013 | 2020 | 54 |
| drive shaft - , e-(left front) | vw | golf | 2013 | 2020 | 54 |
| brake caliper -(right rear) | toyota | corolla | 2019 | 2027 | 53 |
| wheel bearing spindle shaft -(right rear) | toyota | corolla | 2019 | 2027 | 52 |
| brake caliper -(left rear) | toyota | corolla | 2019 | 2027 | 50 |
| shock absorbers rear -(rear) | skoda | octavia | 2013 | 2020 | 49 |
| drive shaft -(left rear) | skoda | octavia | 2013 | 2020 | 48 |
| brake caliper -(right front) | toyota | corolla | 2019 | 2027 | 47 |
| wheel bearing spindle shaft -(left rear) | skoda | octavia | 2013 | 2020 | 46 |
| brake caliper -(left front) | toyota | corolla | 2019 | 2027 | 44 |
| hub rear -(rear) | toyota | corolla | 2002 | 2007 | 44 |
| curtain airbags -(right) | toyota | corolla | 2019 | 2027 | 43 |
| curtain airbags - , e-(left) | vw | golf | 2013 | 2020 | 42 |
| curtain airbags - , e-(right) | vw | golf | 2013 | 2020 | 42 |
