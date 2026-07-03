# Grouping Strategy Fragmentation Analysis

Generated: 2026-06-26 16:37:18 UTC

Input dataset: `datasets/cleaned/clean_master_dataset.csv`
Rows analyzed: 11,321

This report compares whether candidate grouping strategies merge observations or fragment existing groups. It is a structural diagnostic only: no machine learning models are trained and no split files are created.

Definitions used in the tables:

- `previous_groups_split_or_smaller`: previous groups whose rows are distributed across more than one current group.
- `resulting_smaller_fragments`: previous/current overlap fragments that are smaller than their previous group.
- `new_singleton_groups`: current singleton groups created from a previous group that had more than one row.
- `large_previous_groups_split_n_ge_20`: previous groups with at least 20 rows that split.
- `current_groups_merging_previous_groups`: current groups containing rows from more than one previous group.

Important interpretation note: `product_id` and canonical identity groups are not nested. The product-id comparison is therefore a crosswalk diagnostic. OEM and compatibility-year effects are interpreted using isolated comparisons against `canonical(part_name, brand, model)`.

## Main Findings

- Adding OEM increases group count from 593 to 1,624, splits 348 base groups, and creates 309 new singleton groups.
- Adding compatibility years increases group count from 593 to 1,032, splits 270 base groups, and creates 93 new singleton groups.
- In this dataset, OEM fragments the broad part identity more aggressively than compatibility years. That does not automatically make OEM invalid, but it raises a stronger risk that OEM is acting as a noisy fragmentation key rather than a consistently meaningful identity boundary.

## Ordered Candidate Transitions

| comparison | interpretation | previous_groups | current_groups | previous_groups_split_or_smaller | resulting_smaller_fragments | new_singleton_groups | large_previous_groups_split_n_ge_20 | current_groups_merging_previous_groups | rows_in_split_previous_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| product_id -> canonical(part_name, brand, model) | Crosswalk between repeated listing IDs and broad part identity. This is not a nested refinement. | 2619 | 593 | 187 | 374 | 73 | 0 | 404 | 1114 |
| canonical(part_name, brand, model) -> + oem_number | Nested refinement that isolates the effect of adding OEM. | 593 | 1624 | 348 | 1379 | 309 | 270 | 0 | 9772 |
| + oem_number -> + compatibility years | Ordered candidate-to-candidate comparison. This both removes OEM and adds compatibility years, so it should not be interpreted as the isolated effect of years. | 1624 | 1032 | 179 | 386 | 26 | 38 | 405 | 2624 |

## Isolated Field-Addition Effects

| comparison | interpretation | previous_groups | current_groups | previous_groups_split_or_smaller | resulting_smaller_fragments | new_singleton_groups | large_previous_groups_split_n_ge_20 | current_groups_merging_previous_groups | rows_in_split_previous_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical(part_name, brand, model) -> + oem_number | Isolated OEM effect. | 593 | 1624 | 348 | 1379 | 309 | 270 | 0 | 9772 |
| canonical(part_name, brand, model) -> + compatibility years | Isolated compatibility-year effect. | 593 | 1032 | 270 | 709 | 93 | 220 | 0 | 7911 |

## Largest Base Groups Split By OEM

Full CSV: `results/grouping_strategy_fragmentation/base_to_oem_number_split_groups.csv`

| part_name | brand | model | previous_size | refined_group_count | largest_refined_group | smallest_refined_group | singleton_refined_groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drive shaft -(left front) | toyota | corolla | 87 | 12 | 60 | 1 | 8 |
| suspension -(rear) | toyota | corolla | 78 | 10 | 28 | 1 | 3 |
| trailing link rear -(left) | toyota | corolla | 77 | 8 | 35 | 1 | 1 |
| shock absorbers rear -(rear) | toyota | corolla | 73 | 8 | 32 | 2 | 0 |
| trailing link rear -(right) | toyota | corolla | 70 | 7 | 30 | 2 | 0 |
| drive shaft -(right rear) | skoda | octavia | 66 | 5 | 24 | 6 | 0 |
| drive shaft -(right front) | toyota | corolla | 65 | 16 | 26 | 1 | 9 |
| curtain airbags -(right) | toyota | corolla | 63 | 7 | 39 | 1 | 3 |
| hub rear -(rear) | toyota | corolla | 61 | 4 | 50 | 1 | 1 |
| wheel bearing spindle shaft -(left rear) | toyota | corolla | 60 | 7 | 21 | 1 | 2 |
| drive shaft - , e-(left front) | vw | golf | 60 | 3 | 42 | 6 | 0 |
| strut rear -(left) | toyota | corolla | 59 | 9 | 24 | 1 | 3 |
| suspension - , e-(rear) | vw | golf | 59 | 5 | 17 | 6 | 0 |
| suspension -(rear) | skoda | octavia | 57 | 5 | 22 | 6 | 0 |
| strut rear -(right) | toyota | corolla | 55 | 8 | 12 | 1 | 1 |
| curtain airbags -(left) | toyota | corolla | 55 | 5 | 38 | 2 | 0 |
| shock absorbers rear -(rear) | skoda | octavia | 55 | 5 | 15 | 6 | 0 |
| trailing link rear - , e-(left) | vw | golf | 54 | 6 | 12 | 6 | 0 |
| curtain airbags -(right) | skoda | octavia | 54 | 5 | 24 | 6 | 0 |
| brake caliper -(right rear) | toyota | corolla | 53 | 7 | 19 | 1 | 1 |

## Largest Base Groups Split By Compatibility Years

Full CSV: `results/grouping_strategy_fragmentation/base_to_compatibility_years_split_groups.csv`

| part_name | brand | model | previous_size | refined_group_count | largest_refined_group | smallest_refined_group | singleton_refined_groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drive shaft -(left front) | toyota | corolla | 87 | 4 | 70 | 4 | 0 |
| suspension -(rear) | toyota | corolla | 78 | 4 | 59 | 2 | 0 |
| trailing link rear -(left) | toyota | corolla | 77 | 3 | 69 | 2 | 0 |
| shock absorbers rear -(rear) | toyota | corolla | 73 | 4 | 28 | 6 | 0 |
| trailing link rear -(right) | toyota | corolla | 70 | 3 | 62 | 2 | 0 |
| drive shaft -(right front) | toyota | corolla | 65 | 4 | 41 | 6 | 0 |
| curtain airbags -(right) | toyota | corolla | 63 | 4 | 43 | 1 | 1 |
| hub rear -(rear) | toyota | corolla | 61 | 4 | 44 | 1 | 1 |
| drive shaft - , e-(left front) | vw | golf | 60 | 2 | 54 | 6 | 0 |
| suspension - , e-(rear) | vw | golf | 59 | 5 | 24 | 5 | 0 |
| strut rear -(left) | toyota | corolla | 59 | 4 | 39 | 2 | 0 |
| suspension -(rear) | skoda | octavia | 57 | 3 | 39 | 6 | 0 |
| strut rear -(right) | toyota | corolla | 55 | 5 | 19 | 1 | 1 |
| curtain airbags -(left) | toyota | corolla | 55 | 4 | 41 | 1 | 1 |
| shock absorbers rear -(rear) | skoda | octavia | 55 | 2 | 49 | 6 | 0 |
| trailing link rear - , e-(left) | vw | golf | 54 | 3 | 24 | 12 | 0 |
| shock absorbers rear - , e-(rear) | vw | golf | 53 | 4 | 36 | 5 | 0 |
| trailing link rear -(left) | skoda | octavia | 51 | 2 | 36 | 15 | 0 |
| drive shaft - , e-(right rear) | vw | golf | 50 | 3 | 29 | 3 | 0 |
| drive shaft - , e-(left rear) | vw | golf | 48 | 3 | 32 | 5 | 0 |

## Representative Rows Where Adding OEM Changes Grouping

Full CSV: `results/grouping_strategy_fragmentation/oem_fragmentation_examples.csv`

| row_index | product_id | part_name | brand | model | oem_number | year_start | year_end | price | previous_group_size | refined_group_size | refined_groups_within_previous | singleton_created |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4155 | 65399928 | Drive shaft -(Left front) | toyota | corolla | FI27837687A | 2019 | 2027 | 296 | 87 | 60 | 12 | False |
| 4158 | 64489889 | Drive shaft -(Left front) | toyota | corolla | FI09389104A | 2019 | 2027 | 236.80 | 87 | 7 | 12 | False |
| 4852 | 63710432 | Drive shaft -(Left front) | toyota | corolla | FI30987993A | 2002 | 2007 | 236.80 | 87 | 1 | 12 | True |
| 4263 | 60498741 | Suspension -(Rear) | toyota | corolla | FI27837687A | 2019 | 2027 | 47.40 | 78 | 28 | 10 | False |
| 4260 | 53865601 | Suspension -(Rear) | toyota | corolla | FI05028803A | 2019 | 2027 | 47.40 | 78 | 20 | 10 | False |
| 5038 | 53577432 | Suspension -(Rear) | toyota | corolla | FI02042722A | 2019 | 2027 | 41.40 | 78 | 1 | 10 | True |
| 4286 | 58830600 | Trailing link rear -(Left) | toyota | corolla | FI05028803A | 2019 | 2027 | 71 | 77 | 35 | 8 | False |
| 4289 | 55987887 | Trailing link rear -(Left) | toyota | corolla | FI02042722A | 2019 | 2027 | 71 | 77 | 14 | 8 | False |
| 5078 | 63127200 | Trailing link rear -(Left) | toyota | corolla | FI24637030A | 2019 | 2027 | 71 | 77 | 1 | 8 | True |
| 4215 | 54389246 | Shock absorbers rear -(Rear) | toyota | corolla | FI06376738A | 2002 | 2007 | 60.40 | 73 | 32 | 8 | False |
| 4212 | 53819760 | Shock absorbers rear -(Rear) | toyota | corolla | FI01853355A | 1998 | 2001 | 60.40 | 73 | 12 | 8 | False |
| 4961 | 54248598 | Shock absorbers rear -(Rear) | toyota | corolla | FI09389104A | 2019 | 2027 | 59.20 | 73 | 2 | 8 | False |
| 4296 | 53848794 | Trailing link rear -(Right) | toyota | corolla | FI05028803A | 2019 | 2027 | 71 | 70 | 30 | 7 | False |
| 4292 | 55987886 | Trailing link rear -(Right) | toyota | corolla | FI02042722A | 2019 | 2027 | 71 | 70 | 14 | 7 | False |
| 5087 | 53707144 | Trailing link rear -(Right) | toyota | corolla | FI02154548A | 1993 | 1997 | 59.20 | 70 | 2 | 7 | False |
| 8143 | 62656879 | Drive shaft -(Right rear) | skoda | octavia | FI23403240A | 2013 | 2020 | 225 | 66 | 24 | 5 | False |
| 8141 | 64531098 | Drive shaft -(Right rear) | skoda | octavia | FI09389104A | 2013 | 2020 | 219 | 66 | 18 | 5 | False |
| 8145 | 54159171 | Drive shaft -(Right rear) | skoda | octavia | FI06292622A | 2013 | 2020 | 201.30 | 66 | 6 | 5 | False |
| 4160 | 64546406 | Drive shaft -(Right front) | toyota | corolla | FI27837687A | 2019 | 2027 | 355.20 | 65 | 26 | 16 | False |
| 4874 | 53712062 | Drive shaft -(Right front) | toyota | corolla | FI02154548A | 2002 | 2007 | 177.60 | 65 | 7 | 16 | False |

## Representative Rows Where Adding Compatibility Years Changes Grouping

Full CSV: `results/grouping_strategy_fragmentation/compatibility_year_fragmentation_examples.csv`

| row_index | product_id | part_name | brand | model | year_start | year_end | oem_number | price | previous_group_size | refined_group_size | refined_groups_within_previous | singleton_created |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4155 | 65399928 | Drive shaft -(Left front) | toyota | corolla | 2019 | 2027 | FI27837687A | 296 | 87 | 70 | 4 | False |
| 4159 | 64578967 | Drive shaft -(Left front) | toyota | corolla | 2002 | 2007 | FI07265116A | 213.10 | 87 | 8 | 4 | False |
| 4855 | 54232635 | Drive shaft -(Left front) | toyota | corolla | 2013 | 2018 | FI09389104A | 219 | 87 | 4 | 4 | False |
| 4260 | 53865601 | Suspension -(Rear) | toyota | corolla | 2019 | 2027 | FI05028803A | 47.40 | 78 | 59 | 4 | False |
| 4269 | 53817908 | Suspension -(Rear) | toyota | corolla | 2002 | 2007 | FI01853355A | 47.40 | 78 | 9 | 4 | False |
| 5045 | 61890414 | Suspension -(Rear) | toyota | corolla | 2013 | 2018 | FI27837687A | 35.50 | 78 | 2 | 4 | False |
| 4286 | 58830600 | Trailing link rear -(Left) | toyota | corolla | 2019 | 2027 | FI05028803A | 71 | 77 | 69 | 3 | False |
| 4302 | 54363441 | Trailing link rear -(Left) | toyota | corolla | 1998 | 2001 | FI06376738A | 72.20 | 77 | 6 | 3 | False |
| 5081 | 53722743 | Trailing link rear -(Left) | toyota | corolla | 1993 | 1997 | FI02154548A | 59.20 | 77 | 2 | 3 | False |
| 4215 | 54389246 | Shock absorbers rear -(Rear) | toyota | corolla | 2002 | 2007 | FI06376738A | 60.40 | 73 | 28 | 4 | False |
| 4210 | 61749414 | Shock absorbers rear -(Rear) | toyota | corolla | 2019 | 2027 | FI06509801A | 61.80 | 73 | 23 | 4 | False |
| 4216 | 54389283 | Shock absorbers rear -(Rear) | toyota | corolla | 1993 | 1997 | FI06376738A | 60.40 | 73 | 6 | 4 | False |
| 4291 | 56933451 | Trailing link rear -(Right) | toyota | corolla | 2019 | 2027 | FI09389104A | 71 | 70 | 62 | 3 | False |
| 4301 | 54360056 | Trailing link rear -(Right) | toyota | corolla | 1998 | 2001 | FI06376738A | 72.20 | 70 | 6 | 3 | False |
| 5087 | 53707144 | Trailing link rear -(Right) | toyota | corolla | 1993 | 1997 | FI02154548A | 59.20 | 70 | 2 | 3 | False |
| 4160 | 64546406 | Drive shaft -(Right front) | toyota | corolla | 2019 | 2027 | FI27837687A | 355.20 | 65 | 41 | 4 | False |
| 4164 | 54521427 | Drive shaft -(Right front) | toyota | corolla | 2008 | 2012 | FI11042417A | 296 | 65 | 10 | 4 | False |
| 4869 | 61808613 | Drive shaft -(Right front) | toyota | corolla | 2013 | 2018 | FI05351686A | 236.80 | 65 | 6 | 4 | False |
| 3800 | 65257587 | Curtain airbags -(Right) | toyota | corolla | 2019 | 2027 | FI27837687A | 236.80 | 63 | 43 | 4 | False |
| 3804 | 54331761 | Curtain airbags -(Right) | toyota | corolla | 2002 | 2007 | FI06376738A | 181.20 | 63 | 10 | 4 | False |

## Thesis Interpretation

OEM and compatibility years both split broad part-identity groups, but the split statistics should be interpreted as evidence about grouping behavior rather than as proof that either field is a valid identity boundary. A field that creates many smaller or singleton groups may either capture meaningful compatibility distinctions or fragment otherwise comparable parts.
