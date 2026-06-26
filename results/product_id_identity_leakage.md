# Product-ID-Only Identity Leakage Estimate

Generated: 2026-06-26 16:40:03 UTC

Input dataset: `datasets/cleaned/clean_master_dataset.csv`
Rows analyzed: 11,321

This report estimates potential optimistic leakage when train/test splitting is performed only by `product_id`. A product-id-only split prevents the same listing ID from crossing splits, but it can still allow comparable part identities to appear in both train and test when the same identity is represented by multiple product IDs.

Canonicalization matches the earlier grouping reports: Unicode normalization, lowercase, trimmed whitespace, collapsed internal whitespace, missing-value sentinel, and no fuzzy matching.

## Summary

| strategy | total_identities | multi_product_id_identities | product_ids_in_multi_identities | rows_in_multi_identities | largest_product_ids_per_identity | median_product_ids_per_repeated_identity |
| --- | --- | --- | --- | --- | --- | --- |
| canonical(part_name, brand, model) | 593 | 404 (68.13%) | 2,505 (95.65%) | 10,689 (94.42%) | 31 | 5.0 |
| canonical(part_name, brand, model, oem_number) | 1,624 | 546 (33.62%) | 1,696 (64.76%) | 7,067 (62.42%) | 18 | 2.0 |
| canonical(part_name, brand, model, year_start, year_end) | 1,032 | 552 (53.49%) | 2,249 (85.87%) | 9,447 (83.45%) | 26 | 3.0 |

## Interpretation

- `multi_product_id_identities` are identities that could cross train/test boundaries under product-id-only splitting.
- `product_ids_in_multi_identities` estimates how much of the product-id population is exposed to this risk.
- More restrictive identity keys reduce the measured leakage surface, but may also hide leakage if the added field is noisy or fragments true comparable identities.

## canonical(part_name, brand, model)

Full repeated-identity table: `results/product_id_identity_leakage/part_name_brand_model_repeated_identities.csv`
Product-id-count distribution: `results/product_id_identity_leakage/part_name_brand_model_product_id_distribution.csv`
Largest shared identities CSV: `results/product_id_identity_leakage/part_name_brand_model_largest_shared_identities.csv`

### Distribution Of Repeated Identities

| product_ids_per_identity | identity_count | identity_pct | product_id_count | row_count |
| --- | --- | --- | --- | --- |
| 2 | 70 | 17.33 | 140 | 398 |
| 3 | 29 | 7.18 | 87 | 321 |
| 4 | 42 | 10.40 | 168 | 818 |
| 5 | 132 | 32.67 | 660 | 3768 |
| 6 | 20 | 4.95 | 120 | 572 |
| 7 | 20 | 4.95 | 140 | 711 |
| 8 | 12 | 2.97 | 96 | 510 |
| 9 | 7 | 1.73 | 63 | 339 |
| 10 | 5 | 1.24 | 50 | 255 |
| 11 | 3 | 0.74 | 33 | 137 |
| 12 | 4 | 0.99 | 48 | 152 |
| 13 | 5 | 1.24 | 65 | 183 |
| 14 | 16 | 3.96 | 224 | 607 |
| 15 | 19 | 4.70 | 285 | 765 |
| 16 | 5 | 1.24 | 80 | 203 |
| 17 | 2 | 0.50 | 34 | 106 |
| 20 | 2 | 0.50 | 40 | 103 |
| 21 | 1 | 0.25 | 21 | 55 |
| 22 | 1 | 0.25 | 22 | 60 |
| 23 | 2 | 0.50 | 46 | 115 |
| 25 | 2 | 0.50 | 50 | 157 |
| 26 | 1 | 0.25 | 26 | 61 |
| 29 | 2 | 0.50 | 58 | 150 |
| 30 | 1 | 0.25 | 30 | 78 |
| 31 | 1 | 0.25 | 31 | 65 |

### Largest Shared Identities Top 20

| part_name | brand | model | product_id_count | row_count | rows_per_product_id | product_id_sample | median_price | min_price | max_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drive shaft -(right front) | toyota | corolla | 31 | 65 | 2.10 | 53410756, 53712062, 53713676, 53714980, 53716391, 53716866, 53718651, 54151722, 54413270, 54521427 ... (+21 more) | 296.40 | 153.90 | 357 |
| suspension -(rear) | toyota | corolla | 30 | 78 | 2.60 | 53375391, 53480837, 53577432, 53817908, 53827878, 53865601, 54491760, 54491761, 55960953, 55960954 ... (+20 more) | 47.20 | 35.50 | 47.60 |
| trailing link rear -(left) | toyota | corolla | 29 | 77 | 2.66 | 53722743, 53722744, 53848080, 53848801, 53848805, 53848806, 53869465, 53883746, 53883759, 54363441 ... (+19 more) | 71 | 56.20 | 72.60 |
| shock absorbers rear -(rear) | toyota | corolla | 29 | 73 | 2.52 | 53706183, 53721247, 53819760, 53824358, 53853721, 53853777, 53859811, 53859812, 53881584, 54248598 ... (+19 more) | 60.40 | 59.20 | 62.20 |
| hub rear -(rear) | toyota | corolla | 26 | 61 | 2.35 | 53699458, 53708861, 53710888, 53715555, 53716493, 53717736, 54555112, 54560998, 54562514, 54568829 ... (+16 more) | 94.40 | 47.40 | 119 |
| drive shaft -(left front) | toyota | corolla | 25 | 87 | 3.48 | 53367963, 53374161, 54232635, 54417013, 54521794, 54626514, 56269960, 59131986, 60059054, 60132522 ... (+15 more) | 294.90 | 153.90 | 297.50 |
| trailing link rear -(right) | toyota | corolla | 25 | 70 | 2.80 | 53707144, 53725967, 53848794, 53848803, 53848811, 53859508, 53860530, 53883758, 54360056, 55987886 ... (+15 more) | 71 | 56.20 | 72.60 |
| curtain airbags -(right) | toyota | corolla | 23 | 63 | 2.74 | 53663768, 53991822, 54082891, 54331761, 54470946, 54479779, 54492177, 59391659, 61845408, 62404856 ... (+13 more) | 235.90 | 59.20 | 261.80 |
| wheel bearing spindle shaft -(right rear) | toyota | corolla | 23 | 52 | 2.26 | 53853364, 53855121, 53855165, 56933447, 57222704, 58825525, 59131976, 59241761, 60059059, 60139249 ... (+13 more) | 189.75 | 165.80 | 208.30 |
| wheel bearing spindle shaft -(left rear) | toyota | corolla | 22 | 60 | 2.73 | 53853342, 53869656, 56933448, 58825527, 59131977, 59242297, 60059060, 60139248, 60419506, 60496437 ... (+12 more) | 177.90 | 165.80 | 208.30 |
| curtain airbags -(left) | toyota | corolla | 21 | 55 | 2.62 | 53449758, 53659100, 53672406, 54470943, 54476116, 54492143, 59391660, 59624287, 61845319, 62465163 ... (+11 more) | 226.10 | 59.20 | 261.80 |
| brake caliper -(right rear) | toyota | corolla | 20 | 53 | 2.65 | 53608052, 53850683, 53859861, 54238812, 56933443, 57220992, 58755576, 59131987, 59249691, 62404920 ... (+10 more) | 177.90 | 168.70 | 238 |
| brake caliper -(left rear) | toyota | corolla | 20 | 50 | 2.50 | 53608961, 53859868, 53860486, 53867130, 54232531, 56933444, 57220993, 58755577, 59131988, 59249690 ... (+10 more) | 177.70 | 165.80 | 238 |
| strut rear -(left) | toyota | corolla | 17 | 59 | 3.47 | 53589483, 53618513, 53650880, 53664653, 53679064, 53682952, 53689539, 54008914, 54034252, 54321498 ... (+7 more) | 53.30 | 23.70 | 59.50 |
| brake caliper -(right front) | toyota | corolla | 17 | 47 | 2.76 | 53859828, 53860450, 53860490, 53895810, 53928356, 57220994, 58717890, 60058882, 61076237, 62404916 ... (+7 more) | 177.60 | 142.10 | 238 |
| brake caliper -(left front) | toyota | corolla | 16 | 44 | 2.75 | 53606571, 53859829, 53928357, 56933442, 57220995, 60058883, 60496428, 63310200, 63350262, 64546414 ... (+6 more) | 177.60 | 118.40 | 238 |
| distributors - | toyota | corolla | 16 | 40 | 2.50 | 53687628, 53832764, 53832935, 53914784, 53954187, 54310467, 54335221, 54341466, 54345881, 54475593 ... (+6 more) | 83.70 | 59.20 | 95.20 |
| gear box 5 speed - | toyota | corolla | 16 | 40 | 2.50 | 53400892, 53413369, 53889068, 54560231, 54777180, 56989378, 59855589, 60930698, 61139026, 63216307 ... (+6 more) | 651.60 | 517.40 | 833 |
| starter gasoline - | toyota | corolla | 16 | 40 | 2.50 | 53397731, 53705104, 53725225, 53727493, 53821469, 53913410, 54528712, 54648612, 56268870, 58355946 ... (+6 more) | 94.70 | 82.90 | 106.70 |
| passenger airbag - | toyota | corolla | 16 | 39 | 2.44 | 53402114, 53455139, 53471242, 53713033, 59140468, 60058722, 61076256, 63158119, 63310195, 63350253 ... (+6 more) | 296 | 281.20 | 333.20 |

## canonical(part_name, brand, model, oem_number)

Full repeated-identity table: `results/product_id_identity_leakage/part_name_brand_model_oem_number_repeated_identities.csv`
Product-id-count distribution: `results/product_id_identity_leakage/part_name_brand_model_oem_number_product_id_distribution.csv`
Largest shared identities CSV: `results/product_id_identity_leakage/part_name_brand_model_oem_number_largest_shared_identities.csv`

### Distribution Of Repeated Identities

| product_ids_per_identity | identity_count | identity_pct | product_id_count | row_count |
| --- | --- | --- | --- | --- |
| 2 | 287 | 52.56 | 574 | 2491 |
| 3 | 120 | 21.98 | 360 | 1634 |
| 4 | 61 | 11.17 | 244 | 1078 |
| 5 | 34 | 6.23 | 170 | 678 |
| 6 | 12 | 2.20 | 72 | 286 |
| 7 | 9 | 1.65 | 63 | 208 |
| 8 | 3 | 0.55 | 24 | 52 |
| 9 | 7 | 1.28 | 63 | 216 |
| 10 | 3 | 0.55 | 30 | 82 |
| 11 | 4 | 0.73 | 44 | 91 |
| 12 | 2 | 0.37 | 24 | 63 |
| 13 | 1 | 0.18 | 13 | 60 |
| 14 | 1 | 0.18 | 14 | 38 |
| 15 | 1 | 0.18 | 15 | 40 |
| 18 | 1 | 0.18 | 18 | 50 |

### Largest Shared Identities Top 20

| part_name | brand | model | oem_number | product_id_count | row_count | rows_per_product_id | product_id_sample | median_price | min_price | max_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hub rear -(rear) | toyota | corolla | fi15710056a | 18 | 50 | 2.78 | 54555112, 54560998, 54562514, 54568829, 54568941, 54574650, 54576727, 54581766, 54581788, 54584616 ... (+8 more) | 94.55 | 47.40 | 95.20 |
| distributors vacuum regulator - | toyota | corolla | fi15710056a | 15 | 40 | 2.67 | 54570238, 54570295, 54570367, 54570499, 54572869, 54573063, 54573661, 54573914, 54574618, 54575400 ... (+5 more) | 100.60 | 100.20 | 101.10 |
| fuel pump electric - | toyota | corolla | fi15710056a | 14 | 38 | 2.71 | 54549667, 54549674, 54553339, 54557769, 54558231, 54558291, 54569407, 54572196, 54574086, 54575016 ... (+4 more) | 100.80 | 100.20 | 107.10 |
| drive shaft -(left front) | toyota | corolla | fi27837687a | 13 | 60 | 4.62 | 53367963, 53374161, 59131986, 60059054, 60496432, 63310210, 63350268, 64546407, 65057801, 65399928 ... (+3 more) | 296 | 165.80 | 297.50 |
| sensor abs - | toyota | corolla | fi15710056a | 12 | 37 | 3.08 | 54552575, 54558254, 54558373, 54559216, 54560130, 54561122, 54568794, 54572494, 54573990, 54576365 ... (+2 more) | 47.40 | 41.40 | 59.50 |
| passenger airbag - | toyota | corolla | fi27837687a | 12 | 26 | 2.17 | 53402114, 53455139, 53471242, 59140468, 60058722, 61076256, 63310195, 63350253, 64466975, 64546429 ... (+2 more) | 296 | 284.20 | 297.50 |
| trailing link rear -(left) | toyota | corolla | fi05028803a | 11 | 35 | 3.18 | 53848080, 53848801, 53848805, 53848806, 53869465, 53883746, 53883759, 58755574, 58825996, 58830597 ... (+1 more) | 71 | 59.20 | 71.40 |
| wheel bearing spindle shaft -(left rear) | toyota | corolla | fi27837687a | 11 | 21 | 1.91 | 59131977, 60059060, 60496437, 61076252, 62404844, 63310215, 63350279, 64546424, 65107270, 65680827 ... (+1 more) | 177.60 | 176.90 | 178.50 |
| engine gasoline - | toyota | corolla | fi27837687a | 11 | 20 | 1.82 | 59078050, 59140483, 62989746, 63305356, 63352462, 64538310, 65107242, 65176844, 65433218, 66412411 ... (+1 more) | 3670.40 | 3433.60 | 3689 |
| wheel bearing spindle shaft -(right rear) | toyota | corolla | fi27837687a | 11 | 15 | 1.36 | 59131976, 60059059, 60496436, 61076249, 62404848, 63310214, 63350278, 64546423, 65107272, 65680826 ... (+1 more) | 177.60 | 176.90 | 178.50 |
| trailing link rear -(right) | toyota | corolla | fi05028803a | 10 | 30 | 3 | 53848794, 53848803, 53848811, 53859508, 53860530, 53883758, 58755575, 58825995, 58830598, 58830599 | 71 | 59.20 | 71.40 |
| suspension -(rear) | toyota | corolla | fi27837687a | 10 | 28 | 2.80 | 53375391, 53480837, 60498741, 60498767, 61890414, 61890420, 62122633, 62122634, 65399910, 65399986 | 47.40 | 35.50 | 47.60 |
| abs hydraulic aggregate - | toyota | corolla | fi27837687a | 10 | 24 | 2.40 | 59140465, 60058714, 60496419, 61076241, 62404925, 63310160, 63350249, 65250408, 65680819, 66223587 | 589.90 | 532.80 | 595 |
| curtain airbags -(right) | toyota | corolla | fi27837687a | 9 | 39 | 4.33 | 62404856, 63310181, 63350254, 64546891, 64976693, 65257587, 65399988, 65434861, 65680833 | 238 | 94.70 | 261.80 |
| curtain airbags -(left) | toyota | corolla | fi27837687a | 9 | 38 | 4.22 | 53449758, 63310187, 63350255, 64546896, 64976694, 65257588, 65400008, 65434862, 65680834 | 236.80 | 94.70 | 261.80 |
| brake caliper -(right front) | toyota | corolla | fi27837687a | 9 | 34 | 3.78 | 60058882, 61076237, 62404916, 63310199, 63350261, 64546413, 65107262, 65250413, 65399920 | 177.60 | 165.80 | 178.50 |
| brake caliper -(left front) | toyota | corolla | fi27837687a | 9 | 32 | 3.56 | 60058883, 60496428, 63310200, 63350262, 64546414, 65107263, 65250414, 65399917, 65594835 | 177.60 | 118.40 | 178.50 |
| drive shaft -(right front) | toyota | corolla | fi27837687a | 9 | 26 | 2.89 | 53410756, 60059053, 60410436, 63350267, 64546406, 65057800, 65399932, 65865835, 66434290 | 353.90 | 177.60 | 357 |
| steering wheel airbag - | toyota | corolla | fi27837687a | 9 | 24 | 2.67 | 53447491, 53487808, 59140469, 60058723, 63350256, 64546410, 64976681, 65257590, 65680798 | 296 | 260.50 | 297.50 |
| motor cushion - | toyota | corolla | fi06292622a | 9 | 23 | 2.56 | 54084378, 54084790, 54085064, 54098489, 54099109, 54100965, 54105228, 54121433, 54142195 | 23.70 | 5.90 | 23.80 |

## canonical(part_name, brand, model, year_start, year_end)

Full repeated-identity table: `results/product_id_identity_leakage/part_name_brand_model_year_start_year_end_repeated_identities.csv`
Product-id-count distribution: `results/product_id_identity_leakage/part_name_brand_model_year_start_year_end_product_id_distribution.csv`
Largest shared identities CSV: `results/product_id_identity_leakage/part_name_brand_model_year_start_year_end_largest_shared_identities.csv`

### Distribution Of Repeated Identities

| product_ids_per_identity | identity_count | identity_pct | product_id_count | row_count |
| --- | --- | --- | --- | --- |
| 2 | 201 | 36.41 | 402 | 1744 |
| 3 | 119 | 21.56 | 357 | 1737 |
| 4 | 76 | 13.77 | 304 | 1349 |
| 5 | 52 | 9.42 | 260 | 1335 |
| 6 | 28 | 5.07 | 168 | 711 |
| 7 | 14 | 2.54 | 98 | 419 |
| 8 | 12 | 2.17 | 96 | 300 |
| 9 | 14 | 2.54 | 126 | 415 |
| 10 | 3 | 0.54 | 30 | 75 |
| 11 | 8 | 1.45 | 88 | 295 |
| 12 | 5 | 0.91 | 60 | 144 |
| 13 | 5 | 0.91 | 65 | 207 |
| 14 | 1 | 0.18 | 14 | 23 |
| 15 | 5 | 0.91 | 75 | 197 |
| 16 | 1 | 0.18 | 16 | 44 |
| 17 | 2 | 0.36 | 34 | 106 |
| 20 | 2 | 0.36 | 40 | 103 |
| 22 | 2 | 0.36 | 44 | 122 |
| 23 | 1 | 0.18 | 23 | 52 |
| 26 | 1 | 0.18 | 26 | 69 |

### Largest Shared Identities Top 20

| part_name | brand | model | year_start | year_end | product_id_count | row_count | rows_per_product_id | product_id_sample | median_price | min_price | max_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trailing link rear -(left) | toyota | corolla | 2019 | 2027 | 26 | 69 | 2.65 | 53848080, 53848801, 53848805, 53848806, 53869465, 53883746, 53883759, 55987887, 55987889, 55987891 ... (+16 more) | 71 | 56.20 | 71.40 |
| wheel bearing spindle shaft -(right rear) | toyota | corolla | 2019 | 2027 | 23 | 52 | 2.26 | 53853364, 53855121, 53855165, 56933447, 57222704, 58825525, 59131976, 59241761, 60059059, 60139249 ... (+13 more) | 189.75 | 165.80 | 208.30 |
| trailing link rear -(right) | toyota | corolla | 2019 | 2027 | 22 | 62 | 2.82 | 53848794, 53848803, 53848811, 53859508, 53860530, 53883758, 55987886, 55987888, 55987890, 55987892 ... (+12 more) | 71 | 56.20 | 71.40 |
| wheel bearing spindle shaft -(left rear) | toyota | corolla | 2019 | 2027 | 22 | 60 | 2.73 | 53853342, 53869656, 56933448, 58825527, 59131977, 59242297, 60059060, 60139248, 60419506, 60496437 ... (+12 more) | 177.90 | 165.80 | 208.30 |
| brake caliper -(right rear) | toyota | corolla | 2019 | 2027 | 20 | 53 | 2.65 | 53608052, 53850683, 53859861, 54238812, 56933443, 57220992, 58755576, 59131987, 59249691, 62404920 ... (+10 more) | 177.90 | 168.70 | 238 |
| brake caliper -(left rear) | toyota | corolla | 2019 | 2027 | 20 | 50 | 2.50 | 53608961, 53859868, 53860486, 53867130, 54232531, 56933444, 57220993, 58755577, 59131988, 59249690 ... (+10 more) | 177.70 | 165.80 | 238 |
| suspension -(rear) | toyota | corolla | 2019 | 2027 | 17 | 59 | 3.47 | 53577432, 53865601, 58825993, 58825994, 59230183, 59230184, 60129271, 60498741, 60498767, 61749419 ... (+7 more) | 47.20 | 35.50 | 47.60 |
| brake caliper -(right front) | toyota | corolla | 2019 | 2027 | 17 | 47 | 2.76 | 53859828, 53860450, 53860490, 53895810, 53928356, 57220994, 58717890, 60058882, 61076237, 62404916 ... (+7 more) | 177.60 | 142.10 | 238 |
| brake caliper -(left front) | toyota | corolla | 2019 | 2027 | 16 | 44 | 2.75 | 53606571, 53859829, 53928357, 56933442, 57220995, 60058883, 60496428, 63310200, 63350262, 64546414 ... (+6 more) | 177.60 | 118.40 | 238 |
| hybrid batteri - | toyota | corolla | 2019 | 2027 | 15 | 40 | 2.67 | 54137921, 57028353, 57887357, 58926177, 58991908, 59217624, 60117429, 60785799, 61156018, 62529942 ... (+5 more) | 1300.10 | 1124.80 | 1462.20 |
| hybrid inverter - | toyota | corolla | 2019 | 2027 | 15 | 40 | 2.67 | 54137927, 54232530, 54248125, 57231214, 58198976, 59274610, 60063118, 61074258, 61156102, 61905395 ... (+5 more) | 1085.45 | 355.20 | 1190 |
| other control unit - | toyota | corolla | 2019 | 2027 | 15 | 40 | 2.67 | 53608931, 53608953, 53861120, 54103835, 54137955, 59244561, 60119199, 60125605, 60500078, 63020235 ... (+5 more) | 438.75 | 270 | 476 |
| engine gasoline - | toyota | corolla | 2019 | 2027 | 15 | 39 | 2.60 | 53605194, 53848511, 56939596, 59078050, 59140483, 62989746, 63117883, 63305356, 63352462, 64538310 ... (+5 more) | 3656.10 | 3433.60 | 3689 |
| fuel filling pipe / tube - | toyota | corolla | 1998 | 2001 | 15 | 38 | 2.53 | 53821389, 53826004, 54517984, 54527967, 54527972, 54527974, 54555307, 54555440, 54560132, 54571032 ... (+5 more) | 59 | 35.50 | 59.50 |
| shock absorbers rear -(rear) | toyota | corolla | 2019 | 2027 | 14 | 23 | 1.64 | 53853721, 53853777, 53859811, 53859812, 53881584, 54248598, 58825523, 58825992, 60410274, 61070966 ... (+4 more) | 59.20 | 59.20 | 62.20 |
| drive shaft -(left front) | toyota | corolla | 2019 | 2027 | 13 | 70 | 5.38 | 59131986, 60059054, 60496432, 61770180, 63310210, 63350268, 64489889, 64546407, 65399928, 65680788 ... (+3 more) | 294.90 | 235.90 | 297.50 |
| kamera utvändig - | toyota | corolla | 2019 | 2027 | 13 | 38 | 2.92 | 53838464, 53858477, 53881624, 53928354, 54129716, 58761312, 58925051, 59862593, 60492667, 61918245 ... (+3 more) | 653.45 | 296 | 714 |
| abs hydraulic aggregate - | toyota | corolla | 2019 | 2027 | 13 | 36 | 2.77 | 57231205, 59140465, 60058714, 60496419, 61076241, 62404925, 63026248, 63310160, 63350249, 65250408 ... (+3 more) | 590.95 | 532.80 | 595 |
| automatic gear - | toyota | corolla | 2019 | 2027 | 13 | 36 | 2.77 | 54137995, 57035004, 59078044, 59131981, 59269879, 61063287, 62117685, 62404956, 63117910, 63127471 ... (+3 more) | 2960 | 2604.80 | 3054.50 |
| gear lever - | toyota | corolla | 2019 | 2027 | 13 | 27 | 2.08 | 53606551, 53886312, 54145488, 58718927, 60061114, 60467442, 61092453, 62404893, 63017018, 63125341 ... (+3 more) | 177.60 | 171.70 | 189.40 |
