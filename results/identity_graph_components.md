# Identity Graph Connected Component Analysis

Generated: 2026-06-26 16:43:25 UTC

Input dataset: `datasets/cleaned/clean_master_dataset.csv`
Rows analyzed: 11,321

Each row is treated as a node. Edges are created when two rows share the same canonical `product_id` or the same candidate identity. This report analyzes graph structure only; it does not create train/test splits.

A component can contain more than one identity label when product_id links bridge small canonical identity differences. This is useful for leakage analysis because the connected component is the unit that would need to stay together to block both repeated-listing and identity-sharing paths.

Canonicalization matches the earlier grouping reports: Unicode normalization, lowercase, trimmed whitespace, collapsed internal whitespace, missing-value sentinel, and no fuzzy matching.

## Summary

| graph | connected_components | average_component_size | median_component_size | maximum_component_size | minimum_component_size | singleton_components |
| --- | --- | --- | --- | --- | --- | --- |
| product_id + canonical(part_name, brand, model) | 467 | 24.24 | 30.00 | 87 | 1 | 23 (4.93%) |
| product_id + canonical(part_name, brand, model, year_start, year_end) | 886 | 12.78 | 8.00 | 71 | 1 | 82 (9.26%) |

## product_id + canonical(part_name, brand, model)

Row-level component export: `results/identity_graph_components/product_id_part_name_brand_model_row_components.csv`
Component summary CSV: `results/identity_graph_components/product_id_part_name_brand_model_component_summary.csv`
Component-size histogram CSV: `results/identity_graph_components/product_id_part_name_brand_model_component_size_histogram.csv`

### Component Size Histogram

| component_size | component_count | row_count | component_pct |
| --- | --- | --- | --- |
| 1 | 23 | 23 | 4.93 |
| 2 | 6 | 12 | 1.28 |
| 3 | 4 | 12 | 0.86 |
| 4 | 3 | 12 | 0.64 |
| 5 | 2 | 10 | 0.43 |
| 6 | 89 | 534 | 19.06 |
| 7 | 2 | 14 | 0.43 |
| 8 | 3 | 24 | 0.64 |
| 9 | 1 | 9 | 0.21 |
| 10 | 1 | 10 | 0.21 |
| 12 | 25 | 300 | 5.35 |
| 13 | 2 | 26 | 0.43 |
| 18 | 14 | 252 | 3 |
| 19 | 3 | 57 | 0.64 |
| 20 | 1 | 20 | 0.21 |
| 22 | 1 | 22 | 0.21 |
| 23 | 1 | 23 | 0.21 |
| 24 | 33 | 792 | 7.07 |
| 25 | 6 | 150 | 1.28 |
| 26 | 1 | 26 | 0.21 |
| 27 | 3 | 81 | 0.64 |
| 29 | 5 | 145 | 1.07 |
| 30 | 131 | 3930 | 28.05 |
| 31 | 1 | 31 | 0.21 |
| 32 | 2 | 64 | 0.43 |
| 33 | 3 | 99 | 0.64 |
| 34 | 2 | 68 | 0.43 |
| 35 | 1 | 35 | 0.21 |
| 36 | 6 | 216 | 1.28 |
| 37 | 3 | 111 | 0.64 |
| 38 | 3 | 114 | 0.64 |
| 39 | 8 | 312 | 1.71 |
| 40 | 29 | 1160 | 6.21 |
| 41 | 1 | 41 | 0.21 |
| 42 | 12 | 504 | 2.57 |
| 43 | 1 | 43 | 0.21 |
| 47 | 1 | 47 | 0.21 |
| 48 | 8 | 384 | 1.71 |
| 50 | 1 | 50 | 0.21 |
| 51 | 1 | 51 | 0.21 |
| 53 | 1 | 53 | 0.21 |
| 54 | 5 | 270 | 1.07 |
| 55 | 2 | 110 | 0.43 |
| 56 | 1 | 56 | 0.21 |
| 60 | 4 | 240 | 0.86 |
| 62 | 2 | 124 | 0.43 |
| 63 | 1 | 63 | 0.21 |
| 64 | 1 | 64 | 0.21 |
| 66 | 1 | 66 | 0.21 |
| 70 | 1 | 70 | 0.21 |
| 71 | 1 | 71 | 0.21 |
| 74 | 1 | 74 | 0.21 |
| 79 | 1 | 79 | 0.21 |
| 80 | 1 | 80 | 0.21 |
| 87 | 1 | 87 | 0.21 |

### Top 20 Largest Connected Components

| component_id | component_size | product_id_count | identity_count | product_id_sample | identity_sample | min_row_index |
| --- | --- | --- | --- | --- | --- | --- |
| product_id_part_name_brand_model_component_000001 | 87 | 25 | 1 | 53367963, 53374161, 54232635, 54417013, 54521794, 54626514, 56269960, 59131986, 60059054, 60132522, 60224934, 60496432 ... (+13 more) | drive shaft -(left front) \| toyota \| corolla | 4155 |
| product_id_part_name_brand_model_component_000002 | 80 | 30 | 2 | 53375391, 53480837, 53577432, 53817908, 53827878, 53865601, 54491760, 54491761, 55960953, 55960954, 56269534, 56269536 ... (+18 more) | suspension -(rear) \| toyota \| corolla \|\| suspension-(rear) \| toyota \| corolla | 4260 |
| product_id_part_name_brand_model_component_000003 | 79 | 29 | 2 | 53722743, 53722744, 53848080, 53848801, 53848805, 53848806, 53869465, 53883746, 53883759, 54363441, 55987887, 55987889 ... (+17 more) | trailing link rear -(left) \| toyota \| corolla \|\| trailing link rear-(left) \| toyota \| corolla | 4286 |
| product_id_part_name_brand_model_component_000004 | 74 | 29 | 2 | 53706183, 53721247, 53819760, 53824358, 53853721, 53853777, 53859811, 53859812, 53881584, 54248598, 54389246, 54389283 ... (+17 more) | shock absorbers rear -(rear) \| toyota \| corolla \|\| shock absorbers rear-(rear) \| toyota \| corolla | 4210 |
| product_id_part_name_brand_model_component_000005 | 71 | 33 | 2 | 53410756, 53712062, 53713676, 53714980, 53716391, 53716866, 53718651, 54151722, 54413270, 54521427, 54564547, 54626477 ... (+21 more) | drive shaft -(right front) \| toyota \| corolla \|\| drive shaft-(right front) \| toyota \| corolla | 4160 |
| product_id_part_name_brand_model_component_000006 | 70 | 25 | 1 | 53707144, 53725967, 53848794, 53848803, 53848811, 53859508, 53860530, 53883758, 54360056, 55987886, 55987888, 55987890 ... (+13 more) | trailing link rear -(right) \| toyota \| corolla | 4291 |
| product_id_part_name_brand_model_component_000007 | 66 | 11 | 1 | 53375244, 53503260, 53589567, 54159171, 54272539, 54284856, 58539226, 61748921, 62656879, 63591065, 64531098 | drive shaft -(right rear) \| skoda \| octavia | 8141 |
| product_id_part_name_brand_model_component_000008 | 64 | 26 | 2 | 53699458, 53708861, 53710888, 53715555, 53716493, 53717736, 54555112, 54560998, 54562514, 54568829, 54568941, 54574650 ... (+14 more) | hub rear -(rear) \| toyota \| corolla \|\| hub rear-(rear) \| toyota \| corolla | 4270 |
| product_id_part_name_brand_model_component_000009 | 63 | 23 | 1 | 53663768, 53991822, 54082891, 54331761, 54470946, 54479779, 54492177, 59391659, 61845408, 62404856, 62465161, 63216298 ... (+11 more) | curtain airbags -(right) \| toyota \| corolla | 3800 |
| product_id_part_name_brand_model_component_000011 | 62 | 22 | 2 | 53853342, 53869656, 56933448, 58825527, 59131977, 59242297, 60059060, 60139248, 60419506, 60496437, 61076252, 61751482 ... (+10 more) | wheel bearing spindle shaft -(left rear) \| toyota \| corolla \|\| wheel bearing spindle shaft-(left rear) \| toyota \| corolla | 4240 |
| product_id_part_name_brand_model_component_000010 | 62 | 17 | 2 | 53589483, 53618513, 53650880, 53664653, 53679064, 53682952, 53689539, 54008914, 54034252, 54321498, 54479232, 54554195 ... (+5 more) | strut rear -(left) \| toyota \| corolla \|\| strut rear-(left) \| toyota \| corolla | 4225 |
| product_id_part_name_brand_model_component_000013 | 60 | 10 | 2 | 53449037, 53710187, 54245118, 55068239, 55068240, 57147922, 61159009, 61159010, 64947839, 65057619 | suspension - , e-(rear) \| vw \| golf \|\| suspension- , e-(rear) \| vw \| golf | 574 |
| product_id_part_name_brand_model_component_000014 | 60 | 10 | 2 | 53369770, 53409730, 53441870, 54680056, 59351317, 59351332, 59673189, 59673190, 60226034, 60226036 | shock absorbers rear -(rear) \| skoda \| octavia \|\| shock absorbers rear-(rear) \| skoda \| octavia | 8235 |
| product_id_part_name_brand_model_component_000015 | 60 | 10 | 2 | 54242387, 54280417, 54430941, 54443628, 54443630, 54447761, 56665933, 56665934, 56911282, 63591631 | suspension -(rear) \| skoda \| octavia \|\| suspension-(rear) \| skoda \| octavia | 8245 |
| product_id_part_name_brand_model_component_000012 | 60 | 10 | 1 | 53938689, 53938690, 54223177, 54244809, 54245168, 54247541, 54260459, 54263144, 64251591, 64496424 | drive shaft - , e-(left front) \| vw \| golf | 465 |
| product_id_part_name_brand_model_component_000016 | 56 | 21 | 2 | 53449758, 53659100, 53672406, 54470943, 54476116, 54492143, 59391660, 59624287, 61845319, 62465163, 63310187, 63350255 ... (+9 more) | curtain airbags -(left) \| toyota \| corolla \|\| curtain airbags-(left) \| toyota \| corolla | 3795 |
| product_id_part_name_brand_model_component_000017 | 55 | 20 | 2 | 53608052, 53850683, 53859861, 54238812, 56933443, 57220992, 58755576, 59131987, 59249691, 62404920, 62418504, 63127091 ... (+8 more) | brake caliper -(right rear) \| toyota \| corolla \|\| brake caliper-(right rear) \| toyota \| corolla | 3845 |
| product_id_part_name_brand_model_component_000018 | 55 | 15 | 1 | 53467029, 53662663, 53674446, 53721223, 53792102, 54007262, 54338198, 54471239, 54484620, 54485592, 54486882, 54490443 ... (+3 more) | strut rear -(right) \| toyota \| corolla | 4220 |
| product_id_part_name_brand_model_component_000019 | 54 | 9 | 2 | 53862511, 54069563, 54253079, 54268110, 54291801, 54362669, 54416445, 60073919, 64667066 | drive shaft - , e-(left rear) \| vw \| golf \|\| drive shaft- , e-(left rear) \| vw \| golf | 470 |
| product_id_part_name_brand_model_component_000020 | 54 | 9 | 2 | 53713162, 53719558, 53721299, 53725158, 53824510, 53828430, 53923221, 54389700, 56566915 | shock absorbers rear - , e-(rear) \| vw \| golf \|\| shock absorbers rear- , e-(rear) \| vw \| golf | 564 |

## product_id + canonical(part_name, brand, model, year_start, year_end)

Row-level component export: `results/identity_graph_components/product_id_part_name_brand_model_year_start_year_end_row_components.csv`
Component summary CSV: `results/identity_graph_components/product_id_part_name_brand_model_year_start_year_end_component_summary.csv`
Component-size histogram CSV: `results/identity_graph_components/product_id_part_name_brand_model_year_start_year_end_component_size_histogram.csv`

### Component Size Histogram

| component_size | component_count | row_count | component_pct |
| --- | --- | --- | --- |
| 1 | 82 | 82 | 9.26 |
| 2 | 23 | 46 | 2.60 |
| 3 | 13 | 39 | 1.47 |
| 4 | 11 | 44 | 1.24 |
| 5 | 14 | 70 | 1.58 |
| 6 | 277 | 1662 | 31.26 |
| 7 | 22 | 154 | 2.48 |
| 8 | 9 | 72 | 1.02 |
| 9 | 6 | 54 | 0.68 |
| 10 | 7 | 70 | 0.79 |
| 11 | 7 | 77 | 0.79 |
| 12 | 122 | 1464 | 13.77 |
| 13 | 14 | 182 | 1.58 |
| 14 | 9 | 126 | 1.02 |
| 15 | 2 | 30 | 0.23 |
| 16 | 6 | 96 | 0.68 |
| 17 | 6 | 102 | 0.68 |
| 18 | 78 | 1404 | 8.80 |
| 19 | 7 | 133 | 0.79 |
| 20 | 2 | 40 | 0.23 |
| 21 | 5 | 105 | 0.56 |
| 22 | 2 | 44 | 0.23 |
| 23 | 4 | 92 | 0.45 |
| 24 | 43 | 1032 | 4.85 |
| 25 | 4 | 100 | 0.45 |
| 26 | 3 | 78 | 0.34 |
| 27 | 3 | 81 | 0.34 |
| 28 | 3 | 84 | 0.34 |
| 29 | 6 | 174 | 0.68 |
| 30 | 41 | 1230 | 4.63 |
| 31 | 5 | 155 | 0.56 |
| 32 | 2 | 64 | 0.23 |
| 33 | 2 | 66 | 0.23 |
| 34 | 2 | 68 | 0.23 |
| 36 | 11 | 396 | 1.24 |
| 38 | 1 | 38 | 0.11 |
| 39 | 1 | 39 | 0.11 |
| 40 | 6 | 240 | 0.68 |
| 41 | 1 | 41 | 0.11 |
| 42 | 5 | 210 | 0.56 |
| 43 | 1 | 43 | 0.11 |
| 44 | 1 | 44 | 0.11 |
| 47 | 2 | 94 | 0.23 |
| 48 | 3 | 144 | 0.34 |
| 50 | 1 | 50 | 0.11 |
| 53 | 1 | 53 | 0.11 |
| 54 | 3 | 162 | 0.34 |
| 55 | 1 | 55 | 0.11 |
| 61 | 1 | 61 | 0.11 |
| 62 | 2 | 124 | 0.23 |
| 66 | 1 | 66 | 0.11 |
| 70 | 1 | 70 | 0.11 |
| 71 | 1 | 71 | 0.11 |

### Top 20 Largest Connected Components

| component_id | component_size | product_id_count | identity_count | product_id_sample | identity_sample | min_row_index |
| --- | --- | --- | --- | --- | --- | --- |
| product_id_part_name_brand_model_year_start_year_end_component_000001 | 71 | 26 | 2 | 53848080, 53848801, 53848805, 53848806, 53869465, 53883746, 53883759, 55987887, 55987889, 55987891, 55987893, 56933452 ... (+14 more) | trailing link rear -(left) \| toyota \| corolla \| 2019 \| 2027 \|\| trailing link rear-(left) \| toyota \| corolla \| 2019 \| 2027 | 4286 |
| product_id_part_name_brand_model_year_start_year_end_component_000002 | 70 | 13 | 1 | 59131986, 60059054, 60496432, 61770180, 63310210, 63350268, 64489889, 64546407, 65399928, 65680788, 66223598, 66434289 ... (+1 more) | drive shaft -(left front) \| toyota \| corolla \| 2019 \| 2027 | 4155 |
| product_id_part_name_brand_model_year_start_year_end_component_000003 | 66 | 11 | 1 | 53375244, 53503260, 53589567, 54159171, 54272539, 54284856, 58539226, 61748921, 62656879, 63591065, 64531098 | drive shaft -(right rear) \| skoda \| octavia \| 2013 \| 2020 | 8141 |
| product_id_part_name_brand_model_year_start_year_end_component_000004 | 62 | 22 | 2 | 53853342, 53869656, 56933448, 58825527, 59131977, 59242297, 60059060, 60139248, 60419506, 60496437, 61076252, 61751482 ... (+10 more) | wheel bearing spindle shaft -(left rear) \| toyota \| corolla \| 2019 \| 2027 \|\| wheel bearing spindle shaft-(left rear) \| toyota \| corolla \| 2019 \| 2027 | 4240 |
| product_id_part_name_brand_model_year_start_year_end_component_000005 | 62 | 22 | 1 | 53848794, 53848803, 53848811, 53859508, 53860530, 53883758, 55987886, 55987888, 55987890, 55987892, 56933451, 58755575 ... (+10 more) | trailing link rear -(right) \| toyota \| corolla \| 2019 \| 2027 | 4291 |
| product_id_part_name_brand_model_year_start_year_end_component_000006 | 61 | 17 | 2 | 53577432, 53865601, 58825993, 58825994, 59230183, 59230184, 60129271, 60498741, 60498767, 61749419, 61749421, 62122633 ... (+5 more) | suspension -(rear) \| toyota \| corolla \| 2019 \| 2027 \|\| suspension-(rear) \| toyota \| corolla \| 2019 \| 2027 | 4260 |
| product_id_part_name_brand_model_year_start_year_end_component_000007 | 55 | 20 | 2 | 53608052, 53850683, 53859861, 54238812, 56933443, 57220992, 58755576, 59131987, 59249691, 62404920, 62418504, 63127091 ... (+8 more) | brake caliper -(right rear) \| toyota \| corolla \| 2019 \| 2027 \|\| brake caliper-(right rear) \| toyota \| corolla \| 2019 \| 2027 | 3845 |
| product_id_part_name_brand_model_year_start_year_end_component_000010 | 54 | 9 | 2 | 53369770, 53409730, 53441870, 59351317, 59351332, 59673189, 59673190, 60226034, 60226036 | shock absorbers rear -(rear) \| skoda \| octavia \| 2013 \| 2020 \|\| shock absorbers rear-(rear) \| skoda \| octavia \| 2013 \| 2020 | 8235 |
| product_id_part_name_brand_model_year_start_year_end_component_000008 | 54 | 9 | 1 | 53938689, 53938690, 54223177, 54244809, 54245168, 54247541, 54260459, 54263144, 64496424 | drive shaft - , e-(left front) \| vw \| golf \| 2013 \| 2020 | 465 |
| product_id_part_name_brand_model_year_start_year_end_component_000009 | 54 | 9 | 1 | 53451718, 53485361, 53604578, 54252481, 54252635, 59583068, 60484610, 60973822, 63439416 | curtain airbags -(right) \| skoda \| octavia \| 2013 \| 2020 | 7709 |
| product_id_part_name_brand_model_year_start_year_end_component_000011 | 53 | 23 | 2 | 53853364, 53855121, 53855165, 56933447, 57222704, 58825525, 59131976, 59241761, 60059059, 60139249, 60496436, 61076249 ... (+11 more) | wheel bearing spindle shaft -(right rear) \| toyota \| corolla \| 2019 \| 2027 \|\| wheel bearing spindle shaft-(right rear) \| toyota \| corolla \| 2019 \| 2027 | 4245 |
| product_id_part_name_brand_model_year_start_year_end_component_000012 | 50 | 20 | 1 | 53608961, 53859868, 53860486, 53867130, 54232531, 56933444, 57220993, 58755577, 59131988, 59249690, 60058881, 62404922 ... (+8 more) | brake caliper -(left rear) \| toyota \| corolla \| 2019 \| 2027 | 3840 |
| product_id_part_name_brand_model_year_start_year_end_component_000013 | 48 | 18 | 2 | 53606571, 53859829, 53860488, 53879067, 53928357, 56933442, 57220995, 60058883, 60496428, 63310200, 63350262, 64546414 ... (+6 more) | brake caliper -(left front) \| toyota \| corolla \| 2019 \| 2027 \|\| brake caliper-(left front) \| toyota \| corolla \| 2019 \| 2027 | 3830 |
| product_id_part_name_brand_model_year_start_year_end_component_000015 | 48 | 8 | 2 | 53397794, 53409888, 56848565, 62238856, 62662904, 62781323, 65037610, 65651373 | wheel bearing spindle shaft -(left rear) \| skoda \| octavia \| 2013 \| 2020 \|\| wheel bearing spindle shaft-(left rear) \| skoda \| octavia \| 2013 \| 2020 | 8276 |
| product_id_part_name_brand_model_year_start_year_end_component_000014 | 48 | 8 | 1 | 53418003, 53503261, 53584385, 54107088, 54676853, 58539225, 62656774, 66395063 | drive shaft -(left rear) \| skoda \| octavia \| 2013 \| 2020 | 8137 |
| product_id_part_name_brand_model_year_start_year_end_component_000016 | 47 | 17 | 1 | 53859828, 53860450, 53860490, 53895810, 53928356, 57220994, 58717890, 60058882, 61076237, 62404916, 63310199, 63350261 ... (+5 more) | brake caliper -(right front) \| toyota \| corolla \| 2019 \| 2027 | 3835 |
| product_id_part_name_brand_model_year_start_year_end_component_000017 | 47 | 9 | 2 | 53699458, 56097887, 57180324, 57182287, 57894044, 57894045, 58671369, 58671370, 66735158 | hub rear -(rear) \| toyota \| corolla \| 2002 \| 2007 \|\| hub rear-(rear) \| toyota \| corolla \| 2002 \| 2007 | 4274 |
| product_id_part_name_brand_model_year_start_year_end_component_000018 | 44 | 11 | 2 | 54151722, 60059053, 63083525, 63350267, 64489888, 64546406, 65399932, 65865835, 66223983, 66434290, 66630808 | drive shaft -(right front) \| toyota \| corolla \| 2019 \| 2027 \|\| drive shaft-(right front) \| toyota \| corolla \| 2019 \| 2027 | 4160 |
| product_id_part_name_brand_model_year_start_year_end_component_000019 | 43 | 8 | 1 | 62404856, 63310181, 63350254, 64546891, 65257587, 65399988, 65680833, 66455069 | curtain airbags -(right) \| toyota \| corolla \| 2019 \| 2027 | 3800 |
| product_id_part_name_brand_model_year_start_year_end_component_000022 | 42 | 7 | 2 | 63310187, 63350255, 64546896, 65257588, 65400008, 65680834, 66455068 | curtain airbags -(left) \| toyota \| corolla \| 2019 \| 2027 \|\| curtain airbags-(left) \| toyota \| corolla \| 2019 \| 2027 | 3795 |
