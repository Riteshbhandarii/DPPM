# Group-Level Target Price Variability

Generated: 2026-06-26 16:46:28 UTC

Input dataset: `datasets/cleaned/clean_master_dataset.csv`
Rows analyzed: 11,321
Target column: `price`

This report measures whether candidate grouping definitions collect observations with similar target prices. High within-group variance suggests a grouping rule may be too coarse. A high singleton rate suggests a grouping rule may be too strict for evaluation or modeling diagnostics.

Variance, standard deviation, and coefficient of variation are computed only for groups with at least two observations. Singleton groups are reported separately rather than assigned zero variance.

## Summary

| strategy | groups | singletons | non_singleton_groups | avg_size | median_size | avg_variance | median_variance | avg_cv | median_cv | max_variance | max_cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| product_id | 2,619 | 834 (31.84%) | 1,785 | 4.32 | 6.00 | 4.38 | 0.10 | 0.0031 | 0.0031 | 205.17 | 0.1173 |
| canonical(part_name, brand, model) | 593 | 96 (16.19%) | 497 | 19.09 | 18.00 | 1443.55 | 21.51 | 0.0724 | 0.0473 | 59353.76 | 0.7157 |
| canonical(part_name, brand, model, oem_number) | 1,624 | 405 (24.94%) | 1,219 | 6.97 | 6.00 | 418.37 | 0.20 | 0.0230 | 0.0031 | 106134.04 | 0.7157 |
| canonical(part_name, brand, model, year_start, year_end) | 1,032 | 189 (18.31%) | 843 | 10.97 | 6.00 | 643.38 | 0.75 | 0.0447 | 0.0032 | 59353.76 | 0.6137 |

## product_id

Full per-group table: `results/group_price_variability/product_id_price_variability.csv`
Highest variance CSV: `results/group_price_variability/product_id_highest_variance.csv`
Lowest variance CSV: `results/group_price_variability/product_id_lowest_variance.csv`

### Highest Variance Identities Top 20

| product_id | listing_count | mean_price | median_price | std_price | price_variance | coefficient_of_variation | min_price | max_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 53935379 | 6 | 4616.30 | 4614.85 | 14.32 | 205.1667 | 0.0031 | 4599.70 | 4641.00 | 41.30 |
| 64483026 | 6 | 4616.30 | 4614.85 | 14.32 | 205.1667 | 0.0031 | 4599.70 | 4641.00 | 41.30 |
| 66544405 | 6 | 4385.47 | 4384.10 | 13.60 | 185.0589 | 0.0031 | 4369.70 | 4408.90 | 39.20 |
| 65552499 | 6 | 4379.57 | 4378.20 | 13.57 | 184.2689 | 0.0031 | 4363.80 | 4403.00 | 39.20 |
| 66395040 | 5 | 4025.20 | 4025.60 | 13.56 | 183.8920 | 0.0034 | 4010.00 | 4046.00 | 36.00 |
| 63425718 | 6 | 4367.75 | 4366.40 | 13.55 | 183.4892 | 0.0031 | 4352.00 | 4391.10 | 39.10 |
| 65116265 | 6 | 4320.38 | 4319.05 | 13.40 | 179.5114 | 0.0031 | 4304.80 | 4343.50 | 38.70 |
| 62661162 | 5 | 4141.06 | 4139.10 | 13.39 | 179.4184 | 0.0032 | 4127.90 | 4165.00 | 37.10 |
| 54279403 | 6 | 4267.12 | 4265.80 | 13.23 | 175.1214 | 0.0031 | 4251.70 | 4289.90 | 38.20 |
| 57343770 | 6 | 4261.20 | 4259.90 | 13.22 | 174.8167 | 0.0031 | 4245.80 | 4284.00 | 38.20 |
| 62801577 | 6 | 112.48 | 118.20 | 13.19 | 173.9881 | 0.1173 | 83.00 | 119.00 | 36.00 |
| 53429849 | 6 | 4213.87 | 4212.55 | 13.07 | 170.7289 | 0.0031 | 4198.70 | 4236.40 | 37.70 |
| 59824465 | 6 | 4142.83 | 4141.55 | 12.85 | 165.2389 | 0.0031 | 4127.90 | 4165.00 | 37.10 |
| 60213093 | 6 | 4142.83 | 4141.55 | 12.85 | 165.2389 | 0.0031 | 4127.90 | 4165.00 | 37.10 |
| 53368462 | 6 | 4095.48 | 4094.20 | 12.72 | 161.7581 | 0.0031 | 4080.70 | 4117.40 | 36.70 |
| 58360841 | 6 | 4095.48 | 4094.20 | 12.72 | 161.7581 | 0.0031 | 4080.70 | 4117.40 | 36.70 |
| 61922680 | 6 | 4083.65 | 4082.40 | 12.68 | 160.7358 | 0.0031 | 4068.90 | 4105.50 | 36.60 |
| 63205720 | 6 | 4083.65 | 4082.40 | 12.68 | 160.7358 | 0.0031 | 4068.90 | 4105.50 | 36.60 |
| 53387859 | 6 | 3965.28 | 3964.05 | 12.31 | 151.5581 | 0.0031 | 3951.00 | 3986.50 | 35.50 |
| 54025932 | 6 | 3965.28 | 3964.05 | 12.31 | 151.5581 | 0.0031 | 3951.00 | 3986.50 | 35.50 |

### Lowest Variance Identities Top 20

Singleton groups are excluded from this table.

| product_id | listing_count | mean_price | median_price | std_price | price_variance | coefficient_of_variation | min_price | max_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 66891089 | 3 | 9.50 | 9.50 | 0.00 | 0.0000 | 0.0000 | 9.50 | 9.50 | 0.00 |
| 53394183 | 2 | 47.40 | 47.40 | 0.00 | 0.0000 | 0.0000 | 47.40 | 47.40 | 0.00 |
| 65455165 | 2 | 94.40 | 94.40 | 0.00 | 0.0000 | 0.0000 | 94.40 | 94.40 | 0.00 |
| 53621930 | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| 53950695 | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| 54064944 | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| 54115321 | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| 54115323 | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| 53950212 | 6 | 13.02 | 13.00 | 0.04 | 0.0014 | 0.0029 | 13.00 | 13.10 | 0.10 |
| 53964588 | 6 | 14.22 | 14.20 | 0.04 | 0.0014 | 0.0026 | 14.20 | 14.30 | 0.10 |
| 54059287 | 6 | 14.22 | 14.20 | 0.04 | 0.0014 | 0.0026 | 14.20 | 14.30 | 0.10 |
| 54068339 | 6 | 14.22 | 14.20 | 0.04 | 0.0014 | 0.0026 | 14.20 | 14.30 | 0.10 |
| 53821928 | 6 | 11.83 | 11.80 | 0.05 | 0.0022 | 0.0040 | 11.80 | 11.90 | 0.10 |
| 54090871 | 6 | 11.83 | 11.80 | 0.05 | 0.0022 | 0.0040 | 11.80 | 11.90 | 0.10 |
| 54121096 | 6 | 11.83 | 11.80 | 0.05 | 0.0022 | 0.0040 | 11.80 | 11.90 | 0.10 |
| 54244928 | 6 | 11.83 | 11.80 | 0.05 | 0.0022 | 0.0040 | 11.80 | 11.90 | 0.10 |
| 54433078 | 3 | 35.43 | 35.40 | 0.05 | 0.0022 | 0.0013 | 35.40 | 35.50 | 0.10 |
| 53783208 | 6 | 17.77 | 17.80 | 0.05 | 0.0022 | 0.0027 | 17.70 | 17.80 | 0.10 |
| 54039164 | 6 | 17.77 | 17.80 | 0.05 | 0.0022 | 0.0027 | 17.70 | 17.80 | 0.10 |
| 54039165 | 6 | 17.77 | 17.80 | 0.05 | 0.0022 | 0.0027 | 17.70 | 17.80 | 0.10 |

## canonical(part_name, brand, model)

Full per-group table: `results/group_price_variability/part_name_brand_model_price_variability.csv`
Highest variance CSV: `results/group_price_variability/part_name_brand_model_highest_variance.csv`
Lowest variance CSV: `results/group_price_variability/part_name_brand_model_lowest_variance.csv`

### Highest Variance Identities Top 20

| part_name | brand | model | listing_count | mean_price | median_price | std_price | price_variance | coefficient_of_variation | min_price | max_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| engine diesel - , e- | vw | golf | 28 | 3884.21 | 3970.10 | 243.63 | 59353.7564 | 0.0627 | 3488.70 | 4117.40 | 628.70 |
| automatic gear - | skoda | octavia | 24 | 4336.65 | 4381.15 | 235.09 | 55267.0708 | 0.0542 | 3951.00 | 4641.00 | 690.00 |
| automatic gear - , e- | vw | golf | 30 | 3598.94 | 3549.90 | 219.06 | 47986.2151 | 0.0609 | 3305.20 | 3927.00 | 621.80 |
| hybrid inverter - | toyota | corolla | 40 | 1013.58 | 1085.45 | 209.18 | 43756.8824 | 0.2064 | 355.20 | 1190.00 | 834.80 |
| engine diesel - | skoda | octavia | 30 | 3888.50 | 3910.85 | 209.12 | 43730.5597 | 0.0538 | 3529.30 | 4165.00 | 635.70 |
| main cylinder - | toyota | corolla | 38 | 494.91 | 592.00 | 204.50 | 41820.8438 | 0.4132 | 59.20 | 595.00 | 535.80 |
| engine gasoline - , e- | vw | golf | 30 | 4096.08 | 4100.45 | 186.24 | 34686.6136 | 0.0455 | 3753.40 | 4343.50 | 590.10 |
| gear box 6 speed - | toyota | corolla | 29 | 1287.92 | 1356.60 | 185.87 | 34548.0405 | 0.1443 | 592.00 | 1428.00 | 836.00 |
| engine gasoline - | skoda | octavia | 23 | 4309.72 | 4262.40 | 171.37 | 29366.5919 | 0.0398 | 4127.90 | 4641.00 | 513.10 |
| brake servo - | toyota | corolla | 38 | 473.11 | 508.70 | 166.41 | 27693.0680 | 0.3517 | 118.40 | 595.00 | 476.60 |
| brake servo - , e- | vw | golf | 30 | 265.93 | 188.80 | 163.02 | 26575.3553 | 0.6130 | 176.90 | 595.00 | 418.10 |
| automatic gear- | toyota | corolla | 3 | 2920.83 | 2975.00 | 144.42 | 20858.5489 | 0.0494 | 2723.20 | 3064.30 | 341.10 |
| kamera utvändig - | toyota | corolla | 39 | 609.21 | 652.40 | 142.67 | 20353.8283 | 0.2342 | 296.00 | 714.00 | 418.00 |
| automatic gear - | toyota | corolla | 37 | 2912.37 | 2960.00 | 126.01 | 15877.6836 | 0.0433 | 2604.80 | 3054.50 | 449.70 |
| gear box 5 speed - , e- | vw | golf | 21 | 1983.10 | 2005.00 | 117.02 | 13693.8133 | 0.0590 | 1804.10 | 2142.00 | 337.90 |
| abs hydraulic pump - | toyota | corolla | 39 | 474.31 | 530.90 | 102.63 | 10532.7069 | 0.2164 | 236.80 | 547.40 | 310.60 |
| hybrid batteri - | toyota | corolla | 40 | 1287.90 | 1300.10 | 94.08 | 8850.3242 | 0.0730 | 1124.80 | 1462.20 | 337.40 |
| gear box 6 speed - , e- | vw | golf | 30 | 2199.25 | 2138.60 | 93.87 | 8811.8805 | 0.0427 | 2122.90 | 2380.00 | 257.10 |
| gear box 5 speed- , e- | vw | golf | 3 | 1929.97 | 1963.50 | 92.73 | 8599.6022 | 0.0480 | 1803.40 | 2023.00 | 219.60 |
| engine diesel - | toyota | corolla | 38 | 901.34 | 935.40 | 89.62 | 8032.3098 | 0.0994 | 522.20 | 952.00 | 429.80 |

### Lowest Variance Identities Top 20

Singleton groups are excluded from this table.

| part_name | brand | model | listing_count | mean_price | median_price | std_price | price_variance | coefficient_of_variation | min_price | max_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rear sensor -(centre) | toyota | corolla | 4 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| alternator- | toyota | corolla | 2 | 142.10 | 142.10 | 0.00 | 0.0000 | 0.0000 | 142.10 | 142.10 | 0.00 |
| bensin/dieselvärmare styrenhet- | skoda | octavia | 2 | 94.40 | 94.40 | 0.00 | 0.0000 | 0.0000 | 94.40 | 94.40 | 0.00 |
| brake disc-(front) | skoda | octavia | 2 | 11.80 | 11.80 | 0.00 | 0.0000 | 0.0000 | 11.80 | 11.80 | 0.00 |
| drive shaft-(left front) | toyota | corolla | 2 | 236.80 | 236.80 | 0.00 | 0.0000 | 0.0000 | 236.80 | 236.80 | 0.00 |
| dörr styrenhet- , e-(right) | vw | golf | 2 | 106.80 | 106.80 | 0.00 | 0.0000 | 0.0000 | 106.80 | 106.80 | 0.00 |
| gear lever- , e- | vw | golf | 2 | 176.90 | 176.90 | 0.00 | 0.0000 | 0.0000 | 176.90 | 176.90 | 0.00 |
| main cylinder- | toyota | corolla | 2 | 60.40 | 60.40 | 0.00 | 0.0000 | 0.0000 | 60.40 | 60.40 | 0.00 |
| passenger airbag- | toyota | corolla | 2 | 296.50 | 296.50 | 0.00 | 0.0000 | 0.0000 | 296.50 | 296.50 | 0.00 |
| rear axle beam- | toyota | corolla | 2 | 415.20 | 415.20 | 0.00 | 0.0000 | 0.0000 | 415.20 | 415.20 | 0.00 |
| sump- | skoda | octavia | 2 | 107.10 | 107.10 | 0.00 | 0.0000 | 0.0000 | 107.10 | 107.10 | 0.00 |
| airbag front sensor - , e-(left) | vw | golf | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| airbag front sensor - , e-(right) | vw | golf | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| rear sensor -(exterior) | toyota | corolla | 3 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| brake shield - , e-(rear) | vw | golf | 6 | 11.83 | 11.80 | 0.05 | 0.0022 | 0.0040 | 11.80 | 11.90 | 0.10 |
| brake disc -(rear) | skoda | octavia | 11 | 11.85 | 11.90 | 0.05 | 0.0025 | 0.0042 | 11.80 | 11.90 | 0.10 |
| fuel tank lid - , e- | vw | golf | 29 | 17.75 | 17.70 | 0.05 | 0.0025 | 0.0028 | 17.70 | 17.80 | 0.10 |
| other control unit- | skoda | octavia | 2 | 237.15 | 237.15 | 0.05 | 0.0025 | 0.0002 | 237.10 | 237.20 | 0.10 |
| ignition diesel relay- | skoda | octavia | 2 | 29.65 | 29.65 | 0.05 | 0.0025 | 0.0017 | 29.60 | 29.70 | 0.10 |
| ignition module - , e- | vw | golf | 12 | 17.75 | 17.75 | 0.05 | 0.0025 | 0.0028 | 17.70 | 17.80 | 0.10 |

## canonical(part_name, brand, model, oem_number)

Full per-group table: `results/group_price_variability/part_name_brand_model_oem_number_price_variability.csv`
Highest variance CSV: `results/group_price_variability/part_name_brand_model_oem_number_highest_variance.csv`
Lowest variance CSV: `results/group_price_variability/part_name_brand_model_oem_number_lowest_variance.csv`

### Highest Variance Identities Top 20

| part_name | brand | model | oem_number | listing_count | mean_price | median_price | std_price | price_variance | coefficient_of_variation | min_price | max_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| automatic gear - | skoda | octavia | fi27837687a | 12 | 4290.79 | 4293.10 | 325.78 | 106134.0374 | 0.0759 | 3951.00 | 4641.00 | 690.00 |
| engine diesel - , e- | vw | golf | fi09389104a | 14 | 3818.93 | 3899.60 | 270.74 | 73298.9706 | 0.0709 | 3502.80 | 4117.40 | 614.60 |
| engine diesel - | skoda | octavia | fi05351686a | 6 | 4043.98 | 4134.20 | 217.41 | 47268.9114 | 0.0538 | 3558.60 | 4165.00 | 606.40 |
| kamera utvändig - | toyota | corolla | fi05028803a | 11 | 522.08 | 707.60 | 206.39 | 42596.8397 | 0.3953 | 296.00 | 714.00 | 418.00 |
| engine diesel - , e- | vw | golf | fi10331575a | 8 | 3848.86 | 3957.00 | 202.00 | 40804.8023 | 0.0525 | 3488.70 | 3986.50 | 497.80 |
| brake servo - , e- | vw | golf | fi27837687a | 23 | 291.80 | 189.40 | 178.31 | 31795.3448 | 0.6111 | 176.90 | 595.00 | 418.10 |
| automatic gear - | toyota | corolla | fi09389104a | 7 | 2965.76 | 3038.00 | 129.26 | 16707.0596 | 0.0436 | 2723.20 | 3054.50 | 331.30 |
| automatic gear - | toyota | corolla | fi27837687a | 21 | 2915.22 | 2960.00 | 112.94 | 12754.3732 | 0.0387 | 2604.80 | 2975.00 | 370.20 |
| main cylinder - | toyota | corolla | fi27837687a | 26 | 571.58 | 592.00 | 102.49 | 10504.6751 | 0.1793 | 59.20 | 595.00 | 535.80 |
| gear box 5 speed - | toyota | corolla | fi06018105a | 6 | 789.45 | 827.90 | 88.34 | 7803.9092 | 0.1119 | 592.00 | 833.00 | 241.00 |
| hybrid batteri - | toyota | corolla | fi09389104a | 18 | 1295.07 | 1297.55 | 83.60 | 6989.6378 | 0.0646 | 1219.50 | 1462.20 | 242.70 |
| injection control unit - | toyota | corolla | fi27837687a | 14 | 524.21 | 533.65 | 83.32 | 6941.9412 | 0.1589 | 355.20 | 595.00 | 239.80 |
| engine gasoline - | toyota | corolla | fi27837687a | 20 | 3635.81 | 3670.40 | 76.70 | 5883.1399 | 0.0211 | 3433.60 | 3689.00 | 255.40 |
| abs hydraulic pump - | toyota | corolla | fi15710056a | 5 | 296.00 | 296.00 | 64.85 | 4205.5680 | 0.2191 | 236.80 | 414.40 | 177.60 |
| engine gasoline - | skoda | octavia | fi09389104a | 12 | 4204.98 | 4208.35 | 63.50 | 4031.7669 | 0.0151 | 4127.90 | 4289.90 | 162.00 |
| airbag control unit - | toyota | corolla | fi27837687a | 8 | 319.70 | 354.55 | 61.55 | 3788.7950 | 0.1925 | 213.10 | 357.00 | 143.90 |
| engine gasoline - , e- | vw | golf | fi27837687a | 18 | 4131.00 | 4100.45 | 60.18 | 3621.1878 | 0.0146 | 4068.90 | 4236.40 | 167.50 |
| hybrid inverter - | toyota | corolla | fi27837687a | 4 | 414.40 | 414.40 | 59.20 | 3504.6400 | 0.1429 | 355.20 | 473.60 | 118.40 |
| drive shaft -(right front) | toyota | corolla | fi27837687a | 26 | 318.27 | 353.90 | 57.32 | 3285.1827 | 0.1801 | 177.60 | 357.00 | 179.40 |
| drive shaft - , e-(right rear) | vw | golf | fi10331575a | 9 | 276.46 | 237.20 | 56.18 | 3156.3247 | 0.2032 | 235.90 | 357.00 | 121.10 |

### Lowest Variance Identities Top 20

Singleton groups are excluded from this table.

| part_name | brand | model | oem_number | listing_count | mean_price | median_price | std_price | price_variance | coefficient_of_variation | min_price | max_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fuel filling pipe / tube - | toyota | corolla | fi11042417a | 5 | 47.40 | 47.40 | 0.00 | 0.0000 | 0.0000 | 47.40 | 47.40 | 0.00 |
| brake caliper -(right front) | toyota | corolla | fi05028803a | 4 | 142.10 | 142.10 | 0.00 | 0.0000 | 0.0000 | 142.10 | 142.10 | 0.00 |
| rear sensor -(centre) | toyota | corolla | fi27837687a | 4 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| shock absorbers rear -(rear) | toyota | corolla | fi02154548a | 4 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| brake caliper -(left rear) | toyota | corolla | fi09389104a | 3 | 219.00 | 219.00 | 0.00 | 0.0000 | 0.0000 | 219.00 | 219.00 | 0.00 |
| brake caliper -(right rear) | toyota | corolla | fi09389104a | 3 | 219.00 | 219.00 | 0.00 | 0.0000 | 0.0000 | 219.00 | 219.00 | 0.00 |
| contact roll airbag - | toyota | corolla | fi05028803a | 3 | 177.60 | 177.60 | 0.00 | 0.0000 | 0.0000 | 177.60 | 177.60 | 0.00 |
| distributors - | toyota | corolla | fi15710056a | 3 | 71.00 | 71.00 | 0.00 | 0.0000 | 0.0000 | 71.00 | 71.00 | 0.00 |
| engine casing - | toyota | corolla | fi07265116a | 3 | 48.00 | 48.00 | 0.00 | 0.0000 | 0.0000 | 48.00 | 48.00 | 0.00 |
| injection control unit - | toyota | corolla | fi02154548a | 3 | 355.20 | 355.20 | 0.00 | 0.0000 | 0.0000 | 355.20 | 355.20 | 0.00 |
| power steering control unit - | toyota | corolla | fi11042417a | 3 | 177.60 | 177.60 | 0.00 | 0.0000 | 0.0000 | 177.60 | 177.60 | 0.00 |
| rear axle beam - | toyota | corolla | fi05028803a | 3 | 296.00 | 296.00 | 0.00 | 0.0000 | 0.0000 | 296.00 | 296.00 | 0.00 |
| actuator loom - | toyota | corolla | fi06509801a | 2 | 50.60 | 50.60 | 0.00 | 0.0000 | 0.0000 | 50.60 | 50.60 | 0.00 |
| air purifier - | skoda | octavia | fi24637030a | 2 | 94.40 | 94.40 | 0.00 | 0.0000 | 0.0000 | 94.40 | 94.40 | 0.00 |
| airbag front sensor -(left) | toyota | corolla | fi03986645a | 2 | 47.40 | 47.40 | 0.00 | 0.0000 | 0.0000 | 47.40 | 47.40 | 0.00 |
| airbag front sensor -(left) | toyota | corolla | fi05351686a | 2 | 41.40 | 41.40 | 0.00 | 0.0000 | 0.0000 | 41.40 | 41.40 | 0.00 |
| airbag front sensor -(left) | toyota | corolla | fi10331575a | 2 | 23.70 | 23.70 | 0.00 | 0.0000 | 0.0000 | 23.70 | 23.70 | 0.00 |
| airbag krocksensor - | toyota | corolla | fi09515254a | 2 | 53.30 | 53.30 | 0.00 | 0.0000 | 0.0000 | 53.30 | 53.30 | 0.00 |
| airbag krocksensor - | toyota | corolla | fi24637030a | 2 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| alternator - | toyota | corolla | fi11042417a | 2 | 118.40 | 118.40 | 0.00 | 0.0000 | 0.0000 | 118.40 | 118.40 | 0.00 |

## canonical(part_name, brand, model, year_start, year_end)

Full per-group table: `results/group_price_variability/part_name_brand_model_year_start_year_end_price_variability.csv`
Highest variance CSV: `results/group_price_variability/part_name_brand_model_year_start_year_end_highest_variance.csv`
Lowest variance CSV: `results/group_price_variability/part_name_brand_model_year_start_year_end_lowest_variance.csv`

### Highest Variance Identities Top 20

| part_name | brand | model | year_start | year_end | listing_count | mean_price | median_price | std_price | price_variance | coefficient_of_variation | min_price | max_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| engine diesel - , e- | vw | golf | 2013 | 2020 | 28 | 3884.21 | 3970.10 | 243.63 | 59353.7564 | 0.0627 | 3488.70 | 4117.40 | 628.70 |
| hybrid inverter - | toyota | corolla | 2019 | 2027 | 40 | 1013.58 | 1085.45 | 209.18 | 43756.8824 | 0.2064 | 355.20 | 1190.00 | 834.80 |
| engine diesel - | skoda | octavia | 2013 | 2020 | 30 | 3888.50 | 3910.85 | 209.12 | 43730.5597 | 0.0538 | 3529.30 | 4165.00 | 635.70 |
| engine gasoline - , e- | vw | golf | 2013 | 2020 | 30 | 4096.08 | 4100.45 | 186.24 | 34686.6136 | 0.0455 | 3753.40 | 4343.50 | 590.10 |
| gear box 6 speed - | toyota | corolla | 2008 | 2012 | 29 | 1287.92 | 1356.60 | 185.87 | 34548.0405 | 0.1443 | 592.00 | 1428.00 | 836.00 |
| automatic gear - , e- | vw | golf | 2013 | 2020 | 24 | 3522.15 | 3494.60 | 174.53 | 30462.4367 | 0.0496 | 3305.20 | 3808.00 | 502.80 |
| brake servo - , e- | vw | golf | 2013 | 2020 | 29 | 268.89 | 188.80 | 165.01 | 27227.7544 | 0.6137 | 176.90 | 595.00 | 418.10 |
| automatic gear- | toyota | corolla | 2019 | 2027 | 3 | 2920.83 | 2975.00 | 144.42 | 20858.5489 | 0.0494 | 2723.20 | 3064.30 | 341.10 |
| kamera utvändig - | toyota | corolla | 2019 | 2027 | 38 | 611.22 | 653.45 | 143.99 | 20731.7560 | 0.2356 | 296.00 | 714.00 | 418.00 |
| automatic gear - | toyota | corolla | 2019 | 2027 | 36 | 2911.05 | 2960.00 | 127.49 | 16253.9708 | 0.0438 | 2604.80 | 3054.50 | 449.70 |
| gear box 5 speed - , e- | vw | golf | 2013 | 2020 | 21 | 1983.10 | 2005.00 | 117.02 | 13693.8133 | 0.0590 | 1804.10 | 2142.00 | 337.90 |
| automatic gear - | skoda | octavia | 2020 | 2027 | 18 | 4460.44 | 4391.90 | 111.10 | 12342.7769 | 0.0249 | 4363.80 | 4641.00 | 277.20 |
| engine diesel - | toyota | corolla | 2002 | 2007 | 25 | 893.82 | 935.40 | 106.91 | 11429.1893 | 0.1196 | 522.20 | 952.00 | 429.80 |
| hybrid batteri - | toyota | corolla | 2019 | 2027 | 40 | 1287.90 | 1300.10 | 94.08 | 8850.3242 | 0.0730 | 1124.80 | 1462.20 | 337.40 |
| gear box 6 speed - , e- | vw | golf | 2013 | 2020 | 30 | 2199.25 | 2138.60 | 93.87 | 8811.8805 | 0.0427 | 2122.90 | 2380.00 | 257.10 |
| gear box 5 speed- , e- | vw | golf | 2013 | 2020 | 3 | 1929.97 | 1963.50 | 92.73 | 8599.6022 | 0.0480 | 1803.40 | 2023.00 | 219.60 |
| gear box 5 speed - | toyota | corolla | 2002 | 2007 | 26 | 673.92 | 652.20 | 91.95 | 8454.0954 | 0.1364 | 532.80 | 833.00 | 300.20 |
| gear box 6 speed- | toyota | corolla | 2008 | 2012 | 5 | 1349.50 | 1368.50 | 89.26 | 7966.6280 | 0.0661 | 1179.40 | 1428.00 | 248.60 |
| gear box 5 speed - | toyota | corolla | 2008 | 2012 | 7 | 733.66 | 769.60 | 88.31 | 7799.4396 | 0.1204 | 517.40 | 773.50 | 256.10 |
| engine gasoline - | toyota | corolla | 2019 | 2027 | 39 | 3611.58 | 3656.10 | 69.92 | 4889.1085 | 0.0194 | 3433.60 | 3689.00 | 255.40 |

### Lowest Variance Identities Top 20

Singleton groups are excluded from this table.

| part_name | brand | model | year_start | year_end | listing_count | mean_price | median_price | std_price | price_variance | coefficient_of_variation | min_price | max_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| suspension -(rear) | toyota | corolla | 2008 | 2012 | 8 | 35.50 | 35.50 | 0.00 | 0.0000 | 0.0000 | 35.50 | 35.50 | 0.00 |
| rear sensor -(centre) | toyota | corolla | 2019 | 2027 | 4 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| starter gasoline - | toyota | corolla | 2013 | 2018 | 4 | 94.70 | 94.70 | 0.00 | 0.0000 | 0.0000 | 94.70 | 94.70 | 0.00 |
| mass air-flow sensor - | toyota | corolla | 2008 | 2012 | 3 | 71.00 | 71.00 | 0.00 | 0.0000 | 0.0000 | 71.00 | 71.00 | 0.00 |
| sensor abs - | toyota | corolla | 1993 | 1997 | 3 | 41.40 | 41.40 | 0.00 | 0.0000 | 0.0000 | 41.40 | 41.40 | 0.00 |
| bensin/dieselvärmare styrenhet- | skoda | octavia | 2013 | 2020 | 2 | 94.40 | 94.40 | 0.00 | 0.0000 | 0.0000 | 94.40 | 94.40 | 0.00 |
| brake disc-(front) | skoda | octavia | 2005 | 2013 | 2 | 11.80 | 11.80 | 0.00 | 0.0000 | 0.0000 | 11.80 | 11.80 | 0.00 |
| coil - | toyota | corolla | 1998 | 2001 | 2 | 47.40 | 47.40 | 0.00 | 0.0000 | 0.0000 | 47.40 | 47.40 | 0.00 |
| coil - | toyota | corolla | 2019 | 2027 | 2 | 47.40 | 47.40 | 0.00 | 0.0000 | 0.0000 | 47.40 | 47.40 | 0.00 |
| dörr styrenhet- , e-(right) | vw | golf | 2013 | 2020 | 2 | 106.80 | 106.80 | 0.00 | 0.0000 | 0.0000 | 106.80 | 106.80 | 0.00 |
| other relay - | toyota | corolla | 1998 | 2001 | 2 | 35.50 | 35.50 | 0.00 | 0.0000 | 0.0000 | 35.50 | 35.50 | 0.00 |
| rear axle beam- | toyota | corolla | 2019 | 2027 | 2 | 415.20 | 415.20 | 0.00 | 0.0000 | 0.0000 | 415.20 | 415.20 | 0.00 |
| starter diesel- | toyota | corolla | 2008 | 2012 | 2 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| steering wheel airbag - | toyota | corolla | 2008 | 2012 | 2 | 260.50 | 260.50 | 0.00 | 0.0000 | 0.0000 | 260.50 | 260.50 | 0.00 |
| suspension -(rear) | toyota | corolla | 2013 | 2018 | 2 | 35.50 | 35.50 | 0.00 | 0.0000 | 0.0000 | 35.50 | 35.50 | 0.00 |
| trailing link rear -(left) | toyota | corolla | 1993 | 1997 | 2 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| trailing link rear -(right) | toyota | corolla | 1993 | 1997 | 2 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
| airbag front sensor - , e-(left) | vw | golf | 2004 | 2009 | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| airbag front sensor - , e-(right) | vw | golf | 2004 | 2009 | 6 | 5.90 | 5.90 | 0.00 | 0.0000 | 0.0000 | 5.90 | 5.90 | 0.00 |
| rear sensor -(exterior) | toyota | corolla | 2019 | 2027 | 3 | 59.20 | 59.20 | 0.00 | 0.0000 | 0.0000 | 59.20 | 59.20 | 0.00 |
