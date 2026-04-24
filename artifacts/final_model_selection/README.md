# Final Model Selection

## Final Winner

**Random Forest** is the current final strict winner.

- Feature variant: `trusted_recommended_features_without_oem_number`
- Config: `raw_half_features_leaf_1`
- Strict CV MAE: **34.4796**
- Strict CV RMSE: **70.3158**
- Strict CV R2: **0.9864**
- Strict CV median AE: **12.3629**

This is the result to report as the final tuned strict Random Forest result.

## Strict Tuned Finalists

Strict model selection used `datasets/splits/train_grouped.csv` only. The CV
folds were grouped by `part_name + brand + model + oem_number`.

| Rank | Model | Feature variant | Config | MAE | RMSE | R2 | Median AE |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Random Forest | `trusted_recommended_features_without_oem_number` | `raw_half_features_leaf_1` | **34.4796** | **70.3158** | **0.9864** | **12.3629** |
| 2 | XGBoost | `trusted_recommended_features_without_date_offsets_without_oem_number` | `random_search_065` | 40.3583 | 87.1192 | 0.9789 | 16.7802 |

## Baseline Comparators

Linear Ridge and CatBoost are baseline comparators, not final tuned winners.
Their clean strict reruns are currently pending on Puhti. These rows should be
refreshed after the jobs finish.

| Model | Current row status | Current MAE | Current RMSE | Current R2 | Current median AE |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear Ridge | Pending clean rerun | 53.7993 | 154.8746 | 0.9326 | 16.8031 |
| CatBoost | Pending clean rerun | 99.5891 | 262.0683 | 0.7968 | 31.1834 |

## Interpretation

Removing `oem_number` improved the strict Random Forest result compared with the
previous strict RF run with OEM information. This supports using the no-OEM RF
as the final model because it performs best while reducing dependence on exact
OEM-level identity.
