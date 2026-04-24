# Final Strict Model Selection

Final model selection is based on strict part-identity grouped cross-validation on
`datasets/splits/train_grouped.csv`. Folds are grouped by
`part_name + brand + model + oem_number`.

| Model | Role | Feature variant | MAE | RMSE | R2 | Median AE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Random forest | Strict tuned finalist | `trusted_recommended_features_without_oem_number` | 34.4796 | 70.3158 | 0.9864 | 12.3629 |
| XGBoost | Strict tuned finalist | `trusted_recommended_features_without_date_offsets_without_oem_number` | 40.3583 | 87.1192 | 0.9789 | 16.7802 |
| Linear ridge | Strict baseline comparator | `trusted_recommended_features_without_listing_dates` | 53.7993 | 154.8746 | 0.9326 | 16.8031 |
| CatBoost | Strict baseline comparator | `trusted_recommended_features_without_date_offsets` | 99.5891 | 262.0683 | 0.7968 | 31.1834 |

The final strict model-selection winner is random forest without `oem_number`.
The no-OEM result also reduces dependence on exact OEM-level identity information
while improving the previous strict random-forest result.

