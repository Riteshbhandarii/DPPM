# Results

The headline numbers in one place. Every figure here comes from a committed
artifact; the linked documents carry the method and the full tables.

## Connected-component test set (2026-07-10, run once)

Frozen Random Forest, refit on train+validation (9,625 rows), scored once on the
untouched test set of the connected-component split (1,696 rows). Baselines fitted on the same rows.

| Predictor | MAE (EUR) | Median AE (EUR) | RMSE (EUR) | R2 | MdAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 69.46 | 29.37 | 182.41 | 0.911 | 29.7% |
| Subcategory median | 66.15 | 15.32 | 200.94 | 0.892 | 16.4% |
| Global median | 216.08 | 58.25 | - | - | 71.4% |

- MAE difference to the subcategory median: +3.31 EUR, 95% CI [-2.88, +8.59], a tie.
- Median AE difference: +14.04 EUR, CI [+11.51, +18.23], the model is worse.
- By price band the model is worse below 100 EUR, better only at 500-1,000 EUR
  (67 rows), and ties elsewhere. Listings above 1,000 EUR are 4.4% of rows and
  45.8% of total absolute error.
- R2 is not a useful headline here: the lookup table alone reaches 0.892.

Source: `artifacts/strict_final_holdout/`. Full tables, bootstrap method and SHAP:
[the model comparison](STRICT_MODEL_COMPARISON.md), sections 8-11.

## Test-set error by brand and category

Same test set, split by group. RF minus subcategory-median MAE with a paired
bootstrap 95% CI; a positive value means the model is worse. Brand and vehicle
are the same split here (one model per brand).

| Group | n | RF MAE | Median-lookup MAE | Difference CI |
| --- | ---: | ---: | ---: | :---: |
| Skoda | 573 | 56.75 | 50.79 | [-2.26, +14.07] |
| Toyota | 559 | 114.94 | 123.13 | [-24.03, +4.37] |
| VW | 564 | 37.30 | 25.26 | [+7.70, +16.20] |
| Airbag | 122 | 42.14 | 125.94 | [-101.17, -65.64] |
| Brakes | 203 | 30.67 | 33.99 | [-10.86, +4.25] |
| Electric / sensor | 489 | 38.63 | 19.45 | [+15.50, +22.82] |
| Engine | 201 | 52.45 | 76.50 | [-65.64, +9.17] |
| Fuel | 336 | 50.10 | 27.40 | [+17.37, +28.08] |
| Gearbox / axle | 182 | 269.22 | 259.75 | [-7.23, +25.88] |
| Exterior / suspension | 163 | 68.55 | 52.48 | [+5.79, +27.05] |

Source: `artifacts/holdout_subgroup_errors/`, built by
`scripts/holdout_subgroup_errors.py` from the saved predictions (no refit).

## September 2026 live validation

Current Varaosahaku.fi listings, saved by hand, scored against the frozen model
with nothing refitted. From each part x vehicle cell the dearest, the middle and
the cheapest listing are reported. MdAPE is the median absolute percentage error.

**Trained vehicles, 11 parts each** (baseline: the vehicle's own subcategory median)

| Vehicle | Dearest RF / baseline | Middle RF / baseline | Cheapest RF / baseline |
| --- | :---: | :---: | :---: |
| Corolla | 20.9% / 30.9% | 54.7% / 97.0% | 200.6% / 315.0% |
| Golf | 35.4% / 30.8% | 92.2% / 77.5% | 200.1% / 372.0% |
| Octavia | 28.4% / 21.0% | 55.9% / 70.8% | 243.6% / 342.8% |

**Unseen vehicles, 9 parts each** (baseline: all-brand subcategory median, a
weaker comparator, so these gaps are not comparable with the table above)

| Vehicle | Dearest RF / baseline | Middle RF / baseline | Cheapest RF / baseline |
| --- | :---: | :---: | :---: |
| Focus | 50.6% / 36.2% | 72.4% / 65.8% | 279.1% / 271.9% |
| Qashqai | 17.1% / 37.4% | 105.4% / 58.1% | 331.9% / 163.9% |
| V70 | 49.7% / 29.2% | 68.6% / 58.3% | 209.6% / 138.0% |

- Percentage error grows as the listing gets cheaper on every vehicle. The
  model never predicts below about 60 EUR (lowest scored prediction 59.55 EUR),
  so any listing cheaper than that is over-predicted.
- Whether the model or the median lookup is closer depends on the vehicle; no
  pooled "the model wins" result holds across all six.

Collected listings: `results/september_live_validation/september_listings.csv`.
`scripts/predict_september_listings.py` runs the frozen model on them and writes
`results/september_live_validation/september_predictions.csv` with the prediction
and the baseline.
