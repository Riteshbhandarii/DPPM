# Strict Split Model Comparison — Protocol and Bias Controls

This document records how the stage-1 model comparison on the strict connected-component
split was run (notebook `notebooks/05_strict_training/01_strict_model_comparison.ipynb`,
artifacts in `artifacts/strict_model_comparison/`), why it was designed this way, where
every setting came from, and which bias controls were in place.

## 1. What was trained

Four regression models, each predicting the listing price in euros:

| Model | Target scale | Configurations | Source of configurations |
| --- | --- | --- | --- |
| Ridge (linear baseline) | log(price) | 6 alphas: 0.01–3.0 around 0.05 | The originally selected linear baseline was Ridge(alpha=0.05) with log target; OLS was tested and rejected earlier. Linear tuning is cheap, so the known alpha is bracketed by a small grid. |
| Random Forest | raw and log variants | 5 anchor configs | The five finalist configurations from the original tuned RF workflow (`src/tree_modeling.py::RANDOM_FOREST_CONFIGS`). |
| XGBoost | raw and log variants | 5 anchor configs | The five finalist configurations from the original tuned XGBoost workflow (`src/tree_modeling.py::XGBOOST_CONFIGS`). |
| CatBoost | raw and log variants | 5 configs | The configuration used in the original CatBoost evaluation (`raw_rmse_depth7`) plus close known variants. |

Each configuration was crossed with the model's **trusted feature variants** from the
shared feature catalog (`src/tree_modeling.py::build_feature_catalog`) — the same
catalog, exclusions, and leakage-risk feature rules used in the original workflow.

Two trivial anchors were evaluated alongside: predict the training-set global median
price, and predict the training-set median price of the listing's subcategory.

## 2. Data and evaluation

- Training data: `datasets/splits_strict/train_strict.csv` (7,930 rows).
- Scoring data: `datasets/splits_strict/validation_strict.csv` (1,695 rows), used once
  per candidate as a fixed validation set.
- The split is the frozen connected-component split (seed 32, provenance and input
  sha256 in `datasets/splits_strict/strict_split_summary.json`). The test split was not
  read, loaded, or touched at any point in this stage.
- Metrics: MAE (primary, euro-interpretable), RMSE, R², median absolute error. Models
  trained on log(price) have predictions transformed back to euros (with clipping
  safeguards) before metrics are computed, so all models are compared on the same scale.
- Ranking rule (fixed before running): best candidate per model by validation MAE, with
  RMSE as the tiebreaker; the two best models by validation MAE proceed to the tuning
  stage.

## 3. Why this design

1. **Only the split changes.** The procedure — fixed-validation comparison of each
   model's known configurations across trusted feature variants — is the same procedure
   the original training notebooks used on the product-id grouped split. Keeping the
   procedure constant makes the strict-vs-original result difference attributable to the
   evaluation design, not to a changed training recipe.
2. **Known configurations, not library defaults.** Library defaults would handicap
   models arbitrarily (default XGBoost is far from competitive; default Ridge alpha is
   1.0). Giving every model its previously selected best-known setup is the equal-effort
   comparison: same data, same features, same metric, best-known settings each.
3. **No random-search tuning at this stage.** Tuning all four models under
   cross-validation would multiply compute for models that will not survive the
   comparison. The staged funnel (compare all → tune the top two → one holdout
   evaluation of the winner) spends the tuning budget only where it matters. This is the
   pre-registered procedure; the top-2 rule was fixed before results existed.

## 4. Bias and leakage controls

| Risk | Control |
| --- | --- |
| Repeated listings of the same part crossing train/validation | Frozen connected-component split: no `product_id`, no strict identity key, and no connected component appears in more than one split (three assertions, all PASS at generation; identity-key overlap re-asserted at runtime in notebook cell 5). |
| Preprocessing leaking validation statistics | All preprocessing (median imputation, most-frequent imputation, one-hot encoding with `min_frequency=5`) lives inside the model pipeline and is fitted on training data only. Categorical levels for XGBoost/CatBoost are derived from training data only. |
| Future-looking features inflating results | Trusted feature variants exclude full-history listing variables (`times_observed`, `price_change_*`, `observed_span_days`, last-seen/midpoint offsets) and identifiers (`product_id`, `scrape_date`, merge keys) — same exclusion lists as the original workflow (`src/tree_modeling.py`). |
| Early stopping peeking at the scored data | XGBoost and CatBoost early-stop on an inner 10% component-grouped carve of the training split (`GroupShuffleSplit`, seed 42). The validation split is never used for early stopping. (This corrects a weakness of the original grouped-CV code, which early-stopped on the fold being scored.) |
| Selection bias / test-set contamination | Procedure and top-2 rule fixed in advance; the strict test split is reserved for exactly one evaluation of the single winner (`03_strict_final_holdout.ipynb`, guarded to run once). |
| Irreproducibility | Fixed seeds throughout (split seed 32; model/carve seeds 42); per-candidate results and best-summary JSONs saved under `artifacts/strict_model_comparison/`; the component rule used for grouping is proven identical to the frozen split's rule by `tests/test_strict_protocol.py`. |

## 5. Known limitations (stated, not hidden)

- **Anchor configurations were tuned under the original product-id split**, so they may
  slightly favor settings that suited that regime. Mitigation: the two finalists are
  re-tuned from scratch under the strict protocol in the tuning stage; the final claim
  never rests on old-split tuning.
- **A single fixed validation split has sampling variance.** Mitigation: the tuning
  stage ranks candidates by 4-fold component-grouped cross-validation; the comparison
  stage only needs to be accurate enough to pick two finalists.
- The dataset-level limitations (asking prices, one marketplace, three model families,
  short scrape window) are documented in `docs/DATASET_CHARACTERISTICS.md` and apply to
  every model equally.

## 6. Stage-1 result (2026-07-07)

| Model | Validation MAE (€) | RMSE (€) | R² | Median AE (€) |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 92.05 | 247.57 | 0.847 | 15.89 |
| Random Forest | 92.93 | 239.15 | 0.858 | 23.51 |
| XGBoost | 106.91 | 281.80 | 0.802 | 27.09 |
| CatBoost | 168.82 | 527.97 | 0.306 | 33.15 |
| (dummy: subcategory median) | 120.54 | — | — | 23.50 |
| (dummy: global median) | 238.27 | — | — | 59.10 |

Per the pre-registered rule, **Ridge and Random Forest** proceed to the tuning stage.

## 7. Stage-2 strict tuning result (2026-07-07)

Stage-2 tuning used the strict connected-component protocol on the frozen training split. The test split was still not read, loaded, or touched.

| Model | Feature set | Strict CV MAE (€) | RMSE (€) | R² | Median AE (€) |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge | Trusted recommended features without listing dates without OEM number | 106.71 | 323.23 | 0.675 | 23.78 |
| Random Forest | Trusted extended Traficom stack without OEM number | **105.33** | **265.63** | **0.765** | 34.85 |

Under the primary MAE criterion, Random Forest is the strict tuning winner.

## 8. Stage-3 final holdout result (2026-07-10) — run once

The frozen Random Forest winner was refit on `train_strict + validation_strict`
(9,625 rows) and evaluated exactly once on the untouched `test_strict.csv`
(1,696 rows). Notebook `03_strict_final_holdout.ipynb`; artifacts in
`artifacts/strict_final_holdout/`. The run-once guard is now consumed.

Pre-flight verification: 23/23 tests passing, no pre-existing holdout artifacts,
zero `product_id` overlap between fit and test, zero connected components spanning
splits. Run from `main` at `b530e59`, i.e. after the winner was documented — the
pre-registered ordering held.

| Metric | Value | Bootstrap 95% CI (B=10,000, seed 32) |
| --- | ---: | :---: |
| MAE | 69.46 € | [61.70, 77.96] |
| Median AE | 29.37 € | [27.85, 31.92] |
| RMSE | 182.41 € | — |
| R² | 0.9113 | [0.8871, 0.9310] |
| MdAPE | 29.74 % | — |

**These numbers are final.** They are not comparable to the stage-2 cross-validation
MAE of 105.33 €: each CV fold trained on roughly 5,948 rows, while the final model
trained on 9,625 (+62%). k-fold CV is a pessimistically biased estimate of the refit
model, and the long tail benefits most from additional comparables. The stage-2 number
is *selection* evidence; the holdout number is the *generalization* estimate. They must
not be presented as a before/after improvement.

## 9. Baseline comparison on the holdout (2026-07-10)

The two trivial anchors were fitted on `train_strict + validation_strict` — the same fit
scope as the Random Forest — and scored on the same test split
(`scripts/holdout_baseline_comparison.py`,
`artifacts/strict_final_holdout/holdout_baseline_comparison.json`). No model was fitted,
selected, or re-predicted; the Random Forest predictions were read from the frozen
artifact. 15 test rows carry a subcategory unseen in the fit data and fall back to the
global median.

| Predictor | MAE (€) | Median AE (€) | RMSE (€) | R² | MdAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest (frozen winner) | 69.46 | 29.37 | **182.41** | **0.9113** | 29.74 |
| Dummy: subcategory median | **66.15** | **15.32** | 200.94 | 0.8924 | **16.43** |
| Dummy: global median | 216.08 | 58.25 | — | — | 71.42 |

Paired bootstrap on the same rows (B=10,000, seed 32), Random Forest minus dummy, where
a positive difference means the model is worse:

- **ΔMAE = +3.31 €**, CI [−2.88, +8.59] — **not significant**. The model and the lookup
  table tie on mean absolute error.
- **Δmedian AE = +14.04 €**, CI [+11.51, +18.23] — the model is **significantly worse**
  on the typical listing.
- The dummy is closer to the true price on **66.0 %** of test rows.

### Segment-wise (paired bootstrap on MAE, positive = model worse)

| Price segment | n | RF MAE (€) | Dummy MAE (€) | Δ MAE (€) | 95% CI | Verdict |
| --- | ---: | ---: | ---: | ---: | :---: | --- |
| < 50 | 360 | 38.08 | 19.02 | +19.06 | [+14.83, +23.25] | **RF worse** |
| 50–100 | 528 | 20.49 | 11.50 | +8.99 | [+6.71, +11.48] | **RF worse** |
| 100–200 | 318 | 47.01 | 42.73 | +4.28 | [−0.89, +9.28] | tie |
| 200–500 | 348 | 60.24 | 62.70 | −2.46 | [−12.95, +8.36] | tie |
| 500–1 000 | 67 | 51.31 | 192.27 | −140.96 | [−262.18, −48.35] | **RF better** |
| > 1 000 | 75 | 718.94 | 679.61 | +39.33 | [−12.15, +85.78] | tie |

The model's only significant win is the 500–1 000 € band: 67 rows, 4.0 % of the test
split, holding 2.9 % of its total absolute error.

### Error structure of the frozen model

The tail dominates the mean. Listings above 1 000 € are 4.4 % of the test split but carry
**45.8 %** of total absolute error; excluding them, test MAE falls from 69.46 € to
**39.41 €**. In that segment the model **over-predicts systematically** (mean signed error
+480.96 €, median +897.91 €; 61.3 % of rows predicted too high). Relative error by segment
(MdAPE): 78.13 % below 50 €, 28.00 % at 50–100 €, 40.88 % at 100–200 €, **12.31 %** at
200–500 € (best), 5.58 % at 500–1 000 €, 34.75 % above 1 000 €.

## 10. Interpretation

A three-line `groupby` on `subcategory` matches a tuned 373-tree Random Forest across
61 features. The model's predictions correlate with the dummy's at **0.9841**: it has
largely re-learned the subcategory median. Its incremental contribution over that lookup
is a **17.6 % reduction in squared error** (RMSE 182.41 € vs 200.94 €), concentrated in
higher-priced, less conventional listings — real, but modest.

The reported R² of 0.9113 must therefore be read carefully: the subcategory-median dummy
alone scores **0.8924**. Almost all of the explained variance is the informativeness of a
human-assigned category label, not learned structure. R² should not be presented as
headline model performance.

The substantive reading is that used spare-part **asking** prices in this marketplace are
largely *administered by subcategory convention* rather than discovered per listing. A
model trained on those prices learns the convention. This is consistent with, and now
evidence for, the project's standing claim boundary: DPPM is a **market-consistency tool,
not a valuation tool**.

**Scope of the negative result.** This establishes that *the frozen Random Forest* does not
beat the heuristic on typical listings. It does **not** establish that no model can: in
stage-2 cross-validation, Ridge beat the subcategory-median dummy by roughly 24 % on MAE
and 32 % on median AE. Ridge was deliberately **not** evaluated on the holdout, because
selecting it after seeing the Random Forest's result would be selection on the test set
(`docs/DESIGN_DECISIONS.md`, 2026-07-10). The winner was in any case decided by a 1.38 €
margin on mean CV MAE (105.33 vs 106.71) against a fold standard deviation of 24–34 € — a
coin flip on a metric dominated by the tail, on a target whose median is 100.60 € and mean
270.79 €. The choice of primary metric determined the winner. That is a methodological
finding, and it belongs in the discussion chapter.
