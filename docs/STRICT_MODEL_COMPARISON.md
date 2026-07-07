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
| Selection bias / test-set contamination | Procedure and top-2 rule fixed in advance; the strict test split is reserved for exactly one evaluation of the single winner (`02_strict_final_holdout.ipynb`, guarded to run once). |
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
