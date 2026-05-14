# DPPM

DPPM stands for **Dismantler Price Prediction Model**. This repository contains an AMK/Bachelor thesis project on predicting **used automotive spare-part listing prices** from Varaosahaku.fi marketplace listings combined with Traficom-derived Finnish vehicle registry summary features.

The project is designed as a **proof-of-concept decision-support tool** for price review. It is not intended as an automated pricing authority or a definitive market-valuation system.

## Project overview

The thesis addresses a practical pricing problem for dismantler spare-part listings. The workflow combines:

- repeated marketplace listing snapshots from **Varaosahaku.fi**
- Traficom-derived **brand-level and model-level registry summary features**
- cleaning, integration, and grouped evaluation designed to reduce leakage risk

The goal is to estimate an expected listing price from available listing, vehicle, and registry-context information.

## Repository contents

The repository currently includes:

- a **Playwright crawler** for collecting marketplace listing snapshots
- **preprocessing and integration notebooks** for cleaning, merging, and split creation
- **cleaned datasets and grouped train/validation/test splits**
- **model training notebooks and scripts** for linear, random forest, XGBoost, and CatBoost experiments
- **SHAP explainability outputs** and analysis scripts
- a **Streamlit prototype** for interactive decision-support use
- a **FastAPI service** for API-style prediction
- **tests** for serving and UI helper logic

## Current implementation status

The technical implementation is largely in place:

- the final cleaned dataset exists: `datasets/cleaned/clean_master_dataset.csv`
- grouped train/validation/test splits exist under `datasets/splits/`
- multiple model families have been trained and compared
- stricter **part-identity grouped evaluation** has been completed
- SHAP-based explanation workflows exist for the main model paths
- both Streamlit and FastAPI proof-of-concept interfaces exist

The repository should therefore be read as a near-complete thesis implementation plus remaining thesis-writing and presentation work.

## Data and preprocessing status

The main processed data artifacts include:

- `datasets/cleaned/clean_master_dataset.csv`: final cleaned modeling dataset with **11,321 rows**
- `datasets/splits/train_grouped.csv`: grouped training split with **7,954 rows**
- `datasets/splits/validation_grouped.csv`: grouped validation split with **1,689 rows**
- `datasets/splits/test_grouped.csv`: grouped test split with **1,678 rows**

Repeated listing observations were intentionally preserved where useful for listing-history construction. The grouped split keeps all observations from the same `product_id` listing group in exactly one split.

## Model roles

The repository distinguishes three model roles:

- **Operational/UI model**: the context-rich listing-price model used in the demo interface
- **Strict thesis model**: the stricter random-forest modeling path used for the main thesis result
- **Robustness/conservative variant**: a stricter variant with selected listing-history/time features removed to test sensitivity

## Key results summary

The current final model direction is **Random Forest**.  
The main practical evaluation metric is **MAE**, because it is directly interpretable in euros.

### Fixed validation comparison

| Model | Feature set | Validation MAE | Validation RMSE | Validation R2 |
| --- | --- | ---: | ---: | ---: |
| Random forest | trusted recommended features without listing dates | **18.2409** | 48.6056 | 0.9927 |
| XGBoost | trusted recommended features | 18.8845 | **44.5546** | **0.9938** |

### Product-id grouped CV comparison

| Model | Grouped CV MAE | Grouped CV RMSE | Grouped CV R2 |
| --- | ---: | ---: | ---: |
| Random forest | **28.0424 +/- 4.7105** | 75.5137 | 0.9816 |
| XGBoost | 28.9228 +/- 7.3198 | **74.5482** | **0.9819** |

### Stricter part-identity grouped CV comparison

| Model | Part-identity grouped CV MAE | RMSE | R2 | Median AE |
| --- | ---: | ---: | ---: | ---: |
| Random forest, no `oem_number` | **34.4796 +/- 2.7151** | **70.3158** | **0.9864** | **12.3629** |
| XGBoost, no date offsets/no `oem_number` | 40.3583 +/- 4.4666 | 87.1192 | 0.9789 | 16.7802 |
| Linear ridge, clean rerun | 53.6425 +/- 2.8193 | 152.9400 | 0.9343 | 16.5550 |
| CatBoost, clean rerun | 78.8475 +/- 11.3952 | 206.9250 | 0.8789 | 26.4075 |

In short, the validation and grouped-CV results are strong, but the stricter part-identity evaluation gives the more conservative estimate for unseen comparable part identities. Under that stricter setting, Random Forest remains the strongest direction.

### How to read the evaluation layers

The repository now contains four complementary evaluation layers:

- **Fixed validation split**: about **18 EUR MAE** for model selection in the most optimistic development setting
- **Product-id grouped CV**: about **28 EUR MAE** as a stability check across listing-group folds
- **Strict part-identity grouped CV**: about **34 EUR MAE** as the conservative robustness estimate for unseen part identities
- **Held-out grouped test**: about **22 EUR MAE** on the untouched product-id grouped test split

Taken together, the results suggest that expected performance ranges from the more optimistic listing-group setting to the more conservative unseen-part-identity setting. For thesis interpretation, the strict part-identity result is the safest scientific claim, while the grouped test result remains useful as the final held-out estimate under the original product-id split design.

## Why is R2 so high?

The high R2 values should be interpreted carefully.

- Spare-part listing prices have strong **repeated-listing** and **comparable-item** structure.
- Taxonomy variables such as `subcategory`, `part_name`, and `category` explain a large amount of price variation because different spare-part types naturally occupy very different price ranges.
- Product-id grouping prevents the **same exact listing** from crossing train/validation/test boundaries.
- However, highly similar part profiles can still exist under different `product_id` values.

For that reason, very high R2 should **not** be read as perfect general market-value prediction. It is better understood as strong predictive performance within a structured comparable-listing setting, which is why the stricter part-identity evaluation was added.

## Why the stricter part-identity evaluation was needed

Product-id grouping protects against leakage from repeated observations of the same listing, but it does not fully prevent highly similar part identities from appearing in different folds under different listing IDs.

To test this more conservatively, a stricter evaluation grouped by:

```text
part_name + brand + model + oem_number
```

This keeps highly similar part identities within the same fold and gives a more cautious robustness estimate for unseen comparable part profiles.

## Why part category and subcategory matter so much

Spare-part type is naturally one of the strongest price drivers. Engines, gearboxes, control units, body parts, lights, sensors, and small interior parts have very different typical price ranges.

For that reason, taxonomy variables such as `subcategory`, `part_name`, and `category` are expected to dominate SHAP importance. This does **not** mean the model discovered causal market laws. It means the fitted model relies heavily on part taxonomy when predicting listing prices.

## SHAP explainability

SHAP is used here to explain **model behavior**.

- **Global explanations** show which features influence predictions most overall.
- **Local explanations** show why one specific prediction is higher or lower.

The repository supports three explanation paths:

- **Operational/context-rich SHAP**
- **Strict final-model SHAP**
- **Conservative SHAP variant** with selected listing-history/time fields removed

SHAP explanations in this project should be interpreted as **model-specific explanations**, not causal proof.

## Demo / prototype

The repository includes two proof-of-concept interfaces:

- **Streamlit prototype** for interactive decision-support use
- **FastAPI service** for API-style prediction

Both are intended for thesis demonstration and proof-of-concept evaluation, not as production-hardened deployment targets.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install firefox
streamlit run app/streamlit_app.py
```

FastAPI can be started with:

```bash
uvicorn app.fastapi_app:app --reload
```

## Repository structure

- `crawler/`: Playwright crawler for marketplace snapshots
- `notebooks/`: preprocessing, integration, analysis, and training notebooks
- `datasets/`: cleaned, merged, split, and Traficom-derived CSV data
- `scripts/`: tuning, evaluation, export, and utility scripts
- `artifacts/`: model artifacts, tuning outputs, and SHAP outputs
- `app/`: Streamlit and FastAPI proof-of-concept applications
- `src/`: shared modeling and serving utilities
- `tests/`: focused regression tests

## Remaining work

The remaining thesis-stage work is mainly:

- final thesis writing and polishing
- results chapter writing and tightening
- final consistency checks and reruns where needed
- literature review alignment with the implemented workflow
- final SHAP and prototype presentation material preparation if needed

## Summary

DPPM is an applied thesis project that combines marketplace listing data and Traficom-derived registry context to estimate spare-part listing prices. The end-to-end workflow, cleaned datasets, grouped evaluation, model comparisons, explainability tooling, and prototype interfaces are already in place. The current final model direction is Random Forest, while the stricter part-identity evaluation provides the more conservative estimate for thesis interpretation.
