# DPPM

DPPM stands for **Dismantler Price Prediction Model**. This repository contains an applied machine-learning thesis project for estimating used vehicle spare-part **listing prices** from dismantler marketplace listings.

The project combines marketplace listing data from Varaosahaku.fi with Traficom-derived Finnish passenger-car registry features. It covers data collection, cleaning, integration, leakage-aware grouped splitting, model comparison, explainability work with SHAP, and two proof-of-concept serving interfaces: a Streamlit prototype and a FastAPI service.

The practical purpose is to support dismantler-facing price review by producing an expected listing-price estimate and an interpretable explanation from available listing, vehicle, and registry-derived features. The system is intended as a thesis proof of concept and decision-support prototype, not as an automated pricing authority.

The repository currently includes a Playwright crawler, processed datasets, grouped train/validation/test splits, model-training notebooks, saved model artifacts, SHAP analysis outputs, application code, and focused tests.

## Key achievements

- Built a thesis-oriented workflow that links dismantler spare-part listings with Traficom-derived brand and model features.
- Preserved repeated listing observations where useful for listing-history features while using grouped train/validation/test splits to reduce leakage risk.
- Compared linear regression, random forest, XGBoost, and CatBoost regressors on the prepared dataset.
- Selected and exported a random-forest model bundle for local serving.
- Added SHAP-based explanation tooling for analysis and local Streamlit prediction explanations.
- Implemented both a Streamlit decision-support prototype and a FastAPI prediction service.

## Thesis scope and contribution

This project should be read as an applied ML thesis workflow and proof-of-concept decision-support system. Its contribution is the end-to-end construction and evaluation of a spare-part asking-price estimation pipeline that combines:

- scraped marketplace listing attributes,
- Traficom-derived vehicle-population and first-registration features,
- leakage-aware grouped splitting for repeated listings,
- comparative regression modeling, and
- prototype model serving with explainability support.

The work does not claim to produce a production-ready pricing system or a definitive market valuation engine. The model estimates expected listing prices from the available dataset and feature representation.

## Results snapshot

The best documented validation result currently available in this repository is the random forest trained with the selected recommended feature set. Model selection was performed on grouped training and validation splits, followed by one held-out grouped test evaluation.

| Stage | Model | Feature set | MAE | RMSE | R2 |
| --- | --- | --- | ---: | ---: | ---: |
| Validation | Random forest | trusted recommended features without listing dates | 18.1299 | 48.5480 | 0.9927 |
| Held-out test | Random forest | trusted recommended features without listing dates | 22.4695 | 62.6210 | 0.9903 |

Metric values above are retained from the existing repository documentation. If final thesis reporting changes, update this table and the detailed results section below from the final evaluated artifacts.

## Quickstart

Create a local environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install firefox
```

Run the Streamlit prototype:

```bash
streamlit run app/streamlit_app.py
```

Run the FastAPI service:

```bash
uvicorn app.fastapi_app:app --reload
```

Run the crawler for one brand/model pair:

```bash
python3 -m crawler --brand Toyota --model Corolla
```

Run the test suite:

```bash
pytest
```

## Repository structure

The repository is organized by responsibility:

- `app/`: user-facing application entrypoints and Streamlit helper modules.
  - `app/streamlit_app.py`: compact Streamlit entrypoint for the proof-of-concept UI.
  - `app/ui_helpers.py`: option cleaning, part-selection logic, comparable-range helpers, and display formatting.
  - `app/shap_utils.py`: local per-prediction SHAP explanation helpers.
  - `app/fastapi_app.py`: minimal FastAPI serving entrypoint.
- `src/`: model-serving and training code shared outside the UI.
  - `src/random_forest_serving.py`: bundle loading and prediction-range helpers.
  - `src/tree_modeling.py`: training-oriented utilities.
- `crawler/`: Playwright-based crawler package for collecting marketplace listings.
- `datasets/`: cleaned, merged, split, and Traficom-derived CSV datasets used across the thesis workflow.
- `artifacts/`: saved model bundles, tuning outputs, and SHAP analysis artifacts.
- `notebooks/`: thesis pipeline notebooks grouped into preprocessing, integration, analysis, and training stages.
- `scripts/`: reproducible command-line and Puhti batch entrypoints for tuning, evaluation, export, and crawling.
- `tests/`: focused regression tests for serving logic, FastAPI behavior, and Streamlit helper behavior.
- `requirements.txt`: pinned local thesis/demo dependencies, including SHAP-compatible analysis versions.

Generated clutter such as notebook checkpoints, temporary notebook mirrors, local caches, and `__pycache__` directories is intentionally excluded from the repository.

## Current tracked data artifacts

The repository contains processed outputs, so the pipeline can be inspected without rerunning every notebook.

- `datasets/cleaned/clean_master_dataset.csv`: final cleaned modeling dataset with 11,321 rows.
- `datasets/splits/train_grouped.csv`: grouped training split with 7,954 rows.
- `datasets/splits/validation_grouped.csv`: grouped validation split with 1,689 rows.
- `datasets/splits/test_grouped.csv`: grouped test split with 1,678 rows.
- `datasets/splits/group_split_assignment.csv`: listing-group split assignment table.
- `datasets/traficom_outputs/brand_summary.csv` and `datasets/traficom_outputs/model_summary.csv`: brand-level and model-level registry summaries.
- `datasets/traficom_outputs/brand_firstreg_summary.csv` and `datasets/traficom_outputs/model_firstreg_summary.csv`: registry lifecycle summaries based on first-registration history.

## Project workflow

The notebooks are organized as a sequential pipeline:

1. **Preprocess source data**
   - Build Traficom registry summary tables in `notebooks/01_preprocessing/01_preprocess_traficom.ipynb`.
   - Clean model-specific listing exports in the other preprocessing notebooks.
2. **Merge snapshot files**
   - Combine repeated crawler exports by brand/model in `notebooks/02_integration/01_loading_and_merging.ipynb`.
3. **Integrate registry features**
   - Merge marketplace listing data with Traficom-derived brand and model summary tables in the integration notebooks.
4. **Clean the master dataset**
   - Preserve repeated listings across scrape dates.
   - Remove only clear same-day duplicates.
   - Add conservative formatting fixes, mileage imputation, and listing-history features.
5. **Create leakage-aware grouped splits**
   - Keep all observations from the same listing group in exactly one split.
   - Save split files under `datasets/splits/`.
6. **Train and compare models**
   - Evaluate baseline listing-only features, listing plus Traficom features, and broader recommended feature sets.
   - Compare linear regression, random forest, XGBoost, and CatBoost models.
7. **Explain and serve the selected model**
   - Use SHAP analysis for model interpretation.
   - Serve the exported random-forest bundle through Streamlit and FastAPI proof-of-concept interfaces.

## Environment setup

The repository includes a pinned root `requirements.txt` for the local thesis/demo workflow.

Recommended local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install firefox
```

Important version notes:

- `scikit-learn==1.7.2` matches the saved random-forest bundle.
- `numpy==1.26.4` and `shap==0.47.1` are pinned to avoid the numerically broken SHAP behavior seen with newer combinations during notebook analysis.
- If you use Puhti modules instead of a local virtual environment, verify the runtime with:

```python
import shap, sklearn, numpy as np, pandas as pd
print(shap.__version__, sklearn.__version__, np.__version__, pd.__version__)
```

## Running the crawler

From the repository root:

```bash
python3 -m crawler --brand Toyota --model Corolla
```

The crawler:

- launches Firefox through Playwright,
- writes output CSVs to `crawler/crawler_datasets/new/`,
- stamps results with the scrape date, and
- currently limits scraping to `5` parts per subcategory for stability.

The category allowlist and crawler settings live in `crawler/src/crawler_config.py`.

For Puhti batch execution, `scripts/batch/crawler.sh` runs several brand/model pairs sequentially through SLURM. That script is environment-specific and assumes:

- CSC module-based Python
- a project path of `/scratch/project_2017273/DPPM`

Adjust those values before reusing it elsewhere.

## Notebook guide

- `notebooks/01_preprocessing/01_preprocess_traficom.ipynb`: builds cleaned Traficom brand and model summary tables.
- `notebooks/02_integration/04_dataset_cleaning.ipynb`: creates the final cleaned master dataset and listing-history features.
- `notebooks/02_integration/05_post_split.ipynb`: creates grouped train/validation/test splits and validates leakage safety.
- `notebooks/03_analysis/01_data_analysis.ipynb`: exploratory analysis of the prepared dataset.
- `notebooks/04_training/01_baseline_linear_regression.ipynb`: linear-model baseline and enriched-feature experiments.
- `notebooks/04_training/02_random_forest.ipynb`: random forest experiments with grouped validation.
- `notebooks/04_training/03_xgboost.ipynb`: XGBoost experiments with grouped validation.
- `notebooks/04_training/04_catboost.ipynb`: CatBoost experiments with native categorical handling.
- `notebooks/04_training/05_random_forest_shap_analysis.ipynb`: SHAP-based analysis notebook for the final random-forest bundle, including diagnostic checks before larger runs.
- `scripts/tune_random_forest.py` and `scripts/tune_xgboost.py`: reproducible Puhti-oriented tuning entrypoints for the final random forest and XGBoost searches.

Across the training notebooks, the main reported validation metric is MAE, with supporting checks such as MSE, R-squared, and MAPE.

## Results

The table below summarizes the best documented validation result currently available in the repository. All values are from the grouped validation split in `datasets/splits/validation_grouped.csv` with 1,689 rows.

| Model | Selected feature set | Raw columns | Validation MAE | Validation RMSE | Validation R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear regression | trusted recommended features | 66 | 42.3640 | 153.2220 | 0.9270 |
| Random forest | trusted recommended features without listing dates | 66 | 18.1299 | 48.5480 | 0.9927 |
| XGBoost | trusted recommended features without date offsets without `oem_number` | 65 | 20.4223 | 48.9317 | 0.9926 |
| CatBoost | trusted recommended features without date offsets | 66 | 46.0379 | 95.7289 | 0.9715 |

The Puhti-based scripted tuning kept the random-forest configuration as the best documented validation result in the repository, while the widened XGBoost search closed the gap substantially and reached a competitive validation score.

## Final test result

After model selection was completed on the grouped training and validation splits, the selected random-forest configuration was retrained on `datasets/splits/train_grouped.csv` plus `datasets/splits/validation_grouped.csv` and evaluated once on the held-out `datasets/splits/test_grouped.csv` split.

| Model | Selected feature set | Test MAE | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: |
| Random forest | trusted recommended features without listing dates | 22.4695 | 62.6210 | 0.9903 |

The final held-out test result suggests that the selected random-forest model performs consistently on the available grouped split, while showing a moderate drop compared with validation performance. It should be interpreted as evidence for a proof-of-concept listing-price estimation tool, not as a guarantee of production performance.

## Current implementation status

The repository is beyond model-training-only status. It currently contains:

- a final saved random-forest deployment bundle under `artifacts/random_forest_final/full_data_bundle`
- a working Streamlit decision-support prototype in `app/streamlit_app.py`, supported by `app/ui_helpers.py` and `app/shap_utils.py`
- a working FastAPI serving layer in `app/fastapi_app.py`
- serving helpers for bundle loading and prediction in `src/random_forest_serving.py`
- automated tests covering Streamlit helper logic, serving logic, and FastAPI behavior under `tests/`

The current product framing is a **proof-of-concept decision-support tool**, not a production-ready automated pricing system.

## Running the demo apps

Run the Streamlit prototype:

```bash
streamlit run app/streamlit_app.py
```

Run the FastAPI app locally:

```bash
uvicorn app.fastapi_app:app --reload
```

The saved model bundle is expected at `artifacts/random_forest_final/full_data_bundle` unless a different path is provided for the FastAPI service through `MODEL_BUNDLE_DIR`.

## Deploying the tool

The repository currently supports two practical deployment shapes.

### 1. Streamlit demo deployment

Use this when the goal is to present the proof-of-concept as an interactive decision-support tool.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install firefox
streamlit run app/streamlit_app.py
```

Notes:

- The app expects the saved model bundle at `artifacts/random_forest_final/full_data_bundle`.
- The UI uses the random-forest point estimate plus a comparable listing-price range from `reference_rows.csv`.
- The `Why this price?` block computes a live local SHAP explanation for the submitted row.
- This is the simplest deployment option for demos, thesis presentations, and supervisor review.

### 2. FastAPI model service deployment

Use this when the goal is to expose the model as a backend API.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000
```

Optional bundle override:

```bash
MODEL_BUNDLE_DIR=/absolute/path/to/full_data_bundle uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000
```

The main endpoints are:

- `GET /health`
- `GET /model-info`
- `POST /predict`

Example request:

```json
{
  "rows": [
    {
      "part_name": "Brake Caliper",
      "quality_grade": "A1",
      "oem_number": "FI02042722A",
      "mileage": 224000,
      "brand": "toyota",
      "model": "corolla",
      "category": "brakes",
      "subcategory": "right rear",
      "year_start": 2019,
      "year_end": 2027,
      "repair_status": "original_valid",
      "brand_is_known_model_family": 1,
      "model_total_registered": 0,
      "model_median_vehicle_age": 0,
      "model_mean_vehicle_age": 0,
      "model_median_mileage": 0,
      "model_mean_mileage": 0,
      "model_median_engine_cc": 0,
      "model_median_power_kw": 0,
      "model_median_mass_kg": 0,
      "model_share_of_market": 0,
      "model_share_within_brand": 0,
      "model_share_over_10y": 0,
      "model_share_over_200k_km": 0,
      "model_automatic_share": 0,
      "model_petrol_share": 0,
      "model_diesel_share": 0,
      "model_ev_share": 0,
      "model_hybrid_share": 0,
      "model_turbo_share": 0,
      "model_firstreg_total_2014_2026": 0,
      "model_firstreg_year_span": 0,
      "model_firstreg_peak_year": 0,
      "model_firstreg_peak_count": 0,
      "model_firstreg_recent_share": 0,
      "model_firstreg_old_share": 0,
      "model_firstreg_weighted_year": 0,
      "brand_total_registered": 0,
      "brand_median_vehicle_age": 0,
      "brand_mean_vehicle_age": 0,
      "brand_median_mileage": 0,
      "brand_mean_mileage": 0,
      "brand_median_engine_cc": 0,
      "brand_median_power_kw": 0,
      "brand_median_mass_kg": 0,
      "brand_share_of_market": 0,
      "brand_share_over_10y": 0,
      "brand_share_over_200k_km": 0,
      "brand_automatic_share": 0,
      "brand_petrol_share": 0,
      "brand_diesel_share": 0,
      "brand_ev_share": 0,
      "brand_hybrid_share": 0,
      "brand_turbo_share": 0,
      "brand_firstreg_total_2014_2026": 0,
      "brand_firstreg_year_span": 0,
      "brand_firstreg_peak_year": 0,
      "brand_firstreg_peak_count": 0,
      "brand_firstreg_recent_share": 0,
      "brand_firstreg_old_share": 0,
      "brand_firstreg_weighted_year": 0,
      "mileage_missing_flag": 0,
      "observations_so_far": 1,
      "days_since_first_seen_so_far": 0
    }
  ]
}
```

### Deployment recommendation for this thesis

For thesis/demo use, deploy the Streamlit app as the presentation layer and treat the FastAPI app as the backend/service layer. The current repository is best understood as a proof-of-concept deployment target, not a production-hardened system.

## Current behavior of the demo UI

- The main point estimate comes from the final random-forest model.
- The displayed comparable range is based on historical reference-row listing prices for similar cases in `reference_rows.csv`.
- The UI is intentionally simplified for operator-facing use and proof-of-concept demonstration.
- The `Why this price?` section uses local SHAP values to explain the submitted prediction.

## Scope and limitations

- The prediction target is the observed marketplace listing price / asking price, not an independently verified transaction price or true market value.
- The dataset is based on the crawler coverage and processed snapshots available in this repository.
- Repeated listings are intentionally preserved for listing-history features, but this requires grouped splitting to avoid leakage across train, validation, and test data.
- The model can reflect biases, sparsity, and taxonomy inconsistencies in the source listings.
- `subcategory` should be treated as a mixed taxonomy/location field rather than a fully standardized hierarchy.
- The Streamlit and FastAPI apps are proof-of-concept interfaces and are not production-hardened.
- SHAP explanations are useful for model interpretation, but they describe model behavior rather than causal effects.

## Remaining work

Based on the current repository state, useful follow-up work includes:

- continue validating and refining the SHAP explanation layer for final thesis reporting,
- improve UI label/display normalization for presentation quality,
- expand or refresh crawler coverage if the dataset needs more brands, models, or observations per subcategory,
- add stronger deployment hardening if the tool is later moved beyond thesis/demo use.

## Notes on dataset behavior

- The same marketplace listing can appear on multiple scrape dates.
- Those repeated observations are intentionally preserved for listing-history features.
- Split creation is group-based rather than row-based to avoid leakage across repeated listings.
- `subcategory` should be treated as a mixed taxonomy/location field rather than a fully standardized hierarchy.

## License

This project is licensed under the MIT License. See `LICENSE`.
