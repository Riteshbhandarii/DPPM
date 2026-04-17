# DPPM

DPPM stands for **Dismantler Price Prediction Model**. This repository contains a thesis-oriented workflow for predicting used car spare-part prices from dismantler marketplace data.

The project combines:

- scraped spare-part listings from Varaosahaku.fi
- Finnish Traficom M1 passenger-car registry summaries
- notebook-based cleaning, integration, analysis, and model training

The current repository includes a working crawler, intermediate datasets, final grouped train/validation/test splits, and regression experiments with linear regression, random forest, XGBoost, and CatBoost.

## Repository structure

- `crawler/`: Playwright-based crawler package for collecting marketplace listings.
- `crawler/crawler_datasets/`: archived crawler outputs; new runs are written under `crawler/crawler_datasets/new/`.
- `datasets/traficom_outputs/`: cleaned Traficom summary tables used for enrichment.
- `datasets/merged/`: merged listing snapshots and listing-plus-Traficom outputs.
- `datasets/cleaned/`: cleaned modeling datasets, including the final master dataset.
- `datasets/splits/`: leakage-safe grouped train/validation/test splits.
- `notebooks/01_preprocessing/`: Traficom preprocessing and per-model source cleaning.
- `notebooks/02_integration/`: snapshot merging, dataset integration, cleaning, and split creation.
- `notebooks/03_analysis/`: exploratory analysis of the prepared dataset.
- `notebooks/04_training/`: model training and validation notebooks.
- `scripts/batch/`: Puhti-oriented SLURM batch scripts for crawling, tuning, testing, and model export.

## Current tracked data artifacts

The repository already contains processed outputs, so you can inspect the full pipeline without rerunning everything.

- `datasets/cleaned/clean_master_dataset.csv`: final cleaned modeling dataset with 11,321 rows.
- `datasets/splits/train_grouped.csv`: grouped training split with 7,954 rows.
- `datasets/splits/validation_grouped.csv`: grouped validation split with 1,689 rows.
- `datasets/splits/test_grouped.csv`: grouped test split with 1,678 rows.
- `datasets/traficom_outputs/brand_summary.csv` and `datasets/traficom_outputs/model_summary.csv`: brand-level and model-level registry summaries.
- `datasets/traficom_outputs/brand_firstreg_summary.csv` and `datasets/traficom_outputs/model_firstreg_summary.csv`: registry lifecycle summaries based on first-registration history.

## Project workflow

The notebooks are organized as a sequential pipeline:

1. **Preprocess source data**
   - Build Traficom market summary tables in `notebooks/01_preprocessing/01_preprocess_traficom.ipynb`.
   - Clean model-specific listing exports in the other preprocessing notebooks.
2. **Merge snapshot files**
   - Combine repeated crawler exports by brand/model in `notebooks/02_integration/01_loading_and_merging.ipynb`.
3. **Integrate registry features**
   - Merge listing data with Traficom summary tables in the integration notebooks.
4. **Clean the master dataset**
   - Preserve repeated listings across scrape dates.
   - Remove only clear same-day duplicates.
   - Add conservative formatting fixes, mileage imputation, and listing-history features.
5. **Create leakage-safe grouped splits**
   - Keep all observations from the same listing group in exactly one split.
   - Save split files under `datasets/splits/`.
6. **Train and compare models**
   - Evaluate baseline listing-only features, listing plus Traficom features, and broader recommended feature sets.

## Environment setup

This repository does not currently include a pinned `requirements.txt` or `pyproject.toml`, so dependencies need to be installed manually.

The codebase uses:

- Python
- Jupyter notebooks
- pandas and NumPy
- matplotlib and seaborn
- scikit-learn
- XGBoost
- CatBoost
- Beautiful Soup
- Playwright

One reasonable local setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost beautifulsoup4 playwright jupyter ipython
playwright install firefox
```

## Running the crawler

From the repository root:

```bash
python3 -m crawler --brand Toyota --model Corolla
```

The crawler:

- launches Firefox through Playwright
- writes output CSVs to `crawler/crawler_datasets/new/`
- stamps results with the scrape date
- currently limits scraping to `5` parts per subcategory for stability

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
- `scripts/tune_random_forest.py` and `scripts/tune_xgboost.py`: reproducible Puhti-oriented tuning entrypoints for the final random forest and XGBoost searches.

Across the training notebooks, the main reported validation metric is MAE, with supporting checks such as MSE, R-squared, and MAPE.

## Results

The table below summarizes the best **trusted validation result** currently available in the repository. All values are from the grouped validation split in `datasets/splits/validation_grouped.csv` with 1,689 rows.

| Model | Selected feature set | Raw columns | Validation MAE | Validation RMSE | Validation R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear regression | trusted recommended features | 66 | 42.3640 | 153.2220 | 0.9270 |
| Random forest | trusted recommended features without listing dates | 66 | 18.1299 | 48.5480 | 0.9927 |
| XGBoost | trusted recommended features without date offsets without `oem_number` | 65 | 20.4223 | 48.9317 | 0.9926 |
| CatBoost | trusted recommended features without date offsets | 66 | 46.0379 | 95.7289 | 0.9715 |

The latest Puhti-based scripted tuning kept the random forest configuration as the strongest trusted validation result in the repository, while the widened XGBoost search closed the gap substantially and reached a much more competitive validation score.

## Final test result

After model selection was completed on the grouped training and validation splits, the selected random forest configuration was retrained on `datasets/splits/train_grouped.csv` plus `datasets/splits/validation_grouped.csv` and evaluated once on the held-out `datasets/splits/test_grouped.csv` split.

| Model | Selected feature set | Test MAE | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: |
| Random forest | trusted recommended features without listing dates | 22.4695 | 62.6210 | 0.9903 |

The final held-out test result indicates that the selected random forest model remains strong as a spare-part price estimation and decision-support tool, while showing a moderate generalization drop compared with the validation split.

## Remaining work

Based on the current repo state, the main work still to be done is:

- Freeze the environment in a `requirements.txt` or `pyproject.toml`, since setup is currently manual.
- Save the final preprocessing and model artifact in a deployment-friendly format.
- Build the planned FastAPI and Streamlit layers around the selected final model.
- Expand or refresh crawler coverage if the dataset needs more brands, models, or observations per subcategory.

## Notes on dataset behavior

- The same marketplace listing can appear on multiple scrape dates.
- Those repeated observations are intentionally preserved for listing-history features.
- Split creation is group-based rather than row-based to avoid leakage across repeated listings.
- `subcategory` should be treated as a mixed taxonomy/location field rather than a fully standardized hierarchy.

## License

This project is licensed under the MIT License. See `LICENSE`.
