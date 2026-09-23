# Development

Setup, repository layout, the data files that matter, and how CI is set up.

## Setup

Python 3.12 (`.python-version`), dependencies pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py        # demo UI
uvicorn app.fastapi_app:app --reload      # API
make verify PYTHON=.venv/bin/python       # split and test-set result still reproduce (read-only)
make test PYTHON=.venv/bin/python
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `crawler/` | Playwright crawler for marketplace snapshots |
| `notebooks/` | Preprocessing, integration, analysis and training notebooks |
| `datasets/` | Cleaned, merged, split and registry-derived CSV data |
| `scripts/` | Tuning, evaluation, analysis and export scripts |
| `artifacts/` | Model artifacts, tuning outputs, holdout and SHAP outputs |
| `results/` | September live validation: chosen listings and figures |
| `app/` | Streamlit and FastAPI prototypes |
| `src/` | Shared modelling and serving code |
| `tests/` | Regression tests |
| `docs/` | Project documentation |

## Data files

| File | Rows | Note |
| --- | ---: | --- |
| `datasets/cleaned/clean_master_dataset.csv` | 11,321 | Modelling dataset, frozen |
| `datasets/splits_strict/train_strict.csv` | 7,930 | Connected-component split, seed 32 |
| `datasets/splits_strict/validation_strict.csv` | 1,695 | |
| `datasets/splits_strict/test_strict.csv` | 1,696 | Used once on 2026-07-10; never score a model on it again |
| `datasets/splits/*_grouped.csv` | 7,954 / 1,689 / 1,678 | Historical product-id split, optimistic baseline only |

The connected-component split keeps every connected component (rows linked by the same
`product_id` or the same `part_name + brand + model + year_start + year_end`)
in one split. Provenance and leakage checks:
`datasets/splits_strict/strict_split_summary.json`. Which files are frozen and
how each is produced: [PIPELINE.md](PIPELINE.md).

## Model roles

| Role | Purpose |
| --- | --- |
| Reported model | Random Forest selected under component-grouped CV |
| Operational model | Context-rich model used by the demo interface |
| Conservative variant | Listing-history and time features removed, to test sensitivity |

## Explainability

Only `artifacts/strict_final_shap/` (from `scripts/run_strict_shap.py`) explains
the reported model. `artifacts/final_model_shap/`,
`artifacts/final_model_shap_conservative/` and `artifacts/random_forest_shap/`
date from April 2026, before the connected-component split, and explain a different model.
They are kept as history.

## Keeping the repository clean

Datasets, split files, selection summaries, SHAP outputs and evaluation
artifacts are the evidence trail. Do not delete them as routine cleanup.

Ignored instead: Python caches, local virtual environments, Playwright runtime
files and `node_modules/`, and raw source data too large for the repository
(Traficom downloads, raw crawls, the September saved pages).

## CI

CI installs `requirements.txt`, imports the core modules and runs `pytest` on
every pull request. It does not run the crawler, notebooks, training, SHAP or
artifact-generation scripts. Tests run without setting `PYTHONPATH`.
