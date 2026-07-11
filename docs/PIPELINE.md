# Reproducible pipeline — run order & artifacts

End-to-end order from raw data to the frozen thesis results, with the artifact each stage emits
and which artifacts are **frozen** (source-of-truth, never regenerate). For design rationale see
[ARCHITECTURE.md](ARCHITECTURE.md); for the modelling decisions see
[STRICT_MODEL_COMPARISON.md](STRICT_MODEL_COMPARISON.md).

> **Folder numbers are not the run order.** `01_preprocessing/02–04` depend on `02_integration/01`.
> Follow the data dependencies below, not the directory names.
>
> Run everything from the repo root with the project venv (`.venv/bin/python`). All notebooks
> resolve the repo root themselves (`PROJECT_ROOT` bootstrap), so they no longer contain absolute
> paths and can run from any working directory.

## Stage graph (input → output)

| # | Stage | Entry point | Reads | Writes |
|---|-------|-------------|-------|--------|
| 0 | Crawl marketplace | `crawler/` (Playwright) | Varaosahaku.fi | `crawler/crawler_datasets/*.csv` |
| 0 | Traficom open data | *(manual download)* | Traficom | `datasets/traficom_datasets/*.csv` *(gitignored)* |
| 1 | Consolidate listings per model | `notebooks/02_integration/01_loading_and_merging.ipynb` | crawler CSVs | `datasets/merged/dppm_{vw_golf_e_golf,skoda_octavia,toyota_corolla_yaris}.csv` |
| 2 | Registry summaries | `notebooks/01_preprocessing/01_preprocess_traficom.ipynb` | `datasets/traficom_datasets/*` | `datasets/traficom_outputs/{brand,model}_summary.csv`, `{brand,model}_firstreg_summary.csv` |
| 3 | Clean listings per model | `notebooks/01_preprocessing/{02,03,04}_preprocess_*.ipynb` | `datasets/merged/dppm_*.csv` | `datasets/cleaned/{toyota,VW,skoda}_cleaned.csv` |
| 4 | Integrate listings | `notebooks/02_integration/02_dataset_integration.ipynb` | the 3 cleaned per-model files | `datasets/cleaned/cleaned_and_merged_pricedataset.csv` |
| 5 | Join registry | `notebooks/02_integration/03_final_dataset_merging.ipynb` | integrated listings + registry summaries | `datasets/merged/price_traficom_merged.csv` |
| 6 | Final clean → **MASTER** | `notebooks/02_integration/04_dataset_cleaning.ipynb` | `price_traficom_merged.csv` | 🧊 `datasets/cleaned/clean_master_dataset.csv` (11,321 rows) |
| 7 | Strict split | `scripts/generate_strict_split.py` | `clean_master_dataset.csv` | 🧊 `datasets/splits_strict/{train,validation,test}_strict.csv` (7,930 / 1,695 / 1,696; seed 32) |
| 8 | Model selection | `scripts/run_strict_model_tuning.py` | strict train/validation | 🧊 `artifacts/strict_model_tuning/` (RF winner + Ridge finalist) |
| 9 | **Final holdout — RUN ONCE** | `notebooks/05_strict_training/03_strict_final_holdout.ipynb` | frozen split + frozen selection | 🧊 `artifacts/strict_final_holdout/` (MAE 69.46 €) |
| 10 | Descriptive analyses *(holdout-safe)* | see below | frozen split/artifacts (read-only) | `artifacts/{strict_final_shap,registry_ablation,learning_curve,…}/` |

🧊 = **frozen source-of-truth. Do not regenerate.**

### Stage 10 — descriptive analyses (holdout-safe, CV/read-only)
All run from the frozen split and never re-fit selection or touch the holdout as a selection signal:

- `scripts/run_strict_shap.py` → `artifacts/strict_final_shap/` — SHAP attribution (subcategory ≈ 66%).
- `scripts/holdout_baseline_comparison.py` — RF vs subcategory-median dummy on the holdout (descriptive).
- `scripts/registry_ablation.py` → `artifacts/registry_ablation/` — marginal value of the Traficom block (≤1%).
- `scripts/learning_curve.py` → `artifacts/learning_curve/` — data-limited vs signal-limited (component-subsampled CV).

## Frozen artifacts — never regenerate

The thesis cites these exact files; re-running the stages that produce them would break provenance
(and the holdout can **never** run again):

- `datasets/cleaned/clean_master_dataset.csv` — the training data all models were fit on.
- `datasets/splits_strict/*` — the strict connected-component split (seed 32).
- `artifacts/strict_model_tuning/*` — the frozen selection (RF winner, pre-registered).
- `artifacts/strict_final_holdout/*` — the one-shot holdout (guarded by `I_CONFIRM_SELECTION_IS_FROZEN`).

## Reproducing the *results* from the frozen master (the part that matters)

You do **not** need to re-run stages 0–6. Starting from the committed `clean_master_dataset.csv`:

```bash
.venv/bin/python scripts/generate_strict_split.py        # -> datasets/splits_strict/ (byte-identical, seed 32)
.venv/bin/python scripts/run_strict_model_tuning.py      # -> artifacts/strict_model_tuning/ (RF winner)
# Stage 9 holdout is FROZEN — do not re-run; read artifacts/strict_final_holdout/.
.venv/bin/python scripts/run_strict_shap.py              # descriptive
.venv/bin/python scripts/registry_ablation.py            # descriptive
.venv/bin/python scripts/learning_curve.py               # descriptive
```

Each results script re-derives the frozen selection number as a self-check (e.g. the ablation and
learning-curve scripts assert the RF with-registry CV MAE reproduces **105.334 €** before reporting).

## Data-prep stages (0–6) — provenance, not for re-running

The preprocessing/integration notebooks are retained so the master's derivation is auditable, and
they are now path-portable. They are **not** wired for one-command re-execution and should not be
re-run to regenerate frozen inputs (`clean_master_dataset.csv` is the source of truth). Converting
them into consolidated, verified scripts is tracked as a **future** cleanup (issue #47, deferred).
