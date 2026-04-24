#!/bin/bash -l
#SBATCH --job-name=rf-shap-cons
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/rf_shap_cons_%j.out
#SBATCH --error=logs/rf_shap_cons_%j.err

set -euo pipefail

cd /scratch/project_2017273/DPPM
mkdir -p logs

source .venv_catboost/bin/activate

python scripts/analyze_final_rf_shap_global.py \
  --output-dir artifacts/final_model_shap_conservative \
  --drop-feature observations_so_far \
  --drop-feature days_since_first_seen_so_far \
  --drop-feature first_seen_day_offset

python scripts/analyze_final_rf_shap_examples.py \
  --output-dir artifacts/final_model_shap_conservative \
  --drop-feature observations_so_far \
  --drop-feature days_since_first_seen_so_far \
  --drop-feature first_seen_day_offset
