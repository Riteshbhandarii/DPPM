#!/bin/bash
#SBATCH --job-name=rf_shap
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/rf_shap_%j.out
#SBATCH --error=logs/rf_shap_%j.err

set -euo pipefail

cd /scratch/project_2017273/DPPM
mkdir -p logs

source .venv_catboost/bin/activate

python scripts/analyze_final_rf_shap_global.py
python scripts/analyze_final_rf_shap_examples.py
