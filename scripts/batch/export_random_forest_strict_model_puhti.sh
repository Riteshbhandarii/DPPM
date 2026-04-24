#!/bin/bash -l
#SBATCH --job-name=dppm-rf-strict-export
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=dppm_rf_strict_export_%j.out
#SBATCH --error=dppm_rf_strict_export_%j.err

set -euo pipefail

module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd /scratch/project_2017273/DPPM

echo "HOST: $(hostname)"
echo "TIME: $(date)"
python3 -V

python3 -u scripts/export_random_forest_model.py \
  --train-path datasets/splits/train_grouped.csv \
  --validation-path datasets/splits/validation_grouped.csv \
  --test-path datasets/splits/test_grouped.csv \
  --tuning-summary-path artifacts/random_forest_tuning_strict/best_tuning_summary.json \
  --evaluation-summary-path artifacts/final_model_behavior/overall_metrics.json \
  --output-dir artifacts/random_forest_final_strict
