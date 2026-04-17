#!/bin/bash -l
#SBATCH --job-name=dppm-rf-test
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=dppm_rf_test_%j.out
#SBATCH --error=dppm_rf_test_%j.err

set -u
set -o pipefail

module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd /scratch/project_2017273/DPPM

echo "HOST: $(hostname)"
echo "TIME: $(date)"
python3 -V

python3 -u scripts/evaluate_random_forest_test.py \
  --train-path datasets/splits/train_grouped.csv \
  --validation-path datasets/splits/validation_grouped.csv \
  --test-path datasets/splits/test_grouped.csv \
  --tuning-summary-path artifacts/random_forest_tuning/best_tuning_summary.json \
  --output-dir artifacts/random_forest_test
