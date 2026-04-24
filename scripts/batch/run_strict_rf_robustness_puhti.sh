#!/bin/bash -l
#SBATCH --job-name=rf-robust
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/rf_robust_%j.out
#SBATCH --error=logs/rf_robust_%j.err

set -euo pipefail

module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd /scratch/project_2017273/DPPM
mkdir -p logs

echo "HOST: $(hostname)"
echo "TIME: $(date)"
python3 -V

python3 -u scripts/run_strict_robustness_check.py \
  --model random_forest \
  --data-path datasets/splits/train_grouped.csv \
  --rf-summary-path artifacts/random_forest_tuning_strict/best_tuning_summary.json \
  --output-dir artifacts/robustness_checks/strict_models
