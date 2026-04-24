#!/bin/bash -l
#SBATCH --job-name=xgb-robust
#SBATCH --account=project_2017273
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/xgb_robust_%j.out
#SBATCH --error=logs/xgb_robust_%j.err

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
  --model xgboost \
  --data-path datasets/splits/train_grouped.csv \
  --xgb-summary-path artifacts/xgboost_tuning_strict/best_tuning_summary.json \
  --xgboost-device cuda \
  --output-dir artifacts/robustness_checks/strict_models
