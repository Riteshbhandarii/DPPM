#!/bin/bash -l
#SBATCH --job-name=dppm-xgb-strict
#SBATCH --account=project_2017273
#SBATCH --partition=gpu
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=dppm_xgb_strict_%j.out
#SBATCH --error=dppm_xgb_strict_%j.err

set -u
set -o pipefail

module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd /scratch/project_2017273/DPPM

echo "HOST: $(hostname)"
echo "TIME: $(date)"
python3 -V

python3 -u scripts/tune_xgboost_strict.py \
  --data-path datasets/splits/train_grouped.csv \
  --output-dir artifacts/xgboost_tuning_strict \
  --xgboost-device cuda \
  --cv-splits 4 \
  --random-trials 72 \
  --refinement-trials 24 \
  --top-k-finalists 12 \
  --random-seed 42
