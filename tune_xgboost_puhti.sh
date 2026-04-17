#!/bin/bash -l
#SBATCH --job-name=dppm-xgb-tune
#SBATCH --account=project_2017273
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=dppm_xgb_tune_%j.out
#SBATCH --error=dppm_xgb_tune_%j.err

set -u
set -o pipefail

module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd /scratch/project_2017273/DPPM

echo "HOST: $(hostname)"
echo "TIME: $(date)"
python3 -V

python3 scripts/tune_xgboost.py \
  --train-path datasets/splits/train_grouped.csv \
  --validation-path datasets/splits/validation_grouped.csv \
  --output-dir artifacts/xgboost_tuning \
  --xgboost-device cuda \
  --cv-splits 4
