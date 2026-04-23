#!/bin/bash -l
#SBATCH --job-name=dppm-rf-strict
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=dppm_rf_strict_%j.out
#SBATCH --error=dppm_rf_strict_%j.err

set -u
set -o pipefail

module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd /scratch/project_2017273/DPPM

echo "HOST: $(hostname)"
echo "TIME: $(date)"
python3 -V

python3 -u scripts/tune_random_forest_strict.py \
  --data-path datasets/splits/train_grouped.csv \
  --output-dir artifacts/random_forest_tuning_strict \
  --cv-splits 5 \
  --random-trials 72 \
  --refinement-trials 24 \
  --top-k-finalists 12 \
  --random-seed 42
