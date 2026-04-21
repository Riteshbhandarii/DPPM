#!/bin/bash -l
#SBATCH --job-name=dppm-rf-prune
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=dppm_rf_prune_%j.out
#SBATCH --error=dppm_rf_prune_%j.err

set -u
set -o pipefail

module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd /scratch/project_2017273/DPPM

echo "HOST: $(hostname)"
echo "TIME: $(date)"
python3 -V

python3 -u scripts/prune_random_forest.py \
  --train-path datasets/splits/train_grouped.csv \
  --validation-path datasets/splits/validation_grouped.csv \
  --output-dir artifacts/random_forest_final
