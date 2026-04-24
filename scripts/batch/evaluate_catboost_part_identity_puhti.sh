#!/bin/bash
#SBATCH --job-name=pi_cat
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/pi_cat_%j.out
#SBATCH --error=logs/pi_cat_%j.err

set -euo pipefail

cd /scratch/project_2017273/DPPM
mkdir -p logs

source .venv_catboost/bin/activate

python scripts/evaluate_catboost_part_identity.py
