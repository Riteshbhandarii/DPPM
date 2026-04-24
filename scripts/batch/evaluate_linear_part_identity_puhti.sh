#!/bin/bash
#SBATCH --job-name=pi_linear
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/pi_linear_%j.out
#SBATCH --error=logs/pi_linear_%j.err

set -euo pipefail

cd /scratch/project_2017273/DPPM
mkdir -p logs

source .venv_catboost/bin/activate

python scripts/evaluate_linear_part_identity.py
