#!/bin/bash
#SBATCH --job-name=pi_xgb
#SBATCH --account=project_2017273
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/pi_xgb_%j.out
#SBATCH --error=logs/pi_xgb_%j.err

set -euo pipefail

cd /scratch/project_2017273/DPPM
mkdir -p logs

module load python-data

python scripts/evaluate_xgboost_part_identity.py --xgboost-device cuda
