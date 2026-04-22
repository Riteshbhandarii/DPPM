#!/bin/bash
#SBATCH --job-name=r2_quick
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/r2_quick_%j.out
#SBATCH --error=logs/r2_quick_%j.err

set -euo pipefail

cd /scratch/project_2017273/DPPM
mkdir -p logs

module load python-data

python scripts/audit_r2_credibility.py --quick
