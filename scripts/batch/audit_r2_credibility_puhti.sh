#!/bin/bash
#SBATCH --job-name=r2_audit
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/r2_audit_%j.out
#SBATCH --error=logs/r2_audit_%j.err

set -euo pipefail

cd /scratch/project_2017273/DPPM
mkdir -p logs

module load python-data

python scripts/audit_r2_credibility.py
