sacct -j 34049731 --format=JobID,State,Elapsed,Start,End
#!/bin/bash -l
#SBATCH --job-name=dppm-rf-tune
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=dppm_rf_tune_%j.out
#SBATCH --error=dppm_rf_tune_%j.err

set -u
set -o pipefail

module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd /scratch/project_2017273/DPPM

echo "HOST: $(hostname)"
echo "TIME: $(date)"
python3 -V

python3 -u scripts/tune_random_forest.py \
  --train-path datasets/splits/train_grouped.csv \
  --validation-path datasets/splits/validation_grouped.csv \
  --output-dir artifacts/random_forest_tuning \
  --cv-splits 4 \
  --random-trials 24 \
  --top-k-finalists 8 \
  --random-seed 42
