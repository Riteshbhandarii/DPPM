#!/bin/bash
#SBATCH --account=project_200XXXX  # Your Puhti project
#SBATCH --partition=small
#SBATCH --array=1-3
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --output=scraper_%A_%a.out
#SBATCH --error=scraper_%A_%a.err

module purge
module load python-data

BRAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt | cut -d' ' -f1)
MODEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt | cut -d' ' -f2-)

echo "Starting scraper for ${BRAND} ${MODEL} at $(date)"
python3 src/crawler.py --brand "${BRAND}" --model "${MODEL}"
echo "Finished scraping ${BRAND} ${MODEL} at $(date)"
