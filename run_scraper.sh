#!/bin/bash
#SBATCH --job-name=dppm_scraper
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=1-3
#SBATCH --output=scraper_%a.out
#SBATCH --error=scraper_%a.err

module purge
module load python-data

# Read brand and model from models.txt based on array index
BRAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt | cut -d',' -f1)
MODEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt | cut -d',' -f2)

echo "Starting scraper for $BRAND $MODEL at $(date)"

# Run the scraper - ADD QUOTES around $MODEL!
python src/crawler.py --brand "$BRAND" --model "$MODEL"

echo "Finished scraping $BRAND $MODEL at $(date)"
