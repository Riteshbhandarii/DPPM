#!/bin/bash
#SBATCH --job-name=dppm_scraper
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=scraper_%j.log
#SBATCH --error=scraper_%j.err

module load python-data

pip install --user playwright pandas beautifulsoup4 lxml
python -m playwright install chromium

echo "Started: $(date)"

srun --ntasks=1 --cpus-per-task=2 --exclusive \
    python3 src/crawler.py --brand Toyota --model Corolla &

srun --ntasks=1 --cpus-per-task=2 --exclusive \
    python3 src/crawler.py --brand Skoda --model Octavia &

srun --ntasks=1 --cpus-per-task=2 --exclusive \
    python3 src/crawler.py --brand VW --model "Golf,-e_Golf" &

wait

echo "Completed: $(date)"
ls -lh dppm_*.csv
