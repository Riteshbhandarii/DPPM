#!/bin/bash -l
#SBATCH --job-name=crawler
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=crawler_%j.out
#SBATCH --error=crawler_%j.err

# SLURM job to run the crawler sequentially for multiple brand/model pairs.
# Uses a login shell (-l) so the module system is available on Puhti.

set -u
set -o pipefail

# Basic job metadata for logs
echo "JOB STARTED"
echo "HOST: $(hostname)"
echo "TIME: $(date)"

# Load a modern Python via python-data (required for zoneinfo, pandas, etc.)
# The extra module path is needed for Tykky-provided modules on Puhti.
module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

# Move to project root so package imports work correctly
cd ~/DPPM

# Sanity checks (useful for debugging SLURM jobs)
echo "PWD:    $(pwd)"
echo "PYTHON: $(which python3)"
python3 -V
python3 -c "from zoneinfo import ZoneInfo; print('zoneinfo OK')"

# Ensure package markers exist so relative imports work with python -m
[ -f crawler/__init__.py ] || touch crawler/__init__.py
[ -f crawler/src/__init__.py ] || touch crawler/src/__init__.py

# Run the crawler for a single brand/model pair
run_one () {
  local BRAND="$1"
  local MODEL="$2"
  echo "START: ${BRAND} | ${MODEL}"
  echo "TIME:  $(date)"
  python3 -m crawler.src.crawler --brand "${BRAND}" --model "${MODEL}"
  local RC=$?
  echo "END:   ${BRAND} | ${MODEL} (exit=${RC})"
  echo "TIME:  $(date)"
  return $RC
}

# Run sequentially; continue even if one run fails
FAIL=0
run_one "Toyota" "Corolla" || FAIL=1
run_one "Skoda" "Octavia" || FAIL=1
run_one "VW" "Golf,-e_Golf" || FAIL=1

# Final job status
echo "ALL DONE. FAIL=${FAIL}"
echo "TIME: $(date)"
exit $FAIL
