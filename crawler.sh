#!/bin/bash -l
#SBATCH --job-name=crawler
#SBATCH --account=project_2017273
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=crawler_%j.out
#SBATCH --error=crawler_%j.err

set -u
set -o pipefail

echo "JOB STARTED"
echo "HOST: $(hostname)"
echo "TIME: $(date)"

# Make sure module can see python-data
module --force purge
module use --append /appl/soft/ai/tykky/modulefiles
module load python-data

cd ~/DPPM

echo "PWD:    $(pwd)"
echo "PYTHON: $(which python3)"
python3 -V
python3 -c "from zoneinfo import ZoneInfo; print('zoneinfo OK')"

# Ensure package structure exists (required for python -m ... with relative imports)
[ -f crawler/__init__.py ] || touch crawler/__init__.py
[ -f crawler/src/__init__.py ] || touch crawler/src/__init__.py

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

FAIL=0
run_one "Toyota" "Corolla" || FAIL=1
run_one "Skoda" "Octavia" || FAIL=1
run_one "VW" "Golf,-e_Golf" || FAIL=1

echo "ALL DONE. FAIL=${FAIL}"
echo "TIME: $(date)"
exit $FAIL
