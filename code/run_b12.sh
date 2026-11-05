#!/usr/bin/env bash
# Paper B, items B1 + B2 — Linux launcher. Run from this folder.
#   2-Xeon Linux computer:  bash run_b12.sh 0 33
#   (the Windows computer keeps using RUN_B12.ps1 -First 34 -Last 50)
# Workers run in the background and keep running if the terminal is closed.
FIRST=${1:-0}; LAST=${2:-33}; TOTAL=${3:-51}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
export HMM6_REPO="$PWD" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
for w in $(seq "$FIRST" "$LAST"); do
  nohup "$PY" -I b12_run.py "$w" "$TOTAL" "b12_w$w.csv" > "b12_w$w.log" 2>&1 &
done
echo "Started workers $FIRST..$LAST of $TOTAL."
echo "Progress:  grep -h ',sym,' b12_w*.csv | wc -l"
echo "Running:   pgrep -fc b12_run.py"
echo "Errors:    grep -l Error b12_w*.log"
