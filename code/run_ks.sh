#!/usr/bin/env bash
# Linux: bash run_ks.sh 0 33      (Windows computer: .\RUN_KS.ps1 -First 34 -Last 50)
FIRST=${1:-0}; LAST=${2:-33}; TOTAL=${3:-51}; PY=${PYTHON:-python3}
cd "$(dirname "$0")"; export HMM6_REPO="$PWD" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
for w in $(seq "$FIRST" "$LAST"); do nohup "$PY" -I ks_rerun.py "$w" "$TOTAL" "ks_w$w.csv" > "ks_w$w.log" 2>&1 & done
echo "Started $FIRST..$LAST. Running: pgrep -fc ks_rerun.py   Progress: grep -h ',sym,' ks_w*.csv | wc -l"
