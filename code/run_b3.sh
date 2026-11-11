#!/usr/bin/env bash
# Linux (2 Xeons): bash run_b3.sh 0 33        Windows (1 Xeon): .\RUN_B3.ps1 -First 34 -Last 50
FIRST=${1:-0}; LAST=${2:-33}; TOTAL=${3:-51}; PY=${PYTHON:-python3}
cd "$(dirname "$0")"; export HMM6_REPO="$PWD" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$PY" -c "import numba" 2>/dev/null && echo "numba found (fast HMM-6 fits)" || echo "WARNING: numba not found; HMM-6 fits will be ~10x slower (pip install numba)"
for w in $(seq "$FIRST" "$LAST"); do nohup "$PY" -I b3_var.py "$w" "$TOTAL" "b3_w$w.csv" > "b3_w$w.log" 2>&1 & done
echo "Started $FIRST..$LAST of $TOTAL."
echo "Running: pgrep -fc b3_var.py   Progress (paths): grep -h ',jump02,HMM-6,0.01,' b3_w*.csv | wc -l   Errors: grep -l Error b3_w*.log"
