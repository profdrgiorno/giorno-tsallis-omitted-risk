#!/bin/bash
# $1 = worker id (0/1): splits reps 0-149 / 150-299
w=$1; r0=$((w*150)); r1=$((r0+150))
for sc in negative_gamma_jumps gaussian_contamination correct; do for n in 1000 2000 3000; do
  OPENBLAS_NUM_THREADS=1 python3 -I hmm9_srj.py $sc $n $r0 $r1 out/${sc}_${n}_w${w}.csv > out/${sc}_${n}_w${w}.log 2>&1
done; done
echo DONE > out/done_w${w}
