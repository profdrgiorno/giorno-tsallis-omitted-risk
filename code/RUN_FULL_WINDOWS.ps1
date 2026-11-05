# Paper B jump arm on the 4,000-path manifest. Run from this folder.
# $Repo: local clone of https://github.com/profdrgiorno/Giorno_garch_hmm6 (needs DGP_MANIFEST.csv only)
param([string]$Repo = "..\Giorno_garch_hmm6", [int]$Workers = 8)
$env:HMM6_REPO = (Resolve-Path $Repo).Path
$env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
for ($w = 0; $w -lt $Workers; $w++) {
  Start-Process -NoNewWindow python -ArgumentList "full_srj.py $w $Workers full_w$w.csv" -RedirectStandardOutput "full_w$w.log" -RedirectStandardError "full_w$w.err"
}
# Resumable: rerun the same command after an interruption; finished ids are skipped.
# When done:  python summarise_full.py
