param([int]$First = 34, [int]$Last = 50, [int]$Total = 51)
$here = (Get-Location).Path
python -c "import numba" 2>$null; if ($LASTEXITCODE -eq 0) { Write-Host "numba found" } else { Write-Host "WARNING: numba not found; HMM-6 fits ~10x slower (conda install numba)" }
foreach ($w in $First..$Last) {
  Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoExit", "-Command",
    "cd '$here'; `$env:HMM6_REPO='$here'; `$env:OPENBLAS_NUM_THREADS='1'; `$env:NUMBA_NUM_THREADS='1'; python -I b3_var.py $w $Total b3_w$w.csv"
}
Write-Host "Started workers $First..$Last. Progress: (Get-Content b3_w*.csv | Select-String ',jump02,HMM-6,0.01,').Count"
