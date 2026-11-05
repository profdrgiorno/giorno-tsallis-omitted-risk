# Paper B, items B1 + B2. Run from this folder on each computer.
#   Computer with 2 Xeons : .\RUN_B12.ps1 -First 0  -Last 33
#   Computer with 1 Xeon  : .\RUN_B12.ps1 -First 34 -Last 50
# Each worker opens a minimised window; it is finished when its window returns to the prompt.
param([int]$First = 0, [int]$Last = 33, [int]$Total = 51)
$here = (Get-Location).Path
foreach ($w in $First..$Last) {
  Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoExit", "-Command",
    "cd '$here'; `$env:HMM6_REPO='$here'; `$env:OPENBLAS_NUM_THREADS='1'; `$env:MKL_NUM_THREADS='1'; python -I b12_run.py $w $Total b12_w$w.csv"
}
Write-Host "Started workers $First..$Last of $Total. Progress: (Get-Content b12_w*.csv | Select-String ',sym,').Count"
