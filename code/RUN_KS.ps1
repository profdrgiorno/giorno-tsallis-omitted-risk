param([int]$First = 34, [int]$Last = 50, [int]$Total = 51)
$here = (Get-Location).Path
foreach ($w in $First..$Last) {
  Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoExit", "-Command",
    "cd '$here'; `$env:HMM6_REPO='$here'; `$env:OPENBLAS_NUM_THREADS='1'; python -I ks_rerun.py $w $Total ks_w$w.csv"
}
Write-Host "Started workers $First..$Last. Progress: (Get-Content ks_w*.csv | Select-String ',sym,').Count"
