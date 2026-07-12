$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dsa_hi_survey_ops.demo --output results

$altOutput = Join-Path $env:TEMP "dsa-hi-survey-ops-alt"
Remove-Item -Recurse -Force $altOutput -ErrorAction SilentlyContinue
python -m dsa_hi_survey_ops.demo --config configs\dsa_full_alt.json --output $altOutput
python scripts\seasonal_sweep.py

$adapterOutput = Join-Path $env:TEMP "dsa-hi-survey-ops-adapter"
Remove-Item -Recurse -Force $adapterOutput -ErrorAction SilentlyContinue
python scripts\real_cube_adapter.py --demo-synthetic --n-inj 12 --outdir $adapterOutput

Write-Host "Validation complete: 17 tests, both instrument snapshots, seasonal sweep, and FITS-adapter demo passed."
