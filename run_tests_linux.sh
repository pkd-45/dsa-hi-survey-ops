#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dsa_hi_survey_ops.demo --output results
ALT_OUTPUT="$(mktemp -d)"
trap 'rm -rf "$ALT_OUTPUT"' EXIT
python -m dsa_hi_survey_ops.demo --config configs/dsa_full_alt.json --output "$ALT_OUTPUT"
python scripts/seasonal_sweep.py
ADAPTER_OUTPUT="$(mktemp -d)"
python scripts/real_cube_adapter.py --demo-synthetic --n-inj 12 --outdir "$ADAPTER_OUTPUT"
rm -rf "$ADAPTER_OUTPUT"
printf '%s\n' "Validation complete: 17 tests, both instrument snapshots, seasonal sweep, and FITS-adapter demo passed."
