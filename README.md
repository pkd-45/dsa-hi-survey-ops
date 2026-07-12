# dsa-hi-survey-ops

[![tests](https://github.com/pkd-45/dsa-hi-survey-ops/actions/workflows/tests.yml/badge.svg)](https://github.com/pkd-45/dsa-hi-survey-ops/actions/workflows/tests.yml)

**Catalogue-aware spectral planning and science-product validation for Deep Synoptic Array H I surveys.**

This is an independent, application-focused methods sandbox. It is **not official DSA software** and its reference numbers are **not performance forecasts**.

The project addresses operational tasks named in the Bonn DYNAVERSE postdoctoral advert:

- forward modelling of spectral-line data;
- Doppler/zoom-band setup;
- RFI-aware source selection;
- continuum subtraction;
- science-preserving compression;
- completeness, reliability, and quality assurance.

It complements [`ml-radio-recon`](https://github.com/pkd-45/ml-radio-recon), which demonstrates interferometric forward modelling and machine-learning reconstruction. Here the central problem is different: connect configurable survey products and catalogue-level observing choices to the H I measurements that must survive the pipeline.

## What is new

### 1. Catalogue-aware fixed-width zoom optimisation

For every target, the code propagates:

- optical redshift and redshift uncertainty;
- expected H I line width;
- barycentric motion over an observing block;
- guard channels;
- target priority and expected detectability;
- illustrative RFI occupancy.

It then places a fixed-width nearby-galaxy H I zoom product to maximise weighted target yield while requiring full line containment where possible.

### 2. Seasonal Doppler sensitivity

`scripts/seasonal_sweep.py` evaluates the annual barycentric frequency span for a chosen pointing and compares it with the configured guard band. The decision is sky-position dependent: the same tuning may be adequate for one field but require seasonal retuning for another.

### 3. Science-ready cube acceptance tests

A synthetic rotating H I disk is passed through:

- structured continuum and spectral ripple;
- narrow-band RFI;
- thermal noise;
- robust continuum subtraction;
- channel averaging and quantisation;
- broad/high-completeness and strict/high-confidence source masks.

Acceptance is judged using:

- integrated H I flux;
- velocity centroid;
- W50;
- projected position angle;
- mask completeness and purity.

The goal is not merely a low residual image, but preserved astrophysical measurements.

### 4. FITS-cube injection-recovery adapter

`scripts/real_cube_adapter.py` applies the same validation logic to an existing
three-dimensional FITS H I cube. It selects mostly finite background patches,
injects rotating disks at calibrated matched-filter S/N, matches detections in
position and velocity, and reports completeness plus paired-background-corrected
flux, centroid, and $W_{50}$ recovery. It also compares strict and grown masks and
checks two-channel averaging plus float16 quantisation with correct channel-width
accounting.

The release does **not** bundle a survey cube or claim a result on a named real
data set. Automated validation uses a synthetic FITS round-trip and a structured
noise background so that I/O, NaN handling, S/N calibration, mask recovery, and
compression accounting are deterministic. Supplying a public SoFiA-2, WALLABY,
or THINGS cube activates the same workflow:

```bash
python scripts/real_cube_adapter.py path/to/cube.fits --n-inj 40 \
  --outdir results/my_real_cube
```

This distinction is deliberate: the adapter is real-data-ready, while any
survey-specific result must be generated, archived, and cited separately.

## Instrument-configuration note

The public DSA webpages were evolving when this sandbox was prepared. The shipped configurations encode two **illustrative public reference snapshots accessed on 12 July 2026**, rather than claims about the final instrument design:

- `configs/dsa_full.json`: the `/overview-1` snapshot, including a 4192 x 8 kHz nearby-H I product and 15-minute observing blocks;
- `configs/dsa_full_alt.json`: values from the current `/overview` detailed product list and observing-strategy text, including 4096 x 8 kHz and 20-minute blocks.

The current `/overview` page itself contains differing values between its detailed product list and summary table. Instrument settings are therefore configuration inputs, not assumptions hidden in the algorithms. The JSON loader and tests verify that the optimiser operates under either snapshot. No application claim or scientific conclusion depends on a particular channel count or block duration.

The DSA-110 configuration records the public 1280-1530 MHz range and 8192-channel full-band channelisation. See `PUBLIC_SOURCES.md`.

## Repository layout

```text
dsa-hi-survey-ops/
├── .github/workflows/tests.yml
├── configs/
│   ├── dsa_full.json
│   ├── dsa_full_alt.json
│   ├── dsa110.json
│   └── rfi_example.json
├── src/dsa_hi_survey_ops/
│   ├── instrument_modes.py
│   ├── catalogue.py
│   ├── doppler.py
│   ├── zoom_optimizer.py
│   ├── cube_simulator.py
│   ├── rfi.py
│   ├── continuum.py
│   ├── compression.py
│   ├── source_finding.py
│   ├── qa.py
│   └── demo.py
├── scripts/
│   ├── run_demo.py
│   ├── seasonal_sweep.py
│   └── real_cube_adapter.py
├── tests/
├── results/
├── APPLICATION_INSERT.tex
├── INTERVIEW_BRIEF.md
└── VALIDATION.md
```

## Windows quick start

```powershell
cd D:\work\job\dsa-hi-survey-ops
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_tests_windows.ps1
```

Or manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dsa_hi_survey_ops.demo --output results
python scripts\seasonal_sweep.py
python scripts\real_cube_adapter.py --demo-synthetic --n-inj 12
```

## Linux/macOS quick start

```bash
bash run_tests_linux.sh
```

## Configuration examples

Run the deterministic demonstration with either bundled snapshot:

```bash
python -m dsa_hi_survey_ops.demo --config configs/dsa_full.json --output results/reference
python -m dsa_hi_survey_ops.demo --config configs/dsa_full_alt.json --output results/alternate
python scripts/real_cube_adapter.py --demo-synthetic --n-inj 24 \
  --outdir results/adapter_demo
```

## Reference outputs

The deterministic demo writes:

- `results/reference_report.json`
- `results/synthetic_catalogue.csv`
- `results/target_selection.csv`
- `results/dsa_hi_survey_ops_summary.png`
- `results/adapter_demo/real_cube_adapter_report.json`
- `results/adapter_demo/real_cube_adapter_summary.png`

The main figure has four panels: catalogue frequency intervals and RFI, the fixed-width objective as the zoom centre moves, spectral processing, and broad/strict science-product QA.

## Reference run

```text
RFI-aware zoom centre:       1339.741 MHz
No-RFI zoom centre:          1377.713 MHz
Fixed zoom width:            33.536 MHz
Fully contained targets:     39 / 80
Weighted expected yield:     37.54
Flagged synthetic channels:  [18, 79, 80]
Compression factor:          5.33
Broad-mask QA:               green
Strict-mask QA:              amber
```

These values are deterministic regression outputs for the shipped synthetic example, not expected DSA performance.

## Seasonal example

For the default demonstration pointing (RA=180 deg, Dec=37 deg), the annual barycentric span is approximately 220.8 kHz, or 27.6 channels at 8 kHz. With a 16-channel guard on each edge, the script reports that a fixed tuning is adequate. For the lower-declination test field (Dec=5 deg), the span is approximately 267 kHz, or 33.4 channels, and seasonal retuning is advised. These examples demonstrate pointing dependence rather than a universal DSA requirement.

## Release validation

The v0.2.0 release has 17 deterministic unit tests. Four adapter-specific tests
cover FITS I/O with non-finite voxels, matched-filter S/N calibration, bright
source detection, and integrated-flux preservation under channel averaging plus
float16 quantisation. Continuous integration also executes the structured
background adapter demonstration.

## Honest limitations

- No named real survey cube is bundled or used for a headline result in this release; the FITS adapter is validated by deterministic round-trip tests and must be run separately on downloaded public data.
- The target catalogue is synthetic and the RFI bands are illustrative, not a measured OVRO occupancy database.
- The cube simulator is an intentionally compact rotating-disk model, not a CASA or official DSA simulator.
- Continuum subtraction is per-pixel polynomial fitting; low-rank, Gaussian-process, and learned alternatives are future comparisons.
- Compression uses channel averaging plus scalar quantisation, not a production codec.
- Source finding uses transparent threshold masks rather than SoFiA or a survey production finder.
- The optimiser demonstrates the decision structure; real deployment would use validated catalogues, operational constraints, calibration state, and measured RFI statistics.

## Roadmap

1. Ingest real catalogue uncertainties and H I mass/width priors.
2. Replace illustrative RFI bands with measured occupancy products.
3. Run and archive the FITS adapter on a named public SoFiA-2/WALLABY/THINGS reference cube, then compare against the corresponding catalogue or mask products.
4. Add a SoFiA-compatible catalogue benchmark.
5. Add moment-0, moment-1, tilted-ring, and low-surface-brightness recovery tests.
6. Compare compression and continuum methods at fixed science tolerances and runtime budgets.
7. Expose the optimiser and QA report through configuration files and a lightweight service.

## Author

Pratyush Kumar Das, University of Queensland
ORCID: 0000-0002-4326-8598
