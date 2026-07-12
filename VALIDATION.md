# Validation record

Version: 0.2.0
Date: 12 July 2026

## Commands

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dsa_hi_survey_ops.demo --output results
python -m dsa_hi_survey_ops.demo --config configs/dsa_full_alt.json --output /tmp/dsa-alt
python scripts/seasonal_sweep.py
python scripts/real_cube_adapter.py --demo-synthetic --n-inj 24 --outdir results/adapter_demo
```

The same core checks are run by `.github/workflows/tests.yml` after publication.

## Test coverage

The 17 deterministic tests cover the original planning and cube-QA workflow plus
four FITS-adapter checks:

1. reference-configuration arithmetic;
2. H I redshift-frequency round trip;
3. propagation of redshift uncertainty into line intervals;
4. high-priority catalogue selection by the fixed-width optimiser;
5. RFI-aware movement of the optimum;
6. loading both dated DSA configuration snapshots and running each through the optimiser;
7. annual barycentric-drift scale and retuning decision;
8. recovery of injected narrow-band RFI channels;
9. continuum-subtraction flux preservation in a noise-free control;
10. lossless/no-op compression mode;
11. broad-mask completeness relative to the strict mask;
12. exact self-comparison of line metrics and QA;
13. end-to-end generation of report, tables, and summary figure;
14. FITS round-trip, non-finite voxel handling, and bright injection recovery;
15. direct bright rotating-disk detection;
16. exact matched-filter S/N calibration;
17. integrated-flux preservation under two-channel averaging plus float16 quantisation.

## Verified results

```text
Ran 17 tests
OK
zoom centre: 1339.741 MHz
fully contained targets: 39
weighted yield: 37.54
strict QA: amber
```

The structured-background adapter demonstration is a deterministic software
check, not a real-survey result. With 24 injections it exercises three calibrated
matched-filter S/N tiers, position-and-velocity matching, strict and grown masks,
and compressed-cube recovery. Its JSON and figure are stored under
`results/adapter_demo/`.

## Real-data boundary

The adapter can ingest a public SoFiA-2, WALLABY, THINGS, or other conventional
three-dimensional FITS H I cube. No such external cube is bundled, and this
release does not claim a performance result on a named survey data set. Automated
validation instead uses (a) a synthetic FITS round-trip and (b) a structured
background with non-finite voxels. This verifies the code path while keeping
external data provenance and any survey-specific result explicit.

For an actual public-cube run, archive the input identifier or checksum, command,
software version, JSON report, and figure before citing the result.

## Instrument snapshots

The `/overview-1` snapshot and the current `/overview` page differed in channel
count and observing-block duration when accessed on 12 July 2026. Both are
represented as configurable JSON inputs, and the tests exercise the optimiser
under both. This is a configuration-robustness check, not a claim that either
snapshot is the final operational design.

## Seasonal-sweep verification

The seasonal result is pointing dependent:

- default field, RA=180 deg and Dec=37 deg: annual span about 220.8 kHz
  (27.6 channels at 8 kHz), within a 16-channel guard on each edge;
- lower-declination test field, RA=180 deg and Dec=5 deg: annual span about
  267 kHz (33.4 channels), so seasonal retuning is advised.

Barycentric corrections use Astropy coordinate/time transformations. These are
operational demonstrations, not DSA performance forecasts.
