# Changelog

## 0.2.0 - 2026-07-12

- Added `scripts/real_cube_adapter.py` for injection--recovery on arbitrary
  three-dimensional FITS H I cube backgrounds.
- Added NaN-safe FITS loading and spectral-spacing inference for velocity or
  frequency axes.
- Calibrated every injection to an actual matched-filter peak S/N rather than
  using an empirical flux multiplier.
- Randomised line centres and adapted model widths to the available spectral
  window.
- Added position-and-velocity matching, paired-background-corrected flux,
  centroid and W50 recovery, and strict-versus-grown-mask diagnostics.
- Corrected channel-averaging flux accounting by using the enlarged channel
  width; added float16 byte-compression reporting.
- Added structured-background CLI validation and four adapter-specific tests,
  bringing the deterministic test total to 17.
- Documented that no named real survey cube is bundled or claimed as a release
  result; public-cube runs are separate, reproducible outputs.

## 0.1.2 - 2026-07-12

- Updated package and citation metadata.
- Rewrote instrument documentation to distinguish dated public snapshots from final design claims.
- Added a public JSON configuration loader and exercised the optimiser under both bundled DSA snapshots.
- Added `--config` to the demonstration command.
- Removed unclosed-file warnings and made seasonal tests quiet and current-year based.
- Clarified that seasonal retuning is pointing dependent.
- Added GitHub Actions continuous integration for tests and the deterministic demonstration.

## 0.1.1 - 2026-07-12

- Relabelled configurations as illustrative dated snapshots after finding differing values on public DSA pages.
- Added `configs/dsa_full_alt.json`.
- Renamed the reference-product test to `test_reference_configuration_arithmetic`.
- Added `scripts/seasonal_sweep.py` and a unit test.
- Added independent verification records.

## 0.1.0 - 2026-07-12

- Added full-DSA and DSA-110 reference configurations.
- Added catalogue-aware fixed-width H I zoom-band optimisation.
- Added propagation of barycentric correction, redshift uncertainty, W50, and guard channels.
- Added illustrative RFI-aware target weighting.
- Added rotating-disk H I cube forward model.
- Added narrow-band RFI detection, robust continuum subtraction, channel averaging, and quantisation.
- Added broad and strict masks plus H I flux, centroid, W50, position-angle, completeness, and purity QA.
- Added deterministic tests, reference report, summary figure, and Windows/Linux runners.
