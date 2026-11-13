# Windows release verification

This repository is already published. Use these commands from a fresh clone to verify
that the public release reproduces its reference tests. No repository creation,
history rewrite, or tag recreation is necessary.

```powershell
git clone https://github.com/pkd-45/dsa-hi-survey-ops.git
cd dsa-hi-survey-ops
.\run_tests_windows.ps1
git status --short
gh release view v0.2.0
```

The test script exercises the synthetic science-validation workflow. The FITS-cube
adapter is tested with synthetic input; real survey measurements require the
input file, checksum, command, and results to be independently recorded.

Verify the public README, documented limitations, example plots, test workflow,
licence and tagged release before referring others to a specific version.
