# GitHub release steps for Windows

Run these commands from the extracted project folder only after
`run_tests_windows.ps1` finishes successfully with 17 tests and the adapter demonstration.

```powershell
cd D:\work\job\dsa-hi-survey-ops

git init
git branch -M main
git add .
git status
git commit -m "Initial public release of dsa-hi-survey-ops"

gh repo create pkd-45/dsa-hi-survey-ops `
  --public `
  --description "Catalogue-aware spectral planning and science-product validation for DSA HI surveys." `
  --source=. `
  --remote=origin `
  --push

git tag -a v0.2.0 -m "Application methods release"
git push origin v0.2.0

gh release create v0.2.0 `
  --title "dsa-hi-survey-ops v0.2.0" `
  --notes "Catalogue-aware zoom-band optimisation, seasonal Doppler planning, configurable instrument snapshots, H I science-product validation, and a FITS-cube injection-recovery adapter. Independent methods sandbox; not official DSA software and no named real-survey result is claimed in this release."
```

Then verify:

```powershell
gh repo view pkd-45/dsa-hi-survey-ops --web
gh release view v0.2.0
git status
```

Open the repository in a private/logged-out browser and confirm that the README,
figure, test workflow, licence, and release are visible.
