import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from real_cube_adapter import (  # noqa: E402
    compress_channels,
    detect,
    disk_model,
    load_cube,
    make_demo_background,
    matched_filter,
    robust_noise,
    run,
    scale_source_to_peak_snr,
)


class RealCubeAdapterTests(unittest.TestCase):
    def test_fits_roundtrip_nan_handling_and_recovery(self):
        from astropy.io import fits

        rng = np.random.default_rng(3)
        cube = rng.normal(0, 0.4, (48, 96, 96)).astype(np.float32)
        cube[:, :2, :] = np.nan
        header = fits.Header()
        header["CDELT3"] = 10000.0
        header["CUNIT3"] = "m/s"
        header["CTYPE3"] = "VELO-LSR"

        with tempfile.TemporaryDirectory() as temporary:
            cube_path = pathlib.Path(temporary) / "roundtrip.fits"
            fits.PrimaryHDU(cube[None], header=header).writeto(cube_path)
            loaded, dv_kms = load_cube(cube_path)
            self.assertEqual(loaded.shape, (48, 96, 96))
            self.assertAlmostEqual(dv_kms, 10.0)

            report = run(
                loaded,
                dv_kms,
                n_inj=6,
                flux_snr_targets=(25.0,),
                seed=1,
                outdir=pathlib.Path(temporary) / "adapter",
                background_label="unit-test FITS cube",
            )

        summary = report["per_tier"]["25.0"]
        self.assertEqual(summary["completeness"], 1.0)
        self.assertGreater(summary["median_flux_ratio"], 0.45)
        self.assertLessEqual(summary["median_flux_ratio"], 1.01)
        self.assertGreater(report["original_finite_fraction"], 0.95)

    def test_bright_disk_detected(self):
        rng = np.random.default_rng(5)
        cube = rng.normal(0, 0.4, (48, 64, 64)).astype(np.float32)
        cube += disk_model(
            64,
            48,
            flux=6000.0,
            x0=30,
            y0=32,
            dv_kms=10.0,
            vmax=95.0,
        )
        self.assertGreaterEqual(len(detect(cube)), 1)

    def test_matched_filter_snr_calibration(self):
        background = make_demo_background(nv=48, ny=64, nx=64, seed=11)
        unit_source = disk_model(
            64,
            48,
            flux=1.0,
            x0=31.0,
            y0=32.0,
            dv_kms=10.0,
            vmax=90.0,
        )
        source, achieved, sigma_filtered = scale_source_to_peak_snr(
            unit_source, background, target_snr=12.0
        )
        measured = np.max(matched_filter(source)) / sigma_filtered
        self.assertAlmostEqual(achieved, 12.0, places=5)
        self.assertAlmostEqual(measured, 12.0, places=5)
        self.assertGreater(robust_noise(background), 0.0)

    def test_channel_averaging_preserves_integrated_flux(self):
        source = disk_model(
            64,
            48,
            flux=123.4,
            x0=30.0,
            y0=31.0,
            dv_kms=10.0,
            vmax=90.0,
        )
        compressed, retained = compress_channels(source, factor=2)
        original_flux = source[:retained].sum() * 10.0
        compressed_flux = compressed.astype(np.float32).sum() * 20.0
        self.assertAlmostEqual(original_flux, compressed_flux, delta=0.02 * original_flux)
        self.assertEqual(source[:retained].nbytes / compressed.nbytes, 4.0)


if __name__ == "__main__":
    unittest.main()
