import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsa_hi_survey_ops.compression import compress_cube
from dsa_hi_survey_ops.continuum import subtract_polynomial_continuum
from dsa_hi_survey_ops.cube_simulator import simulate_hi_cube
from dsa_hi_survey_ops.demo import run_demo
from dsa_hi_survey_ops.qa import compare_metrics
from dsa_hi_survey_ops.rfi import detect_narrowband_rfi, replace_flagged_channels
from dsa_hi_survey_ops.source_finding import estimate_noise, make_masks, measure_line


class CubePipelineTests(unittest.TestCase):
    def test_rfi_channels_are_detected(self):
        cube = simulate_hi_cube(rfi_channels=(12, 55), seed=2)
        mask = detect_narrowband_rfi(cube.observed, nsig=5.0)
        self.assertTrue(mask[12])
        self.assertTrue(mask[55])

    def test_continuum_subtraction_preserves_line_flux(self):
        cube = simulate_hi_cube(noise_sigma=0.0, rfi_channels=(), seed=3)
        line_free = np.abs(cube.velocity_kms) > 230.0
        residual, _ = subtract_polynomial_continuum(cube.observed, cube.velocity_kms, line_free)
        recovered = residual.sum()
        truth = cube.truth_line.sum()
        self.assertLess(abs(recovered - truth) / truth, 0.03)

    def test_compression_noop(self):
        rng = np.random.default_rng(1)
        cube = rng.normal(size=(16, 4, 4))
        result = compress_cube(cube, channel_average=1, quant_bits=32)
        np.testing.assert_allclose(result.cube, cube, rtol=1e-6, atol=1e-6)
        self.assertEqual(result.nominal_factor, 1.0)

    def test_broad_mask_is_more_complete(self):
        cube = simulate_hi_cube(seed=7)
        flagged = detect_narrowband_rfi(cube.observed)
        cleaned = replace_flagged_channels(cube.observed, flagged)
        residual, _ = subtract_polynomial_continuum(
            cleaned, cube.velocity_kms, np.abs(cube.velocity_kms) > 230.0
        )
        noise = estimate_noise(residual, np.abs(cube.velocity_kms) > 230.0)
        broad, strict = make_masks(residual, noise)
        broad_comp = np.logical_and(broad, cube.truth_mask).sum() / cube.truth_mask.sum()
        strict_comp = np.logical_and(strict, cube.truth_mask).sum() / cube.truth_mask.sum()
        self.assertGreaterEqual(broad_comp, strict_comp)

    def test_measurement_and_qa_are_finite(self):
        cube = simulate_hi_cube(noise_sigma=0.0, rfi_channels=(), seed=4)
        truth = measure_line(cube.truth_line, cube.velocity_kms, cube.truth_mask)
        recovered = measure_line(cube.truth_line, cube.velocity_kms, cube.truth_mask)
        qa = compare_metrics(truth, recovered, cube.truth_mask, cube.truth_mask)
        self.assertAlmostEqual(qa.flux_error_percent, 0.0)
        self.assertAlmostEqual(qa.centroid_error_kms, 0.0)
        self.assertAlmostEqual(qa.mask_completeness, 1.0)
        self.assertEqual(qa.traffic_light, "green")

    def test_end_to_end_demo_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(tmp, seed=31)
            self.assertTrue(Path(tmp, "reference_report.json").exists())
            self.assertTrue(Path(tmp, "dsa_hi_survey_ops_summary.png").exists())
            self.assertTrue(Path(tmp, "target_selection.csv").exists())
            self.assertGreater(report["zoom_plan"]["fully_contained_targets"], 0)
            self.assertIn(report["processing"]["strict_qa"]["traffic_light"], {"green", "amber", "red"})


if __name__ == "__main__":
    unittest.main()
