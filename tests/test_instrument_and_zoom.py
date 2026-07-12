import pathlib
import sys
import unittest

import numpy as np

from dsa_hi_survey_ops.catalogue import Target, synthetic_catalogue
from dsa_hi_survey_ops.instrument_modes import (
    DSA110,
    DSA_FULL,
    load_instrument_mode,
    observed_hi_frequency_mhz,
    redshift_from_frequency_mhz,
)
from dsa_hi_survey_ops.zoom_optimizer import optimise_zoom_band, target_frequency_interval


class InstrumentAndZoomTests(unittest.TestCase):
    def test_reference_configuration_arithmetic(self):
        zoom = DSA_FULL.product("hi_nearby_zoom")
        self.assertEqual(zoom.n_channels, 4192)
        self.assertAlmostEqual(zoom.channel_width_khz, 8.0)
        self.assertAlmostEqual(zoom.width_mhz, 33.536)
        self.assertEqual(DSA_FULL.observing_block_minutes, 15.0)
        self.assertAlmostEqual(DSA110.product("full_band").width_mhz, 250.0)

    def test_hi_frequency_round_trip(self):
        for redshift in (0.0, 0.03, 0.5, 1.0):
            frequency = observed_hi_frequency_mhz(redshift)
            self.assertAlmostEqual(
                redshift_from_frequency_mhz(frequency), redshift, places=12
            )

    def test_target_interval_grows_with_uncertainty(self):
        base = Target("G", 150, 2, 0.03, 1e-5, 200, 0.8, 1.0)
        uncertain = Target("G", 150, 2, 0.03, 5e-4, 200, 0.8, 1.0)
        corrections = np.array([-10.0, -9.8])
        base_interval = target_frequency_interval(base, corrections)
        uncertain_interval = target_frequency_interval(uncertain, corrections)
        self.assertGreater(
            uncertain_interval.high_mhz - uncertain_interval.low_mhz,
            base_interval.high_mhz - base_interval.low_mhz,
        )

    def test_optimizer_contains_high_weight_cluster(self):
        targets = [
            Target("A", 0, 0, 0.030, 2e-5, 180, 0.95, 3.0),
            Target("B", 0, 0, 0.032, 2e-5, 180, 0.90, 2.0),
            Target("C", 0, 0, 0.075, 2e-5, 180, 0.40, 1.0),
        ]
        plan = optimise_zoom_band(
            targets,
            [-12.0, -11.8],
            DSA_FULL,
            DSA_FULL.product("hi_nearby_zoom"),
            grid_step_mhz=0.1,
        )
        status = {selection.target_id: selection.status for selection in plan.selections}
        self.assertEqual(status["A"], "contained")
        self.assertEqual(status["B"], "contained")
        self.assertEqual(status["C"], "outside")

    def test_rfi_penalty_reduces_overlap(self):
        targets = synthetic_catalogue(40, seed=11)
        product = DSA_FULL.product("hi_nearby_zoom")
        no_rfi = optimise_zoom_band(targets, [-10.0, -9.9], DSA_FULL, product)
        rfi = [(no_rfi.center_mhz - 8.0, no_rfi.center_mhz + 8.0)]
        with_rfi = optimise_zoom_band(
            targets,
            [-10.0, -9.9],
            DSA_FULL,
            product,
            rfi_bands_mhz=rfi,
            rfi_penalty=3.0,
        )
        self.assertLess(with_rfi.rfi_occupancy_fraction, 0.48)


class ConfigVariantsAndSeasonTests(unittest.TestCase):
    def test_alternate_configs_run_through_optimizer(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        targets = synthetic_catalogue(24, seed=19)
        for name in ("dsa_full.json", "dsa_full_alt.json"):
            mode = load_instrument_mode(root / "configs" / name)
            product = mode.product("hi_nearby_zoom")
            plan = optimise_zoom_band(targets, [-10.0, -9.8], mode, product)
            self.assertGreater(product.width_mhz, 30.0)
            self.assertLess(product.width_mhz, 35.0)
            self.assertIn(mode.observing_block_minutes, (15.0, 20.0))
            self.assertGreater(plan.fully_contained_targets, 0)
            self.assertGreaterEqual(plan.low_mhz, mode.band_low_mhz)
            self.assertLessEqual(plan.high_mhz, mode.band_high_mhz)

    def test_seasonal_drift_is_of_annual_earth_orbit_scale(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from seasonal_sweep import sweep

        span, verdict = sweep(
            180.0,
            5.0,
            1339.741,
            16,
            8.0,
            year=2026,
            verbose=False,
        )
        self.assertGreater(span, 50.0)
        self.assertLess(span, 300.0)
        self.assertIn("tuning", verdict)


if __name__ == "__main__":
    unittest.main()
