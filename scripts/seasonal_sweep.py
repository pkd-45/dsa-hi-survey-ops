#!/usr/bin/env python3
"""Seasonal barycentric retuning sweep for a fixed DSA zoom tuning.

For a given pointing, compute the barycentric velocity correction at monthly
intervals, convert it to a frequency drift at the tuned band centre, and report
whether a fixed zoom tuning stays within the configured guard band. The result
is pointing dependent and is an operational planning diagnostic, not an
instrument-performance forecast.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dsa_hi_survey_ops.doppler import barycentric_corrections_kms

C_KMS = 299792.458


def sweep(
    ra_deg: float,
    dec_deg: float,
    centre_mhz: float,
    guard_channels: int,
    channel_khz: float,
    year: int = 2026,
    verbose: bool = True,
) -> tuple[float, str]:
    epochs = [f"{year}-{month:02d}-15T06:00:00" for month in range(1, 13)]
    velocities = np.array(
        [
            barycentric_corrections_kms(
                ra_deg,
                dec_deg,
                epoch,
                duration_minutes=0.0,
                n_samples=2,
            ).mean()
            for epoch in epochs
        ]
    )
    shifts_khz = centre_mhz * 1e3 * velocities / C_KMS
    span_khz = float(shifts_khz.max() - shifts_khz.min())
    guard_khz = guard_channels * channel_khz
    verdict = (
        "single fixed tuning OK across the year"
        if span_khz / 2 <= guard_khz
        else "seasonal retuning advised (drift exceeds guard band)"
    )

    if verbose:
        print(
            f"pointing RA={ra_deg:.3f} Dec={dec_deg:.3f}; "
            f"centre {centre_mhz:.3f} MHz"
        )
        for epoch, velocity, shift in zip(epochs, velocities, shifts_khz):
            print(
                f"  {epoch[:10]}  v_bary={velocity:+7.2f} km/s  "
                f"shift={shift:+9.1f} kHz ({shift / channel_khz:+7.1f} ch)"
            )
        print(
            f"annual drift span: {span_khz:.1f} kHz = "
            f"{span_khz / channel_khz:.1f} channels of {channel_khz:.0f} kHz"
        )
        print(
            f"guard band: {guard_channels} channels = "
            f"{guard_khz:.1f} kHz per edge"
        )
        print(f"VERDICT: {verdict}")
    return span_khz, verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra", type=float, default=180.0)
    parser.add_argument("--dec", type=float, default=37.0)
    parser.add_argument("--centre-mhz", type=float, default=1339.741)
    parser.add_argument("--guard-channels", type=int, default=16)
    parser.add_argument("--channel-khz", type=float, default=8.0)
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()
    sweep(
        args.ra,
        args.dec,
        args.centre_mhz,
        args.guard_channels,
        args.channel_khz,
        year=args.year,
    )
