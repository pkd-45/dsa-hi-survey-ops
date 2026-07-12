from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .catalogue import synthetic_catalogue, write_catalogue_csv
from .compression import compress_cube
from .continuum import subtract_polynomial_continuum
from .cube_simulator import simulate_hi_cube
from .doppler import barycentric_corrections_kms
from .instrument_modes import DSA_FULL, InstrumentMode, load_instrument_mode
from .qa import compare_metrics
from .rfi import detect_narrowband_rfi, replace_flagged_channels
from .source_finding import estimate_noise, make_masks, measure_line
from .zoom_optimizer import optimise_zoom_band, target_frequency_interval


def run_demo(
    output_dir: str | Path = "results",
    seed: int = 123,
    instrument: InstrumentMode = DSA_FULL,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = synthetic_catalogue(80, seed=seed)
    corrections = barycentric_corrections_kms(
        150.0, 2.0, "2026-07-15T03:00:00", duration_minutes=instrument.observing_block_minutes
    )
    rfi_bands = [(1330.2, 1331.0), (1370.0, 1382.0), (1401.2, 1402.0)]
    product = instrument.product("hi_nearby_zoom")
    plan = optimise_zoom_band(
        targets,
        corrections,
        instrument,
        product,
        rfi_bands_mhz=rfi_bands,
    )
    plan_no_rfi = optimise_zoom_band(targets, corrections, instrument, product)
    write_catalogue_csv(targets, output_dir / "synthetic_catalogue.csv")
    with (output_dir / "target_selection.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(plan.selections[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(s) for s in plan.selections)

    cube = simulate_hi_cube(seed=seed)
    channel_rfi = detect_narrowband_rfi(cube.observed)
    cleaned = replace_flagged_channels(cube.observed, channel_rfi)
    line_free = np.abs(cube.velocity_kms) > 230.0
    residual, continuum_model = subtract_polynomial_continuum(
        cleaned, cube.velocity_kms, line_free, degree=2
    )
    compression = compress_cube(residual, channel_average=2, quant_bits=12)
    velocity_compressed = cube.velocity_kms[: compression.cube.shape[0] * 2].reshape(-1, 2).mean(axis=1)
    truth_compressed = cube.truth_line[: compression.cube.shape[0] * 2].reshape(
        compression.cube.shape[0], 2, *cube.truth_line.shape[1:]
    ).mean(axis=1)
    truth_mask_compressed = cube.truth_mask[: compression.cube.shape[0] * 2].reshape(
        compression.cube.shape[0], 2, *cube.truth_mask.shape[1:]
    ).any(axis=1)
    noise_est = estimate_noise(compression.cube, np.abs(velocity_compressed) > 230.0)
    broad_mask, strict_mask = make_masks(compression.cube, noise_est)
    truth_metrics = measure_line(truth_compressed, velocity_compressed, truth_mask_compressed)
    broad_metrics = measure_line(compression.cube, velocity_compressed, broad_mask)
    strict_metrics = measure_line(compression.cube, velocity_compressed, strict_mask)
    broad_qa = compare_metrics(truth_metrics, broad_metrics, truth_mask_compressed, broad_mask)
    strict_qa = compare_metrics(truth_metrics, strict_metrics, truth_mask_compressed, strict_mask)

    intervals = [target_frequency_interval(t, corrections) for t in targets]
    centers = np.linspace(1310.0, 1405.0, 160)
    yield_curve = []
    for center in centers:
        low, high = center - product.width_mhz / 2, center + product.width_mhz / 2
        yield_curve.append(
            sum(i.weight for i in intervals if i.low_mhz >= low and i.high_mhz <= high)
        )

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    ax = axes[0, 0]
    for interval in intervals:
        ax.plot([interval.low_mhz, interval.high_mhz], [interval.weight, interval.weight], alpha=0.35)
    ax.axvspan(plan.low_mhz, plan.high_mhz, alpha=0.18, label="optimised zoom")
    for a, b in rfi_bands:
        ax.axvspan(a, b, alpha=0.18)
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("target weight")
    ax.set_title("catalogue intervals, RFI and selected zoom")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    ax.plot(centers, yield_curve)
    ax.axvline(plan.center_mhz, ls="--", label="RFI-aware optimum")
    ax.axvline(plan_no_rfi.center_mhz, ls=":", label="no-RFI optimum")
    ax.set_xlabel("zoom centre (MHz)")
    ax.set_ylabel("fully-contained weighted yield")
    ax.set_title("fixed-width zoom-band objective")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    raw_spec = cube.observed.sum(axis=(1, 2))
    truth_spec = cube.truth_line.sum(axis=(1, 2))
    residual_spec = residual.sum(axis=(1, 2))
    compressed_spec = compression.cube.sum(axis=(1, 2)) * 2.0
    ax.plot(cube.velocity_kms, raw_spec / np.max(raw_spec), label="observed + continuum")
    ax.plot(cube.velocity_kms, truth_spec / np.max(truth_spec), label="truth H I")
    ax.plot(cube.velocity_kms, residual_spec / np.max(np.abs(residual_spec)), label="continuum-subtracted")
    ax.plot(velocity_compressed, compressed_spec / np.max(np.abs(compressed_spec)), label="compressed")
    ax.set_xlabel("velocity (km s$^{-1}$)")
    ax.set_ylabel("normalised integrated spectrum")
    ax.set_title("science-preserving spectral processing")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    labels = ["|flux| (%)", "centroid", "W50", "PA", "1-C", "1-P"]
    strict_values = [
        abs(strict_qa.flux_error_percent),
        strict_qa.centroid_error_kms,
        strict_qa.w50_error_kms,
        strict_qa.pa_error_deg,
        100 * (1 - strict_qa.mask_completeness),
        100 * (1 - strict_qa.mask_purity),
    ]
    broad_values = [
        abs(broad_qa.flux_error_percent),
        broad_qa.centroid_error_kms,
        broad_qa.w50_error_kms,
        broad_qa.pa_error_deg,
        100 * (1 - broad_qa.mask_completeness),
        100 * (1 - broad_qa.mask_purity),
    ]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, broad_values, width=0.36, label=f"broad ({broad_qa.traffic_light})")
    ax.bar(x + 0.18, strict_values, width=0.36, label=f"strict ({strict_qa.traffic_light})")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("error / missing fraction")
    ax.set_title("science-product QA")
    ax.legend(fontsize=8)

    fig.tight_layout()
    figure_path = output_dir / "dsa_hi_survey_ops_summary.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)

    report = {
        "disclaimer": "Independent application-focused sandbox; not official DSA software or a performance forecast.",
        "instrument": {
            "name": instrument.name,
            "observing_block_minutes": instrument.observing_block_minutes,
            "product": asdict(product),
            "product_width_mhz": product.width_mhz,
        },
        "barycentric_corrections_kms": {
            "minimum": float(corrections.min()),
            "maximum": float(corrections.max()),
            "drift": float(corrections.max() - corrections.min()),
        },
        "zoom_plan": plan.to_dict(),
        "zoom_plan_without_rfi": plan_no_rfi.to_dict(),
        "processing": {
            "flagged_channels": np.flatnonzero(channel_rfi).tolist(),
            "compression": asdict(compression) | {"cube": None},
            "noise_estimate": noise_est,
            "truth_metrics": asdict(truth_metrics),
            "broad_metrics": asdict(broad_metrics),
            "strict_metrics": asdict(strict_metrics),
            "broad_qa": broad_qa.to_dict(),
            "strict_qa": strict_qa.to_dict(),
        },
    }
    with (output_dir / "reference_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DSA H I survey-operations sandbox")
    parser.add_argument("--output", default="results", help="output directory")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--config",
        help="optional JSON instrument configuration; defaults to the bundled reference mode",
    )
    args = parser.parse_args()
    instrument = load_instrument_mode(args.config) if args.config else DSA_FULL
    report = run_demo(args.output, args.seed, instrument=instrument)
    plan = report["zoom_plan"]
    print(f"zoom centre: {plan['center_mhz']:.3f} MHz")
    print(f"fully contained targets: {plan['fully_contained_targets']}")
    print(f"weighted yield: {plan['expected_weighted_yield']:.2f}")
    print(f"strict QA: {report['processing']['strict_qa']['traffic_light']}")


if __name__ == "__main__":
    main()
