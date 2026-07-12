#!/usr/bin/env python3
"""Inject compact rotating H I disks into an existing FITS-cube background.

This adapter extends the synthetic survey-operations demonstration to arbitrary
three-dimensional FITS H I cubes. It is designed for reproducible
injection-recovery tests on public reference cubes such as the SoFiA-2 test
cube, WALLABY cubelets, or THINGS products.

Important scope statement
-------------------------
The repository does not bundle a survey cube and the release validation does
not claim a result on a named real survey data set. The automated tests use a
synthetic FITS round-trip and a structured-noise demonstration to verify the
I/O, source injection, detection, paired-background measurement, and
compression accounting. Supplying a real cube activates exactly the same
workflow.

The recovered flux is measured with paired-background subtraction: the source
mask is determined from the injected cube, then the same voxels from the
pre-injection background are subtracted. This isolates how much injected line
emission the detected mask captures while retaining the effect of the real
background on detection and mask construction.

Usage
-----
  python scripts/real_cube_adapter.py CUBE.fits --n-inj 40
  python scripts/real_cube_adapter.py --demo-synthetic --n-inj 24

Only NumPy, SciPy, Matplotlib, and Astropy are required.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

C_KMS = 299792.458
HI_REST_HZ = 1.42040575177e9


# ---------------------------------------------------------------------------
# Cube I/O and robust statistics
# ---------------------------------------------------------------------------


def _finite_median(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("cube contains no finite voxels")
    return float(np.median(values[finite]))


def sanitise_cube(cube: np.ndarray) -> tuple[np.ndarray, float]:
    """Return float32 cube with non-finite voxels replaced by the finite median.

    Returns the cleaned cube and the original finite-voxel fraction.
    """
    arr = np.asarray(cube, dtype=np.float32)
    finite = np.isfinite(arr)
    fraction = float(finite.mean())
    if fraction == 0.0:
        raise ValueError("cube contains no finite voxels")
    fill = _finite_median(arr)
    return np.where(finite, arr, fill).astype(np.float32, copy=False), fraction


def load_cube(path: str | pathlib.Path) -> tuple[np.ndarray, float | None]:
    """Load a FITS cube and infer the absolute spectral spacing in km/s.

    FITS axis 3 is assumed to be the spectral axis, which maps to NumPy axis 0
    for a conventional three-dimensional cube. Singleton Stokes axes are
    removed. Frequency spacing is converted using RESTFRQ/RESTFREQ when
    available and the H I rest frequency otherwise.
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=np.float32))
        header = hdul[0].header.copy()

    if data.ndim != 3:
        raise ValueError(f"expected a 3D cube after removing singleton axes; got {data.shape}")

    dv_kms: float | None = None
    if "CDELT3" in header:
        delta = abs(float(header["CDELT3"]))
        unit = str(header.get("CUNIT3", "")).strip().lower()
        ctype = str(header.get("CTYPE3", "")).upper()

        if "FREQ" in ctype:
            if unit in {"khz", "khz "}:
                delta_hz = delta * 1e3
            elif unit in {"mhz", "mhz "}:
                delta_hz = delta * 1e6
            else:
                delta_hz = delta
            rest_hz = float(header.get("RESTFRQ", header.get("RESTFREQ", HI_REST_HZ)))
            dv_kms = delta_hz / rest_hz * C_KMS
        elif "km" in unit:
            dv_kms = delta
        elif unit.startswith("m") or unit in {"m/s", "m s-1", "m s^-1"}:
            dv_kms = delta / 1e3

    return data, dv_kms


def robust_noise(cube: np.ndarray) -> float:
    """Median-absolute-deviation noise scale, ignoring non-finite voxels."""
    arr = np.asarray(cube, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("cannot estimate noise from an all-non-finite cube")
    vals = arr[finite]
    med = np.median(vals)
    return float(1.4826 * np.median(np.abs(vals - med)))


def matched_filter(cube: np.ndarray, spat_fwhm: float = 2.0, spec_box: int = 5) -> np.ndarray:
    """Apply the same compact spatial and spectral smoothing used by detection."""
    clean, _ = sanitise_cube(cube)
    smoothed = ndimage.gaussian_filter(
        clean, (0.0, spat_fwhm / 2.355, spat_fwhm / 2.355), mode="nearest"
    )
    return ndimage.uniform_filter1d(smoothed, size=spec_box, axis=0, mode="nearest")


# ---------------------------------------------------------------------------
# Injection model
# ---------------------------------------------------------------------------


def disk_model(
    nx: int,
    nv: int,
    flux: float,
    x0: float,
    y0: float,
    incl_deg: float = 55.0,
    pa_deg: float = 40.0,
    vmax: float = 160.0,
    r_scale: float = 4.0,
    sigma_v: float = 12.0,
    dv_kms: float = 10.0,
    chan0: float | None = None,
) -> np.ndarray:
    """Generate a compact rotating disk with requested integrated line flux.

    ``flux`` is defined such that ``cube.sum() * dv_kms == flux``. The units are
    therefore whatever flux-density units the background uses, multiplied by
    km/s.
    """
    if nx < 8 or nv < 8:
        raise ValueError("disk model requires at least 8 spatial pixels and 8 channels")
    if dv_kms <= 0:
        raise ValueError("dv_kms must be positive")

    yy, xx = np.mgrid[0:nx, 0:nx].astype(float)
    incl = np.deg2rad(float(incl_deg))
    pa = np.deg2rad(float(pa_deg))
    cosi = max(np.cos(incl), 0.08)

    xr = (xx - x0) * np.cos(pa) + (yy - y0) * np.sin(pa)
    yr = (-(xx - x0) * np.sin(pa) + (yy - y0) * np.cos(pa)) / cosi
    radius = np.hypot(xr, yr) + 1e-9

    surface = np.exp(-radius / r_scale)
    surface /= surface.sum()

    vlos = vmax * (1.0 - np.exp(-radius / 2.5)) * (xr / radius) * np.sin(incl)
    centre = nv / 2.0 if chan0 is None else float(chan0)
    channel = np.arange(nv, dtype=float)[:, None, None]
    spectral_offset = channel - centre - vlos[None] / dv_kms
    profile = np.exp(-0.5 * (spectral_offset / (sigma_v / dv_kms)) ** 2)

    cube = surface[None] * profile
    normalisation = float(cube.sum() * dv_kms)
    if normalisation <= 0:
        raise RuntimeError("disk model produced a non-positive normalisation")
    cube *= float(flux) / normalisation
    return cube.astype(np.float32)


def source_halfwidth_channels(
    vmax: float, incl_deg: float, sigma_v: float, dv_kms: float
) -> float:
    """Expected half-width used only for matching a detection to an injection."""
    return (vmax * np.sin(np.deg2rad(incl_deg)) + 2.5 * sigma_v) / dv_kms


def scale_source_to_peak_snr(
    unit_source: np.ndarray,
    background: np.ndarray,
    target_snr: float,
    spat_fwhm: float = 2.0,
    spec_box: int = 5,
) -> tuple[np.ndarray, float, float]:
    """Scale a unit source to a requested matched-filter peak S/N.

    The filtered noise is estimated from the exact background patch used for
    the injection. The returned tuple is ``(scaled_source, achieved_snr,
    filtered_noise)``.
    """
    background_filtered = matched_filter(background, spat_fwhm, spec_box)
    sigma_filtered = max(robust_noise(background_filtered), 1e-12)
    source_filtered = matched_filter(unit_source, spat_fwhm, spec_box)
    unit_peak = float(np.max(source_filtered))
    if unit_peak <= 0:
        raise ValueError("source has no positive matched-filter response")
    scale = float(target_snr) * sigma_filtered / unit_peak
    scaled = unit_source * scale
    achieved = float(np.max(matched_filter(scaled, spat_fwhm, spec_box)) / sigma_filtered)
    return scaled.astype(np.float32), achieved, sigma_filtered


# ---------------------------------------------------------------------------
# Detection and science measurements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    chan: float
    peak_snr: float
    nvox: int
    mask_slices: tuple[slice, slice, slice]
    mask: np.ndarray


def detect(
    cube: np.ndarray,
    nsig: float = 4.5,
    min_vox: int = 6,
    spat_fwhm: float = 2.0,
    spec_box: int = 5,
) -> list[Detection]:
    """Matched-filter threshold detector returning connected 3D components."""
    smoothed = matched_filter(cube, spat_fwhm, spec_box)
    sigma = max(robust_noise(smoothed), 1e-12)
    snr = smoothed / sigma
    labels, nlabels = ndimage.label(snr > nsig)

    detections: list[Detection] = []
    for slices, label_id in zip(ndimage.find_objects(labels), range(1, nlabels + 1)):
        mask = labels[slices] == label_id
        nvox = int(mask.sum())
        if nvox < min_vox:
            continue

        weights = np.where(mask, np.clip(snr[slices], 0.0, None), 0.0)
        total = float(weights.sum())
        if total <= 0:
            continue
        ch, yy, xx = np.mgrid[slices[0], slices[1], slices[2]]
        detections.append(
            Detection(
                x=float((xx * weights).sum() / total),
                y=float((yy * weights).sum() / total),
                chan=float((ch * weights).sum() / total),
                peak_snr=float(weights.max()),
                nvox=nvox,
                mask_slices=slices,
                mask=mask,
            )
        )
    return detections


def match_detection(
    detections: list[Detection],
    x0: float,
    y0: float,
    chan0: float,
    r_match: float,
    dchan_match: float,
) -> Detection | None:
    """Return the closest detection satisfying spatial and spectral limits."""
    candidates: list[tuple[float, Detection]] = []
    for item in detections:
        dr = float(np.hypot(item.x - x0, item.y - y0))
        dc = abs(item.chan - chan0)
        if dr <= r_match and dc <= dchan_match:
            score = (dr / max(r_match, 1e-9)) ** 2 + (dc / max(dchan_match, 1e-9)) ** 2
            candidates.append((score, item))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


def _positive_centroid(spectrum: np.ndarray) -> float | None:
    values = np.clip(np.asarray(spectrum, dtype=float), 0.0, None)
    total = float(values.sum())
    if total <= 0:
        return None
    return float(np.dot(np.arange(values.size, dtype=float), values) / total)


def w50_channels(spectrum: np.ndarray) -> float:
    """Simple full width at half peak in channels."""
    values = np.clip(np.asarray(spectrum, dtype=float), 0.0, None)
    if values.size == 0 or values.max() <= 0:
        return 0.0
    above = np.flatnonzero(values >= 0.5 * values.max())
    return float(above[-1] - above[0] + 1) if above.size else 0.0


def measure_recovery(
    injected_cube: np.ndarray,
    background_cube: np.ndarray,
    source_cube: np.ndarray,
    detection: Detection,
    dv_kms: float,
    x0: float,
    y0: float,
    dilation_iterations: int = 0,
) -> dict[str, float | None]:
    """Measure source recovery using paired-background subtraction.

    The detection mask comes from the noisy injected cube. Subtracting the
    paired pre-injection background in exactly those voxels yields the captured
    injected signal and prevents unrelated real emission from being counted as
    recovered source flux.
    """
    strict_mask = np.zeros_like(injected_cube, dtype=bool)
    strict_mask[detection.mask_slices] = detection.mask
    if dilation_iterations > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        recovery_mask = ndimage.binary_dilation(
            strict_mask, structure=structure, iterations=dilation_iterations
        )
    else:
        recovery_mask = strict_mask

    increment = injected_cube - background_cube
    captured = np.where(recovery_mask, increment, 0.0)
    captured_spectrum = captured.sum(axis=(1, 2))
    true_spectrum = source_cube.sum(axis=(1, 2))

    recovered_flux = float(captured.sum() * dv_kms)
    true_flux = float(source_cube.sum() * dv_kms)
    true_centroid = _positive_centroid(true_spectrum)
    measured_local = _positive_centroid(captured_spectrum)
    measured_centroid = (
        float(detection.mask_slices[0].start + measured_local) if measured_local is not None else None
    )

    centroid_offset = None
    if true_centroid is not None and measured_centroid is not None:
        centroid_offset = float((measured_centroid - true_centroid) * dv_kms)

    return {
        "flux_ratio": recovered_flux / true_flux if true_flux > 0 else None,
        "recovered_flux": recovered_flux,
        "true_flux": true_flux,
        "w50_true_kms": w50_channels(true_spectrum) * dv_kms,
        "w50_measured_kms": w50_channels(captured_spectrum) * dv_kms,
        "centroid_offset_kms": centroid_offset,
        "position_offset_px": float(np.hypot(detection.x - x0, detection.y - y0)),
        "peak_snr": float(detection.peak_snr),
        "nvox": float(recovery_mask.sum()),
    }


def compress_channels(cube: np.ndarray, factor: int = 2) -> tuple[np.ndarray, int]:
    """Average adjacent channels and quantise to float16.

    Returns the compressed float16 cube and the number of original channels
    retained. Integrated flux is preserved when measurements use the enlarged
    channel width ``factor * dv``.
    """
    if factor < 1:
        raise ValueError("compression factor must be at least one")
    nv = cube.shape[0]
    retained = (nv // factor) * factor
    if retained < factor:
        raise ValueError("cube has too few channels for requested averaging")
    trimmed = np.asarray(cube[:retained], dtype=np.float32)
    compressed = trimmed.reshape(retained // factor, factor, *trimmed.shape[1:]).mean(axis=1)
    return compressed.astype(np.float16), retained


def _compressed_channel_coordinate(chan: float, factor: int) -> float:
    """Map an original channel-centre coordinate to an averaged cube."""
    return (float(chan) + 0.5) / factor - 0.5


# ---------------------------------------------------------------------------
# Patch selection and experiment
# ---------------------------------------------------------------------------


def choose_background_patch(
    cube: np.ndarray,
    box: int,
    rng: np.random.Generator,
    max_attempts: int = 80,
) -> tuple[np.ndarray, int, int, float, float]:
    """Choose a mostly finite, not-obviously-source-dominated spatial patch."""
    nv, ny, nx = cube.shape
    best: tuple[float, np.ndarray, int, int, float, float] | None = None

    for _ in range(max_attempts):
        y_start = int(rng.integers(0, ny - box + 1))
        x_start = int(rng.integers(0, nx - box + 1))
        raw = cube[:, y_start : y_start + box, x_start : x_start + box]
        finite_fraction = float(np.isfinite(raw).mean())
        if finite_fraction < 0.95:
            score = 10.0 + (1.0 - finite_fraction)
            contamination = 1.0
        else:
            clean, _ = sanitise_cube(raw)
            sigma = max(robust_noise(clean), 1e-12)
            median = float(np.median(clean))
            contamination = float(np.mean(np.abs(clean - median) > 6.0 * sigma))
            score = contamination + 0.1 * (1.0 - finite_fraction)

        if best is None or score < best[0]:
            clean, _ = sanitise_cube(raw)
            best = (score, clean, y_start, x_start, finite_fraction, contamination)
        if finite_fraction >= 0.995 and contamination <= 0.01:
            break

    if best is None:
        raise RuntimeError("failed to select a background patch")
    _, patch, y_start, x_start, finite_fraction, contamination = best
    return patch, y_start, x_start, finite_fraction, contamination


def make_demo_background(
    nv: int = 48,
    ny: int = 128,
    nx: int = 128,
    seed: int = 7,
) -> np.ndarray:
    """Structured synthetic background used only for automated release checks."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 0.4, (nv, ny, nx)).astype(np.float32)
    correlated = ndimage.gaussian_filter(white, (0.0, 1.2, 1.2), mode="nearest")
    channel_scale = 1.0 + 0.18 * np.sin(np.linspace(0, 3.0 * np.pi, nv))
    cube = (0.72 * white + 0.55 * correlated) * channel_scale[:, None, None]

    yy, xx = np.mgrid[0:ny, 0:nx]
    bowl = -0.22 * np.exp(-((xx - 0.72 * nx) ** 2 + (yy - 0.35 * ny) ** 2) / (2 * 11.0**2))
    cube += bowl[None] * np.cos(np.linspace(0, 2 * np.pi, nv))[:, None, None]
    cube[5:7, :, 20:22] += 1.5  # compact structured artefact for realism
    cube[:, :2, :] = np.nan      # exercise non-finite handling
    return cube.astype(np.float32)


def _median_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def run(
    cube: np.ndarray,
    dv_kms: float,
    n_inj: int = 40,
    flux_snr_targets: tuple[float, ...] = (6.0, 12.0, 25.0),
    seed: int = 20260713,
    outdir: str | pathlib.Path = "results/adapter_demo",
    background_label: str = "supplied FITS cube",
) -> dict:
    """Run injection-recovery on the supplied cube background."""
    if dv_kms is None or not np.isfinite(dv_kms) or dv_kms <= 0:
        raise ValueError("a positive channel spacing in km/s is required")
    if n_inj < 1:
        raise ValueError("n_inj must be positive")
    if not flux_snr_targets:
        raise ValueError("at least one target S/N tier is required")

    raw_cube = np.asarray(cube, dtype=np.float32)
    if raw_cube.ndim != 3:
        raise ValueError(f"expected a 3D cube, got {raw_cube.shape}")
    nv, ny, nx = raw_cube.shape
    box = min(64, ny, nx)
    if box < 32:
        raise ValueError("spatial dimensions must be at least 32 pixels")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    patch_finite: list[float] = []
    patch_contamination: list[float] = []

    for index in range(n_inj):
        target_snr = float(flux_snr_targets[index % len(flux_snr_targets)])
        background, y_start, x_start, finite_fraction, contamination = choose_background_patch(
            raw_cube, box, rng
        )
        patch_finite.append(finite_fraction)
        patch_contamination.append(contamination)

        inclination = float(rng.uniform(25.0, 70.0))
        pa_deg = float(rng.uniform(0.0, 180.0))
        sigma_v = 12.0
        # Keep the model line inside the supplied spectral window.  The
        # velocity range remains galaxy-like but adapts for small test cubes.
        max_halfwidth = max(nv / 2.0 - 6.0, 4.0)
        vmax_limit = (max_halfwidth * dv_kms - 2.5 * sigma_v) / max(
            np.sin(np.deg2rad(inclination)), 0.2
        )
        vmax_upper = min(190.0, max(65.0, vmax_limit))
        vmax_lower = min(110.0, max(55.0, 0.65 * vmax_upper))
        vmax = float(rng.uniform(vmax_lower, vmax_upper))
        dchan = source_halfwidth_channels(vmax, inclination, sigma_v, dv_kms)
        margin = max(int(np.ceil(dchan)) + 4, 8)
        if 2 * margin >= nv:
            raise ValueError(
                "cube has too few spectral channels for the model line width; "
                "use a smaller velocity model or a larger spectral window"
            )
        chan0 = float(rng.uniform(margin, nv - margin - 1))
        x0, y0 = rng.uniform(14.0, box - 14.0, size=2)

        unit_source = disk_model(
            box,
            nv,
            flux=1.0,
            x0=float(x0),
            y0=float(y0),
            incl_deg=inclination,
            pa_deg=pa_deg,
            vmax=vmax,
            sigma_v=sigma_v,
            dv_kms=dv_kms,
            chan0=chan0,
        )
        source, achieved_snr, filtered_sigma = scale_source_to_peak_snr(
            unit_source, background, target_snr
        )
        injected = background + source

        detections = detect(injected)
        matched = match_detection(
            detections,
            float(x0),
            float(y0),
            chan0,
            r_match=3.0,
            dchan_match=dchan,
        )

        row: dict[str, float | int | bool | None] = {
            "target_snr": target_snr,
            "achieved_matched_filter_snr": achieved_snr,
            "filtered_noise": filtered_sigma,
            "detected": matched is not None,
            "injected_flux": float(source.sum() * dv_kms),
            "x_global": float(x_start + x0),
            "y_global": float(y_start + y0),
            "chan0": chan0,
            "incl_deg": inclination,
            "vmax_kms": vmax,
            "patch_finite_fraction": finite_fraction,
            "patch_high_tail_fraction": contamination,
        }

        if matched is not None:
            strict_metrics = measure_recovery(
                injected, background, source, matched, dv_kms, float(x0), float(y0)
            )
            broad_metrics = measure_recovery(
                injected,
                background,
                source,
                matched,
                dv_kms,
                float(x0),
                float(y0),
                dilation_iterations=2,
            )
            row.update(strict_metrics)
            row.update({f"broad_{key}": value for key, value in broad_metrics.items()})

        compressed_injected, retained = compress_channels(injected, factor=2)
        compressed_background, _ = compress_channels(background, factor=2)
        compressed_source, _ = compress_channels(source, factor=2)
        compressed_injected32 = compressed_injected.astype(np.float32)
        compressed_background32 = compressed_background.astype(np.float32)
        compressed_source32 = compressed_source.astype(np.float32)
        compressed_chan0 = _compressed_channel_coordinate(chan0, 2)
        compressed_detections = detect(compressed_injected32, spec_box=3)
        compressed_match = match_detection(
            compressed_detections,
            float(x0),
            float(y0),
            compressed_chan0,
            r_match=3.0,
            dchan_match=max(dchan / 2.0, 1.0),
        )
        row["compressed_detected"] = compressed_match is not None
        row["compression_factor_bytes"] = float(
            injected[:retained].nbytes / compressed_injected.nbytes
        )
        if compressed_match is not None:
            compressed_metrics = measure_recovery(
                compressed_injected32,
                compressed_background32,
                compressed_source32,
                compressed_match,
                dv_kms * 2.0,
                float(x0),
                float(y0),
            )
            row.update({f"compressed_{key}": value for key, value in compressed_metrics.items()})

        rows.append(row)

    per_tier: dict[str, dict] = {}
    for target in flux_snr_targets:
        selected = [row for row in rows if row["target_snr"] == float(target)]
        detected_rows = [row for row in selected if row["detected"]]
        compressed_rows = [row for row in selected if row["compressed_detected"]]
        per_tier[str(float(target))] = {
            "n": len(selected),
            "completeness": len(detected_rows) / len(selected),
            "compressed_completeness": len(compressed_rows) / len(selected),
            "median_strict_flux_ratio": _median_or_none(
                [row.get("flux_ratio") for row in detected_rows]
            ),
            "median_broad_flux_ratio": _median_or_none(
                [row.get("broad_flux_ratio") for row in detected_rows]
            ),
            "median_flux_ratio": _median_or_none([row.get("flux_ratio") for row in detected_rows]),
            "median_abs_centroid_offset_kms": _median_or_none(
                [abs(row["centroid_offset_kms"]) if row.get("centroid_offset_kms") is not None else None for row in detected_rows]
            ),
            "median_abs_w50_error_kms": _median_or_none(
                [
                    abs(float(row["w50_measured_kms"]) - float(row["w50_true_kms"]))
                    if row.get("w50_measured_kms") is not None
                    else None
                    for row in detected_rows
                ]
            ),
            "median_compressed_flux_ratio": _median_or_none(
                [row.get("compressed_flux_ratio") for row in compressed_rows]
            ),
        }

    cleaned, original_finite_fraction = sanitise_cube(raw_cube)
    report = {
        "background_label": background_label,
        "cube_shape": list(raw_cube.shape),
        "original_finite_fraction": original_finite_fraction,
        "noise_mad": robust_noise(cleaned),
        "dv_kms": float(dv_kms),
        "n_injections": n_inj,
        "measurement": "paired-background-corrected detected-mask recovery",
        "median_patch_finite_fraction": float(np.median(patch_finite)),
        "median_patch_high_tail_fraction": float(np.median(patch_contamination)),
        "per_tier": per_tier,
        "rows": rows,
    }

    output = pathlib.Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "real_cube_adapter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=float)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        tier_values = [float(value) for value in flux_snr_targets]
        completeness = [per_tier[str(value)]["completeness"] for value in tier_values]
        compressed_completeness = [
            per_tier[str(value)]["compressed_completeness"] for value in tier_values
        ]
        axes[0].plot(tier_values, completeness, "o-", label="native")
        axes[0].plot(tier_values, compressed_completeness, "s--", label="2-channel + f16")
        axes[0].set_xscale("log")
        axes[0].set_ylim(-0.03, 1.05)
        axes[0].set_xlabel("calibrated matched-filter peak S/N")
        axes[0].set_ylabel("completeness")
        axes[0].legend()

        for target in tier_values:
            selected = [
                row for row in rows if row["target_snr"] == target and row["detected"]
            ]
            axes[1].scatter(
                [target * 0.97] * len(selected),
                [row["flux_ratio"] for row in selected],
                alpha=0.55,
                marker="o",
                label="strict" if target == tier_values[0] else None,
            )
            axes[1].scatter(
                [target * 1.03] * len(selected),
                [row["broad_flux_ratio"] for row in selected],
                alpha=0.55,
                marker="s",
                label="grown mask" if target == tier_values[0] else None,
            )
        axes[1].axhline(1.0, ls="--", c="k", lw=1)
        axes[1].legend()
        axes[1].set_xscale("log")
        axes[1].set_xlabel("calibrated matched-filter peak S/N")
        axes[1].set_ylabel("paired-background flux ratio")
        figure.suptitle(f"Injection-recovery on {background_label}")
        figure.tight_layout()
        figure.savefig(output / "real_cube_adapter_summary.png", dpi=150)
        plt.close(figure)
    except Exception as exc:  # plotting should never invalidate the numerical report
        report["plot_warning"] = str(exc)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject model H I disks into an existing FITS-cube background."
    )
    parser.add_argument("cube", nargs="?", help="path to a 3D FITS H I cube")
    parser.add_argument("--n-inj", type=int, default=40)
    parser.add_argument(
        "--dv-kms", type=float, default=None, help="channel width if not inferable from FITS"
    )
    parser.add_argument(
        "--demo-synthetic",
        action="store_true",
        help="use the structured synthetic release-check background",
    )
    parser.add_argument("--outdir", default="results/adapter_demo")
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    if args.demo_synthetic:
        if args.cube:
            parser.error("provide either a FITS cube or --demo-synthetic, not both")
        cube = make_demo_background()
        dv_kms = args.dv_kms or 10.0
        label = "structured synthetic release-check background"
        print("adapter validation mode: structured synthetic background")
    else:
        if not args.cube:
            parser.error("a FITS cube is required unless --demo-synthetic is specified")
        cube, inferred_dv = load_cube(args.cube)
        dv_kms = args.dv_kms or inferred_dv
        if dv_kms is None:
            parser.error("could not infer channel spacing; provide --dv-kms")
        label = pathlib.Path(args.cube).name
        print(
            f"loaded {args.cube}: shape={cube.shape}, dv={dv_kms:.3f} km/s, "
            f"noise(MAD)={robust_noise(cube):.5g}"
        )

    report = run(
        cube,
        float(dv_kms),
        n_inj=args.n_inj,
        seed=args.seed,
        outdir=args.outdir,
        background_label=label,
    )
    for tier, summary in report["per_tier"].items():
        print(
            f"  S/N {tier}: completeness={summary['completeness']:.2f}, "
            f"compressed={summary['compressed_completeness']:.2f}, "
            f"strict/broad flux ratio={summary['median_strict_flux_ratio']}/{summary['median_broad_flux_ratio']}"
        )
    print(f"wrote {pathlib.Path(args.outdir) / 'real_cube_adapter_report.json'}")


if __name__ == "__main__":
    main()
