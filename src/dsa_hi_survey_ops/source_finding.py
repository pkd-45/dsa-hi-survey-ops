from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class LineMetrics:
    integrated_flux: float
    centroid_kms: float
    w50_kms: float
    position_angle_deg: float
    mask_voxels: int


def estimate_noise(cube: np.ndarray, line_free_channels: np.ndarray | None = None) -> float:
    values = cube[line_free_channels] if line_free_channels is not None else cube
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def make_masks(
    cube: np.ndarray,
    noise_sigma: float,
    broad_sigma: float = 2.3,
    strict_sigma: float = 4.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create broad/high-completeness and strict/high-confidence masks."""
    smoothed = ndimage.gaussian_filter(cube, sigma=(0.7, 1.0, 1.0))
    broad = smoothed > broad_sigma * noise_sigma
    broad = ndimage.binary_dilation(broad, iterations=2)
    strict = smoothed > strict_sigma * noise_sigma
    labels, nlab = ndimage.label(strict)
    clean = np.zeros_like(strict)
    for index in range(1, nlab + 1):
        component = labels == index
        if component.sum() >= 8:
            clean |= component
    strict = ndimage.binary_dilation(clean, iterations=1)
    return broad, strict


def _w50(velocity: np.ndarray, spectrum: np.ndarray) -> float:
    if not np.any(np.isfinite(spectrum)) or np.nanmax(spectrum) <= 0:
        return float("nan")
    half = 0.5 * np.nanmax(spectrum)
    idx = np.flatnonzero(spectrum >= half)
    if idx.size < 2:
        return 0.0
    return float(abs(velocity[idx[-1]] - velocity[idx[0]]))


def measure_line(cube: np.ndarray, velocity_kms: np.ndarray, mask: np.ndarray) -> LineMetrics:
    data = np.where(mask, cube, 0.0)
    positive = np.clip(data, 0.0, None)
    spectrum = positive.sum(axis=(1, 2))
    flux = float(spectrum.sum())
    centroid = float(np.sum(velocity_kms * spectrum) / flux) if flux > 0 else float("nan")
    moment0 = positive.sum(axis=0)
    total = moment0.sum()
    if total > 0:
        yy, xx = np.mgrid[0:moment0.shape[0], 0:moment0.shape[1]].astype(float)
        x0 = float((xx * moment0).sum() / total)
        y0 = float((yy * moment0).sum() / total)
        dx, dy = xx - x0, yy - y0
        ixx = float((moment0 * dx * dx).sum() / total)
        iyy = float((moment0 * dy * dy).sum() / total)
        ixy = float((moment0 * dx * dy).sum() / total)
        pa = 0.5 * np.degrees(np.arctan2(2 * ixy, ixx - iyy))
        pa = float(pa % 180.0)
    else:
        pa = float("nan")
    return LineMetrics(
        integrated_flux=flux,
        centroid_kms=centroid,
        w50_kms=_w50(velocity_kms, spectrum),
        position_angle_deg=pa,
        mask_voxels=int(mask.sum()),
    )
