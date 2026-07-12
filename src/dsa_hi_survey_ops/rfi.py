from __future__ import annotations

import numpy as np
from scipy import ndimage


def detect_narrowband_rfi(cube: np.ndarray, nsig: float = 6.0) -> np.ndarray:
    """Identify channels with anomalous spatially coherent power."""
    if cube.ndim != 3:
        raise ValueError("cube must have shape [channel, y, x]")
    channel_median = np.median(cube, axis=(1, 2))
    baseline = ndimage.median_filter(channel_median, size=11, mode="reflect")
    residual = channel_median - baseline
    sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    if sigma <= 0:
        return np.zeros(cube.shape[0], dtype=bool)
    return np.abs(residual) > nsig * sigma


def replace_flagged_channels(cube: np.ndarray, channel_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate flagged channels for downstream demonstrations."""
    out = np.asarray(cube, dtype=float).copy()
    mask = np.asarray(channel_mask, dtype=bool)
    if mask.shape != (cube.shape[0],):
        raise ValueError("channel_mask shape mismatch")
    good = np.flatnonzero(~mask)
    if good.size < 2:
        raise ValueError("at least two unflagged channels are required")
    x = np.arange(cube.shape[0])
    flat = out.reshape(out.shape[0], -1)
    for col in range(flat.shape[1]):
        flat[:, col] = np.interp(x, good, flat[good, col])
    return flat.reshape(out.shape)
