from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompressionResult:
    cube: np.ndarray
    channel_average: int
    quant_bits: int
    nominal_factor: float
    scale: float


def compress_cube(
    cube: np.ndarray,
    channel_average: int = 1,
    quant_bits: int = 16,
) -> CompressionResult:
    """Channel-average then symmetric integer-quantise a cube.

    The returned cube is de-quantised for direct science-metric comparison.
    """
    if channel_average < 1:
        raise ValueError("channel_average must be >= 1")
    if quant_bits not in (8, 12, 16, 32):
        raise ValueError("quant_bits must be one of 8, 12, 16, 32")
    arr = np.asarray(cube, dtype=float)
    ntrim = arr.shape[0] - arr.shape[0] % channel_average
    arr = arr[:ntrim]
    if channel_average > 1:
        arr = arr.reshape(ntrim // channel_average, channel_average, *arr.shape[1:]).mean(axis=1)
    if quant_bits == 32:
        out = arr.astype(np.float32)
        scale = 1.0
    else:
        qmax = 2 ** (quant_bits - 1) - 1
        peak = float(np.max(np.abs(arr)))
        scale = peak / qmax if peak > 0 else 1.0
        quant = np.clip(np.rint(arr / scale), -qmax, qmax).astype(np.int32)
        out = quant.astype(float) * scale
    factor = channel_average * 32.0 / quant_bits
    return CompressionResult(
        cube=out,
        channel_average=channel_average,
        quant_bits=quant_bits,
        nominal_factor=factor,
        scale=scale,
    )
