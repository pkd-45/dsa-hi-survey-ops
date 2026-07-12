from __future__ import annotations

import numpy as np


def subtract_polynomial_continuum(
    cube: np.ndarray,
    velocity_kms: np.ndarray,
    line_free_mask: np.ndarray,
    degree: int = 2,
    clip_sigma: float = 4.0,
    max_iter: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust per-pixel polynomial continuum subtraction."""
    if cube.ndim != 3:
        raise ValueError("cube must have shape [channel, y, x]")
    if line_free_mask.shape != (cube.shape[0],):
        raise ValueError("line_free_mask shape mismatch")
    if line_free_mask.sum() <= degree + 2:
        raise ValueError("insufficient line-free channels")
    x = velocity_kms / np.max(np.abs(velocity_kms))
    flat = np.asarray(cube, dtype=float).reshape(cube.shape[0], -1)
    model = np.empty_like(flat)
    for column in range(flat.shape[1]):
        y = flat[:, column]
        good = line_free_mask & np.isfinite(y)
        for _ in range(max_iter):
            coeff = np.polyfit(x[good], y[good], degree)
            fit = np.polyval(coeff, x)
            resid = y - fit
            sigma = 1.4826 * np.median(np.abs(resid[good] - np.median(resid[good])))
            if sigma <= 0:
                break
            new_good = good & (np.abs(resid) < clip_sigma * sigma)
            if np.array_equal(new_good, good) or new_good.sum() <= degree + 2:
                break
            good = new_good
        coeff = np.polyfit(x[good], y[good], degree)
        model[:, column] = np.polyval(coeff, x)
    model = model.reshape(cube.shape)
    return cube - model, model
