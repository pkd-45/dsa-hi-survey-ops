from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CubeData:
    velocity_kms: np.ndarray
    truth_line: np.ndarray
    continuum: np.ndarray
    noise: np.ndarray
    rfi: np.ndarray
    observed: np.ndarray
    truth_mask: np.ndarray
    noise_sigma: float


def simulate_hi_cube(
    nchan: int = 128,
    ny: int = 32,
    nx: int = 32,
    dv_kms: float = 6.0,
    total_line_flux: float = 1100.0,
    inclination_deg: float = 58.0,
    pa_deg: float = 32.0,
    vmax_kms: float = 165.0,
    dispersion_kms: float = 12.0,
    scale_length_pix: float = 4.5,
    noise_sigma: float = 0.045,
    continuum_level: float = 0.7,
    ripple_fraction: float = 0.025,
    rfi_channels: tuple[int, ...] = (18, 79, 80),
    seed: int = 9,
) -> CubeData:
    """Synthetic rotating H I disk plus continuum, noise and narrow-band RFI."""
    rng = np.random.default_rng(seed)
    velocity = (np.arange(nchan) - (nchan - 1) / 2) * dv_kms
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x0, y0 = (nx - 1) / 2, (ny - 1) / 2
    pa = np.deg2rad(pa_deg)
    inc = np.deg2rad(inclination_deg)
    xr = (xx - x0) * np.cos(pa) + (yy - y0) * np.sin(pa)
    yr = (-(xx - x0) * np.sin(pa) + (yy - y0) * np.cos(pa)) / np.cos(inc)
    radius = np.hypot(xr, yr) + 1e-8
    surface = np.exp(-radius / scale_length_pix)
    surface /= surface.sum()
    vrot = vmax_kms * (1.0 - np.exp(-radius / 2.5))
    vlos = vrot * (xr / radius) * np.sin(inc)
    line = surface[None, :, :] * np.exp(
        -0.5 * ((velocity[:, None, None] - vlos[None, :, :]) / dispersion_kms) ** 2
    )
    line *= total_line_flux / line.sum()

    xfreq = np.linspace(-1.0, 1.0, nchan)[:, None, None]
    spatial_cont = (
        1.0
        + 0.8 * np.exp(-((xx - 7.0) ** 2 + (yy - 23.0) ** 2) / (2 * 2.2**2))
        + 0.5 * np.exp(-((xx - 25.0) ** 2 + (yy - 9.0) ** 2) / (2 * 1.6**2))
    )
    continuum = continuum_level * spatial_cont[None, :, :] * (
        1.0 + 0.07 * xfreq + 0.05 * xfreq**2
    )
    continuum += continuum_level * ripple_fraction * np.sin(2 * np.pi * 3.1 * xfreq)
    noise = rng.normal(0.0, noise_sigma, line.shape)
    rfi = np.zeros_like(line)
    for channel in rfi_channels:
        if 0 <= channel < nchan:
            pattern = rng.normal(0.0, 1.0, (ny, nx))
            pattern = 0.55 + 0.12 * pattern
            rfi[channel] += pattern
    observed = line + continuum + noise + rfi
    truth_mask = line > (0.008 * line.max())
    return CubeData(
        velocity_kms=velocity,
        truth_line=line,
        continuum=continuum,
        noise=noise,
        rfi=rfi,
        observed=observed,
        truth_mask=truth_mask,
        noise_sigma=noise_sigma,
    )
