from __future__ import annotations

import numpy as np


def barycentric_corrections_kms(
    ra_deg: float,
    dec_deg: float,
    start_utc: str,
    duration_minutes: float = 15.0,
    n_samples: int = 9,
    latitude_deg: float = 37.2338889,
    longitude_deg: float = -118.2822222,
    height_m: float = 1222.0,
) -> np.ndarray:
    """Barycentric radial-velocity correction over a DSA observing block.

    Defaults use the Owens Valley Radio Observatory coordinates. Astropy's IERS
    auto-download is disabled so the demonstration remains deterministic and
    usable offline; bundled Earth-orientation data are sufficient here.
    """
    if duration_minutes < 0:
        raise ValueError("duration_minutes must be non-negative")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False
    location = EarthLocation.from_geodetic(
        lon=longitude_deg * u.deg,
        lat=latitude_deg * u.deg,
        height=height_m * u.m,
    )
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    t0 = Time(start_utc, scale="utc")
    times = t0 + np.linspace(0.0, duration_minutes, n_samples) * u.minute
    correction = target.radial_velocity_correction(
        kind="barycentric", obstime=times, location=location
    )
    return correction.to_value(u.km / u.s)
