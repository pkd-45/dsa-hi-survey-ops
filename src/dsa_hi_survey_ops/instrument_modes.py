from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

C_KMS = 299_792.458
HI_REST_MHZ = 1420.40575177


@dataclass(frozen=True)
class SpectralProduct:
    name: str
    n_channels: int
    channel_width_khz: float
    purpose: str

    @property
    def width_mhz(self) -> float:
        return self.n_channels * self.channel_width_khz / 1000.0

    def velocity_resolution_kms(self, frequency_mhz: float) -> float:
        if frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be positive")
        return C_KMS * (self.channel_width_khz / 1000.0) / frequency_mhz


@dataclass(frozen=True)
class InstrumentMode:
    name: str
    band_low_mhz: float
    band_high_mhz: float
    observing_block_minutes: float
    products: tuple[SpectralProduct, ...]

    def product(self, name: str) -> SpectralProduct:
        for product in self.products:
            if product.name == name:
                return product
        raise KeyError(f"unknown product {name!r} for {self.name}")


def instrument_mode_from_dict(config: dict[str, Any]) -> InstrumentMode:
    """Build an :class:`InstrumentMode` from a JSON-compatible mapping.

    Instrument parameters are intentionally configuration data rather than
    assumptions embedded in the optimisation code. This allows published
    reference snapshots or later operational values to be swapped without
    changing the planning and QA algorithms.
    """
    try:
        band_low, band_high = config["frequency_range_mhz"]
        products_config = config["spectral_products"]
        products = tuple(
            SpectralProduct(
                name=name,
                n_channels=int(values["n_channels"]),
                channel_width_khz=float(values["channel_width_khz"]),
                purpose=str(values.get("purpose", name.replace("_", " "))),
            )
            for name, values in products_config.items()
        )
        mode = InstrumentMode(
            name=str(config["name"]),
            band_low_mhz=float(band_low),
            band_high_mhz=float(band_high),
            observing_block_minutes=float(config["observing_block_minutes"]),
            products=products,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid instrument configuration: {exc}") from exc

    if mode.band_low_mhz <= 0 or mode.band_high_mhz <= mode.band_low_mhz:
        raise ValueError("frequency_range_mhz must be positive and increasing")
    if mode.observing_block_minutes <= 0:
        raise ValueError("observing_block_minutes must be positive")
    for product in mode.products:
        if product.n_channels <= 0 or product.channel_width_khz <= 0:
            raise ValueError("spectral products require positive channel counts and widths")
        if product.width_mhz > mode.band_high_mhz - mode.band_low_mhz:
            raise ValueError(f"spectral product {product.name!r} exceeds instrument band")
    return mode


def load_instrument_mode(path: str | Path) -> InstrumentMode:
    """Load an instrument reference configuration from JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return instrument_mode_from_dict(json.load(handle))


DSA_FULL = InstrumentMode(
    name=(
        "Illustrative DSA reference configuration "
        "(deepsynoptic.org/overview-1 snapshot, accessed 12 July 2026)"
    ),
    band_low_mhz=700.0,
    band_high_mhz=2000.0,
    observing_block_minutes=15.0,
    products=(
        SpectralProduct(
            name="hi_zlt1",
            n_channels=5600,
            channel_width_khz=130.0,
            purpose="H I survey to z < 1",
        ),
        SpectralProduct(
            name="hi_nearby_zoom",
            n_channels=4192,
            channel_width_khz=8.0,
            purpose="nearby-galaxy H I zoom product (<100 Mpc)",
        ),
        SpectralProduct(
            name="hi_galactic",
            n_channels=960,
            channel_width_khz=1.0,
            purpose="Galactic H I product",
        ),
    ),
)

DSA110 = InstrumentMode(
    name="DSA-110 public specifications",
    band_low_mhz=1280.0,
    band_high_mhz=1530.0,
    observing_block_minutes=15.0,
    products=(
        SpectralProduct(
            name="full_band",
            n_channels=8192,
            channel_width_khz=(1530.0 - 1280.0) * 1000.0 / 8192.0,
            purpose="public DSA-110 full-band channelisation",
        ),
    ),
)


def observed_hi_frequency_mhz(redshift: float) -> float:
    if redshift <= -1:
        raise ValueError("redshift must be greater than -1")
    return HI_REST_MHZ / (1.0 + redshift)


def redshift_from_frequency_mhz(frequency_mhz: float) -> float:
    if frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be positive")
    return HI_REST_MHZ / frequency_mhz - 1.0
