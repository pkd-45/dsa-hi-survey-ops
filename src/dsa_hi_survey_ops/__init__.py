__version__ = "0.2.0"

"""Catalogue-aware planning and science-product validation for DSA H I surveys."""

from .catalogue import Target, synthetic_catalogue
from .instrument_modes import (
    DSA110,
    DSA_FULL,
    HI_REST_MHZ,
    InstrumentMode,
    SpectralProduct,
    load_instrument_mode,
)
from .zoom_optimizer import optimise_zoom_band, target_frequency_interval

__all__ = [
    "Target",
    "synthetic_catalogue",
    "DSA110",
    "DSA_FULL",
    "HI_REST_MHZ",
    "InstrumentMode",
    "SpectralProduct",
    "load_instrument_mode",
    "optimise_zoom_band",
    "target_frequency_interval",
]
