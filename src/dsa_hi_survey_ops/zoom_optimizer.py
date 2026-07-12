from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .catalogue import Target
from .instrument_modes import C_KMS, HI_REST_MHZ, InstrumentMode, SpectralProduct


@dataclass(frozen=True)
class TargetInterval:
    target_id: str
    low_mhz: float
    high_mhz: float
    center_mhz: float
    weight: float
    w50_kms: float


@dataclass(frozen=True)
class TargetSelection:
    target_id: str
    low_mhz: float
    high_mhz: float
    weight: float
    contained: bool
    overlap_fraction: float
    rfi_overlap_fraction: float
    effective_weight: float
    status: str


@dataclass(frozen=True)
class ZoomPlan:
    instrument: str
    product: str
    center_mhz: float
    low_mhz: float
    high_mhz: float
    width_mhz: float
    n_channels: int
    channel_width_khz: float
    expected_weighted_yield: float
    fully_contained_targets: int
    partial_targets: int
    edge_or_outside_targets: int
    rfi_occupancy_fraction: float
    selections: tuple[TargetSelection, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["selections"] = [asdict(s) for s in self.selections]
        return data


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def target_frequency_interval(
    target: Target,
    barycentric_corrections_kms: Iterable[float],
    guard_each_side_kms: float = 40.0,
    redshift_sigma_multiplier: float = 2.0,
) -> TargetInterval:
    corr = np.asarray(list(barycentric_corrections_kms), dtype=float)
    if corr.size == 0 or not np.all(np.isfinite(corr)):
        raise ValueError("barycentric corrections must be finite and non-empty")
    if guard_each_side_kms < 0:
        raise ValueError("guard_each_side_kms cannot be negative")
    z_lo = max(-0.999999, target.redshift - redshift_sigma_multiplier * target.redshift_sigma)
    z_hi = target.redshift + redshift_sigma_multiplier * target.redshift_sigma
    nu_high_z = HI_REST_MHZ / (1.0 + z_lo)
    nu_low_z = HI_REST_MHZ / (1.0 + z_hi)
    center = HI_REST_MHZ / (1.0 + target.redshift)
    half_line = center * (0.5 * target.w50_kms + guard_each_side_kms) / C_KMS
    corr_low = center * corr.min() / C_KMS
    corr_high = center * corr.max() / C_KMS
    low = nu_low_z + corr_low - half_line
    high = nu_high_z + corr_high + half_line
    return TargetInterval(
        target_id=target.target_id,
        low_mhz=float(low),
        high_mhz=float(high),
        center_mhz=float(0.5 * (low + high)),
        weight=float(target.weight),
        w50_kms=float(target.w50_kms),
    )


def optimise_zoom_band(
    targets: list[Target],
    corrections_kms: Iterable[float],
    instrument: InstrumentMode,
    product: SpectralProduct,
    rfi_bands_mhz: list[tuple[float, float]] | None = None,
    guard_each_side_kms: float = 40.0,
    partial_credit: float = 0.20,
    rfi_penalty: float = 0.80,
    grid_step_mhz: float | None = None,
) -> ZoomPlan:
    """Optimise a fixed-width zoom product over a target catalogue.

    Full line containment receives full target weight. Partial containment gets
    a small configurable credit proportional to the interval overlap. Target
    weight is reduced by overlap with supplied RFI bands, and a separate window
    occupancy penalty discourages centres dominated by unusable spectrum.
    """
    if not targets:
        raise ValueError("targets must be non-empty")
    if not (0 <= partial_credit <= 1):
        raise ValueError("partial_credit must lie in [0, 1]")
    rfi_bands_mhz = rfi_bands_mhz or []
    intervals = [
        target_frequency_interval(t, corrections_kms, guard_each_side_kms)
        for t in targets
    ]
    width = product.width_mhz
    min_center = instrument.band_low_mhz + width / 2
    max_center = instrument.band_high_mhz - width / 2
    if min_center >= max_center:
        raise ValueError("product width exceeds instrument band")
    if grid_step_mhz is None:
        grid_step_mhz = max(product.channel_width_khz / 1000.0 * 32.0, 0.02)
    grid = np.arange(min_center, max_center + 0.5 * grid_step_mhz, grid_step_mhz)
    candidates = np.unique(
        np.concatenate(
            [
                grid,
                np.array([i.center_mhz for i in intervals]),
                np.array([i.low_mhz + width / 2 for i in intervals]),
                np.array([i.high_mhz - width / 2 for i in intervals]),
            ]
        )
    )
    candidates = candidates[(candidates >= min_center) & (candidates <= max_center)]

    total_weight = sum(i.weight for i in intervals)
    best_score = -np.inf
    best_center = float(candidates[0])
    for center in candidates:
        low, high = center - width / 2, center + width / 2
        rfi_occ = sum(_interval_overlap(low, high, a, b) for a, b in rfi_bands_mhz) / width
        score = 0.0
        for interval in intervals:
            span = interval.high_mhz - interval.low_mhz
            overlap = _interval_overlap(low, high, interval.low_mhz, interval.high_mhz)
            frac = min(1.0, overlap / max(span, 1e-12))
            contained = interval.low_mhz >= low and interval.high_mhz <= high
            rfi_target = sum(
                _interval_overlap(interval.low_mhz, interval.high_mhz, a, b)
                for a, b in rfi_bands_mhz
            ) / max(span, 1e-12)
            usable_weight = interval.weight * max(0.0, 1.0 - rfi_target)
            score += usable_weight if contained else partial_credit * frac * usable_weight
        score -= rfi_penalty * rfi_occ * total_weight
        if score > best_score:
            best_score = score
            best_center = float(center)

    low, high = best_center - width / 2, best_center + width / 2
    selections: list[TargetSelection] = []
    for interval in intervals:
        span = interval.high_mhz - interval.low_mhz
        overlap = _interval_overlap(low, high, interval.low_mhz, interval.high_mhz)
        frac = min(1.0, overlap / max(span, 1e-12))
        contained = interval.low_mhz >= low and interval.high_mhz <= high
        rfi_target = min(
            1.0,
            sum(
                _interval_overlap(interval.low_mhz, interval.high_mhz, a, b)
                for a, b in rfi_bands_mhz
            )
            / max(span, 1e-12),
        )
        effective = interval.weight * (1.0 - rfi_target)
        if contained:
            status = "contained"
        elif frac > 0:
            status = "partial"
            effective *= partial_credit * frac
        else:
            status = "outside"
            effective = 0.0
        selections.append(
            TargetSelection(
                target_id=interval.target_id,
                low_mhz=interval.low_mhz,
                high_mhz=interval.high_mhz,
                weight=interval.weight,
                contained=contained,
                overlap_fraction=frac,
                rfi_overlap_fraction=rfi_target,
                effective_weight=effective,
                status=status,
            )
        )
    rfi_occ = min(
        1.0,
        sum(_interval_overlap(low, high, a, b) for a, b in rfi_bands_mhz) / width,
    )
    return ZoomPlan(
        instrument=instrument.name,
        product=product.name,
        center_mhz=best_center,
        low_mhz=low,
        high_mhz=high,
        width_mhz=width,
        n_channels=product.n_channels,
        channel_width_khz=product.channel_width_khz,
        expected_weighted_yield=sum(s.effective_weight for s in selections),
        fully_contained_targets=sum(s.status == "contained" for s in selections),
        partial_targets=sum(s.status == "partial" for s in selections),
        edge_or_outside_targets=sum(s.status == "outside" for s in selections),
        rfi_occupancy_fraction=rfi_occ,
        selections=tuple(selections),
    )
