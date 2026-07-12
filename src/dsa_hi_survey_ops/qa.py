from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .source_finding import LineMetrics


@dataclass(frozen=True)
class QAMetrics:
    flux_error_percent: float
    centroid_error_kms: float
    w50_error_kms: float
    pa_error_deg: float
    mask_completeness: float
    mask_purity: float
    traffic_light: str

    def to_dict(self) -> dict:
        return asdict(self)


def _angle_difference_deg(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    delta = abs(a - b) % 180.0
    return float(min(delta, 180.0 - delta))


def compare_metrics(
    truth: LineMetrics,
    recovered: LineMetrics,
    truth_mask: np.ndarray,
    recovered_mask: np.ndarray,
) -> QAMetrics:
    flux_error = 100.0 * (recovered.integrated_flux - truth.integrated_flux) / truth.integrated_flux
    centroid_error = abs(recovered.centroid_kms - truth.centroid_kms)
    w50_error = abs(recovered.w50_kms - truth.w50_kms)
    pa_error = _angle_difference_deg(recovered.position_angle_deg, truth.position_angle_deg)
    intersection = np.logical_and(truth_mask, recovered_mask).sum()
    completeness = intersection / max(truth_mask.sum(), 1)
    purity = intersection / max(recovered_mask.sum(), 1)
    checks = [
        abs(flux_error) <= 10.0,
        centroid_error <= 12.0,
        w50_error <= 24.0,
        pa_error <= 12.0,
        completeness >= 0.65,
        purity >= 0.50,
    ]
    if all(checks):
        light = "green"
    elif sum(checks) >= 4:
        light = "amber"
    else:
        light = "red"
    return QAMetrics(
        flux_error_percent=float(flux_error),
        centroid_error_kms=float(centroid_error),
        w50_error_kms=float(w50_error),
        pa_error_deg=float(pa_error),
        mask_completeness=float(completeness),
        mask_purity=float(purity),
        traffic_light=light,
    )
