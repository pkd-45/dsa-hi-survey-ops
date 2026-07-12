from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Target:
    target_id: str
    ra_deg: float
    dec_deg: float
    redshift: float
    redshift_sigma: float
    w50_kms: float
    detect_probability: float
    priority: float

    @property
    def weight(self) -> float:
        return self.detect_probability * self.priority


def synthetic_catalogue(
    n_targets: int = 80,
    seed: int = 123,
    ra_center_deg: float = 150.0,
    dec_center_deg: float = 2.0,
) -> list[Target]:
    """Create a deterministic, application-scale catalogue.

    The redshift distribution intentionally spans more than one nearby-HI zoom
    window so the optimiser must trade target yield, line containment, and RFI.
    """
    if n_targets < 1:
        raise ValueError("n_targets must be positive")
    rng = np.random.default_rng(seed)
    redshift = np.concatenate(
        [
            rng.normal(0.030, 0.005, n_targets // 2),
            rng.normal(0.058, 0.007, n_targets - n_targets // 2),
        ]
    )
    rng.shuffle(redshift)
    redshift = np.clip(redshift, 0.005, 0.085)
    w50 = np.clip(rng.lognormal(np.log(210.0), 0.35, n_targets), 55.0, 520.0)
    z_sigma = rng.uniform(1.0e-5, 1.2e-4, n_targets)
    detect_probability = np.clip(rng.beta(5.0, 2.2, n_targets), 0.15, 0.99)
    priority = rng.choice([1.0, 1.5, 2.0, 3.0], n_targets, p=[0.46, 0.28, 0.18, 0.08])
    ra = ra_center_deg + rng.normal(0.0, 0.55, n_targets)
    dec = dec_center_deg + rng.normal(0.0, 0.55, n_targets)
    return [
        Target(
            target_id=f"G{i:04d}",
            ra_deg=float(ra[i]),
            dec_deg=float(dec[i]),
            redshift=float(redshift[i]),
            redshift_sigma=float(z_sigma[i]),
            w50_kms=float(w50[i]),
            detect_probability=float(detect_probability[i]),
            priority=float(priority[i]),
        )
        for i in range(n_targets)
    ]


def write_catalogue_csv(targets: list[Target], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(t) | {"weight": t.weight} for t in targets]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
