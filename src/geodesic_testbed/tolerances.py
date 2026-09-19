"""Tolerance propagation through Jacobi fundamental solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .jacobi import JacobiTrace

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PathTolerance:
    """Bounds on one path's starting pose in the transverse plane."""

    lateral: float = 0.0
    heading: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.lateral) or self.lateral < 0.0:
            raise ValueError("lateral tolerance must be finite and nonnegative")
        if not np.isfinite(self.heading) or self.heading < 0.0:
            raise ValueError("heading tolerance must be finite and nonnegative")

    def envelope(self, trace: JacobiTrace, *, combination: str = "worst-case") -> Array:
        lateral = np.abs(trace.position_basis) * self.lateral
        heading = np.abs(trace.angle_basis) * self.heading
        if combination == "worst-case":
            return lateral + heading
        if combination == "rss":
            return np.hypot(lateral, heading)
        raise ValueError("combination must be 'worst-case' or 'rss'")
