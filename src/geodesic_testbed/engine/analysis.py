# SPDX-License-Identifier: MPL-2.0
"""Fitting helpers used to turn sweeps into numbers with error bars.

A convergence claim is only as good as the window it was fitted on, so every
fit here records which samples it used and why the others were dropped.  Two
effects routinely corrupt naive log-log fits in this testbed:

* a *roundoff floor* -- once a discretisation error falls below ~1e-15 relative
  it stops shrinking, and including those levels flattens the slope;
* a *higher-order ceiling* -- the leading term of an expansion only dominates
  while it is small, and including coarse samples bends the slope the other way.

Both are handled by an explicit window rather than by eyeballing a plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PowerLawFit:
    """Least-squares fit of ``y = prefactor * x**exponent`` in log-log space."""

    exponent: float
    prefactor: float
    r_squared: float
    n_used: int
    n_total: int
    x_min: float | None
    x_max: float | None
    dropped_below_floor: int
    dropped_above_ceiling: int

    def predict(self, x) -> np.ndarray:
        return self.prefactor * np.asarray(x, dtype=float) ** self.exponent

    def invert(self, y: float) -> float:
        """Value of ``x`` at which the fitted law reaches ``y``."""
        return float((y / self.prefactor) ** (1.0 / self.exponent))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_power_law(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    y_floor: float = 0.0,
    y_ceiling: float = np.inf,
) -> PowerLawFit:
    """Fit a power law on the samples that lie inside ``(y_floor, y_ceiling)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    below = finite & (y <= y_floor)
    above = finite & (y >= y_ceiling)
    keep = finite & ~below & ~above & (y > 0.0)
    n_used = int(keep.sum())
    if n_used < 2:
        return PowerLawFit(
            exponent=float("nan"),
            prefactor=float("nan"),
            r_squared=float("nan"),
            n_used=n_used,
            n_total=int(x.size),
            x_min=None,
            x_max=None,
            dropped_below_floor=int(below.sum()),
            dropped_above_ceiling=int(above.sum()),
        )
    log_x = np.log10(x[keep])
    log_y = np.log10(y[keep])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    residual = log_y - (slope * log_x + intercept)
    total = log_y - log_y.mean()
    denominator = float(np.sum(total**2))
    r_squared = 1.0 if denominator == 0.0 else 1.0 - float(np.sum(residual**2)) / denominator
    return PowerLawFit(
        exponent=float(slope),
        prefactor=float(10.0**intercept),
        r_squared=float(r_squared),
        n_used=n_used,
        n_total=int(x.size),
        x_min=float(x[keep].min()),
        x_max=float(x[keep].max()),
        dropped_below_floor=int(below.sum()),
        dropped_above_ceiling=int(above.sum()),
    )


def successive_orders(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> list[float | None]:
    """Local order between consecutive refinement levels, ``log(e_i/e_j)/log(x_i/x_j)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out: list[float | None] = []
    for index in range(len(x) - 1):
        if y[index] <= 0.0 or y[index + 1] <= 0.0 or x[index] == x[index + 1]:
            out.append(None)
            continue
        out.append(float(np.log(y[index] / y[index + 1]) / np.log(x[index] / x[index + 1])))
    return out


def relative_error(measured, reference) -> np.ndarray:
    """Relative error, falling back to absolute error where the reference vanishes."""
    measured = np.asarray(measured, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = np.abs(reference)
    return np.where(scale > 0.0, np.abs(measured - reference) / np.where(scale > 0.0, scale, 1.0),
                    np.abs(measured - reference))
