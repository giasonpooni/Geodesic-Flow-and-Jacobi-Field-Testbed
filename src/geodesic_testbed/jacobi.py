"""Analytical and numerical transverse Jacobi-field propagation.

For a unit-speed geodesic on a surface, the scalar transverse variation
obeys

    j''(s) + K(s) j(s) = 0.

Two fundamental solutions are propagated:

* ``a``: ``a(0)=1, a'(0)=0`` for initial lateral displacement;
* ``b``: ``b(0)=0, b'(0)=1`` for initial heading displacement.

Any first-order transverse variation is ``j = a*j0 + b*j0_prime``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
Curvature = float | Callable[[float], float]


def _readonly(values: ArrayLike) -> Array:
    array = np.asarray(values, dtype=float).copy()
    array.setflags(write=False)
    return array


def _arclength_grid(values: ArrayLike) -> Array:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("arclength must be a one-dimensional grid with at least two samples")
    if not np.all(np.isfinite(grid)):
        raise ValueError("arclength must be finite")
    if grid[0] != 0.0:
        raise ValueError("arclength must start at zero")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("arclength must be strictly increasing")
    return grid


def _finite_scalar(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class JacobiTrace:
    """Fundamental transverse variations sampled along one geodesic."""

    arclength: Array
    position_basis: Array
    position_rate: Array
    angle_basis: Array
    angle_rate: Array
    gaussian_curvature: Array
    method: str

    def __post_init__(self) -> None:
        arrays = (
            self.arclength,
            self.position_basis,
            self.position_rate,
            self.angle_basis,
            self.angle_rate,
            self.gaussian_curvature,
        )
        size = np.asarray(self.arclength).size
        if any(np.asarray(value).shape != (size,) for value in arrays):
            raise ValueError("every trace field must be a vector on the arclength grid")
        for name in (
            "arclength",
            "position_basis",
            "position_rate",
            "angle_basis",
            "angle_rate",
            "gaussian_curvature",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def separation(self, initial_offset: float, initial_angle: float) -> Array:
        """Return the signed first-order transverse separation.

        ``initial_offset`` has units of length. ``initial_angle`` is the
        dimensionless small-angle perturbation in radians.
        """
        offset = _finite_scalar(initial_offset, "initial_offset")
        angle = _finite_scalar(initial_angle, "initial_angle")
        return self.position_basis * offset + self.angle_basis * angle

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "arclength": self.arclength.tolist(),
            "gaussian_curvature": self.gaussian_curvature.tolist(),
            "position_basis": self.position_basis.tolist(),
            "position_rate": self.position_rate.tolist(),
            "angle_basis": self.angle_basis.tolist(),
            "angle_rate": self.angle_rate.tolist(),
        }


def constant_curvature_trace(arclength: ArrayLike, curvature: float) -> JacobiTrace:
    """Evaluate the exact Jacobi fundamental solutions for constant ``K``."""
    grid = _arclength_grid(arclength)
    K = _finite_scalar(curvature, "curvature")
    if K > 0.0:
        omega = np.sqrt(K)
        position = np.cos(omega * grid)
        position_rate = -omega * np.sin(omega * grid)
        angle = np.sin(omega * grid) / omega
        angle_rate = np.cos(omega * grid)
    elif K < 0.0:
        omega = np.sqrt(-K)
        position = np.cosh(omega * grid)
        position_rate = omega * np.sinh(omega * grid)
        angle = np.sinh(omega * grid) / omega
        angle_rate = np.cosh(omega * grid)
    else:
        position = np.ones_like(grid)
        position_rate = np.zeros_like(grid)
        angle = grid.copy()
        angle_rate = np.ones_like(grid)
    return JacobiTrace(
        arclength=grid,
        position_basis=position,
        position_rate=position_rate,
        angle_basis=angle,
        angle_rate=angle_rate,
        gaussian_curvature=np.full_like(grid, K),
        method="constant-curvature-analytic",
    )


def _curvature_function(curvature: Curvature) -> Callable[[float], float]:
    if callable(curvature):
        def checked(s: float) -> float:
            return _finite_scalar(curvature(s), "curvature(s)")

        return checked
    value = _finite_scalar(curvature, "curvature")
    return lambda _s: value


def _rhs(s: float, state: Array, curvature: Callable[[float], float]) -> Array:
    K = curvature(s)
    a, da, b, db = state
    return np.asarray([da, -K * a, db, -K * b], dtype=float)


def integrate_jacobi(arclength: ArrayLike, curvature: Curvature) -> JacobiTrace:
    """Integrate the two Jacobi basis fields with fixed-output-step RK4.

    The caller controls the integration resolution through ``arclength``.
    A later mesh/CAD adapter can supply sampled Gaussian curvature as an
    interpolating callable without changing the application contracts.
    """
    grid = _arclength_grid(arclength)
    K = _curvature_function(curvature)
    states = np.empty((grid.size, 4), dtype=float)
    states[0] = np.asarray([1.0, 0.0, 0.0, 1.0])
    curvature_samples = np.empty(grid.size, dtype=float)
    curvature_samples[0] = K(0.0)
    for index in range(grid.size - 1):
        s0 = float(grid[index])
        h = float(grid[index + 1] - grid[index])
        y0 = states[index]
        k1 = _rhs(s0, y0, K)
        k2 = _rhs(s0 + 0.5 * h, y0 + 0.5 * h * k1, K)
        k3 = _rhs(s0 + 0.5 * h, y0 + 0.5 * h * k2, K)
        k4 = _rhs(s0 + h, y0 + h * k3, K)
        states[index + 1] = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        curvature_samples[index + 1] = K(float(grid[index + 1]))
    return JacobiTrace(
        arclength=grid,
        position_basis=states[:, 0],
        position_rate=states[:, 1],
        angle_basis=states[:, 2],
        angle_rate=states[:, 3],
        gaussian_curvature=curvature_samples,
        method="rk4",
    )


def finite_angular_separation(
    arclength: ArrayLike,
    curvature: float,
    angle_delta: float,
) -> Array:
    """Exact finite distance between rays with a shared origin and angle.

    This reference is available for constant-curvature model spaces. It is
    used to measure where the first-order prediction
    ``abs(angle_delta) * angle_basis`` ceases to approximate finite path
    separation.
    """
    s = np.asarray(arclength, dtype=float)
    if s.ndim != 1 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("arclength must be a finite, nonnegative vector")
    K = _finite_scalar(curvature, "curvature")
    delta = abs(_finite_scalar(angle_delta, "angle_delta"))
    if K > 0.0:
        radius = 1.0 / np.sqrt(K)
        scaled = s / radius
        cosine = np.cos(scaled) ** 2 + np.sin(scaled) ** 2 * np.cos(delta)
        return radius * np.arccos(np.clip(cosine, -1.0, 1.0))
    if K < 0.0:
        radius = 1.0 / np.sqrt(-K)
        scaled = s / radius
        cosh_distance = np.cosh(scaled) ** 2 - np.sinh(scaled) ** 2 * np.cos(delta)
        return radius * np.arccosh(np.maximum(cosh_distance, 1.0))
    return 2.0 * s * np.sin(0.5 * delta)
