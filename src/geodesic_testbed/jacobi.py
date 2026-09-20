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

from .engine.integrators import integrate_on_grid
from .engine.record import (
    CalibrationBinding,
    FirstOrderValidity,
    Provenance,
    Resolution,
    StartingCovariance,
    TransferRecord,
    Units,
    UpstreamArtefact,
    digest,
)
from .engine.spaceforms import asin_k, sin_k
from .engine.surfaces import TRANSFER_INITIAL_STATE
from .engine.transfer import constant_curvature_transfer

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

    def as_transfer_record(
        self,
        *,
        units: Units | None = None,
        observation_mode: str = "intrinsic-surface-distance",
        relative_tolerance: float = 1.0e-6,
        covariance: StartingCovariance | None = None,
        calibration: CalibrationBinding | None = None,
        upstream: tuple[UpstreamArtefact, ...] = (),
        provenance: Provenance | None = None,
    ) -> TransferRecord:
        """Present this trace as the public transfer record.

        Where the curvature is constant the first-order validity range is not a
        guess: the exact separation expands as
        ``d = eps sn_K(s) [1 - cn_K(s)^2 eps^2/24 + ...]``, so the largest
        heading perturbation holding a given relative tolerance anywhere on the
        path is ``sqrt(24 tol) / max|cn_K(s)|``. That bound is filled in here;
        on a varying profile it is left not established rather than invented.
        """
        constant = bool(np.ptp(self.gaussian_curvature) < 1e-12)
        if constant:
            worst = float(np.max(np.abs(self.position_basis)))
            limit = (
                float(np.sqrt(24.0 * relative_tolerance) / worst) if worst > 0.0 else None
            )
            validity = FirstOrderValidity(
                basis="closed-form: relative error is cn_K(s)^2 eps^2 / 24",
                relative_tolerance=float(relative_tolerance),
                max_heading=limit,
            )
        else:
            validity = FirstOrderValidity.not_established(
                "curvature varies along the path; no closed-form eps^2 coefficient"
            )
        return TransferRecord(
            arclength=self.arclength,
            gaussian_curvature=self.gaussian_curvature,
            a=self.position_basis,
            a_rate=self.position_rate,
            b=self.angle_basis,
            b_rate=self.angle_rate,
            frame="transverse-to-gamma, parallel-transported",
            units=units or Units(),
            source_digest=digest(
                {
                    "curvature": self.gaussian_curvature.tolist()[:1] if constant
                    else "varying",
                    "constant": constant,
                    "span": [float(self.arclength[0]), float(self.arclength[-1])],
                }
            ),
            resolution=Resolution(
                method=self.method,
                samples=int(self.arclength.size),
                max_step=float(np.max(np.diff(self.arclength))),
                uniform=bool(
                    np.allclose(np.diff(self.arclength), self.arclength[1] - self.arclength[0])
                ),
            ),
            validity=validity,
            observation_mode=observation_mode,
            domain="constant-curvature" if constant else "declared-curvature-profile",
            covariance=covariance
            or StartingCovariance.not_declared(
                "computed from a declared curvature profile; no starting pose "
                "distribution exists here"
            ),
            provenance=(
                provenance
                or Provenance(
                    note="Jacobi transfer from a declared curvature profile"
                )
            ).with_upstream(*upstream),
            calibration=calibration
            or CalibrationBinding.unbound("no instrument took part in this computation"),
            # No geometry and no chart, and both absences are the right answer:
            # a curvature profile is not an embedding, so there are no points to
            # sample, no frame to write down in ambient coordinates and no
            # parameterisation to run off the edge of. A consumer needing a
            # position must go to a record built from a surface.
            geometry=None,
            chart=None,
            path_type="geodesic",
            path_type_basis=(
                "declared: K(s) is given as the curvature along a geodesic, which "
                "is what makes j'' + K j = 0 the right equation"
            ),
        )

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
    """Evaluate the exact Jacobi fundamental solutions for constant ``K``.

    Delegates to :func:`geodesic_testbed.engine.transfer.constant_curvature_transfer`,
    which is the same closed form the constant-curvature stage is verified
    against, so this reference and that one cannot drift apart.
    """
    grid = _arclength_grid(arclength)
    K = _finite_scalar(curvature, "curvature")
    exact = constant_curvature_transfer(grid, K)
    return JacobiTrace(
        arclength=grid,
        position_basis=exact.a,
        position_rate=exact.a_rate,
        angle_basis=exact.b,
        angle_rate=exact.b_rate,
        gaussian_curvature=np.full(grid.size, K, dtype=float),
        method="constant-curvature-analytic",
    )


def reference_constant_curvature_trace(arclength: ArrayLike, curvature: float) -> JacobiTrace:
    """A second, deliberately separate, closed-form implementation.

    :func:`constant_curvature_trace` routes through the engine so that the
    application layer and the verified solver cannot drift apart. That leaves
    nothing independent to compare the closed form against -- so this direct
    transcription of ``cos``, ``cosh``, ``sin/omega`` and ``sinh/omega`` is
    kept, and a test holds the two to machine precision. It is not a fallback
    and callers should not use it.
    """
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

    The stepping is the engine's, not a second copy of it: this is
    :func:`geodesic_testbed.engine.integrators.integrate_on_grid` applied to
    the same ``(a, a', b, b')`` state that the surface flow carries, so a
    declared curvature profile and a real surface go through identical
    arithmetic.
    """
    grid = _arclength_grid(arclength)
    K = _curvature_function(curvature)
    states = integrate_on_grid(
        lambda s, state: _rhs(s, state, K),
        np.asarray(TRANSFER_INITIAL_STATE, dtype=float),
        grid,
        method="rk4",
    )
    curvature_samples = np.asarray([K(float(s)) for s in grid], dtype=float)
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

    Evaluated as ``sn_K(d/2) = sn_K(s) sin(delta/2)``, which is the law of
    cosines for all three curvature signs at once and keeps full precision for
    small ``angle_delta`` -- exactly the regime the first-order limit is
    measured in, and exactly where the ``arccos`` and ``arccosh`` forms lose
    most of their digits.
    """
    s = np.asarray(arclength, dtype=float)
    if s.ndim != 1 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("arclength must be a finite, nonnegative vector")
    K = _finite_scalar(curvature, "curvature")
    delta = abs(_finite_scalar(angle_delta, "angle_delta"))
    sign = 1.0 if K > 0.0 else (-1.0 if K < 0.0 else 0.0)
    radius = 1.0 / np.sqrt(abs(K)) if K != 0.0 else 1.0
    scaled = s / radius
    return 2.0 * radius * asin_k(sin_k(scaled, sign) * np.sin(0.5 * delta), sign)
