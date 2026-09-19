"""The Jacobi equation and the two ways this testbed produces a Jacobi field.

For a unit-speed geodesic on a surface of constant curvature ``K``, a
transverse Jacobi field has scalar component ``j`` obeying

.. math::

    j''(s) + K\\, j(s) = 0, \\qquad j(0) = 0,\\ j'(0) = 1,

whose solutions are ``s``, ``sin s`` and ``sinh s`` for ``K = 0, +1, -1``.
Two independent numerical routes to that same object are implemented here:

``integrate_jacobi``
    integrates the scalar ODE directly;

``numerical_separation``
    integrates two nearby geodesics through the ambient flow and measures the
    Riemannian distance between them.

The second route returns a *finite* separation.  ``j`` predicts it only to
first order in the initial angle ``epsilon``; the size of the gap between the
two is the subject of the perturbation sweep, and is the boundary the project
README insists on keeping explicit.
"""

from __future__ import annotations

import numpy as np

from .integrators import Rhs, integrate
from .spaceforms import SpaceForm, sin_k


def jacobi_rhs(K: float) -> Rhs:
    """Right-hand side of ``(j, j')' = (j', -K j)``."""

    def rhs(_s: float, y: np.ndarray) -> np.ndarray:
        return np.stack([y[..., 1], -K * y[..., 0]], axis=-1)

    return rhs


def jacobi_reference(s, K: float) -> np.ndarray:
    """Closed-form solution ``sn_K(s)``: ``s``, ``sin s`` or ``sinh s``."""
    return sin_k(s, K)


def integrate_jacobi(
    K: float,
    *,
    length: float,
    n_steps: int,
    method: str = "rk4",
    j0: float = 0.0,
    jdot0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the Jacobi equation; return ``(s, j, j')``."""
    grid, trajectory = integrate(
        jacobi_rhs(K),
        np.array([j0, jdot0], dtype=float),
        length=length,
        n_steps=n_steps,
        method=method,
    )
    return grid, trajectory[..., 0], trajectory[..., 1]


def integrate_geodesic(
    form: SpaceForm,
    *,
    direction_angle: float = 0.0,
    length: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the ambient geodesic flow from ``p0``; return ``(s, points, velocities)``."""
    p0 = form.base_point()
    v0 = form.rotated_direction(direction_angle)
    grid, trajectory = integrate(
        form.flow_rhs(),
        form.flow_state(p0, v0),
        length=length,
        n_steps=n_steps,
        method=method,
    )
    return grid, trajectory[..., :3], trajectory[..., 3:]


def numerical_separation(
    form: SpaceForm,
    epsilon: float,
    *,
    length: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray]:
    """Distance between two numerically flowed geodesics separated by angle ``epsilon``.

    Both start at ``p0``; the second leaves rotated by ``epsilon`` in the
    tangent plane.  Returns ``(s, distance)``.
    """
    grid, base_points, _ = integrate_geodesic(
        form, direction_angle=0.0, length=length, n_steps=n_steps, method=method
    )
    _, perturbed_points, _ = integrate_geodesic(
        form, direction_angle=epsilon, length=length, n_steps=n_steps, method=method
    )
    return grid, form.distance(base_points, perturbed_points)


def geodesic_position_error(
    form: SpaceForm,
    grid: np.ndarray,
    points: np.ndarray,
    *,
    direction_angle: float = 0.0,
) -> np.ndarray:
    """Ambient error of a flowed geodesic against the closed-form exponential map."""
    p0 = form.base_point()
    v0 = form.rotated_direction(direction_angle)
    exact, _ = form.exp(p0, v0, grid)
    delta = np.asarray(points, dtype=float) - exact
    return np.sqrt(np.sum(delta * delta, axis=-1))


def integrate_geodesic_bundle(
    form: SpaceForm,
    direction_angles,
    *,
    length: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flow a whole fan of geodesics from ``p0`` at once.

    ``direction_angles`` are angles in the tangent plane at ``p0``.  All
    trajectories share one step sequence, so differences between them carry no
    relative timing error.  Returns ``(s, points, velocities)`` with the fan
    on the second axis.
    """
    angles = np.atleast_1d(np.asarray(direction_angles, dtype=float))
    p0 = form.base_point()
    starts = np.stack(
        [form.flow_state(p0, form.rotated_direction(float(a))) for a in angles], axis=0
    )
    grid, trajectory = integrate(
        form.flow_rhs(), starts, length=length, n_steps=n_steps, method=method
    )
    return grid, trajectory[..., :3], trajectory[..., 3:]
