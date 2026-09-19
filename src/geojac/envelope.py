"""The instrument's output: a sensitivity envelope around a nominal path.

Input is a surface, a starting point, a starting heading and a tolerance on how
accurately that heading can be set. Output is, at every arc length along the
path: the local Gaussian curvature, the Jacobi field, how much an initial
aiming error has been amplified, how large an aiming error the tolerance still
allows, and a flag wherever the path passes a focus of neighbouring geodesics
and the path-to-endpoint map stops being well conditioned.

Two independent routes to the Jacobi field are provided, because on a surface
of varying curvature there is no closed form to check either of them against:

``integrate_path``
    solves ``j'' + K(gamma(s)) j = 0`` along the path, coupled to the geodesic;

``finite_difference_jacobi``
    central-differences the flow itself in the initial heading.

The second converges to the first as ``eps -> 0`` with a relative error of
``eps^2/6`` on any surface of constant curvature -- the immersion is isometric,
so the ambient length of the variation field is its length in the surface --
and their agreement is what makes the first trustworthy where nothing else can
check it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .integrators import integrate
from .surfaces import ParametricSurface


@dataclass(frozen=True)
class PathEnvelope:
    """A nominal path and everything the sensitivity readout needs."""

    surface: ParametricSurface
    start: tuple[float, float, float]
    length: float
    n_steps: int
    integrator: str
    arc_length: np.ndarray
    u: np.ndarray
    v: np.ndarray
    points: np.ndarray
    curvature: np.ndarray
    jacobi_field: np.ndarray
    jacobi_derivative: np.ndarray
    speed: np.ndarray

    # -- readout -----------------------------------------------------------
    @property
    def speed_drift(self) -> np.ndarray:
        """How far the unit-speed condition has slipped; nothing re-normalises it."""
        return np.abs(self.speed - 1.0)

    @property
    def amplification(self) -> np.ndarray:
        """``j(s)/s``: aiming error growth relative to a flat surface."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.arc_length > 0.0, self.jacobi_field / self.arc_length, 1.0)

    @property
    def worst_so_far(self) -> np.ndarray:
        """Running maximum of ``|j|``: what a tolerance along the whole path must survive."""
        return np.maximum.accumulate(np.abs(self.jacobi_field))

    def transverse_envelope(self, heading_tolerance: float) -> np.ndarray:
        """First-order half-width of the tube of paths reachable within the tolerance."""
        return float(heading_tolerance) * np.abs(self.jacobi_field)

    def heading_budget(self, transverse_tolerance: float, *, cumulative: bool = True) -> np.ndarray:
        """Largest initial heading error keeping the deviation inside ``transverse_tolerance``.

        With ``cumulative`` the budget is governed by the worst point reached so
        far, which is what a path that must stay inside a tolerance everywhere
        actually requires; without it, by the deviation at that point alone.
        """
        scale = self.worst_so_far if cumulative else np.abs(self.jacobi_field)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(scale > 0.0, float(transverse_tolerance) / scale, np.inf)

    def focus_points(self) -> list[float]:
        """Arc lengths at which the Jacobi field vanishes away from the start.

        These are conjugate points: neighbouring geodesics refocus, the first
        order deviation carries no information about which one you are on, and
        the path-to-endpoint map is ill conditioned there.
        """
        field = self.jacobi_field
        grid = self.arc_length
        found: list[float] = []
        for index in range(1, len(field) - 1):
            if grid[index] <= 0.0:
                continue
            left, right = field[index], field[index + 1]
            if left == 0.0:
                found.append(float(grid[index]))
            elif left * right < 0.0:
                weight = left / (left - right)
                found.append(float(grid[index] + weight * (grid[index + 1] - grid[index])))
        return found

    def summary(self) -> dict[str, Any]:
        focus = self.focus_points()
        return {
            "surface": self.surface.name,
            "description": self.surface.description,
            "start": {"u": self.start[0], "v": self.start[1], "heading": self.start[2]},
            "length": float(self.length),
            "n_steps": int(self.n_steps),
            "integrator": self.integrator,
            "curvature_along_path": {
                "min": float(np.min(self.curvature)),
                "max": float(np.max(self.curvature)),
                "varies": bool(np.ptp(self.curvature) > 1e-12),
            },
            "jacobi_field_at_end": float(self.jacobi_field[-1]),
            "amplification_at_end": float(self.amplification[-1]),
            "max_abs_jacobi_field": float(np.max(np.abs(self.jacobi_field))),
            "max_speed_drift": float(np.max(self.speed_drift)),
            "focus_points": focus,
            "well_conditioned": not focus,
        }


def integrate_paths(
    surface: ParametricSurface,
    *,
    u0: float,
    v0: float,
    headings,
    length: float,
    n_steps: int,
    method: str = "rk4",
) -> list[PathEnvelope]:
    """Flow a fan of geodesics and their Jacobi fields in one pass.

    A whole fan costs barely more than a single path: the state is small and
    the expense is the per-step Python overhead, which the fan shares.
    """
    angles = np.atleast_1d(np.asarray(headings, dtype=float))
    starts = np.stack(
        [surface.initial_state(u0, v0, float(angle)) for angle in angles], axis=0
    )
    grid, trajectory = integrate(
        surface.geodesic_jacobi_rhs(), starts, length=length, n_steps=n_steps, method=method
    )
    envelopes = []
    for index, angle in enumerate(angles):
        u, v = trajectory[:, index, 0], trajectory[:, index, 1]
        du, dv = trajectory[:, index, 2], trajectory[:, index, 3]
        envelopes.append(
            PathEnvelope(
                surface=surface,
                start=(float(u0), float(v0), float(angle)),
                length=float(length),
                n_steps=int(n_steps),
                integrator=method,
                arc_length=grid,
                u=u,
                v=v,
                points=surface.embed(u, v),
                curvature=np.asarray(surface.gaussian_curvature(u, v), dtype=float),
                jacobi_field=trajectory[:, index, 4],
                jacobi_derivative=trajectory[:, index, 5],
                speed=np.asarray(surface.speed(u, v, du, dv), dtype=float),
            )
        )
    return envelopes


def integrate_path(
    surface: ParametricSurface,
    *,
    u0: float,
    v0: float,
    heading: float = 0.0,
    length: float,
    n_steps: int,
    method: str = "rk4",
) -> PathEnvelope:
    """Flow one geodesic and its Jacobi field together along ``surface``."""
    return integrate_paths(
        surface,
        u0=u0,
        v0=v0,
        headings=[heading],
        length=length,
        n_steps=n_steps,
        method=method,
    )[0]


def finite_difference_jacobi(
    surface: ParametricSurface,
    *,
    u0: float,
    v0: float,
    heading: float = 0.0,
    epsilon,
    length: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray]:
    """``|d gamma / d heading|`` by a central difference of the flow itself.

    ``epsilon`` may be a scalar or an array; a whole sweep is flowed in one
    pass.  Returns ``(s, |J|)``, with the sweep on the second axis when
    ``epsilon`` is an array.  This never touches the Jacobi equation, so it is
    a genuinely independent measurement of the same quantity.
    """
    epsilons = np.atleast_1d(np.asarray(epsilon, dtype=float))
    offsets = np.concatenate([heading + epsilons, heading - epsilons])
    starts = np.stack(
        [surface.initial_state(u0, v0, float(angle)) for angle in offsets], axis=0
    )
    grid, trajectory = integrate(
        surface.geodesic_jacobi_rhs(), starts, length=length, n_steps=n_steps, method=method
    )
    count = len(epsilons)
    forward = surface.embed(trajectory[:, :count, 0], trajectory[:, :count, 1])
    backward = surface.embed(trajectory[:, count:, 0], trajectory[:, count:, 1])
    separation = np.sqrt(np.sum((forward - backward) ** 2, axis=-1))
    measured = separation / (2.0 * epsilons)
    return grid, (measured[:, 0] if np.ndim(epsilon) == 0 else measured)


def scan_headings(
    surface: ParametricSurface,
    *,
    u0: float,
    v0: float,
    headings: np.ndarray,
    length: float,
    n_steps: int,
    method: str = "rk4",
) -> list[dict[str, Any]]:
    """Sensitivity of every candidate starting heading, worst first.

    The cheapest useful decision this instrument supports: of the directions a
    scan or a tow could be laid in, which one carries an aiming error the least
    far.
    """
    rows: list[dict[str, Any]] = []
    for envelope in integrate_paths(
        surface,
        u0=u0,
        v0=v0,
        headings=headings,
        length=length,
        n_steps=n_steps,
        method=method,
    ):
        heading = envelope.start[2]
        rows.append(
            {
                "heading": float(heading),
                "heading_degrees": float(np.rad2deg(heading)),
                "max_abs_jacobi_field": float(np.max(np.abs(envelope.jacobi_field))),
                "jacobi_field_at_end": float(envelope.jacobi_field[-1]),
                "amplification_at_end": float(envelope.amplification[-1]),
                "mean_curvature_along_path": float(np.mean(envelope.curvature)),
                "focus_points": envelope.focus_points(),
                "max_speed_drift": float(np.max(envelope.speed_drift)),
            }
        )
    return sorted(rows, key=lambda row: row["max_abs_jacobi_field"])
