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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .integrators import integrate
from .record import (
    FirstOrderValidity,
    Resolution,
    TransferRecord,
    Units,
    digest,
)
from .surfaces import ParametricSurface
from .transfer import FocusEvent, TransferMap


def _chart_report(surface: ParametricSurface, grid, u, v) -> dict[str, Any]:
    """Where, if anywhere, the path left the chart it was computed in.

    Reported rather than raised: a path that runs off the edge of a
    parameterisation is a legitimate thing to have asked for and a useful thing
    to be told about, and the samples before the event are still valid. What is
    not acceptable is returning the samples after it without saying so.
    """
    validity = surface.chart_validity(u, v)
    usable = np.asarray(validity["in_domain"]) & np.asarray(validity["well_conditioned"])
    if bool(np.all(usable)):
        return {
            "valid": True,
            "valid_until": float(grid[-1]),
            "first_invalid_index": None,
            "reason": None,
            "min_conditioning": float(np.min(validity["conditioning"])),
            "min_domain_margin": float(np.min(validity["domain_margin"])),
            "declared_domain": surface.chart.to_dict(),
        }
    index = int(np.argmax(~usable))
    reason = (
        "left the declared parameter domain"
        if not bool(validity["in_domain"][index])
        else "chart became degenerate: EG - F^2 unresolved"
    )
    return {
        "valid": False,
        "valid_until": float(grid[index - 1]) if index > 0 else float(grid[0]),
        "first_invalid_index": index,
        "reason": reason,
        "min_conditioning": float(np.min(validity["conditioning"])),
        "min_domain_margin": float(np.min(validity["domain_margin"])),
        "declared_domain": surface.chart.to_dict(),
    }


def _validated_epsilons(epsilon) -> np.ndarray:
    """Differencing steps must be finite and strictly positive.

    A zero or a NaN divides silently into the separation and produces an
    infinity or a NaN that looks like a Jacobi field, so it is refused here
    rather than discovered three plots later.
    """
    epsilons = np.atleast_1d(np.asarray(epsilon, dtype=float))
    if epsilons.size == 0:
        raise ValueError("epsilon must contain at least one value")
    if not np.all(np.isfinite(epsilons)):
        raise ValueError("epsilon must be finite")
    if np.any(epsilons <= 0.0):
        raise ValueError("epsilon must be strictly positive")
    return epsilons


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
    lateral_basis: np.ndarray
    lateral_rate: np.ndarray
    jacobi_field: np.ndarray
    jacobi_derivative: np.ndarray
    speed: np.ndarray
    chart: dict[str, Any] = field(default_factory=dict)

    # -- readout -----------------------------------------------------------
    @property
    def transfer_map(self) -> TransferMap:
        """``Phi(s)``: both columns of the starting-pose sensitivity."""
        return TransferMap(
            arc_length=self.arc_length,
            a=self.lateral_basis,
            a_rate=self.lateral_rate,
            b=self.jacobi_field,
            b_rate=self.jacobi_derivative,
        )

    def as_transfer_record(
        self,
        *,
        units: Units | None = None,
        observation_mode: str = "ambient-euclidean-chord",
        validity: FirstOrderValidity | None = None,
    ) -> TransferRecord:
        """Present this path as the public transfer record.

        The default observation mode is the ambient chord, because that is what
        this repository can actually compute on a general surface: the
        in-surface distance between two nearby geodesics would be a
        boundary-value problem, and claiming it here would be claiming a
        quantity nothing produces.

        Validity is declared not established by default, for the same reason --
        the ``eps^2`` coefficient is known in closed form only on the constant
        curvature model spaces. A caller who has measured it can pass one in.
        """
        steps = int(self.n_steps)
        return TransferRecord(
            arclength=self.arc_length,
            gaussian_curvature=self.curvature,
            a=self.lateral_basis,
            a_rate=self.lateral_rate,
            b=self.jacobi_field,
            b_rate=self.jacobi_derivative,
            frame="transverse-to-gamma, parallel-transported",
            units=units or Units(),
            source_digest=digest(
                {
                    "surface": self.surface.name,
                    "description": self.surface.description,
                    "start": {"u": self.start[0], "v": self.start[1], "heading": self.start[2]},
                    "length": self.length,
                }
            ),
            resolution=Resolution(
                method=self.integrator,
                samples=steps + 1,
                max_step=float(self.length) / steps,
                uniform=True,
            ),
            validity=validity
            or FirstOrderValidity.not_established(
                "no closed form for the eps^2 coefficient on a varying-curvature surface"
            ),
            observation_mode=observation_mode,
            domain="parametric-surface",
        )

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

    def focus_points(self, *, component: str = "b") -> list[float]:
        """Arc lengths at which a column of the transfer map vanishes.

        Delegates to :meth:`TransferMap.focus_events`, so these are the
        Hermite-refined locations rather than the sample-spacing ones, and
        there is a single implementation of the root finding.
        """
        return self.transfer_map.focus_points(component=component)

    def focus_events(self, *, component: str = "b") -> list[FocusEvent]:
        """The same foci, with their location uncertainty and slope."""
        return self.transfer_map.focus_events(component=component)

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
            "chart": self.chart,
            "focus_points": focus,
            "focus_events": [event.to_dict() for event in self.focus_events()],
            "focus_clearance": self.transfer_map.clearance_from_focus(),
            "crossed_first_conjugate_point": (
                self.transfer_map.crossed_first_conjugate_point()
            ),
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
    # Refuse a start the chart cannot represent, rather than integrating out of
    # a singularity and returning something that looks like an envelope.
    surface.require_valid_chart(u0, v0, where="the starting point")
    starts = np.stack(
        [surface.initial_state(u0, v0, float(angle)) for angle in angles], axis=0
    )
    grid, trajectory = integrate(
        surface.geodesic_transfer_rhs(), starts, length=length, n_steps=n_steps, method=method
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
                lateral_basis=trajectory[:, index, 4],
                lateral_rate=trajectory[:, index, 5],
                jacobi_field=trajectory[:, index, 6],
                jacobi_derivative=trajectory[:, index, 7],
                speed=np.asarray(surface.speed(u, v, du, dv), dtype=float),
                chart=_chart_report(surface, grid, u, v),
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
    epsilons = _validated_epsilons(epsilon)
    offsets = np.concatenate([heading + epsilons, heading - epsilons])
    starts = np.stack(
        [surface.initial_state(u0, v0, float(angle)) for angle in offsets], axis=0
    )
    grid, trajectory = integrate(
        surface.geodesic_transfer_rhs(), starts, length=length, n_steps=n_steps, method=method
    )
    count = len(epsilons)
    forward = surface.embed(trajectory[:, :count, 0], trajectory[:, :count, 1])
    backward = surface.embed(trajectory[:, count:, 0], trajectory[:, count:, 1])
    separation = np.sqrt(np.sum((forward - backward) ** 2, axis=-1))
    measured = separation / (2.0 * epsilons)
    return grid, (measured[:, 0] if np.ndim(epsilon) == 0 else measured)


def finite_difference_lateral(
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
    """``|d gamma / d lateral offset|`` by central-differencing the *start point*.

    The independent check on the first column of the transfer map, and a
    different construction from the heading one. The start is moved a distance
    ``epsilon`` to either side along the geodesic perpendicular to the path,
    and the initial direction at the moved start is the parallel transport of
    the original one along that perpendicular. Parallel transport along a
    geodesic preserves the angle to it, and rotation by a right angle commutes
    with transport, so the transported direction is simply the perpendicular of
    the perpendicular geodesic's own tangent at the displaced point -- with the
    sign that undoes the rotation, opposite on the two sides.

    Returns ``(s, |J|)``, the sweep on the second axis when ``epsilon`` is an
    array. The measured quantity is an ``ambient-euclidean-chord``.
    """
    epsilons = _validated_epsilons(epsilon)
    rhs = surface.geodesic_transfer_rhs()
    tangent = surface.unit_direction(u0, v0, heading)
    normal = surface.perpendicular_direction(u0, v0, *tangent)

    starts = []
    for offset in epsilons:
        for sign in (1.0, -1.0):
            side = surface.state_from_tangent(u0, v0, sign * normal[0], sign * normal[1])
            _, walk = integrate(rhs, side, length=float(offset), n_steps=n_steps, method=method)
            u, v, du, dv = (float(walk[-1, index]) for index in range(4))
            transported = surface.perpendicular_direction(u, v, du, dv)
            # +side started along +normal, so undoing the rotation flips the sign.
            direction = (-sign * transported[0], -sign * transported[1])
            starts.append(surface.state_from_tangent(u, v, *direction))

    grid, trajectory = integrate(
        rhs, np.stack(starts, axis=0), length=length, n_steps=n_steps, method=method
    )
    forward = surface.embed(trajectory[:, 0::2, 0], trajectory[:, 0::2, 1])
    backward = surface.embed(trajectory[:, 1::2, 0], trajectory[:, 1::2, 1])
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
    dead_zone_fraction: float = 0.2,
) -> list[dict[str, Any]]:
    """Rank candidate starting headings by forward angular-error amplification.

    This measures one thing: ``max |b(s)|``, how far an initial aiming error is
    carried. That is **not** the same as robustness, and the scan reports
    enough to see the difference.

    A small ``|b|`` away from the start is a *focus*: neighbouring geodesics
    converge there. Forward separation is indeed small, but the map from
    starting heading to endpoint is ill conditioned, the path is at or past a
    conjugate point, local minimality can be lost, and a family of such paths
    crowds together instead of covering. A route chosen purely by minimum
    amplification will walk straight into one.

    So each row carries both an upper measure (``max_forward_amplification``)
    and a lower one (``focus_margin``: the smallest ``|b|`` after the initial
    dead zone, where ``b`` is small only because it starts at zero). Rows are
    ordered by amplification; a production objective has to weigh both, and
    also boundary clearance, chart validity, path length and coverage, none of
    which are modelled here.
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
        after_dead_zone = envelope.arc_length >= dead_zone_fraction * length
        focus_points = envelope.focus_points()
        rows.append(
            {
                "heading": float(heading),
                "heading_degrees": float(np.rad2deg(heading)),
                "max_forward_amplification": float(np.max(np.abs(envelope.jacobi_field))),
                "focus_margin": float(np.min(np.abs(envelope.jacobi_field[after_dead_zone]))),
                "dead_zone_fraction": float(dead_zone_fraction),
                "jacobi_field_at_end": float(envelope.jacobi_field[-1]),
                "amplification_at_end": float(envelope.amplification[-1]),
                "max_lateral_amplification": float(np.max(np.abs(envelope.lateral_basis))),
                "max_wronskian_drift": float(np.max(envelope.transfer_map.wronskian_drift)),
                "mean_curvature_along_path": float(np.mean(envelope.curvature)),
                "focus_points": focus_points,
                "passes_a_focus": bool(focus_points),
                "max_speed_drift": float(np.max(envelope.speed_drift)),
            }
        )
    return sorted(rows, key=lambda row: row["max_forward_amplification"])
