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

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .contract import validated_covariance
from .integrators import get_integrator, integrate, integrate_guarded
from .record import (
    PERTURBATION_DIRECTIONS,
    CalibrationBinding,
    ChartValidity,
    ConvergenceEstimate,
    GeometryUncertainty,
    PathGeometry,
    ProbeFit,
    Provenance,
    Resolution,
    StartingCovariance,
    TransferRecord,
    Units,
    UpstreamArtefact,
    ValidityEnvelope,
    digest,
)
from .surfaces import ParametricSurface, darboux_frame
from .transfer import FocusEvent, TransferMap


def _chart_report(
    surface: ParametricSurface,
    grid,
    u,
    v,
    *,
    stopped_at: int | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Where, if anywhere, the path left the chart it was computed in.

    Reported rather than raised: a path that runs off the edge of a
    parameterisation is a legitimate thing to have asked for and a useful thing
    to be told about, and the samples before the event are still valid. What is
    not acceptable is returning the samples after it without saying so.

    ``stopped_at`` is for the guarded flow, where the integrator stopped the
    path at the exit and froze it. The frozen samples are copies of the last
    valid state and would pass a validity check, so the exit has to be carried
    in rather than rediscovered -- the whole point of stopping was not to
    compute the invalid ones.
    """
    validity = surface.chart_validity(u, v)
    usable = np.asarray(validity["in_domain"]) & np.asarray(validity["well_conditioned"])
    if stopped_at is not None:
        usable = usable.copy()
        usable[stopped_at:] = False
    if bool(np.all(usable)):
        return {
            "valid": True,
            "valid_until": float(grid[-1]),
            "first_invalid_index": None,
            "reason": None,
            "truncated": False,
            "min_conditioning": float(np.min(validity["conditioning"])),
            "min_domain_margin": float(np.min(validity["domain_margin"])),
            "declared_domain": surface.chart.to_dict(),
        }
    index = int(np.argmax(~usable))
    if reason is None:
        reason = (
            "left the declared parameter domain"
            if not bool(validity["in_domain"][index])
            else "chart became degenerate: EG - F^2 unresolved"
        )
    # Conditioning is reported over the prefix that is still usable. Taken over
    # the whole run it would describe the region the path should never have
    # been integrated into, which is a number about nothing.
    kept = slice(0, max(index, 1))
    return {
        "valid": False,
        "valid_until": float(grid[index - 1]) if index > 0 else float(grid[0]),
        "first_invalid_index": index,
        "reason": reason,
        "truncated": False,
        "min_conditioning": float(np.min(validity["conditioning"][kept])),
        "min_domain_margin": float(np.min(validity["domain_margin"][kept])),
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
    #: Parameter velocities. Kept because the ambient tangent cannot be
    #: recovered from ``u, v`` alone without differencing them, and a
    #: differenced tangent is a different quantity at the order that matters.
    du: np.ndarray
    dv: np.ndarray
    points: np.ndarray
    curvature: np.ndarray
    lateral_basis: np.ndarray
    lateral_rate: np.ndarray
    jacobi_field: np.ndarray
    jacobi_derivative: np.ndarray
    speed: np.ndarray
    chart: dict[str, Any] = field(default_factory=dict)

    # -- chart validity ----------------------------------------------------
    @property
    def chart_valid(self) -> bool:
        """Whether every sample of this path lies in a usable part of the chart."""
        return bool(self.chart.get("valid", True))

    @property
    def valid_samples(self) -> int:
        """How many leading samples are usable. The rest are not results.

        Past the first invalid sample the metric is degenerate or the point is
        outside the declared domain, so the geodesic equation was integrated
        through coefficients that do not describe the surface. Those samples
        are not a worse answer; they are not an answer.
        """
        index = self.chart.get("first_invalid_index")
        return int(self.arc_length.size) if index is None else int(index)

    def truncated_to_chart(self) -> PathEnvelope:
        """The leading prefix of this path that stayed inside the chart.

        Returned as a whole envelope rather than as a mask, so that everything
        computed from it -- the transfer map, the foci, the record -- is
        computed from valid samples only and cannot silently reach past the
        exit. An unbroken path is returned unchanged.
        """
        if self.chart_valid:
            return self
        keep = self.valid_samples
        if keep < 2:
            raise ValueError(
                f"the path left the chart of {self.surface.name} at sample {keep} "
                f"({self.chart.get('reason')}); there is no valid prefix to keep. "
                "Start elsewhere, or shorten the path"
            )
        sliced = {
            name: np.asarray(getattr(self, name))[:keep]
            for name in (
                "arc_length", "u", "v", "du", "dv", "points", "curvature",
                "lateral_basis", "lateral_rate", "jacobi_field", "jacobi_derivative",
                "speed",
            )
        }
        grid = sliced["arc_length"]
        # Every sample that remains is inside the chart, so the truncated
        # envelope *is* valid. What it is not is the path that was asked for,
        # and that stays on the record: ``valid`` describes the samples in
        # hand, ``truncated`` describes what happened to the request.
        return replace(
            self,
            length=float(grid[-1] - grid[0]),
            n_steps=int(keep - 1),
            chart=self.chart | {
                "valid": True,
                "valid_until": float(grid[-1]),
                "first_invalid_index": None,
                "truncated": True,
                "truncated_at": float(grid[-1]),
                "truncation_reason": self.chart.get("reason"),
                "truncated_from_length": float(self.length),
                "truncated_from_samples": int(self.arc_length.size),
            },
            **sliced,
        )

    # -- the path, in the form that crosses the boundary -------------------
    def path_geometry(
        self,
        *,
        coordinate_frame: str = "surface-parameterisation-ambient",
        datum_frame: str = "not-declared",
        uncertainty: GeometryUncertainty | None = None,
    ) -> PathGeometry:
        """Positions, the frame as vectors, and the two normal curvatures.

        The geometry uncertainty defaults to ``analytic`` and not to
        undeclared, because for a surface given by a formula zero *is* the
        answer: there is no reconstruction and nothing to be uncertain about.
        A caller working from a scan replaces it with what the scan measured.
        """
        frame = darboux_frame(self.surface, self.u, self.v, self.du, self.dv)
        return PathGeometry(
            position=self.points,
            tangent=frame["tangent"],
            transverse=frame["transverse"],
            surface_normal=frame["surface_normal"],
            normal_curvature_along=frame["normal_curvature_along"],
            normal_curvature_transverse=frame["normal_curvature_transverse"],
            coordinate_frame=coordinate_frame,
            datum_frame=datum_frame,
            uncertainty=uncertainty
            or GeometryUncertainty.analytic(
                f"{self.surface.name} is given in closed form; its geometry carries "
                "no reconstruction error"
            ),
        )

    def chart_record(self) -> ChartValidity:
        """How much of the requested path the parameterisation could carry."""
        truncated = bool(self.chart.get("truncated", False))
        return ChartValidity(
            samples=int(self.arc_length.size),
            requested_samples=int(
                self.chart.get("truncated_from_samples", self.arc_length.size)
            ),
            truncated=truncated,
            reason=str(self.chart.get("truncation_reason") or "") if truncated else "",
            truncated_at=self.chart.get("truncated_at"),
            requested_length=self.chart.get("truncated_from_length", float(self.length)),
            min_conditioning=self.chart.get("min_conditioning"),
            declared_domain=dict(self.chart.get("declared_domain", {})),
        )

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
        validity: ValidityEnvelope | None = None,
        covariance: StartingCovariance | None = None,
        calibration: CalibrationBinding | None = None,
        upstream: tuple[UpstreamArtefact, ...] = (),
        provenance: Provenance | None = None,
        geometry: PathGeometry | None = None,
        geometry_uncertainty: GeometryUncertainty | None = None,
        coordinate_frame: str = "surface-parameterisation-ambient",
        datum_frame: str = "not-declared",
        convergence: ConvergenceEstimate | None = None,
        include_geometry: bool = True,
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

        The starting covariance and the calibration identifiers default to
        undeclared and unbound, and correctly so: this envelope was computed
        from an analytic surface, and neither a pose distribution nor an
        instrument exists anywhere in that computation. A caller who has one
        attaches it here or with :meth:`TransferRecord.with_covariance`.

        A path that left its chart cannot become a record. The samples past the
        exit were integrated through a degenerate metric and would cross the
        boundary indistinguishable from valid ones -- a downstream consumer
        sees an arclength grid and a transfer map, not a chart. Truncate first,
        explicitly, and the record then describes the prefix that is real.
        """
        if not self.chart_valid:
            raise ValueError(
                f"this path left the chart of {self.surface.name} at arc length "
                f"{self.chart.get('valid_until')} ({self.chart.get('reason')}), so "
                "its later samples were computed from a metric that does not "
                "describe the surface. Call truncated_to_chart() and present that "
                "prefix; a record carries no chart and cannot warn a consumer"
            )
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
                convergence=convergence
                or ConvergenceEstimate.not_established(
                    "no step-doubling comparison was run for this path; call "
                    "estimate_convergence(envelope) to pay for one"
                ),
            ),
            validity=validity
            or ValidityEnvelope.not_established(
                "no closed form for the eps^2 coefficient on a varying-curvature surface"
            ),
            observation_mode=observation_mode,
            domain="parametric-surface",
            covariance=covariance
            or StartingCovariance.not_declared(
                "computed from an analytic surface; no starting pose distribution exists here"
            ),
            provenance=(provenance or Provenance(note=f"geodesic envelope on {self.surface.name}"))
            .with_upstream(*upstream),
            calibration=calibration
            or CalibrationBinding.unbound("no instrument took part in this computation"),
            geometry=(
                geometry
                or self.path_geometry(
                    coordinate_frame=coordinate_frame,
                    datum_frame=datum_frame,
                    uncertainty=geometry_uncertainty,
                )
            )
            if include_geometry
            else None,
            # By construction, not by assertion: what was integrated is the
            # geodesic equation, and the residual in that claim is what the
            # speed drift and the Wronskian measure.
            path_type="geodesic",
            path_type_basis=(
                "by construction: the geodesic equation was integrated, and the "
                "unit-speed and det Phi = 1 invariants are never re-imposed"
            ),
            chart=self.chart_record(),
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


#: What to do with a path that runs off the edge of its parameterisation.
#: ``truncate`` keeps the valid prefix and says so, and is the default because
#: the samples past the exit were integrated through a metric that does not
#: describe the surface -- returning them by default means the careless path is
#: the wrong one. ``raise`` refuses outright; ``report`` returns the whole run
#: with the exit recorded, for the experiments that measure the exit itself.
CHART_EXIT_POLICIES: tuple[str, ...] = ("truncate", "raise", "report")


def integrate_paths(
    surface: ParametricSurface,
    *,
    u0: float,
    v0: float,
    headings,
    length: float,
    n_steps: int,
    method: str = "rk4",
    on_chart_exit: str = "truncate",
) -> list[PathEnvelope]:
    """Flow a fan of geodesics and their Jacobi fields in one pass.

    A whole fan costs barely more than a single path: the state is small and
    the expense is the per-step Python overhead, which the fan shares.

    ``on_chart_exit`` decides what happens to a path that leaves the region in
    which the parameterisation is a chart. The default truncates it to the
    prefix that is real. Each envelope is truncated on its own: one heading in
    a fan running off the edge says nothing about the others.
    """
    if on_chart_exit not in CHART_EXIT_POLICIES:
        raise ValueError(f"on_chart_exit must be one of {CHART_EXIT_POLICIES}")
    angles = np.atleast_1d(np.asarray(headings, dtype=float))
    # Refuse a start the chart cannot represent, rather than integrating out of
    # a singularity and returning something that looks like an envelope.
    surface.require_valid_chart(u0, v0, where="the starting point")
    starts = np.stack(
        [surface.initial_state(u0, v0, float(angle)) for angle in angles], axis=0
    )
    rhs = surface.geodesic_transfer_rhs()
    if on_chart_exit == "report":
        # The one policy that integrates past the exit, because the experiments
        # that measure *where* a chart fails need the samples on both sides.
        grid, trajectory = integrate(
            rhs, starts, length=length, n_steps=n_steps, method=method
        )
        valid = np.full(angles.shape, n_steps + 1, dtype=int)
        reasons: dict[int, str] = {}
    else:
        guard, reasons = _chart_guard(surface)
        grid, trajectory, valid = integrate_guarded(
            rhs, starts, length=length, n_steps=n_steps, method=method, guard=guard
        )

    envelopes = _wrap(
        surface, angles, grid, trajectory, u0, v0, length, n_steps, method,
        valid=valid, reasons=reasons,
    )
    if on_chart_exit == "report":
        return envelopes
    if on_chart_exit == "raise":
        for envelope in envelopes:
            if not envelope.chart_valid:
                raise ValueError(
                    f"the geodesic at heading {envelope.start[2]:g} left the chart of "
                    f"{surface.name} at arc length {envelope.chart['valid_until']:g} "
                    f"({envelope.chart['reason']})"
                )
        return envelopes
    return [
        envelope if envelope.chart_valid else envelope.truncated_to_chart()
        for envelope in envelopes
    ]


def _chart_guard(surface: ParametricSurface) -> tuple[Any, dict[int, str]]:
    """A guard that is ``True`` inside the chart, and the reasons it said no.

    The reasons are collected as the guard runs because they cannot be
    recovered afterwards: a stopped trajectory is frozen at its last *valid*
    state, so nothing in the returned samples remembers what went wrong. That
    is the trade for never computing the invalid samples in the first place.
    """
    reasons: dict[int, str] = {}

    def guard(state: np.ndarray) -> np.ndarray:
        u, v = state[..., 0], state[..., 1]
        validity = surface.chart_validity(u, v)
        in_domain = np.asarray(validity["in_domain"])
        conditioned = np.asarray(validity["well_conditioned"])
        usable = in_domain & conditioned
        for position in np.flatnonzero(np.atleast_1d(~usable)):
            reasons.setdefault(
                int(position),
                "left the declared parameter domain"
                if not bool(np.atleast_1d(in_domain)[position])
                else "chart became degenerate: EG - F^2 unresolved",
            )
        return usable

    return guard, reasons


def _wrap(
    surface: ParametricSurface,
    angles: np.ndarray,
    grid: np.ndarray,
    trajectory: np.ndarray,
    u0: float,
    v0: float,
    length: float,
    n_steps: int,
    method: str,
    valid: np.ndarray,
    reasons: dict[int, str],
) -> list[PathEnvelope]:
    """Wrap each trajectory of a flown fan as an envelope."""
    envelopes = []
    for index, angle in enumerate(angles):
        stopped = int(valid[index])
        stopped_at = None if stopped > int(n_steps) else stopped
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
                du=du,
                dv=dv,
                points=surface.embed(u, v),
                curvature=np.asarray(surface.gaussian_curvature(u, v), dtype=float),
                lateral_basis=trajectory[:, index, 4],
                lateral_rate=trajectory[:, index, 5],
                jacobi_field=trajectory[:, index, 6],
                jacobi_derivative=trajectory[:, index, 7],
                speed=np.asarray(surface.speed(u, v, du, dv), dtype=float),
                chart=_chart_report(
                    surface, grid, u, v,
                    stopped_at=stopped_at,
                    reason=reasons.get(index),
                ),
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
    on_chart_exit: str = "truncate",
) -> PathEnvelope:
    """Flow one geodesic and its Jacobi field together along ``surface``.

    A path that leaves its chart is truncated to the prefix that stayed inside
    it, and the envelope's ``chart`` says where and why. Pass
    ``on_chart_exit="raise"`` to refuse such a path outright, or ``"report"``
    to receive the whole run with the exit recorded.
    """
    return integrate_paths(
        surface,
        u0=u0,
        v0=v0,
        headings=[heading],
        length=length,
        n_steps=n_steps,
        method=method,
        on_chart_exit=on_chart_exit,
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


def estimate_convergence(
    envelope: PathEnvelope, *, refinement: int = 2, probe_covariance=None
) -> ConvergenceEstimate:
    """What this path's numerics cost, per quantity, by halving the step.

    Fixed-step RK4 is kept because its order is known and therefore testable,
    and an order is a statement about the limit rather than about the run in
    hand. This is the run in hand: flow the same path again at ``h / 2`` and
    Richardson-extrapolate at the method's order, so that

    ``error(h) ~ |y(h) - y(h/2)| * 2^p / (2^p - 1)``

    is an absolute error estimate for *the samples the record carries*, in the
    record's own units.

    Reported per quantity, because they do not converge together:

    ``position``
        where the path is, in ambient length.
    ``transfer``
        the worst of the four entries of ``Phi``.
    ``curvature``
        ``K`` along the path, which is evaluated rather than integrated and so
        converges at the rate the *path* does, not the rate ``Phi`` does.
    ``focus``
        the location of the first focus. A root of ``b`` divided by a slope
        that is small precisely where the root is, so it is the entry most
        likely to be the one that matters, and the one a step size chosen by
        looking at ``Phi`` alone will get wrong.
    ``covariance``
        the propagated ``Phi C Phi^T``, quadratic in ``Phi`` and so with
        roughly twice its relative error. The probe is the identity unless one
        is given, which makes the figure the error in ``Phi Phi^T`` -- the
        factor any starting covariance is sandwiched by.

    It costs a second integration, which is why no producer does it by
    default: a record whose convergence says ``not-established`` has not paid
    for one, and that is a true statement rather than a missing feature.
    """
    if refinement < 2:
        raise ValueError("refinement must halve the step at least once")
    order = get_integrator(envelope.integrator).order
    fine = integrate_path(
        envelope.surface,
        u0=envelope.start[0],
        v0=envelope.start[1],
        heading=envelope.start[2],
        length=float(envelope.length),
        n_steps=int(envelope.n_steps) * refinement,
        method=envelope.integrator,
    )
    # A refined run can stop at a different sample if the path leaves the
    # chart, so the comparison is made on the samples both runs actually hold.
    coarse_samples = int(envelope.arc_length.size)
    shared = min(coarse_samples, 1 + (int(fine.arc_length.size) - 1) // refinement)
    if shared < 2:
        return ConvergenceEstimate.not_established(
            "the refined run left the chart before a second sample, so there is "
            "nothing to compare"
        )
    coarse_slice = slice(0, shared)
    fine_slice = slice(0, shared * refinement, refinement)
    if not np.allclose(
        envelope.arc_length[coarse_slice], fine.arc_length[fine_slice], rtol=0.0, atol=1e-12
    ):
        raise ValueError("the refined run does not land on the coarse run's samples")

    scale = refinement**order / (refinement**order - 1.0)

    def gap(coarse_values, fine_values) -> float:
        difference = np.asarray(coarse_values)[coarse_slice] - np.asarray(fine_values)[
            fine_slice
        ]
        return float(np.max(np.abs(difference))) * scale

    transfer = max(
        gap(envelope.lateral_basis, fine.lateral_basis),
        gap(envelope.lateral_rate, fine.lateral_rate),
        gap(envelope.jacobi_field, fine.jacobi_field),
        gap(envelope.jacobi_derivative, fine.jacobi_derivative),
    )
    position = float(
        np.max(
            np.linalg.norm(
                envelope.points[coarse_slice] - fine.points[fine_slice], axis=-1
            )
        )
    ) * scale

    coarse_focus = envelope.focus_points()
    fine_focus = fine.focus_points()
    focus: float | None = None
    note = ""
    if coarse_focus and fine_focus:
        focus = abs(coarse_focus[0] - fine_focus[0]) * scale
    elif bool(coarse_focus) != bool(fine_focus):
        note = (
            "the two resolutions disagree about whether the path has a focus at "
            "all, which is a stronger statement than any error estimate"
        )

    probe = np.eye(2) if probe_covariance is None else validated_covariance(probe_covariance)
    covariance = float(
        np.max(
            np.abs(
                envelope.transfer_map.propagate_covariance(probe)[coarse_slice]
                - fine.transfer_map.propagate_covariance(probe)[fine_slice]
            )
        )
    ) * scale

    return ConvergenceEstimate(
        basis=f"step-doubling: {envelope.integrator} at h and h/{refinement}, "
        f"Richardson-extrapolated at order {order}",
        order=order,
        refinement=refinement,
        position=position,
        transfer=transfer,
        curvature=gap(envelope.curvature, fine.curvature),
        focus=focus,
        covariance=covariance,
        note=note,
    )


#: The ladder of *perturbation magnitudes* the validity probe walks -- the
#: separation between the two starting poses, not the half-separation the
#: central difference takes. That distinction is a factor of four in the
#: fitted coefficient and a factor of two in the bound, and getting it wrong
#: would make the measured envelope disagree with the closed form by exactly
#: that. Geometric, so the ``eps^2`` law can be fitted rather than assumed, and
#: spanning the range a real perturbation lives in: a quarter of a degree to
#: fifteen degrees of heading.
VALIDITY_PROBES: tuple[float, ...] = (
    0.004,
    0.008,
    0.016,
    0.032,
    0.064,
    0.128,
    0.256,
    0.512,
)


def measure_validity_envelope(
    surface: ParametricSurface,
    *,
    u0: float,
    v0: float,
    heading: float,
    length: float,
    n_steps: int,
    relative_tolerance: float,
    tolerance_basis: str,
    probes: tuple[float, ...] = VALIDITY_PROBES,
    directions: tuple[str, ...] = PERTURBATION_DIRECTIONS,
    method: str = "rk4",
    source_digest: str = "",
    observation_mode: str = "ambient-euclidean-chord",
) -> ValidityEnvelope:
    """The largest perturbation the linear map holds to, measured not asserted.

    The reference is the geodesic flow itself, central-differenced. It never
    touches the Jacobi equation, so it is an independent *computational route*
    to the same quantity rather than a rearrangement of the one being checked.
    It is not an independent implementation: the probe and the transfer map it
    is compared against share the surface model, the geodesic right-hand side
    and the integrator, so any error in those is common mode and cancels out of
    the comparison rather than showing up in it. ``reference_method`` and
    ``reference_samples`` are therefore declared -- the bound is only a
    statement about the linearisation while the integrator's truncation error
    sits well below it, and with a second-order method at these steps it does
    not.

    Both columns are exercised, because they are different questions: a fan of
    headings probes ``b`` and a set of laterally displaced starts probes ``a``,
    and the two focus in different places. An envelope established from the
    heading alone would be a bound on one column presented as a bound on the
    map.

    The admitted magnitude is *fitted* rather than read off the ladder. The
    relative error goes as ``C eps^2``, so ``C`` is fitted over the probes that
    are still in that regime and the bound is ``sqrt(tolerance / C)``.
    Reporting the largest rung that happened to pass would quantise the answer
    to the ladder and would round the wrong way -- upwards, admitting a
    perturbation that was never tested.

    It is then clipped to the end of the ladder, and ``probe_limited`` says so.
    On an intrinsically flat surface the lateral column is exact, the fitted
    coefficient is roundoff, and the extrapolated bound comes out in the
    hundreds of radians. What was established there is that the linearisation
    held out to the largest perturbation tested; extrapolating four orders
    beyond the data is not a measurement.

    The measured quantity is an ambient chord on both sides, which is why the
    mode is carried: comparing a chord with an in-surface separation differs at
    exactly the order this is measuring.
    """
    for direction in directions:
        if direction not in PERTURBATION_DIRECTIONS:
            raise ValueError(f"directions must be drawn from {PERTURBATION_DIRECTIONS}")
    if not directions:
        raise ValueError("a validity envelope must exercise at least one column")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if not tolerance_basis or tolerance_basis == "not-declared":
        raise ValueError(
            "a validity envelope must say how its tolerance was chosen before it is "
            "measured, not after"
        )
    probes = tuple(float(value) for value in probes)
    if len(probes) < 3:
        raise ValueError("fitting the eps^2 coefficient needs at least three probes")

    envelope = integrate_paths(
        surface,
        u0=u0,
        v0=v0,
        headings=[heading],
        length=length,
        n_steps=n_steps,
        method=method,
        on_chart_exit="truncate",
    )[0]
    transfer = envelope.transfer_map
    predicted = {"heading": np.abs(transfer.b), "lateral": np.abs(transfer.a)}
    measure = {
        "heading": finite_difference_jacobi,
        "lateral": finite_difference_lateral,
    }

    def relative_error(direction: str, magnitudes):
        """Worst and end-of-route relative departure, per magnitude."""
        grid, measured = measure[direction](
            surface,
            u0=u0,
            v0=v0,
            heading=heading,
            # The declared magnitudes are separations; a central difference
            # straddles the nominal, so each side is half of one.
            epsilon=0.5 * np.asarray(magnitudes, dtype=float),
            length=length,
            n_steps=n_steps,
            method=method,
        )
        reference = predicted[direction][: grid.size]
        scale = np.maximum(np.abs(reference), np.max(np.abs(reference)) * 1e-12)
        relative = np.abs(measured[: grid.size] - reference[:, None]) / scale[:, None]
        return np.max(relative, axis=0), relative[-1]

    fits: list[ProbeFit] = []
    pointwise = 0.0
    route = 0.0
    for direction in directions:
        worst, at_end = relative_error(direction, probes)
        fit = _fit_quadratic_regime(probes, worst, tolerance, direction)

        # Held out: re-probe at fractions of the bound the fit just chose.
        # None of these took part in it, and the one at 1.2 is the one that can
        # say the bound is wrong.
        fractions = np.asarray(HELD_OUT_FRACTIONS, dtype=float) * float(fit.bound)
        held_worst, _ = relative_error(direction, fractions)
        fit = replace(
            fit,
            held_out=tuple(
                (float(f), float(e))
                for f, e in zip(HELD_OUT_FRACTIONS, held_worst, strict=True)
            ),
        )
        fits.append(fit)
        pointwise = max(pointwise, float(np.max(worst)))
        route = max(route, float(np.max(at_end)))

    bounds = {fit.direction: fit.bound for fit in fits}
    limited = tuple(fit.direction for fit in fits if fit.probe_limited)

    return ValidityEnvelope(
        basis=(
            "measured: the linear map against the geodesic flow, central-differenced, "
            f"with the eps^2 coefficient fitted over {len(probes)} probes and re-probed "
            f"at {', '.join(f'{f:g}x' for f in HELD_OUT_FRACTIONS)} the fitted bound"
        ),
        observation_mode=observation_mode,
        relative_tolerance=tolerance,
        tolerance_basis=tolerance_basis,
        max_lateral=bounds.get("lateral"),
        max_heading=bounds.get("heading"),
        directions=tuple(directions),
        probe_magnitudes=probes,
        pointwise_error=pointwise,
        route_error=route,
        probe_limited_directions=limited,
        fits=tuple(fits),
        reference="geodesic-flow-central-difference",
        reference_method=method,
        reference_samples=int(n_steps),
        reference_digest=source_digest,
        convergence=estimate_convergence(envelope),
        note=(
            f"{surface.name}; the bound is where the fitted quadratic term reaches the "
            "declared tolerance, not the largest probe that happened to pass"
            + (
                ". It is clipped to the end of the ladder for "
                f"{', '.join(limited)}: the linearisation held to the tolerance across "
                "every probe there, so what was established is that it holds that far, "
                "not how much further"
                if limited
                else ""
            )
        ),
    )


#: Fractions of the fitted bound the linearisation is re-probed at, after the
#: fit. None of them took part in it. At 1.2 the relative error must exceed the
#: declared tolerance: a bound the linearisation comfortably survives past is
#: not where it fails, it is wherever the ladder happened to stop.
HELD_OUT_FRACTIONS: tuple[float, ...] = (0.8, 1.0, 1.2)


def _fit_quadratic_regime(
    probes: tuple[float, ...], errors, tolerance: float, direction: str
) -> ProbeFit:
    """``sqrt(tolerance / C)`` with ``C`` fitted on the probes still going as ``eps^2``.

    Probes whose error has left the quadratic law -- into the higher-order
    terms at the top of the ladder, or into the differencing floor at the
    bottom -- would bias ``C``, so the fit keeps the ones whose local log-log
    slope is within half of two. Which ones those were is returned rather than
    discarded: an adaptive selection nobody can inspect is not a measurement,
    it is a number with a provenance of "trust me".
    """
    magnitudes = np.asarray(probes, dtype=float)
    values = np.asarray(errors, dtype=float)
    rejected: list[tuple[float, str]] = []

    positive = values > 0.0
    for magnitude in magnitudes[~positive]:
        rejected.append((float(magnitude), "relative error underflowed to zero"))
    if not np.any(positive):
        return ProbeFit(
            direction=direction,
            coefficient=0.0,
            fitted_probes=(),
            observed_slopes=(),
            rejected_probes=tuple(rejected),
            bound=float(magnitudes[-1]),
            probe_limited=True,
        )

    logs = np.log(values[positive])
    steps = np.log(magnitudes[positive])
    slopes = np.gradient(logs, steps) if steps.size > 1 else np.full(steps.size, 2.0)
    quadratic = np.abs(slopes - 2.0) < 0.5
    for magnitude, slope in zip(magnitudes[positive][~quadratic], slopes[~quadratic],
                                strict=False):
        rejected.append(
            (float(magnitude), f"local log-log slope {float(slope):.3f} is not 2")
        )

    if int(np.count_nonzero(quadratic)) < 2:
        # Too little of the ladder is quadratic to fit on. The smallest probe
        # carries it alone, which is the conservative reading: it gives the
        # largest C the data supports, and therefore the smallest bound.
        index = int(np.argmax(positive))
        coefficient = float(values[index] / magnitudes[index] ** 2)
        used = (float(magnitudes[index]),)
        kept_slopes: tuple[float, ...] = ()
    else:
        used = tuple(float(v) for v in magnitudes[positive][quadratic])
        kept_slopes = tuple(float(v) for v in slopes[quadratic])
        coefficient = float(
            np.exp(np.mean(np.log(values[positive][quadratic]) - 2.0 * np.log(used)))
        )

    ceiling = float(magnitudes[-1])
    if coefficient <= 0.0:  # pragma: no cover - guard
        return ProbeFit(
            direction=direction, coefficient=0.0, fitted_probes=used,
            observed_slopes=kept_slopes, rejected_probes=tuple(rejected),
            bound=ceiling, probe_limited=True,
        )
    fitted = float(np.sqrt(float(tolerance) / coefficient))
    return ProbeFit(
        direction=direction,
        coefficient=coefficient,
        fitted_probes=used,
        observed_slopes=kept_slopes,
        rejected_probes=tuple(rejected),
        bound=min(fitted, ceiling),
        probe_limited=fitted > ceiling,
    )
