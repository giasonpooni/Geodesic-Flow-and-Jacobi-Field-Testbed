"""Choosing a route from constraints the application actually has.

Ranking candidate paths by ``max |b(s)|`` alone selects for paths that pass
through a focus, where forward separation is small because the map from
starting pose to endpoint has collapsed. Guarding that with a fixed focus
margin replaces one arbitrary number with another: a clearance of 0.25 means
nothing without a length scale, and nothing at all to a process whose real
limit is a cross-track tolerance in millimetres.

So the constraints here are the ones a manufacturing or inspection process
states in its own terms. Given the starting-pose tolerance the machine can
actually hold, the transfer map gives

.. math::

    e_\\perp(s)    &= |a(s)|\\,\\delta p + |b(s)|\\,\\delta\\alpha \\\\
    e_\\alpha(s)   &= |a'(s)|\\,\\delta p + |b'(s)|\\,\\delta\\alpha

and a route is feasible when every declared limit holds along the whole path.
Ranking happens only among feasible routes, and by a dimensionless score.

**Why there is no free robustness.** Scaling the transfer map by the tolerance
box, ``Phi~ = S^-1 Phi S`` with ``S = diag(dp, da)``, leaves the determinant
alone, so ``Phi~`` has singular values ``sigma`` and ``1/sigma``. Whatever
direction of starting error the flow contracts, it expands the conjugate
direction by exactly the same factor, and the larger singular value is never
below one. A path that looks insensitive in cross-track has moved the
sensitivity into heading, or towards a focus. That is a theorem about
``det Phi = 1``, not a heuristic, and it is why both bounds are constrained
here rather than one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .contract import validated_covariance
from .observation_model import ObservationModel
from .record import TransferRecord, to_transfer_record
from .tracking import AcquisitionSpec, TrackingOutcome, evaluate_tracking

Array = np.ndarray


@dataclass(frozen=True)
class CoverageSpec:
    """Neighbouring-path geometry, for the coverage and spacing constraints."""

    nominal_spacing: float
    swath_width: float
    initial_heading_delta: float = 0.0

    def __post_init__(self) -> None:
        for name in ("nominal_spacing", "swath_width"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(float(self.initial_heading_delta)):
            raise ValueError("initial_heading_delta must be finite")


@dataclass(frozen=True)
class RouteConstraints:
    """Limits a route must hold along its whole length. ``None`` means unconstrained."""

    max_cross_track_error: float | None = None
    max_heading_error: float | None = None
    min_coverage_margin: float | None = None
    #: The sensor's acquisition and retention schedule, in resolvability and
    #: arc length. This is what replaces a threshold on |b|: dimensionless,
    #: stated by the instrument, and a schedule rather than a single number --
    #: a minimum taken over the whole path is unsatisfiable because rho(0) = 0
    #: on every route, and a minimum taken only after the first crossing lets a
    #: route pass by acquiring at its very last sample.
    acquisition: AcquisitionSpec | None = None
    min_boundary_clearance: float | None = None
    max_path_length: float | None = None
    coverage: CoverageSpec | None = None

    #: Limits that are only meaningful above zero. A tolerance of zero admits
    #: nothing and is almost always a default that was never filled in; a
    #: negative one admits nothing while reading as a limit. Both are refused
    #: here rather than silently making every route infeasible.
    _POSITIVE_LIMITS = (
        "max_cross_track_error",
        "max_heading_error",
        "max_path_length",
    )
    #: Limits where zero is a real declaration -- "touch the boundary but do
    #: not cross it" -- so only negativity and non-finiteness are refused.
    _NONNEGATIVE_LIMITS = ("min_boundary_clearance",)

    def __post_init__(self) -> None:
        if self.min_coverage_margin is not None and self.coverage is None:
            raise ValueError("a coverage margin needs a CoverageSpec to measure against")
        for name in self._POSITIVE_LIMITS:
            declared = getattr(self, name)
            if declared is None:
                continue
            value = float(declared)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive when declared; {value!r} "
                    "admits no route and is not a limit"
                )
        for name in self._NONNEGATIVE_LIMITS:
            declared = getattr(self, name)
            if declared is None:
                continue
            value = float(declared)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative when declared")
        if self.min_coverage_margin is not None and not np.isfinite(
            float(self.min_coverage_margin)
        ):
            raise ValueError("min_coverage_margin must be finite when declared")

    def required_inputs(self) -> tuple[str, ...]:
        """Data a caller must supply for these constraints to be evaluable."""
        needed: list[str] = []
        if self.acquisition is not None:
            needed.extend(("observation", "initial_covariance"))
        if self.min_boundary_clearance is not None:
            needed.append("boundary_clearance")
        return tuple(needed)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["coverage"] = asdict(self.coverage) if self.coverage else None
        payload["acquisition"] = self.acquisition.to_dict() if self.acquisition else None
        return payload


@dataclass(frozen=True)
class ConstraintMargin:
    """One constraint, what it allowed, what the route did, and by how much."""

    name: str
    limit: float
    worst_value: float
    margin: float
    satisfied: bool
    at_arclength: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteAssessment:
    """Whether a route is usable, and among usable ones how it ranks."""

    label: str
    feasible: bool
    binding_constraint: str | None
    amplification_score: float
    max_cross_track_error: float
    max_heading_error: float
    focus_clearance: float
    tracking: dict[str, Any] | None
    crossed_first_conjugate_point: bool
    path_length: float
    margins: list[ConstraintMargin]
    source_digest: str
    observation_mode: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["margins"] = [margin.to_dict() for margin in self.margins]
        return payload


def _margin(
    name: str,
    limit: float | None,
    values: Array,
    grid: Array,
    *,
    upper_bound: bool,
) -> ConstraintMargin | None:
    """Turn a limit and a sampled quantity into a margin, or ``None`` if unconstrained."""
    if limit is None:
        return None
    values = np.asarray(values, dtype=float)
    if upper_bound:
        index = int(np.argmax(values))
        worst = float(values[index])
        margin = float(limit) - worst
    else:
        index = int(np.argmin(values))
        worst = float(values[index])
        margin = worst - float(limit)
    return ConstraintMargin(
        name=name,
        limit=float(limit),
        worst_value=worst,
        margin=margin,
        satisfied=bool(margin >= 0.0),
        at_arclength=float(grid[index]) if grid.size == values.size else None,
    )


def assess_route(
    source: Any,
    *,
    max_lateral: float,
    max_heading: float,
    constraints: RouteConstraints,
    boundary_clearance: Array | None = None,
    observation: ObservationModel | None = None,
    initial_covariance: Array | None = None,
    label: str = "route",
) -> RouteAssessment:
    """Check one candidate route against every declared limit.

    ``boundary_clearance``, when given, is the distance from the path to the
    edge of the workable region at each sample. It has to come from the caller
    because it depends on the part, not on the transfer map.

    ``observation`` and ``initial_covariance`` carry the resolvability
    constraint, which is the one worth using: it asks whether the instrument
    can distinguish the starting poses the tolerance admits, rather than
    whether a dimensionful quantity cleared a chosen number.

    A constraint that was declared but cannot be evaluated is an error, not a
    pass. Skipping it would make a route look feasible *because* the evidence
    for the limit it must meet is missing, which is the one failure mode a
    feasibility check must never have.
    """
    supplied = {
        "observation": observation,
        "initial_covariance": initial_covariance,
        "boundary_clearance": boundary_clearance,
    }
    missing = [name for name in constraints.required_inputs() if supplied[name] is None]
    if missing:
        raise ValueError(
            f"route {label!r} declares constraints needing {', '.join(missing)}, "
            "which were not supplied; an unevaluated constraint must not be "
            "reported as a satisfied one"
        )

    record: TransferRecord = to_transfer_record(source)
    grid = record.arclength
    # Presence was checked above; that a constraint's evidence is *usable* is a
    # separate question with the same answer. A boundary array of the wrong
    # length silently pairs clearances with the wrong arc lengths, and a
    # covariance of the wrong shape fails somewhere deeper with a message about
    # matrices rather than about this route.
    if boundary_clearance is not None:
        clearance_samples = np.asarray(boundary_clearance, dtype=float)
        if clearance_samples.ndim != 1 or clearance_samples.shape != grid.shape:
            raise ValueError(
                f"route {label!r}: boundary_clearance has shape "
                f"{np.shape(boundary_clearance)} but the record has {grid.shape[0]} "
                "samples; a clearance must be given on the record's own arclength "
                "grid, or it is paired with the wrong arc lengths"
            )
        if not np.all(np.isfinite(clearance_samples)):
            raise ValueError(
                f"route {label!r}: boundary_clearance must be finite; a "
                "non-finite clearance compares as satisfied against any limit"
            )
        boundary_clearance = clearance_samples
    if initial_covariance is not None:
        initial_covariance = validated_covariance(
            initial_covariance, f"route {label!r}: C0"
        )
    cross_track = record.cross_track_error(max_lateral, max_heading)
    heading = record.heading_error(max_lateral, max_heading)
    # Reported, never decisive: |a| and |b| carry length per unit of starting
    # error, so a threshold on either is specific to one part size and one
    # angle unit. The geometric fact is still worth publishing beside the
    # instrument's verdict -- they answer different questions.
    focus_clearance = record.clearance_from_focus(component="b")
    lateral_focus_clearance = record.clearance_from_focus(component="a")
    clearance = min(focus_clearance, lateral_focus_clearance)
    length = float(grid[-1] - grid[0])

    margins: list[ConstraintMargin] = []
    for margin in (
        _margin("cross-track-error", constraints.max_cross_track_error, cross_track, grid,
                upper_bound=True),
        _margin("heading-error", constraints.max_heading_error, heading, grid,
                upper_bound=True),
        _margin("path-length", constraints.max_path_length, np.array([length]),
                np.array([grid[-1]]), upper_bound=True),
    ):
        if margin is not None:
            margins.append(margin)

    if constraints.coverage is not None and constraints.min_coverage_margin is not None:
        spacing = np.abs(
            record.separation(
                constraints.coverage.nominal_spacing,
                constraints.coverage.initial_heading_delta,
            )
        )
        # Two neighbouring paths may each drift by the full one-path envelope,
        # in opposite directions.
        widest = spacing + 2.0 * cross_track
        coverage_margin = constraints.coverage.swath_width - widest
        found = _margin(
            "coverage-margin", constraints.min_coverage_margin, coverage_margin, grid,
            upper_bound=False,
        )
        if found is not None:
            margins.append(found)

    tracking: TrackingOutcome | None = None
    if constraints.acquisition is not None:
        assert observation is not None and initial_covariance is not None  # checked above
        rho = observation.resolvability(record, initial_covariance)
        profile = np.min(rho, axis=1) if rho.ndim > 1 else rho
        tracking = evaluate_tracking(grid, profile, constraints.acquisition)
        margins.append(
            ConstraintMargin(
                name="tracking",
                limit=float(constraints.acquisition.hold_threshold),
                worst_value=(
                    tracking.min_resolvability_while_tracked
                    if tracking.min_resolvability_while_tracked is not None
                    else tracking.max_resolvability
                ),
                margin=(
                    # ``or 0.0`` here would turn a genuine resolvability of
                    # exactly zero into the same margin as no measurement.
                    (
                        tracking.min_resolvability_while_tracked
                        if tracking.min_resolvability_while_tracked is not None
                        else 0.0
                    )
                    - float(constraints.acquisition.hold_threshold)
                    if tracking.satisfied
                    else -abs(float(constraints.acquisition.hold_threshold))
                ),
                satisfied=tracking.satisfied,
                at_arclength=tracking.acquisition_arclength,
            )
        )

    if constraints.min_boundary_clearance is not None:
        assert boundary_clearance is not None  # checked above
        found = _margin(
            "boundary-clearance",
            constraints.min_boundary_clearance,
            np.asarray(boundary_clearance, dtype=float),
            grid,
            upper_bound=False,
        )
        if found is not None:
            margins.append(found)

    failed = [margin for margin in margins if not margin.satisfied]
    binding = min(margins, key=lambda m: m.margin).name if margins else None
    return RouteAssessment(
        label=label,
        feasible=not failed,
        binding_constraint=binding,
        amplification_score=record.amplification_score(max_lateral, max_heading),
        max_cross_track_error=float(np.max(cross_track)),
        max_heading_error=float(np.max(heading)),
        focus_clearance=clearance,
        tracking=None if tracking is None else tracking.to_dict(),
        crossed_first_conjugate_point=record.transfer_map().crossed_first_conjugate_point(),
        path_length=length,
        margins=margins,
        source_digest=record.source_digest,
        observation_mode=record.observation_mode,
    )


def rank_routes(
    candidates: dict[str, Any],
    *,
    max_lateral: float,
    max_heading: float,
    constraints: RouteConstraints,
    boundary_clearance: dict[str, Array] | None = None,
    observation: ObservationModel | None = None,
    initial_covariance: Array | None = None,
) -> list[RouteAssessment]:
    """Assess every candidate; feasible ones first, then by dimensionless score.

    Infeasible routes are kept in the result rather than dropped, ordered by how
    badly they miss, because "no route is feasible" needs to say which limit was
    the problem.
    """
    boundary_clearance = boundary_clearance or {}
    assessments = [
        assess_route(
            source,
            max_lateral=max_lateral,
            max_heading=max_heading,
            constraints=constraints,
            boundary_clearance=boundary_clearance.get(label),
            observation=observation,
            initial_covariance=initial_covariance,
            label=label,
        )
        for label, source in candidates.items()
    ]
    return sorted(
        assessments,
        key=lambda a: (
            not a.feasible,
            a.amplification_score if a.feasible else -min(m.margin for m in a.margins),
        ),
    )
