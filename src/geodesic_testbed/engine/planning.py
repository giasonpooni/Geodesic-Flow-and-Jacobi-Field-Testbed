# SPDX-License-Identifier: MPL-2.0
r"""Choosing a route, without collapsing the reasons into one number first.

A route has to satisfy several things at once -- stay inside its tolerance,
stay inside the chart, stay clear of the part's edge and of anything on it, be
observable by the instrument that has to follow it, and not be longer than the
process allows. These are different quantities in different units, and the
temptation is to weight them into a score and rank by it. That buries the
trade-off in a set of weights nobody declared, and the weights are then the
decision.

So this module keeps them apart:

**Observability** is accumulated rather than sampled. ``rho(s)`` asks whether
the instrument can resolve the starting-pose error *at one arc length*; the
Gramian

.. code-block:: text

    W = int_0^L Phi^T H^T R^-1 H Phi ds

asks how much information about that error the whole path yields, and the two
differ: a route can be resolvable at every sample and still tell you almost
nothing about one direction of the starting pose, because ``Phi`` rotates that
direction into the sensor's blind one everywhere at once. The smallest
eigenvalue of the Gramian is what that direction costs, and it is a property of
the path rather than of a sample.

**Boundaries** are computed rather than supplied. ``assess_route`` takes a
clearance array because clearance depends on the part; declaring the part once,
as a region, and computing the array is the same information with the
opportunity for a mismatched grid removed.

**Routes** are generated over both degrees of freedom. A fan of headings from
one point is one family, and the family a manufacturing process actually runs
is a set of parallel offset courses -- a different sweep, over the start point,
which exercises the ``a`` column that a heading fan never touches.

**The cost is a front, not a scalar.** :func:`pareto_front` returns the routes
that nothing else beats on every objective at once. Collapsing that to one
number needs weights, :func:`weighted_cost` demands them explicitly, and it
refuses to normalise an objective that has no declared limit -- because the
limit is what makes the ratio dimensionless, and without it the weights are
carrying units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import validated_covariance
from .envelope import PathEnvelope, integrate_paths
from .integrators import integrate
from .observation_model import ObservationModel
from .record import TransferRecord, to_transfer_record
from .surfaces import ParametricSurface, first_fundamental_form
from .uncertainty import cumulative_quadrature

Array = np.ndarray

#: What a Gramian was made dimensionless by. The choice is a declaration, not a
#: convenience: ``tolerance-box`` asks how well the instrument sees the errors
#: the *tolerance* admits, and ``prior-covariance`` how much it adds to what was
#: already believed. They rank routes differently and both are legitimate.
GRAMIAN_SCALES: tuple[str, ...] = ("tolerance-box", "prior-covariance")


@dataclass(frozen=True)
class Observability:
    """``W = int Phi^T H^T R^-1 H Phi ds``, and what its worst direction costs."""

    arclength: Array
    #: Cumulative, so a caller can see *where* the information arrived rather
    #: than only how much there was by the end.
    cumulative: Array
    scale: Array
    scale_basis: str
    path_length: float
    note: str = ""

    @property
    def total(self) -> Array:
        """The dimensionless Gramian over the whole path."""
        return self.cumulative[-1]

    @property
    def eigenvalues(self) -> Array:
        """Ascending. The smallest is the worst-observed direction."""
        return np.linalg.eigvalsh(self.total)

    @property
    def worst_direction(self) -> Array:
        """The starting-pose direction the path says least about, in scaled units."""
        values, vectors = np.linalg.eigh(self.total)
        return vectors[:, int(np.argmin(values))]

    @property
    def condition_number(self) -> float:
        """``lambda_max / lambda_min`` of the scaled Gramian.

        How much better the path sees its best-observed starting-pose
        direction than its worst. A route can accumulate a great deal of
        information and still be nearly blind in one direction, and that is
        the number that says so -- a trace or a determinant would not.
        """
        values = self.eigenvalues
        smallest = float(values[0])
        if smallest <= 0.0:
            return float("inf")
        return float(values[-1]) / smallest

    def over(self, start: float, end: float) -> Array:
        """Information accumulated between two arc lengths.

        The endpoints are interpolated rather than snapped to the nearest
        sample, for the same reason every other event here is: an interval
        quantised to the grid is known only to the sample spacing, and a
        caller comparing two candidate acquisition windows would be comparing
        their rounding as much as their information.
        """
        grid = np.asarray(self.arclength, dtype=float)
        lower, upper = float(start), float(end)
        if upper <= lower:
            raise ValueError("an interval needs end > start")
        if lower < grid[0] - 1e-12 or upper > grid[-1] + 1e-12:
            raise ValueError(
                f"[{lower}, {upper}] is not inside the path [{grid[0]}, {grid[-1]}]"
            )
        return _interpolated(grid, self.cumulative, upper) - _interpolated(
            grid, self.cumulative, lower
        )

    @property
    def per_unit_length(self) -> Array:
        """Information density. This is the scale-invariant one.

        The accumulated Gramian is not scale invariant and should not be: the
        same physical situation drawn at twice the size has twice the path, and
        a longer path really does carry more information. Dividing by the path
        length removes that and leaves a quantity that is the same function of
        the fractional distance along the path at every size -- the property a
        route criterion has to have, and the one a dimensionful score cannot.
        """
        return self.total / self.path_length

    @property
    def numerical_rank(self) -> int:
        """How many starting-pose directions the path says anything about.

        Reported alongside the eigenvalues because it is the question a
        condition number cannot answer: a rank-deficient Gramian has an
        infinite condition number and a finite one can still be numerically
        rank deficient at the working precision.
        """
        return int(np.linalg.matrix_rank(self.total))

    def to_dict(self) -> dict[str, Any]:
        values = self.eigenvalues
        return {
            "form": "integral",
            # The integral form treats R as a noise *density*: the ds in the
            # quadrature carries units, so the number is per unit arc length
            # and needs the samples independent. The stacked form treats R as
            # the covariance of the measurements taken. They differ by the
            # sample spacing and are not interchangeable.
            "noise_convention": "continuous-density",
            "scale_basis": self.scale_basis,
            "scale": self.scale.tolist(),
            "path_length": float(self.path_length),
            "eigenvalues": values.tolist(),
            "numerical_rank": self.numerical_rank,
            "worst_observed": float(values[0]),
            "best_observed": float(values[-1]),
            "anisotropy": float(values[-1] / values[0]) if values[0] > 0.0 else None,
            "condition_number": self.condition_number,
            "worst_direction": self.worst_direction.tolist(),
            "worst_observed_per_unit_length": float(values[0] / self.path_length),
            "note": self.note,
        }


def observability_gramian(
    source: Any,
    observation: ObservationModel,
    *,
    max_lateral: float | None = None,
    max_heading: float | None = None,
    prior_covariance=None,
) -> Observability:
    """Accumulate ``Phi^T H^T R^-1 H Phi`` along the path, made dimensionless.

    Exactly one scaling must be declared. ``max_lateral``/``max_heading`` give
    the tolerance box, ``prior_covariance`` gives ``C0``; the raw Gramian is
    not reported on its own because its entries do not share units -- ``b`` is
    a length per radian -- so its eigenvalues are not comparable and ranking by
    them would depend on the unit of angle.
    """
    record: TransferRecord = to_transfer_record(source)
    if observation.mode != record.observation_mode:
        raise ValueError(
            f"the record is in {record.observation_mode!r} but this observation "
            f"model reports {observation.mode!r}; R is the noise on the quantity "
            "the model reports, so a Gramian mixing the two is weighting one "
            "quantity's sensitivity by another's noise"
        )
    scale, basis = _declared_scaling(max_lateral, max_heading, prior_covariance)

    phi = record.transfer_map().matrices()
    information = np.linalg.inv(observation.noise_covariance)
    projected = observation.matrix.T @ information @ observation.matrix
    integrand = np.einsum("sji,jk,skl->sil", phi, projected, phi)
    scaled = np.einsum("ji,sjk,kl->sil", scale, integrand, scale)

    grid = record.arclength
    cumulative = np.stack(
        [
            np.stack(
                [
                    cumulative_quadrature(scaled[:, 0, 0], grid),
                    cumulative_quadrature(scaled[:, 0, 1], grid),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    cumulative_quadrature(scaled[:, 0, 1], grid),
                    cumulative_quadrature(scaled[:, 1, 1], grid),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    return Observability(
        arclength=grid,
        cumulative=cumulative,
        scale=scale,
        scale_basis=basis,
        path_length=float(grid[-1] - grid[0]),
        note=(
            "accumulated over the whole path; rho(s) is the same question asked "
            "at one arc length, and a route can pass that everywhere while "
            "telling you almost nothing about one direction"
        ),
    )


# -- boundaries ------------------------------------------------------------


@dataclass(frozen=True)
class ChartBoundary:
    """Clearance to the edge of the declared parameter domain, in the surface's metric.

    A parameter-domain box is stated in ``u`` and ``v``, and a clearance has to
    be a length on the part, so the margin in each coordinate is converted
    through the metric: ``sqrt(E) du`` and ``sqrt(G) dv``. That is a
    first-order estimate of the distance to the edge and is stated as one --
    exact only where the coordinate curves are orthogonal and the metric is
    locally constant, which is where a chart is well conditioned anyway.

    A periodic coordinate has no edge, and contributes no constraint rather
    than a large one.
    """

    surface: ParametricSurface
    note: str = ""

    def clearance(self, envelope: PathEnvelope) -> Array:
        chart = self.surface.chart
        E, F, G = first_fundamental_form(self.surface.jet_at(envelope.u, envelope.v))
        del F
        margins = []
        if chart.u_period is None:
            if np.isfinite(chart.u_min):
                margins.append(np.sqrt(E) * (envelope.u - chart.u_min))
            if np.isfinite(chart.u_max):
                margins.append(np.sqrt(E) * (chart.u_max - envelope.u))
        if chart.v_period is None:
            if np.isfinite(chart.v_min):
                margins.append(np.sqrt(G) * (envelope.v - chart.v_min))
            if np.isfinite(chart.v_max):
                margins.append(np.sqrt(G) * (chart.v_max - envelope.v))
        if not margins:
            return np.full(envelope.arc_length.size, np.inf)
        return np.min(np.stack(margins, axis=0), axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "chart-boundary",
            "surface": self.surface.name,
            "domain": self.surface.chart.to_dict(),
            "basis": "first-order: sqrt(E) du and sqrt(G) dv, exact where the chart is orthogonal",
            "note": self.note,
        }


@dataclass(frozen=True)
class ObstacleDiscs:
    """Keep-out discs on the part, given by their centres and radii.

    Clearance is measured as the **ambient chord** from the path to each
    centre, minus the radius. The chord is never longer than the distance
    within the surface, so this underestimates the clearance -- which is the
    direction an obstacle constraint has to err in. A route this declares
    clear really is clear; a route it rejects might have been fine.
    """

    centres: Array
    radii: Array
    note: str = ""

    def __post_init__(self) -> None:
        centres = np.asarray(self.centres, dtype=float)
        radii = np.atleast_1d(np.asarray(self.radii, dtype=float))
        if centres.ndim != 2 or centres.shape[1] != 3:
            raise ValueError("obstacle centres must be (m, 3) ambient points")
        if radii.shape != (centres.shape[0],):
            raise ValueError("every obstacle needs exactly one radius")
        if np.any(radii < 0.0) or not np.all(np.isfinite(radii)):
            raise ValueError("obstacle radii must be finite and nonnegative")
        centres.setflags(write=False)
        radii.setflags(write=False)
        object.__setattr__(self, "centres", centres)
        object.__setattr__(self, "radii", radii)

    def clearance(self, envelope: PathEnvelope) -> Array:
        if self.centres.shape[0] == 0:
            return np.full(envelope.arc_length.size, np.inf)
        separations = np.linalg.norm(
            envelope.points[:, None, :] - self.centres[None, :, :], axis=-1
        )
        return np.min(separations - self.radii[None, :], axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "obstacle-discs",
            "count": int(self.centres.shape[0]),
            "radii": self.radii.tolist(),
            "basis": "ambient chord to the centre, minus the radius -- a lower bound",
            "note": self.note,
        }


def combined_clearance(envelope: PathEnvelope, *regions: Any) -> Array:
    """The tightest clearance any declared region allows, sample by sample."""
    if not regions:
        raise ValueError(
            "a boundary clearance with no declared region is not a constraint; "
            "declare the part's edge, its obstacles, or neither"
        )
    return np.min(
        np.stack([region.clearance(envelope) for region in regions], axis=0), axis=0
    )


# -- generating routes -----------------------------------------------------


def route_label(family: str, index: int, quantity: float) -> str:
    """A stable name for a route: its family, its index, and what generated it.

    Built from **declared** values only -- the family, the position in the
    sweep, and the heading or offset the sweep asked for -- and never from
    anything the solver computed. A start point reached by flowing a
    perpendicular geodesic differs in its last bits between platforms, and a
    label formatted from it can therefore round to a different string on a
    different machine. That label then lands in a committed report and moves
    its content hash, which is the failure this repository has already had once
    and fixed once, at the 97.5-degree heading of the torus scan.

    The index alone would be enough to be stable, and is not enough to be
    read; the quantity alone would be readable and not stable. Both, and the
    index is what makes collisions impossible whatever the formatting does.
    """
    return f"{family}/{int(index):02d}/{float(quantity):+.6g}"


def heading_fan(
    surface: ParametricSurface,
    *,
    u0: float,
    v0: float,
    count: int,
    length: float,
    n_steps: int,
    start_degrees: float = 0.0,
    span_degrees: float = 360.0,
) -> dict[str, PathEnvelope]:
    """Every heading from one start point. One family, and the easy one."""
    if int(count) < 2:
        raise ValueError("a fan needs at least two headings")
    degrees = start_degrees + span_degrees * np.arange(count) / float(count)
    envelopes = integrate_paths(
        surface, u0=u0, v0=v0, headings=np.deg2rad(degrees),
        length=length, n_steps=n_steps,
    )
    return {
        route_label("fan", index, float(degree)): envelope
        for index, (degree, envelope) in enumerate(
            zip(degrees, envelopes, strict=True)
        )
    }


def offset_courses(
    surface: ParametricSurface,
    *,
    u0: float,
    v0: float,
    heading: float,
    spacing: float,
    count: int,
    length: float,
    n_steps: int,
) -> dict[str, PathEnvelope]:
    """Parallel courses either side of a seed, which is what a process runs.

    The family a heading fan cannot produce. Offsetting the *start point*
    transversely exercises the ``a`` column -- how a lateral placement error
    propagates -- which a fan over headings never touches, and the two columns
    focus in different places, so a route family chosen on one says nothing
    about the other.

    ``spacing`` is a distance **on the part**, and it is realised exactly: each
    course starts where a geodesic perpendicular to the seed reaches that arc
    length, with its initial direction the parallel transport of the seed's
    along that perpendicular. Stepping linearly in ``u`` and ``v`` instead
    would be right to first order and wrong by ``O(K spacing^2)`` -- about two
    parts in a thousand on a unit torus at a spacing of a tenth, which is the
    same size as the effects this repository measures.

    This is the construction the lateral column is already checked with, which
    is the point: a course family offset any other way would not be the family
    the ``a`` column describes.
    """
    if int(count) < 1:
        raise ValueError("at least one course")
    if float(spacing) <= 0.0:
        raise ValueError("course spacing must be positive")
    surface.require_valid_chart(u0, v0, where="the seed course's start")
    tangent = surface.unit_direction(u0, v0, float(heading))
    normal = surface.perpendicular_direction(u0, v0, *tangent)
    rhs = surface.geodesic_transfer_rhs()
    offsets = (np.arange(count) - (count - 1) / 2.0) * float(spacing)

    courses: dict[str, PathEnvelope] = {}
    for index, offset in enumerate(offsets):
        if offset == 0.0:
            start_u, start_v, direction = u0, v0, tangent
        else:
            sign = 1.0 if offset > 0.0 else -1.0
            side = surface.state_from_tangent(
                u0, v0, sign * normal[0], sign * normal[1]
            )
            _, walk = integrate(
                rhs, side, length=abs(float(offset)), n_steps=n_steps
            )
            start_u, start_v = float(walk[-1, 0]), float(walk[-1, 1])
            transported = surface.perpendicular_direction(
                start_u, start_v, float(walk[-1, 2]), float(walk[-1, 3])
            )
            # The +side left along +normal, so undoing that rotation flips the
            # sign -- the same bookkeeping the lateral-column check does.
            direction = (-sign * transported[0], -sign * transported[1])
        surface.require_valid_chart(
            start_u, start_v, where=f"the course offset by {offset:g}"
        )
        envelope = integrate_paths(
            surface, u0=float(start_u), v0=float(start_v),
            headings=[float(surface.heading_of(start_u, start_v, *direction))],
            length=length, n_steps=n_steps,
        )[0]
        # Keyed by the offset that was *asked for*, not by the point the flow
        # landed on: the offset is (i - (n-1)/2) * spacing, exact small
        # integers against a declared spacing, and identical on every machine.
        courses[route_label("course", index, float(offset))] = envelope
    return courses


# -- the front -------------------------------------------------------------

#: Which way is better for each declared objective.
DIRECTIONS: tuple[str, ...] = ("lower-is-better", "higher-is-better")


@dataclass(frozen=True)
class Objective:
    """One declared quantity a route is judged on, and the limit that scales it.

    ``limit`` is what makes the value dimensionless: the ratio of a cross-track
    error to the cross-track tolerance is a pure number and comparable with the
    ratio of a path length to the process maximum, while the two raw values are
    not comparable with anything. An objective with no limit can still be
    ranked -- a front needs only an ordering -- but it cannot be weighted into
    a scalar, and :func:`weighted_cost` says so rather than quietly using the
    raw number and letting the weights carry the units.
    """

    name: str
    direction: str
    limit: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if self.limit is not None:
            value = float(self.limit)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"the limit on {self.name!r} must be finite and positive")
            object.__setattr__(self, "limit", value)

    def normalise(self, value: float) -> float:
        """The value as a fraction of its declared limit, oriented so lower is better."""
        if self.limit is None:
            raise ValueError(
                f"objective {self.name!r} declares no limit, so its value cannot be "
                "made dimensionless; a weight applied to the raw number would be "
                "carrying its units"
            )
        ratio = float(value) / self.limit
        return ratio if self.direction == "lower-is-better" else 1.0 / max(ratio, 1e-300)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "limit": self.limit,
            "note": self.note,
        }


def _oriented(objective: Objective, value: float) -> float:
    """The raw value, sign-flipped so that smaller is always better."""
    return float(value) if objective.direction == "lower-is-better" else -float(value)


def pareto_front(
    routes: dict[str, dict[str, float]], objectives: tuple[Objective, ...]
) -> dict[str, Any]:
    """The routes nothing else beats on every objective at once.

    The honest answer to a multi-objective choice, and usually a short list.
    Ranking by a weighted sum instead does not remove the trade-off; it hides
    it inside weights that then make the decision, and it can pick a route that
    is worse than another on *every* declared quantity if the weights are odd
    enough.
    """
    if not objectives:
        raise ValueError("a front needs at least one declared objective")
    names = list(routes)
    missing = {
        name: [o.name for o in objectives if o.name not in routes[name]] for name in names
    }
    incomplete = {name: keys for name, keys in missing.items() if keys}
    if incomplete:
        raise ValueError(
            f"these routes do not report every declared objective: {incomplete}. A "
            "route missing an objective must not be ranked as though it passed it"
        )

    values = {
        name: np.array([_oriented(o, routes[name][o.name]) for o in objectives])
        for name in names
    }
    front: list[str] = []
    dominated_by: dict[str, str] = {}
    for name in names:
        mine = values[name]
        beaten_by = None
        for other in names:
            if other == name:
                continue
            theirs = values[other]
            if np.all(theirs <= mine) and np.any(theirs < mine):
                beaten_by = other
                break
        if beaten_by is None:
            front.append(name)
        else:
            dominated_by[name] = beaten_by
    return {
        "objectives": [o.to_dict() for o in objectives],
        "front": front,
        "front_size": len(front),
        "dominated": dominated_by,
        "note": (
            "nothing on the front is beaten on every objective at once; choosing "
            "among them needs a declared preference, which is not a property of "
            "the geometry"
        ),
    }


def weighted_cost(
    values: dict[str, float], objectives: tuple[Objective, ...], weights: dict[str, float]
) -> dict[str, Any]:
    """Collapse the front to one number, with weights the caller had to declare.

    Every objective must have a declared limit, so every term is a pure ratio
    and the weights are pure numbers rather than the carriers of a unit
    conversion nobody wrote down. Every objective must have a weight, because
    an omitted one is a weight of zero chosen by accident.
    """
    declared = {o.name for o in objectives}
    if set(weights) != declared:
        raise ValueError(
            f"weights must name exactly the declared objectives {sorted(declared)}, "
            f"not {sorted(weights)}; an objective with no weight has been given "
            "zero by accident rather than by decision"
        )
    if any(float(weight) < 0.0 for weight in weights.values()):
        raise ValueError("a negative weight turns an objective into its opposite")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"the weights sum to {total!r} rather than 1; a scale on the whole cost "
            "is not a preference between its terms and hides one of the two"
        )
    terms = {o.name: o.normalise(values[o.name]) for o in objectives}
    return {
        "cost": float(sum(weights[name] * term for name, term in terms.items())),
        "terms": terms,
        "weights": {name: float(weight) for name, weight in weights.items()},
        "note": "every term is a ratio to a declared limit, so the weights carry no units",
    }


__all__ = [
    "DIRECTIONS",
    "GRAMIAN_SCALES",
    "ChartBoundary",
    "ObstacleDiscs",
    "Objective",
    "Observability",
    "combined_clearance",
    "heading_fan",
    "observability_gramian",
    "offset_courses",
    "pareto_front",
    "route_label",
    "weighted_cost",
]


def _interpolated(grid: Array, values: Array, at: float) -> Array:
    """Linear interpolation of a stack of matrices at one arc length."""
    position = float(np.clip(at, grid[0], grid[-1]))
    index = int(np.clip(np.searchsorted(grid, position, side="right") - 1, 0, grid.size - 2))
    span = grid[index + 1] - grid[index]
    weight = 0.0 if span <= 0.0 else (position - grid[index]) / span
    return values[index] + weight * (values[index + 1] - values[index])


@dataclass(frozen=True)
class StackedObservability:
    """``W = A^T R^-1 A`` over the stacked observation vector.

    The other form of the same question, and genuinely a different object.
    ``int Phi^T H^T R^-1 H Phi ds`` treats ``R`` as a noise *density* and needs
    the samples to be independent; this treats ``R`` as the covariance of the
    measurements actually taken, and is the only form that admits a correlated
    one. A filter correlates arc lengths, so that is not an exotic case -- and
    a filtered campaign that ranked routes by the integral form would be
    counting information it did not have, because the samples it averaged were
    already averages of each other.

    On a uniform grid with a stationary ``R`` the two agree up to the sample
    spacing: ``W_integral ~ h W_stacked``. The conversion is declared here
    rather than left implicit, because it is exactly the factor that decides
    whether two campaigns at different sampling rates are comparable.
    """

    arclength: Array
    total: Array
    scale: Array
    scale_basis: str
    noise_structure: str
    samples: int
    note: str = ""

    @property
    def eigenvalues(self) -> Array:
        return np.linalg.eigvalsh(self.total)

    @property
    def worst_direction(self) -> Array:
        values, vectors = np.linalg.eigh(self.total)
        return vectors[:, int(np.argmin(values))]

    @property
    def condition_number(self) -> float:
        values = self.eigenvalues
        smallest = float(values[0])
        return float("inf") if smallest <= 0.0 else float(values[-1]) / smallest

    @property
    def numerical_rank(self) -> int:
        """How many starting-pose directions these measurements constrain."""
        return int(np.linalg.matrix_rank(self.total))

    def to_dict(self) -> dict[str, Any]:
        values = self.eigenvalues
        return {
            "form": "stacked",
            # Per-sample, not a density: R here is the covariance of the
            # measurements actually taken, so no ds appears and the samples
            # need not be independent.
            "noise_convention": "per-sample-covariance",
            "scale_basis": self.scale_basis,
            "scale": self.scale.tolist(),
            "noise_structure": self.noise_structure,
            "samples": int(self.samples),
            "eigenvalues": values.tolist(),
            "numerical_rank": self.numerical_rank,
            "worst_observed": float(values[0]),
            "best_observed": float(values[-1]),
            "condition_number": self.condition_number,
            "worst_direction": self.worst_direction.tolist(),
            "note": self.note,
        }


def stacked_observability(
    source: Any,
    observation: ObservationModel,
    noise,
    *,
    max_lateral: float | None = None,
    max_heading: float | None = None,
    prior_covariance=None,
    window: tuple[float, float] | None = None,
) -> StackedObservability:
    """``A^T R^-1 A`` with the full ``R``, made dimensionless the same way.

    ``window`` restricts the calculation to the samples inside an arc-length
    interval, and does so by inverting the *submatrix* of ``R`` rather than
    taking a submatrix of ``R^-1``. Those are different matrices whenever the
    noise is correlated, and only the first is the information the measurements
    in that window carry on their own -- the second is what they carry given
    the ones outside it, which is not a quantity a route planner can act on
    before those have been taken.
    """
    from .output_covariance import NoiseModel, stacked_operator

    if not isinstance(noise, NoiseModel):
        raise TypeError(
            "pass a NoiseModel: the structure of R -- stationary, per sample or "
            "correlated -- is what decides whether this form or the integral one "
            "is the right question, so it is not inferred from an array's shape"
        )
    record: TransferRecord = to_transfer_record(source)
    if observation.mode != record.observation_mode:
        raise ValueError(
            f"the record is in {record.observation_mode!r} but this observation "
            f"model reports {observation.mode!r}"
        )
    scale, basis = _declared_scaling(max_lateral, max_heading, prior_covariance)

    grid = record.arclength
    operator = stacked_operator(record, observation)
    covariance = noise.stacked(grid.size)
    width = len(observation.outputs)

    if window is not None:
        lower, upper = (float(value) for value in window)
        if upper <= lower:
            raise ValueError("a window needs end > start")
        inside = np.flatnonzero((grid >= lower) & (grid <= upper))
        if inside.size < 1:
            raise ValueError(f"no sample lies in [{lower}, {upper}]")
        rows = np.concatenate([inside * width + offset for offset in range(width)])
        rows.sort()
        operator = operator[rows]
        covariance = covariance[np.ix_(rows, rows)]
        samples = int(inside.size)
    else:
        samples = int(grid.size)

    information = np.linalg.inv(covariance)
    total = scale.T @ (operator.T @ information @ operator) @ scale
    return StackedObservability(
        arclength=grid,
        total=0.5 * (total + total.T),
        scale=scale,
        scale_basis=basis,
        noise_structure=noise.structure,
        samples=samples,
        note=(
            "A^T R^-1 A over the measurements actually taken; on a uniform grid "
            "with a stationary R this is the integral Gramian divided by the "
            "sample spacing"
        ),
    )


def _declared_scaling(
    max_lateral: float | None, max_heading: float | None, prior_covariance
) -> tuple[Array, str]:
    """The one scaling the caller declared, or an error naming both options."""
    box = max_lateral is not None or max_heading is not None
    if box == (prior_covariance is not None):
        raise ValueError(
            "declare exactly one scaling: a tolerance box (max_lateral and "
            "max_heading) or a prior covariance. The Gramian's entries do not "
            "share units, so an unscaled one cannot be ranked by"
        )
    if box:
        if max_lateral is None or max_heading is None:
            raise ValueError("a tolerance box needs both max_lateral and max_heading")
        if float(max_lateral) <= 0.0 or float(max_heading) <= 0.0:
            raise ValueError("both tolerance scales must be positive")
        return np.diag([float(max_lateral), float(max_heading)]), "tolerance-box"
    return np.linalg.cholesky(validated_covariance(prior_covariance, "C0")), "prior-covariance"
