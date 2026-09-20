"""Stage two: the same two objects on surfaces where the curvature varies.

The constant-curvature stage could check every number against a closed form.
Here there is none, so the verification has to come from somewhere else. Four
sources are used, in decreasing order of strength:

1. **Anchoring.** The general machinery is run on the surfaces whose answer is
   already known -- a plate, a rolled sheet, a sphere, a pseudosphere -- and
   must reproduce ``s``, ``s``, ``sin s`` and ``sinh s``. If it cannot recover
   the calibrated case it has no business on a saddle.
2. **Two independent routes.** The Jacobi field is obtained both by integrating
   ``j'' + K(gamma(s)) j = 0`` along the path and by central-differencing the
   geodesic flow itself in the initial heading. The difference between them is
   second order in ``eps`` on every surface, with coefficient

   .. math::

       \frac{\\mathrm{cn}_K(s)^2 + \\kappa_n^2\\,\\mathrm{sn}_K(s)^2}{6},

   where ``kappa_n`` is the normal curvature transverse to the path. The first
   term is intrinsic -- the finite variation is not the derivative -- and the
   second is the price of measuring a straight-line chord in space instead of
   a distance in the surface. On a plate (no bending) and on a unit sphere
   (``cn^2 + sn^2 = 1``) the two terms collapse to exactly ``1/6``, and there
   the whole finite-``eps`` answer is known in closed form and is checked
   sample by sample. Elsewhere the exponent is checked and the coefficient
   reported, because a reconstructed pair of 3-D points gives a chord and
   someone will need that number.
3. **Self-convergence.** With no reference solution, the step size is halved
   against a finer run of the same solver and the order of accuracy is fitted.
4. **Invariants.** Unit speed is never re-imposed, so its drift is a free
   diagnostic; and the numerically assembled Gaussian curvature is compared
   with the analytic one wherever the surface knows it.

The cost of the convenience of finite-difference derivatives is measured, not
assumed: every surface that supplies analytic derivatives is also run without
them, and the difference is reported.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from .. import __version__
from .analysis import fit_power_law, successive_orders
from .contract import GeometryUncertainty
from .envelope import (
    PathEnvelope,
    estimate_convergence,
    finite_difference_jacobi,
    finite_difference_lateral,
    integrate_path,
    integrate_paths,
    scan_headings,
)
from .experiment import _check, _jsonable, content_hash
from .observation import catalogue as observation_catalogue
from .observation_model import ObservationModel
from .planning import (
    ChartBoundary,
    Objective,
    combined_clearance,
    heading_fan,
    observability_gramian,
    offset_courses,
    pareto_front,
)
from .prediction import (
    chord_from_intrinsic,
    chord_from_tangent,
    first_order_prediction,
    has_closed_form_separation,
    intrinsic_from_tangent,
)
from .routing import CoverageSpec, RouteConstraints, rank_routes
from .surfaces import (
    Chart,
    ParametricSurface,
    cylinder,
    hyperbolic_paraboloid,
    plane,
    pseudosphere,
    sphere,
    torus,
)
from .tracking import AcquisitionSpec
from .uncertainty import (
    budget,
    calibration_transform,
    fixture_datum,
    path_registration,
    sensor_noise,
    starting_pose,
    surface_reconstruction,
)

REPORT_SCHEMA = "geodesic-jacobi-surfaces-v3"
SUPERSEDES = "geodesic-jacobi-surfaces-v2"


@dataclass(frozen=True)
class SurfaceCase:
    """A surface together with a nominal path on it."""

    key: str
    surface: ParametricSurface
    u0: float
    v0: float
    heading: float
    length: float
    reference: str | None = None
    chord_is_closed_form: bool = False

    def closed_form(self, s: np.ndarray) -> np.ndarray | None:
        """The heading column ``b(s)``."""
        if self.reference is None:
            return None
        return {"s": lambda x: x, "sin(s)": np.sin, "sinh(s)": np.sinh}[self.reference](s)

    def lateral_closed_form(self, s: np.ndarray) -> np.ndarray | None:
        """The lateral column ``a(s) = cn_K(s)``."""
        if self.reference is None:
            return None
        return {"s": np.ones_like, "sin(s)": np.cos, "sinh(s)": np.cosh}[self.reference](s)

    def closed_form_variation(self, s: float, epsilons: np.ndarray) -> np.ndarray | None:
        """Exact ``|J|`` from a central difference, where the chord is known exactly.

        On the plate and on the unit sphere the ambient chord between the two
        perturbed geodesics is ``2 sn_K(s) sin(eps)`` exactly, so the measured
        quantity is ``sn_K(s) sin(eps)/eps`` with no expansion involved.
        """
        if not self.chord_is_closed_form:
            return None
        reference = self.closed_form(np.asarray(s, dtype=float))
        return np.abs(reference) * np.sin(epsilons) / epsilons


def default_cases() -> tuple[SurfaceCase, ...]:
    return (
        SurfaceCase("plate", plane(), 0.0, 0.0, 0.6, 2.0, "s", chord_is_closed_form=True),
        SurfaceCase("rolled-sheet", cylinder(1.0), 0.0, 0.0, 0.6, 2.0, "s"),
        SurfaceCase(
            "spherical-cap", sphere(1.0), np.pi / 2, 0.0, 0.6, 4.0, "sin(s)",
            chord_is_closed_form=True,
        ),
        SurfaceCase("pseudosphere", pseudosphere(), 1.2, 0.0, 0.6, 1.5, "sinh(s)"),
        SurfaceCase("saddle", hyperbolic_paraboloid(1.0), 0.3, 0.2, 0.6, 2.0, None),
        SurfaceCase("torus", torus(2.0, 1.0), 0.3, 0.2, 0.6, 2.0, None),
    )


@dataclass(frozen=True)
class SurfaceConfig:
    n_steps: int = 2000
    finite_difference_steps: int = 1000
    reference_steps: int = 2560
    step_counts: tuple[int, ...] = field(default_factory=lambda: (40, 80, 160, 320))
    self_convergence_length: float = 2.0
    # Keep the declared grid platform-stable.  Recomputing it with
    # ``np.logspace`` can change the final bit across NumPy/BLAS builds,
    # making a committed report appear stale even when the experiment is
    # unchanged.
    epsilons: tuple[float, ...] = (
        1.0e-4,
        1.7782794100389228e-4,
        3.1622776601683794e-4,
        5.623413251903491e-4,
        1.0e-3,
        1.7782794100389228e-3,
        3.1622776601683794e-3,
        5.623413251903491e-3,
        1.0e-2,
        1.7782794100389228e-2,
        3.1622776601683794e-2,
        5.6234132519034905e-2,
        1.0e-1,
    )
    two_route_arc_length: float = 1.0
    two_route_steps: int = 1000
    fit_floor: float = 1e-8
    fit_ceiling: float = 1e-3

    #: Two resolutions, four times apart, so that the error budget's own order
    #: can be read off it: a budget that does not fall as h^4 is not measuring
    #: the truncation error of a fourth-order method.
    budget_step_counts: tuple[int, ...] = (160, 640)
    #: Below this a "true error" is the reference's own accumulated rounding,
    #: and comparing an estimate against it compares two pieces of noise.
    budget_roundoff_floor: float = 1e-12
    jet_sensitivity_steps: int = 400
    jet_relative_steps: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    chain_steps: int = 800
    #: Small enough that the second-order corrections dominate, large enough
    #: that the finite-difference reference is not cancellation-limited.
    chain_epsilons: tuple[float, ...] = (5e-3, 2.5e-3)

    #: The declared example campaign the uncertainty budget is assembled from.
    #: Every one of these is a *declaration*, not a measurement: nothing here
    #: has been on a bench, and the budget exists to show the shape of the
    #: arithmetic and which term would dominate, not to characterise hardware.
    budget_path_steps: int = 400
    scan_position_sigma: float = 2.0e-4
    scan_normal_sigma: float = 1.0e-4
    scan_curvature_sigma: float = 1.0e-3
    fixture_lateral_sigma: float = 2.0e-4
    fixture_heading_sigma: float = 1.0e-4
    calibration_offset_sigma: float = 5.0e-5
    registration_sigma: float = 2.0e-3
    sensor_correlation_length: float = 0.05

    #: Route planning. The limits are what make each objective dimensionless;
    #: they are declared here and they decide nothing on their own -- the front
    #: is reported, and collapsing it needs weights nobody has declared.
    planning_route_count: int = 8
    planning_course_spacing: float = 0.1
    planning_amplification_limit: float = 5.0
    planning_observability_limit: float = 1.0e3
    planning_clearance_limit: float = 0.05

    transverse_tolerances: tuple[float, ...] = (1e-3, 1e-2)
    heading_scan_count: int = 24
    heading_scan_key: str = "torus"
    # Long enough for the geodesics to leave the neighbourhood they started in
    # and sample both signs of the torus's curvature; at a short path length
    # every heading looks much the same and there is no decision to make.
    heading_scan_length: float = 6.0
    heading_scan_steps: int = 3000
    # Reported, not used as a criterion: a heading whose Jacobi field dips this
    # low away from the start is near a focus.
    focus_margin_report_level: float = 0.25

    # What actually decides a route. These are process limits in units of the
    # radius of curvature and radians, not properties of the mathematics, and
    # they are the thing a user is expected to replace.
    route_tolerance_lateral: float = 1.0e-3
    route_tolerance_heading: float = 1.0e-3
    route_max_cross_track_error: float = 2.0e-2
    route_max_heading_error: float = 5.0e-2
    route_nominal_spacing: float = 0.10
    route_swath_width: float = 0.14
    route_min_coverage_margin: float = 0.0
    # The instrument, in units of the radius of curvature. A 25 um scanner on a
    # 300 mm coupon, and a starting heading held to a tenth of a degree.
    route_measurement_sigma: float = 25.0e-6 / 0.3
    route_heading_sigma_degrees: float = 0.1
    route_lateral_sigma: float = 0.0
    # The sensor's acquisition schedule. Two thresholds, because one chatters
    # on noise; a window, a latency limit and a tolerated loss, because a route
    # acquired at its last sample was never tracked.
    route_acquire_threshold: float = 5.0
    route_hold_threshold: float = 3.0
    route_acquisition_window: float = 0.05
    route_max_acquisition_distance: float = 1.0
    route_min_tracked_distance: float = 3.0
    route_max_loss_distance: float = 0.1
    curve_samples: int = 200

    analytic_tolerance: float = 1e-11
    finite_difference_tolerance: float = 1e-7
    order_tolerance: float = 0.2
    exponent_tolerance: float = 0.05
    coefficient_tolerance: float = 1e-2
    chord_closed_form_tolerance: float = 1e-9
    exactness_threshold: float = 1e-13
    speed_drift_tolerance: float = 1e-9
    wronskian_tolerance: float = 1e-9
    curvature_tolerance: float = 1e-11
    curvature_tolerance_fd: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _uses_finite_differences(surface: ParametricSurface) -> bool:
    return surface.jet is None


# ---------------------------------------------------------------------------
# 1. anchoring: recover the calibrated answers the general way
# ---------------------------------------------------------------------------
def anchor_to_closed_forms(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.reference is None:
            continue
        envelope = integrate_path(
            case.surface,
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            length=case.length,
            n_steps=config.n_steps,
        )
        expected = case.closed_form(envelope.arc_length)
        finite_difference = _uses_finite_differences(case.surface)
        rows.append(
            {
                "case": case.key,
                "surface": case.surface.name,
                "description": case.surface.description,
                "reference": case.reference,
                "derivatives": "finite-difference" if finite_difference else "analytic",
                "tolerance": (
                    config.finite_difference_tolerance
                    if finite_difference
                    else config.analytic_tolerance
                ),
                "lateral_reference": {"s": "1", "sin(s)": "cos(s)", "sinh(s)": "cosh(s)"}[
                    case.reference
                ],
                "max_jacobi_error": float(np.max(np.abs(envelope.jacobi_field - expected))),
                "max_lateral_error": float(
                    np.max(np.abs(envelope.lateral_basis - case.lateral_closed_form(
                        envelope.arc_length
                    )))
                ),
                "max_wronskian_drift": float(
                    np.max(envelope.transfer_map.wronskian_drift)
                ),
                "max_speed_drift": float(np.max(envelope.speed_drift)),
                "max_curvature_error": _curvature_error(case.surface, envelope),
                "focus_points": envelope.focus_points(),
            }
        )
    return rows


def _curvature_error(surface: ParametricSurface, envelope: PathEnvelope) -> float | None:
    if surface.exact_curvature is None:
        return None
    exact = np.asarray(surface.exact_curvature(envelope.u, envelope.v), dtype=float)
    return float(np.max(np.abs(envelope.curvature - exact)))


# ---------------------------------------------------------------------------
# 2. two independent routes to the same Jacobi field
# ---------------------------------------------------------------------------
def compare_two_routes(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    epsilons = np.asarray(config.epsilons, dtype=float)
    rows: list[dict[str, Any]] = []
    for case in cases:
        length = max(config.two_route_arc_length, 0.0)
        envelope = integrate_path(
            case.surface,
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            length=length,
            n_steps=config.two_route_steps,
        )
        from_equation = abs(float(envelope.jacobi_field[-1]))
        _, measured = finite_difference_jacobi(
            case.surface,
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            epsilon=epsilons,
            length=length,
            n_steps=config.two_route_steps,
        )
        deviation = np.abs(measured[-1] / from_equation - 1.0)
        fit = fit_power_law(
            epsilons, deviation, y_floor=config.fit_floor, y_ceiling=config.fit_ceiling
        )
        constant_curvature = bool(np.ptp(envelope.curvature) < 1e-9)
        exact = case.closed_form_variation(length, epsilons)
        chord_error = (
            None if exact is None else float(np.max(np.abs(measured[-1] / exact - 1.0)))
        )
        rows.append(
            {
                "case": case.key,
                "surface": case.surface.name,
                "arc_length": float(length),
                "observation_mode": "ambient-euclidean-chord",
                "jacobi_from_equation": from_equation,
                "constant_curvature": constant_curvature,
                "expected_exponent": 2.0,
                "fitted_exponent": float(fit.exponent),
                "fitted_coefficient": float(fit.prefactor),
                "intrinsic_coefficient": 1.0 / 6.0,
                "chord_excess_coefficient": float(fit.prefactor - 1.0 / 6.0),
                "chord_closed_form": (
                    "|J| = sn_K(s) sin(eps)/eps" if exact is not None else None
                ),
                "chord_closed_form_max_relative_error": chord_error,
                "fit": fit.to_dict(),
                "samples": [
                    {"epsilon": float(e), "relative_difference": float(d)}
                    for e, d in zip(epsilons, deviation, strict=True)
                ],
            }
        )
    return rows


def compare_lateral_route(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    """The first column of ``Phi``, checked the way the second one is.

    ``b`` is validated by perturbing the initial heading. ``a`` gets its own,
    structurally different perturbation: the start point is moved sideways
    along the perpendicular geodesic and the initial direction is parallel
    transported to it. Nothing about that construction touches the Jacobi
    equation, so agreement is evidence rather than bookkeeping.
    """
    epsilons = np.asarray(config.epsilons, dtype=float)
    rows: list[dict[str, Any]] = []
    for case in cases:
        length = config.two_route_arc_length
        envelope = integrate_path(
            case.surface,
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            length=length,
            n_steps=config.two_route_steps,
        )
        from_equation = abs(float(envelope.lateral_basis[-1]))
        _, measured = finite_difference_lateral(
            case.surface,
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            epsilon=epsilons,
            length=length,
            n_steps=config.two_route_steps,
        )
        deviation = np.abs(measured[-1] / from_equation - 1.0)
        exact = bool(np.max(deviation) < config.exactness_threshold)
        fit = fit_power_law(
            epsilons, deviation, y_floor=config.fit_floor, y_ceiling=config.fit_ceiling
        )
        rows.append(
            {
                "case": case.key,
                "surface": case.surface.name,
                "arc_length": float(length),
                "observation_mode": "ambient-euclidean-chord",
                "lateral_from_equation": from_equation,
                "lateral_from_displaced_start": float(measured[-1][0]),
                "regime": "exact-to-roundoff" if exact else "second-order",
                "expected_exponent": 2.0,
                "fitted_exponent": None if exact else float(fit.exponent),
                "max_deviation": float(np.max(deviation)),
                "fit": fit.to_dict(),
                "samples": [
                    {"epsilon": float(e), "relative_difference": float(d)}
                    for e, d in zip(epsilons, deviation, strict=True)
                ],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 3. self-convergence where no reference solution exists
# ---------------------------------------------------------------------------
def measure_self_convergence(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        length = config.self_convergence_length
        reference = integrate_path(
            case.surface,
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            length=length,
            n_steps=config.reference_steps,
        )
        steps, jacobi_errors, position_errors, levels = [], [], [], []
        for n_steps in config.step_counts:
            if config.reference_steps % n_steps:
                raise ValueError("every level must be a divisor of the reference resolution")
            stride = config.reference_steps // n_steps
            coarse = integrate_path(
                case.surface,
                u0=case.u0,
                v0=case.v0,
                heading=case.heading,
                length=length,
                n_steps=n_steps,
            )
            jacobi_error = float(
                np.max(np.abs(coarse.jacobi_field - reference.jacobi_field[::stride]))
            )
            position_error = float(
                np.max(
                    np.linalg.norm(coarse.points - reference.points[::stride], axis=-1)
                )
            )
            steps.append(length / n_steps)
            jacobi_errors.append(jacobi_error)
            position_errors.append(position_error)
            levels.append(
                {
                    "n_steps": int(n_steps),
                    "h": float(length / n_steps),
                    "jacobi_error": jacobi_error,
                    "position_error": position_error,
                }
            )
        jacobi_fit = fit_power_law(steps, jacobi_errors, y_floor=config.exactness_threshold)
        position_fit = fit_power_law(steps, position_errors, y_floor=config.exactness_threshold)
        exact = (
            max(max(jacobi_errors), max(position_errors)) < config.exactness_threshold
        )
        rows.append(
            {
                "case": case.key,
                "surface": case.surface.name,
                "length": float(length),
                "reference_steps": int(config.reference_steps),
                "expected_order": 4,
                "regime": "exact-to-roundoff" if exact else "converging",
                "max_error_over_levels": float(max(max(jacobi_errors), max(position_errors))),
                "fitted_order_jacobi": None if exact else float(jacobi_fit.exponent),
                "fitted_order_position": None if exact else float(position_fit.exponent),
                "jacobi_fit": jacobi_fit.to_dict(),
                "position_fit": position_fit.to_dict(),
                "successive_orders_jacobi": successive_orders(steps, jacobi_errors),
                "levels": levels,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 4. the cost of not supplying analytic derivatives
# ---------------------------------------------------------------------------
def measure_route_planning(config: SurfaceConfig, cases) -> dict[str, Any]:
    """Route selection with every declared quantity kept apart.

    Four things this establishes, none of which a single ranking scalar can.

    **Accumulated observability is not sampled resolvability.** The Gramian
    ``int Phi^T H^T R^-1 H Phi ds`` answers a question about the whole path
    that ``rho(s)`` answers about a sample, and its worst eigenvalue names the
    direction of starting-pose error the route says least about. Its *density*
    -- per unit path length -- is scale invariant, which is the property a
    route criterion has to have and the reason the accumulated figure is
    reported beside it rather than instead of it.

    **The offset family is not the heading family.** A fan over headings
    exercises the ``b`` column; a set of parallel courses offset from a seed
    exercises ``a``. The two columns focus in different places, so a route
    family chosen on one says nothing about the other, and this sweep reports
    both.

    **Boundaries are computed from the declared part.** The chart's own edge,
    through the surface metric, rather than a clearance array supplied on a
    grid that might not match.

    **The answer is a front.** The routes nothing else beats on every declared
    objective at once, with the dominated ones and what beat them. Collapsing
    that needs weights, and the weights would then be the decision.
    """
    case = next(entry for entry in cases if entry.key == config.heading_scan_key)
    observation, initial_covariance = _instrument(config)
    lateral = config.route_tolerance_lateral
    heading = config.route_tolerance_heading

    families: dict[str, dict[str, Any]] = {}
    for family, routes in (
        (
            "heading-fan",
            heading_fan(
                case.surface, u0=case.u0, v0=case.v0,
                count=config.planning_route_count,
                length=config.heading_scan_length,
                n_steps=config.heading_scan_steps,
            ),
        ),
        (
            "offset-courses",
            offset_courses(
                case.surface, u0=case.u0, v0=case.v0, heading=case.heading,
                spacing=config.planning_course_spacing,
                count=config.planning_route_count,
                length=config.heading_scan_length,
                n_steps=config.heading_scan_steps,
            ),
        ),
    ):
        boundary = ChartBoundary(case.surface)
        rows: dict[str, dict[str, float]] = {}
        detail: list[dict[str, Any]] = []
        for label, envelope in routes.items():
            record = envelope.as_transfer_record(
                observation_mode="ambient-euclidean-chord"
            )
            gramian = observability_gramian(
                record, observation, max_lateral=lateral, max_heading=heading
            )
            clearance = combined_clearance(envelope, boundary)
            rho = observation.resolvability(record, initial_covariance)
            profile = np.min(rho, axis=1) if rho.ndim > 1 else rho
            values = {
                "amplification": record.amplification_score(lateral, heading),
                "cross_track_error": float(
                    np.max(record.cross_track_error(lateral, heading))
                ),
                "heading_error": float(np.max(record.heading_error(lateral, heading))),
                "worst_observed": float(gramian.eigenvalues[0]),
                "boundary_clearance": float(np.min(clearance)),
                "path_length": float(record.arclength[-1] - record.arclength[0]),
            }
            rows[label] = values
            detail.append(
                values
                | {
                    "label": label,
                    "start": {
                        "u": float(envelope.start[0]),
                        "v": float(envelope.start[1]),
                        "heading": float(envelope.start[2]),
                    },
                    "observability": gramian.to_dict(),
                    "min_resolvability_after_start": float(np.min(profile[1:])),
                    "focus_points": record.focus_events(),
                    "chart_complete": bool(record.chart.complete),
                }
            )
        objectives = _planning_objectives(config)
        families[family] = {
            "routes": len(rows),
            "objectives": [o.to_dict() for o in objectives],
            "front": pareto_front(rows, objectives),
            "detail": [
                {key: value for key, value in row.items() if key != "focus_points"}
                for row in detail
            ],
            "observability_anisotropy": [
                row["observability"]["anisotropy"] for row in detail
            ],
        }

    return {
        "case": case.key,
        "surface": case.surface.name,
        "families": families,
        "scale_invariance": _gramian_scale_invariance(config),
        "monotone": _gramian_monotonicity(config, case),
        "front_is_really_undominated": _front_is_sound(families),
        "families_differ": _families_differ(families),
        "note": (
            "the two families exercise different columns of Phi; a route chosen "
            "on one says nothing about the other"
        ),
    }


def _gramian_scale_invariance(config: SurfaceConfig) -> dict[str, Any]:
    """The same physical situation at twice the size, in information density.

    The accumulated Gramian doubles, and should: twice the path really does
    carry twice the information. Per unit path length it must not move at all,
    and that is the property a route criterion needs -- the same reason ``rho``
    is checked this way and a threshold on ``|b|`` cannot be.
    """
    densities = []
    totals = []
    for factor in (1.0, 2.0):
        envelope = integrate_path(
            sphere(factor), u0=np.pi / 2, v0=0.0, heading=0.6,
            length=3.0 * factor, n_steps=config.heading_scan_steps,
        )
        record = envelope.as_transfer_record(observation_mode="ambient-euclidean-chord")
        gramian = observability_gramian(
            record,
            ObservationModel.transverse_only(
                config.route_measurement_sigma * factor,
                mode="ambient-euclidean-chord",
            ),
            max_lateral=config.route_tolerance_lateral * factor,
            max_heading=config.route_tolerance_heading,
        )
        densities.append(np.linalg.eigvalsh(gramian.per_unit_length))
        totals.append(np.linalg.eigvalsh(gramian.total))
    reference = np.maximum(np.abs(densities[0]), 1e-300)
    return {
        "identity": "W / L is unchanged when the whole situation is drawn at twice the size",
        "density_relative_difference": float(
            np.max(np.abs(densities[1] - densities[0]) / reference)
        ),
        "accumulated_ratio": float(np.max(totals[1] / np.maximum(totals[0], 1e-300))),
        "expected_accumulated_ratio": 2.0,
        "densities": [values.tolist() for values in densities],
    }


def _gramian_monotonicity(config: SurfaceConfig, case) -> dict[str, Any]:
    """Information accumulates: ``W(s2) - W(s1)`` is positive semi-definite.

    Not a tautology about the code -- it is a statement that the integrand
    ``Phi^T H^T R^-1 H Phi`` is itself positive semi-definite everywhere, which
    is what makes the Gramian a measure of information rather than an integral
    that happens to grow.
    """
    envelope = integrate_path(
        case.surface, u0=case.u0, v0=case.v0, heading=case.heading,
        length=config.heading_scan_length, n_steps=config.heading_scan_steps,
    )
    record = envelope.as_transfer_record(observation_mode="ambient-euclidean-chord")
    observation, _ = _instrument(config)
    gramian = observability_gramian(
        record, observation,
        max_lateral=config.route_tolerance_lateral,
        max_heading=config.route_tolerance_heading,
    )
    increments = np.diff(gramian.cumulative, axis=0)
    smallest = float(np.min(np.linalg.eigvalsh(increments)))
    return {
        "identity": "W(s2) - W(s1) is positive semi-definite for s2 > s1",
        "most_negative_increment_eigenvalue": smallest,
        "samples": int(increments.shape[0]),
    }


def _front_is_sound(families: dict[str, Any]) -> dict[str, Any]:
    """Every route said to be dominated really is, on every objective at once.

    The front is the deliverable, so the claim behind it is checked rather than
    trusted: for each dominated route, the route named as beating it must be no
    worse on every declared objective and strictly better on one.
    """
    violations = []
    for family, payload in families.items():
        values = {row["label"]: row for row in payload["detail"]}
        objectives = payload["objectives"]
        for loser, winner in payload["front"]["dominated"].items():
            better_everywhere = True
            strictly_better_somewhere = False
            for objective in objectives:
                name, direction = objective["name"], objective["direction"]
                mine, theirs = values[loser][name], values[winner][name]
                if direction == "higher-is-better":
                    mine, theirs = -mine, -theirs
                if theirs > mine:
                    better_everywhere = False
                if theirs < mine:
                    strictly_better_somewhere = True
            if not (better_everywhere and strictly_better_somewhere):
                violations.append({"family": family, "loser": loser, "winner": winner})
    return {"violations": len(violations), "detail": violations}


def _families_differ(families: dict[str, Any]) -> dict[str, Any]:
    """The offset family is not the heading family rediscovered.

    If offsetting the start point produced the same routes as fanning the
    heading, one of the two columns of ``Phi`` would never be exercised and the
    whole reason for generating beyond a fan would be gone.

    Compared by *start point*, not by label. The labels are now prefixed by
    their family and so are disjoint by construction, which would make a
    comparison of them a test of the prefix rather than of the geometry.
    """
    # Compared exactly, not rounded. Rounding to a fixed number of decimals
    # would reintroduce the very thing the route labels were just fixed to
    # avoid -- two starts merging or separating on the last bit. A fan's
    # starts are the *same float*, so they compare equal exactly, and the
    # courses are spaced far enough apart that nothing is near a tie.
    starts = {
        name: {(row["start"]["u"], row["start"]["v"]) for row in payload["detail"]}
        for name, payload in families.items()
    }
    fan, courses = starts["heading-fan"], starts["offset-courses"]
    return {
        "compared_by": "start point, since the labels are disjoint by prefix",
        "shared_starts": len(fan & courses),
        "distinct_fan_starts": len(fan),
        "distinct_course_starts": len(courses),
    }


def _planning_objectives(config: SurfaceConfig) -> tuple[Objective, ...]:
    """The declared quantities, each with the limit that makes it dimensionless."""
    return (
        Objective(
            "amplification", "lower-is-better", limit=config.planning_amplification_limit,
            note="worst dimensionless gain of the tolerance box anywhere on the path",
        ),
        Objective(
            "cross_track_error", "lower-is-better",
            limit=config.route_max_cross_track_error,
        ),
        Objective(
            "heading_error", "lower-is-better", limit=config.route_max_heading_error,
        ),
        Objective(
            "worst_observed", "higher-is-better",
            limit=config.planning_observability_limit,
            note="smallest eigenvalue of the dimensionless accumulated Gramian",
        ),
        Objective(
            "boundary_clearance", "higher-is-better",
            limit=config.planning_clearance_limit,
        ),
        Objective(
            "path_length", "lower-is-better", limit=config.heading_scan_length,
        ),
    )


def measure_uncertainty_budget(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    """The whole budget on each path, and which term actually dominates.

    Every number here comes from the declared instrument and the declared
    tolerance box -- the same ones the route decision uses -- plus a declared
    registration and fixture. Nothing is fitted and nothing is a limit: the
    output is a breakdown, and what an acceptable total is belongs to a
    protocol.

    Two things it establishes that no single figure can. The **systematic
    fraction**: how much of the worst-sample variance is one unknown repeated,
    which is exactly the part that averaging more samples along the path does
    not touch. And **rank**: a budget of purely systematic terms is singular,
    because a perfectly correlated error is perfectly predictable, so a
    campaign that forgot to declare its sensor's noise finds out here rather
    than in a Cholesky failure three layers down.
    """
    lateral = config.route_tolerance_lateral
    heading = config.route_tolerance_heading
    rows: list[dict[str, Any]] = []
    for case in cases:
        envelope = integrate_path(
            case.surface, u0=case.u0, v0=case.v0, heading=case.heading,
            length=case.length, n_steps=config.budget_path_steps,
        )
        record = envelope.as_transfer_record(observation_mode="ambient-euclidean-chord")
        scanned = record.with_geometry_uncertainty(
            GeometryUncertainty(
                position=config.scan_position_sigma,
                normal=config.scan_normal_sigma,
                curvature=config.scan_curvature_sigma,
                basis="assumed",
                note="a declared example scan, not a measured one",
            )
        )
        assembled = budget(
            scanned,
            starting_pose(
                scanned,
                np.diag(
                    [
                        config.route_lateral_sigma**2,
                        float(np.deg2rad(config.route_heading_sigma_degrees)) ** 2,
                    ]
                ),
                basis="the declared instrument's starting-pose uncertainty",
            ),
            surface_reconstruction(scanned, lateral, heading),
            fixture_datum(
                scanned,
                lateral_sigma=config.fixture_lateral_sigma,
                heading_sigma=config.fixture_heading_sigma,
                basis="declared",
            ),
            calibration_transform(
                scanned, sigma=config.calibration_offset_sigma, basis="declared"
            ),
            path_registration(
                scanned, lateral, heading,
                sigma=config.registration_sigma, basis="declared",
            ),
            sensor_noise(
                scanned,
                sigma=config.route_measurement_sigma,
                correlation_length=config.sensor_correlation_length,
                basis="declared",
            ),
            note=f"{case.key}: declared example, no measured input",
        )
        systematic_only = budget(
            scanned,
            *[c for c in assembled.contributions if c.structure == "systematic"],
        )
        payload = assembled.to_dict()
        payload |= {
            "case": case.key,
            "surface": case.surface.name,
            "systematic_only_is_singular": not systematic_only.is_positive_definite(),
            "starting_pose_share_at_worst": payload["shares_at_worst"]["starting pose"],
        }
        rows.append(payload)
    return rows


def measure_prediction_chain(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    """Does naming the transformations actually close the gap to a measurement?

    The finite-difference route is an *ambient chord*: it flows two geodesics
    at plus and minus ``eps`` and measures the straight-line distance between
    reconstructed 3-D points. The transfer map is a *first-order tangent
    vector*. Between them sit two second-order corrections, and this sweep asks
    whether applying them -- as named transformations on a prediction object,
    rather than as a relabelling -- reproduces the measurement.

    Where the curvature is constant, both corrections are available and the
    full chain closes: the disagreement falls from the size of the effect to
    the numerical floor. Where it varies, only the chord correction is
    computable, ``chord_from_tangent`` applies it and declares that the other
    is missing, and the residual that remains is exactly the term it declared.

    The comparison is scaled by the largest separation on the path rather than
    taken pointwise, because the spherical cap crosses its conjugate point: the
    separation passes through zero there, and a relative error against zero is
    not a measure of anything.
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        envelope = integrate_path(
            case.surface, u0=case.u0, v0=case.v0, heading=case.heading,
            length=case.length, n_steps=config.chain_steps,
        )
        record = envelope.as_transfer_record(observation_mode="ambient-euclidean-chord")
        closed_form = has_closed_form_separation(record)
        levels = []
        for epsilon in config.chain_epsilons:
            _, measured = finite_difference_jacobi(
                case.surface, u0=case.u0, v0=case.v0, heading=case.heading,
                epsilon=float(epsilon), length=case.length, n_steps=config.chain_steps,
            )
            # The finite-difference pair is separated by 2 eps, and it reports a
            # magnitude: it has no sign to lose, which the transfer map does.
            chord = measured * 2.0 * float(epsilon)
            tangent = first_order_prediction(record, 0.0, 2.0 * float(epsilon))
            chain = (
                chord_from_intrinsic(intrinsic_from_tangent(tangent, record), record)
                if closed_form
                else chord_from_tangent(tangent, record)
            )
            scale = max(float(np.max(np.abs(tangent.values))), 1e-300)
            levels.append(
                {
                    "epsilon": float(epsilon),
                    "first_order_disagreement": float(
                        np.max(np.abs(chord - np.abs(tangent.values)))
                    )
                    / scale,
                    "chain_disagreement": float(
                        np.max(np.abs(chord - np.abs(chain.values)))
                    )
                    / scale,
                    "stages": list(chain.chain),
                    "intrinsic_correction": chain.extra["intrinsic_correction"],
                }
            )
        worst_chain = max(level["chain_disagreement"] for level in levels)
        best_ratio = min(
            level["first_order_disagreement"] / max(level["chain_disagreement"], 1e-300)
            for level in levels
        )
        rows.append(
            {
                "case": case.key,
                "surface": case.surface.name,
                "curvature_is_constant": bool(closed_form),
                "chain": list(levels[0]["stages"]),
                "intrinsic_correction": levels[0]["intrinsic_correction"],
                "worst_chain_disagreement": worst_chain,
                "smallest_improvement_factor": best_ratio,
                "chain_is_never_worse": all(
                    level["chain_disagreement"] <= level["first_order_disagreement"]
                    for level in levels
                ),
                "levels": levels,
            }
        )
    return rows


def measure_error_budget(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    """A path-level error budget: what the step costs, per quantity, per case.

    Two things are measured, and the second is what makes the first evidence.

    **The budget.** ``estimate_convergence`` halves the step and Richardson
    -extrapolates at the method's order, giving an absolute error estimate for
    position, the transfer map, the curvature, the focus location and the
    propagated covariance -- separately, because they do not converge together.

    **Whether the budget is true.** Three of these cases have a closed form for
    ``b(s)``, so the actual error is available and the estimate can be checked
    against it rather than trusted. An error estimate that understates the
    error is worse than no estimate, because it is acted on.

    The comparison is only meaningful where there is a truncation error to
    resolve: on the plate and the rolled sheet, ``b(s) = s`` is reproduced to
    roundoff at every step size, so both the estimate and the truth are
    1e-14-sized and their ratio is a ratio of noise. Those cases are reported
    with ``resolved: false`` and carry no check, which is the same discipline
    the convergence sweeps use for an exact method.
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        entries = []
        for n_steps in config.budget_step_counts:
            envelope = integrate_path(
                case.surface,
                u0=case.u0,
                v0=case.v0,
                heading=case.heading,
                length=case.length,
                n_steps=n_steps,
            )
            estimate = estimate_convergence(envelope)
            exact = case.closed_form(envelope.arc_length)
            truth = (
                None
                if exact is None
                else float(np.max(np.abs(envelope.jacobi_field - exact)))
            )
            # Below this the "true error" is the accumulated rounding of the
            # reference itself, so the ratio compares two pieces of noise.
            resolved = truth is not None and truth > config.budget_roundoff_floor
            entries.append(
                {
                    "n_steps": int(n_steps),
                    "step": float(case.length) / n_steps,
                    "estimate": estimate.to_dict(),
                    "true_transfer_error": truth,
                    "resolved": bool(resolved),
                    "estimate_over_truth": (
                        float(estimate.transfer / truth) if resolved else None
                    ),
                }
            )
        coarse, fine = entries[0], entries[-1]
        refinement = fine["n_steps"] / coarse["n_steps"]
        observed_order = None
        if coarse["estimate"]["transfer"] and fine["estimate"]["transfer"]:
            ratio = coarse["estimate"]["transfer"] / fine["estimate"]["transfer"]
            if ratio > 1.0:
                observed_order = float(np.log(ratio) / np.log(refinement))
        rows.append(
            {
                "case": case.key,
                "surface": case.surface.name,
                "reference": case.reference,
                "method": "rk4",
                "declared_order": 4,
                "refinement_between_levels": float(refinement),
                "observed_order": observed_order,
                "any_resolved": any(entry["resolved"] for entry in entries),
                "worst_estimate_over_truth": max(
                    (
                        abs(entry["estimate_over_truth"] - 1.0)
                        for entry in entries
                        if entry["estimate_over_truth"] is not None
                    ),
                    default=None,
                ),
                "levels": entries,
            }
        )
    return rows


def measure_jet_step_sensitivity(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    """What the finite-difference jet's step size costs *along a whole path*.

    ``derivative_convergence`` asks whether the jet at one point converges as
    the differencing step shrinks. That is necessary and it is not the budget:
    a jet error enters the geodesic equation at every step, and what a consumer
    needs to know is what it does to the path, to ``Phi``, to the curvature and
    to the focus after the whole integration.

    So the same path is flowed with the analytic jet and with the fallback at a
    range of relative steps, and the differences are reported per quantity. The
    curve has the shape every finite difference has -- truncation falling as
    the step shrinks, cancellation rising as it shrinks further -- and the
    interesting number is the best any step achieves, because that is the floor
    on a surface that has no analytic jet at all.
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        if _uses_finite_differences(case.surface):
            continue
        analytic = integrate_path(
            case.surface, u0=case.u0, v0=case.v0, heading=case.heading,
            length=case.length, n_steps=config.jet_sensitivity_steps,
        )
        focus = analytic.focus_points()
        levels = []
        for relative_step in config.jet_relative_steps:
            numeric = integrate_path(
                replace(case.surface, jet=None, fd_relative_step=float(relative_step)),
                u0=case.u0, v0=case.v0, heading=case.heading,
                length=case.length, n_steps=config.jet_sensitivity_steps,
            )
            numeric_focus = numeric.focus_points()
            levels.append(
                {
                    "fd_relative_step": float(relative_step),
                    "position": float(
                        np.max(np.linalg.norm(analytic.points - numeric.points, axis=-1))
                    ),
                    "transfer": float(
                        max(
                            np.max(np.abs(analytic.lateral_basis - numeric.lateral_basis)),
                            np.max(np.abs(analytic.lateral_rate - numeric.lateral_rate)),
                            np.max(np.abs(analytic.jacobi_field - numeric.jacobi_field)),
                            np.max(
                                np.abs(
                                    analytic.jacobi_derivative - numeric.jacobi_derivative
                                )
                            ),
                        )
                    ),
                    "curvature": float(
                        np.max(np.abs(analytic.curvature - numeric.curvature))
                    ),
                    "focus": (
                        abs(focus[0] - numeric_focus[0])
                        if focus and numeric_focus
                        else None
                    ),
                    "found_the_same_number_of_foci": len(focus) == len(numeric_focus),
                }
            )
        best = min(levels, key=lambda level: level["transfer"])
        worst = max(levels, key=lambda level: level["transfer"])
        default = min(
            levels,
            key=lambda level: abs(
                level["fd_relative_step"] - float(case.surface.fd_relative_step)
            ),
        )
        rows.append(
            {
                "case": case.key,
                "surface": case.surface.name,
                "n_steps": int(config.jet_sensitivity_steps),
                "default_relative_step": float(case.surface.fd_relative_step),
                "best_relative_step": best["fd_relative_step"],
                "best_transfer_error": best["transfer"],
                "worst_transfer_error": worst["transfer"],
                "default_transfer_error": default["transfer"],
                #: How much the declared default gives away against the best
                #: step for this surface. A default cannot be optimal for every
                #: surface; what it must not be is arbitrary, and this is the
                #: number that says which.
                "default_over_best": (
                    float(default["transfer"] / best["transfer"])
                    if best["transfer"] > 0.0
                    else None
                ),
                "default_over_worst": (
                    float(default["transfer"] / worst["transfer"])
                    if worst["transfer"] > 0.0
                    else None
                ),
                "spread_factor": (
                    float(worst["transfer"] / best["transfer"])
                    if best["transfer"] > 0.0
                    else None
                ),
                "every_step_found_the_same_foci": all(
                    level["found_the_same_number_of_foci"] for level in levels
                ),
                "levels": levels,
            }
        )
    return rows


def measure_finite_difference_cost(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if _uses_finite_differences(case.surface):
            continue
        analytic = integrate_path(
            case.surface,
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            length=case.length,
            n_steps=config.finite_difference_steps,
        )
        numeric = integrate_path(
            replace(case.surface, jet=None),
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            length=case.length,
            n_steps=config.finite_difference_steps,
        )
        scale = max(float(np.max(np.abs(analytic.jacobi_field))), 1e-300)
        rows.append(
            {
                "case": case.key,
                "surface": case.surface.name,
                "fd_relative_step": float(case.surface.fd_relative_step),
                "fd_step_at_start": float(case.surface.fd_step_at(case.u0, case.v0)),
                "fd_convergence": replace(case.surface, jet=None).derivative_convergence(
                    case.u0, case.v0
                ),
                "max_jacobi_difference": float(
                    np.max(np.abs(analytic.jacobi_field - numeric.jacobi_field))
                ),
                "max_relative_jacobi_difference": float(
                    np.max(np.abs(analytic.jacobi_field - numeric.jacobi_field)) / scale
                ),
                "max_curvature_difference": float(
                    np.max(np.abs(analytic.curvature - numeric.curvature))
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 5. the instrument readout
# ---------------------------------------------------------------------------
def build_envelopes(config: SurfaceConfig, cases) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        envelope = integrate_path(
            case.surface,
            u0=case.u0,
            v0=case.v0,
            heading=case.heading,
            length=case.length,
            n_steps=config.n_steps,
        )
        summary = envelope.summary()
        summary["max_wronskian_drift"] = float(
            np.max(envelope.transfer_map.wronskian_drift)
        )
        summary["max_lateral_amplification"] = float(np.max(np.abs(envelope.lateral_basis)))
        summary["lateral_focus_points"] = envelope.transfer_map.focus_points(component="a")
        # The record carries its own error budget, because a consumer deciding
        # whether a focus at s = 3.1416 is located well enough to plan against
        # needs the error and not the step size it came from.
        record = envelope.as_transfer_record(
            convergence=estimate_convergence(envelope)
        )
        singular = record.scaled_singular_values(
            config.route_tolerance_lateral, config.route_tolerance_heading
        )
        # The invariant, formed as a 2x2 determinant. Reading it off the
        # product of singular values instead would measure the SVD's accuracy
        # on the tolerance box's aspect ratio rather than the integrator's on
        # the surface, and the two differ by two orders of magnitude.
        summary["scaled_determinant_error"] = float(
            np.max(
                np.abs(
                    record.scaled_determinant(
                        config.route_tolerance_lateral, config.route_tolerance_heading
                    )
                    - 1.0
                )
            )
        )
        summary["scaled_singular_value_product_error"] = float(
            np.max(np.abs(singular[:, 0] * singular[:, 1] - 1.0))
        )

        # The path geometry the record now carries, checked rather than added.
        # Euler's theorem: the normal curvatures in any two orthogonal tangent
        # directions sum to twice the mean curvature. H comes from the second
        # fundamental form directly and the two kappa_n from the Darboux frame
        # along the flowed path, so the identity ties the frame the record
        # publishes to the surface it claims to be on -- on every surface, with
        # no closed form required.
        geometry = envelope.path_geometry()
        mean = np.asarray(case.surface.mean_curvature(envelope.u, envelope.v), dtype=float)
        summary["geometry"] = {
            "euler_identity": "kappa_n(along) + kappa_n(across) = 2H",
            "euler_residual": float(
                np.max(
                    np.abs(
                        geometry.normal_curvature_along
                        + geometry.normal_curvature_transverse
                        - 2.0 * mean
                    )
                )
            ),
            "frame_orientation_residual": geometry.orientation_residual(),
            "max_tangent_norm_error": float(
                np.max(np.abs(np.linalg.norm(geometry.tangent, axis=-1) - 1.0))
            ),
            "max_frame_inner_product": float(
                np.max(
                    np.abs(
                        np.einsum("ij,ij->i", geometry.tangent, geometry.surface_normal)
                    )
                )
            ),
            "normal_curvature_along": {
                "min": float(np.min(geometry.normal_curvature_along)),
                "max": float(np.max(geometry.normal_curvature_along)),
            },
            "normal_curvature_transverse": {
                "min": float(np.min(geometry.normal_curvature_transverse)),
                "max": float(np.max(geometry.normal_curvature_transverse)),
            },
        }
        summary["min_scaled_singular_value_max"] = float(np.min(singular[:, 0]))
        summary["amplification_score"] = float(np.max(singular[:, 0]))
        summary["record"] = record.to_dict(include_samples=False) | {
            "round_trip_error": float(
                max(
                    np.max(np.abs(record.a - envelope.lateral_basis)),
                    np.max(np.abs(record.b - envelope.jacobi_field)),
                    np.max(np.abs(record.arclength - envelope.arc_length)),
                )
            )
        }
        probes = np.linspace(0.0, case.length, 5)[1:]
        stride_index = [int(round(p / (case.length / config.n_steps))) for p in probes]
        summary["budget"] = [
            {
                "transverse_tolerance": float(tolerance),
                "at": [
                    {
                        "arc_length": float(envelope.arc_length[index]),
                        "max_heading_error": float(
                            envelope.heading_budget(tolerance)[index]
                        ),
                        "max_heading_error_degrees": float(
                            np.rad2deg(envelope.heading_budget(tolerance)[index])
                        ),
                        "envelope_half_width_at_1_degree": float(
                            envelope.transverse_envelope(np.deg2rad(1.0))[index]
                        ),
                    }
                    for index in stride_index
                ],
            }
            for tolerance in config.transverse_tolerances
        ]
        stride = max(1, config.n_steps // config.curve_samples)
        summary["curve"] = [
            {
                "arc_length": float(s),
                "gaussian_curvature": float(k),
                "jacobi_field": float(j),
            }
            for s, k, j in zip(
                envelope.arc_length[::stride],
                envelope.curvature[::stride],
                envelope.jacobi_field[::stride],
                strict=True,
            )
        ]
        rows.append(summary | {"case": case.key})
    return rows


def _acquisition(config: SurfaceConfig) -> AcquisitionSpec:
    return AcquisitionSpec(
        acquire_threshold=config.route_acquire_threshold,
        hold_threshold=config.route_hold_threshold,
        acquisition_window=config.route_acquisition_window,
        max_acquisition_distance=config.route_max_acquisition_distance,
        min_tracked_distance=config.route_min_tracked_distance,
        max_loss_distance=config.route_max_loss_distance,
        processing="offline",
        note="a declared example schedule; a real one comes from the sensor",
    )


def _route_constraints(config: SurfaceConfig, *, acquisition: AcquisitionSpec | None = None):
    return RouteConstraints(
        max_cross_track_error=config.route_max_cross_track_error,
        max_heading_error=config.route_max_heading_error,
        min_coverage_margin=config.route_min_coverage_margin,
        acquisition=acquisition,
        coverage=CoverageSpec(
            nominal_spacing=config.route_nominal_spacing,
            swath_width=config.route_swath_width,
        ),
    )


def _instrument(config: SurfaceConfig) -> tuple[ObservationModel, np.ndarray]:
    """The scanner and the starting-pose uncertainty the route is judged against."""
    model = ObservationModel.transverse_only(
        config.route_measurement_sigma,
        mode="ambient-euclidean-chord",
        calibration_id="declared-example",
        reconstruction_version="none",
        note="a 25 um metrology system on a 300 mm coupon, expressed in units of R",
    )
    covariance = np.diag(
        [
            config.route_lateral_sigma**2,
            float(np.deg2rad(config.route_heading_sigma_degrees)) ** 2,
        ]
    )
    return model, covariance


def _resolvability_scale_invariance(config: SurfaceConfig) -> dict[str, Any]:
    """The property a dimensionful focus threshold cannot have.

    Take the same geometry at two sizes: a unit sphere with a path of length L
    and a scanner of standard deviation ``sigma``, against a sphere of radius 2
    with a path of length 2L and a scanner of ``2 sigma`` -- the identical
    physical situation drawn at twice the size. ``rho`` must be the same
    function of the fractional distance along the path. A threshold on ``|b|``,
    which has units of length per radian, would double.
    """
    length = 3.0
    heading = float(np.deg2rad(config.route_heading_sigma_degrees))
    profiles = []
    for radius in (1.0, 2.0):
        model = ObservationModel.transverse_only(
            config.route_measurement_sigma * radius, mode="ambient-euclidean-chord"
        )
        envelope = integrate_path(
            sphere(radius),
            u0=float(np.pi / 2),
            v0=0.0,
            heading=float(np.pi / 2),
            length=length * radius,
            n_steps=600,
        )
        profiles.append(model.resolvability(envelope, np.diag([0.0, heading**2]))[:, 0])
    difference = float(np.max(np.abs(profiles[0] - profiles[1])))
    scale = float(np.max(np.abs(profiles[0]))) or 1.0
    return {
        "radii": [1.0, 2.0],
        "max_absolute_difference": difference,
        "max_relative_difference": difference / scale,
        "note": "same physical situation at two sizes; rho must not move",
    }


def measure_chart_rescaling_invariance(cases) -> dict[str, Any]:
    """The defect that retired ``sqrt(EG - F^2) / max(E, G)``.

    Reparameterise the torus by ``v -> v/3``. Nothing about the surface or the
    regularity of the chart changes -- only how fast the second coordinate runs.
    The retired criterion falls by exactly that factor, because with
    ``r_v -> 3 r_v`` the numerator picks up a 3 and the denominator a 9; it
    therefore reports an anisotropic chart as a nearly degenerate one.
    ``chart_orthogonality``, built on the direction-normalised Jacobian,
    depends only on the angle between the coordinate directions and does not
    move.
    """
    base = next(case.surface for case in cases if case.key == "torus")
    factor = 3.0
    stretched = ParametricSurface(
        name=f"{base.name} reparameterised by v -> v/{factor:.0f}",
        position=lambda u, v: base.position(u, factor * v),
        chart=Chart(u_min=-10.0, u_max=10.0, v_min=-10.0, v_max=10.0),
    )
    u, v = np.meshgrid(
        np.linspace(0.1, 1.2, 24), np.linspace(0.1, 1.2, 24), indexing="ij"
    )
    u, v = u.ravel(), v.ravel()

    def retired(surface: ParametricSurface) -> float:
        E, F, G = surface.first_fundamental_form(u, v)
        return float(np.min(np.sqrt(E * G - F**2) / np.maximum(E, G)))

    before, after = retired(base), retired(stretched)
    orthogonality = (
        float(np.min(base.chart_orthogonality(u, v))),
        float(np.min(stretched.chart_orthogonality(u, v))),
    )
    return {
        "surface": base.name,
        "reparameterisation": f"v -> v/{factor:.0f}",
        "retired_criterion": "sqrt(EG - F^2) / max(E, G)",
        "retired_before": before,
        "retired_after": after,
        "retired_ratio": before / after,
        "expected_ratio": factor,
        "orthogonality_before": orthogonality[0],
        "orthogonality_after": orthogonality[1],
        "orthogonality_absolute_difference": abs(orthogonality[0] - orthogonality[1]),
        "note": (
            "the chart is exactly as regular after the rescaling as before, so a "
            "conditioning number that moves is measuring the parameterisation "
            "rather than the chart"
        ),
    }


def measure_focus_versus_resolvability(config: SurfaceConfig, cases) -> dict[str, Any]:
    """Two statements that correlate here, and are not the same statement.

    A *geometric focus* is a zero of a transfer column: a property of the
    surface and the path, present whatever instrument is pointed at it. *Low
    resolvability* is a property of the whole chain, and moves when the
    tolerance or the scanner moves. Collapsing one into the other would let a
    route be called "clear of a focus" because the scanner happened to be good,
    or a genuine conjugate point be reported wherever a sensor was noisy.

    Two measurements separate them:

    * a path with **no focus anywhere** driven below threshold by shrinking the
      admitted starting uncertainty -- low ``rho``, no focus;
    * a point a declared distance from a **real** conjugate point, resolved by
      sharpening the scanner -- a focus nearby, and ``rho`` above threshold
      anyway.
    """
    heading_sigma = float(np.deg2rad(config.route_heading_sigma_degrees))

    # 1. The plate: b(s) = s, so there is no focus at any arc length. Shrink
    #    the admitted heading error and the scanner stops being able to see the
    #    difference between the poses the tolerance allows -- with no focus
    #    anywhere near it.
    plate = next(case for case in cases if case.key == "plate")
    flat = integrate_path(
        plate.surface, u0=plate.u0, v0=plate.v0, heading=plate.heading,
        length=plate.length, n_steps=config.n_steps,
    )
    instrument, _ = _instrument(config)
    tiny = heading_sigma / 1.0e4
    rho_flat = instrument.resolvability(flat, np.diag([0.0, tiny**2]))[:, 0]
    flat_focus_points = [float(event.arc_length) for event in flat.transfer_map.focus_events()]

    # 2. The spherical cap: b(s) = sin(s), a genuine conjugate point at pi.
    #    At a declared offset from it the signal is small but not zero, so a
    #    sharp enough scanner still separates the admitted poses.
    offsets = [0.1, 0.05, 0.01]
    cap = integrate_path(
        sphere(1.0), u0=float(np.pi / 2), v0=0.0, heading=float(np.pi / 2),
        length=float(np.pi) + 0.5, n_steps=config.n_steps,
    )
    transfer = cap.transfer_map
    focus = [float(event.arc_length) for event in transfer.focus_events()]
    near_focus = []
    for offset in offsets:
        target = focus[0] - offset if focus else float(np.pi) - offset
        index = int(np.argmin(np.abs(cap.arc_length - target)))
        signal = abs(float(transfer.b[index])) * heading_sigma
        # The metrology sigma at which rho reaches the acquire threshold here.
        required = signal / float(config.route_acquire_threshold)
        near_focus.append(
            {
                "offset_from_focus": float(offset),
                "arclength": float(cap.arc_length[index]),
                "abs_b": abs(float(transfer.b[index])),
                "rho_with_declared_scanner": signal / float(config.route_measurement_sigma),
                "metrology_sigma_to_reach_acquire_threshold": float(required),
                "metrology_sigma_micrometres_on_a_300mm_coupon": float(
                    required * 300_000.0
                ),
                "resolvable_with_declared_scanner": bool(
                    signal / float(config.route_measurement_sigma)
                    >= float(config.route_acquire_threshold)
                ),
            }
        )

    return {
        "no_focus_but_unresolvable": {
            "case": "plate",
            "surface": plate.surface.name,
            "heading_sigma": tiny,
            "heading_sigma_ratio_to_declared": tiny / heading_sigma,
            "n_focus_points": len(flat_focus_points),
            "max_resolvability": float(np.max(rho_flat)),
            "acquire_threshold": float(config.route_acquire_threshold),
            "note": (
                "b(s) = s has no zero on this path, so there is no focus at any "
                "arc length; rho is below the acquire threshold everywhere "
                "regardless, because the tolerance admits less than the scanner "
                "can see"
            ),
        },
        "focus_nearby_but_resolvable": {
            "case": "spherical-cap",
            "surface": "sphere(R=1)",
            "focus_points": focus,
            "declared_metrology_sigma": float(config.route_measurement_sigma),
            "samples": near_focus,
            "note": (
                "|b| is small near a conjugate point but not zero, so the "
                "metrology sigma that resolves the admitted poses there is "
                "finite; a focus does not by itself make a path unobservable"
            ),
        },
        "note": (
            "a focus is a fact about the surface and the path; resolvability is "
            "a fact about the surface, the tolerance and the instrument together. "
            "Both are reported, and the route constraint uses the second"
        ),
    }


def heading_label(degrees: float) -> str:
    """A stable name for one candidate heading.

    One decimal place, not zero. The scan's headings are multiples of 7.5
    degrees, so half of them land exactly on a rounding boundary -- and
    ``f"{97.5:.0f}"`` is ``'98'`` while ``f"{97.49999999999999:.0f}"`` is
    ``'97'``. Which of the two a ``2 pi k / n`` division produces is a property
    of the platform's last bit, so with a zero-decimal label the *name of the
    recommended route* moved between machines, and the figure that looks a
    route up by name could not find it. A decimal place puts the label off the
    boundary entirely.
    """
    return f"{float(degrees):.1f}deg"


def scan_for_robust_heading(config: SurfaceConfig, cases) -> dict[str, Any]:
    """Rank starting headings, then choose among them by declared process limits.

    The ranking scalar is forward angular-error amplification, and it is
    reported as such. It is not the decision: a heading scores well by passing
    through a focus, where the endpoint map has collapsed, so the recommendation
    comes from :func:`~geodesic_testbed.engine.routing.rank_routes` against
    limits the process states -- cross-track error, downstream heading error,
    coverage margin -- rather than against a focus margin chosen here.

    To show that the recommendation really does follow the declared constraint,
    the same candidates are ranked a second time with a focus-clearance limit
    added, and both answers are recorded.
    """
    case = next(item for item in cases if item.key == config.heading_scan_key)
    headings = np.linspace(0.0, np.pi, config.heading_scan_count, endpoint=False)
    rows = scan_headings(
        case.surface,
        u0=case.u0,
        v0=case.v0,
        headings=headings,
        length=config.heading_scan_length,
        n_steps=config.heading_scan_steps,
    )
    envelopes = integrate_paths(
        case.surface,
        u0=case.u0,
        v0=case.v0,
        headings=headings,
        length=config.heading_scan_length,
        n_steps=config.heading_scan_steps,
    )
    candidates = {
        heading_label(np.rad2deg(heading)): envelope
        for heading, envelope in zip(headings, envelopes, strict=True)
    }

    observation, initial_covariance = _instrument(config)

    def ranked(acquisition: AcquisitionSpec | None) -> list[Any]:
        return rank_routes(
            candidates,
            max_lateral=config.route_tolerance_lateral,
            max_heading=config.route_tolerance_heading,
            constraints=_route_constraints(config, acquisition=acquisition),
            observation=observation,
            initial_covariance=initial_covariance,
        )

    declared = ranked(None)
    stricter = ranked(_acquisition(config))
    lowest, highest = rows[0], rows[-1]
    return {
        "case": case.key,
        "surface": case.surface.name,
        "ranking_scalar": "minimum-forward-angular-error-amplification",
        "ranking_note": (
            "max |b(s)| is how far an aiming error is carried, not robustness. "
            "It is reported because it is the cleanest single number about the "
            "heading column, and it does not decide anything: the recommendation "
            "below comes from declared process limits."
        ),
        "start": {"u": case.u0, "v": case.v0},
        "length": float(config.heading_scan_length),
        "n_headings": int(config.heading_scan_count),
        "focus_margin_report_level": float(config.focus_margin_report_level),
        "focus_margin_note": (
            "|b| has units of length per radian, so this level is specific to "
            "this part's size and to radians. It is reported, never used as a "
            "criterion; the criterion is resolvability below."
        ),
        "lowest_amplification": lowest,
        "highest_amplification": highest,
        "amplification_ratio": float(
            highest["max_forward_amplification"] / lowest["max_forward_amplification"]
        ),
        "n_headings_passing_a_focus": sum(1 for row in rows if row["passes_a_focus"]),
        "route_selection": {
            "tolerance": {
                "lateral": float(config.route_tolerance_lateral),
                "heading": float(config.route_tolerance_heading),
            },
            "constraints": _route_constraints(config).to_dict(),
            "n_feasible": sum(1 for entry in declared if entry.feasible),
            "recommended": declared[0].to_dict() if declared[0].feasible else None,
            "binding_constraints": sorted(
                {entry.binding_constraint for entry in declared if entry.binding_constraint}
            ),
            "assessments": [entry.to_dict() for entry in declared],
        },
        "route_selection_with_tracking": {
            "note": (
                "The same candidates, with the constraint that the scanner "
                "acquires the path and does not lose it. rho = |b| sigma_alpha / "
                "sigma_measurement is dimensionless, so unlike a threshold on |b| "
                "it does not move when the part is rescaled or the angle unit "
                "changes; and a schedule rather than a minimum, because rho(0) = 0 "
                "on every route and a route acquired at its last sample was never "
                "tracked."
            ),
            "observation_model": observation.to_dict(),
            "initial_covariance": initial_covariance.tolist(),
            "acquisition": _acquisition(config).to_dict(),
            "n_feasible": sum(1 for entry in stricter if entry.feasible),
            "recommended": stricter[0].to_dict() if stricter[0].feasible else None,
            "outcomes": sorted(
                {
                    entry.tracking["outcome"]
                    for entry in stricter
                    if entry.tracking is not None
                }
            ),
            "scale_invariance": _resolvability_scale_invariance(config),
            "assessments": [entry.to_dict() for entry in stricter],
        },
        "headings": rows,
    }


# ---------------------------------------------------------------------------
# checks and report
# ---------------------------------------------------------------------------
def collect_checks(results: dict[str, Any], config: SurfaceConfig) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    for row in results["anchored_to_closed_forms"]:
        checks.append(
            _check(
                f"surface-anchor/{row['case']}",
                f"the general solver recovers j(s) = {row['reference']} on {row['surface']}",
                row["max_jacobi_error"],
                row["tolerance"],
            )
        )
        checks.append(
            _check(
                f"surface-speed/{row['case']}",
                "unit speed holds along the path without ever being re-imposed",
                row["max_speed_drift"],
                config.speed_drift_tolerance,
            )
        )
        if row["max_curvature_error"] is not None:
            checks.append(
                _check(
                    f"surface-curvature/{row['case']}",
                    "Gaussian curvature assembled from the jet matches the analytic value",
                    row["max_curvature_error"],
                    (
                        config.curvature_tolerance_fd
                        if row["derivatives"] == "finite-difference"
                        else config.curvature_tolerance
                    ),
                )
            )

    for row in results["anchored_to_closed_forms"]:
        checks.append(
            _check(
                f"surface-anchor-lateral/{row['case']}",
                f"the lateral column is recovered as a(s) = {row['lateral_reference']} "
                f"on {row['surface']}",
                row["max_lateral_error"],
                row["tolerance"],
            )
        )

    for row in results["envelopes"]:
        checks.append(
            _check(
                f"surface-wronskian/{row['case']}",
                "det Phi = a b' - a' b stays 1 along the path, an invariant the "
                "solver never enforces",
                row["max_wronskian_drift"],
                config.wronskian_tolerance,
            )
        )

    for row in results["lateral_route"]:
        if row["regime"] == "exact-to-roundoff":
            checks.append(
                _check(
                    f"surface-lateral-route/{row['case']}",
                    "two parallel geodesics on a flat surface stay exactly as far "
                    "apart as they started, so a(s) is recovered exactly",
                    row["max_deviation"],
                    config.exactness_threshold,
                )
            )
            continue
        checks.append(
            _check(
                f"surface-lateral-route/{row['case']}",
                "moving the start point sideways and parallel-transporting the "
                "direction recovers a(s), to second order in the offset",
                abs(row["fitted_exponent"] - row["expected_exponent"]),
                config.exponent_tolerance,
            )
        )

    for row in results["two_routes"]:
        checks.append(
            _check(
                f"surface-two-routes/{row['case']}",
                "differencing the flow and integrating the Jacobi equation agree, "
                "with a difference of order eps^2",
                abs(row["fitted_exponent"] - row["expected_exponent"]),
                config.exponent_tolerance,
            )
        )
        if row["chord_closed_form_max_relative_error"] is not None:
            checks.append(
                _check(
                    f"surface-two-routes-chord/{row['case']}",
                    "where the chord between the two perturbed geodesics is known "
                    "exactly, the measured variation matches it at every eps",
                    row["chord_closed_form_max_relative_error"],
                    config.chord_closed_form_tolerance,
                )
            )
            checks.append(
                _check(
                    f"surface-two-routes-coefficient/{row['case']}",
                    "and its eps^2 coefficient is the purely intrinsic 1/6",
                    abs(row["fitted_coefficient"] * 6.0 - 1.0),
                    config.coefficient_tolerance,
                )
            )
        else:
            checks.append(
                _check(
                    f"surface-chord-excess/{row['case']}",
                    "measuring a chord in space instead of a distance in the "
                    "surface adds to the eps^2 term, it never subtracts",
                    row["chord_excess_coefficient"],
                    0.0,
                    comparison=">=",
                )
            )

    for row in results["self_convergence"]:
        if row["regime"] == "exact-to-roundoff":
            checks.append(
                _check(
                    f"surface-self-convergence/{row['case']}",
                    "the flow is integrated exactly at every step size, so there "
                    "is no order to measure",
                    row["max_error_over_levels"],
                    config.exactness_threshold,
                )
            )
            continue
        checks.append(
            _check(
                f"surface-self-convergence/{row['case']}",
                "rk4 on the coupled geodesic-Jacobi system converges at order 4 "
                "against a finer run of itself",
                abs(row["fitted_order_jacobi"] - row["expected_order"]),
                config.order_tolerance,
            )
        )
        checks.append(
            _check(
                f"surface-self-convergence-position/{row['case']}",
                "so does the path itself",
                abs(row["fitted_order_position"] - row["expected_order"]),
                config.order_tolerance,
            )
        )

    for row in results["finite_difference_cost"]:
        checks.append(
            _check(
                f"surface-fd-cost/{row['case']}",
                "a surface given without analytic derivatives still tracks the "
                "analytic one closely enough to be useful",
                row["max_relative_jacobi_difference"],
                config.finite_difference_tolerance,
            )
        )

    flat = {row["case"]: row for row in results["envelopes"]}
    checks.append(
        _check(
            "surface-rolled-sheet-is-flat",
            "rolling a sheet into a cylinder does not change path sensitivity: "
            "the amplification stays 1",
            abs(flat["rolled-sheet"]["amplification_at_end"] - 1.0),
            1e-12,
        )
    )
    checks.append(
        _check(
            "surface-focus-found",
            "the spherical cap refocuses at s = pi and is flagged ill conditioned",
            abs(flat["spherical-cap"]["focus_points"][0] - float(np.pi))
            if flat["spherical-cap"]["focus_points"]
            else None,
            1e-8,
        )
    )
    checks.append(
        _check(
            "surface-no-spurious-focus",
            "no focus is reported on the negatively curved cases, where none exists",
            float(
                len(flat["saddle"]["focus_points"]) + len(flat["pseudosphere"]["focus_points"])
            ),
            0.0,
        )
    )

    for row in results["error_budget"]:
        if row["observed_order"] is not None:
            checks.append(
                _check(
                    f"budget-order/{row['case']}",
                    "the step-doubling error budget itself falls as h^4, so it is "
                    "measuring the truncation error of a fourth-order method rather "
                    "than something else",
                    abs(row["observed_order"] - row["declared_order"]),
                    0.15,
                )
            )
        if row["worst_estimate_over_truth"] is not None:
            checks.append(
                _check(
                    f"budget-estimate-matches-the-truth/{row['case']}",
                    "where a closed form leaves a truncation error to resolve, the "
                    "estimate reproduces the actual error -- an error estimate that "
                    "understates the error is worse than none, because it is acted on",
                    row["worst_estimate_over_truth"],
                    0.02,
                )
            )

    planning = results["route_planning"]
    checks.append(
        _check(
            "gramian-density-is-scale-invariant",
            "the accumulated observability per unit path length is unchanged when "
            "the same physical situation is drawn at twice the size -- the "
            "property a route criterion must have, and a dimensionful score "
            "cannot",
            planning["scale_invariance"]["density_relative_difference"],
            1e-9,
        )
    )
    checks.append(
        _check(
            "gramian-accumulates-with-path-length",
            "while the accumulated figure doubles, because twice the path really "
            "does carry twice the information",
            abs(
                planning["scale_invariance"]["accumulated_ratio"]
                - planning["scale_invariance"]["expected_accumulated_ratio"]
            ),
            1e-9,
        )
    )
    checks.append(
        _check(
            "gramian-is-monotone",
            "W(s2) - W(s1) is positive semi-definite, so the Gramian measures "
            "information rather than merely growing",
            -planning["monotone"]["most_negative_increment_eigenvalue"],
            1e-12,
        )
    )
    checks.append(
        _check(
            "pareto-front-is-sound",
            "every route the front calls dominated is beaten on every declared "
            "objective at once, and strictly on at least one",
            float(planning["front_is_really_undominated"]["violations"]),
            0.0,
        )
    )
    checks.append(
        _check(
            "route-families-are-different-families",
            "offsetting the start point does not rediscover the heading fan; the "
            "two exercise different columns of Phi, compared by start point "
            "rather than by a label the families prefix differently",
            float(planning["families_differ"]["shared_starts"]),
            0.0,
        )
    )
    checks.append(
        _check(
            "heading-fan-shares-one-start",
            "and a fan is a fan: every one of its routes leaves the same point, "
            "so it moves only the b column",
            float(planning["families_differ"]["distinct_fan_starts"]),
            1.0,
        )
    )
    for family, payload in planning["families"].items():
        checks.append(
            _check(
                f"pareto-front-is-not-everything/{family}",
                "and the front is a selection rather than the whole set, so the "
                "declared objectives are actually in tension",
                float(payload["front"]["front_size"]),
                float(payload["routes"]) - 0.5,
            )
        )

    for row in results["uncertainty_budget"]:
        checks.append(
            _check(
                f"budget-is-invertible/{row['case']}",
                "the declared budget can whiten a residual: at least one term has "
                "full rank, which in practice means the sensor's own noise",
                0.0 if row["positive_definite"] else 1.0,
                0.0,
            )
        )
        checks.append(
            _check(
                f"budget-systematic-terms-are-singular/{row['case']}",
                "and the systematic terms alone cannot, because a perfectly "
                "correlated error is perfectly predictable -- the distinction a "
                "per-sample variance sum would have erased",
                0.0 if row["systematic_only_is_singular"] else 1.0,
                0.0,
            )
        )
        checks.append(
            _check(
                f"budget-shares-account-for-everything/{row['case']}",
                "the breakdown sums to the total, so no contribution is unaccounted",
                abs(sum(row["shares_at_worst"].values()) - 1.0),
                1e-12,
            )
        )

    for row in results["prediction_chain"]:
        if row["curvature_is_constant"]:
            checks.append(
                _check(
                    f"prediction-chain-closes/{row['case']}",
                    "naming the two second-order transformations closes the gap to "
                    "an independently computed ambient chord, from the size of the "
                    "effect down to the numerical floor",
                    row["worst_chain_disagreement"],
                    1e-9,
                )
            )
            checks.append(
                _check(
                    f"prediction-chain-is-worth-it/{row['case']}",
                    "and by three orders of magnitude or more, which is why the "
                    "transformations are objects and not labels",
                    row["smallest_improvement_factor"],
                    1000.0,
                    comparison=">=",
                )
            )
        else:
            checks.append(
                _check(
                    f"prediction-chain-declares-the-missing-term/{row['case']}",
                    "where the curvature varies only the chord correction is "
                    "computable, and the prediction says the other was not applied "
                    "rather than appearing complete",
                    0.0 if row["intrinsic_correction"].startswith("not-applied") else 1.0,
                    0.0,
                )
            )
            checks.append(
                _check(
                    f"prediction-chain-never-worse/{row['case']}",
                    "and the part that is applied never moves the prediction away "
                    "from the measurement",
                    0.0 if row["chain_is_never_worse"] else 1.0,
                    0.0,
                )
            )

    for row in results["jet_step_sensitivity"]:
        checks.append(
            _check(
                f"jet-step-foci-are-stable/{row['case']}",
                "every differencing step in the sweep finds the same number of foci: "
                "the jet's step may cost accuracy, and it may not invent or erase a "
                "conjugate point",
                0.0 if row["every_step_found_the_same_foci"] else 1.0,
                0.0,
            )
        )
        if row["default_over_worst"] is not None:
            checks.append(
                _check(
                    f"jet-step-default-is-at-the-good-end/{row['case']}",
                    "the declared relative step is within 1e-3 of the worst step in "
                    "the sweep, on every surface -- a default cannot be optimal "
                    "everywhere, but it must not be arbitrary",
                    row["default_over_worst"],
                    1e-3,
                )
            )

    for row in results["envelopes"]:
        geometry = row["geometry"]
        checks.append(
            _check(
                f"surface-euler-identity/{row['case']}",
                "the two normal curvatures the record carries sum to 2H, tying the "
                "frame it publishes to the surface it claims to be on",
                geometry["euler_residual"],
                1e-12,
            )
        )
        checks.append(
            _check(
                f"surface-frame-is-orthonormal/{row['case']}",
                "the carried frame is a right-handed Darboux triad: "
                "transverse = normal x tangent, and the three are orthonormal",
                max(
                    geometry["frame_orientation_residual"],
                    geometry["max_tangent_norm_error"],
                    geometry["max_frame_inner_product"],
                ),
                1e-12,
            )
        )
        checks.append(
            _check(
                f"surface-no-free-robustness/{row['case']}",
                "conjugating by the tolerance box leaves det Phi = 1, so no path "
                "contracts every starting-pose error at once -- checked as the "
                "determinant itself, not as a product of singular values",
                row["scaled_determinant_error"],
                config.wronskian_tolerance,
            )
        )
        checks.append(
            _check(
                f"surface-amplification-floor/{row['case']}",
                "and the larger of the two is never below 1 anywhere on the path",
                row["min_scaled_singular_value_max"],
                1.0 - 1e-12,
                comparison=">=",
            )
        )
        checks.append(
            _check(
                f"surface-record/{row['case']}",
                "the path presents itself as a transfer record that reproduces its "
                "own arrays and declares units, resolution and observation mode",
                row["record"]["round_trip_error"],
                0.0,
            )
        )

    scan = results["heading_scan"]
    selection = scan["route_selection"]
    checks.append(
        _check(
            "surface-route-feasible",
            "at least one candidate route meets every declared process limit, and "
            "the recommendation is one of them",
            float(selection["n_feasible"]),
            1.0,
            comparison=">=",
        )
    )
    resolvable = scan["route_selection_with_tracking"]
    checks.append(
        _check(
            "surface-route-follows-constraints",
            "adding the sensor's acquisition schedule changes which route is "
            "recommended, so the answer follows a declared instrument limit and "
            "not a threshold chosen here",
            0.0
            if (
                selection["recommended"] is not None
                and resolvable["recommended"] is not None
                and selection["recommended"]["label"] != resolvable["recommended"]["label"]
            )
            else 1.0,
            0.0,
        )
    )
    checks.append(
        _check(
            "surface-tracking-discriminates",
            "the schedule separates routes that hold the track from routes that "
            "lose it, rather than passing everything or nothing",
            float(len(resolvable["outcomes"])),
            2.0,
            comparison=">=",
        )
    )
    outcome_by_label = {
        entry["label"]: entry["tracking"]["outcome"]
        for entry in resolvable["assessments"]
        if entry["tracking"] is not None
    }
    declared_label = selection["recommended"]["label"] if selection["recommended"] else None
    checks.append(
        _check(
            "surface-route-declared-choice-is-not-trackable",
            "the route the declared process limits pick is one the scanner does "
            "not hold, which is why the instrument and not the ranking scalar "
            "decides",
            0.0
            if declared_label is not None
            and outcome_by_label.get(declared_label) not in (None, "TRACKED")
            else 1.0,
            0.0,
        )
    )
    checks.append(
        _check(
            "surface-tracking-outcome-per-route",
            "every candidate route carries its own recorded tracking outcome, so "
            "the figure's marking is read from the report rather than assumed",
            float(len(outcome_by_label)),
            float(scan["n_headings"]),
            comparison=">=",
        )
    )
    lost_labels = {
        label for label, outcome in outcome_by_label.items() if outcome != "TRACKED"
    }
    focus_labels = {
        heading_label(row["heading_degrees"])
        for row in scan["headings"]
        if row["passes_a_focus"]
    }
    checks.append(
        _check(
            "surface-track-loss-coincides-with-focus-in-this-configuration",
            "for THIS torus, starting uncertainty, H, scanner noise and schedule, "
            "the routes the scanner cannot hold are exactly the routes that pass "
            "through a focus -- corroboration from two independent computations, "
            "not a general identity: see surface-unresolvable-without-a-focus and "
            "surface-resolvable-beside-a-focus for the two ways they come apart",
            float(len(lost_labels ^ focus_labels)),
            0.0,
        )
    )

    distinct = results["focus_versus_resolvability"]
    unresolvable = distinct["no_focus_but_unresolvable"]
    checks.append(
        _check(
            "surface-unresolvable-without-a-focus",
            "shrinking the admitted starting uncertainty drives rho below the "
            "acquire threshold on a path with no focus at any arc length, so low "
            "resolvability is not evidence of a focus",
            float(unresolvable["n_focus_points"])
            + (
                0.0
                if unresolvable["max_resolvability"] < unresolvable["acquire_threshold"]
                else 1.0
            ),
            0.0,
        )
    )
    nearest = distinct["focus_nearby_but_resolvable"]["samples"][-1]
    checks.append(
        _check(
            "surface-resolvable-beside-a-focus",
            "and |b| beside a genuine conjugate point is small but not zero, so "
            "the metrology sigma that resolves the admitted poses there is "
            "finite -- a focus does not by itself make a path unobservable",
            0.0 if np.isfinite(nearest["metrology_sigma_to_reach_acquire_threshold"])
            and nearest["metrology_sigma_to_reach_acquire_threshold"] > 0.0
            else 1.0,
            0.0,
        )
    )
    checks.append(
        _check(
            "surface-focus-is-instrument-independent",
            "the conjugate point on the spherical cap is at pi whatever scanner "
            "is pointed at it, which is what makes it a different quantity from "
            "rho",
            abs(distinct["focus_nearby_but_resolvable"]["focus_points"][0] - float(np.pi)),
            1e-9,
        )
    )
    checks.append(
        _check(
            "surface-resolvability-scale-invariant",
            "rho is the same for the same physical situation drawn at twice the "
            "size, which is what a threshold on |b| could never be",
            resolvable["scale_invariance"]["max_relative_difference"],
            1e-9,
        )
    )

    rescaling = results["chart_rescaling_invariance"]
    checks.append(
        _check(
            "surface-chart-orthogonality-rescaling-invariant",
            "reparameterising a coordinate does not change how orthogonal the "
            "chart is, so the conditioning number measures the chart and not "
            "the parameterisation",
            rescaling["orthogonality_absolute_difference"],
            1e-9,
        )
    )
    checks.append(
        _check(
            "surface-retired-chart-criterion-is-not",
            "and the criterion this replaced moves by exactly the rescaling "
            "factor on the same regular chart, which is why it was replaced",
            abs(rescaling["retired_ratio"] - rescaling["expected_ratio"]),
            1e-6,
        )
    )

    for row in results["envelopes"]:
        checks.append(
            _check(
                f"surface-chart-validity/{row['case']}",
                "the path stays inside the declared parameter domain and the "
                "chart stays conditioned along all of it",
                0.0 if row["chart"]["valid"] else 1.0,
                0.0,
            )
        )
    checks.append(
        _check(
            "surface-heading-scan",
            "starting heading measurably changes how far an aiming error is "
            "carried, so there is a choice to make",
            scan["amplification_ratio"],
            1.1,
            comparison=">=",
        )
    )
    checks.append(
        _check(
            "surface-heading-scan-drift",
            "every candidate heading in the scan held unit speed",
            max(row["max_speed_drift"] for row in scan["headings"]),
            config.speed_drift_tolerance,
        )
    )
    checks.append(
        _check(
            "surface-heading-scan-wronskian",
            "and every one of them held det Phi = 1",
            max(row["max_wronskian_drift"] for row in scan["headings"]),
            config.wronskian_tolerance,
        )
    )
    checks.append(
        _check(
            "surface-heading-scan-focus-tradeoff",
            "the best-scoring heading buys its score with a focus, so the ranking "
            "scalar alone is not a robustness criterion",
            0.0 if scan["lowest_amplification"]["passes_a_focus"] else 1.0,
            0.0,
        )
    )
    return checks


def run_surface_experiment(config: SurfaceConfig | None = None) -> dict[str, Any]:
    config = config or SurfaceConfig()
    cases = default_cases()
    results = {
        "anchored_to_closed_forms": anchor_to_closed_forms(config, cases),
        "two_routes": compare_two_routes(config, cases),
        "lateral_route": compare_lateral_route(config, cases),
        "self_convergence": measure_self_convergence(config, cases),
        "finite_difference_cost": measure_finite_difference_cost(config, cases),
        "error_budget": measure_error_budget(config, cases),
        "prediction_chain": measure_prediction_chain(config, cases),
        "uncertainty_budget": measure_uncertainty_budget(config, cases),
        "route_planning": measure_route_planning(config, cases),
        "jet_step_sensitivity": measure_jet_step_sensitivity(config, cases),
        "envelopes": build_envelopes(config, cases),
        "chart_rescaling_invariance": measure_chart_rescaling_invariance(cases),
        "focus_versus_resolvability": measure_focus_versus_resolvability(config, cases),
        "heading_scan": scan_for_robust_heading(config, cases),
    }
    checks = collect_checks(results, config)
    core = _jsonable(
        {
            "schema": REPORT_SCHEMA,
            "supersedes": SUPERSEDES,
            "schema_changes": [
                "adds results.route_planning: the accumulated observability "
                "Gramian, boundaries computed from the declared part, a second "
                "route family that exercises the a column, and a Pareto front "
                "instead of a ranking scalar",
                "adds results.uncertainty_budget: every declared source of error, "
                "its shape, and which one dominates -- the starting pose is one "
                "term and rarely the largest",
                "adds results.prediction_chain: the named transformations from "
                "Phi dz0 to an ambient chord, against an independent measurement",
                "adds results.error_budget: a per-quantity step-doubling error "
                "estimate, checked against the closed forms where one exists",
                "adds results.jet_step_sensitivity: what the finite-difference "
                "jet's relative step costs along a whole path, not at one point",
                "envelopes: adds the path geometry the record now carries -- the "
                "Darboux frame as vectors and both normal curvatures -- checked "
                "against Euler's theorem on every surface",
                "envelopes.record: now the full v2 contract, carrying the path "
                "geometry, the chart validity, the path type and the convergence "
                "estimate alongside the transfer map",
                "heading_scan.route_selection: routes are chosen by declared process "
                "limits and a sensor acquisition schedule, not by a threshold on |b|",
                "envelopes: adds chart validity, the transfer record, and the "
                "scaled-transfer singular values",
                "adds results.lateral_route: the a column checked by moving the start point",
                "adds the lateral column and the Wronskian to the anchors and envelopes",
                "heading_scan: max_abs_jacobi_field -> max_forward_amplification, adds "
                "focus_margin and passes_a_focus, most/least_tolerant -> "
                "lowest/highest_amplification, sensitivity_ratio -> amplification_ratio",
                "adds results.chart_rescaling_invariance: the anisotropic-rescaling "
                "defect that retired sqrt(EG - F^2)/max(E, G)",
                "adds results.focus_versus_resolvability: a geometric focus and low "
                "instrument resolvability are separate quantities, and both are reported",
                "route_selection: min_focus_clearance is removed, not deprecated; a "
                "dimensionful clearance may be reported but may not decide",
                "tracking outcomes carry acquisition_window_started_at / "
                "acquisition_declared_at and loss_started_at / track_loss_declared_at, "
                "because a causal instrument cannot act on a window before it closes",
                "heading_scan.route_selection_with_tracking: adds per-route "
                "assessments so each candidate carries its own tracking outcome",
                "adds observation_modes and tags each comparison with one",
            ],
            "experiment": "curvature-aware-path-sensitivity-on-parametric-surfaces",
            "claim_scope": "numerical-verification-anchored-to-the-constant-curvature-stage",
            "depends_on": "geodesic-jacobi-report-v3",
            "observation_modes": observation_catalogue(),
            "cases": [
                {
                    "case": case.key,
                    "surface": case.surface.name,
                    "description": case.surface.description,
                    "start": {"u": case.u0, "v": case.v0, "heading": case.heading},
                    "length": case.length,
                    "closed_form": case.reference,
                }
                for case in cases
            ],
            "config": config.to_dict(),
            "results": results,
            "checks": checks,
        }
    )
    report = dict(core)
    report["summary"] = {
        "n_checks": len(checks),
        "n_failed": sum(1 for check in checks if not check["passed"]),
        "all_passed": all(check["passed"] for check in checks),
        "failed": [check["id"] for check in checks if not check["passed"]],
    }
    report["content_hash"] = content_hash(core)
    report["environment"] = {
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(terse=True),
    }
    return report
