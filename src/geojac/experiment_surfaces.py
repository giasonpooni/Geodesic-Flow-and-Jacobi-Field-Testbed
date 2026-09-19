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
   reported, because a camera measures chords and someone will need that
   number.
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

from . import __version__
from .analysis import fit_power_law, successive_orders
from .envelope import (
    PathEnvelope,
    finite_difference_jacobi,
    integrate_path,
    scan_headings,
)
from .experiment import _check, _jsonable, content_hash
from .surfaces import (
    ParametricSurface,
    cylinder,
    hyperbolic_paraboloid,
    plane,
    pseudosphere,
    sphere,
    torus,
)

REPORT_SCHEMA = "geodesic-jacobi-surfaces-v1"


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
        if self.reference is None:
            return None
        return {"s": lambda x: x, "sin(s)": np.sin, "sinh(s)": np.sinh}[self.reference](s)

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
    epsilons: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.logspace(-4.0, -1.0, 13))
    )
    two_route_arc_length: float = 1.0
    two_route_steps: int = 1000
    fit_floor: float = 1e-8
    fit_ceiling: float = 1e-3

    transverse_tolerances: tuple[float, ...] = (1e-3, 1e-2)
    heading_scan_count: int = 24
    heading_scan_key: str = "torus"
    # Long enough for the geodesics to leave the neighbourhood they started in
    # and sample both signs of the torus's curvature; at a short path length
    # every heading looks much the same and there is no decision to make.
    heading_scan_length: float = 6.0
    heading_scan_steps: int = 3000
    curve_samples: int = 200

    analytic_tolerance: float = 1e-11
    finite_difference_tolerance: float = 1e-7
    order_tolerance: float = 0.2
    exponent_tolerance: float = 0.05
    coefficient_tolerance: float = 1e-2
    chord_closed_form_tolerance: float = 1e-9
    exactness_threshold: float = 1e-13
    speed_drift_tolerance: float = 1e-9
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
                "max_jacobi_error": float(np.max(np.abs(envelope.jacobi_field - expected))),
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
                "fd_step": float(case.surface.fd_step),
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


def scan_for_robust_heading(config: SurfaceConfig, cases) -> dict[str, Any]:
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
    best, worst = rows[0], rows[-1]
    return {
        "case": case.key,
        "surface": case.surface.name,
        "start": {"u": case.u0, "v": case.v0},
        "length": float(config.heading_scan_length),
        "n_headings": int(config.heading_scan_count),
        "most_tolerant": best,
        "least_tolerant": worst,
        "sensitivity_ratio": float(
            worst["max_abs_jacobi_field"] / best["max_abs_jacobi_field"]
        ),
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

    scan = results["heading_scan"]
    checks.append(
        _check(
            "surface-heading-scan",
            "starting heading measurably changes how far an aiming error is "
            "carried, so there is a choice to make",
            scan["sensitivity_ratio"],
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
    return checks


def run_surface_experiment(config: SurfaceConfig | None = None) -> dict[str, Any]:
    config = config or SurfaceConfig()
    cases = default_cases()
    results = {
        "anchored_to_closed_forms": anchor_to_closed_forms(config, cases),
        "two_routes": compare_two_routes(config, cases),
        "self_convergence": measure_self_convergence(config, cases),
        "finite_difference_cost": measure_finite_difference_cost(config, cases),
        "envelopes": build_envelopes(config, cases),
        "heading_scan": scan_for_robust_heading(config, cases),
    }
    checks = collect_checks(results, config)
    core = _jsonable(
        {
            "schema": REPORT_SCHEMA,
            "experiment": "curvature-aware-path-sensitivity-on-parametric-surfaces",
            "claim_scope": "numerical-verification-anchored-to-the-constant-curvature-stage",
            "depends_on": "geodesic-jacobi-report-v1",
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
        "geojac_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(terse=True),
    }
    return report
