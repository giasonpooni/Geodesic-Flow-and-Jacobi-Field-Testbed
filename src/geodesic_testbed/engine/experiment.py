"""The experiment: four sweeps, one report, explicit pass/fail checks.

The testbed answers four questions, in order of increasing ambition.

1. *Does the Jacobi solver converge, and at the rate its method promises?*
   Integrate ``j'' + K j = 0`` against ``s``, ``sin s``, ``sinh s`` with
   methods of order 1, 2 and 4 over a ladder of step sizes.

2. *Does the geodesic flow itself converge?*  Same ladder, but integrating the
   ambient second-order geodesic equation and comparing with the closed-form
   exponential map -- plus how far the numerical state drifts off the unit
   tangent bundle.

3. *Over what range of perturbations is the Jacobi field actually predictive?*
   A Jacobi field is a first-order variation, not a finite distance.  Sweeping
   the initial angle ``epsilon`` exposes both sides of that statement.  From
   above, a modelling error that grows like ``eps^3``.  From below, an
   integrator noise floor -- but only when the two trajectories do not share
   their discretisation error, so the sweep measures the separation twice:
   once with the whole fan advanced by one shared step sequence, where the
   error is common-mode and cancels, and once with the two geodesics flowed at
   different resolutions, where it does not and the floor appears.

4. *What does all of that say about a real path?*  The same numbers, restated as
   an instrument would report them: how much an initial aiming error is
   amplified after a path of length ``s``, how large an initial error a given
   transverse tolerance allows, and what the first-order prediction would get
   wrong at the aiming errors a physical bench can actually set.

5. *What happens at a zero of the Jacobi field?*  On the sphere ``sin s``
   vanishes at ``s = pi``.  That conjugate point is where the geodesic stops
   being minimising, which is the second boundary this project refuses to blur.

Every quantitative claim ends up in :func:`collect_checks` with a threshold
attached, so the report either passes or it does not.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .. import __version__
from .analysis import fit_power_law, successive_orders
from .canonical import CANONICAL_DIGITS, content_hash
from .canonical import jsonable as _jsonable
from .flows import (
    geodesic_position_error,
    integrate_geodesic,
    integrate_geodesic_bundle,
    integrate_jacobi,
    jacobi_reference,
)
from .integrators import get_integrator, integrate, step_ladder
from .observation import catalogue as observation_catalogue
from .record import TransferRecord
from .spaceforms import SpaceForm, all_space_forms
from .transfer import (
    INITIAL_STATE,
    TransferMap,
    constant_curvature_transfer,
    transfer_from_trajectory,
    transfer_rhs,
)
from .uncertainty import curvature_sensitivity

REPORT_SCHEMA = "geodesic-jacobi-report-v3"
SUPERSEDES = "geodesic-jacobi-report-v2"


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything the experiment is allowed to depend on."""

    arc_length: float = 2.0
    step_counts: tuple[int, ...] = field(default_factory=lambda: step_ladder(10, 9))
    integrators: tuple[str, ...] = ("euler", "midpoint", "rk4")

    # Discretisation errors stop shrinking once they reach double-precision
    # noise; levels below this floor are excluded from order fits.
    roundoff_floor: float = 1e-12
    # Below this, a method is reproducing the exact solution and no order can
    # be measured (true of every method on the flat model, where the solutions
    # are linear in the arc length).
    exactness_threshold: float = 1e-13

    sample_arc_lengths: tuple[float, ...] = (0.5, 1.0, 2.0)
    epsilon_min_exponent: float = -8.0
    epsilon_max_exponent: float = 0.0
    n_epsilons: int = 33
    perturbation_steps: int = 1000
    relative_fit_floor: float = 1e-12
    relative_fit_ceiling: float = 1e-4
    tolerances: tuple[float, ...] = (1e-3, 1e-6)

    # Second flow of the same fan, deliberately at a different resolution from
    # the first, so that the two trajectories do not share their local error.
    mixed_resolution_steps: int = 64
    noise_tolerance: float = 1e-3
    noise_blindness_threshold: float = 0.5

    # Instrument readout.  Lengths are in units of the radius of curvature.
    sensitivity_arc_lengths: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0)
    transverse_tolerances: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    bench_angles_degrees: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    bench_arc_length: float = 2.0
    bench_steps: int = 2000

    conjugate_span_multiple: float = 2.0
    conjugate_steps: int = 20000

    order_tolerance: float = 0.15
    # Below this the measured determinant is dominated by its own accumulated
    # rounding (about n * eps), not by the method's drift.
    wronskian_theory_floor: float = 1e-9
    wronskian_theory_tolerance: float = 1e-3
    # Deliberately coarse: a refinement that only helps at fine resolution is
    # not worth having.
    focus_refinement_span: float = 4.0
    focus_refinement_samples: int = 21
    exponent_tolerance: float = 0.05
    coefficient_tolerance: float = 0.01
    epsilon_star_tolerance: float = 0.01
    drift_tolerance: float = 1e-9

    def epsilons(self) -> np.ndarray:
        return np.logspace(self.epsilon_min_exponent, self.epsilon_max_exponent, self.n_epsilons)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


# ---------------------------------------------------------------------------
# sweep 1: the Jacobi equation itself
# ---------------------------------------------------------------------------
def sweep_jacobi_convergence(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for form in all_space_forms():
        for method in config.integrators:
            steps: list[float] = []
            max_errors: list[float] = []
            endpoint_errors: list[float] = []
            levels: list[dict[str, Any]] = []
            for n_steps in config.step_counts:
                grid, j, _ = integrate_jacobi(
                    form.K, length=config.arc_length, n_steps=n_steps, method=method
                )
                error = np.abs(j - jacobi_reference(grid, form.K))
                h = config.arc_length / n_steps
                steps.append(h)
                max_errors.append(float(error.max()))
                endpoint_errors.append(float(error[-1]))
                levels.append(
                    {
                        "n_steps": int(n_steps),
                        "h": float(h),
                        "max_error": float(error.max()),
                        "endpoint_error": float(error[-1]),
                    }
                )
            rows.append(
                _convergence_row(
                    config,
                    form=form,
                    method=method,
                    quantity="jacobi-field",
                    reference=form.reference_solution_name,
                    steps=steps,
                    errors=max_errors,
                    endpoint_errors=endpoint_errors,
                    levels=levels,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# sweep 2: the geodesic flow
# ---------------------------------------------------------------------------
def sweep_flow_convergence(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for form in all_space_forms():
        for method in config.integrators:
            steps: list[float] = []
            max_errors: list[float] = []
            endpoint_errors: list[float] = []
            levels: list[dict[str, Any]] = []
            finest_drift: dict[str, float] = {}
            for n_steps in config.step_counts:
                grid, points, velocities = integrate_geodesic(
                    form, length=config.arc_length, n_steps=n_steps, method=method
                )
                error = geodesic_position_error(form, grid, points)
                residuals = form.constraint_residuals(points, velocities)
                drift = {key: float(np.max(value)) for key, value in residuals.items()}
                h = config.arc_length / n_steps
                steps.append(h)
                max_errors.append(float(error.max()))
                endpoint_errors.append(float(error[-1]))
                levels.append(
                    {
                        "n_steps": int(n_steps),
                        "h": float(h),
                        "max_error": float(error.max()),
                        "endpoint_error": float(error[-1]),
                        "constraint_drift": drift,
                    }
                )
                finest_drift = drift
            row = _convergence_row(
                config,
                form=form,
                method=method,
                quantity="geodesic-position",
                reference="exp(p0, v0, s)",
                steps=steps,
                errors=max_errors,
                endpoint_errors=endpoint_errors,
                levels=levels,
            )
            row["finest_constraint_drift"] = finest_drift
            rows.append(row)
    return rows


def _convergence_row(
    config: ExperimentConfig,
    *,
    form: SpaceForm,
    method: str,
    quantity: str,
    reference: str,
    steps: list[float],
    errors: list[float],
    endpoint_errors: list[float],
    levels: list[dict[str, Any]],
) -> dict[str, Any]:
    integrator = get_integrator(method)
    exact = max(errors) < config.exactness_threshold
    fit = fit_power_law(steps, errors, y_floor=config.roundoff_floor)
    return {
        "curvature": form.K,
        "curvature_label": form.label,
        "surface": form.name,
        "quantity": quantity,
        "reference": reference,
        "integrator": method,
        "expected_order": integrator.order,
        "regime": "exact-to-roundoff" if exact else "converging",
        "fitted_order": None if exact else float(fit.exponent),
        "order_error": None if exact else float(abs(fit.exponent - integrator.order)),
        "fit": fit.to_dict(),
        "successive_orders": successive_orders(steps, errors),
        "max_error_over_levels": float(max(errors)),
        "min_error_over_levels": float(min(errors)),
        "finest_endpoint_error": float(endpoint_errors[-1]),
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# sweep 2b: the Wronskian, an invariant the integrator never enforces
# ---------------------------------------------------------------------------
_ONE_STEP_DETERMINANT = {
    "euler": "1 + K h^2",
    "midpoint": "1 + K^2 h^4 / 4",
    "rk4": "(1 - K h^2/2 + K^2 h^4/24)^2 + K (h - K h^3/6)^2",
}


def _wronskian_theory_drift(K: float, h: float, n_steps: int, method: str) -> float:
    """``|d^n - 1|``: the exact Wronskian drift of a fixed-step method on constant K."""
    if method == "euler":
        excess = K * h**2
    elif method == "midpoint":
        excess = K**2 * h**4 / 4.0
    elif method == "rk4":
        # det(alpha I + beta A) = alpha^2 + K beta^2, with the rk4 stability
        # polynomial closed on A^2 = -K I. Expanding this to leading order gives
        # -K^3 h^6 / 72, but the untruncated form is what the solver actually
        # applies, and at h = 0.2 the two already differ by half a percent.
        alpha = 1.0 - K * h**2 / 2.0 + K**2 * h**4 / 24.0
        beta = h - K * h**3 / 6.0
        excess = alpha * alpha + K * beta * beta - 1.0
    else:  # pragma: no cover - guard
        raise ValueError(f"no closed-form Wronskian drift for {method!r}")
    # d^n - 1 with d = 1 + excess, evaluated without losing the small excess.
    return float(abs(np.expm1(n_steps * np.log1p(excess))))


def sweep_wronskian(config: ExperimentConfig) -> list[dict[str, Any]]:
    """How well each method conserves ``det Phi = a b' - a' b = 1``.

    The Jacobi equation has no first-derivative term, so its Wronskian is
    exactly conserved and starts at 1. That is not an approximation being
    measured against a reference -- it is an identity, and any departure is
    entirely the integrator's.

    It is also structural rather than accumulated, and for a constant ``K`` it
    is known exactly. The one-step map is a polynomial in ``hA`` with
    ``A^2 = -K I``, so each method has its own one-step determinant, and since
    the determinant of a product is the product of determinants, ``n`` equal
    steps give ``det Phi = d^n`` with no approximation at all:

    ====== ================================================= ============
    method one-step ``d``                                    order in h
    ====== ================================================= ============
    euler  ``1 + K h^2``                                     1
    mid    ``1 + K^2 h^4 / 4``                               3
    rk4    ``(1 - Kh^2/2 + K^2h^4/24)^2 + K(h - Kh^3/6)^2``  5
    ====== ================================================= ============

    (rk4's is ``1 - K^3 h^6 / 72`` to leading order, but the untruncated form
    is what the method applies and the two already differ by half a percent at
    ``h = 0.2``.)

    So this sweep measures three more cleanly separated slopes -- from a
    quantity that needed no reference solution -- and can be checked not just
    for its slope but against the exact value ``|d^n - 1|``.
    """
    rows: list[dict[str, Any]] = []
    expected = {"euler": 1, "midpoint": 3, "rk4": 5}
    for form in all_space_forms():
        for method in config.integrators:
            steps, drifts, levels = [], [], []
            for n_steps in config.step_counts:
                grid, trajectory = integrate(
                    transfer_rhs(form.K),
                    np.asarray(INITIAL_STATE, dtype=float),
                    length=config.arc_length,
                    n_steps=n_steps,
                    method=method,
                )
                drift = float(
                    np.max(transfer_from_trajectory(grid, trajectory).wronskian_drift)
                )
                h = config.arc_length / n_steps
                theory = _wronskian_theory_drift(form.K, h, int(n_steps), method)
                steps.append(h)
                drifts.append(drift)
                levels.append(
                    {
                        "n_steps": int(n_steps),
                        "h": float(h),
                        "drift": drift,
                        "theory_drift": theory,
                        "relative_error": (
                            float(abs(drift / theory - 1.0)) if theory > 0.0 else None
                        ),
                    }
                )
            exact = max(drifts) < config.exactness_threshold
            resolved = [
                level["relative_error"]
                for level in levels
                if level["theory_drift"] > config.wronskian_theory_floor
            ]
            fit = fit_power_law(steps, drifts, y_floor=config.roundoff_floor)
            order = expected[method]
            rows.append(
                {
                    "curvature": form.K,
                    "curvature_label": form.label,
                    "integrator": method,
                    "invariant": "det Phi = a b' - a' b = 1",
                    "expected_drift_order": order,
                    "regime": "exact-to-roundoff" if exact else "drifting",
                    "fitted_drift_order": None if exact else float(fit.exponent),
                    "drift_order_error": (
                        None if exact or order is None else float(abs(fit.exponent - order))
                    ),
                    "finest_drift": float(drifts[-1]),
                    "coarsest_drift": float(drifts[0]),
                    "one_step_determinant": _ONE_STEP_DETERMINANT[method],
                    "max_relative_error_vs_theory": (
                        float(max(resolved)) if resolved else None
                    ),
                    "levels_resolved_above_roundoff": len(resolved),
                    "fit": fit.to_dict(),
                    "levels": levels,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# sweep 2c: what det Phi = 1 costs a route planner
# ---------------------------------------------------------------------------
def sweep_transfer_determinant(config: ExperimentConfig) -> list[dict[str, Any]]:
    """The conserved Wronskian, and its consequence for a route planner.

    Scaling the transfer map by the tolerance box, ``S^-1 Phi S``, makes its
    entries pure ratios so that paths can be compared -- and leaves the
    determinant at 1, so the two singular values are reciprocal. Whatever
    direction of starting-pose error the flow contracts, it expands the
    conjugate direction by the same factor, and the larger is never below one.

    That is the precise sense in which a low ``max |b|`` is error
    redistribution rather than robustness, so it is measured rather than
    asserted, on a range of tolerance boxes with very different aspect ratios.

    **The invariant is checked directly, not through the singular values.**
    ``sigma_1 sigma_2 = |det Phi|`` is an identity, and testing ``det Phi = 1``
    by forming that product tests the SVD as much as the flow: ``sigma_1``
    comes back with a relative error of order ``eps``, so on the ``10^5``
    aspect-ratio box the product is accurate only to about ``10^-12`` and the
    threshold is really measuring how badly conditioned the box was. The
    determinant of the 2x2 itself is two products and a subtraction and holds
    to machine precision on every box, so that is what carries the check.
    The product error is still reported, as a measurement of the SVD rather
    than of the geometry.
    """
    boxes = ((1e-3, 1e-3), (1e-2, 1e-4), (1e-5, 1e-1))
    rows: list[dict[str, Any]] = []
    for form in all_space_forms():
        grid = np.linspace(0.0, config.arc_length, 401)
        phi = constant_curvature_transfer(grid, form.K)
        record = TransferRecord(
            arclength=grid,
            gaussian_curvature=np.full_like(grid, form.K),
            a=phi.a, a_rate=phi.a_rate, b=phi.b, b_rate=phi.b_rate,
            domain="constant-curvature",
        )
        determinant_error = float(np.max(np.abs(record.determinant - 1.0)))
        scaled_determinant_error = 0.0
        product_error = 0.0
        smallest = float("inf")
        per_box = []
        for lateral, heading in boxes:
            scaled = float(np.max(np.abs(record.scaled_determinant(lateral, heading) - 1.0)))
            singular = record.scaled_singular_values(lateral, heading)
            product = float(np.max(np.abs(singular[:, 0] * singular[:, 1] - 1.0)))
            scaled_determinant_error = max(scaled_determinant_error, scaled)
            product_error = max(product_error, product)
            smallest = min(smallest, float(np.min(singular[:, 0])))
            per_box.append(
                {
                    "max_lateral": lateral,
                    "max_heading": heading,
                    "aspect_ratio": float(max(lateral, heading) / min(lateral, heading)),
                    "amplification_score": float(np.max(singular[:, 0])),
                    "min_largest_singular_value": float(np.min(singular[:, 0])),
                    "scaled_determinant_error": scaled,
                    "singular_value_product_error": product,
                }
            )
        rows.append(
            {
                "curvature": form.K,
                "curvature_label": form.label,
                "identity": "det Phi = a b' - a' b = 1, and sigma_1 sigma_2 = |det Phi|",
                "determinant_error": determinant_error,
                "scaled_determinant_error": scaled_determinant_error,
                #: Reported, not decisive: this is the SVD's accuracy on a
                #: badly conditioned box, not the flow's accuracy.
                "singular_value_product_error": product_error,
                "min_largest_singular_value": smallest,
                "boxes": per_box,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# sweep 2d: how well a focus is located
# ---------------------------------------------------------------------------
def sweep_focus_refinement(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Refined against linear root location, where the answer is known exactly."""
    rows: list[dict[str, Any]] = []
    for form in all_space_forms():
        if form.K <= 0.0:
            continue  # only the sphere has a focus
        grid = np.linspace(0.0, config.focus_refinement_span, config.focus_refinement_samples)
        phi = constant_curvature_transfer(grid, form.K)
        event = phi.focus_events(component="b")[0]
        values = phi.b
        index = int(np.argmax(np.sign(values[:-1]) * np.sign(values[1:]) < 0.0))
        weight = values[index] / (values[index] - values[index + 1])
        linear = float(grid[index] + weight * (grid[index + 1] - grid[index]))
        exact = float(np.pi / np.sqrt(form.K))
        rows.append(
            {
                "curvature": form.K,
                "curvature_label": form.label,
                "exact": exact,
                "sample_spacing": float(grid[1] - grid[0]),
                "refined": event.arc_length,
                "refined_error": float(abs(event.arc_length - exact)),
                "linear": linear,
                "linear_error": float(abs(linear - exact)),
                "reported_uncertainty": event.location_uncertainty,
                "derivative_at_focus": event.derivative,
                "method": "cubic-Hermite, Newton with bisection safeguard",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# sweep 3: where the first-order variation stops being predictive
# ---------------------------------------------------------------------------
def sweep_first_order_validity(config: ExperimentConfig) -> list[dict[str, Any]]:
    epsilons = config.epsilons()
    span = max(config.sample_arc_lengths)
    rows: list[dict[str, Any]] = []
    for form in all_space_forms():
        angles = np.concatenate([[0.0], epsilons])
        grid, points, _ = integrate_geodesic_bundle(
            form, angles, length=span, n_steps=config.perturbation_steps, method="rk4"
        )
        base = points[:, 0, :]
        fan = points[:, 1:, :]
        flowed = form.distance(base[:, None, :], fan)

        # The same fan again, but with the base geodesic and the perturbed ones
        # advanced by different step sequences, so their truncation errors no
        # longer cancel in the difference.
        coarse_steps = config.mixed_resolution_steps
        coarser_steps = coarse_steps // 2
        grid_a, points_a, _ = integrate_geodesic_bundle(
            form, [0.0], length=span, n_steps=coarse_steps, method="rk4"
        )
        grid_b, points_b, _ = integrate_geodesic_bundle(
            form, epsilons, length=span, n_steps=coarser_steps, method="rk4"
        )

        for s_value in config.sample_arc_lengths:
            index = _grid_index(grid, s_value, span, config.perturbation_steps)
            index_a = _grid_index(grid_a, s_value, span, coarse_steps)
            index_b = _grid_index(grid_b, s_value, span, coarser_steps)
            mixed = form.distance(points_a[index_a, 0, :], points_b[index_b, :, :])
            exact = form.exact_separation(s_value, epsilons)
            first_order = form.first_order_separation(s_value, epsilons)
            numeric = flowed[index]
            with np.errstate(divide="ignore", invalid="ignore"):
                relative_model = np.abs(exact / first_order - 1.0)
                relative_numeric = np.abs(numeric / first_order - 1.0)
                numeric_vs_exact = np.abs(numeric - exact) / np.abs(exact)
                mixed_vs_exact = np.abs(mixed - exact) / np.abs(exact)
            fit = fit_power_law(
                epsilons,
                relative_model,
                y_floor=config.relative_fit_floor,
                y_ceiling=config.relative_fit_ceiling,
            )
            theory_coefficient = float(form.relative_deviation_coefficient(s_value))
            epsilon_star = []
            for tolerance in config.tolerances:
                theory = float(form.epsilon_for_relative_tolerance(s_value, tolerance))
                measured = float(fit.invert(tolerance))
                epsilon_star.append(
                    {
                        "tolerance": float(tolerance),
                        "epsilon_star_measured": measured,
                        "epsilon_star_theory": theory,
                        "relative_difference": float(abs(measured / theory - 1.0))
                        if np.isfinite(theory) and theory > 0.0
                        else None,
                    }
                )
            usable = numeric_vs_exact < 1e-6
            mixed_flow_exact = float(np.max(mixed_vs_exact)) < 1e-13
            rows.append(
                {
                    "curvature": form.K,
                    "curvature_label": form.label,
                    "surface": form.name,
                    "arc_length": float(s_value),
                    "jacobi_field": float(jacobi_reference(s_value, form.K)),
                    "law": "sn_K(d/2) = sn_K(s) sin(eps/2)",
                    "observation_mode": "intrinsic-surface-distance",
                    "expected_exponent": 2.0,
                    "fitted_exponent": float(fit.exponent),
                    "fitted_coefficient": float(fit.prefactor),
                    "theory_coefficient": theory_coefficient,
                    "coefficient_relative_error": (
                        float(abs(fit.prefactor / theory_coefficient - 1.0))
                        if theory_coefficient > 0.0
                        else None
                    ),
                    "fit": fit.to_dict(),
                    "epsilon_star": epsilon_star,
                    "flow_agreement": {
                        "shared_step_sequence": {
                            "comment": (
                                "one step sequence for the whole fan: the truncation "
                                "error is common-mode and cancels in the separation"
                            ),
                            "integrator": "rk4",
                            "n_steps": int(config.perturbation_steps),
                            "h": float(span / config.perturbation_steps),
                            "max_relative_error": float(np.max(numeric_vs_exact)),
                            "relative_error_at_smallest_epsilon": float(numeric_vs_exact[0]),
                            "smallest_usable_epsilon": float(epsilons[usable].min())
                            if usable.any()
                            else None,
                        },
                        "mixed_resolution": {
                            "comment": (
                                "base and perturbed geodesic flowed at different "
                                "resolutions: the error no longer cancels, and below "
                                "eps_noise the separation drowns in it"
                            ),
                            "integrator": "rk4",
                            "n_steps_base": int(coarse_steps),
                            "n_steps_perturbed": int(coarser_steps),
                            "regime": "exact-flow" if mixed_flow_exact else "noise-floor",
                            "noise_tolerance": float(config.noise_tolerance),
                            "epsilon_noise_measured": _noise_threshold(
                                epsilons, mixed_vs_exact, config.noise_tolerance
                            ),
                            "max_relative_error": float(np.max(mixed_vs_exact)),
                            "relative_error_at_smallest_epsilon": float(mixed_vs_exact[0]),
                            "relative_error_at_largest_epsilon": float(mixed_vs_exact[-1]),
                            "lowest_decade_slope": _lowest_decade_slope(
                                epsilons, mixed_vs_exact
                            ),
                            "timelike_difference_samples": int(
                                np.sum(form.chord_squared(
                                    points_a[index_a, 0, :], points_b[index_b, :, :]
                                ) < 0.0)
                            ),
                        },
                    },
                    "samples": [
                        {
                            "epsilon": float(e),
                            "separation_exact": float(ex),
                            "separation_first_order": float(fo),
                            "separation_flowed": float(nu),
                            "relative_deviation_model": float(rm),
                            "relative_deviation_flowed": float(rn),
                            "flowed_vs_exact_relative": float(ne),
                            "mixed_resolution_vs_exact_relative": float(mx),
                        }
                        for e, ex, fo, nu, rm, rn, ne, mx in zip(
                            epsilons,
                            exact,
                            first_order,
                            numeric,
                            relative_model,
                            relative_numeric,
                            numeric_vs_exact,
                            mixed_vs_exact,
                            strict=True,
                        )
                    ],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# sweep 4: the same numbers, restated as an instrument readout
# ---------------------------------------------------------------------------
UNITS_NOTE = (
    "Lengths are in units of the radius of curvature R (the models are K = 0, +1, -1, "
    "so R = 1); a spherical coupon of radius 300 mm makes s = 1 a 300 mm path. "
    "Angles are radians unless the field name ends in _degrees."
)


def sweep_path_sensitivity(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Amplification, tolerance budget and bench predictions, per surface.

    Nothing here is new physics -- it is the closed-form ``sn_K`` and the
    measured validity limit, rearranged into the three questions someone
    planning a path on a curved surface actually asks: how much does an aiming
    error grow, how large an aiming error may I allow, and where does the
    first-order answer stop being one.
    """
    rows: list[dict[str, Any]] = []
    bench_angles = np.deg2rad(np.asarray(config.bench_angles_degrees, dtype=float))
    for form in all_space_forms():
        entries: list[dict[str, Any]] = []
        for s_value in config.sensitivity_arc_lengths:
            field = float(jacobi_reference(s_value, form.K))
            entries.append(
                {
                    "arc_length": float(s_value),
                    "jacobi_field": field,
                    "amplification_vs_flat": float(field / s_value),
                    "tolerance_budget": [
                        {
                            "transverse_tolerance": float(tolerance),
                            "max_initial_angle": float(tolerance / field),
                            "max_initial_angle_degrees": float(
                                np.rad2deg(tolerance / field)
                            ),
                        }
                        for tolerance in config.transverse_tolerances
                    ],
                    "first_order_validity": [
                        {
                            "relative_tolerance": float(tolerance),
                            "max_initial_angle": float(
                                form.epsilon_for_relative_tolerance(s_value, tolerance)
                            ),
                            "max_initial_angle_degrees": float(
                                np.rad2deg(
                                    form.epsilon_for_relative_tolerance(s_value, tolerance)
                                )
                            ),
                        }
                        for tolerance in config.tolerances
                    ],
                }
            )

        # An independent numerical check on the numbers a bench would be handed.
        grid, points, _ = integrate_geodesic_bundle(
            form,
            np.concatenate([[0.0], bench_angles]),
            length=config.bench_arc_length,
            n_steps=config.bench_steps,
            method="rk4",
        )
        flowed = form.distance(points[-1, 0, :], points[-1, 1:, :])
        exact = form.exact_separation(config.bench_arc_length, bench_angles)
        first_order = form.first_order_separation(config.bench_arc_length, bench_angles)
        rows.append(
            {
                "curvature": form.K,
                "curvature_label": form.label,
                "surface": form.name,
                "units": UNITS_NOTE,
                "by_arc_length": entries,
                "bench_predictions": {
                    "arc_length": float(config.bench_arc_length),
                    "integrator": "rk4",
                    "n_steps": int(config.bench_steps),
                    "max_flow_vs_closed_form_relative_error": float(
                        np.max(np.abs(flowed - exact) / np.abs(exact))
                    ),
                    "angles": [
                        {
                            "angle_degrees": float(degrees),
                            "angle": float(radians),
                            "separation_first_order": float(prediction),
                            "separation_exact": float(truth),
                            "separation_flowed": float(measured),
                            "first_order_relative_error": float(
                                abs(prediction / truth - 1.0)
                            ),
                        }
                        for degrees, radians, prediction, truth, measured in zip(
                            config.bench_angles_degrees,
                            bench_angles,
                            first_order,
                            exact,
                            flowed,
                            strict=True,
                        )
                    ],
                },
            }
        )
    return rows


# ---------------------------------------------------------------------------
# sweep 5: the conjugate point on the sphere
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# sweep 2e: what a curvature error costs, through the Jacobi Green's function
# ---------------------------------------------------------------------------
def sweep_curvature_sensitivity(config: ExperimentConfig) -> list[dict[str, Any]]:
    """The uncertainty budget's curvature term, against a re-integration.

    If the surface the path was flowed along has ``K + dK`` and the part has
    ``K``, the variation equation picks up a source and the prediction moves by

    ``dj(s) = -int_0^s [b(s) a(t) - a(s) b(t)] dK j(t) dt``

    -- an integral over the record's own samples, with no Wronskian in the
    denominator because ``det Phi = 1`` exactly. That is the whole reason the
    cost of a mis-fitted surface is computable downstream from a transfer
    record and nothing else.

    It is checked the only way a sensitivity can be: by actually perturbing the
    curvature and re-integrating. The residual is the ``dK^2`` term, so it has
    to fall linearly as ``dK`` shrinks, and that slope is the check -- a
    formula that was merely close would not have it.
    """
    rows: list[dict[str, Any]] = []
    grid = np.linspace(0.0, config.arc_length, 2001)
    for form in all_space_forms():
        base = constant_curvature_transfer(grid, form.K)
        record = TransferRecord(
            arclength=grid,
            gaussian_curvature=np.full_like(grid, form.K),
            a=base.a, a_rate=base.a_rate, b=base.b, b_rate=base.b_rate,
            domain="constant-curvature",
        )
        sensitivity = curvature_sensitivity(record, 0.0, 1.0)
        levels = []
        for delta in (1e-5, 1e-6, 1e-7):
            perturbed = constant_curvature_transfer(grid, form.K + delta)
            measured = perturbed.b - base.b
            predicted = delta * sensitivity
            scale = max(float(np.max(np.abs(measured))), 1e-300)
            levels.append(
                {
                    "delta_curvature": float(delta),
                    "max_abs_measured": float(np.max(np.abs(measured))),
                    "relative_residual": float(np.max(np.abs(predicted - measured))) / scale,
                }
            )
        # The order is read from the two *coarsest* perturbations. The finest
        # one is there to show where the floor is, and it is on it: at
        # dK = 1e-7 the difference of two transfer maps has cancelled seven
        # digits, so its residual is partly roundoff and fitting through it
        # would measure the subtraction rather than the formula.
        coarse, fine = levels[0], levels[1]
        ratio = coarse["relative_residual"] / max(fine["relative_residual"], 1e-300)
        order = float(
            np.log(ratio)
            / np.log(coarse["delta_curvature"] / fine["delta_curvature"])
        )
        rows.append(
            {
                "curvature": form.K,
                "curvature_label": form.label,
                "kernel": "G(s, t) = b(s) a(t) - a(s) b(t), det Phi = 1",
                "worst_relative_residual": max(
                    level["relative_residual"] for level in levels
                ),
                "residual_order_in_delta": order,
                "levels": levels,
            }
        )
    return rows


def study_conjugate_point(config: ExperimentConfig) -> dict[str, Any]:
    form = SpaceForm(1.0)
    span = config.conjugate_span_multiple * np.pi
    n_steps = config.conjugate_steps
    grid, j, j_rate = integrate_jacobi(form.K, length=span, n_steps=n_steps, method="rk4")
    # One root finder for the whole repository: the Hermite refinement in
    # TransferMap, fed the heading column this sweep already has.
    events = TransferMap(
        arc_length=grid, a=np.ones_like(grid), a_rate=np.zeros_like(grid),
        b=j, b_rate=j_rate,
    ).focus_events(component="b")
    first_zero = events[0].arc_length if events else None
    first_zero_uncertainty = events[0].location_uncertainty if events else None

    flow_grid, points, _ = integrate_geodesic(form, length=span, n_steps=n_steps, method="rk4")
    flowed_distance = form.distance(form.base_point(), points)
    true_distance = np.where(flow_grid <= np.pi, flow_grid, 2.0 * np.pi - flow_grid)
    deficit = flow_grid - flowed_distance

    past = (flow_grid > np.pi + 0.1) & (flow_grid < 2.0 * np.pi - 0.1)
    predicted_deficit = 2.0 * (flow_grid - np.pi)
    probes = [0.5 * np.pi, np.pi, 1.5 * np.pi, 1.9 * np.pi]
    probe_rows = []
    for probe in probes:
        index = int(round(probe / (span / n_steps)))
        probe_rows.append(
            {
                "arc_length": float(flow_grid[index]),
                "flowed_distance_to_start": float(flowed_distance[index]),
                "minimising": bool(deficit[index] < 1e-8),
                "length_excess": float(deficit[index]),
            }
        )

    return {
        "curvature": form.K,
        "curvature_label": form.label,
        "statement": (
            "sin(s) vanishes at s = pi; past that conjugate point the geodesic "
            "continues to solve the geodesic equation but is no longer minimising"
        ),
        "integrator": "rk4",
        "n_steps": int(n_steps),
        "h": float(span / n_steps),
        "jacobi_first_zero": first_zero,
        "jacobi_first_zero_error": None if first_zero is None else float(abs(first_zero - np.pi)),
        "jacobi_first_zero_uncertainty": first_zero_uncertainty,
        "root_finder": "cubic-Hermite, Newton with bisection safeguard",
        "max_distance_error": float(np.max(np.abs(flowed_distance - true_distance))),
        "return_to_start_distance": float(flowed_distance[-1]),
        "max_length_excess_model_error": float(
            np.max(np.abs(deficit[past] - predicted_deficit[past]))
        ),
        "probes": probe_rows,
        "curve": [
            {
                "arc_length": float(s),
                "jacobi_field": float(value),
                "flowed_distance_to_start": float(distance),
            }
            for s, value, distance in zip(
                flow_grid[:: max(1, n_steps // 200)],
                j[:: max(1, n_steps // 200)],
                flowed_distance[:: max(1, n_steps // 200)],
                strict=True,
            )
        ],
    }


def _noise_threshold(
    epsilons: np.ndarray, relative_error: np.ndarray, tolerance: float
) -> float | None:
    """Smallest perturbation resolved to ``tolerance``, and still resolved above it.

    Scanned from the large-``epsilon`` end so that a single lucky sample inside
    the noise floor cannot be mistaken for the threshold.
    """
    threshold: float | None = None
    for index in range(len(epsilons) - 1, -1, -1):
        if relative_error[index] <= tolerance:
            threshold = float(epsilons[index])
        else:
            break
    return threshold


def _lowest_decade_slope(epsilons: np.ndarray, values: np.ndarray) -> float | None:
    """Log-log slope across the smallest decade of the sweep, as a diagnostic."""
    low = epsilons <= epsilons[0] * 10.0
    if low.sum() < 2 or np.any(values[low] <= 0.0):
        return None
    slope, _ = np.polyfit(np.log10(epsilons[low]), np.log10(values[low]), 1)
    return float(slope)


def _grid_index(grid: np.ndarray, s_value: float, span: float, n_steps: int) -> int:
    """Index of ``s_value`` on a uniform grid, refusing to silently snap to a neighbour."""
    index = int(round(s_value / (span / n_steps)))
    if not (0 <= index < grid.size) or not np.isclose(grid[index], s_value, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"sample arc length {s_value} does not lie on a grid of {n_steps} steps over "
            f"[0, {span}]"
        )
    return index


def _first_positive_zero(grid: np.ndarray, values: np.ndarray) -> float | None:
    """Linearly interpolated first sign change at positive arc length."""
    for index in range(1, len(values) - 1):
        if grid[index] <= 0.0:
            continue
        left, right = values[index], values[index + 1]
        if left == 0.0:
            return float(grid[index])
        if left * right < 0.0:
            weight = left / (left - right)
            return float(grid[index] + weight * (grid[index + 1] - grid[index]))
    return None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def _check(
    identifier: str,
    statement: str,
    value: float | None,
    threshold: float,
    comparison: str = "<=",
) -> dict[str, Any]:
    if value is None or not np.isfinite(value):
        passed = False
    elif comparison == "<=":
        passed = bool(value <= threshold)
    elif comparison == ">=":
        passed = bool(value >= threshold)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown comparison {comparison!r}")
    return {
        "id": identifier,
        "statement": statement,
        "value": None if value is None else float(value),
        "threshold": float(threshold),
        "comparison": comparison,
        "passed": passed,
    }


def collect_checks(results: dict[str, Any], config: ExperimentConfig) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    for key, label in (
        ("jacobi_ode_convergence", "jacobi"),
        ("geodesic_flow_convergence", "flow"),
    ):
        for row in results[key]:
            identifier = f"{label}-order/{row['curvature_label']}/{row['integrator']}"
            if row["regime"] == "exact-to-roundoff":
                checks.append(
                    _check(
                        identifier,
                        f"{label} on {row['curvature_label']} is reproduced to roundoff by "
                        f"{row['integrator']} (linear solution, no order to measure)",
                        row["max_error_over_levels"],
                        config.exactness_threshold,
                    )
                )
            else:
                checks.append(
                    _check(
                        identifier,
                        f"measured order of {row['integrator']} on {label} "
                        f"{row['curvature_label']} matches its formal order "
                        f"{row['expected_order']}",
                        row["order_error"],
                        config.order_tolerance,
                    )
                )
                checks.append(
                    _check(
                        f"{label}-fit-quality/{row['curvature_label']}/{row['integrator']}",
                        "log-log convergence fit is clean (r^2)",
                        row["fit"]["r_squared"],
                        0.999,
                        comparison=">=",
                    )
                )

    for row in results["geodesic_flow_convergence"]:
        if row["integrator"] != "rk4":
            continue
        drift = row["finest_constraint_drift"]
        for name, value in drift.items():
            checks.append(
                _check(
                    f"flow-drift/{row['curvature_label']}/{name}",
                    f"rk4 state stays on the unit tangent bundle ({name} residual) "
                    f"at the finest step",
                    value,
                    config.drift_tolerance,
                )
            )

    for row in results["wronskian_conservation"]:
        identifier = f"wronskian/{row['curvature_label']}/{row['integrator']}"
        if row["regime"] == "exact-to-roundoff":
            checks.append(
                _check(
                    identifier,
                    f"{row['integrator']} conserves det Phi exactly on "
                    f"{row['curvature_label']}",
                    row["coarsest_drift"],
                    config.exactness_threshold,
                )
            )
        else:
            checks.append(
                _check(
                    identifier,
                    f"drift of det Phi under {row['integrator']} falls at order "
                    f"{row['expected_drift_order']}, as its one-step determinant "
                    f"{row['one_step_determinant']} says",
                    row["drift_order_error"],
                    config.order_tolerance,
                )
            )
            checks.append(
                _check(
                    f"wronskian-exact/{row['curvature_label']}/{row['integrator']}",
                    "and matches the exact value |d^n - 1| wherever that is above "
                    "roundoff, not merely its slope",
                    row["max_relative_error_vs_theory"],
                    config.wronskian_theory_tolerance,
                )
            )
    for form in all_space_forms():
        if form.K == 0.0:
            continue
        by_method = {
            row["integrator"]: row
            for row in results["wronskian_conservation"]
            if row["curvature_label"] == form.label
        }
        checks.append(
            _check(
                f"wronskian-discriminates/{form.label}",
                "the invariant has teeth: at the same finest step euler has lost "
                "it while rk4 still holds it to roundoff",
                by_method["euler"]["finest_drift"] / max(by_method["rk4"]["finest_drift"], 1e-16),
                1e6,
                comparison=">=",
            )
        )

    for row in results["first_order_validity"]:
        tag = f"{row['curvature_label']}/s={row['arc_length']:g}"
        checks.append(
            _check(
                f"variation-exponent/{tag}",
                "relative error of the first-order prediction grows like eps^2",
                abs(row["fitted_exponent"] - row["expected_exponent"]),
                config.exponent_tolerance,
            )
        )
        checks.append(
            _check(
                f"variation-coefficient/{tag}",
                "fitted leading coefficient matches cn_K(s)^2 / 24",
                row["coefficient_relative_error"],
                config.coefficient_tolerance,
            )
        )
        for entry in row["epsilon_star"]:
            checks.append(
                _check(
                    f"variation-epsilon-star/{tag}/tol={entry['tolerance']:g}",
                    "measured validity limit of the first-order prediction matches theory",
                    entry["relative_difference"],
                    config.epsilon_star_tolerance,
                )
            )
        checks.append(
            _check(
                f"flow-vs-closed-form/{tag}",
                "separation of an rk4-flowed fan matches the closed form at every "
                "perturbation, the truncation error being common-mode",
                row["flow_agreement"]["shared_step_sequence"]["max_relative_error"],
                1e-9,
            )
        )
        mixed = row["flow_agreement"]["mixed_resolution"]
        if mixed["regime"] == "exact-flow":
            checks.append(
                _check(
                    f"flow-noise-floor/{tag}",
                    "the flat geodesic flow is exact at every resolution, so "
                    "mismatched step sizes introduce no separation error",
                    mixed["max_relative_error"],
                    1e-13,
                )
            )
        else:
            checks.append(
                _check(
                    f"flow-noise-floor/{tag}",
                    "a coarse mismatched flow still resolves the largest "
                    "perturbation in the sweep",
                    mixed["relative_error_at_largest_epsilon"],
                    1e-4,
                )
            )
            checks.append(
                _check(
                    f"flow-noise-blindness/{tag}",
                    "the same coarse mismatched flow carries no information at "
                    "all about the smallest perturbation in the sweep",
                    mixed["relative_error_at_smallest_epsilon"],
                    config.noise_blindness_threshold,
                    comparison=">=",
                )
            )

    for row in results["path_sensitivity"]:
        label = row["curvature_label"]
        worst = 0.0
        for entry in row["by_arc_length"]:
            for budget in entry["tolerance_budget"]:
                round_trip = budget["max_initial_angle"] * entry["jacobi_field"]
                worst = max(
                    worst, abs(round_trip / budget["transverse_tolerance"] - 1.0)
                )
        checks.append(
            _check(
                f"sensitivity-budget/{label}",
                "an initial aiming error at the tolerance budget produces exactly "
                "the tolerated transverse separation",
                worst,
                1e-12,
            )
        )
        checks.append(
            _check(
                f"sensitivity-bench/{label}",
                "separations tabulated for the bench angles are reproduced by an "
                "independent rk4 flow",
                row["bench_predictions"]["max_flow_vs_closed_form_relative_error"],
                1e-10,
            )
        )

    for row in results["transfer_determinant"]:
        checks.append(
            _check(
                f"transfer-determinant/{row['curvature_label']}",
                "det Phi = a b' - a' b is 1 at every arc length, formed directly "
                "rather than as a product of singular values",
                row["determinant_error"],
                1e-13,
            )
        )
        checks.append(
            _check(
                f"transfer-determinant-scaled/{row['curvature_label']}",
                "and conjugating by the tolerance box leaves it alone, on boxes "
                "with aspect ratios from 1 to 10^4",
                row["scaled_determinant_error"],
                1e-13,
            )
        )
        checks.append(
            _check(
                f"transfer-amplification-floor/{row['curvature_label']}",
                "so the larger is never below 1: a path cannot be robust to every "
                "direction of starting-pose error at once",
                row["min_largest_singular_value"],
                1.0 - 1e-12,
                comparison=">=",
            )
        )

    for row in results["focus_refinement"]:
        checks.append(
            _check(
                f"focus-refinement/{row['curvature_label']}",
                "the Hermite-refined focus beats a linear one by orders of "
                "magnitude at the same sample spacing",
                row["refined_error"],
                row["linear_error"] / 100.0,
            )
        )
        checks.append(
            _check(
                f"focus-uncertainty/{row['curvature_label']}",
                "and the reported location uncertainty bounds the error it left",
                row["refined_error"],
                row["reported_uncertainty"],
            )
        )

    conjugate = results["conjugate_point"]
    for row in results["curvature_sensitivity"]:
        checks.append(
            _check(
                f"curvature-sensitivity/{row['curvature_label']}",
                "the Green's-function sensitivity to a curvature bias reproduces an "
                "actual re-integration at the perturbed curvature",
                row["worst_relative_residual"],
                1e-4,
            )
        )
        checks.append(
            _check(
                f"curvature-sensitivity-order/{row['curvature_label']}",
                "and its residual falls linearly in dK, which is what makes it the "
                "derivative rather than something merely close to it",
                abs(row["residual_order_in_delta"] - 1.0),
                0.05,
            )
        )

    checks.append(
        _check(
            "conjugate-point/jacobi-zero",
            "numerically integrated Jacobi field vanishes at s = pi",
            conjugate["jacobi_first_zero_error"],
            1e-9,
        )
    )
    checks.append(
        _check(
            "conjugate-point/distance-model",
            "flowed distance back to the start reproduces min(s, 2pi - s)",
            conjugate["max_distance_error"],
            1e-9,
        )
    )
    checks.append(
        _check(
            "conjugate-point/length-excess",
            "past the conjugate point the arc length exceeds the distance by 2(s - pi)",
            conjugate["max_length_excess_model_error"],
            1e-9,
        )
    )
    checks.append(
        _check(
            "conjugate-point/closes-at-2pi",
            "the flowed great circle closes on its starting point at s = 2pi",
            conjugate["return_to_start_distance"],
            1e-8,
        )
    )
    return checks


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def run_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    """Run every sweep and assemble the machine-readable report."""
    config = config or ExperimentConfig()
    results = {
        "jacobi_ode_convergence": sweep_jacobi_convergence(config),
        "geodesic_flow_convergence": sweep_flow_convergence(config),
        "wronskian_conservation": sweep_wronskian(config),
        "transfer_determinant": sweep_transfer_determinant(config),
        "focus_refinement": sweep_focus_refinement(config),
        "first_order_validity": sweep_first_order_validity(config),
        "path_sensitivity": sweep_path_sensitivity(config),
        "curvature_sensitivity": sweep_curvature_sensitivity(config),
        "conjugate_point": study_conjugate_point(config),
    }
    checks = collect_checks(results, config)
    core = _jsonable(
        {
            "schema": REPORT_SCHEMA,
            "supersedes": SUPERSEDES,
            "schema_changes": [
                "observation_modes: support is now per domain, not one boolean",
                "adds results.curvature_sensitivity: the uncertainty budget's "
                "curvature term, checked against a re-integration",
                f"every float is canonicalised to {CANONICAL_DIGITS} significant "
                "digits before hashing, so the content hash is reproducible "
                "across platforms",
                "transfer_determinant: the invariant is checked as det Phi "
                "directly, not as a product of singular values, which measured "
                "the SVD's conditioning rather than the flow's accuracy",
                "adds results.focus_refinement: Hermite-refined focus location "
                "against a linear one",
                "conjugate_point: adds the refined root's uncertainty and names "
                "the root finder",
            ],
            "experiment": "geodesic-flow-and-jacobi-field-testbed",
            "claim_scope": "numerical-verification-against-closed-form-solutions",
            "curvatures": [form.K for form in all_space_forms()],
            "observation_modes": observation_catalogue(),
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


def write_report(report: dict[str, Any], path) -> None:
    import pathlib

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
