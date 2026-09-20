"""The uncertainty budget: every source, its shape, and what it actually costs.

The starting pose is one term and rarely the largest. What these tests pin is
the part a per-sample variance cannot express: most of the terms are
*systematic*, one unknown felt the same way at every sample, and that is a
rank-one covariance rather than a diagonal. The distinction decides whether
averaging along the path helps at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed import constant_curvature_trace
from geodesic_testbed.boundary import GeometryUncertainty
from geodesic_testbed.engine.envelope import integrate_path
from geodesic_testbed.engine.surfaces import torus
from geodesic_testbed.engine.uncertainty import (
    Contribution,
    UncertaintyBudget,
    budget,
    calibration_transform,
    cumulative_quadrature,
    curvature_greens_function,
    curvature_sensitivity,
    fixture_datum,
    path_registration,
    sensor_noise,
    starting_pose,
    surface_reconstruction,
)

GRID = np.linspace(0.0, 2.0, 801)
BOX = (0.1, 0.0035)


def _record():
    return constant_curvature_trace(GRID, 0.7).as_transfer_record()


def _surface_record():
    return integrate_path(
        torus(200.0, 60.0), u0=0.3, v0=0.2, heading=0.6, length=240.0, n_steps=400
    ).as_transfer_record()


# -- the curvature term ----------------------------------------------------


def test_the_curvature_sensitivity_is_the_derivative_not_an_approximation() -> None:
    """Perturb K, re-integrate, and the residual falls linearly in dK."""
    curvature = 0.7
    base = constant_curvature_trace(GRID, curvature)
    sensitivity = curvature_sensitivity(base.as_transfer_record(), 0.0, 1.0)

    residuals = []
    for delta in (1e-5, 1e-6):
        measured = constant_curvature_trace(GRID, curvature + delta).angle_basis - (
            base.angle_basis
        )
        predicted = delta * sensitivity
        scale = float(np.max(np.abs(measured)))
        residuals.append(float(np.max(np.abs(predicted - measured))) / scale)

    assert residuals[0] < 1e-4
    assert residuals[0] / residuals[1] == pytest.approx(10.0, rel=0.05)


def test_the_greens_function_separates_and_is_causal() -> None:
    """Rank two before the causal mask, and zero above the diagonal after it.

    The two properties belong to different objects and it matters which. The
    separation ``b(s)a(t) - a(s)b(t)`` is what makes the double integral two
    cumulative ones; causality is imposed by the *limit* of the integral, not
    by a rank-destroying mask on a matrix nobody forms.
    """
    record = _record()
    separated = np.outer(record.b, record.a) - np.outer(record.a, record.b)
    assert np.linalg.matrix_rank(separated, tol=1e-10) == 2

    kernel = curvature_greens_function(record)
    assert float(np.max(np.abs(np.triu(kernel, k=1)))) == 0.0
    assert np.allclose(np.tril(separated), np.tril(kernel))


def test_the_separated_form_agrees_with_the_kernel_it_replaces() -> None:
    """``O(n)`` instead of ``O(n^2)``, and the same number."""
    record = _record()
    nominal = record.separation(*BOX)
    kernel = curvature_greens_function(record)
    direct = -np.array(
        [
            np.trapezoid(
                (kernel[index, : index + 1] * nominal[: index + 1]),
                record.arclength[: index + 1],
            )
            for index in range(record.arclength.size)
        ]
    )
    separated = curvature_sensitivity(record, *BOX)
    scale = float(np.max(np.abs(separated)))
    assert float(np.max(np.abs(direct - separated))) / scale < 1e-5


def test_the_cumulative_quadrature_is_fourth_order_on_a_uniform_grid() -> None:
    """Second order would put a 1e-6 floor under everything built on it."""
    errors = []
    for samples in (201, 401):
        grid = np.linspace(0.0, 1.3, samples)
        values = np.exp(grid) * np.cos(3.0 * grid)
        exact = (
            np.exp(grid) * (np.cos(3.0 * grid) + 3.0 * np.sin(3.0 * grid)) - 1.0
        ) / 10.0
        errors.append(float(np.max(np.abs(cumulative_quadrature(values, grid) - exact))))
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.2)


def test_the_quadrature_falls_back_rather_than_applying_simpson_to_unequal_steps() -> None:
    grid = np.concatenate([np.linspace(0.0, 1.0, 51), np.linspace(1.05, 2.0, 20)])
    values = np.ones_like(grid)
    assert cumulative_quadrature(values, grid)[-1] == pytest.approx(2.0, abs=1e-12)


def test_an_odd_and_an_even_sample_count_both_integrate_to_the_end() -> None:
    for samples in (100, 101):
        grid = np.linspace(0.0, 1.0, samples)
        result = cumulative_quadrature(grid**2, grid)
        assert result[-1] == pytest.approx(1.0 / 3.0, rel=1e-9)
        assert np.all(np.diff(result) > 0.0)


# -- shapes ----------------------------------------------------------------


def test_a_systematic_term_is_rank_one_and_a_noise_term_is_not() -> None:
    """One unknown against a fresh draw per sample: the whole distinction."""
    record = _record()
    systematic = calibration_transform(record, sigma=0.01)
    noise = sensor_noise(record, sigma=0.01)

    assert systematic.structure == "systematic"
    assert systematic.rank == 1
    assert noise.structure == "independent"
    assert noise.rank == record.arclength.size
    # Same per-sample sigma, completely different objects.
    assert np.allclose(systematic.sigma, noise.sigma)


def test_a_correlated_noise_term_sits_between_the_two() -> None:
    record = _record()
    white = sensor_noise(record, sigma=0.01)
    correlated = sensor_noise(record, sigma=0.01, correlation_length=0.5)
    assert correlated.structure == "correlated"
    assert np.allclose(np.diag(white.covariance), np.diag(correlated.covariance))
    assert float(np.max(np.abs(correlated.covariance - white.covariance))) > 0.0
    assert correlated.covariance[0, -1] > 0.0
    assert white.covariance[0, -1] == 0.0


def test_a_starting_pose_term_is_rank_two_because_there_are_two_unknowns() -> None:
    record = _record()
    pose = starting_pose(record, np.diag([1e-4, 1e-6]))
    assert pose.rank == 2
    assert pose.structure == "systematic"
    # And it is Phi C0 Phi^T: the transverse diagonal is a^2 sp^2 + b^2 sa^2.
    assert np.allclose(
        np.diag(pose.covariance), record.a**2 * 1e-4 + record.b**2 * 1e-6
    )


def test_a_registration_error_is_largest_where_the_prediction_is_steepest() -> None:
    """Not where it is largest, which is the point of carrying j' separately."""
    record = _record()
    contribution = path_registration(record, *BOX, sigma=0.01)
    separation = np.abs(record.separation(*BOX))
    assert int(np.argmax(contribution.sigma)) != int(np.argmax(separation))
    assert np.allclose(
        contribution.sigma, 0.01 * np.abs(record.heading_change(*BOX))
    )


# -- the budget ------------------------------------------------------------


def _budget(record):
    return budget(
        record,
        starting_pose(
            record, np.diag([(BOX[0] / 3) ** 2, (BOX[1] / 3) ** 2]),
            basis="3-sigma tolerance box",
        ),
        path_registration(record, *BOX, sigma=0.5),
        calibration_transform(record, sigma=0.01),
        sensor_noise(record, sigma=0.025, correlation_length=5.0),
    )


def test_the_total_is_a_sum_of_matrices_and_not_of_variances() -> None:
    record = _surface_record()
    assembled = _budget(record)
    assert np.allclose(
        assembled.total,
        sum(c.covariance for c in assembled.contributions),
    )
    # Off-diagonal structure survives, which a variance sum would destroy.
    assert float(np.max(np.abs(assembled.total - np.diag(np.diag(assembled.total))))) > 0.0


def test_the_breakdown_names_what_to_fix() -> None:
    record = _surface_record()
    assembled = _budget(record)
    shares = assembled.shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert assembled.dominant().name == max(shares, key=shares.get)
    assert 0.0 <= assembled.systematic_fraction <= 1.0


def test_a_budget_of_only_systematic_terms_is_singular_and_says_so() -> None:
    """A perfectly correlated error is perfectly predictable. That is not a bug."""
    record = _surface_record()
    systematic_only = budget(
        record,
        calibration_transform(record, sigma=0.01),
        path_registration(record, *BOX, sigma=0.5),
    )
    assert not systematic_only.is_positive_definite()
    assert systematic_only.systematic_fraction == pytest.approx(1.0)

    with_noise = budget(
        record,
        calibration_transform(record, sigma=0.01),
        path_registration(record, *BOX, sigma=0.5),
        sensor_noise(record, sigma=0.025),
    )
    assert with_noise.is_positive_definite()


def test_the_surface_term_reads_the_records_own_declared_uncertainty() -> None:
    """How well the surface is known belongs to the scan, not to this call."""
    record = _surface_record()

    # An analytic surface really does contribute zero here, and the term says
    # so rather than being left out -- a budget that omits a term and a budget
    # that shows it as zero are different documents.
    exact = surface_reconstruction(record, *BOX)
    assert exact.basis == "analytic"
    assert float(np.max(exact.sigma)) == 0.0

    scanned = record.with_geometry_uncertainty(
        GeometryUncertainty(
            position=0.05, normal=1e-4, curvature=2e-7, basis="as-built-scan"
        )
    )
    contribution = surface_reconstruction(scanned, *BOX)
    assert contribution.basis == "as-built-scan"
    assert contribution.rank == 1
    assert float(np.max(contribution.sigma)) > 0.0


def test_a_record_that_declares_no_geometry_uncertainty_refuses_to_guess_one() -> None:
    record = _surface_record().with_geometry_uncertainty(
        GeometryUncertainty.not_declared("the scan has not been processed yet")
    )
    with pytest.raises(ValueError, match="declares no geometry uncertainty"):
        surface_reconstruction(record, *BOX)


def test_a_fixture_offset_propagates_exactly_as_a_starting_pose_error_does() -> None:
    record = _surface_record()
    fixture = fixture_datum(record, lateral_sigma=0.02, heading_sigma=1e-4)
    equivalent = starting_pose(record, np.diag([0.02**2, 1e-8]))
    assert np.allclose(fixture.covariance, equivalent.covariance)
    assert fixture.kind == "fixture-datum"


# -- what a budget refuses -------------------------------------------------


def test_a_contribution_must_say_where_its_number_came_from() -> None:
    record = _record()
    with pytest.raises(ValueError, match="on what basis"):
        Contribution(
            kind="other", name="a guess",
            covariance=np.eye(record.arclength.size),
            structure="independent", basis="",
        )


def test_an_empty_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="not a budget"):
        UncertaintyBudget(arclength=GRID, contributions=())


def test_two_contributions_may_not_share_a_name() -> None:
    record = _record()
    with pytest.raises(ValueError, match="share a name"):
        budget(
            record,
            sensor_noise(record, sigma=0.01, name="noise"),
            sensor_noise(record, sigma=0.02, name="noise"),
        )


def test_every_term_must_be_on_the_same_path() -> None:
    record = _record()
    other = constant_curvature_trace(np.linspace(0.0, 2.0, 401), 0.7).as_transfer_record()
    with pytest.raises(ValueError, match="every term must be on the same path"):
        budget(record, sensor_noise(other, sigma=0.01))


def test_a_negative_definite_contribution_is_refused() -> None:
    record = _record()
    with pytest.raises(ValueError, match="positive semi-definite"):
        Contribution(
            kind="other", name="wrong sign",
            covariance=-np.eye(record.arclength.size),
            structure="independent", basis="declared",
        )
