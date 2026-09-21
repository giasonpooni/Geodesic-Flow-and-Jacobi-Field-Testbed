# SPDX-License-Identifier: MPL-2.0
"""The 2x2 transfer map and its invariant."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed.engine.integrators import integrate
from geodesic_testbed.engine.transfer import (
    INITIAL_STATE,
    TransferMap,
    constant_curvature_transfer,
    transfer_from_trajectory,
    transfer_rhs,
)

GRID = np.linspace(0.0, 2.0, 41)
CURVATURES = [0.0, 1.0, -1.0, 4.0, -0.25, 0.09]


@pytest.mark.parametrize("K", CURVATURES)
def test_closed_form_starts_as_the_identity(K: float) -> None:
    phi = constant_curvature_transfer(GRID, K)
    assert phi.a[0] == pytest.approx(1.0)
    assert phi.a_rate[0] == pytest.approx(0.0, abs=1e-15)
    assert phi.b[0] == pytest.approx(0.0, abs=1e-15)
    assert phi.b_rate[0] == pytest.approx(1.0)
    assert np.allclose(phi.matrices()[0], np.eye(2), atol=1e-15)


@pytest.mark.parametrize("K", CURVATURES)
def test_the_wronskian_is_one_for_the_closed_form(K: float) -> None:
    assert np.max(constant_curvature_transfer(GRID, K).wronskian_drift) < 1e-14


@pytest.mark.parametrize("K", CURVATURES)
def test_both_columns_solve_the_jacobi_equation(K: float) -> None:
    step = 1e-4
    s = np.linspace(0.3, 1.7, 15)
    for column in ("a", "b"):

        def sample(x, column=column):
            return getattr(constant_curvature_transfer(x, K), column)

        second = (sample(s + step) - 2.0 * sample(s) + sample(s - step)) / step**2
        assert np.max(np.abs(second + K * sample(s))) < 1e-6


@pytest.mark.parametrize("K", CURVATURES)
def test_rk4_reproduces_the_closed_form_transfer_map(K: float) -> None:
    grid, trajectory = integrate(
        transfer_rhs(K), np.asarray(INITIAL_STATE, float), length=2.0, n_steps=4000
    )
    numeric = transfer_from_trajectory(grid, trajectory)
    exact = constant_curvature_transfer(grid, K)
    for column in ("a", "a_rate", "b", "b_rate"):
        assert np.allclose(getattr(numeric, column), getattr(exact, column), atol=1e-11)
    assert np.max(numeric.wronskian_drift) < 1e-13


def test_a_varying_curvature_profile_is_accepted() -> None:
    phi = transfer_from_trajectory(
        *integrate(
            transfer_rhs(lambda s: 1.0 + 0.5 * s),
            np.asarray(INITIAL_STATE, float),
            length=1.0,
            n_steps=2000,
        )
    )
    assert np.max(phi.wronskian_drift) < 1e-13


def test_propagation_is_linear_and_matches_the_matrix_product() -> None:
    phi = constant_curvature_transfer(GRID, -1.0)
    lateral, heading = 3e-3, -7e-4
    offset, angle = phi.propagate(lateral, heading)
    stacked = phi.matrices() @ np.array([lateral, heading])
    assert np.allclose(offset, stacked[:, 0])
    assert np.allclose(angle, stacked[:, 1])
    doubled = phi.propagate(2 * lateral, 2 * heading)[0]
    assert np.allclose(doubled, 2.0 * offset)


def test_the_worst_case_box_bound_is_attained_at_a_corner() -> None:
    phi = constant_curvature_transfer(GRID, 1.0)
    lateral, heading = 2e-3, 5e-4
    bound = phi.worst_case_offset(lateral, heading)
    corners = [
        np.abs(phi.propagate(sx * lateral, sy * heading)[0])
        for sx in (1, -1)
        for sy in (1, -1)
    ]
    assert np.allclose(bound, np.max(corners, axis=0))
    assert np.all(phi.rss_offset(lateral, heading) <= bound + 1e-15)


def test_covariance_propagation_is_congruent_and_stays_positive() -> None:
    phi = constant_curvature_transfer(GRID, -1.0)
    covariance = np.array([[4e-6, 1e-7], [1e-7, 9e-8]])
    propagated = phi.propagate_covariance(covariance)
    assert propagated.shape == (GRID.size, 2, 2)
    assert np.allclose(propagated[0], covariance)
    assert np.allclose(propagated, np.swapaxes(propagated, -1, -2))
    # det Phi = 1, so a congruence by Phi preserves the determinant exactly.
    assert np.allclose(np.linalg.det(propagated), np.linalg.det(covariance), rtol=1e-10)
    assert np.all(np.linalg.eigvalsh(propagated) > 0.0)
    # The lateral variance is the worst-case box bound squared only at a corner;
    # here it must at least match the first row of Phi C Phi^T.
    assert np.allclose(propagated[:, 0, 0], phi.a**2 * covariance[0, 0]
                       + 2.0 * phi.a * phi.b * covariance[0, 1]
                       + phi.b**2 * covariance[1, 1])


def test_a_bad_covariance_is_refused() -> None:
    phi = constant_curvature_transfer(GRID, 0.0)
    with pytest.raises(ValueError):
        phi.propagate_covariance(np.eye(3))
    with pytest.raises(ValueError):
        phi.propagate_covariance(np.array([[1.0, 2.0], [3.0, 1.0]]))


def test_focus_points_separate_the_two_columns() -> None:
    """On the unit sphere b vanishes at pi and a at pi/2; they are not the same event."""
    grid = np.linspace(0.0, 4.0, 4001)
    phi = constant_curvature_transfer(grid, 1.0)
    assert phi.focus_points(component="b")[0] == pytest.approx(np.pi, abs=1e-6)
    assert phi.focus_points(component="a")[0] == pytest.approx(np.pi / 2, abs=1e-6)
    flat = constant_curvature_transfer(grid, 0.0)
    assert flat.focus_points(component="a") == []
    assert flat.focus_points(component="b") == []


def test_an_arbitrary_map_reports_its_own_determinant() -> None:
    phi = TransferMap(
        arc_length=np.array([0.0, 1.0]),
        a=np.array([1.0, 2.0]),
        a_rate=np.array([0.0, 1.0]),
        b=np.array([0.0, 1.0]),
        b_rate=np.array([1.0, 1.5]),
    )
    assert np.allclose(phi.determinant, [1.0, 2.0 * 1.5 - 1.0 * 1.0])
