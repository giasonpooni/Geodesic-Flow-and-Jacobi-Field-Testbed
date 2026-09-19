"""Integrator tests, on a problem with nothing to do with geometry."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed.engine.analysis import fit_power_law
from geodesic_testbed.engine.integrators import (
    DEFAULT_INTEGRATORS,
    get_integrator,
    integrate,
    step_ladder,
)


def exponential_rhs(_s: float, y: np.ndarray) -> np.ndarray:
    return y


@pytest.mark.parametrize("method", DEFAULT_INTEGRATORS)
def test_measured_order_matches_the_formal_order(method: str) -> None:
    counts = step_ladder(8, 7)
    steps, errors = [], []
    for n_steps in counts:
        grid, trajectory = integrate(
            exponential_rhs, np.array([1.0]), length=1.0, n_steps=n_steps, method=method
        )
        errors.append(float(abs(trajectory[-1, 0] - np.exp(grid[-1]))))
        steps.append(1.0 / n_steps)
    fit = fit_power_law(steps, errors, y_floor=1e-13)
    assert fit.exponent == pytest.approx(get_integrator(method).order, abs=0.05)
    assert fit.r_squared > 0.999


@pytest.mark.parametrize("method", DEFAULT_INTEGRATORS)
def test_batched_integration_matches_one_trajectory_at_a_time(method: str) -> None:
    starts = np.array([[1.0], [2.0], [-0.5]])
    _, batched = integrate(
        exponential_rhs, starts, length=0.7, n_steps=13, method=method
    )
    for index in range(starts.shape[0]):
        _, single = integrate(
            exponential_rhs, starts[index], length=0.7, n_steps=13, method=method
        )
        assert np.array_equal(batched[:, index, :], single)


def test_grid_is_rebuilt_from_the_index_and_so_does_not_drift() -> None:
    grid, _ = integrate(exponential_rhs, np.array([1.0]), length=1.0, n_steps=3000)
    assert grid[-1] == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(np.diff(grid), 1.0 / 3000, atol=1e-18)


def test_step_ladder_is_geometric() -> None:
    assert step_ladder(10, 4) == (10, 20, 40, 80)


def test_a_zero_step_integration_is_refused() -> None:
    with pytest.raises(ValueError):
        integrate(exponential_rhs, np.array([1.0]), length=1.0, n_steps=0)


def test_unknown_integrator_is_refused() -> None:
    with pytest.raises(KeyError):
        get_integrator("verlet")
