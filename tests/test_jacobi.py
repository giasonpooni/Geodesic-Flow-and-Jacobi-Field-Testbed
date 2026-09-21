# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed import (
    constant_curvature_trace,
    finite_angular_separation,
    integrate_jacobi,
)


@pytest.mark.parametrize("curvature", [0.0, 1.0, -1.0, 0.25, -0.25])
def test_rk4_matches_constant_curvature_solution(curvature: float) -> None:
    grid = np.linspace(0.0, 2.0, 401)
    exact = constant_curvature_trace(grid, curvature)
    numerical = integrate_jacobi(grid, curvature)
    assert np.max(np.abs(numerical.position_basis - exact.position_basis)) < 2.0e-10
    assert np.max(np.abs(numerical.angle_basis - exact.angle_basis)) < 2.0e-10


def test_variable_curvature_is_supported() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    trace = integrate_jacobi(grid, lambda s: 0.25 + 0.1 * s)
    assert trace.position_basis[0] == 1.0
    assert trace.angle_basis[0] == 0.0
    assert trace.position_rate[0] == 0.0
    assert trace.angle_rate[0] == 1.0
    assert np.all(np.isfinite(trace.angle_basis))


@pytest.mark.parametrize("curvature", [0.0, 1.0, -1.0])
def test_finite_angular_separation_converges_to_jacobi_prediction(curvature: float) -> None:
    grid = np.linspace(0.0, 1.0, 101)
    basis = np.abs(constant_curvature_trace(grid, curvature).angle_basis)
    coarse_delta = 1.0e-1
    fine_delta = 1.0e-2
    coarse = finite_angular_separation(grid, curvature, coarse_delta) / coarse_delta
    fine = finite_angular_separation(grid, curvature, fine_delta) / fine_delta
    coarse_error = float(np.max(np.abs(coarse - basis)))
    fine_error = float(np.max(np.abs(fine - basis)))
    assert fine_error < coarse_error / 50.0


def test_invalid_arclength_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="start at zero"):
        constant_curvature_trace([0.1, 0.2], 0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        integrate_jacobi([0.0, 0.2, 0.2], 0.0)
