"""The two routes to a Jacobi field, and their agreement."""

from __future__ import annotations

import numpy as np
import pytest

from geojac.jacobi import (
    geodesic_position_error,
    integrate_geodesic,
    integrate_geodesic_bundle,
    integrate_jacobi,
    jacobi_reference,
    numerical_separation,
)
from geojac.spaceforms import SpaceForm, all_space_forms

FORMS = all_space_forms()
IDS = [form.name for form in FORMS]


@pytest.mark.parametrize("form", FORMS, ids=IDS)
def test_solver_reproduces_the_closed_form_jacobi_field(form: SpaceForm) -> None:
    grid, field, derivative = integrate_jacobi(form.K, length=2.0, n_steps=2000)
    assert np.allclose(field, jacobi_reference(grid, form.K), atol=1e-12)
    assert derivative[0] == pytest.approx(1.0)


@pytest.mark.parametrize("form", FORMS, ids=IDS)
def test_flowed_geodesic_reproduces_the_exponential_map(form: SpaceForm) -> None:
    grid, points, velocities = integrate_geodesic(
        form, direction_angle=0.4, length=2.0, n_steps=2000
    )
    assert np.max(geodesic_position_error(form, grid, points, direction_angle=0.4)) < 1e-12
    residuals = form.constraint_residuals(points, velocities)
    assert max(float(np.max(value)) for value in residuals.values()) < 1e-12


@pytest.mark.parametrize("form", FORMS, ids=IDS)
def test_finite_separation_converges_to_the_first_order_prediction(form: SpaceForm) -> None:
    """The point of the whole exercise: eps j(s) is the limit, not the answer."""
    s = 1.5
    ratios = []
    for epsilon in (1e-2, 1e-3, 1e-4):
        grid, distance = numerical_separation(form, epsilon, length=s, n_steps=1500)
        assert grid[-1] == pytest.approx(s)
        ratios.append(float(distance[-1] / (epsilon * jacobi_reference(s, form.K))))
    # Each tenfold reduction in eps should cut the deficit by a hundred.
    deficits = [abs(ratio - 1.0) for ratio in ratios]
    assert deficits[0] > deficits[1] > deficits[2]
    assert deficits[0] / deficits[1] == pytest.approx(100.0, rel=0.05)
    assert deficits[1] / deficits[2] == pytest.approx(100.0, rel=0.05)


@pytest.mark.parametrize("form", FORMS, ids=IDS)
def test_flowed_separation_agrees_with_the_closed_form(form: SpaceForm) -> None:
    epsilon = 3e-3
    _, distance = numerical_separation(form, epsilon, length=2.0, n_steps=2000)
    exact = form.exact_separation(2.0, epsilon)
    assert distance[-1] == pytest.approx(float(exact), rel=1e-10)


def test_bundle_matches_separately_flowed_geodesics() -> None:
    form = SpaceForm(-1.0)
    angles = [0.0, 0.05, 0.2]
    _, points, _ = integrate_geodesic_bundle(form, angles, length=1.0, n_steps=64)
    for index, angle in enumerate(angles):
        _, single, _ = integrate_geodesic(
            form, direction_angle=angle, length=1.0, n_steps=64
        )
        assert np.array_equal(points[:, index, :], single)


def test_the_sphere_refocuses_at_the_conjugate_point() -> None:
    """Every geodesic leaving a point on the sphere meets again at s = pi."""
    form = SpaceForm(1.0)
    grid, distance = numerical_separation(form, 0.5, length=np.pi, n_steps=6000)
    assert grid[-1] == pytest.approx(np.pi)
    assert distance[-1] < 1e-11
    assert distance[len(distance) // 2] > 0.4
