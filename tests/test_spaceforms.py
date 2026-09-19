"""Geometry tests.

Where it is possible the expected value is computed a second, independent way
-- by the textbook law of cosines, by a finite-difference of the defining ODE,
or from the hyperboloid constraint -- rather than by calling the same helper
the test is meant to check.
"""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed.engine.spaceforms import (
    SUPPORTED_CURVATURES,
    SpaceForm,
    all_space_forms,
    asin_k,
    conjugate_distance,
    cos_k,
    sin_k,
)

ARCS = np.linspace(0.0, 2.0, 41)


@pytest.mark.parametrize("form", all_space_forms(), ids=lambda f: f.name)
def test_reference_solution_solves_the_jacobi_equation(form: SpaceForm) -> None:
    step = 1e-4
    s = np.linspace(0.2, 2.0, 25)
    second = (sin_k(s + step, form.K) - 2 * sin_k(s, form.K) + sin_k(s - step, form.K)) / step**2
    assert np.max(np.abs(second + form.K * sin_k(s, form.K))) < 1e-6


@pytest.mark.parametrize("form", all_space_forms(), ids=lambda f: f.name)
def test_generalised_pythagoras(form: SpaceForm) -> None:
    identity = cos_k(ARCS, form.K) ** 2 + form.K * sin_k(ARCS, form.K) ** 2
    assert np.allclose(identity, 1.0, atol=1e-13)


@pytest.mark.parametrize("form", all_space_forms(), ids=lambda f: f.name)
def test_asin_k_inverts_sin_k(form: SpaceForm) -> None:
    values = np.linspace(0.0, 0.9, 19)
    assert np.allclose(sin_k(asin_k(values, form.K), form.K), values, atol=1e-13)


@pytest.mark.parametrize("form", all_space_forms(), ids=lambda f: f.name)
def test_exponential_map_stays_on_the_unit_tangent_bundle(form: SpaceForm) -> None:
    p0 = form.base_point()
    v0 = form.rotated_direction(0.7)
    points, velocities = form.exp(p0, v0, ARCS)
    residuals = form.constraint_residuals(points, velocities)
    for name, value in residuals.items():
        assert np.max(value) < 1e-12, name


@pytest.mark.parametrize("form", all_space_forms(), ids=lambda f: f.name)
def test_distance_along_a_geodesic_is_its_arc_length(form: SpaceForm) -> None:
    p0 = form.base_point()
    points, _ = form.exp(p0, form.rotated_direction(-0.3), ARCS)
    assert np.allclose(form.distance(p0, points), ARCS, atol=1e-13)


def test_sphere_distance_is_accurate_at_both_ends_of_its_range() -> None:
    """arcsin saturates near the antipode; the arctan2 form used here must not."""
    form = SpaceForm(1.0)
    p0 = form.base_point()
    arcs = np.array([1e-12, 1e-8, 1.0, np.pi - 1e-8, np.pi])
    points, _ = form.exp(p0, form.rotated_direction(0.0), arcs)
    measured = form.distance(p0, points)
    assert np.allclose(measured / np.maximum(arcs, 1e-300), 1.0, rtol=1e-7)


def test_exact_separation_matches_the_textbook_laws_of_cosines() -> None:
    epsilon = 0.37
    arcs = np.linspace(0.1, 1.4, 14)

    plane = SpaceForm(0.0)
    assert np.allclose(
        plane.exact_separation(arcs, epsilon), 2.0 * arcs * np.sin(epsilon / 2.0), atol=1e-14
    )

    sphere = SpaceForm(1.0)
    spherical = np.arccos(np.cos(arcs) ** 2 + np.sin(arcs) ** 2 * np.cos(epsilon))
    assert np.allclose(sphere.exact_separation(arcs, epsilon), spherical, atol=1e-12)

    hyperbolic = SpaceForm(-1.0)
    hyperbolic_law = np.arccosh(np.cosh(arcs) ** 2 - np.sinh(arcs) ** 2 * np.cos(epsilon))
    assert np.allclose(hyperbolic.exact_separation(arcs, epsilon), hyperbolic_law, atol=1e-12)


@pytest.mark.parametrize("form", all_space_forms(), ids=lambda f: f.name)
def test_leading_correction_to_the_first_order_prediction(form: SpaceForm) -> None:
    """d = eps sn_K(s) [1 - cn_K(s)^2 eps^2 / 24 + O(eps^4)]."""
    s = 1.3
    epsilon = 1e-3
    exact = form.exact_separation(s, epsilon)
    first_order = form.first_order_separation(s, epsilon)
    measured = (1.0 - exact / first_order) / epsilon**2
    assert measured == pytest.approx(form.relative_deviation_coefficient(s), rel=1e-5)


@pytest.mark.parametrize("form", all_space_forms(), ids=lambda f: f.name)
def test_epsilon_for_relative_tolerance_round_trips(form: SpaceForm) -> None:
    tolerance = 1e-6
    s = 0.9
    epsilon = float(form.epsilon_for_relative_tolerance(s, tolerance))
    exact = form.exact_separation(s, epsilon)
    first_order = form.first_order_separation(s, epsilon)
    assert abs(1.0 - exact / first_order) == pytest.approx(tolerance, rel=1e-3)


def test_hyperbolic_difference_can_stop_being_spacelike() -> None:
    """Two points pushed off the hyperboloid have no distance left between them."""
    form = SpaceForm(-1.0)
    on_surface = form.base_point()
    drifted = on_surface + np.array([0.0, 0.0, 1e-3])
    assert form.chord_squared(on_surface, drifted) < 0.0
    assert form.distance(on_surface, drifted) == 0.0


def test_conjugate_distance_only_exists_on_the_sphere() -> None:
    assert conjugate_distance(1.0) == pytest.approx(np.pi)
    assert conjugate_distance(0.0) is None
    assert conjugate_distance(-1.0) is None
    assert sin_k(conjugate_distance(1.0), 1.0) == pytest.approx(0.0, abs=1e-15)


def test_only_the_three_model_curvatures_are_accepted() -> None:
    assert SUPPORTED_CURVATURES == (0.0, 1.0, -1.0)
    with pytest.raises(ValueError):
        SpaceForm(0.5)
