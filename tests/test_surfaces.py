# SPDX-License-Identifier: MPL-2.0
"""Parametric surfaces: the differential geometry, before any path is flowed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from geodesic_testbed.engine.surfaces import (
    CATALOGUE,
    Chart,
    ParametricSurface,
    built_in,
    christoffel,
    cylinder,
    first_fundamental_form,
    gaussian_curvature,
    hyperbolic_paraboloid,
    plane,
    pseudosphere,
    sphere,
    torus,
    unit_normal,
)

SAMPLE_POINTS = [(0.3, 0.2), (0.9, -0.4), (1.2, 1.1)]


def _surfaces() -> list[ParametricSurface]:
    return [factory() for factory in CATALOGUE.values()]


@pytest.mark.parametrize("surface", _surfaces(), ids=lambda s: s.name)
@pytest.mark.parametrize("point", SAMPLE_POINTS)
def test_curvature_from_the_jet_matches_the_analytic_value(surface, point) -> None:
    u, v = point
    measured = float(surface.gaussian_curvature(u, v))
    expected = float(surface.exact_curvature(u, v))
    assert measured == pytest.approx(expected, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("surface", _surfaces(), ids=lambda s: s.name)
def test_finite_difference_derivatives_recover_the_analytic_ones(surface) -> None:
    numeric = replace(surface, jet=None)
    for u, v in SAMPLE_POINTS:
        assert float(numeric.gaussian_curvature(u, v)) == pytest.approx(
            float(surface.gaussian_curvature(u, v)), abs=1e-6
        )


@pytest.mark.parametrize("surface", _surfaces(), ids=lambda s: s.name)
@pytest.mark.parametrize("heading", [0.0, 0.7, 2.4, -1.3])
def test_unit_direction_really_has_unit_speed(surface, heading) -> None:
    u, v = 0.9, 0.4
    du, dv = surface.unit_direction(u, v, heading)
    assert float(surface.speed(u, v, du, dv)) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("surface", _surfaces(), ids=lambda s: s.name)
def test_heading_is_an_angle_in_the_surface_not_in_the_chart(surface) -> None:
    """Two headings differing by ``theta`` really meet at ``theta`` on the surface."""
    u, v = 0.9, 0.4
    E, F, G = surface.first_fundamental_form(u, v)
    first = surface.unit_direction(u, v, 0.0)
    second = surface.unit_direction(u, v, 0.8)
    inner = (
        E * first[0] * second[0]
        + F * (first[0] * second[1] + first[1] * second[0])
        + G * first[1] * second[1]
    )
    assert float(inner) == pytest.approx(np.cos(0.8), abs=1e-12)


@pytest.mark.parametrize("surface", _surfaces(), ids=lambda s: s.name)
def test_the_normal_is_a_unit_vector_orthogonal_to_the_surface(surface) -> None:
    jet = surface.jet_at(0.9, 0.4)
    normal = unit_normal(jet)
    assert float(np.dot(normal, normal)) == pytest.approx(1.0, abs=1e-14)
    assert float(np.dot(normal, jet.ru)) == pytest.approx(0.0, abs=1e-14)
    assert float(np.dot(normal, jet.rv)) == pytest.approx(0.0, abs=1e-14)


def test_flat_surfaces_have_no_christoffel_symbols_in_these_charts() -> None:
    for surface in (plane(), cylinder(1.0)):
        symbols = christoffel(surface.jet_at(0.7, -0.2))
        assert np.allclose(np.asarray(symbols, dtype=float), 0.0, atol=1e-14)


def test_the_sphere_metric_is_the_textbook_one() -> None:
    surface = sphere(2.0)
    E, F, G = first_fundamental_form(surface.jet_at(0.7, 0.3))
    assert float(E) == pytest.approx(4.0)
    assert float(F) == pytest.approx(0.0, abs=1e-15)
    assert float(G) == pytest.approx(4.0 * np.sin(0.7) ** 2)


def test_curvature_scales_with_the_radius() -> None:
    for radius in (0.5, 1.0, 3.0):
        assert float(sphere(radius).gaussian_curvature(1.0, 0.5)) == pytest.approx(
            1.0 / radius**2, rel=1e-12
        )


def test_the_torus_changes_the_sign_of_its_curvature() -> None:
    surface = torus(2.0, 1.0)
    assert float(surface.gaussian_curvature(0.0, 0.0)) > 0.0  # outer equator
    assert float(surface.gaussian_curvature(np.pi, 0.0)) < 0.0  # inner equator
    assert float(surface.gaussian_curvature(np.pi / 2, 0.0)) == pytest.approx(0.0, abs=1e-15)


def test_the_saddle_is_flattest_far_from_its_centre() -> None:
    surface = hyperbolic_paraboloid(1.0)
    centre = float(surface.gaussian_curvature(0.0, 0.0))
    away = float(surface.gaussian_curvature(2.0, 2.0))
    assert centre == pytest.approx(-1.0)
    assert away == pytest.approx(-1.0 / (1.0 + 8.0) ** 2)  # K = -1/(1 + u^2 + v^2)^2
    assert abs(away) < abs(centre) / 50.0


def test_the_pseudosphere_really_is_constant_negative() -> None:
    surface = pseudosphere()
    values = [float(surface.gaussian_curvature(u, 0.3)) for u in (0.5, 1.0, 2.0, 3.0)]
    assert values == pytest.approx([-1.0] * 4, abs=1e-12)


def test_gaussian_curvature_is_independent_of_how_the_normal_is_scaled() -> None:
    """The implementation skips normalising the normal; that must not change K."""
    surface = torus(2.0, 1.0)
    jet = surface.jet_at(0.9, 0.4)
    normal = unit_normal(jet)
    E, F, G = first_fundamental_form(jet)
    L = float(np.dot(jet.ruu, normal))
    M = float(np.dot(jet.ruv, normal))
    N = float(np.dot(jet.rvv, normal))
    textbook = (L * N - M * M) / float(E * G - F * F)
    assert float(gaussian_curvature(jet)) == pytest.approx(textbook, rel=1e-12)


def test_unknown_surface_is_refused() -> None:
    assert set(CATALOGUE) == {"plane", "cylinder", "sphere", "pseudosphere", "saddle", "torus"}
    with pytest.raises(KeyError):
        built_in("klein-bottle")


def test_chart_orthogonality_survives_an_anisotropic_reparameterisation() -> None:
    """The defect that retired ``sqrt(EG - F^2) / max(E, G)``.

    Running one coordinate three times as fast leaves the surface, and the
    regularity of the chart, untouched. The retired criterion falls by exactly
    that factor; orthogonality does not move.
    """
    base = torus(2.0, 1.0)
    stretched = ParametricSurface(
        name="torus with v running three times as fast",
        position=lambda u, v: base.position(u, 3.0 * v),
        chart=Chart(u_min=-10.0, u_max=10.0, v_min=-10.0, v_max=10.0),
    )
    u = np.linspace(0.1, 1.2, 17)
    v = np.linspace(0.1, 1.2, 17)

    def retired(surface: ParametricSurface) -> float:
        E, F, G = surface.first_fundamental_form(u, v)
        return float(np.min(np.sqrt(E * G - F**2) / np.maximum(E, G)))

    assert retired(base) / retired(stretched) == pytest.approx(3.0, rel=1e-6)
    assert float(np.min(base.chart_orthogonality(u, v))) == pytest.approx(1.0, abs=1e-9)
    assert float(np.min(stretched.chart_orthogonality(u, v))) == pytest.approx(1.0, abs=1e-9)
