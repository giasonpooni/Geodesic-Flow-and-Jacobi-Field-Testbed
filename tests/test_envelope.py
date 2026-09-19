"""The sensitivity envelope, and the two independent routes to a Jacobi field."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed.engine.envelope import (
    finite_difference_jacobi,
    integrate_path,
    integrate_paths,
    scan_headings,
)
from geodesic_testbed.engine.surfaces import cylinder, plane, pseudosphere, sphere, torus

ANCHORS = [
    ("plane", plane(), 0.0, 0.0, lambda s: s),
    ("cylinder", cylinder(1.0), 0.0, 0.0, lambda s: s),
    ("sphere", sphere(1.0), np.pi / 2, 0.0, np.sin),
    ("pseudosphere", pseudosphere(), 1.2, 0.0, np.sinh),
]


@pytest.mark.parametrize("name,surface,u0,v0,reference", ANCHORS, ids=[a[0] for a in ANCHORS])
def test_the_general_solver_recovers_the_closed_form(name, surface, u0, v0, reference) -> None:
    envelope = integrate_path(surface, u0=u0, v0=v0, heading=0.6, length=1.5, n_steps=1500)
    assert np.max(np.abs(envelope.jacobi_field - reference(envelope.arc_length))) < 1e-11
    assert np.max(envelope.speed_drift) < 1e-12


def test_a_rolled_sheet_has_exactly_the_sensitivity_of_the_flat_one() -> None:
    """Intrinsic curvature is what matters; bending a sheet does not change it."""
    flat = integrate_path(plane(), u0=0.0, v0=0.0, heading=0.6, length=2.0, n_steps=800)
    rolled = integrate_path(cylinder(1.0), u0=0.0, v0=0.0, heading=0.6, length=2.0, n_steps=800)
    assert np.allclose(flat.jacobi_field, rolled.jacobi_field, atol=1e-13)
    assert flat.amplification[-1] == pytest.approx(1.0, abs=1e-13)


def test_the_two_routes_agree_to_second_order_in_the_step() -> None:
    surface = torus(2.0, 1.0)
    envelope = integrate_path(surface, u0=0.3, v0=0.2, heading=0.6, length=1.0, n_steps=1000)
    reference = abs(float(envelope.jacobi_field[-1]))
    deviations = []
    for epsilon in (1e-2, 5e-3, 2.5e-3):
        _, measured = finite_difference_jacobi(
            surface, u0=0.3, v0=0.2, heading=0.6, epsilon=epsilon, length=1.0, n_steps=1000
        )
        deviations.append(abs(float(measured[-1]) / reference - 1.0))
    assert deviations[0] / deviations[1] == pytest.approx(4.0, rel=0.02)
    assert deviations[1] / deviations[2] == pytest.approx(4.0, rel=0.02)


def test_the_chord_excess_on_a_cylinder_matches_its_normal_curvature() -> None:
    """The eps^2 coefficient is ``(cn^2 + kappa_n^2 sn^2)/6``, kappa_n transverse.

    On a unit cylinder at heading ``theta`` the transverse normal curvature is
    ``sin^2(theta)``, and the surface is intrinsically flat, so at ``s = 1``
    the coefficient must be ``(1 + sin^4(theta)) / 6``.
    """
    heading = 0.6
    surface = cylinder(1.0)
    envelope = integrate_path(
        surface, u0=0.0, v0=0.0, heading=heading, length=1.0, n_steps=1000
    )
    reference = abs(float(envelope.jacobi_field[-1]))
    epsilon = 1e-2
    _, measured = finite_difference_jacobi(
        surface, u0=0.0, v0=0.0, heading=heading, epsilon=epsilon, length=1.0, n_steps=1000
    )
    coefficient = abs(float(measured[-1]) / reference - 1.0) / epsilon**2
    expected = (1.0 + np.sin(heading) ** 4) / 6.0
    assert coefficient == pytest.approx(expected, rel=2e-3)


def test_the_chord_is_exact_on_a_plate_and_on_a_sphere() -> None:
    epsilons = np.array([1e-3, 1e-2, 1e-1])
    for surface, u0, reference in ((plane(), 0.0, 1.0), (sphere(1.0), np.pi / 2, np.sin(1.0))):
        _, measured = finite_difference_jacobi(
            surface, u0=u0, v0=0.0, heading=0.6, epsilon=epsilons, length=1.0, n_steps=1000
        )
        exact = reference * np.sin(epsilons) / epsilons
        assert np.allclose(measured[-1], exact, rtol=1e-9)


def test_a_focus_is_found_where_the_field_vanishes_and_nowhere_else() -> None:
    focused = integrate_path(
        sphere(1.0), u0=np.pi / 2, v0=0.0, heading=0.6, length=4.0, n_steps=4000
    )
    assert focused.focus_points()[0] == pytest.approx(np.pi, abs=1e-8)
    assert not focused.summary()["well_conditioned"]

    spreading = integrate_path(
        pseudosphere(), u0=1.2, v0=0.0, heading=0.6, length=1.5, n_steps=1500
    )
    assert spreading.focus_points() == []
    assert spreading.summary()["well_conditioned"]


def test_the_tolerance_budget_is_the_inverse_of_the_worst_deviation() -> None:
    envelope = integrate_path(
        sphere(1.0), u0=np.pi / 2, v0=0.0, heading=0.6, length=2.0, n_steps=800
    )
    tolerance = 1e-3
    budget = envelope.heading_budget(tolerance)
    assert np.all(budget[1:] * envelope.worst_so_far[1:] == pytest.approx(tolerance))
    # The cumulative budget can only tighten as the path gets longer.
    assert np.all(np.diff(budget[1:]) <= 1e-18)
    pointwise = envelope.heading_budget(tolerance, cumulative=False)
    assert np.all(pointwise[1:] >= budget[1:] - 1e-18)


def test_the_envelope_half_width_is_linear_in_the_tolerance() -> None:
    envelope = integrate_path(torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=400)
    assert np.allclose(
        envelope.transverse_envelope(2e-3), 2.0 * envelope.transverse_envelope(1e-3)
    )


def test_a_fan_matches_the_paths_flowed_one_at_a_time() -> None:
    surface = torus(2.0, 1.0)
    headings = [0.0, 0.5, 1.2]
    fan = integrate_paths(
        surface, u0=0.3, v0=0.2, headings=headings, length=1.0, n_steps=200
    )
    for envelope, heading in zip(fan, headings, strict=True):
        single = integrate_path(
            surface, u0=0.3, v0=0.2, heading=heading, length=1.0, n_steps=200
        )
        assert np.array_equal(envelope.jacobi_field, single.jacobi_field)


def test_the_heading_scan_is_ranked_and_finds_a_real_spread() -> None:
    rows = scan_headings(
        torus(2.0, 1.0),
        u0=0.3,
        v0=0.2,
        headings=np.linspace(0.0, np.pi, 8, endpoint=False),
        length=6.0,
        n_steps=1500,
    )
    worst = [row["max_forward_amplification"] for row in rows]
    assert worst == sorted(worst)
    assert worst[-1] / worst[0] > 2.0


def test_the_scan_reports_the_focus_that_low_amplification_can_hide() -> None:
    """Cheap forward separation is sometimes bought with an ill-conditioned map."""
    rows = scan_headings(
        torus(2.0, 1.0),
        u0=0.3,
        v0=0.2,
        headings=np.linspace(0.0, np.pi, 16, endpoint=False),
        length=6.0,
        n_steps=1500,
    )
    assert rows[0]["passes_a_focus"] is True
    assert rows[0]["focus_margin"] < 1e-2
    assert rows[-1]["passes_a_focus"] is False
    clear = [row for row in rows if not row["passes_a_focus"]]
    assert clear, "some heading must be clear of a focus for the trade-off to exist"
    assert min(row["max_forward_amplification"] for row in clear) > rows[0][
        "max_forward_amplification"
    ]
    for row in rows:
        assert row["max_wronskian_drift"] < 1e-9
