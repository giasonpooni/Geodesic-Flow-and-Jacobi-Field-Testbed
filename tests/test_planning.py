"""Choosing a route without collapsing the reasons into one number first.

What is pinned here: that accumulated observability is a different question
from sampled resolvability, that it is scale invariant in the only form that
can be, that boundaries are computed from the declared part rather than handed
in on a grid that might not match, that offsetting the start point is a
genuinely different family from fanning the heading, and that the answer is a
front whose domination claims are true.
"""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed.engine.envelope import integrate_path, integrate_paths
from geodesic_testbed.engine.observation_model import ObservationModel
from geodesic_testbed.engine.planning import (
    ChartBoundary,
    Objective,
    ObstacleDiscs,
    combined_clearance,
    heading_fan,
    observability_gramian,
    offset_courses,
    pareto_front,
    route_label,
    stacked_observability,
    weighted_cost,
)
from geodesic_testbed.engine.surfaces import plane, pseudosphere, sphere, torus

SCANNER = ObservationModel.transverse_only(
    2.5e-5, mode="ambient-euclidean-chord", calibration_id="declared-example"
)
BOX = {"max_lateral": 1e-3, "max_heading": 1e-3}


def _record(surface=None, **overrides):
    settings = {
        "u0": 0.3, "v0": 0.2, "heading": 0.6, "length": 2.0, "n_steps": 400
    } | overrides
    return integrate_path(surface or torus(2.0, 1.0), **settings).as_transfer_record(
        observation_mode="ambient-euclidean-chord"
    )


# -- the Gramian -----------------------------------------------------------


def test_the_gramian_density_is_scale_invariant_and_the_total_is_not() -> None:
    """Twice the path really does carry twice the information. Per length, none."""
    densities, totals = [], []
    for factor in (1.0, 2.0):
        record = _record(
            sphere(factor), u0=np.pi / 2, v0=0.0, length=3.0 * factor, n_steps=600
        )
        gramian = observability_gramian(
            record,
            ObservationModel.transverse_only(
                2.5e-5 * factor, mode="ambient-euclidean-chord"
            ),
            max_lateral=1e-3 * factor, max_heading=1e-3,
        )
        densities.append(np.linalg.eigvalsh(gramian.per_unit_length))
        totals.append(gramian.eigenvalues)

    assert np.allclose(densities[0], densities[1], rtol=1e-12, atol=0.0)
    assert np.allclose(totals[1], 2.0 * totals[0], rtol=1e-12, atol=0.0)


def test_information_only_accumulates() -> None:
    """``W(s2) - W(s1)`` is positive semi-definite, sample by sample."""
    gramian = observability_gramian(_record(), SCANNER, **BOX)
    increments = np.diff(gramian.cumulative, axis=0)
    assert float(np.min(np.linalg.eigvalsh(increments))) > -1e-12


def test_the_gramian_names_the_direction_the_path_says_least_about() -> None:
    gramian = observability_gramian(_record(), SCANNER, **BOX)
    values = gramian.eigenvalues
    assert values[0] <= values[-1]
    direction = gramian.worst_direction
    assert float(np.linalg.norm(direction)) == pytest.approx(1.0)
    quadratic = float(direction @ gramian.total @ direction)
    assert quadratic == pytest.approx(values[0], rel=1e-9)


def test_accumulated_observability_is_not_sampled_resolvability() -> None:
    """A route can be resolvable everywhere and still say little about one direction."""
    record = _record(length=6.0, n_steps=1200)
    gramian = observability_gramian(record, SCANNER, **BOX)
    rho = SCANNER.resolvability(record, np.diag([1e-6, 1e-6]))
    profile = np.min(rho, axis=1) if rho.ndim > 1 else rho

    # The Gramian is anisotropic even where rho is comfortably above zero.
    assert float(np.min(profile[1:])) > 0.0
    assert gramian.eigenvalues[-1] / gramian.eigenvalues[0] > 2.0


def test_a_gramian_needs_exactly_one_declared_scaling() -> None:
    """Unscaled, its entries do not share units and cannot be ranked by."""
    record = _record()
    with pytest.raises(ValueError, match="exactly one scaling"):
        observability_gramian(record, SCANNER)
    with pytest.raises(ValueError, match="exactly one scaling"):
        observability_gramian(
            record, SCANNER, max_lateral=1e-3, max_heading=1e-3,
            prior_covariance=np.eye(2) * 1e-6,
        )


def test_a_prior_scaled_gramian_asks_a_different_question() -> None:
    record = _record()
    box = observability_gramian(record, SCANNER, **BOX)
    prior = observability_gramian(
        record, SCANNER, prior_covariance=np.diag([4e-6, 1e-6])
    )
    assert box.scale_basis == "tolerance-box"
    assert prior.scale_basis == "prior-covariance"
    assert not np.allclose(box.total, prior.total)


def test_a_gramian_mixing_observation_modes_is_refused() -> None:
    record = _record()
    with pytest.raises(ValueError, match="weighting one quantity"):
        observability_gramian(
            record,
            ObservationModel.transverse_only(
                2.5e-5, mode="intrinsic-surface-distance"
            ),
            **BOX,
        )


# -- boundaries ------------------------------------------------------------


def test_the_chart_boundary_is_computed_from_the_declared_domain() -> None:
    surface = pseudosphere()
    envelope = integrate_path(surface, u0=1.0, v0=0.0, heading=0.3, length=1.0, n_steps=200)
    clearance = ChartBoundary(surface).clearance(envelope)

    assert clearance.shape == envelope.arc_length.shape
    assert np.all(clearance > 0.0)
    # The path runs towards the u edge, so the clearance has to change.
    assert float(np.ptp(clearance)) > 0.0


def test_a_chart_with_no_edges_contributes_no_constraint() -> None:
    """Both of the torus's coordinates wrap, so there is nothing to clear."""
    surface = torus(2.0, 1.0)
    envelope = integrate_path(surface, u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=200)
    assert np.all(np.isinf(ChartBoundary(surface).clearance(envelope)))


def test_an_obstacle_clearance_is_a_lower_bound_and_errs_the_safe_way() -> None:
    """The ambient chord is never longer than the in-surface distance."""
    surface = sphere(1.0)
    envelope = integrate_path(
        surface, u0=np.pi / 2, v0=0.0, heading=0.6, length=2.0, n_steps=200
    )
    centre = envelope.points[100] + np.array([0.0, 0.0, 0.5])
    discs = ObstacleDiscs(centres=centre[None, :], radii=[0.1])
    clearance = discs.clearance(envelope)
    chord = np.linalg.norm(envelope.points - centre, axis=-1)
    assert np.allclose(clearance, chord - 0.1)
    assert float(np.min(clearance)) < float(np.min(chord))


def test_the_tightest_declared_region_wins() -> None:
    surface = pseudosphere()
    envelope = integrate_path(surface, u0=1.0, v0=0.0, heading=0.3, length=1.0, n_steps=200)
    boundary = ChartBoundary(surface)
    discs = ObstacleDiscs(
        centres=envelope.points[[50]] + np.array([[0.0, 0.0, 0.2]]), radii=[0.05]
    )
    combined = combined_clearance(envelope, boundary, discs)
    assert np.all(combined <= boundary.clearance(envelope) + 1e-12)
    assert np.all(combined <= discs.clearance(envelope) + 1e-12)


def test_a_clearance_with_no_declared_region_is_not_a_constraint() -> None:
    envelope = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=1.0, n_steps=100
    )
    with pytest.raises(ValueError, match="not a constraint"):
        combined_clearance(envelope)


def test_an_obstacle_needs_one_radius_each() -> None:
    with pytest.raises(ValueError, match="exactly one radius"):
        ObstacleDiscs(centres=np.zeros((3, 3)), radii=[0.1, 0.2])


# -- generating routes -----------------------------------------------------


def test_a_route_label_is_built_from_declared_values_only() -> None:
    """The bug this exists to prevent already happened once, at 97.5 degrees.

    Two properties, and the second is why the index is there. A label must not
    move when a value moves in its last bits -- two headings a machine cannot
    tell apart get the same text. And a label must still be unique when that
    happens, because it keys a dict that lands in a committed report; the index
    is what guarantees that, whatever the formatting does.
    """
    assert route_label("fan", 3, 97.5) == route_label("fan", 3, 97.49999999999999)
    assert route_label("fan", 3, 97.5) != route_label("fan", 4, 97.5)
    assert route_label("fan", 3, 97.5) != route_label("course", 3, 97.5)
    assert "97.5" in route_label("fan", 3, 97.5)


def test_a_route_label_never_carries_a_computed_start_point() -> None:
    """A course's start is reached by flowing, so it is not stable enough to key on."""
    surface = torus(2.0, 1.0)
    courses = offset_courses(
        surface, u0=0.3, v0=0.2, heading=0.6, spacing=0.1, count=5,
        length=1.0, n_steps=100,
    )
    for label, envelope in courses.items():
        assert f"{envelope.start[0]:.3f}" not in label
        assert f"{envelope.start[1]:.3f}" not in label
    # The offsets are (i - (n-1)/2) * spacing: exact small integers against a
    # declared spacing, and identical on every machine.
    assert set(courses) == {
        route_label("course", index, offset)
        for index, offset in enumerate((-0.2, -0.1, 0.0, 0.1, 0.2))
    }


def test_a_heading_fan_shares_a_start_and_an_offset_family_does_not() -> None:
    surface = torus(2.0, 1.0)
    fan = heading_fan(surface, u0=0.3, v0=0.2, count=6, length=2.0, n_steps=200)
    courses = offset_courses(
        surface, u0=0.3, v0=0.2, heading=0.6, spacing=0.1, count=5,
        length=2.0, n_steps=200,
    )
    assert len({envelope.start[:2] for envelope in fan.values()}) == 1
    assert len({envelope.start[:2] for envelope in courses.values()}) == 5
    assert not set(fan) & set(courses)


def test_parallel_courses_have_different_heading_coordinates_on_a_curved_part() -> None:
    """They are parallel by transport, and the coordinate angle is not the test.

    Their heading *coordinates* differ, and the spread is the holonomy of the
    perpendicular each one was carried along. A family built by giving every
    course the same heading number would not be parallel at all -- it would be
    a fan with the starts moved.
    """
    curved = offset_courses(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, spacing=0.1, count=5,
        length=1.0, n_steps=200,
    )
    headings = sorted(envelope.start[2] for envelope in curved.values())
    assert len(set(headings)) == 5
    assert headings[-1] - headings[0] > 1e-3

    flat = offset_courses(
        plane(), u0=0.0, v0=0.0, heading=0.6, spacing=0.1, count=5,
        length=1.0, n_steps=200,
    )
    flat_headings = [envelope.start[2] for envelope in flat.values()]
    assert float(np.ptp(flat_headings)) < 1e-12


def test_offset_courses_are_spaced_by_a_length_on_the_part() -> None:
    """Not by a step in u and v, which would depend on the parameterisation."""
    surface = torus(2.0, 1.0)
    courses = offset_courses(
        surface, u0=0.3, v0=0.2, heading=0.6, spacing=0.1, count=3,
        length=1.0, n_steps=100,
    )
    starts = np.array([envelope.points[0] for envelope in courses.values()])
    gaps = np.linalg.norm(np.diff(starts, axis=0), axis=-1)
    assert np.allclose(gaps, 0.1, rtol=2e-3)


def test_the_offset_family_exercises_the_column_a_fan_does_not() -> None:
    """A lateral placement error is the ``a`` column, and a fan never moves it."""
    surface = plane()
    courses = offset_courses(
        surface, u0=0.0, v0=0.0, heading=0.6, spacing=0.2, count=3,
        length=1.0, n_steps=100,
    )
    starts = [envelope.start[:2] for envelope in courses.values()]
    assert len(set(starts)) == 3


def test_a_fan_of_one_and_a_course_of_none_are_refused() -> None:
    surface = torus(2.0, 1.0)
    with pytest.raises(ValueError, match="at least two headings"):
        heading_fan(surface, u0=0.3, v0=0.2, count=1, length=1.0, n_steps=50)
    with pytest.raises(ValueError, match="spacing must be positive"):
        offset_courses(
            surface, u0=0.3, v0=0.2, heading=0.6, spacing=0.0, count=3,
            length=1.0, n_steps=50,
        )


# -- the front -------------------------------------------------------------


OBJECTIVES = (
    Objective("amplification", "lower-is-better", limit=2.0),
    Objective("observed", "higher-is-better", limit=1000.0),
    Objective("length", "lower-is-better", limit=3.0),
)


def test_a_route_beaten_on_everything_is_dominated_and_the_rest_are_not() -> None:
    routes = {
        "a": {"amplification": 1.2, "observed": 2000.0, "length": 2.0},
        "b": {"amplification": 1.9, "observed": 3000.0, "length": 2.0},
        "c": {"amplification": 1.5, "observed": 1500.0, "length": 2.5},
    }
    result = pareto_front(routes, OBJECTIVES)
    assert set(result["front"]) == {"a", "b"}
    assert result["dominated"] == {"c": "a"}


def test_the_front_keeps_a_route_that_wins_on_one_objective_only() -> None:
    """Which is the whole point: the trade-off is reported, not resolved."""
    routes = {
        "cheap": {"amplification": 1.0, "observed": 100.0, "length": 2.9},
        "visible": {"amplification": 1.99, "observed": 9000.0, "length": 2.9},
    }
    result = pareto_front(routes, OBJECTIVES)
    assert set(result["front"]) == {"cheap", "visible"}
    assert not result["dominated"]


def test_a_route_missing_an_objective_is_not_ranked_as_though_it_passed() -> None:
    routes = {
        "a": {"amplification": 1.2, "observed": 2000.0, "length": 2.0},
        "b": {"amplification": 1.0},
    }
    with pytest.raises(ValueError, match="do not report every declared objective"):
        pareto_front(routes, OBJECTIVES)


def test_a_front_with_no_declared_objectives_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one declared objective"):
        pareto_front({"a": {}}, ())


def test_collapsing_the_front_needs_weights_that_were_declared() -> None:
    values = {"amplification": 1.2, "observed": 2000.0, "length": 2.0}
    cost = weighted_cost(
        values, OBJECTIVES,
        {"amplification": 0.5, "observed": 0.3, "length": 0.2},
    )
    assert cost["terms"]["amplification"] == pytest.approx(0.6)
    assert cost["terms"]["observed"] == pytest.approx(0.5)
    assert cost["cost"] == pytest.approx(0.5 * 0.6 + 0.3 * 0.5 + 0.2 * (2.0 / 3.0))


def test_an_objective_left_out_of_the_weights_was_zeroed_by_accident() -> None:
    values = {"amplification": 1.2, "observed": 2000.0, "length": 2.0}
    with pytest.raises(ValueError, match="by accident"):
        weighted_cost(values, OBJECTIVES, {"amplification": 0.5, "observed": 0.5})


def test_weights_that_do_not_sum_to_one_hide_a_scale_in_a_preference() -> None:
    values = {"amplification": 1.2, "observed": 2000.0, "length": 2.0}
    with pytest.raises(ValueError, match="rather than 1"):
        weighted_cost(
            values, OBJECTIVES,
            {"amplification": 1.0, "observed": 1.0, "length": 1.0},
        )


def test_an_objective_with_no_limit_can_be_ranked_but_not_weighted() -> None:
    """The limit is what makes the ratio dimensionless."""
    unlimited = Objective("amplification", "lower-is-better")
    routes = {"a": {"amplification": 1.0}, "b": {"amplification": 2.0}}
    assert pareto_front(routes, (unlimited,))["front"] == ["a"]
    with pytest.raises(ValueError, match="declares no limit"):
        weighted_cost({"amplification": 1.0}, (unlimited,), {"amplification": 1.0})


def test_a_negative_weight_turns_an_objective_into_its_opposite() -> None:
    values = {"amplification": 1.2, "observed": 2000.0, "length": 2.0}
    with pytest.raises(ValueError, match="its opposite"):
        weighted_cost(
            values, OBJECTIVES,
            {"amplification": -0.5, "observed": 1.0, "length": 0.5},
        )


# -- the two forms of the Gramian ------------------------------------------


def _gramian_record(n_steps: int = 200):
    from geodesic_testbed.engine.contract import Units

    envelope = integrate_paths(
        sphere(1.0), u0=np.pi / 2, v0=0.0, headings=[0.5], length=1.4, n_steps=n_steps
    )[0]
    return envelope.as_transfer_record(
        units=Units(length="metre", angle="radian"),
        observation_mode="ambient-euclidean-chord",
    )


def test_the_condition_number_says_what_the_total_information_does_not() -> None:
    """A path can accumulate a great deal and still be nearly blind one way."""
    from geodesic_testbed.engine.observation_model import ObservationModel

    record = _gramian_record()
    model = ObservationModel.transverse_only(5e-5, mode="ambient-euclidean-chord")
    gramian = observability_gramian(
        record, model, max_lateral=2e-4, max_heading=np.deg2rad(0.1)
    )
    values = gramian.eigenvalues
    assert gramian.condition_number == pytest.approx(values[-1] / values[0])
    assert gramian.condition_number > 1.0
    assert gramian.to_dict()["condition_number"] == gramian.condition_number


def test_information_over_an_interval_is_interpolated_rather_than_snapped() -> None:
    """Two windows differing by less than a sample spacing must differ.

    An interval quantised to the grid is known only to the sample spacing, and
    a planner comparing two candidate acquisition windows would be comparing
    their rounding as much as their information.
    """
    from geodesic_testbed.engine.observation_model import ObservationModel

    record = _gramian_record(n_steps=100)
    model = ObservationModel.transverse_only(5e-5, mode="ambient-euclidean-chord")
    gramian = observability_gramian(
        record, model, max_lateral=2e-4, max_heading=np.deg2rad(0.1)
    )
    spacing = float(record.arclength[1] - record.arclength[0])

    whole = gramian.over(float(record.arclength[0]), float(record.arclength[-1]))
    assert np.allclose(whole, gramian.total, rtol=1e-12, atol=0.0)

    near = gramian.over(0.2, 0.7)
    nudged = gramian.over(0.2, 0.7 + 0.3 * spacing)
    assert not np.allclose(near, nudged, rtol=1e-9, atol=0.0)

    first = gramian.over(0.0, 0.7)
    second = gramian.over(0.7, float(record.arclength[-1]))
    assert np.allclose(first + second, gramian.total, rtol=1e-10, atol=0.0)

    with pytest.raises(ValueError, match="end > start"):
        gramian.over(0.7, 0.2)
    with pytest.raises(ValueError, match="not inside the path"):
        gramian.over(0.0, 99.0)


def test_the_stacked_gramian_is_the_integral_one_divided_by_the_spacing() -> None:
    """Two different objects with a declared conversion between them.

    The integral form treats ``R`` as a noise density; the stacked form treats
    it as the covariance of the measurements actually taken. On a uniform grid
    with a stationary ``R`` they differ by the sample spacing, and a campaign
    comparing two sampling rates without that factor is comparing nothing.

    The agreement is first order in the spacing, not exact: a sum over the
    samples is a rectangle rule, and it differs from the integral by the end
    correction. So the claim is that the discrepancy *halves when the sampling
    doubles*, which is a statement about the relationship, where a fixed
    tolerance would only be a statement about one grid.
    """
    from geodesic_testbed.engine.observation_model import ObservationModel
    from geodesic_testbed.engine.output_covariance import NoiseModel

    model = ObservationModel.transverse_only(5e-5, mode="ambient-euclidean-chord")
    noise = NoiseModel.from_observation_model(model, basis="bench characterisation")
    ratios = []
    for n_steps in (200, 400, 800):
        record = _gramian_record(n_steps=n_steps)
        spacing = float(record.arclength[1] - record.arclength[0])
        integral = observability_gramian(
            record, model, max_lateral=2e-4, max_heading=np.deg2rad(0.1)
        )
        stacked = stacked_observability(
            record, model, noise, max_lateral=2e-4, max_heading=np.deg2rad(0.1)
        )
        ratios.append(float(integral.total[0, 0] / (spacing * stacked.total[0, 0])))
    gaps = [abs(ratio - 1.0) for ratio in ratios]
    assert gaps[-1] < 2e-3, f"the two forms disagree by {gaps[-1]:.3e} at 800 samples"
    for coarse, fine in zip(gaps, gaps[1:], strict=False):
        assert coarse / fine == pytest.approx(2.0, rel=0.05), (
            f"the discrepancy should halve with the spacing; {coarse:.3e} -> {fine:.3e}"
        )


def test_a_correlated_noise_carries_less_information_than_an_independent_one() -> None:
    """The case the integral form cannot express, and gets wrong by assuming away.

    A filter makes neighbouring samples averages of each other. Ranking routes
    by the integral Gramian on filtered data counts information that was never
    collected; the stacked form with the full ``R`` does not.
    """
    from geodesic_testbed.engine.observation_model import ObservationModel
    from geodesic_testbed.engine.output_covariance import NoiseModel

    record = _gramian_record(n_steps=120)
    model = ObservationModel.transverse_only(5e-5, mode="ambient-euclidean-chord")
    n = record.arclength.size
    variance = 5e-5**2

    independent = NoiseModel.stationary(
        np.array([[variance]]), ("transverse",), basis="bench characterisation"
    )
    lag = np.exp(-np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / 8.0)
    correlated = NoiseModel(
        blocks=variance * lag,
        structure="correlated",
        outputs=("transverse",),
        basis="a declared filter group delay",
    )

    box = {"max_lateral": 2e-4, "max_heading": np.deg2rad(0.1)}
    white = stacked_observability(record, model, independent, **box)
    smoothed = stacked_observability(record, model, correlated, **box)

    assert np.trace(smoothed.total) < 0.5 * np.trace(white.total), (
        "correlated noise has to carry visibly less information, or the full R "
        "is not reaching the calculation"
    )
    assert smoothed.noise_structure == "correlated"
    assert white.noise_structure == "stationary"


def test_a_window_inverts_the_submatrix_of_R_and_not_a_submatrix_of_its_inverse() -> None:
    """Different matrices whenever the noise is correlated, and only one is usable.

    The information the measurements in a window carry *on their own* is what
    a planner can act on before the others have been taken. A submatrix of
    ``R^-1`` is what they carry *given* the rest, which is a different and
    unobtainable quantity.
    """
    from geodesic_testbed.engine.observation_model import ObservationModel
    from geodesic_testbed.engine.output_covariance import NoiseModel, stacked_operator

    record = _gramian_record(n_steps=120)
    model = ObservationModel.transverse_only(5e-5, mode="ambient-euclidean-chord")
    n = record.arclength.size
    lag = np.exp(-np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / 8.0)
    correlated = NoiseModel(
        blocks=5e-5**2 * lag,
        structure="correlated",
        outputs=("transverse",),
        basis="a declared filter group delay",
    )
    box = {"max_lateral": 2e-4, "max_heading": np.deg2rad(0.1)}
    window = (0.3, 0.9)
    windowed = stacked_observability(record, model, correlated, window=window, **box)

    grid = record.arclength
    inside = np.flatnonzero((grid >= window[0]) & (grid <= window[1]))
    assert windowed.samples == inside.size
    scale = np.diag([2e-4, np.deg2rad(0.1)])
    operator = stacked_operator(record, model)[inside]
    sub = correlated.stacked(n)[np.ix_(inside, inside)]
    expected = scale.T @ (operator.T @ np.linalg.inv(sub) @ operator) @ scale
    assert np.allclose(windowed.total, expected, rtol=1e-9, atol=0.0)

    wrong = scale.T @ (
        operator.T @ np.linalg.inv(correlated.stacked(n))[np.ix_(inside, inside)] @ operator
    ) @ scale
    assert not np.allclose(windowed.total, wrong, rtol=1e-3, atol=0.0), (
        "the two readings have to differ, or the distinction is not being made"
    )


def test_the_stacked_gramian_refuses_a_bare_array_for_R() -> None:
    from geodesic_testbed.engine.observation_model import ObservationModel

    record = _gramian_record(n_steps=60)
    model = ObservationModel.transverse_only(5e-5, mode="ambient-euclidean-chord")
    with pytest.raises(TypeError, match="pass a NoiseModel"):
        stacked_observability(
            record, model, np.array([[1e-9]]), max_lateral=2e-4, max_heading=0.01
        )


def test_the_stacked_gramian_needs_exactly_one_declared_scaling() -> None:
    from geodesic_testbed.engine.observation_model import ObservationModel
    from geodesic_testbed.engine.output_covariance import NoiseModel

    record = _gramian_record(n_steps=60)
    model = ObservationModel.transverse_only(5e-5, mode="ambient-euclidean-chord")
    noise = NoiseModel.from_observation_model(model, basis="bench characterisation")
    with pytest.raises(ValueError, match="exactly one scaling"):
        stacked_observability(record, model, noise)
    with pytest.raises(ValueError, match="exactly one scaling"):
        stacked_observability(
            record, model, noise, max_lateral=1e-4, max_heading=1e-3,
            prior_covariance=np.diag([1e-8, 1e-6]),
        )
