"""Route selection from declared process limits, not from a chosen number."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed import (
    AcquisitionSpec,
    CoverageSpec,
    ObservationModel,
    RouteConstraints,
    assess_route,
    rank_routes,
)
from geodesic_testbed.engine.envelope import integrate_paths
from geodesic_testbed.engine.surfaces import torus

SIGMA_ALPHA = np.deg2rad(0.1)
SIGMA_MEASUREMENT = 25e-6 / 0.3
HEADING_ONLY = np.diag([0.0, SIGMA_ALPHA**2])
TOLERANCE = {"max_lateral": 1e-3, "max_heading": 1e-3}


def _candidates(count: int = 12):
    headings = np.linspace(0.0, np.pi, count, endpoint=False)
    envelopes = integrate_paths(
        torus(2.0, 1.0), u0=0.3, v0=0.2, headings=headings, length=6.0, n_steps=1500
    )
    return {
        f"{np.rad2deg(h):.0f}deg": e for h, e in zip(headings, envelopes, strict=True)
    }


def test_a_route_reports_every_declared_limit_and_which_one_binds() -> None:
    candidates = _candidates()
    constraints = RouteConstraints(
        max_cross_track_error=2e-2,
        max_heading_error=5e-2,
        min_coverage_margin=0.0,
        coverage=CoverageSpec(nominal_spacing=0.10, swath_width=0.14),
    )
    assessment = assess_route(
        candidates["90deg"], constraints=constraints, label="90deg", **TOLERANCE
    )
    names = {margin.name for margin in assessment.margins}
    assert names == {"cross-track-error", "heading-error", "coverage-margin"}
    assert assessment.binding_constraint in names
    tightest = min(assessment.margins, key=lambda m: m.margin)
    assert assessment.binding_constraint == tightest.name
    assert assessment.feasible == all(m.satisfied for m in assessment.margins)


def test_an_unconstrained_limit_is_simply_absent() -> None:
    assessment = assess_route(
        _candidates()["90deg"], constraints=RouteConstraints(), label="x", **TOLERANCE
    )
    assert assessment.margins == []
    assert assessment.feasible is True
    assert assessment.binding_constraint is None


def test_infeasible_routes_are_kept_and_ordered_by_how_badly_they_miss() -> None:
    constraints = RouteConstraints(max_cross_track_error=5e-3)
    ranked = rank_routes(_candidates(), constraints=constraints, **TOLERANCE)
    feasible = [a for a in ranked if a.feasible]
    infeasible = [a for a in ranked if not a.feasible]
    assert ranked[: len(feasible)] == feasible, "feasible routes come first"
    misses = [-min(m.margin for m in a.margins) for a in infeasible]
    assert misses == sorted(misses)


def test_tightening_a_limit_can_only_shrink_the_feasible_set() -> None:
    candidates = _candidates()
    loose = rank_routes(
        candidates, constraints=RouteConstraints(max_cross_track_error=5e-2), **TOLERANCE
    )
    tight = rank_routes(
        candidates, constraints=RouteConstraints(max_cross_track_error=5e-3), **TOLERANCE
    )
    loose_set = {a.label for a in loose if a.feasible}
    tight_set = {a.label for a in tight if a.feasible}
    assert tight_set <= loose_set


SCHEDULE = AcquisitionSpec(
    acquire_threshold=5.0,
    hold_threshold=3.0,
    acquisition_window=0.05,
    max_acquisition_distance=1.0,
    min_tracked_distance=3.0,
    max_loss_distance=0.1,
)


def test_the_tracking_constraint_uses_the_instrument_not_a_threshold_on_b() -> None:
    candidates = _candidates()
    model = ObservationModel.transverse_only(SIGMA_MEASUREMENT)
    without = rank_routes(
        candidates,
        constraints=RouteConstraints(max_cross_track_error=2e-2),
        **TOLERANCE,
    )
    with_tracking = rank_routes(
        candidates,
        constraints=RouteConstraints(max_cross_track_error=2e-2, acquisition=SCHEDULE),
        observation=model,
        initial_covariance=HEADING_ONLY,
        **TOLERANCE,
    )
    assert all(a.tracking is None for a in without)
    assert all(a.tracking is not None for a in with_tracking)
    assert {a.label for a in with_tracking if a.feasible} <= {
        a.label for a in without if a.feasible
    }
    # The routes the instrument cannot follow are the ones that pass a focus.
    for assessment in with_tracking:
        if assessment.crossed_first_conjugate_point:
            assert assessment.tracking["outcome"] == "TRACK_LOST"


def test_a_route_acquired_at_its_last_sample_does_not_pass_vacuously() -> None:
    """The failure a "minimum after first crossing" rule would have allowed."""
    candidates = _candidates()
    model = ObservationModel.transverse_only(SIGMA_MEASUREMENT)
    late = AcquisitionSpec(
        acquire_threshold=5.0, hold_threshold=3.0, min_tracked_distance=5.9
    )
    ranked = rank_routes(
        candidates,
        constraints=RouteConstraints(acquisition=late),
        observation=model,
        initial_covariance=HEADING_ONLY,
        **TOLERANCE,
    )
    outcomes = {a.tracking["outcome"] for a in ranked}
    assert outcomes - {"TRACKED"}, "some route must fail the tracked-distance floor"


def test_a_coverage_margin_without_a_coverage_spec_is_refused() -> None:
    with pytest.raises(ValueError):
        RouteConstraints(min_coverage_margin=0.0)


def test_boundary_clearance_comes_from_the_caller() -> None:
    candidates = _candidates()
    envelope = candidates["90deg"]
    clearance = np.full(envelope.arc_length.shape, 0.5)
    clearance[100:] = 0.01
    assessment = assess_route(
        envelope,
        constraints=RouteConstraints(min_boundary_clearance=0.1),
        boundary_clearance=clearance,
        label="90deg",
        **TOLERANCE,
    )
    assert not assessment.feasible
    assert assessment.binding_constraint == "boundary-clearance"
