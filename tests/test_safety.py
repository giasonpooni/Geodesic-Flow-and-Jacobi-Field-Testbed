# SPDX-License-Identifier: MPL-2.0
"""The ways this runtime is allowed to fail, and the ways it is not.

Each test here pins a failure mode that produced a *plausible number* rather
than an error: samples integrated through a degenerate metric, a resolvability
minimum taken over the stretch where there was no track, an invariant checked
through a decomposition whose own conditioning dominated the answer, a limit
declared as zero, an evidence array of the wrong length. None of them raised;
all of them reported.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from geodesic_testbed import constant_curvature_trace
from geodesic_testbed.boundary import StartingCovariance, Units
from geodesic_testbed.engine.canonical import CANONICAL_DIGITS, canonical_float, jsonable
from geodesic_testbed.engine.contract import validated_covariance
from geodesic_testbed.engine.envelope import integrate_path, integrate_paths
from geodesic_testbed.engine.integrators import integrate_guarded
from geodesic_testbed.engine.observation import require_available
from geodesic_testbed.engine.observation_model import ObservationModel
from geodesic_testbed.engine.record import TransferRecord
from geodesic_testbed.engine.routing import RouteConstraints, assess_route
from geodesic_testbed.engine.surfaces import pseudosphere, torus
from geodesic_testbed.engine.tracking import TRACK_LOST, AcquisitionSpec, evaluate_tracking

#: A start on the pseudosphere whose geodesics run off the chart at three very
#: different arc lengths, so a fan exercises truncation per path.
EXITING = {"u0": 0.6, "v0": 0.0, "length": 6.0, "n_steps": 600}


# -- a path that leaves its chart -----------------------------------------


def test_integration_stops_at_the_chart_edge_instead_of_running_past_it() -> None:
    surface = pseudosphere()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        envelope = integrate_path(surface, heading=1.2, **EXITING)

    assert envelope.chart["truncated"] is True
    assert envelope.arc_length[-1] < EXITING["length"]
    assert np.all(np.isfinite(envelope.jacobi_field))
    assert np.all(np.isfinite(envelope.curvature))
    # Every sample that survived is inside the chart, so the envelope in hand
    # is valid; what it is not is the path that was asked for.
    assert envelope.chart_valid
    assert envelope.chart["truncation_reason"]


def test_each_path_in_a_fan_is_truncated_on_its_own() -> None:
    """One heading running off the edge says nothing about the others."""
    fan = integrate_paths(pseudosphere(), headings=[0.0, 1.2, 2.5], **EXITING)
    lengths = [float(envelope.arc_length[-1]) for envelope in fan]
    assert lengths[0] > lengths[1] > lengths[2]
    assert all(envelope.chart["truncated"] for envelope in fan)


def test_a_path_that_stays_inside_its_chart_is_untouched() -> None:
    fan = integrate_paths(
        torus(2.0, 1.0), u0=0.3, v0=0.2, headings=np.linspace(0.0, 1.0, 5),
        length=3.0, n_steps=300,
    )
    for envelope in fan:
        assert envelope.chart_valid
        assert envelope.chart["truncated"] is False
        assert envelope.arc_length.size == 301


def test_the_raise_policy_refuses_a_path_that_leaves_the_chart() -> None:
    with pytest.raises(ValueError, match="left the chart"):
        integrate_path(pseudosphere(), heading=1.2, on_chart_exit="raise", **EXITING)


def test_an_unknown_chart_exit_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="on_chart_exit"):
        integrate_path(torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.5,
                       length=1.0, n_steps=50, on_chart_exit="carry on")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_invalid_samples_cannot_become_a_record() -> None:
    """A record carries no chart, so it cannot warn a consumer. It must not exist.

    The warnings are filtered because this is the one policy that integrates
    past the exit on purpose, and the overflow it produces there is the
    demonstration, not an accident -- it is what the other policies exist to
    avoid.
    """
    reported = integrate_path(pseudosphere(), heading=1.2, on_chart_exit="report", **EXITING)
    assert not reported.chart_valid
    with pytest.raises(ValueError, match="left the chart"):
        reported.as_transfer_record()

    truncated = reported.truncated_to_chart()
    record = truncated.as_transfer_record()
    assert record.grid["samples"] == truncated.arc_length.size
    assert np.all(np.isfinite(record.b))


def test_the_guard_freezes_rather_than_advancing_a_stopped_trajectory() -> None:
    """The point of stopping is that the right-hand side is never called past it."""
    seen: list[float] = []

    def rhs(s, y):
        seen.append(float(np.max(y)))
        return np.ones_like(y)

    grid, trajectory, valid = integrate_guarded(
        rhs, np.zeros((2, 1)), length=1.0, n_steps=10, method="euler",
        guard=lambda y: (y[..., 0] < np.array([0.25, 0.75])),
    )
    assert list(valid) == [3, 8]
    assert max(seen) < 0.8


# -- resolvability after a track is lost ----------------------------------


def test_the_minimum_while_tracked_excludes_the_samples_after_the_loss() -> None:
    grid = np.linspace(0.0, 10.0, 1001)
    rho = np.where(grid < 5.0, 6.0, 0.05)
    outcome = evaluate_tracking(
        grid, rho,
        AcquisitionSpec(acquire_threshold=5.0, hold_threshold=3.0, max_loss_distance=0.1),
    )
    assert outcome.outcome == TRACK_LOST
    # 0.05 is how deep the failure went, not how much margin the instrument had
    # while it was working. The latter is the only thing the field is for.
    assert outcome.min_resolvability_while_tracked == pytest.approx(6.0)
    assert outcome.min_resolvability_while_tracked > 3.0


def test_a_loss_at_the_moment_of_acquisition_reports_no_tracked_minimum() -> None:
    grid = np.linspace(0.0, 10.0, 1001)
    rho = np.where(grid < 0.5, 6.0, 0.05)
    outcome = evaluate_tracking(
        grid, rho,
        AcquisitionSpec(acquire_threshold=5.0, hold_threshold=3.0, max_loss_distance=0.1),
    )
    assert outcome.outcome == TRACK_LOST
    assert outcome.min_resolvability_while_tracked == pytest.approx(6.0)


# -- the invariant, checked directly --------------------------------------


def test_the_determinant_is_checked_without_a_decomposition() -> None:
    """The 2x2 determinant holds on a box the singular values cannot.

    The identity ``sigma_1 sigma_2 = |det Phi|`` is exact and the arithmetic is
    not: on a tolerance box with an aspect ratio of 10^4 the SVD returns
    ``sigma_1`` to a relative ``eps``, and the product lands near 1 by orders of
    magnitude less than the determinant does.
    """
    record = constant_curvature_trace(np.linspace(0.0, 2.0, 401), -1.0).as_transfer_record()
    direct = float(np.max(np.abs(record.scaled_determinant(1e-5, 1e-1) - 1.0)))
    singular = record.scaled_singular_values(1e-5, 1e-1)
    through_svd = float(np.max(np.abs(singular[:, 0] * singular[:, 1] - 1.0)))

    assert direct < 1e-13
    assert direct < through_svd


def test_conjugation_leaves_the_determinant_where_it_was() -> None:
    record = constant_curvature_trace(np.linspace(0.0, 2.0, 401), 1.0).as_transfer_record()
    for box in ((1e-3, 1e-3), (1e-2, 1e-4), (1e-5, 1e-1)):
        assert np.allclose(record.scaled_determinant(*box), record.determinant, atol=1e-14)


# -- artefacts that have to be reproducible -------------------------------


def test_floats_are_canonicalised_relative_to_their_own_size() -> None:
    """Rounding is relative, so a 1e-16 residual is still a 1e-16 residual."""
    assert canonical_float(2.220446049250313e-16) == pytest.approx(2.22044604925e-16)
    assert canonical_float(1.0 + 4.5e-16) == 1.0
    assert canonical_float(-0.0) == 0.0
    assert str(canonical_float(-0.0)) == "0.0"


def test_canonicalisation_removes_the_digits_platforms_disagree_on() -> None:
    value = 1.2345678901234567
    perturbed = np.nextafter(np.nextafter(value, 2.0), 2.0)
    assert value != perturbed
    assert jsonable({"x": value}) == jsonable({"x": perturbed})


def test_canonicalisation_keeps_more_precision_than_any_declared_threshold() -> None:
    """The tightest check in the repository is 1e-14; twelve digits is relative."""
    assert CANONICAL_DIGITS >= 12
    assert canonical_float(1.2345e-14) == pytest.approx(1.2345e-14, rel=1e-11)


def test_non_finite_values_become_null_rather_than_invalid_json() -> None:
    assert jsonable({"a": float("nan"), "b": float("inf")}) == {"a": None, "b": None}


# -- declared limits and the evidence for them ----------------------------


def _route_record():
    return constant_curvature_trace(np.linspace(0.0, 2.0, 201), 1.0).as_transfer_record()


@pytest.mark.parametrize(
    "limits",
    [
        {"max_cross_track_error": 0.0},
        {"max_cross_track_error": -1.0},
        {"max_heading_error": float("nan")},
        {"max_path_length": 0.0},
        {"min_boundary_clearance": -0.1},
    ],
)
def test_a_limit_that_admits_nothing_is_refused_at_declaration(limits) -> None:
    with pytest.raises(ValueError):
        RouteConstraints(**limits)


def test_boundary_clearance_must_be_on_the_records_own_grid() -> None:
    record = _route_record()
    with pytest.raises(ValueError, match="arclength grid"):
        assess_route(
            record, max_lateral=1e-3, max_heading=1e-3,
            constraints=RouteConstraints(min_boundary_clearance=0.01),
            boundary_clearance=np.ones(record.arclength.size - 3),
        )


def test_a_non_finite_boundary_clearance_is_refused() -> None:
    record = _route_record()
    clearance = np.ones(record.arclength.size)
    clearance[17] = np.nan
    with pytest.raises(ValueError, match="finite"):
        assess_route(
            record, max_lateral=1e-3, max_heading=1e-3,
            constraints=RouteConstraints(min_boundary_clearance=0.01),
            boundary_clearance=clearance,
        )


def test_a_covariance_of_the_wrong_shape_is_named_as_such() -> None:
    record = _route_record()
    constraints = RouteConstraints(
        acquisition=AcquisitionSpec(acquire_threshold=5.0, hold_threshold=3.0)
    )
    with pytest.raises(ValueError, match="2x2"):
        assess_route(
            record, max_lateral=1e-3, max_heading=1e-3, constraints=constraints,
            observation=ObservationModel.transverse_only(
                2.5e-5, mode="intrinsic-surface-distance"
            ),
            initial_covariance=np.eye(3),
        )


def test_one_covariance_validator_serves_the_whole_repository() -> None:
    for bad in (np.eye(3), [[1.0, 2.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]]):
        with pytest.raises(ValueError):
            validated_covariance(np.asarray(bad, dtype=float))
    assert validated_covariance(np.diag([4.0, 9.0])).shape == (2, 2)


# -- a mode that the domain cannot produce --------------------------------


def test_a_record_cannot_claim_a_quantity_its_domain_does_not_have() -> None:
    """Each half names something real; the pair is what is false."""
    grid = np.linspace(0.0, 1.0, 51)
    fields = {
        "arclength": grid,
        "gaussian_curvature": np.zeros_like(grid),
        "a": np.ones_like(grid),
        "a_rate": np.zeros_like(grid),
        "b": grid,
        "b_rate": np.ones_like(grid),
    }
    with pytest.raises(ValueError, match="unavailable on 'parametric-surface'"):
        TransferRecord(
            **fields, domain="parametric-surface",
            observation_mode="intrinsic-surface-distance",
        )
    with pytest.raises(KeyError, match="unknown domain"):
        TransferRecord(**fields, domain="a-real-part")


def test_a_curvature_profile_is_its_own_domain() -> None:
    """Not a model space -- K varies -- and not a surface: there is no embedding."""
    grid = np.linspace(0.0, 1.0, 201)
    trace = constant_curvature_trace(grid, 1.0)
    assert trace.as_transfer_record().domain == "constant-curvature"
    require_available("intrinsic-surface-distance", "declared-curvature-profile")
    with pytest.raises(ValueError, match="unavailable"):
        require_available("ambient-euclidean-chord", "declared-curvature-profile")


def test_a_surface_record_may_not_be_relabelled_into_a_chord_it_has_no_embedding_for() -> None:
    record = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=200
    ).as_transfer_record()
    assert record.observation_mode == "ambient-euclidean-chord"
    with pytest.raises(ValueError, match="unavailable"):
        integrate_path(
            torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=200
        ).as_transfer_record(observation_mode="intrinsic-surface-distance")


# -- the covariance a record declares -------------------------------------


def test_a_declared_covariance_must_match_the_records_units_and_frame() -> None:
    record = _route_record()
    with pytest.raises(ValueError, match="micrometre"):
        record.with_covariance(
            StartingCovariance.from_tolerance_box(
                1.0, 1.0, units=Units(length="micrometre", angle="radian")
            )
        )
