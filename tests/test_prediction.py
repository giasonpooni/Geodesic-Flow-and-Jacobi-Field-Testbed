# SPDX-License-Identifier: MPL-2.0
"""The chain from the transfer map to something a sensor could have reported.

Four stages, each an object carrying the transformation that produced it. The
tests here pin the two properties that make that worth doing: the chain cannot
run backwards, so a chord can never be relabelled an intrinsic distance; and
where the transformations are all available, applying them actually closes the
gap to an independently computed measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed import constant_curvature_trace
from geodesic_testbed.engine.envelope import finite_difference_jacobi, integrate_path
from geodesic_testbed.engine.observation_model import (
    ObservationModel,
    TemporalFilter,
    operator_digest,
)
from geodesic_testbed.engine.prediction import (
    STAGES,
    Prediction,
    chord_from_intrinsic,
    chord_from_tangent,
    first_order_prediction,
    has_closed_form_separation,
    instrument_from_chord,
    intrinsic_from_tangent,
    predict_chord,
    residual_statistics,
)
from geodesic_testbed.engine.surfaces import sphere, torus

GRID = np.linspace(0.0, 1.5, 151)


def _trace(curvature: float = 1.0):
    return constant_curvature_trace(GRID, curvature).as_transfer_record()


def _surface(length: float = 1.5, n_steps: int = 800):
    return integrate_path(
        sphere(1.0), u0=np.pi / 2, v0=0.0, heading=0.6, length=length, n_steps=n_steps
    )


# -- the chain has a direction --------------------------------------------


def test_each_stage_carries_the_operation_that_produced_it() -> None:
    record = _trace()
    tangent = first_order_prediction(record, 0.0, 0.01)
    intrinsic = intrinsic_from_tangent(tangent, record)

    assert tangent.stage == "first-order-tangent"
    assert tangent.observation_mode == "first-order-tangent-separation"
    assert intrinsic.stage == "intrinsic-surface-distance"
    assert len(intrinsic.chain) == 2
    assert "sn_K(d/2)" in intrinsic.chain[1]


def test_the_chain_cannot_run_backwards() -> None:
    """A chord relabelled an intrinsic distance is the confusion this prevents."""
    record = _trace()
    intrinsic = intrinsic_from_tangent(first_order_prediction(record, 0.0, 0.01), record)
    with pytest.raises(ValueError, match="would be a relabelling"):
        intrinsic.advanced("first-order-tangent", intrinsic.values, "undo")


def test_the_first_order_prediction_is_not_a_distance() -> None:
    """It is signed, and past a focus it is negative. A distance is not."""
    envelope = _surface(length=4.0)
    record = envelope.as_transfer_record()
    tangent = first_order_prediction(record, 0.0, 0.01)
    assert float(np.min(tangent.values)) < 0.0
    assert record.focus_events()


def test_an_intrinsic_distance_is_refused_where_the_curvature_varies() -> None:
    record = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=1.5, n_steps=200
    ).as_transfer_record()
    assert not has_closed_form_separation(record)
    with pytest.raises(ValueError, match="boundary-value problem"):
        intrinsic_from_tangent(first_order_prediction(record, 0.0, 0.01), record)


def test_the_closed_form_follows_the_path_not_the_domain() -> None:
    """A sphere reached the general way still has constant curvature."""
    record = _surface().as_transfer_record()
    assert record.domain == "parametric-surface"
    assert has_closed_form_separation(record)
    intrinsic = intrinsic_from_tangent(first_order_prediction(record, 0.0, 0.01), record)
    assert "hypothesis is about the path" in intrinsic.extra["justification"]


def test_a_chord_needs_an_embedding_and_a_curvature_profile_has_none() -> None:
    record = _trace()
    tangent = first_order_prediction(record, 0.0, 0.01)
    with pytest.raises(ValueError, match="carries no geometry"):
        chord_from_tangent(tangent, record)


# -- the transformations are right ----------------------------------------


def test_the_intrinsic_correction_reproduces_the_closed_form_coefficient() -> None:
    """``d = eps sn_K(s) [1 - cn_K(s)^2 eps^2 / 24 + ...]``, and the residual is the next term."""
    from geodesic_testbed.engine.spaceforms import cos_k

    for curvature in (1.0, 0.0, -1.0):
        record = _trace(curvature)
        residuals = []
        for epsilon in (0.02, 0.01):
            tangent = first_order_prediction(record, 0.0, epsilon)
            intrinsic = intrinsic_from_tangent(tangent, record)
            keep = np.abs(tangent.values) > 1e-9
            measured = 1.0 - intrinsic.values[keep] / tangent.values[keep]
            predicted = cos_k(GRID[keep], curvature) ** 2 * epsilon**2 / 24.0
            residuals.append(float(np.max(np.abs(measured / predicted - 1.0))))
        # The residual is the eps^4 term, so halving eps quarters it.
        assert residuals[0] < 1e-3
        assert residuals[0] / residuals[1] == pytest.approx(4.0, rel=0.05)


def test_the_full_chain_reproduces_an_independently_measured_chord() -> None:
    """Flowing two geodesics and measuring between them knows nothing of this chain."""
    length, n_steps, epsilon = 1.5, 800, 5e-3
    envelope = _surface(length, n_steps)
    record = envelope.as_transfer_record(observation_mode="ambient-euclidean-chord")
    _, measured = finite_difference_jacobi(
        sphere(1.0), u0=np.pi / 2, v0=0.0, heading=0.6,
        epsilon=epsilon, length=length, n_steps=n_steps,
    )
    chord = measured * 2.0 * epsilon

    tangent = first_order_prediction(record, 0.0, 2.0 * epsilon)
    chain = chord_from_intrinsic(intrinsic_from_tangent(tangent, record), record)
    scale = float(np.max(np.abs(tangent.values)))

    before = float(np.max(np.abs(chord - np.abs(tangent.values)))) / scale
    after = float(np.max(np.abs(chord - np.abs(chain.values)))) / scale
    assert before > 1e-6
    assert after < 1e-9
    assert chain.extra["intrinsic_correction"] == "applied"


def test_a_chord_only_prediction_declares_the_term_it_did_not_apply() -> None:
    record = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=1.5, n_steps=400
    ).as_transfer_record()
    chain = chord_from_tangent(first_order_prediction(record, 0.0, 0.01), record)
    assert chain.stage == "ambient-euclidean-chord"
    assert chain.extra["intrinsic_correction"].startswith("not-applied")
    assert "has NOT been applied" in chain.note


def test_predict_chord_runs_the_whole_chain_and_stops_before_the_instrument() -> None:
    record = _surface().as_transfer_record(observation_mode="ambient-euclidean-chord")
    chain = predict_chord(record, heading=0.01)
    assert chain.stage == "ambient-euclidean-chord"
    assert len(chain.chain) == 3
    assert predict_chord(record, heading=0.01, to_stage="first-order-tangent").stage == (
        STAGES[0]
    )
    with pytest.raises(ValueError, match="needs an observation model"):
        predict_chord(record, heading=0.01, to_stage="instrument-output")


# -- the instrument stage --------------------------------------------------


def _instrument_stage(*, with_covariance: bool = True, temporal_filter=None, filter_matrix=None):
    record = _surface(length=1.0, n_steps=100).as_transfer_record(
        observation_mode="ambient-euclidean-chord"
    )
    chord = predict_chord(record, heading=0.01)
    observation = ObservationModel.full_pose(
        2.5e-5, 1.0e-4, mode="ambient-euclidean-chord", calibration_id="bench-cal-1"
    )
    return record, instrument_from_chord(
        chord, observation, record,
        initial_covariance=np.diag([1e-6, 1e-6]) if with_covariance else None,
        temporal_filter=temporal_filter,
        filter_matrix=filter_matrix,
    )


def test_the_instrument_stage_projects_and_carries_the_covariance_to_judge_against() -> None:
    record, stage = _instrument_stage()
    assert stage.stage == "instrument-output"
    assert stage.values.shape == (record.arclength.size, 2)
    assert stage.covariance.shape == (record.arclength.size, 2, 2)
    # The spread the residual is judged against is not the measurement noise
    # alone: it also carries the starting covariance through Phi.
    _, bare = _instrument_stage(with_covariance=False)
    assert np.all(np.diagonal(stage.covariance, axis1=1, axis2=2) >= np.diagonal(
        bare.covariance, axis1=1, axis2=2
    ) - 1e-18)
    assert "C0" in stage.transformation


def test_a_mismatched_observation_mode_is_refused_rather_than_relabelled() -> None:
    record = _surface(length=1.0, n_steps=100).as_transfer_record(
        observation_mode="ambient-euclidean-chord"
    )
    chord = predict_chord(record, heading=0.01)
    with pytest.raises(ValueError, match="rather than relabelling"):
        instrument_from_chord(
            chord,
            ObservationModel.transverse_only(
                2.5e-5, mode="intrinsic-surface-distance"
            ),
            record,
        )


def test_a_filter_moves_the_prediction_and_the_noise_together() -> None:
    """``F H Phi dz0`` against ``F R F^T``, never a filtered side against the raw R."""
    record, unfiltered = _instrument_stage()
    size = record.arclength.size
    smoother = np.eye(size)
    smoother[1:-1, :-2] += 0.25 * np.eye(size - 2)
    smoother[1:-1, 2:] += 0.25 * np.eye(size - 2)
    smoother[1:-1, 1:-1] -= 0.5 * np.eye(size - 2)
    declared = TemporalFilter(
        identifier="three-point", version="1.0", causal=False,
        parameters={"window": 3}, tuned_on="none",
    )
    _, filtered = _instrument_stage(temporal_filter=declared, filter_matrix=smoother)

    assert filtered.extra["filtered"] is True
    assert "three-point" in filtered.transformation
    assert filtered.extra["filter_operator_digest"] == operator_digest(smoother)
    assert not np.allclose(filtered.values, unfiltered.values)
    # The filter mixes arc lengths, so the covariance stops being a per-sample
    # stack and becomes one matrix over every scalar residual at once.
    assert unfiltered.covariance.shape == (size, 2, 2)
    assert filtered.covariance.shape == (size * 2, size * 2)
    off_diagonal = filtered.covariance.copy()
    for index in range(size):
        off_diagonal[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = 0.0
    assert np.max(np.abs(off_diagonal)) > 0.0

    statistics = residual_statistics(filtered.values, filtered)
    assert statistics.correlated is True


def test_a_filter_needs_both_its_name_and_the_operator_that_ran() -> None:
    """A boolean cannot tell the declared operator from another with the same name."""
    declared = TemporalFilter(identifier="three-point", version="1.0", causal=False)
    with pytest.raises(ValueError, match="both its declaration and its operator"):
        _instrument_stage(temporal_filter=declared)
    with pytest.raises(ValueError, match="both its declaration and its operator"):
        _instrument_stage(filter_matrix=np.eye(101))


# -- the comparison statistic ----------------------------------------------


def test_a_residual_drawn_from_the_declared_covariance_whitens_to_unit_variance() -> None:
    """Reduced chi-square near one says the residual is the size the model predicts."""
    record, stage = _instrument_stage()
    rng = np.random.default_rng(20260920)
    factors = np.linalg.cholesky(stage.covariance)
    noise = np.einsum(
        "ijk,ik->ij", factors, rng.standard_normal(stage.values.shape)
    )
    statistics = residual_statistics(stage.values + noise, stage)

    assert statistics.degrees_of_freedom == stage.values.size
    assert statistics.reduced_chi_square == pytest.approx(1.0, abs=0.15)
    assert statistics.whitened.shape == stage.values.shape


def test_a_residual_ten_times_too_large_shows_up_in_the_reduced_chi_square() -> None:
    record, stage = _instrument_stage()
    rng = np.random.default_rng(20260920)
    factors = np.linalg.cholesky(stage.covariance)
    noise = 10.0 * np.einsum(
        "ijk,ik->ij", factors, rng.standard_normal(stage.values.shape)
    )
    statistics = residual_statistics(stage.values + noise, stage)
    assert statistics.reduced_chi_square > 50.0


def test_the_whitened_residual_sees_what_a_maximum_absolute_error_cannot() -> None:
    """Two residuals of the same size, in directions the instrument resolves differently."""
    record, stage = _instrument_stage()
    sigma = np.sqrt(np.diagonal(stage.covariance, axis1=1, axis2=2))
    size = 3.0 * float(np.min(sigma))

    transverse_only = np.zeros_like(stage.values)
    transverse_only[:, 0] = size
    heading_only = np.zeros_like(stage.values)
    heading_only[:, 1] = size

    first = residual_statistics(stage.values + transverse_only, stage)
    second = residual_statistics(stage.values + heading_only, stage)

    assert first.max_abs_residual == pytest.approx(second.max_abs_residual)
    assert first.max_abs_whitened != pytest.approx(second.max_abs_whitened)


def test_a_residual_with_no_declared_spread_is_not_evidence() -> None:
    record = _surface(length=1.0, n_steps=100).as_transfer_record(
        observation_mode="ambient-euclidean-chord"
    )
    chord = predict_chord(record, heading=0.01)
    assert chord.covariance is None
    with pytest.raises(ValueError, match="not evidence"):
        residual_statistics(chord.values, chord)


def test_a_measurement_on_a_different_grid_is_refused() -> None:
    _, stage = _instrument_stage()
    with pytest.raises(ValueError, match="the same grid"):
        residual_statistics(stage.values[:-1], stage)


def test_a_supplied_covariance_is_recorded_as_such() -> None:
    _, stage = _instrument_stage()
    supplied = np.broadcast_to(np.eye(2) * 1e-8, stage.covariance.shape).copy()
    statistics = residual_statistics(stage.values, stage, covariance=supplied)
    assert "supplied by the caller" in statistics.note
    assert residual_statistics(stage.values, stage).note.startswith("whitened against")


def test_a_prediction_serialises_its_chain_without_its_samples() -> None:
    _, stage = _instrument_stage()
    payload = stage.to_dict()
    assert payload["stage"] == "instrument-output"
    assert payload["has_covariance"] is True
    assert len(payload["chain"]) == 4
    assert "values" not in payload
    assert "values" in stage.to_dict(include_samples=True)


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(ValueError, match="stage must be one of"):
        Prediction(
            stage="whatever-the-scanner-said",
            values=np.zeros(5),
            arclength=np.linspace(0.0, 1.0, 5),
            transformation="none",
        )
