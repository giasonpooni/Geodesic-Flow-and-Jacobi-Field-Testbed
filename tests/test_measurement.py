"""path-sensitivity-observation-v1: the shape a physical trial has to arrive in."""

from __future__ import annotations

import json

import numpy as np
import pytest

from geodesic_testbed import (
    MEASUREMENT_SCHEMA,
    MeasurementRecord,
    Perturbation,
    Uncertainty,
    apply_filter,
    compare,
)
from geodesic_testbed.engine.observation_model import FilteredPrediction, operator_digest

#: A three-sample smoother, standing in for the bench's real one.
SMOOTHER = np.array([
    [0.75, 0.25, 0.00],
    [0.25, 0.50, 0.25],
    [0.00, 0.25, 0.75],
])
SMOOTHER_DIGEST = operator_digest(SMOOTHER)


def _filtered(values, *, matrix=SMOOTHER, identifier="rts-smoother", version="1.2.0"):
    """A prediction put through the bench's declared operator."""
    return apply_filter(
        matrix, values, identifier=identifier, version=version, causal=False
    )


def _record(**overrides) -> MeasurementRecord:
    fields = {
        "coupon_id": "plate-01",
        "run_id": "r003",
        "role": "validation",
        "observation_mode": "ambient-euclidean-chord",
        "observation_mode_version": 1,
        "geometry_model": "fitted-bicubic",
        "geometry_model_digest": "sha256:aaa",
        "reconstruction_version": "recon-0.0",
        "calibration_id": "bench-cal-2026-09",
        "filter_identifier": "rts-smoother",
        "filter_version": "1.2.0",
        "filter_causal": False,
        "filter_operator_digest": SMOOTHER_DIGEST,
        "filter_tuned_on": "calibration-set-A",
        "units": {"length": "mm", "angle": "radian"},
        "coordinate_frame": "coupon-datum-A",
        "datum_frame": "fixture-A",
        "calibration_transform_digest": "sha256:bbb",
        "as_built_scan_digest": "sha256:ccc",
        "raw_data_digest": "sha256:ddd",
        "prediction_report_digest": "sha256:eee",
        "perturbation": Perturbation(commanded_lateral=0.0, commanded_heading=0.0175),
        "uncertainty": Uncertainty(
            metrology=0.012, fixture=0.008, repeatability=0.005, surface_geometry=0.010
        ),
        "arclength": [0.0, 100.0, 200.0],
        "signed_transverse_separation": [0.0, 1.72, 3.44],
    }
    return MeasurementRecord(**(fields | overrides))


def test_a_complete_trial_serialises_with_its_schema_and_digest() -> None:
    record = _record()
    payload = record.to_dict()
    assert payload["schema"] == MEASUREMENT_SCHEMA
    assert json.loads(json.dumps(payload, default=str))["coupon_id"] == "plate-01"
    assert record.digest().startswith("sha256:")
    assert _record().digest() == record.digest()
    assert _record(run_id="r004").digest() != record.digest()


def test_the_uncertainty_budget_combines_in_quadrature() -> None:
    record = _record()
    assert record.uncertainty.total() == pytest.approx(
        np.sqrt(0.012**2 + 0.008**2 + 0.005**2 + 0.010**2)
    )
    assert record.uncertainty.to_dict()["total"] == record.uncertainty.total()


def test_signal_to_noise_is_reported_so_an_unresolvable_trial_is_visible() -> None:
    resolvable = _record()
    assert resolvable.signal_to_noise > 100.0
    faint = _record(signed_transverse_separation=[0.0, 0.01, 0.02])
    assert faint.signal_to_noise < 3.0

    # With no declared bar the module reports the ratio and draws no conclusion.
    verdict = compare(faint, _filtered([0.0, 0.009, 0.019]))
    assert verdict["signal_to_noise"] == pytest.approx(faint.signal_to_noise)
    assert verdict["meets_declared_resolvability"] is None

    # The bar comes from the instrument protocol.
    assert compare(
        faint, _filtered([0.0, 0.009, 0.019]), resolvability_threshold=3.0
    )["meets_declared_resolvability"] is False
    assert compare(
        faint, _filtered([0.0, 0.009, 0.019]), resolvability_threshold=1.0
    )["meets_declared_resolvability"] is True


def test_a_comparison_refuses_a_mode_mismatch() -> None:
    record = _record()
    with pytest.raises(ValueError, match="convert one before comparing"):
        compare(
            record, _filtered([0.0, 1.7, 3.4]), mode="intrinsic-surface-distance"
        )
    verdict = compare(record, _filtered([0.0, 1.70, 3.40]))
    assert verdict["observation_mode"] == "ambient-euclidean-chord"
    assert verdict["role"] == "validation"
    assert verdict["filter"]["operator_digest"] == SMOOTHER_DIGEST


def test_a_filtered_trial_cannot_be_compared_with_an_unfiltered_prediction() -> None:
    """F changes the model: y_f = F H Phi dz0 + F eta, and R_f = F R F^T."""
    record = _record()
    with pytest.raises(ValueError, match="F R F"):
        compare(record, [0.0, 1.70, 3.40])
    unfiltered = _record(
        filter_identifier="none", filter_version="", filter_causal=True,
        filter_operator_digest="",
    )
    assert compare(unfiltered, [0.0, 1.70, 3.40])["filter"]["identifier"] == "none"

    # ... and filtering only the prediction is the same bias the other way up.
    with pytest.raises(ValueError, match="biases it towards agreement"):
        compare(unfiltered, _filtered([0.0, 1.70, 3.40]))


def test_a_different_operator_with_the_same_name_is_refused() -> None:
    """The failure a boolean flag cannot catch."""
    record = _record()
    impostor = np.array([
        [0.50, 0.50, 0.00],
        [0.10, 0.80, 0.10],
        [0.00, 0.50, 0.50],
    ])
    assert operator_digest(impostor) != SMOOTHER_DIGEST

    # Same identifier, same version, different matrix: the label agrees and the
    # operator does not, which is exactly how a filter manufactures agreement.
    with pytest.raises(ValueError, match="operator digest"):
        compare(record, _filtered([0.0, 1.70, 3.40], matrix=impostor))

    # And the same operator under a different name is refused too.
    with pytest.raises(ValueError, match="identifier"):
        compare(record, _filtered([0.0, 1.70, 3.40], identifier="butterworth"))


def test_a_causal_prediction_cannot_stand_in_for_an_offline_one() -> None:
    record = _record()
    causal_artifact = FilteredPrediction(
        values=np.array([0.0, 1.70, 3.40]),
        identifier="rts-smoother",
        version="1.2.0",
        operator_digest=SMOOTHER_DIGEST,
        causal=True,
    )
    with pytest.raises(ValueError, match="causal"):
        compare(record, causal_artifact)


def test_applying_the_filter_carries_the_output_covariance() -> None:
    """R_f = F R F^T, and a filter correlates samples that were independent."""
    artifact = apply_filter(
        SMOOTHER, [0.0, 1.70, 3.40], identifier="rts-smoother", version="1.2.0",
        causal=False, noise_covariance=0.04,
    )
    expected = SMOOTHER @ (0.04 * np.eye(3)) @ SMOOTHER.T
    assert artifact.noise_covariance == pytest.approx(expected)
    off_diagonal = artifact.noise_covariance[0, 1]
    assert off_diagonal != 0.0, "a filter correlates neighbouring samples"


def test_rejected_samples_need_a_declared_rule() -> None:
    with pytest.raises(ValueError, match="outlier rule"):
        _record(rejected_sample_mask=[False, True, False])
    kept = _record(rejected_sample_mask=[False, True, False], outlier_rule="3-sigma on residual")
    assert kept.outlier_rule


def test_a_trial_that_cannot_be_traced_is_refused() -> None:
    for missing in (
        "geometry_model_digest",
        "calibration_transform_digest",
        "as_built_scan_digest",
        "raw_data_digest",
        "prediction_report_digest",
    ):
        with pytest.raises(ValueError, match="not evidence"):
            _record(**{missing: ""})


def test_the_separation_convention_is_pinned() -> None:
    """A nearest-point distance differs at the order the campaign is resolving."""
    with pytest.raises(ValueError, match="matched nominal arc length"):
        _record(separation_convention="nearest-point")


def test_calibration_and_validation_are_kept_apart() -> None:
    assert _record(role="calibration").role == "calibration"
    with pytest.raises(ValueError):
        _record(role="demo")


def test_a_stale_mode_version_is_refused() -> None:
    with pytest.raises(ValueError, match="version"):
        _record(observation_mode_version=2)


def test_a_measurement_covariance_must_be_a_covariance() -> None:
    assert _record(measurement_covariance=[[1.44e-4]]).measurement_covariance is not None
    for bad in ([[1.0, 2.0], [2.0, 1.0]], [[float("nan")]], [[1.0, 0.0]]):
        with pytest.raises(ValueError):
            _record(measurement_covariance=bad)
