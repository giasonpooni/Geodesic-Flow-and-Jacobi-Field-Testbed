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
    good = np.diag([1.44e-4, 1.44e-4, 1.44e-4]).tolist()
    assert _record(measurement_covariance=good).measurement_covariance is not None
    for bad in (
        [[1.0, 2.0], [2.0, 1.0]],       # not positive semidefinite
        [[float("nan")]],                # not finite
        [[1.0, 0.0]],                    # not square
        np.array([[1.0, 2.0], [3.0, 4.0]]).tolist(),  # not symmetric
    ):
        with pytest.raises(ValueError):
            _record(measurement_covariance=bad)


def test_a_measurement_covariance_of_the_wrong_size_is_refused() -> None:
    """Square, symmetric and positive semidefinite, and still the wrong matrix.

    A 1x1 covariance on a trial that measured three separations passes every
    other check here. It is a covariance -- of some other trial -- and pairing
    it with these arc lengths is the failure mode the shape bound exists for.
    """
    with pytest.raises(ValueError, match="1x1 and this trial measured 3"):
        _record(measurement_covariance=[[1.44e-4]])


# -- binding a trial to the record the prediction came from ----------------


def _prediction_record(**overrides):
    """A transfer record on the trial's own arclength, in the trial's units."""
    from geodesic_testbed import constant_curvature_trace
    from geodesic_testbed.boundary import Units

    grid = np.array([0.0, 100.0, 200.0])
    return constant_curvature_trace(grid, 0.0).as_transfer_record(
        units=overrides.pop("units", Units(length="mm", angle="radian")),
        observation_mode="ambient-euclidean-chord",
        **overrides,
    )


def test_a_prediction_record_in_other_units_is_refused() -> None:
    """Metres against millimetres agrees in shape and is wrong by a thousand."""
    from geodesic_testbed.boundary import Units

    with pytest.raises(ValueError, match="length unit"):
        compare(
            _record(),
            _filtered([0.0, 1.70, 3.40]),
            prediction_source=_prediction_record(units=Units(length="m", angle="radian")),
        )


def test_a_prediction_bound_to_another_calibration_is_refused() -> None:
    from geodesic_testbed.boundary import CalibrationBinding

    bound = _prediction_record(
        calibration=CalibrationBinding(
            calibration_ids=("bench-cal-2025-01",),
            registration_id="sha256:bbb",
            reconstruction_version="recon-0.0",
        )
    )
    with pytest.raises(ValueError, match="different instrument state"):
        compare(_record(), _filtered([0.0, 1.70, 3.40]), prediction_source=bound)


def test_a_prediction_bound_to_the_trials_own_calibration_agrees() -> None:
    from geodesic_testbed.boundary import CalibrationBinding

    trial = _record()
    bound = _prediction_record(
        calibration=CalibrationBinding(
            calibration_ids=(trial.calibration_id,),
            registration_id=trial.calibration_transform_digest,
            reconstruction_version=trial.reconstruction_version,
        )
    )
    result = compare(trial, _filtered([0.0, 1.70, 3.40]), prediction_source=bound)
    assert result["calibration"]["agreed"] is True


def test_an_unbound_prediction_reports_that_no_tie_was_established() -> None:
    """Allowed -- a prediction from geometry has no calibration -- and stated."""
    result = compare(
        _record(), _filtered([0.0, 1.70, 3.40]), prediction_source=_prediction_record()
    )
    assert result["calibration"]["agreed"] is None
    assert result["calibration"]["prediction"]["bound"] is False
    assert result["calibration"]["trial"]["calibration_ids"] == ["bench-cal-2026-09"]


# -- the comparison statistic ----------------------------------------------


def test_a_comparison_with_no_covariance_says_so_rather_than_implying_a_verdict() -> None:
    """The scalars are a summary of the residual, not a statistic about it."""
    verdict = compare(_record(), _filtered([0.0, 1.70, 3.40]))
    statistics = verdict["residual_statistics"]
    assert statistics["available"] is False
    assert "no covariance" in statistics["reason"]
    assert "not a statistic" in statistics["reason"]


def test_a_trial_that_declares_a_covariance_is_compared_against_it() -> None:
    """Declaring one and not using it was the defect: it is the whole comparison."""
    record = _record(measurement_covariance=np.diag([4e-4, 4e-4, 4e-4]).tolist())
    verdict = compare(record, _filtered([0.0, 1.70, 3.40]))
    statistics = verdict["residual_statistics"]
    assert statistics["available"] is True
    assert statistics["source"] == "trial-declared"
    assert statistics["degrees_of_freedom"] == 3
    assert statistics["band"]["lower"] < statistics["band"]["upper"]
    assert statistics["verdict"] in {
        "consistent", "covariance-too-large", "residual-too-large"
    }
    assert 0.0 <= statistics["probability_less_than"] <= 1.0


def test_the_band_is_closed_at_both_ends_so_an_inflated_covariance_fails() -> None:
    """A budget large enough to cover everything is rejected, not rewarded.

    The same residual against two covariances: one honest, one a hundred times
    too large. A one-sided test passes both. Only the lower limit distinguishes
    them, and overstating is the direction a measurement budget usually errs.
    """
    def unfiltered(**overrides):
        return _record(
            filter_identifier="none",
            filter_version="",
            filter_causal=True,
            filter_operator_digest="",
            **overrides,
        )

    # The residual is [0, 0.02, 0.04]; a variance of 6.7e-4 makes the reduced
    # chi-square one by construction, and a hundred times that does not.
    predicted = [0.0, 1.70, 3.40]
    honest = compare(
        unfiltered(measurement_covariance=np.diag([6.7e-4] * 3).tolist()), predicted
    )["residual_statistics"]
    inflated = compare(
        unfiltered(measurement_covariance=np.diag([6.7e-2] * 3).tolist()), predicted
    )["residual_statistics"]

    assert honest["reduced_chi_square"] == pytest.approx(1.0, rel=0.02)

    assert honest["verdict"] == "consistent"
    assert inflated["verdict"] == "covariance-too-large"
    assert inflated["chi_square"] < honest["chi_square"]


def test_a_residual_far_larger_than_the_declared_covariance_is_rejected() -> None:
    verdict = compare(
        _record(measurement_covariance=np.diag([1e-8] * 3).tolist()),
        _filtered([0.0, 1.50, 3.00]),
    )["residual_statistics"]
    assert verdict["verdict"] == "residual-too-large"
    assert verdict["max_abs_whitened_residual"] > 10.0


def test_an_assembled_covariance_of_the_wrong_size_is_refused() -> None:
    """Two matrices that are each fine and are not the same comparison."""
    from geodesic_testbed.engine.output_covariance import OutputCovariance

    wrong = OutputCovariance(
        arclength=np.array([0.0, 1.0, 2.0, 3.0]),
        outputs=("signed-transverse-separation",),
        blocks={"observation-noise": np.diag([1e-4] * 4)},
    )
    with pytest.raises(ValueError, match="not the same comparison"):
        compare(_record(), _filtered([0.0, 1.70, 3.40]), covariance=wrong)


def test_an_assembled_covariance_takes_precedence_over_the_trial_s_own() -> None:
    """The trial knows its instrument; the assembled total knows the whole chain."""
    from geodesic_testbed.engine.output_covariance import OutputCovariance

    assembled = OutputCovariance(
        arclength=np.array([0.0, 100.0, 200.0]),
        outputs=("signed-transverse-separation",),
        blocks={
            "observation-noise": np.diag([4e-4] * 3),
            "shared-parameters": np.full((3, 3), 9e-4),
        },
    )
    verdict = compare(
        _record(measurement_covariance=np.diag([4e-4] * 3).tolist()),
        _filtered([0.0, 1.70, 3.40]),
        covariance=assembled,
    )["residual_statistics"]
    assert verdict["source"] == "assembled"
    assert set(verdict["shares"]) == {"observation-noise", "shared-parameters"}
    assert sum(verdict["shares"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [
    [[-1e-30]], [[1e-30, 2e-30], [2e-30, 1e-30]],
    [[0, 1e-300], [1e-300, 1]], [[True]], [[10**1000]], [],
])
def test_measurement_covariance_does_not_have_an_absolute_acceptance_floor(bad):
    with pytest.raises(ValueError):
        _record(measurement_covariance=bad)


def test_measurement_covariance_retains_singular_mixed_unit_entries():
    """Extreme scales survive the correlation normalisation without overflow.

    Sized to the trial's three separations rather than to 2x2: the covariance
    is *of* the observation vector, so its shape is bound to it, and a 2x2 here
    would be a covariance of some other trial. The entries are what this test
    is about and they are unchanged.
    """
    matrix = [
        [1e-300, 1.0, 0.0],
        [1.0, 1e300, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert _record(measurement_covariance=matrix).measurement_covariance == matrix


def test_comparison_rechecks_mutated_filtered_covariance():
    artifact = apply_filter(
        SMOOTHER, [0.0, 1.7, 3.4], identifier="rts-smoother", version="1.2.0",
        causal=False, noise_covariance=0.04,
    )
    artifact.noise_covariance.setflags(write=True)
    artifact.noise_covariance[0, 0] = -1e-30
    with pytest.raises(ValueError, match="negative variance"):
        compare(_record(), artifact)


def test_measurement_covariance_is_rechecked_before_export_and_comparison():
    record = _record(measurement_covariance=[[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]])
    record.measurement_covariance[0][0] = -1e-30
    with pytest.raises(ValueError, match="negative variance"):
        record.to_dict()
    with pytest.raises(ValueError, match="negative variance"):
        record.digest()
    with pytest.raises(ValueError, match="negative variance"):
        compare(record, _filtered([0.0, 1.7, 3.4]))

