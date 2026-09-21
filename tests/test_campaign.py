"""The coupon programme, and the bookkeeping failures it refuses to accept.

No coupon has been cut. What is tested here is the *programme*: that it is
ordered, that it perturbs both axes, that it keeps calibration and validation
apart by coupon, that it demands achieved perturbations, and that it repeats at
a second scale. Every one of those is a way a campaign produces a full set of
plots and no evidence.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from geodesic_testbed import MeasurementRecord, Perturbation, Uncertainty
from geodesic_testbed.engine.campaign import (
    CAMPAIGN_STATUS,
    CouponProgram,
    CouponStage,
    PerturbationPlan,
    SharedDifferential,
    conformance,
    default_program,
    intrinsic_flatness_control,
)

ARCLENGTH = [0.0, 100.0, 200.0]


def _trial(
    *,
    stage: str,
    coupon: str,
    run: str,
    role: str = "validation",
    lateral: float = 0.0,
    heading: float = 0.0175,
    achieved: bool = True,
    scale: float = 1.0,
    separation=None,
    mode: str = "ambient-euclidean-chord",
    measurement_covariance=None,
) -> MeasurementRecord:
    return MeasurementRecord(
        coupon_id=coupon,
        run_id=run,
        role=role,
        observation_mode=mode,
        observation_mode_version=1,
        geometry_model="fitted-bicubic",
        geometry_model_digest="sha256:aaa",
        reconstruction_version="recon-0.0",
        calibration_id="bench-cal-2026-09",
        filter_identifier="none",
        filter_version="",
        filter_causal=True,
        filter_operator_digest="",
        units={"length": "mm", "angle": "radian"},
        coordinate_frame="coupon-datum-A",
        datum_frame="fixture-A",
        calibration_transform_digest="sha256:bbb",
        as_built_scan_digest="sha256:ccc",
        raw_data_digest=f"sha256:{run}",
        prediction_report_digest="sha256:eee",
        measurement_covariance=measurement_covariance,
        perturbation=Perturbation(
            commanded_lateral=lateral,
            commanded_heading=heading,
            achieved_lateral=lateral * 1.01 if achieved else None,
            achieved_heading=heading * 0.99 if achieved else None,
            achieved_uncertainty_lateral=1e-3 if achieved else None,
            achieved_uncertainty_heading=1e-5 if achieved else None,
        ),
        uncertainty=Uncertainty(
            metrology=0.012, fixture=0.008, repeatability=0.005, surface_geometry=0.010
        ),
        arclength=list(ARCLENGTH),
        signed_transverse_separation=(
            list(separation) if separation is not None else [0.0, 1.72, 3.44]
        ),
        extra={"stage": stage, "scale": scale},
    )


def _complete_set() -> list[MeasurementRecord]:
    """The smallest set that satisfies the programme."""
    trials: list[MeasurementRecord] = []
    index = 0
    for stage in default_program().stages:
        for scale in (1.0, 2.0):
            for role, coupon in (("calibration", "c"), ("validation", "v")):
                for lateral, heading in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.0175)):
                    index += 1
                    trials.append(
                        _trial(
                            stage=stage.key,
                            coupon=f"{stage.key}-{coupon}-{scale:g}",
                            run=f"r{index:04d}",
                            role=role,
                            lateral=lateral,
                            heading=heading,
                            scale=scale,
                        )
                    )
    return trials


# -- the programme itself --------------------------------------------------


def test_the_programme_is_declared_in_advance_and_has_no_data() -> None:
    program = default_program()
    assert program.status == CAMPAIGN_STATUS == "not-started"
    assert [stage.order for stage in program.stages] == [1, 2, 3, 4]
    assert program.total_runs > 0


def test_the_first_stage_is_a_differential_control_and_says_so() -> None:
    """Its prediction is that two coupons agree, so the systematics cancel."""
    program = default_program()
    plate = program.stage("flat-plate")
    cylinder = program.stage("rolled-cylinder")
    assert plate.paired_with == "rolled-cylinder"
    assert cylinder.paired_with == "flat-plate"
    assert "cancel" in cylinder.falsifies or "cancel" in cylinder.note


def test_every_later_stage_names_what_it_rests_on() -> None:
    program = default_program()
    for stage in program.stages:
        for needed in stage.depends_on:
            assert program.stage(needed).order < stage.order
    assert program.stage("as-built-varying").depends_on == ("spherical-cap",)


def test_a_programme_at_one_scale_is_refused() -> None:
    """A dimensionless claim tested at one size has not been tested."""
    program = default_program()
    with pytest.raises(ValueError, match="one scale"):
        CouponProgram(stages=program.stages, plan=program.plan, scales=(1.0,))


def test_a_plan_that_perturbs_only_the_heading_is_refused() -> None:
    """It measures b and says nothing about a, while producing a full set of plots."""
    with pytest.raises(ValueError, match="says nothing|measures the a column"):
        PerturbationPlan(lateral=(0.0,), heading=(0.0, 1.0))


def test_a_plan_with_no_control_run_is_refused() -> None:
    with pytest.raises(ValueError, match="zero-zero control"):
        PerturbationPlan(lateral=(0.5, 1.0), heading=(0.5, 1.0))


def test_a_plan_with_a_single_replicate_is_refused() -> None:
    with pytest.raises(ValueError, match="repeatability"):
        PerturbationPlan(lateral=(0.0, 1.0), heading=(0.0, 1.0), replicates=1)


def test_a_stage_order_with_a_gap_is_refused() -> None:
    plan = default_program().plan
    stages = (
        CouponStage(key="a", order=1, geometry="", question="", falsifies=""),
        CouponStage(key="b", order=3, geometry="", question="", falsifies=""),
    )
    with pytest.raises(ValueError, match="no gaps"):
        CouponProgram(stages=stages, plan=plan)


# -- conformance -----------------------------------------------------------


def test_a_complete_set_satisfies_the_programme() -> None:
    report = conformance(default_program(), _complete_set())
    assert report.satisfied, report.failures
    assert not report.stages_missing
    assert report.counts["scales"] == [1.0, 2.0]


def test_a_missing_stage_is_reported_rather_than_assumed_fine() -> None:
    trials = [t for t in _complete_set() if t.extra["stage"] != "as-built-varying"]
    report = conformance(default_program(), trials)
    assert not report.satisfied
    assert report.stages_missing == ("as-built-varying",)


def test_a_stage_read_before_its_prerequisite_is_refused() -> None:
    trials = [
        t for t in _complete_set() if t.extra["stage"] in ("spherical-cap",)
    ]
    report = conformance(default_program(), trials)
    assert any("depends on" in failure for failure in report.failures)


def test_a_differential_stage_without_its_pair_is_refused() -> None:
    trials = [t for t in _complete_set() if t.extra["stage"] != "rolled-cylinder"]
    report = conformance(default_program(), trials)
    assert any("differential test" in failure for failure in report.failures)


def test_a_coupon_used_for_both_calibration_and_validation_is_refused() -> None:
    """The split has to be by coupon: two runs share its geometry and fixturing."""
    trials = _complete_set()
    for trial in trials:
        if trial.extra["stage"] == "flat-plate" and trial.role == "validation":
            index = trials.index(trial)
            trials[index] = _trial(
                stage="flat-plate", coupon="flat-plate-c-1", run=trial.run_id,
                role="validation", scale=trial.extra["scale"],
            )
    report = conformance(default_program(), trials)
    assert any("both" in failure and "calibration" in failure for failure in report.failures)


def test_a_stage_that_never_perturbs_the_lateral_axis_is_refused() -> None:
    trials = [
        t
        for t in _complete_set()
        if not (t.extra["stage"] == "spherical-cap" and t.perturbation.commanded_lateral)
    ]
    report = conformance(default_program(), trials)
    assert any("a column" in failure for failure in report.failures)


def test_commanded_without_achieved_is_refused() -> None:
    """The gap between them is a starting-pose error of the kind under test."""
    trials = _complete_set()
    trials[0] = _trial(
        stage=trials[0].extra["stage"], coupon=trials[0].coupon_id,
        run=trials[0].run_id, role=trials[0].role, achieved=False,
        scale=trials[0].extra["scale"],
    )
    report = conformance(default_program(), trials)
    assert any("achieved perturbation" in failure for failure in report.failures)


def test_one_scale_is_refused_because_the_claim_is_dimensionless() -> None:
    trials = [t for t in _complete_set() if t.extra["scale"] == 1.0]
    report = conformance(default_program(), trials)
    assert any("dimensionless" in failure for failure in report.failures)


def test_a_stage_mixing_observation_modes_is_refused() -> None:
    trials = _complete_set()
    trials[0] = _trial(
        stage=trials[0].extra["stage"], coupon=trials[0].coupon_id,
        run=trials[0].run_id, role=trials[0].role,
        scale=trials[0].extra["scale"], mode="intrinsic-surface-distance",
    )
    report = conformance(default_program(), trials)
    assert any("mixes observation modes" in failure for failure in report.failures)


def test_a_trial_that_names_no_stage_cannot_be_evidence_for_one() -> None:
    trials = _complete_set()
    orphan = _trial(stage="", coupon="x", run="r9999")
    report = conformance(default_program(), [*trials, orphan])
    assert any("names no stage" in failure for failure in report.failures)


# -- the flatness control --------------------------------------------------


#: Both coupons feel the calibration scale identically; only the cylinder's
#: fixture was re-established when it was mounted. The first cancels in the
#: difference and the second does not, which is the whole point of measuring
#: the cancellation rather than asserting it.
def _shared(samples: int = 3, *, fixture_differs: bool = True) -> SharedDifferential:
    calibration = np.ones((samples, 1))
    fixture_plate = np.zeros((samples, 1))
    fixture_cylinder = np.full((samples, 1), 1.0 if fixture_differs else 0.0)
    return SharedDifferential(
        names=("calibration-scale", "fixture-datum"),
        kinds=("calibration-transform", "fixture-datum"),
        covariance=np.diag([4e-4, 9e-4]),
        plate_jacobian=np.hstack([calibration, fixture_plate]),
        cylinder_jacobian=np.hstack([calibration, fixture_cylinder]),
        basis="a certificate and a fixture repeatability study",
    )


def _independent(samples: int = 3, sigma: float = 0.02):
    return (np.eye(samples) * sigma**2).tolist()


def test_the_intrinsic_and_observation_nulls_are_separate_hypotheses() -> None:
    """The transfer maps agree; the ambient chords do not, and must not.

    Testing ``y_c - y_p`` against zero would reject a correct runtime on a
    coupon pair it predicts perfectly, because the cylinder has a transverse
    normal curvature the plate does not.
    """
    plate = [_trial(stage="flat-plate", coupon="p1", run="p1", lateral=0.5)]
    cylinder = [_trial(stage="rolled-cylinder", coupon="c1", run="c1", lateral=0.5)]
    result = intrinsic_flatness_control(plate, cylinder)

    assert "same transfer map" in result["intrinsic_null"]
    assert "not tested" in result["observation_null"]
    assert result["matched_pairs"] == 1
    assert result["differential_covariance"] == "differential_covariance_not_established"
    assert result["worst_reduced_chi_square"] is None
    assert result["status"] == "not-started"


def test_the_residual_is_measured_against_the_predicted_difference_not_zero() -> None:
    """``r_D = (y_c - y_p) - (yhat_c - yhat_p)``.

    A coupon pair whose observed difference is exactly the predicted chord
    correction has residual zero and must pass, even though the two measured
    separations are nowhere near identical.
    """
    predicted = np.array([0.0, 0.05, 0.11])
    # The cylinder reads the plate plus the predicted chord correction, plus a
    # residual of about one sigma -- which is what a consistent trial looks
    # like. An exactly zero residual is a different outcome, tested below.
    plate = [
        _trial(stage="flat-plate", coupon="p1", run="p1",
               separation=[0.0, 1.72, 3.44], measurement_covariance=_independent())
    ]
    cylinder = [
        _trial(stage="rolled-cylinder", coupon="c1", run="c1",
               separation=[0.03, 1.73, 3.57], measurement_covariance=_independent())
    ]
    result = intrinsic_flatness_control(
        plate, cylinder, predicted_difference=predicted, shared=_shared()
    )
    row = result["pairs"][0]
    assert result["all_consistent"] is True
    assert row["max_abs_residual"] < 0.05

    # Against zero instead, the same pair would look like a 0.13 disagreement:
    # nearly three times the residual the correct null leaves.
    assert row["max_abs_observed_difference"] > 2.5 * row["max_abs_residual"]


def test_a_residual_of_exactly_zero_is_too_good_and_the_band_says_so() -> None:
    """The lower limit is not decoration.

    A measured difference that reproduces the prediction to the last digit,
    against a declared uncertainty of tens of microns, is evidence that the
    uncertainty is overstated or that the two sides are not independent -- not
    evidence of agreement.
    """
    predicted = np.array([0.0, 0.05, 0.11])
    plate = [
        _trial(stage="flat-plate", coupon="p1", run="p1",
               separation=[0.0, 1.72, 3.44], measurement_covariance=_independent())
    ]
    cylinder = [
        _trial(stage="rolled-cylinder", coupon="c1", run="c1",
               separation=[0.0, 1.77, 3.55], measurement_covariance=_independent())
    ]
    result = intrinsic_flatness_control(
        plate, cylinder, predicted_difference=predicted, shared=_shared()
    )
    row = result["pairs"][0]
    assert row["max_abs_residual"] == pytest.approx(0.0, abs=1e-12)
    assert row["statistic"]["verdict"] == "covariance-too-large"
    assert result["all_consistent"] is False


def test_a_real_disagreement_still_fails() -> None:
    predicted = np.array([0.0, 0.05, 0.11])
    plate = [
        _trial(stage="flat-plate", coupon="p1", run="p1",
               separation=[0.0, 1.72, 3.44], measurement_covariance=_independent())
    ]
    cylinder = [
        _trial(stage="rolled-cylinder", coupon="c1", run="c1",
               separation=[0.0, 2.77, 4.55], measurement_covariance=_independent())
    ]
    result = intrinsic_flatness_control(
        plate, cylinder, predicted_difference=predicted, shared=_shared()
    )
    assert result["all_consistent"] is False
    assert result["pairs"][0]["statistic"]["verdict"] == "residual-too-large"


def test_without_a_declared_cross_covariance_nothing_is_tested() -> None:
    """Quadrature assumes independence, which is the opposite of cancellation."""
    predicted = np.zeros(3)
    plate = [_trial(stage="flat-plate", coupon="p1", run="p1")]
    cylinder = [_trial(stage="rolled-cylinder", coupon="c1", run="c1")]
    result = intrinsic_flatness_control(plate, cylinder, predicted_difference=predicted)
    assert result["differential_covariance"] == "differential_covariance_not_established"
    assert result["pairs"][0]["statistic"] is None
    assert result["all_consistent"] is None


def test_a_declared_joint_covariance_is_enough_on_its_own() -> None:
    predicted = np.zeros(3)
    plate = [_trial(stage="flat-plate", coupon="p1", run="p1")]
    cylinder = [_trial(stage="rolled-cylinder", coupon="c1", run="c1")]
    result = intrinsic_flatness_control(
        plate, cylinder,
        predicted_difference=predicted,
        joint_covariance=np.eye(3) * 4e-4,
    )
    assert result["differential_covariance"] == "established"
    assert result["pairs"][0]["statistic"] is not None


def test_cancellation_is_measured_rather_than_asserted() -> None:
    """A shared parameter cancels only where both coupons felt it identically.

    The calibration scale is common to both and drops out. The fixture datum
    was re-established when the cylinder was mounted, so it does not -- and the
    reported fraction says which situation the control is actually in.
    """
    both_common = _shared(fixture_differs=False)
    assert both_common.uncancelled_fraction() == pytest.approx(0.0, abs=1e-15)

    partly_common = _shared(fixture_differs=True)
    assert partly_common.uncancelled_fraction() > 0.1, (
        "a fixture that was re-established does not cancel, and a control that "
        "assumed it did would understate its own uncertainty"
    )
    surviving = partly_common.difference_block()
    assert np.allclose(surviving, 9e-4 * np.ones((3, 3)))


def test_pairing_uses_achieved_perturbations_and_not_commanded_ones() -> None:
    """Two runs can share a command and receive different starting poses."""
    plate = [_trial(stage="flat-plate", coupon="p1", run="p1", heading=0.0175)]
    drifted = _trial(stage="rolled-cylinder", coupon="c1", run="c1", heading=0.0175)
    drifted = replace(
        drifted,
        perturbation=Perturbation(
            commanded_lateral=0.0,
            commanded_heading=0.0175,
            # Same command, a starting pose two hundred sigma away.
            achieved_heading=0.0175 * 0.99 + 2e-3,
            achieved_lateral=0.0,
            achieved_uncertainty_lateral=1e-3,
            achieved_uncertainty_heading=1e-5,
        ),
    )
    with pytest.raises(ValueError, match="achieved perturbations differ"):
        intrinsic_flatness_control(plate, [drifted])


def test_a_trial_with_no_achieved_perturbation_cannot_be_paired() -> None:
    plate = [_trial(stage="flat-plate", coupon="p1", run="p1")]
    cylinder = [_trial(stage="rolled-cylinder", coupon="c1", run="c1", achieved=False)]
    with pytest.raises(ValueError, match="achieved perturbations are not reported"):
        intrinsic_flatness_control(plate, cylinder)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observation_mode", "first-order-tangent-separation"),
        ("units", {"length": "m", "angle": "radian"}),
        ("coordinate_frame", "some-other-frame"),
        ("datum_frame", "fixture-B"),
        ("calibration_id", "bench-cal-2025-01"),
        ("reconstruction_version", "recon-9.9"),
    ),
)
def test_two_trials_that_are_not_differenceable_are_refused(field: str, value) -> None:
    """The difference of two different quantities has no null hypothesis."""
    plate = [_trial(stage="flat-plate", coupon="p1", run="p1")]
    cylinder = [replace(_trial(stage="rolled-cylinder", coupon="c1", run="c1"),
                        **{field: value})]
    with pytest.raises(ValueError, match="differenceable"):
        intrinsic_flatness_control(plate, cylinder)


def test_a_filtered_trial_is_not_differenceable_against_an_unfiltered_one() -> None:
    plate = [_trial(stage="flat-plate", coupon="p1", run="p1")]
    cylinder = [replace(
        _trial(stage="rolled-cylinder", coupon="c1", run="c1"),
        filter_identifier="rts-smoother", filter_version="1.2.0",
        filter_operator_digest="sha256:f", filter_causal=False,
    )]
    with pytest.raises(ValueError, match="differenceable"):
        intrinsic_flatness_control(plate, cylinder)


def test_the_flatness_control_needs_both_coupons() -> None:
    plate = [_trial(stage="flat-plate", coupon="p1", run="p1")]
    with pytest.raises(ValueError, match="needs trials from both"):
        intrinsic_flatness_control(plate, [])


def test_a_conforming_set_still_claims_no_physical_result() -> None:
    """Conformance is about bookkeeping. It says nothing about agreement.

    A set of trials can satisfy every structural rule here and still disagree
    with the prediction entirely -- which is the point: the programme is what
    makes a disagreement mean something, not what decides whether there is one.
    """
    report = conformance(default_program(), _complete_set())
    assert report.satisfied
    assert default_program().status == "not-started"
    assert "satisfied" in report.to_dict()
    assert "agreement" not in str(report.to_dict())
