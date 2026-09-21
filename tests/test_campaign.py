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
    DifferentialCase,
    DifferentialCovariance,
    PerturbationPlan,
    SharedDifferential,
    campaign_flatness_control,
    conformance,
    default_program,
    evaluate_differential_case,
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
#: difference and the second does not, which is why the effect is measured
#: rather than asserted.
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


def _case(**overrides) -> DifferentialCase:
    fields = {
        "plate_run_id": "p1",
        "cylinder_run_id": "c1",
        "predicted_difference": np.array([0.0, 0.05, 0.11]),
        "prediction_digest": "sha256:prediction",
    }
    fields.update(overrides)
    return DifferentialCase(**fields)


# -- the two nulls ---------------------------------------------------------


def test_the_residual_is_measured_against_the_predicted_difference_not_zero() -> None:
    """``r_D = (y_c - y_p) - (yhat_c - yhat_p)``.

    A coupon pair whose observed difference is the predicted chord correction
    plus about one sigma is consistent, even though the two measured
    separations are nowhere near identical. Tested against zero the same pair
    would read as a disagreement several times larger.
    """
    plate = _trial(stage="flat-plate", coupon="p1", run="p1",
                   separation=[0.0, 1.72, 3.44], measurement_covariance=_independent())
    cylinder = _trial(stage="rolled-cylinder", coupon="c1", run="c1",
                      separation=[0.03, 1.73, 3.57], measurement_covariance=_independent())
    row = evaluate_differential_case(_case(shared_parameters=_shared()), plate, cylinder)

    assert row["statistic"]["accepted"] is True
    assert row["max_abs_residual"] < 0.05
    assert row["max_abs_observed_difference"] > 2.5 * row["max_abs_residual"]
    assert row["prediction_digest"] == "sha256:prediction"


def test_a_real_disagreement_still_fails() -> None:
    plate = _trial(stage="flat-plate", coupon="p1", run="p1",
                   separation=[0.0, 1.72, 3.44], measurement_covariance=_independent())
    cylinder = _trial(stage="rolled-cylinder", coupon="c1", run="c1",
                      separation=[0.0, 2.77, 4.55], measurement_covariance=_independent())
    row = evaluate_differential_case(_case(shared_parameters=_shared()), plate, cylinder)
    assert row["statistic"]["accepted"] is False
    assert row["statistic"]["verdict"] == "upper-tail-inconsistent"
    assert row["statistic"]["interpretations"]


def test_a_residual_of_exactly_zero_is_rejected_by_the_lower_tail() -> None:
    """The lower limit is not decoration, and the verdict does not diagnose.

    A measured difference reproducing the prediction to the last digit against
    a declared uncertainty of tens of microns is inconsistent with that
    uncertainty. Which of several causes produced it -- an overstated
    covariance, a prediction not independent of the observation, parameters
    fitted on the evaluated data -- the statistic cannot say, so the result
    lists them instead of picking one.
    """
    plate = _trial(stage="flat-plate", coupon="p1", run="p1",
                   separation=[0.0, 1.72, 3.44], measurement_covariance=_independent())
    cylinder = _trial(stage="rolled-cylinder", coupon="c1", run="c1",
                      separation=[0.0, 1.77, 3.55], measurement_covariance=_independent())
    row = evaluate_differential_case(_case(shared_parameters=_shared()), plate, cylinder)

    assert row["max_abs_residual"] == pytest.approx(0.0, abs=1e-12)
    assert row["statistic"]["verdict"] == "lower-tail-inconsistent"
    assert "the declared covariance is overstated" in row["statistic"]["interpretations"]
    assert any(
        "not independent" in reason for reason in row["statistic"]["interpretations"]
    ), "an overstated covariance is one explanation among several, not the verdict"


# -- the covariance, and what the subtraction did to it --------------------


def test_without_a_declared_cross_covariance_nothing_is_tested() -> None:
    """Quadrature assumes independence, which is the opposite of cancellation."""
    plate = _trial(stage="flat-plate", coupon="p1", run="p1")
    cylinder = _trial(stage="rolled-cylinder", coupon="c1", run="c1")
    row = evaluate_differential_case(_case(), plate, cylinder)
    assert row["differential_covariance"] == "not-established"
    assert row["statistic"] is None


def test_a_declared_difference_covariance_is_enough_on_its_own() -> None:
    plate = _trial(stage="flat-plate", coupon="p1", run="p1")
    cylinder = _trial(stage="rolled-cylinder", coupon="c1", run="c1")
    row = evaluate_differential_case(
        _case(
            difference_covariance=DifferentialCovariance(
                matrix=np.eye(3) * 4e-4,
                basis="a declared differential repeatability study",
                independent_sources=("sensor-noise",),
            )
        ),
        plate,
        cylinder,
    )
    assert row["differential_covariance"] == "established"
    assert row["covariance"]["accounts_for"] == ["sensor-noise"]
    assert row["statistic"] is not None


def test_a_joint_covariance_becomes_the_difference_through_its_selector() -> None:
    """``Sigma_D = D Sigma_joint D^T`` with ``D = [-I  I]``.

    The cross-covariance blocks are where the cancellation actually lives, and
    a caller holding the 2n x 2n matrix should not be forming the difference by
    hand. Perfectly correlated trials cancel to nothing; uncorrelated ones add.
    """
    block = np.eye(3) * 4e-4
    correlated = np.block([[block, block], [block, block]])
    case = DifferentialCase.from_joint(
        joint_covariance=correlated + np.eye(6) * 1e-12,
        basis="a declared joint study",
        plate_run_id="p1",
        cylinder_run_id="c1",
        predicted_difference=np.zeros(3),
        prediction_digest="sha256:prediction",
    )
    assert np.allclose(case.difference_covariance.matrix, np.eye(3) * 2e-12, atol=1e-15)

    independent = np.block([[block, np.zeros((3, 3))], [np.zeros((3, 3)), block]])
    apart = DifferentialCase.from_joint(
        joint_covariance=independent,
        basis="a declared joint study",
        plate_run_id="p1",
        cylinder_run_id="c1",
        predicted_difference=np.zeros(3),
        prediction_digest="sha256:prediction",
    )
    assert np.allclose(apart.difference_covariance.matrix, np.eye(3) * 8e-4)


def test_an_odd_sized_joint_covariance_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\(2n, 2n\)"):
        DifferentialCase.from_joint(
            joint_covariance=np.eye(5),
            basis="a study",
            plate_run_id="p1",
            cylinder_run_id="c1",
            predicted_difference=np.zeros(2),
            prediction_digest="sha256:p",
        )


@pytest.mark.parametrize(
    ("label", "fixture_cylinder", "expected"),
    (
        ("identical Jacobians cancel entirely", "same", 0.0),
        ("one coupon alone leaves it intact", "plate-only", 1.0),
        ("opposite Jacobians amplify", "opposite", 2.0),
    ),
)
def test_the_variance_ratio_is_not_a_fraction(
    label: str, fixture_cylinder: str, expected: float
) -> None:
    """It reaches two when the two coupons felt the parameter oppositely.

    Differencing then *amplifies* the shared uncertainty instead of removing
    it -- the outcome a name like "uncancelled fraction" would have quietly
    excluded, and the reason the value is not clipped.
    """
    plate_jacobian = np.ones((3, 1))
    cylinder_jacobian = {
        "same": np.ones((3, 1)),
        "plate-only": np.zeros((3, 1)),
        "opposite": -np.ones((3, 1)),
    }[fixture_cylinder]
    shared = SharedDifferential(
        names=("fixture-datum",),
        kinds=("fixture-datum",),
        covariance=np.array([[9e-4]]),
        plate_jacobian=plate_jacobian,
        cylinder_jacobian=cylinder_jacobian,
        basis="a fixture repeatability study",
    )
    assert shared.differential_to_separate_variance_ratio() == pytest.approx(
        expected, abs=1e-12
    ), label


def test_a_source_declared_on_both_sides_is_refused() -> None:
    """The same calibration uncertainty in the independent parts and the shared
    block is counted twice, which is not conservative."""
    with pytest.raises(ValueError, match="counted twice"):
        DifferentialCovariance(
            matrix=np.eye(3) * 4e-4,
            basis="a study",
            independent_sources=("calibration-transform",),
            shared_sources=("calibration-transform",),
        )


# -- pairing and compatibility ---------------------------------------------


def test_pairing_uses_achieved_perturbations_and_not_commanded_ones() -> None:
    """Two runs can share a command and receive different starting poses."""
    plate = _trial(stage="flat-plate", coupon="p1", run="p1", heading=0.0175)
    drifted = replace(
        _trial(stage="rolled-cylinder", coupon="c1", run="c1", heading=0.0175),
        perturbation=Perturbation(
            commanded_lateral=0.0,
            commanded_heading=0.0175,
            achieved_heading=0.0175 * 0.99 + 2e-3,
            achieved_lateral=0.0,
            achieved_uncertainty_lateral=1e-3,
            achieved_uncertainty_heading=1e-5,
        ),
    )
    with pytest.raises(ValueError, match="achieved perturbations differ"):
        evaluate_differential_case(_case(), plate, drifted)


def test_the_matching_threshold_is_a_declared_policy_not_a_constant() -> None:
    """One sigma is a protocol's choice, so the case carries it."""
    plate = _trial(stage="flat-plate", coupon="p1", run="p1", heading=0.0175)
    drifted = replace(
        _trial(stage="rolled-cylinder", coupon="c1", run="c1", heading=0.0175),
        perturbation=Perturbation(
            commanded_lateral=0.0,
            commanded_heading=0.0175,
            achieved_heading=0.0175 * 0.99 + 2e-5,
            achieved_lateral=0.0,
            achieved_uncertainty_lateral=1e-3,
            achieved_uncertainty_heading=1e-5,
        ),
    )
    with pytest.raises(ValueError, match="achieved perturbations differ"):
        evaluate_differential_case(_case(), plate, drifted)
    row = evaluate_differential_case(
        _case(achieved_match_sigmas=3.0), plate, drifted
    )
    assert row["plate_achieved"] is not None
    assert row["cylinder_achieved"] is not None
    assert row["plate_achieved"] != row["cylinder_achieved"]


def test_a_trial_with_no_achieved_perturbation_cannot_be_paired() -> None:
    plate = _trial(stage="flat-plate", coupon="p1", run="p1")
    cylinder = _trial(stage="rolled-cylinder", coupon="c1", run="c1", achieved=False)
    with pytest.raises(ValueError, match="achieved perturbations are not reported"):
        evaluate_differential_case(_case(), plate, cylinder)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observation_mode", "first-order-tangent-separation"),
        ("units", {"length": "m", "angle": "radian"}),
        ("coordinate_frame", "some-other-frame"),
        ("datum_frame", "fixture-B"),
        ("calibration_id", "bench-cal-2025-01"),
        ("calibration_transform_digest", "sha256:a-different-transform"),
        ("reconstruction_version", "recon-9.9"),
        ("outlier_rule", "5-sigma on residual"),
    ),
)
def test_two_trials_that_are_not_differenceable_are_refused(field: str, value) -> None:
    """The difference of two different quantities has no null hypothesis."""
    plate = _trial(stage="flat-plate", coupon="p1", run="p1")
    cylinder = replace(_trial(stage="rolled-cylinder", coupon="c1", run="c1"),
                       **{field: value})
    with pytest.raises(ValueError, match="cannot be differenced"):
        evaluate_differential_case(_case(), plate, cylinder)


def test_a_calibration_id_that_matches_a_different_transform_is_refused() -> None:
    """The id names a state; the digest is the transform that actually ran."""
    plate = _trial(stage="flat-plate", coupon="p1", run="p1")
    cylinder = replace(
        _trial(stage="rolled-cylinder", coupon="c1", run="c1"),
        calibration_transform_digest="sha256:a-different-transform",
    )
    assert plate.calibration_id == cylinder.calibration_id
    with pytest.raises(ValueError, match="calibration_transform_digest"):
        evaluate_differential_case(_case(), plate, cylinder)


def test_a_case_that_names_other_runs_is_refused() -> None:
    plate = _trial(stage="flat-plate", coupon="p1", run="p1")
    cylinder = _trial(stage="rolled-cylinder", coupon="c1", run="c9")
    with pytest.raises(ValueError, match="this case is for"):
        evaluate_differential_case(_case(), plate, cylinder)


def test_a_case_must_name_the_prediction_it_came_from() -> None:
    with pytest.raises(ValueError, match="name the prediction"):
        _case(prediction_digest="")


# -- the campaign-wide result ----------------------------------------------


def test_a_campaign_boolean_is_withheld_while_any_pair_is_untested() -> None:
    """One tested pair among several must not report the campaign consistent."""
    plate_a = _trial(stage="flat-plate", coupon="p1", run="p1",
                     separation=[0.0, 1.72, 3.44], measurement_covariance=_independent())
    cylinder_a = _trial(stage="rolled-cylinder", coupon="c1", run="c1",
                        separation=[0.03, 1.73, 3.57],
                        measurement_covariance=_independent())
    # The second pair declares no covariance at all, so it cannot be tested.
    plate_b = _trial(stage="flat-plate", coupon="p2", run="p2", separation=[0.0, 1.72, 3.44])
    cylinder_b = _trial(stage="rolled-cylinder", coupon="c2", run="c2",
                        separation=[0.03, 1.73, 3.57])

    result = campaign_flatness_control(
        [
            _case(shared_parameters=_shared()),
            _case(plate_run_id="p2", cylinder_run_id="c2", shared_parameters=_shared()),
        ],
        [plate_a, cylinder_a, plate_b, cylinder_b],
    )
    assert result["matched_pairs"] == 2
    assert result["tested_pairs"] == 1
    assert result["untested_pairs"] == 1
    assert result["covariance_status"] == "partial"
    assert result["all_tested_consistent"] is None, (
        "a boolean here would say the campaign is consistent while half of it was "
        "never examined"
    )


def test_a_fully_tested_campaign_does_report_a_boolean() -> None:
    plate = _trial(stage="flat-plate", coupon="p1", run="p1",
                   separation=[0.0, 1.72, 3.44], measurement_covariance=_independent())
    cylinder = _trial(stage="rolled-cylinder", coupon="c1", run="c1",
                      separation=[0.03, 1.73, 3.57], measurement_covariance=_independent())
    result = campaign_flatness_control(
        [_case(shared_parameters=_shared())], [plate, cylinder]
    )
    assert result["covariance_status"] == "complete"
    assert result["all_tested_consistent"] is True
    assert result["status"] == "not-started"


def test_each_pair_is_judged_against_its_own_prediction() -> None:
    """A single array broadcast across unlike conditions is the failure here.

    Two pairs at different perturbations have different predicted differences.
    Given each its own, both are consistent; given both the first's, the second
    is not.
    """
    # The cylinder reads the plate plus the pair's own predicted difference,
    # plus a residual of about one sigma. An exactly zero residual is a
    # different outcome -- the lower tail -- and not what "consistent" means.
    residual = (0.03, -0.02, 0.02)

    def pair(suffix: str, offset: float):
        return (
            _trial(stage="flat-plate", coupon=f"p{suffix}", run=f"p{suffix}",
                   separation=[0.0, 1.72, 3.44], measurement_covariance=_independent()),
            _trial(stage="rolled-cylinder", coupon=f"c{suffix}", run=f"c{suffix}",
                   separation=[0.0 + residual[0],
                               1.72 + offset + residual[1],
                               3.44 + 2 * offset + residual[2]],
                   measurement_covariance=_independent()),
        )

    plate_a, cylinder_a = pair("1", 0.05)
    plate_b, cylinder_b = pair("2", 0.30)
    records = [plate_a, cylinder_a, plate_b, cylinder_b]

    own = campaign_flatness_control(
        [
            _case(predicted_difference=np.array([0.0, 0.05, 0.10]),
                  shared_parameters=_shared()),
            _case(plate_run_id="p2", cylinder_run_id="c2",
                  predicted_difference=np.array([0.0, 0.30, 0.60]),
                  shared_parameters=_shared()),
        ],
        records,
    )
    assert own["covariance_status"] == "complete"
    assert own["all_tested_consistent"] is True

    shared_prediction = campaign_flatness_control(
        [
            _case(predicted_difference=np.array([0.0, 0.05, 0.10]),
                  shared_parameters=_shared()),
            _case(plate_run_id="p2", cylinder_run_id="c2",
                  predicted_difference=np.array([0.0, 0.05, 0.10]),
                  shared_parameters=_shared()),
        ],
        records,
    )
    assert shared_prediction["all_tested_consistent"] is False


def test_a_campaign_with_no_cases_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one declared pair"):
        campaign_flatness_control([], [_trial(stage="flat-plate", coupon="p1", run="p1")])


def test_a_case_naming_a_missing_trial_is_refused() -> None:
    with pytest.raises(ValueError, match="not among the supplied trials"):
        campaign_flatness_control(
            [_case()], [_trial(stage="flat-plate", coupon="p1", run="p1")]
        )


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
