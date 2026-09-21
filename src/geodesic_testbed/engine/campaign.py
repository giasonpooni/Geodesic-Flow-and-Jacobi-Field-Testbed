"""The coupon programme this prediction would have to survive, declared in advance.

No coupon has been cut and no trial has been run. What this module supplies is
the *programme*: which coupons, in which order, perturbed how, split between
calibration and validation, and at how many scales -- written down before there
is any data, because a campaign design chosen after seeing the residuals is not
a test of anything.

The stages are ordered by what each one can falsify on its own.

1. **Flat plate against rolled cylinder.** A control before an experiment, and
   a *differential* one: the two coupons have the same intrinsic geometry, so
   they must have the same transfer map. Two things about it are easy to state
   too strongly, and :func:`intrinsic_flatness_control` states them carefully.

   The prediction is **not** that the two measured separations are identical.
   They are identical in ``Phi``; at finite perturbation the cylinder has a
   transverse normal curvature the plate does not, so the two ambient-chord
   predictions differ at second order -- the same effect that makes the
   cylinder's validity envelope 15.7% tighter than the plate's. The intrinsic
   null and the observation-space null are separate hypotheses and the second
   is tested against a *predicted* difference, not against zero.

   And the systematic budget does not cancel merely by being named in both
   budgets. A shared parameter contributes
   ``(J_c - J_p) C_theta (J_c - J_p)^T`` to the difference, which vanishes only
   where the two coupons felt it identically; a fixture datum re-established
   when the second coupon was mounted did not. What cancels is measured and
   reported, not assumed.

   It is still the stage to run first, and it is still the only one whose
   prediction needs no curvature at all.

2. **Spherical cap.** The first stage with something to measure rather than
   nullify: focusing, and past ``s = pi R`` a conjugate point. The prediction
   is not a number but a *location*, which is a far harder thing to get right
   by accident than a magnitude.

3. **As-built saddle or torus.** Varying curvature, where there is no closed
   form and the general machinery is all there is -- and, on an as-built part,
   the first stage where the surface itself is measured rather than declared,
   so the geometry-uncertainty term in the budget stops being zero.

4. **Repeat at a second scale.** The claim this repository makes about route
   decisions is that they are dimensionless. A second radius, with the
   tolerance and the instrument scaled to match, is what turns that from an
   argument into a measurement.

Three rules apply to every stage, and each one exists because campaigns fail on
bookkeeping far more often than on physics:

* **Perturb both axes.** A starting pose is wrong in two independent ways and
  the two columns of ``Phi`` focus in different places. A programme that
  perturbs only the heading measures ``b`` and says nothing whatever about
  ``a``, while producing a full set of plots.
* **Keep calibration and validation apart.** A trial used to tune the model and
  again to demonstrate it proves nothing, and the separation has to be by
  *coupon*, not only by run: two runs on the same coupon share its as-built
  geometry, its fixturing and its calibration.
* **Use the achieved perturbation.** What was commanded is an intent. The
  difference between commanded and achieved is a starting-pose error of exactly
  the kind being measured, so a comparison against the commanded value has the
  quantity under test in its own input.

:func:`conformance` checks a submitted set of
:class:`~geodesic_testbed.engine.measurement.MeasurementRecord` against all of
this and returns what is missing. It is deliberately a *checker* rather than a
loader: the programme is the deliverable now, and the data is what it is
waiting for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .measurement import ROLES, MeasurementRecord

Array = np.ndarray

#: What the whole programme is waiting on. It moves when data arrives, and
#: nothing in this repository may report a physical agreement while it says
#: ``not-started``.
CAMPAIGN_STATUS = "not-started"


@dataclass(frozen=True)
class CouponStage:
    """One stage: a coupon, the question it settles, and what it needs first."""

    key: str
    order: int
    geometry: str
    question: str
    falsifies: str
    #: Stages whose evidence this one rests on. A saddle result read before the
    #: spherical cap has established that the solver locates a focus is a
    #: result about two things at once.
    depends_on: tuple[str, ...] = ()
    #: Coupons that must be present *together*, because the stage's prediction
    #: is about their difference rather than about either one.
    paired_with: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "order": self.order,
            "geometry": self.geometry,
            "question": self.question,
            "falsifies": self.falsifies,
            "depends_on": list(self.depends_on),
            "paired_with": self.paired_with,
            "note": self.note,
        }


@dataclass(frozen=True)
class PerturbationPlan:
    """What is commanded at the start of each run, and how the runs are split.

    ``lateral`` and ``heading`` are separate lists on purpose. A grid of the
    two crossed together would be larger than a coupon programme can afford and
    would confound the columns anyway; what is needed is each axis exercised on
    its own, plus the zero-zero control that says what the measurement reads
    when nothing was perturbed.
    """

    lateral: tuple[float, ...]
    heading: tuple[float, ...]
    replicates: int = 3
    #: Fraction of coupons reserved for validation. By coupon and not by run:
    #: two runs on the same coupon share its as-built geometry and fixturing.
    validation_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name in ("lateral", "heading"):
            values = tuple(float(value) for value in getattr(self, name))
            if not values:
                raise ValueError(f"the plan must command at least one {name} perturbation")
            if not all(np.isfinite(values)):
                raise ValueError(f"every commanded {name} perturbation must be finite")
            object.__setattr__(self, name, values)
        if not any(value != 0.0 for value in self.lateral):
            raise ValueError(
                "the plan commands no nonzero lateral perturbation, so it measures "
                "the a column not at all while producing a full set of results"
            )
        if not any(value != 0.0 for value in self.heading):
            raise ValueError("the plan commands no nonzero heading perturbation")
        if 0.0 not in self.lateral or 0.0 not in self.heading:
            raise ValueError(
                "the plan has no zero-zero control run; without one, nothing says "
                "what the measurement reads when the start was not perturbed"
            )
        if int(self.replicates) < 2:
            raise ValueError(
                "a single run per condition cannot separate repeatability from the "
                "effect being measured"
            )
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must lie strictly between 0 and 1")

    @property
    def runs_per_coupon(self) -> int:
        """Both axes swept separately, sharing the one zero-zero control."""
        conditions = (len(self.lateral) - 1) + (len(self.heading) - 1) + 1
        return int(conditions * self.replicates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lateral": list(self.lateral),
            "heading": list(self.heading),
            "replicates": int(self.replicates),
            "validation_fraction": float(self.validation_fraction),
            "runs_per_coupon": self.runs_per_coupon,
        }


@dataclass(frozen=True)
class CouponProgram:
    """The whole programme: stages, perturbations, and the scales it repeats at."""

    stages: tuple[CouponStage, ...]
    plan: PerturbationPlan
    #: Radii, as multiples of the first. At least two, or the dimensionless
    #: claim is an argument rather than a measurement.
    scales: tuple[float, ...] = (1.0, 2.0)
    status: str = CAMPAIGN_STATUS
    note: str = ""

    def __post_init__(self) -> None:
        keys = [stage.key for stage in self.stages]
        if len(keys) != len(set(keys)):
            raise ValueError("two stages share a key")
        orders = [stage.order for stage in self.stages]
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("stage orders must be 1..n with no gaps or repeats")
        for stage in self.stages:
            paired = (stage.paired_with,) if stage.paired_with else ()
            for needed in (*stage.depends_on, *paired):
                if needed not in keys:
                    raise ValueError(f"stage {stage.key!r} names unknown stage {needed!r}")
            for needed in stage.depends_on:
                if self.stage(needed).order >= stage.order:
                    raise ValueError(
                        f"stage {stage.key!r} depends on {needed!r}, which does not "
                        "come before it"
                    )
        if len(self.scales) < 2:
            raise ValueError(
                "a programme at one scale cannot test a dimensionless claim; the "
                "whole point of rho and of S^-1 Phi S is that they do not move "
                "when the part does"
            )
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "scales", tuple(float(s) for s in self.scales))

    def stage(self, key: str) -> CouponStage:
        for stage in self.stages:
            if stage.key == key:
                return stage
        raise KeyError(f"unknown stage {key!r}; have {[s.key for s in self.stages]}")

    @property
    def total_runs(self) -> int:
        return len(self.stages) * len(self.scales) * self.plan.runs_per_coupon

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stages": [stage.to_dict() for stage in self.stages],
            "plan": self.plan.to_dict(),
            "scales": list(self.scales),
            "total_runs": self.total_runs,
            "note": self.note,
        }


def default_program() -> CouponProgram:
    """The programme as declared. Nothing has been built against it."""
    return CouponProgram(
        stages=(
            CouponStage(
                key="flat-plate",
                order=1,
                geometry="flat plate, K = 0",
                question="does the bench reproduce a sensitivity that is exactly known?",
                falsifies="the measurement chain, before any curvature is involved",
                paired_with="rolled-cylinder",
                note="b(s) = s exactly; any departure is the bench, not the model",
            ),
            CouponStage(
                key="rolled-cylinder",
                order=2,
                geometry="the same sheet rolled onto a drum, K = 0",
                question="is path sensitivity intrinsic?",
                falsifies=(
                    "the claim that only intrinsic curvature carries a pose error -- "
                    "a differential test whose common systematics cancel"
                ),
                depends_on=("flat-plate",),
                paired_with="flat-plate",
                note=(
                    "the strongest early stage: the prediction is that the two "
                    "coupons agree, so calibration offset, registration shift and "
                    "fixture datum are common and cancel in the difference"
                ),
            ),
            CouponStage(
                key="spherical-cap",
                order=3,
                geometry="spherical cap, K = 1/R^2 > 0",
                question="does the predicted focus appear where it is predicted?",
                falsifies="the conjugate-point location, which is a place and not a size",
                depends_on=("flat-plate", "rolled-cylinder"),
            ),
            CouponStage(
                key="as-built-varying",
                order=4,
                geometry="as-built saddle or torus, K varying along the path",
                question="does the general machinery hold where no closed form does?",
                falsifies=(
                    "the varying-curvature solver and, for the first time, the "
                    "surface reconstruction: on an as-built part the geometry "
                    "uncertainty term stops being zero"
                ),
                depends_on=("spherical-cap",),
            ),
        ),
        plan=PerturbationPlan(
            lateral=(0.0, 0.25, 0.5, 1.0),
            heading=(0.0, 0.25, 0.5, 1.0),
            replicates=3,
        ),
        scales=(1.0, 2.0),
        note=(
            "perturbations are fractions of the declared tolerance box, so the "
            "programme is stated in the same dimensionless terms the route "
            "criterion is"
        ),
    )


# -- conformance -----------------------------------------------------------


@dataclass(frozen=True)
class ConformanceReport:
    """What a submitted set of trials has, and what the programme still wants."""

    satisfied: bool
    stages_covered: tuple[str, ...]
    stages_missing: tuple[str, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    counts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "stages_covered": list(self.stages_covered),
            "stages_missing": list(self.stages_missing),
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "counts": dict(self.counts),
        }


def _stage_of(record: MeasurementRecord) -> str:
    """Which stage a trial belongs to, from its own declared fields."""
    return str(record.extra.get("stage", ""))


def _scale_of(record: MeasurementRecord) -> float | None:
    value = record.extra.get("scale")
    return None if value is None else float(value)


def conformance(
    program: CouponProgram, records: list[MeasurementRecord]
) -> ConformanceReport:
    """Check a submitted set of trials against the declared programme.

    Every failure here is a bookkeeping failure, and every one of them is
    cheap to design out and expensive to discover after the coupons have been
    scrapped. None of them is about whether the model agreed.
    """
    failures: list[str] = []
    warnings: list[str] = []
    by_stage: dict[str, list[MeasurementRecord]] = {}
    for record in records:
        key = _stage_of(record)
        if not key:
            failures.append(
                f"trial {record.run_id!r} names no stage; a trial that cannot be "
                "placed in the programme cannot be evidence for any part of it"
            )
            continue
        try:
            program.stage(key)
        except KeyError:
            failures.append(f"trial {record.run_id!r} names unknown stage {key!r}")
            continue
        by_stage.setdefault(key, []).append(record)

    covered = tuple(stage.key for stage in program.stages if stage.key in by_stage)
    missing = tuple(stage.key for stage in program.stages if stage.key not in by_stage)

    for stage in program.stages:
        trials = by_stage.get(stage.key)
        if not trials:
            continue

        # The stage is only usable once the ones it rests on are in hand.
        absent = [key for key in stage.depends_on if key not in by_stage]
        if absent:
            failures.append(
                f"stage {stage.key!r} has trials but depends on {absent}, which do "
                "not; a result read before its prerequisite is a result about two "
                "things at once"
            )
        if stage.paired_with and stage.paired_with not in by_stage:
            failures.append(
                f"stage {stage.key!r} is a differential test against "
                f"{stage.paired_with!r}, which has no trials; on its own it "
                "measures an absolute accuracy it was not designed to establish"
            )

        # Both axes, or the a column was never touched.
        lateral = {abs(t.perturbation.commanded_lateral) for t in trials}
        heading = {abs(t.perturbation.commanded_heading) for t in trials}
        if not any(value > 0.0 for value in lateral):
            failures.append(
                f"stage {stage.key!r} perturbs no starting lateral offset, so it "
                "says nothing about the a column of Phi"
            )
        if not any(value > 0.0 for value in heading):
            failures.append(f"stage {stage.key!r} perturbs no starting heading")
        if not (0.0 in lateral and 0.0 in heading):
            warnings.append(
                f"stage {stage.key!r} has no zero-zero control run, so nothing says "
                "what the bench reads when the start was not perturbed"
            )

        # Roles, separated by coupon and not only by run.
        roles = {role: set() for role in ROLES}
        for trial in trials:
            roles[trial.role].add(trial.coupon_id)
        shared = roles["calibration"] & roles["validation"]
        if shared:
            failures.append(
                f"stage {stage.key!r} uses coupon(s) {sorted(shared)} for both "
                "calibration and validation; two runs on one coupon share its "
                "as-built geometry, its fixturing and its calibration, so the "
                "split has to be by coupon"
            )
        for role in ROLES:
            if not roles[role]:
                failures.append(f"stage {stage.key!r} has no {role} coupons")

        # Achieved, not commanded.
        unachieved = [
            trial.run_id
            for trial in trials
            if trial.perturbation.achieved_lateral is None
            or trial.perturbation.achieved_heading is None
        ]
        if unachieved:
            failures.append(
                f"stage {stage.key!r}: trials {sorted(unachieved)[:4]} report no "
                "achieved perturbation. The difference between commanded and "
                "achieved is a starting-pose error of exactly the kind under test, "
                "so comparing against the commanded value puts the quantity being "
                "measured into the prediction's own input"
            )
        unquantified = [
            trial.run_id
            for trial in trials
            if trial.perturbation.achieved_uncertainty_lateral is None
            or trial.perturbation.achieved_uncertainty_heading is None
        ]
        if unquantified:
            warnings.append(
                f"stage {stage.key!r}: trials {sorted(unquantified)[:4]} report an "
                "achieved perturbation with no uncertainty on it, so it cannot "
                "enter the budget as the starting-pose term"
            )

        # One observation mode per stage, or the trials are not comparable.
        modes = {trial.observation_mode for trial in trials}
        if len(modes) > 1:
            failures.append(
                f"stage {stage.key!r} mixes observation modes {sorted(modes)}; they "
                "differ at the order the campaign is trying to resolve"
            )

    # Scales, for the dimensionless claim.
    scales = {
        _scale_of(record) for record in records if _scale_of(record) is not None
    }
    if len(scales) < 2:
        failures.append(
            f"the trials cover {len(scales)} scale(s) and the programme declares "
            f"{len(program.scales)}; a dimensionless claim tested at one size has "
            "not been tested"
        )

    return ConformanceReport(
        satisfied=not failures and not missing,
        stages_covered=covered,
        stages_missing=missing,
        failures=tuple(failures),
        warnings=tuple(warnings),
        counts={
            "trials": len(records),
            "trials_per_stage": {key: len(value) for key, value in by_stage.items()},
            "coupons": len({record.coupon_id for record in records}),
            "scales": sorted(scale for scale in scales),
        },
    )


#: What must agree, field for field, before two trials can be differenced at
#: all. Anything on this list that differs is not a small correction to the
#: comparison -- it is a comparison between two different quantities, and the
#: difference of two different quantities has no null hypothesis.
DIFFERENCEABLE_FIELDS: tuple[str, ...] = (
    "observation_mode",
    "observation_mode_version",
    "units",
    "coordinate_frame",
    "datum_frame",
    "filter_identifier",
    "filter_version",
    "filter_operator_digest",
    "filter_causal",
    "calibration_id",
    "reconstruction_version",
)

#: How close two achieved perturbations must be, relative to their combined
#: declared uncertainty, before the pair is treated as one condition. Two runs
#: can share a command and receive measurably different starting poses, and the
#: gap between commanded and achieved is a starting-pose error of exactly the
#: kind under test.
ACHIEVED_MATCH_SIGMAS = 1.0


@dataclass(frozen=True)
class SharedDifferential:
    """The parameters both coupons felt, and how differently each felt them.

    This is the object that decides whether a differential control actually
    cancels anything. For a parameter ``theta`` shared by the two coupons,

    .. code-block:: text

        Sigma_D,theta = (J_c - J_p) C_theta (J_c - J_p)^T

    which is zero only when ``J_c == J_p``. A calibration scale applied to both
    coupons through the same transform does cancel; a fixture datum that was
    re-established when the second coupon was mounted does not, and calling it
    common because it has the same *name* is how a differential control comes
    to look more powerful than it is.

    The cancellation is therefore *measured* here and reported as
    ``cancelled_fraction``, rather than asserted in a docstring.
    """

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    covariance: Array
    plate_jacobian: Array
    cylinder_jacobian: Array
    basis: str
    note: str = ""

    def __post_init__(self) -> None:
        from .contract import validated_covariance

        if not self.names:
            raise ValueError("a shared-parameter block must name its parameters")
        if len(self.kinds) != len(self.names):
            raise ValueError("every shared parameter needs a kind")
        if not self.basis:
            raise ValueError(
                "a shared-parameter block must say how C_theta was arrived at; "
                "'these errors are common' is a claim, not a basis"
            )
        count = len(self.names)
        object.__setattr__(
            self, "covariance", validated_covariance(self.covariance, "C_theta", count)
        )
        for name in ("plate_jacobian", "cylinder_jacobian"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.ndim != 2 or array.shape[1] != count:
                raise ValueError(f"{name} must be (n, {count}): one column per parameter")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.plate_jacobian.shape != self.cylinder_jacobian.shape:
            raise ValueError(
                "the two Jacobians are on different grids, so their difference is "
                "not a difference at matched arc lengths"
            )
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "kinds", tuple(self.kinds))

    @property
    def samples(self) -> int:
        return int(self.plate_jacobian.shape[0])

    def difference_block(self) -> Array:
        """``(J_c - J_p) C (J_c - J_p)^T``: what survives the subtraction."""
        gap = self.cylinder_jacobian - self.plate_jacobian
        return gap @ self.covariance @ gap.T

    def uncancelled_fraction(self) -> float:
        """How much of the shared variance the difference failed to remove.

        Zero when the two coupons felt the parameter identically, one when the
        subtraction achieved nothing. It is the number that says whether this
        control is differential in fact rather than in intent.
        """
        survives = float(np.trace(self.difference_block()))
        separately = float(
            np.trace(self.plate_jacobian @ self.covariance @ self.plate_jacobian.T)
            + np.trace(self.cylinder_jacobian @ self.covariance @ self.cylinder_jacobian.T)
        )
        if separately <= 0.0:
            return 0.0
        return survives / separately

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "kinds": list(self.kinds),
            "basis": self.basis,
            "samples": self.samples,
            "uncancelled_fraction": self.uncancelled_fraction(),
            "note": self.note,
        }


def _differenceable(left: MeasurementRecord, right: MeasurementRecord) -> list[str]:
    """Which of the fields that must agree do not."""
    return [
        name
        for name in DIFFERENCEABLE_FIELDS
        if getattr(left, name) != getattr(right, name)
    ]


def _achieved(record: MeasurementRecord) -> tuple[float, float] | None:
    perturbation = record.perturbation
    if perturbation.achieved_lateral is None or perturbation.achieved_heading is None:
        return None
    return (float(perturbation.achieved_lateral), float(perturbation.achieved_heading))


def _achieved_sigma(record: MeasurementRecord) -> tuple[float, float]:
    perturbation = record.perturbation
    return (
        float(perturbation.achieved_uncertainty_lateral or 0.0),
        float(perturbation.achieved_uncertainty_heading or 0.0),
    )


def intrinsic_flatness_control(
    plate: list[MeasurementRecord],
    cylinder: list[MeasurementRecord],
    *,
    predicted_difference=None,
    joint_covariance=None,
    shared: SharedDifferential | None = None,
    coverage: float = 0.95,
) -> dict[str, Any]:
    """Stage one's test, stated as two hypotheses rather than one.

    Bending a sheet onto a drum does not change its intrinsic geometry, so the
    plate and the rolled cylinder have the same transfer map -- this runtime
    computes them to agree to 1e-13. That is the **intrinsic null**, and it is
    a statement about ``Phi``.

    It is *not* a statement about what an instrument reports. The two coupons
    differ in transverse normal curvature, so their ambient-chord predictions
    differ at second order in the perturbation -- the same branch that measured
    the transfer maps agreeing to 1e-13 measured the cylinder's validity
    envelope as 15.7% tighter than the plate's, for exactly that reason. At
    finite perturbation ``y_c != y_p`` even though the intrinsic sensitivities
    agree. So the **observation-space null** is

    .. code-block:: text

        r_D = (y_c - y_p) - (yhat_c - yhat_p)

    and testing ``y_c - y_p`` against zero instead would reject a correct
    runtime on a coupon pair it predicts perfectly. ``predicted_difference`` is
    ``yhat_c - yhat_p`` and is required to test it; without one, this function
    reports the raw difference as a description and says in
    ``observation_null`` that no test was performed.

    **Cancellation is measured, not assumed.** The differential covariance is

    .. code-block:: text

        Sigma_D = Sigma_p + Sigma_c - Sigma_pc - Sigma_cp

    and the cross terms exist only if something says what the two coupons
    share. Combining two scalar uncertainties in quadrature -- which is what
    this function used to do -- assumes independence, which is the opposite of
    the cancellation the docstring was claiming. Supply either a
    ``joint_covariance`` for the difference directly, or a
    :class:`SharedDifferential` from which ``(J_c - J_p) C (J_c - J_p)^T`` is
    built. With neither, the result says
    ``differential_covariance_not_established`` and no chi-square is computed.

    Trials are paired on **achieved** perturbations, not commanded ones, and
    only when the achieved values agree to within their own declared
    uncertainty. Two runs can share a command and receive measurably different
    starting poses, and that gap is a starting-pose error of precisely the kind
    under test.
    """
    from .output_covariance import OutputCovariance

    if not plate or not cylinder:
        raise ValueError(
            "the flatness control is a comparison between two coupons and needs "
            "trials from both; either one alone is a different experiment"
        )

    matched: list[tuple[MeasurementRecord, MeasurementRecord]] = []
    refused: list[dict[str, Any]] = []
    for left in plate:
        for right in cylinder:
            clashes = _differenceable(left, right)
            if clashes:
                refused.append(
                    {
                        "plate_run": left.run_id,
                        "cylinder_run": right.run_id,
                        "reason": f"these differ and must not: {clashes}",
                    }
                )
                continue
            if list(left.arclength) != list(right.arclength):
                refused.append(
                    {
                        "plate_run": left.run_id,
                        "cylinder_run": right.run_id,
                        "reason": "different arclength grids, so the samples are not paired",
                    }
                )
                continue
            left_achieved, right_achieved = _achieved(left), _achieved(right)
            if left_achieved is None or right_achieved is None:
                refused.append(
                    {
                        "plate_run": left.run_id,
                        "cylinder_run": right.run_id,
                        "reason": (
                            "achieved perturbations are not reported; the control "
                            "pairs on what the rig did, not on what it was told"
                        ),
                    }
                )
                continue
            sigmas = [
                np.hypot(a, b)
                for a, b in zip(_achieved_sigma(left), _achieved_sigma(right), strict=True)
            ]
            gaps = [abs(a - b) for a, b in zip(left_achieved, right_achieved, strict=True)]
            if any(sigma <= 0.0 for sigma in sigmas):
                refused.append(
                    {
                        "plate_run": left.run_id,
                        "cylinder_run": right.run_id,
                        "reason": (
                            "an achieved perturbation was reported with no uncertainty, "
                            "so there is no scale on which to call two of them the same"
                        ),
                    }
                )
                continue
            if any(
                gap > ACHIEVED_MATCH_SIGMAS * sigma
                for gap, sigma in zip(gaps, sigmas, strict=True)
            ):
                refused.append(
                    {
                        "plate_run": left.run_id,
                        "cylinder_run": right.run_id,
                        "reason": (
                            f"achieved perturbations differ by {gaps} against a combined "
                            f"uncertainty of {sigmas}; propagate the difference through "
                            "the transfer map or do not pair them"
                        ),
                    }
                )
                continue
            matched.append((left, right))

    if not matched:
        raise ValueError(
            "no plate trial is differenceable against a cylinder trial at a matched "
            f"achieved perturbation. {len(refused)} pair(s) were refused: "
            f"{refused[:3]}"
        )

    predicted = (
        None if predicted_difference is None
        else np.asarray(predicted_difference, dtype=float)
    )

    rows: list[dict[str, Any]] = []
    for left, right in matched:
        observed = np.asarray(right.signed_transverse_separation, dtype=float) - np.asarray(
            left.signed_transverse_separation, dtype=float
        )
        row: dict[str, Any] = {
            "plate_run": left.run_id,
            "cylinder_run": right.run_id,
            "achieved_lateral": _achieved(left),
            "achieved_heading": _achieved(right),
            "samples": int(observed.size),
            "max_abs_observed_difference": float(np.max(np.abs(observed))),
        }
        if predicted is None:
            row["observation_null"] = "not-tested: no predicted difference was declared"
            row["residual"] = None
        else:
            if predicted.shape != observed.shape:
                raise ValueError(
                    f"the predicted difference is {predicted.shape} and the observed "
                    f"one is {observed.shape}; they are not the same comparison"
                )
            residual = observed - predicted
            row["max_abs_residual"] = float(np.max(np.abs(residual)))
            row["residual"] = residual

        total = _differential_covariance(left, right, joint_covariance, shared)
        if total is None:
            row["differential_covariance"] = "not-established"
            row["statistic"] = None
        else:
            row["differential_covariance"] = "established"
            if row.get("residual") is not None:
                covariance = OutputCovariance(
                    arclength=np.asarray(left.arclength, dtype=float),
                    outputs=("signed-transverse-separation",),
                    blocks={"observation-noise": total},
                    note="Sigma_p + Sigma_c - Sigma_pc - Sigma_cp",
                )
                row["statistic"] = covariance.accepts(row["residual"], coverage=coverage)
            else:
                row["statistic"] = None
        row.pop("residual", None)
        rows.append(row)

    established = [row for row in rows if row["differential_covariance"] == "established"]
    tested = [row for row in established if row["statistic"] is not None]
    return {
        "status": CAMPAIGN_STATUS,
        "intrinsic_null": (
            "the two coupons have the same transfer map: bending a sheet does not "
            "change its intrinsic geometry, and this runtime computes them equal "
            "to 1e-13"
        ),
        "observation_null": (
            "r_D = (y_c - y_p) - (yhat_c - yhat_p). The raw difference is NOT "
            "expected to vanish: the cylinder has a transverse normal curvature the "
            "plate does not, so the two ambient-chord predictions differ at second "
            "order in the perturbation"
            if predicted is not None
            else "not tested: no predicted difference was declared, so the observed "
            "difference below is a description and not evidence"
        ),
        "matched_pairs": len(rows),
        "refused_pairs": refused,
        "differential_covariance": (
            "established" if established else "differential_covariance_not_established"
        ),
        "shared": None if shared is None else shared.to_dict(),
        "worst_reduced_chi_square": (
            max(row["statistic"]["reduced"] for row in tested) if tested else None
        ),
        "all_consistent": (
            all(row["statistic"]["accepted"] for row in tested) if tested else None
        ),
        "note": (
            "cancellation is measured rather than assumed: a shared parameter only "
            "cancels where the two coupons felt it identically, and "
            "shared.uncancelled_fraction reports how much survived the subtraction. "
            "What counts as agreement is the instrument protocol's to declare, and "
            f"the campaign status is {CAMPAIGN_STATUS!r}"
        ),
        "pairs": rows,
    }


def _differential_covariance(
    plate: MeasurementRecord,
    cylinder: MeasurementRecord,
    joint_covariance,
    shared: SharedDifferential | None,
):
    """``Sigma_p + Sigma_c - Sigma_pc - Sigma_cp``, or ``None`` if nothing says.

    A declared joint covariance is taken as the whole answer. Otherwise the
    independent parts come from each trial's own ``measurement_covariance`` and
    the cross terms from the shared block, which collapses the four-term sum to
    ``Sigma_ind,p + Sigma_ind,c + (J_c - J_p) C (J_c - J_p)^T``.

    ``None`` is the honest answer when neither was supplied. Combining two
    scalar uncertainties in quadrature would assume independence, which is
    precisely the opposite of the cancellation a differential control claims.
    """
    from .contract import validated_covariance

    samples = len(plate.signed_transverse_separation)
    if joint_covariance is not None:
        total = validated_covariance(joint_covariance, "Sigma_D", size=None)
        if total.shape != (samples, samples):
            raise ValueError(
                f"the joint covariance is {total.shape} and the difference has "
                f"{samples} samples"
            )
        return total
    if shared is None:
        return None
    if plate.measurement_covariance is None or cylinder.measurement_covariance is None:
        return None
    if shared.samples != samples:
        raise ValueError(
            f"the shared block is on {shared.samples} samples and the difference has "
            f"{samples}"
        )
    independent = validated_covariance(
        plate.measurement_covariance, "Sigma_ind,plate", size=None
    ) + validated_covariance(
        cylinder.measurement_covariance, "Sigma_ind,cylinder", size=None
    )
    return validated_covariance(
        independent + shared.difference_block(), "Sigma_D", size=None
    )


__all__ = [
    "CAMPAIGN_STATUS",
    "ConformanceReport",
    "CouponProgram",
    "CouponStage",
    "ACHIEVED_MATCH_SIGMAS",
    "DIFFERENCEABLE_FIELDS",
    "PerturbationPlan",
    "SharedDifferential",
    "conformance",
    "default_program",
    "intrinsic_flatness_control",
]
