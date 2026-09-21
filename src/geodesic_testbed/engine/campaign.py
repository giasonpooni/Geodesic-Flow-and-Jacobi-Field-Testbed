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
   too strongly, and :func:`evaluate_differential_case` states them carefully, per pair.

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

   Each pair carries its own :class:`DifferentialCase`: its own predicted
   difference, its own covariance and its own Jacobians. A campaign runs
   several perturbations, replicates and scales, and one array applied to all
   of them would be broadcast across unlike conditions -- agreeing with one
   pair and meaning nothing for the rest.

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
from .uncertainty import CONTRIBUTIONS as UNCERTAINTY_SOURCES

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
    # The id names an instrument state; the digest is the transform that
    # actually ran. Matching the first while permitting different seconds
    # defeats the separation the two fields exist to provide.
    "calibration_transform_digest",
    "reconstruction_version",
    # Two arrays on the same grid are not the same observation if different
    # samples were dropped from them, or dropped under different rules.
    "rejected_sample_mask",
    "outlier_rule",
)

#: Default only. How close two achieved perturbations must be, relative to
#: their combined declared uncertainty, before the pair is treated as one
#: condition. Two runs can share a command and receive measurably different
#: starting poses, and that gap is a starting-pose error of exactly the kind
#: under test -- but *how close is close enough* is a policy an instrument
#: protocol declares, not a mathematical invariant, so every
#: :class:`DifferentialCase` carries its own and this is what it falls back to.
DEFAULT_ACHIEVED_MATCH_SIGMAS = 1.0




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

    The effect is therefore *measured* here, as
    :meth:`differential_to_separate_variance_ratio`, rather than asserted.
    """

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    covariance: Array
    plate_jacobian: Array
    cylinder_jacobian: Array
    basis: str
    #: The grid these Jacobians are indexed on, and the calibrations they were
    #: established against. Sample-indexed arrays on a different grid of the
    #: same length would otherwise pass every shape check: ``(J_c - J_p)`` is
    #: formed sample by sample, so two grids that merely agree in count pair
    #: sensitivities with the wrong arc lengths.
    grid_digest: str = ""
    calibration_ids: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        from .contract import validated_covariance

        if not self.names:
            raise ValueError("a shared-parameter block must name its parameters")
        if len(self.kinds) != len(self.names):
            raise ValueError("every shared parameter needs a kind")
        for kind in self.kinds:
            if kind not in UNCERTAINTY_SOURCES:
                raise ValueError(f"shared parameter kinds must be in {UNCERTAINTY_SOURCES}")
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
        if "calibration-transform" in self.kinds and not self.calibration_ids:
            raise ValueError(
                "this block carries a calibration transform but names no "
                "calibration; a shared calibration uncertainty that does not say "
                "which calibration it is cannot be held against the trials that "
                "claim to share it"
            )
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "kinds", tuple(self.kinds))
        object.__setattr__(self, "calibration_ids", tuple(self.calibration_ids))

    @property
    def samples(self) -> int:
        return int(self.plate_jacobian.shape[0])

    def difference_block(self) -> Array:
        """``(J_c - J_p) C (J_c - J_p)^T``: what survives the subtraction."""
        gap = self.cylinder_jacobian - self.plate_jacobian
        return gap @ self.covariance @ gap.T

    def differential_to_separate_variance_ratio(self) -> float:
        """What the subtraction did to the shared variance. **Not a fraction.**

        .. code-block:: text

            tr[(J_c - J_p) C (J_c - J_p)^T] / (tr[J_p C J_p^T] + tr[J_c C J_c^T])

        Zero when the two coupons felt the parameter identically -- the case a
        differential control is designed for. One when only one of them felt it
        at all, so there was nothing to cancel. **Two** when they felt it
        oppositely, ``J_c = -J_p``: differencing then *amplifies* the shared
        uncertainty rather than removing it, which is the outcome a name like
        "uncancelled fraction" would have quietly excluded. It is not clipped,
        because a value above one is the finding.
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
            "grid_digest": self.grid_digest,
            "calibration_ids": list(self.calibration_ids),
            "differential_to_separate_variance_ratio": (
                self.differential_to_separate_variance_ratio()
            ),
            "note": self.note,
        }


@dataclass(frozen=True)
class DifferentialCovariance:
    """``Sigma_D`` for one pair, with enough provenance to rule out double counting.

    ``Sigma_D = Sigma_p + Sigma_c - Sigma_pc - Sigma_cp``. Supplying the result
    is not enough on its own: when the independent parts and the cross terms
    come from different places, something has to say whether the first already
    contains what the second carries. A calibration uncertainty inside both is
    counted twice, which is not conservative -- it is wrong in the direction
    that looks like caution. This is why the components are
    :class:`IndependentCovariance` and :class:`SharedDifferential` objects
    that declare their sources, and never a record's bare
    ``measurement_covariance``, which says how big it is and nothing about
    what is inside it.

    So each component declares what it accounts for, and construction refuses
    an overlap. ``grid_digest`` pins the arclength grid the matrix belongs to;
    a covariance on a different grid pairs uncertainty with the wrong arc
    lengths and every value check passes for it.
    """

    matrix: Array
    basis: str
    independent_sources: tuple[str, ...] = ()
    shared_sources: tuple[str, ...] = ()
    calibration_ids: tuple[str, ...] = ()
    grid_digest: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        from .contract import validated_covariance

        if not self.basis:
            raise ValueError(
                "a differential covariance must say how it was arrived at; it is "
                "what every residual in the comparison is divided by"
            )
        for field_name in ("independent_sources", "shared_sources"):
            for source in getattr(self, field_name):
                if source not in UNCERTAINTY_SOURCES:
                    raise ValueError(f"{field_name} entries must be in {UNCERTAINTY_SOURCES}")
        overlap = sorted(set(self.independent_sources) & set(self.shared_sources))
        if overlap:
            raise ValueError(
                f"{overlap} is declared in both the independent components and the "
                "shared block, so it is counted twice. Counting an uncertainty "
                "twice is not conservative, it is wrong in the direction that "
                "looks like caution."
            )
        object.__setattr__(
            self, "matrix", validated_covariance(self.matrix, "Sigma_D", size=None)
        )
        for name in ("independent_sources", "shared_sources", "calibration_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def accounts_for(self) -> tuple[str, ...]:
        """Every source this matrix contains, from either side."""
        return tuple(sorted(set(self.independent_sources) | set(self.shared_sources)))

    @property
    def samples(self) -> int:
        return int(self.matrix.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "samples": self.samples,
            "independent_sources": list(self.independent_sources),
            "shared_sources": list(self.shared_sources),
            "accounts_for": list(self.accounts_for),
            "calibration_ids": list(self.calibration_ids),
            "grid_digest": self.grid_digest,
            "note": self.note,
        }


@dataclass(frozen=True)
class IndependentCovariance:
    """One trial's own covariance, with the sources it is declared to contain.

    A bare ``measurement_covariance`` off a record says how big it is and
    nothing about what is inside it. That is enough to validate and not enough
    to add: if it already contains the calibration transform, and a shared
    block carries the calibration transform too, the sum counts it twice and
    no check on either half can see it.

    So the independent components of a differential covariance are declared
    here, not lifted from the records. ``sources`` is what this matrix
    contains; construction of the difference refuses any source that also
    appears in the shared block.
    """

    matrix: Array
    basis: str
    sources: tuple[str, ...]
    calibration_ids: tuple[str, ...] = ()
    grid_digest: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        from .contract import validated_covariance

        if not self.basis:
            raise ValueError(
                "an independent covariance must say how it was arrived at"
            )
        if not self.sources:
            raise ValueError(
                "an independent covariance must name the sources it contains; "
                "an undeclared one cannot be added to a shared block without "
                "risking counting something twice"
            )
        for source in self.sources:
            if source not in UNCERTAINTY_SOURCES:
                raise ValueError(f"sources must be in {UNCERTAINTY_SOURCES}")
        object.__setattr__(
            self, "matrix", validated_covariance(self.matrix, "Sigma_ind", size=None)
        )
        for name in ("sources", "calibration_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def samples(self) -> int:
        return int(self.matrix.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "samples": self.samples,
            "sources": list(self.sources),
            "calibration_ids": list(self.calibration_ids),
            "grid_digest": self.grid_digest,
            "note": self.note,
        }


@dataclass(frozen=True)
class DifferentialCase:
    """One plate-cylinder pair, with the prediction and covariance that are *its*.

    A campaign runs several perturbations, replicates and scales. Each pair has
    a different predicted difference, a different covariance and different
    parameter Jacobians, and a single array applied to all of them would be
    silently broadcast across unlike conditions -- agreeing with one and
    meaning nothing for the rest.

    ``difference_covariance`` is ``Sigma_D``, the ``(n, n)`` covariance of the
    difference itself, and is named for what it is. A *joint* covariance would
    be the ``(2n, 2n)`` block matrix over the two trials stacked, from which
    ``Sigma_D = D Sigma_joint D^T`` with ``D = [-I  I]``;
    :meth:`from_joint` builds a case that way for a caller who has one.
    """

    plate_run_id: str
    cylinder_run_id: str
    predicted_difference: Array
    prediction_digest: str
    difference_covariance: DifferentialCovariance | None = None
    shared_parameters: SharedDifferential | None = None
    #: The two trials' own covariances, *with* the sources they contain. Bare
    #: record covariances are not used: an undeclared matrix cannot be added
    #: to a shared block without risking counting a source twice.
    plate_independent: IndependentCovariance | None = None
    cylinder_independent: IndependentCovariance | None = None
    coverage: float = 0.95
    achieved_match_sigmas: float = DEFAULT_ACHIEVED_MATCH_SIGMAS
    note: str = ""

    def __post_init__(self) -> None:
        if not self.plate_run_id or not self.cylinder_run_id:
            raise ValueError("a case must name both runs it applies to")
        if not self.prediction_digest:
            raise ValueError(
                "a case must name the prediction it was computed from; a predicted "
                "difference with no provenance cannot be replayed or argued with"
            )
        predicted = np.asarray(self.predicted_difference, dtype=float)
        if predicted.ndim != 1 or not predicted.size:
            raise ValueError("the predicted difference must be one value per sample")
        if not np.all(np.isfinite(predicted)):
            raise ValueError("the predicted difference must be finite")
        predicted.setflags(write=False)
        object.__setattr__(self, "predicted_difference", predicted)
        if not 0.0 < float(self.coverage) < 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1")
        if float(self.achieved_match_sigmas) <= 0.0:
            raise ValueError(
                "achieved_match_sigmas is how close two starting poses must be to "
                "count as one condition; zero or negative admits nothing"
            )
        for other, name in (
            (self.difference_covariance, "difference_covariance"),
            (self.shared_parameters, "shared_parameters"),
            (self.plate_independent, "plate_independent"),
            (self.cylinder_independent, "cylinder_independent"),
        ):
            if other is not None and other.samples != predicted.size:
                raise ValueError(
                    f"{name} is on {other.samples} samples and the predicted "
                    f"difference has {predicted.size}"
                )
        # Exactly three states, and nothing between them. A partial one is a
        # case that looks equipped and is not: components without a shared
        # block are silently ignored by the assembly and the pair comes back
        # not-established, which reads as "nobody supplied a covariance"
        # rather than "the covariance you supplied was not usable".
        components = {
            "plate_independent": self.plate_independent,
            "cylinder_independent": self.cylinder_independent,
            "shared_parameters": self.shared_parameters,
        }
        supplied = {name for name, value in components.items() if value is not None}
        complete = self.difference_covariance is not None
        if complete and supplied:
            raise ValueError(
                "a case declares either a complete difference_covariance or the "
                f"components to build one from, not both: this one has both, and "
                f"{sorted(supplied)} would be ignored. Nothing would then say "
                "which produced the covariance a verdict was read against."
            )
        if supplied and supplied != set(components):
            missing = sorted(set(components) - supplied)
            raise ValueError(
                f"this case supplies {sorted(supplied)} and not {missing}. The "
                "component route needs both independent covariances and the "
                "shared block together; a partial set is ignored by the assembly "
                "and comes back not-established, which reads as though nobody "
                "supplied anything."
            )
        if supplied:
            overlap = sorted(
                (set(self.plate_independent.sources) | set(self.cylinder_independent.sources))
                & set(self.shared_parameters.kinds)
            )
            if overlap:
                raise ValueError(
                    f"{overlap} is declared in the independent covariances and in "
                    "the shared block, so the difference would carry it twice"
                )

    @classmethod
    def from_joint(
        cls,
        *,
        joint_covariance,
        basis: str,
        **fields: Any,
    ) -> DifferentialCase:
        """Build a case from a true ``(2n, 2n)`` joint covariance.

        ``Sigma_D = D Sigma_joint D^T`` with ``D = [-I  I]``, which is where
        the cross-covariance blocks actually enter. A caller who has the joint
        matrix should pass it here rather than forming the difference by hand.
        """
        from .contract import validated_covariance

        joint = validated_covariance(joint_covariance, "Sigma_joint", size=None)
        if joint.shape[0] % 2:
            raise ValueError(
                f"a joint covariance is (2n, 2n) over the two trials stacked; "
                f"{joint.shape[0]} is odd"
            )
        samples = joint.shape[0] // 2
        selector = np.hstack([-np.eye(samples), np.eye(samples)])
        difference = selector @ joint @ selector.T
        covariance = DifferentialCovariance(
            matrix=difference,
            basis=basis,
            note="formed as D Sigma_joint D^T with D = [-I  I]",
            **{
                key: fields.pop(key)
                for key in (
                    "independent_sources",
                    "shared_sources",
                    "calibration_ids",
                    "grid_digest",
                )
                if key in fields
            },
        )
        return cls(difference_covariance=covariance, **fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plate_run_id": self.plate_run_id,
            "cylinder_run_id": self.cylinder_run_id,
            "samples": int(self.predicted_difference.size),
            "prediction_digest": self.prediction_digest,
            "coverage": float(self.coverage),
            "achieved_match_sigmas": float(self.achieved_match_sigmas),
            "difference_covariance": (
                None if self.difference_covariance is None
                else self.difference_covariance.to_dict()
            ),
            "shared_parameters": (
                None if self.shared_parameters is None else self.shared_parameters.to_dict()
            ),
            "note": self.note,
        }


def grid_digest(arclength) -> str:
    """A canonical digest of an arclength grid.

    Canonicalised first, so the digest identifies the grid rather than the
    last bits of whatever computed it. A covariance carries the digest of the
    grid it belongs to, and pairing it with any other grid is refused: every
    value check passes for a matrix on the wrong arc lengths.
    """
    from .canonical import content_hash

    return content_hash({"arclength": [float(value) for value in arclength]})


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


def _pairing_refusal(
    plate: MeasurementRecord, cylinder: MeasurementRecord, match_sigmas: float
) -> str | None:
    """Why these two trials cannot be differenced, or ``None`` if they can."""
    clashes = _differenceable(plate, cylinder)
    if clashes:
        return f"these differ and must not: {clashes}"
    if list(plate.arclength) != list(cylinder.arclength):
        return "different arclength grids, so the samples are not paired"
    plate_achieved, cylinder_achieved = _achieved(plate), _achieved(cylinder)
    if plate_achieved is None or cylinder_achieved is None:
        return (
            "achieved perturbations are not reported; the control pairs on what "
            "the rig did, not on what it was told"
        )
    sigmas = [
        float(np.hypot(a, b))
        for a, b in zip(_achieved_sigma(plate), _achieved_sigma(cylinder), strict=True)
    ]
    if any(sigma <= 0.0 for sigma in sigmas):
        return (
            "an achieved perturbation was reported with no uncertainty, so there "
            "is no scale on which to call two of them the same"
        )
    gaps = [abs(a - b) for a, b in zip(plate_achieved, cylinder_achieved, strict=True)]
    if any(gap > match_sigmas * sigma for gap, sigma in zip(gaps, sigmas, strict=True)):
        return (
            f"achieved perturbations differ by {gaps} against a combined uncertainty "
            f"of {sigmas} at {match_sigmas} sigma; propagate the difference through "
            "the transfer map or do not pair them"
        )
    return None


def _case_covariance(
    case: DifferentialCase, plate: MeasurementRecord, cylinder: MeasurementRecord
):
    """``Sigma_D`` for this pair, or ``None`` when nothing established one.

    A declared :class:`DifferentialCovariance` is the whole answer. Otherwise
    it is built from the two *declared* independent components and the shared
    block, which collapses the four-term sum to

    .. code-block:: text

        Sigma_ind,p + Sigma_ind,c + (J_c - J_p) C (J_c - J_p)^T

    The independent parts are the case's, never the records' bare
    ``measurement_covariance``. That is the difference between a closed
    double-counting route and a docstring claiming one: an undeclared matrix
    says how big it is and nothing about what is inside it, so adding it to a
    shared block that may carry the same source is exactly the mistake the
    overlap check exists to catch, and it cannot catch it.

    ``None`` is the honest answer when neither route was supplied.
    """
    if case.difference_covariance is not None:
        return case.difference_covariance
    shared = case.shared_parameters
    if shared is None:
        return None
    # __post_init__ guarantees both components are present alongside a shared
    # block, and that their sources are disjoint from its kinds.
    plate_part = case.plate_independent
    cylinder_part = case.cylinder_independent
    return DifferentialCovariance(
        matrix=plate_part.matrix + cylinder_part.matrix + shared.difference_block(),
        basis=(
            f"{plate_part.basis} and {cylinder_part.basis} as the independent "
            f"parts, and {shared.basis} for the shared block"
        ),
        independent_sources=tuple(
            sorted(set(plate_part.sources) | set(cylinder_part.sources))
        ),
        shared_sources=shared.kinds,
        calibration_ids=tuple(
            sorted(
                set(plate_part.calibration_ids)
                | set(cylinder_part.calibration_ids)
                | set(shared.calibration_ids)
            )
        ),
        grid_digest=plate_part.grid_digest,
        note="Sigma_ind,p + Sigma_ind,c + (J_c - J_p) C (J_c - J_p)^T",
    )


def _check_bindings(
    case: DifferentialCase, plate: MeasurementRecord, cylinder: MeasurementRecord
) -> None:
    """Hold the declared digests and ids against what the trials actually say.

    Provenance fields that nothing verifies are labels. Each of these is a way
    a case can be attached to the wrong evidence and produce a plausible
    number: a covariance on another run's grid, a component certified against
    a calibration neither trial used, a predicted difference from a different
    report.
    """
    expected_grid = grid_digest(plate.arclength)
    for component, name in (
        (case.difference_covariance, "difference_covariance"),
        (case.plate_independent, "plate_independent"),
        (case.cylinder_independent, "cylinder_independent"),
        (case.shared_parameters, "shared_parameters"),
    ):
        if component is None:
            continue
        if not component.grid_digest:
            raise ValueError(
                f"{name} carries no grid_digest, so nothing says which arclength "
                "grid it belongs to; a covariance on the wrong grid pairs "
                "uncertainty with the wrong arc lengths and passes every value check"
            )
        if component.grid_digest != expected_grid:
            raise ValueError(
                f"{name} is for grid {component.grid_digest} and these trials are "
                f"on {expected_grid}"
            )
        declared = set(component.calibration_ids)
        if declared and not declared <= {plate.calibration_id, cylinder.calibration_id}:
            raise ValueError(
                f"{name} is certified against {sorted(declared)} and these trials "
                f"ran under {sorted({plate.calibration_id, cylinder.calibration_id})}"
            )
    reports = {plate.prediction_report_digest, cylinder.prediction_report_digest}
    # Set equality, not membership. Membership passes when the case matches one
    # trial and not the other, which is the ambiguous half of a binding: the
    # field declares *one* prediction, so both trials must have been compared
    # against it. Two different per-record reports need two parent digests or a
    # pair-prediction manifest, which is the adapter's to introduce -- not this
    # single field quietly meaning either.
    if reports != {case.prediction_digest}:
        raise ValueError(
            f"the case's predicted difference cites {case.prediction_digest!r}, and "
            f"these trials were compared against {sorted(reports)}. One prediction "
            "digest must be the prediction report of both trials; a digest matching "
            "one of them describes a comparison that was never made."
        )


def evaluate_differential_case(
    case: DifferentialCase,
    plate: MeasurementRecord,
    cylinder: MeasurementRecord,
) -> dict[str, Any]:
    """One pair, against the prediction and covariance declared for *it*.

    Returns the residual statistics when a covariance was established and says
    why not when one was not. The observation-space null is

    .. code-block:: text

        r_D = (y_c - y_p) - (yhat_c - yhat_p)

    and never ``y_c - y_p`` against zero: the cylinder has a transverse normal
    curvature the plate does not, so the two ambient-chord predictions differ
    at second order in the perturbation even though the transfer maps agree.
    """
    from .output_covariance import OutputCovariance

    if plate.run_id != case.plate_run_id or cylinder.run_id != case.cylinder_run_id:
        raise ValueError(
            f"this case is for {case.plate_run_id!r} against {case.cylinder_run_id!r}, "
            f"not {plate.run_id!r} against {cylinder.run_id!r}"
        )
    refusal = _pairing_refusal(plate, cylinder, float(case.achieved_match_sigmas))
    if refusal is not None:
        raise ValueError(
            f"{case.plate_run_id!r} and {case.cylinder_run_id!r} cannot be "
            f"differenced: {refusal}"
        )
    _check_bindings(case, plate, cylinder)

    observed = np.asarray(cylinder.signed_transverse_separation, dtype=float) - np.asarray(
        plate.signed_transverse_separation, dtype=float
    )
    if observed.shape != case.predicted_difference.shape:
        raise ValueError(
            f"the case predicts {case.predicted_difference.size} samples and the "
            f"trials carry {observed.size}"
        )
    residual = observed - case.predicted_difference

    row: dict[str, Any] = {
        "plate_run": plate.run_id,
        "cylinder_run": cylinder.run_id,
        "plate_achieved": _achieved(plate),
        "cylinder_achieved": _achieved(cylinder),
        "samples": int(observed.size),
        "prediction_digest": case.prediction_digest,
        "max_abs_observed_difference": float(np.max(np.abs(observed))),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }

    covariance = _case_covariance(case, plate, cylinder)
    if covariance is None:
        row["differential_covariance"] = "not-established"
        row["covariance"] = None
        row["statistic"] = None
        return row
    if covariance.samples != observed.size:
        raise ValueError(
            f"Sigma_D is on {covariance.samples} samples and the difference has "
            f"{observed.size}"
        )
    total = OutputCovariance(
        arclength=np.asarray(plate.arclength, dtype=float),
        outputs=("signed-transverse-separation",),
        blocks={"observation-noise": covariance.matrix},
        note="Sigma_p + Sigma_c - Sigma_pc - Sigma_cp",
    )
    row["differential_covariance"] = "established"
    row["covariance"] = covariance.to_dict()
    row["statistic"] = total.accepts(residual, coverage=float(case.coverage))
    return row


def campaign_flatness_control(
    cases: list[DifferentialCase], records: list[MeasurementRecord]
) -> dict[str, Any]:
    """Every declared pair, each against its own prediction and covariance.

    The campaign-wide boolean is reported **only** when every matched pair was
    actually tested. One tested pair among eight would otherwise let
    ``all_consistent`` read ``True`` while seven were never examined, which is
    the shape of a result that is worse than no result.
    """
    counts: dict[str, int] = {}
    for record in records:
        counts[record.run_id] = counts.get(record.run_id, 0) + 1
    duplicates = sorted(run_id for run_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"two or more trials share the run ids {duplicates}. A run id is how a "
            "case names its evidence, so a duplicate silently decides which trial "
            "the comparison used"
        )
    by_run = {record.run_id: record for record in records}
    if not cases:
        raise ValueError(
            "a differential control needs at least one declared pair; a campaign "
            "with no cases has nothing to test"
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        missing = [
            run_id
            for run_id in (case.plate_run_id, case.cylinder_run_id)
            if run_id not in by_run
        ]
        if missing:
            raise ValueError(
                f"a case names {missing}, which is not among the supplied trials"
            )
        rows.append(
            evaluate_differential_case(
                case, by_run[case.plate_run_id], by_run[case.cylinder_run_id]
            )
        )

    tested = [row for row in rows if row["statistic"] is not None]
    untested = [row for row in rows if row["statistic"] is None]
    if not untested:
        status = "complete"
    elif tested:
        status = "partial"
    else:
        status = "differential_covariance_not_established"

    return {
        "status": CAMPAIGN_STATUS,
        "intrinsic_null": (
            "the two coupons have the same transfer map: bending a sheet does not "
            "change its intrinsic geometry, and this runtime computes them equal "
            "to 1e-13"
        ),
        "observation_null": (
            "r_D = (y_c - y_p) - (yhat_c - yhat_p). The raw difference is NOT "
            "expected to vanish: the cylinder has a transverse normal curvature "
            "the plate does not, so the two ambient-chord predictions differ at "
            "second order in the perturbation"
        ),
        "matched_pairs": len(rows),
        "tested_pairs": len(tested),
        "untested_pairs": len(untested),
        "covariance_status": status,
        # Only a boolean when there is nothing left out of it.
        "all_tested_consistent": (
            all(row["statistic"]["accepted"] for row in tested)
            if tested and not untested
            else None
        ),
        "worst_reduced_chi_square": (
            max(row["statistic"]["reduced"] for row in tested) if tested else None
        ),
        "note": (
            "cancellation is measured rather than assumed: a shared parameter only "
            "cancels where the two coupons felt it identically, and "
            "differential_to_separate_variance_ratio reports what the subtraction "
            "did to it -- two when they felt it oppositely, which amplifies. What "
            "counts as agreement is the instrument protocol's to declare, and the "
            f"campaign status is {CAMPAIGN_STATUS!r}"
        ),
        "pairs": rows,
    }


__all__ = [
    "CAMPAIGN_STATUS",
    "ConformanceReport",
    "CouponProgram",
    "CouponStage",
    "DEFAULT_ACHIEVED_MATCH_SIGMAS",
    "DIFFERENCEABLE_FIELDS",
    "DifferentialCase",
    "DifferentialCovariance",
    "IndependentCovariance",
    "PerturbationPlan",
    "SharedDifferential",
    "campaign_flatness_control",
    "conformance",
    "grid_digest",
    "evaluate_differential_case",
    "default_program",
]
