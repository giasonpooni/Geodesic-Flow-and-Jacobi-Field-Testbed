"""The record a physical trial has to produce for its numbers to count.

Nothing in this repository has been measured. What this module supplies is the
shape of the evidence: ``path-sensitivity-observation-v1``, the record a bench
emits per trial and that a prediction is compared against.

It exists as a contract rather than as data because the failure modes of a
validation campaign are bookkeeping failures, and they are cheap to design out
and expensive to discover afterwards:

* a measurement compared against a prediction in a *different* observation mode
  -- an in-surface distance against a reconstructed chord -- differs at exactly
  the order the campaign is trying to resolve;
* a trial used both to tune the model and to demonstrate it proves nothing, so
  every record declares its ``role``;
* a separation measured as nearest-point distance rather than at matched
  nominal arc length is a different quantity, and the difference is second
  order in the perturbation -- again the order of interest;
* a filtered trajectory compared against an unfiltered prediction, with the
  raw sensor's uncertainty: a linear filter ``F`` turns the comparison into
  ``F H Phi dz0`` against ``F R F^T``, and reporting the raw ``R`` understates
  the uncertainty and correlates samples that are being treated as
  independent. The filter is therefore part of the record, with its version,
  its parameters, whether it was causal, and what it was tuned on;
* a result that cannot be traced back to the calibration and the raw data it
  came from cannot be re-analysed when the calibration turns out to be wrong.

Every field here is one of those. The record carries digests rather than the
data itself, so it stays small enough to publish beside the result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .contract import CalibrationBinding, validated_covariance
from .observation import mode as observation_mode
from .observation_model import FilteredPrediction
from .output_covariance import OutputCovariance
from .record import TransferRecord, to_transfer_record

MEASUREMENT_SCHEMA = "path-sensitivity-observation-v1"

#: What a trial is allowed to be used for. Keeping these apart is the single
#: most important discipline in the campaign.
ROLES: tuple[str, ...] = ("calibration", "validation")


@dataclass(frozen=True)
class Perturbation:
    """The starting-pose offset that was commanded, and what was achieved."""

    commanded_lateral: float
    commanded_heading: float
    achieved_lateral: float | None = None
    achieved_heading: float | None = None
    achieved_uncertainty_lateral: float | None = None
    achieved_uncertainty_heading: float | None = None

    def __post_init__(self) -> None:
        for name in ("commanded_lateral", "commanded_heading"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        for name in ("achieved_lateral", "achieved_heading"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when reported")
        for name in ("achieved_uncertainty_lateral", "achieved_uncertainty_heading"):
            value = getattr(self, name)
            if value is None:
                continue
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative when reported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Uncertainty:
    """The budget, split so that a disagreement can be attributed."""

    metrology: float
    fixture: float
    repeatability: float
    surface_geometry: float
    combined: float | None = None

    def __post_init__(self) -> None:
        # A negative or non-finite component silently produces a misleading
        # signal-to-noise ratio downstream -- often an infinite one, which
        # reads as "perfectly resolved".
        for name in ("metrology", "fixture", "repeatability", "surface_geometry"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} uncertainty must be finite and nonnegative")
        if self.combined is not None:
            value = float(self.combined)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("combined uncertainty must be finite and nonnegative")

    def total(self) -> float:
        """Combined standard uncertainty, root-sum-square unless one was given."""
        if self.combined is not None:
            return float(self.combined)
        return float(
            np.sqrt(
                self.metrology**2
                + self.fixture**2
                + self.repeatability**2
                + self.surface_geometry**2
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"total": self.total()}


@dataclass(frozen=True)
class MeasurementRecord:
    """One physical trial, in the form a comparison can actually use."""

    coupon_id: str
    run_id: str
    role: str
    observation_mode: str
    observation_mode_version: int
    geometry_model: str
    geometry_model_digest: str
    reconstruction_version: str
    calibration_id: str
    filter_identifier: str
    filter_version: str
    filter_causal: bool
    #: Digest of the filter operator actually applied. A name and a version say
    #: which filter was meant; this says which one ran, and it is what a
    #: filtered prediction has to match.
    filter_operator_digest: str
    units: dict[str, str]
    coordinate_frame: str
    datum_frame: str
    calibration_transform_digest: str
    as_built_scan_digest: str
    raw_data_digest: str
    prediction_report_digest: str
    perturbation: Perturbation
    uncertainty: Uncertainty
    measurement_covariance: list[list[float]] | None = None
    filter_parameters: dict[str, Any] = field(default_factory=dict)
    filter_group_delay: float = 0.0
    filter_tuned_on: str = ""
    input_sampling_rate: float | None = None
    output_sampling_rate: float | None = None
    rejected_sample_mask: list[bool] = field(default_factory=list)
    outlier_rule: str = ""
    arclength: list[float] = field(default_factory=list)
    signed_transverse_separation: list[float] = field(default_factory=list)
    separation_convention: str = "matched-nominal-arclength"
    run_order_index: int | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, not {self.role!r}")
        declared = observation_mode(self.observation_mode)
        if declared.version != self.observation_mode_version:
            raise ValueError(
                f"{self.observation_mode!r} is at version {declared.version}, but this "
                f"record was written against version {self.observation_mode_version}"
            )
        if self.separation_convention != "matched-nominal-arclength":
            raise ValueError(
                "separation must be measured at matched nominal arc length; a "
                "nearest-point distance is a different quantity and differs at "
                "the order this campaign is trying to resolve"
            )
        if len(self.arclength) != len(self.signed_transverse_separation):
            raise ValueError("arclength and separation must have the same length")
        if self.rejected_sample_mask and len(self.rejected_sample_mask) != len(
            self.arclength
        ):
            raise ValueError("the rejected-sample mask must cover every sample")
        if any(self.rejected_sample_mask) and not self.outlier_rule:
            raise ValueError(
                "samples were rejected but no outlier rule was declared; a "
                "discarded observation with no stated rule is not evidence"
            )
        if self.filter_identifier != "none":
            if not self.filter_version:
                raise ValueError("a declared filter must carry a version")
            if not self.filter_operator_digest:
                raise ValueError(
                    "a filtered trial must carry the digest of the operator that "
                    "filtered it; a prediction cannot otherwise prove it used the "
                    "same one"
                )
        if len(self.arclength) < 2:
            raise ValueError("a trial needs at least two samples")
        grid = np.asarray(self.arclength, dtype=float)
        if not np.all(np.isfinite(grid)):
            raise ValueError("arclength must be finite")
        if not np.all(np.diff(grid) > 0.0):
            raise ValueError(
                "arclength must be strictly increasing; a repeated or reversed "
                "sample is not a position along the path"
            )
        if not np.all(np.isfinite(np.asarray(self.signed_transverse_separation, dtype=float))):
            raise ValueError("signed_transverse_separation must be finite")
        if self.uncertainty.total() <= 0.0:
            raise ValueError(
                "the combined uncertainty must be positive; a trial with none "
                "declared reports an infinite signal-to-noise ratio"
            )
        for name in ("input_sampling_rate", "output_sampling_rate"):
            value = getattr(self, name)
            if value is None:
                continue
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive when declared")
        if not {"length", "angle"} <= set(self.units):
            raise ValueError("units must declare both 'length' and 'angle'")
        if self.measurement_covariance is not None:
# Admitted or refused, never repaired -- the shared validator in
            # contract.py, with size=None because this covariance is over the
            # trial's whole observation vector rather than a 2x2 pose.
            matrix = validated_covariance(
                self.measurement_covariance, "measurement_covariance", size=None
            )
            # Being a covariance is not enough: it has to be a covariance *of
            # this trial*. A 1x1 on a three-sample run passes every value check
            # above and is a covariance of something else, so the shape is
            # bound to the observation vector it claims to describe.
            expected = len(self.signed_transverse_separation)
            if not expected:
                raise ValueError(
                    "a measurement covariance was declared but the trial carries no "
                    "measured separations for it to be the covariance of"
                )
            if matrix.shape[0] != expected:
                raise ValueError(
                    f"measurement_covariance is {matrix.shape[0]}x{matrix.shape[0]} and "
                    f"this trial measured {expected} separations; a covariance that is "
                    "not on the observation vector pairs uncertainty with the wrong "
                    "arc lengths"
                )
        for name in (
            "geometry_model_digest",
            "calibration_transform_digest",
            "as_built_scan_digest",
            "raw_data_digest",
            "prediction_report_digest",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be set; an untraceable trial is not evidence")

    @property
    def signal_to_noise(self) -> float:
        """Largest measured separation against the combined uncertainty.

        A campaign whose residual of interest is comparable with this number
        cannot resolve the effect it is looking for, and is better re-planned
        than run.
        """
        # ``total`` is positive by construction, checked at __post_init__.
        return float(
            np.max(np.abs(self.signed_transverse_separation)) / self.uncertainty.total()
        )

    def to_dict(self) -> dict[str, Any]:
        if self.measurement_covariance is not None:
            validated_covariance(self.measurement_covariance, "measurement_covariance", size=None)
        payload = asdict(self)
        payload["schema"] = MEASUREMENT_SCHEMA
        payload["perturbation"] = self.perturbation.to_dict()
        payload["uncertainty"] = self.uncertainty.to_dict()
        payload["signal_to_noise"] = self.signal_to_noise
        return payload

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def prediction_binding(record: MeasurementRecord) -> CalibrationBinding:
    """The trial's calibration state, in the contract's vocabulary.

    The instrument names its calibration one way and the boundary contract
    another; this is the single place the two are translated, so a comparison
    can ask whether a prediction and a trial share an instrument state without
    either side learning the other's field names.
    """
    return CalibrationBinding(
        calibration_ids=(record.calibration_id,) if record.calibration_id else (),
        registration_id=record.calibration_transform_digest,
        reconstruction_version=record.reconstruction_version,
        instrument_id=record.geometry_model,
        note=f"from trial {record.run_id}",
    )


def compare(
    record: MeasurementRecord,
    predicted_separation,
    *,
    mode: str | None = None,
    resolvability_threshold: float | None = None,
    prediction_source: Any = None,
    covariance: OutputCovariance | None = None,
    coverage: float = 0.95,
):
    """Residual of a prediction against a trial, refusing a mismatched comparison.

    ``predicted_separation`` is sampled on the trial's own ``arclength``. For a
    trial that was filtered it must be a
    :class:`~geodesic_testbed.engine.observation_model.FilteredPrediction`,
    whose identifier, version and *operator digest* all have to match the
    record: a prediction is only comparable with a filtered measurement if it
    went through the same operator, and a boolean claim that it did is not
    evidence. Build one with
    :func:`~geodesic_testbed.engine.observation_model.apply_filter`.

    Two checks are the point of this function. The mode check: an in-surface
    prediction against a reconstructed chord disagrees at second order in the
    perturbation, which is the same order as the model's own failure, so the
    comparison would read as a model failure that is really a
    units-of-measurement error. The filter check: the same, for the smoothing.

    ``covariance`` is the assembled ``Sigma_y`` the residual is judged against
    -- ``A C0 A^T + J C_theta J^T + R + Sigma_num``, from
    :mod:`~geodesic_testbed.engine.output_covariance`. When it is supplied, or
    when the trial declares its own ``measurement_covariance``, the result
    carries the whitened residual, the chi-square with its degrees of freedom,
    a *two-sided* acceptance band and the empirical interval coverage. When
    neither is available the result says so in ``residual_statistics`` rather
    than leaving the scalars to be read as a verdict: a maximum absolute
    residual throws away the covariance, cannot be compared between
    instruments, and cannot be held to any threshold that is not already in
    the measurement's own units.

    ``resolvability_threshold``, when given, is the signal-to-noise bar the
    *instrument protocol* declares. This module reports the ratio and never
    invents the bar.

    ``prediction_source`` is the transfer record the prediction came from. It
    is optional because a prediction can be handed over as bare numbers, and
    that is the case worth discouraging: supplying the record lets this
    function check the two things the numbers cannot carry. Units, because a
    prediction in metres against a trial in millimetres is a thousandfold error
    that agrees in shape. And calibration, because a record *bound* to one
    instrument state and a trial run under another are not comparable however
    well they agree. A record that declares no calibration is not an error --
    a prediction from an analytic surface correctly declares none -- but the
    result says so rather than leaving the reader to assume a tie that was
    never established.
    """
    if record.measurement_covariance is not None:
        validated_covariance(record.measurement_covariance, "measurement_covariance", size=None)
    expected = mode or record.observation_mode
    if expected != record.observation_mode:
        raise ValueError(
            f"prediction is in {expected!r} but the trial measured "
            f"{record.observation_mode!r}; convert one before comparing"
        )

    prediction_record: TransferRecord | None = (
        None if prediction_source is None else to_transfer_record(prediction_source)
    )
    calibration: dict[str, Any] = {
        "trial": prediction_binding(record).to_dict(),
        "prediction": (
            None if prediction_record is None else prediction_record.calibration.to_dict()
        ),
        "agreed": None,
    }
    if prediction_record is not None:
        trial_units = dict(record.units)
        declared = prediction_record.units.to_dict()
        clashes = [
            (key, declared[key], trial_units[key])
            for key in declared
            if key in trial_units and declared[key] != trial_units[key]
        ]
        if clashes:
            detail = "; ".join(
                f"the prediction's {key} unit is {mine!r} but the trial's is {theirs!r}"
                for key, mine, theirs in clashes
            )
            raise ValueError(
                f"{detail}. A comparison in two different units agrees in shape and "
                "disagrees by a scale factor nothing else here would catch"
            )
        bound = prediction_record.calibration
        if bound.bound:
            agreed = bound.agrees_with(prediction_binding(record))
            calibration["agreed"] = agreed
            if not agreed:
                raise ValueError(
                    f"the prediction is bound to calibration {sorted(bound.calibration_ids)} "
                    f"and the trial ran under {record.calibration_id!r} at reconstruction "
                    f"{record.reconstruction_version!r}; a prediction bound to a different "
                    "instrument state is not comparable with this measurement"
                )

    filtered = isinstance(predicted_separation, FilteredPrediction)
    if record.filter_identifier != "none":
        if not filtered:
            raise ValueError(
                f"the trial was filtered by {record.filter_identifier!r} "
                f"{record.filter_version!r}, so the prediction must be a "
                "FilteredPrediction carrying the operator it went through, and "
                "be compared against F R F^T"
            )
        artifact: FilteredPrediction = predicted_separation
        mismatches = [
            (name, mine, theirs)
            for name, mine, theirs in (
                ("identifier", record.filter_identifier, artifact.identifier),
                ("version", record.filter_version, artifact.version),
                ("operator digest", record.filter_operator_digest,
                 artifact.operator_digest),
            )
            if mine != theirs
        ]
        if mismatches:
            detail = "; ".join(
                f"the trial's filter {name} is {mine!r} but the prediction's is {theirs!r}"
                for name, mine, theirs in mismatches
            )
            raise ValueError(
                f"{detail}. A prediction filtered by a different operator is not "
                "comparable with this measurement, whatever the two are called"
            )
        if record.filter_causal != artifact.causal:
            raise ValueError(
                "the trial and the prediction disagree about whether the filter was "
                "causal; a zero-phase smoother cannot stand in for a real-time one"
            )
        predicted = np.asarray(artifact.values, dtype=float)
    elif filtered:
        raise ValueError(
            "the prediction was filtered but the trial was not; filtering one side "
            "of a comparison biases it towards agreement"
        )
    else:
        predicted = np.asarray(predicted_separation, dtype=float)

    measured = np.asarray(record.signed_transverse_separation, dtype=float)
    if predicted.shape != measured.shape:
        raise ValueError("prediction must be sampled on the trial's arclength")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("the prediction must be finite")
    residual = measured - predicted
    total = record.uncertainty.total()
    snr = record.signal_to_noise
    result: dict[str, Any] = {
        "observation_mode": record.observation_mode,
        "role": record.role,
        "max_abs_residual": float(np.max(np.abs(residual))),
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
        "residual_over_uncertainty": float(np.max(np.abs(residual)) / total),
        "signal_to_noise": snr,
        "resolvability_threshold": (
            None if resolvability_threshold is None else float(resolvability_threshold)
        ),
        "meets_declared_resolvability": (
            None if resolvability_threshold is None
            else bool(snr >= float(resolvability_threshold))
        ),
        "calibration": calibration,
        "prediction_report_digest": record.prediction_report_digest,
        "raw_data_digest": record.raw_data_digest,
        "filter": {
            "identifier": record.filter_identifier,
            "version": record.filter_version,
            "causal": record.filter_causal,
            "operator_digest": record.filter_operator_digest,
            "prediction_filtered": filtered,
        },
    }
    if filtered and predicted_separation.noise_covariance is not None:
        validated_covariance(
            predicted_separation.noise_covariance, "filtered R", predicted.size
        )
        result["filtered_noise_covariance_shape"] = list(
            np.shape(predicted_separation.noise_covariance)
        )
    result["residual_statistics"] = _residual_statistics(
        record, residual, covariance=covariance, coverage=coverage
    )
    return result


def _residual_statistics(
    record: MeasurementRecord,
    residual: np.ndarray,
    *,
    covariance: OutputCovariance | None,
    coverage: float,
) -> dict[str, Any]:
    """The covariance-aware half of the comparison, or a statement of why not.

    A scalar summary of a residual is not a comparison statistic. It cannot be
    compared between instruments, it has no distribution, and the only bar that
    can be put against it is one already in the measurement's units -- which is
    a declared limit smuggled in as arithmetic. What replaces it is the
    whitened residual and a chi-square with stated degrees of freedom, against
    a band that is closed at *both* ends.
    """
    if covariance is None and record.measurement_covariance is None:
        return {
            "available": False,
            "reason": (
                "no covariance: the trial declares no measurement_covariance and none "
                "was assembled for the comparison. The scalars above are a summary of "
                "the residual and not a statistic about it."
            ),
        }
    total = covariance
    source = "assembled"
    if total is None:
        source = "trial-declared"
        total = OutputCovariance(
            arclength=np.asarray(record.arclength, dtype=float),
            outputs=("signed-transverse-separation",),
            blocks={
                "observation-noise": np.asarray(record.measurement_covariance, dtype=float)
            },
            note="the covariance the trial itself declared",
        )
    if total.degrees_of_freedom != residual.size:
        raise ValueError(
            f"the covariance is over {total.degrees_of_freedom} scalars and the "
            f"residual has {residual.size}; they are not the same comparison"
        )
    outcome = total.accepts(residual, coverage=coverage)
    whitened = total.whiten(residual)
    return {
        "available": True,
        "source": source,
        "chi_square": outcome["statistic"],
        "degrees_of_freedom": outcome["degrees_of_freedom"],
        "reduced_chi_square": outcome["reduced"],
        "probability_less_than": outcome["probability_less_than"],
        "band": outcome["band"],
        "verdict": outcome["verdict"],
        "accepted": outcome["accepted"],
        "max_abs_whitened_residual": float(np.max(np.abs(whitened))),
        "interval_coverage": total.interval_coverage(residual, sigmas=1.0),
        "shares": outcome["shares"],
    }
