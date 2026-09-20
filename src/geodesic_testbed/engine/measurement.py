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

from .observation import mode as observation_mode
from .observation_model import FilteredPrediction
from .transfer import _validated_covariance

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
            _validated_covariance(self.measurement_covariance, "measurement_covariance", size=None)
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
            _validated_covariance(self.measurement_covariance, "measurement_covariance", size=None)
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


def compare(
    record: MeasurementRecord,
    predicted_separation,
    *,
    mode: str | None = None,
    resolvability_threshold: float | None = None,
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

    ``resolvability_threshold``, when given, is the signal-to-noise bar the
    *instrument protocol* declares. This module reports the ratio and never
    invents the bar.
    """
    if record.measurement_covariance is not None:
        _validated_covariance(record.measurement_covariance, "measurement_covariance", size=None)
    expected = mode or record.observation_mode
    if expected != record.observation_mode:
        raise ValueError(
            f"prediction is in {expected!r} but the trial measured "
            f"{record.observation_mode!r}; convert one before comparing"
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
        _validated_covariance(
            predicted_separation.noise_covariance, "filtered R", predicted.size
        )
        result["filtered_noise_covariance_shape"] = list(
            np.shape(predicted_separation.noise_covariance)
        )
    return result
