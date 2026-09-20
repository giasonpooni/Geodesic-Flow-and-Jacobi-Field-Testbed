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
        if self.filter_identifier != "none" and not self.filter_version:
            raise ValueError("a declared filter must carry a version")
        if len(self.arclength) < 2:
            raise ValueError("a trial needs at least two samples")
        if not {"length", "angle"} <= set(self.units):
            raise ValueError("units must declare both 'length' and 'angle'")
        if self.measurement_covariance is not None:
            matrix = np.asarray(self.measurement_covariance, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("measurement_covariance must be square")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("measurement_covariance must be finite")
            if np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[0] < -1e-12:
                raise ValueError("measurement_covariance must be positive semidefinite")
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
        total = self.uncertainty.total()
        if total <= 0.0:
            return float("inf")
        return float(np.max(np.abs(self.signed_transverse_separation)) / total)

    def to_dict(self) -> dict[str, Any]:
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
    filtered_prediction: bool = False,
):
    """Residual of a prediction against a trial, refusing a mode mismatch.

    ``predicted_separation`` is sampled on the trial's own ``arclength``. The
    mode check is the point of the function: an in-surface prediction against a
    reconstructed chord disagrees at second order in the perturbation, which is
    the same order as the model's own failure, so the comparison would read as
    a model failure that is really a units-of-measurement error.
    """
    expected = mode or record.observation_mode
    if expected != record.observation_mode:
        raise ValueError(
            f"prediction is in {expected!r} but the trial measured "
            f"{record.observation_mode!r}; convert one before comparing"
        )
    if record.filter_identifier != "none" and not filtered_prediction:
        raise ValueError(
            f"the trial was filtered by {record.filter_identifier!r} "
            f"{record.filter_version!r}, so the prediction must be passed through "
            "the same filter and compared against F R F^T; pass "
            "filtered_prediction=True once it has been"
        )
    predicted = np.asarray(predicted_separation, dtype=float)
    measured = np.asarray(record.signed_transverse_separation, dtype=float)
    if predicted.shape != measured.shape:
        raise ValueError("prediction must be sampled on the trial's arclength")
    residual = measured - predicted
    total = record.uncertainty.total()
    return {
        "observation_mode": record.observation_mode,
        "role": record.role,
        "max_abs_residual": float(np.max(np.abs(residual))),
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
        "residual_over_uncertainty": (
            float(np.max(np.abs(residual)) / total) if total > 0.0 else float("inf")
        ),
        "signal_to_noise": record.signal_to_noise,
        "resolvable": bool(record.signal_to_noise >= 3.0),
        "prediction_report_digest": record.prediction_report_digest,
        "raw_data_digest": record.raw_data_digest,
        "filter": {
            "identifier": record.filter_identifier,
            "version": record.filter_version,
            "causal": record.filter_causal,
            "prediction_filtered": filtered_prediction,
        },
    }
