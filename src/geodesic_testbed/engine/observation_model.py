"""From a transfer map to what an instrument would actually see.

The transfer map says how a starting-pose error propagates:

.. math::

    \\delta z(s) = \\Phi(s)\\, \\delta z_0,
    \\qquad \\delta z = (\\delta_\\perp,\\ \\delta_\\alpha)^\\mathsf{T}.

No instrument reports ``delta z``. It reports some linear functional of it,
corrupted by its own noise:

.. math::

    y(s) = H(s)\\,\\delta z(s) + \\eta(s),
    \\qquad
    \\operatorname{Cov}(y) = H\\Phi\\, C_0\\, \\Phi^\\mathsf{T} H^\\mathsf{T} + R .

``H`` is the observation mode made concrete -- a scanner that reports only
transverse deviation is ``[1, 0]``; one that also tracks orientation is the
identity -- and ``R`` is the metrology covariance. Together they are what turns
a verified transfer map into a prediction that a bench can falsify.

**What replaces a threshold on |b|.** The repository previously called a path
"clear of a focus" when ``|b(s)|`` stayed above 0.25, which is a number with
units of length per radian: it is specific to a unit-radius torus, and scaling
the same physical part or switching from radians to degrees changes the
verdict. The observable statement has none of those problems:

.. math::

    \\rho(s) = \\frac{\\sqrt{\\operatorname{diag}(H\\Phi C_0 \\Phi^\\mathsf{T} H^\\mathsf{T})}}
                    {\\sqrt{\\operatorname{diag} R}} .

With ``H = [1, 0]`` and a starting uncertainty in heading alone this is exactly
``|b(s)| sigma_alpha / sigma_measurement``: a fact about a surface, a tolerance
and a scanner together, and dimensionless.

**``rho`` is not a focus, and must not be renamed one.** The two are different
statements that happen to correlate:

* a *geometric focus* is a zero of the relevant transfer component -- a
  property of the surface and the path alone, present whatever instrument is
  pointed at it, and reported by ``focus_points`` on the transfer map;
* *low resolvability* is a property of the whole measurement chain. Shrink the
  admitted starting uncertainty ``C_0`` and ``rho`` falls everywhere, with no
  focus anywhere near; sharpen ``R`` enough and ``rho`` stays above threshold
  arbitrarily close to a genuine conjugate point, because a good enough
  instrument still separates the two starting poses.

So both are reported, and they answer different questions. The route
constraint uses ``rho``, because feasibility is a question about the
instrument. The mathematical report keeps the conjugate-point locations,
because those are a claim about the geometry that no scanner can change.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .contract import validated_covariance
from .observation import mode as observation_mode
from .record import TransferRecord, to_transfer_record

Array = np.ndarray

#: The state the transfer map propagates, in order.
STATE_COMPONENTS = ("transverse", "heading")


#: One covariance validator for the whole repository, in the contract module
#: where the vocabulary lives. Two implementations of "is this a covariance"
#: eventually disagree, and the one that is laxer is the one that gets used.
_validated_covariance = validated_covariance


@dataclass(frozen=True)
class TemporalFilter:
    r"""A declared filter, and the fact that it changes the observation model.

    Filtering is legitimate and often necessary, but it is part of the
    instrument, not invisible preprocessing. A linear filter ``F`` applied to a
    measured trajectory changes what is being compared:

    .. math::

        y_f = F H \Phi\, \delta z_0 + F \eta,
        \qquad R_f = F R F^\mathsf{T},

    and ``R_f`` is correlated across samples even when ``R`` was not. Comparing
    filtered measurements against an unfiltered prediction while keeping the
    original ``R`` understates the uncertainty and can manufacture agreement.

    This record carries the identity of the filter so a comparison can refuse
    that mismatch; :func:`filtered_noise_covariance` performs the ``F R F^T``
    algebra for an explicit operator.
    """

    identifier: str
    version: str
    causal: bool
    group_delay: float = 0.0
    input_rate: float | None = None
    output_rate: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    tuned_on: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def operator_digest(filter_matrix) -> str:
    """A digest of the filter operator itself, not of its name.

    An identifier and a version say which filter was *meant*; this says which
    one was *applied*. A comparison that checks only the name will accept a
    prediction smoothed by a different operator carrying the same label, which
    is precisely how a filter manufactures agreement unnoticed.
    """
    F = np.ascontiguousarray(np.asarray(filter_matrix, dtype=float))
    hasher = hashlib.sha256()
    hasher.update(str(F.shape).encode("utf-8"))
    hasher.update(F.tobytes())
    return "sha256:" + hasher.hexdigest()[:32]


def filtered_noise_covariance(filter_matrix, noise_covariance) -> Array:
    """``F R F^T`` for an explicit linear filter over samples.

    ``R`` may be a scalar variance, a per-sample vector, or a full covariance.
    The result is dense: a filter correlates samples that were independent, and
    pretending otherwise is the error this function exists to make visible.
    """
    F = np.asarray(filter_matrix, dtype=float)
    if F.ndim != 2:
        raise ValueError("the filter must be a matrix over samples")
    if not np.all(np.isfinite(F)):
        raise ValueError("the filter matrix must be finite")
    R = np.asarray(noise_covariance, dtype=float)
    if R.ndim == 0:
        R = np.eye(F.shape[1]) * float(R)
    elif R.ndim == 1:
        R = np.diag(R)
    if R.shape != (F.shape[1], F.shape[1]):
        raise ValueError("R must be square and match the filter's input length")
    R = _validated_covariance(R, "R", R.shape[0])
    return F @ R @ F.T


@dataclass(frozen=True)
class FilteredPrediction:
    """A prediction that has been through a named filter, and can prove it.

    A boolean "yes, it was filtered" is an assertion by the caller. This is
    evidence: the identifier and version say which filter was intended, and
    ``operator_digest`` says which matrix was actually applied. A comparison
    can then refuse a prediction filtered by a *different* operator wearing the
    same label -- the failure a boolean cannot catch.

    Build one with :func:`apply_filter` rather than by hand, so the digest and
    the values come from the same operator.
    """

    values: Array
    identifier: str
    version: str
    operator_digest: str
    causal: bool
    parameters: dict[str, Any] = field(default_factory=dict)
    noise_covariance: Array | None = None
    group_delay: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("a filtered prediction must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        if not self.identifier or self.identifier == "none":
            raise ValueError("a filtered prediction must name the filter that made it")
        if not self.version:
            raise ValueError("a declared filter must carry a version")
        if not self.operator_digest:
            raise ValueError(
                "a filtered prediction must carry the digest of the operator that "
                "produced it; a name alone does not identify a filter"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "version": self.version,
            "operator_digest": self.operator_digest,
            "causal": self.causal,
            "parameters": dict(self.parameters),
            "group_delay": float(self.group_delay),
            "has_output_covariance": self.noise_covariance is not None,
            "note": self.note,
        }


def apply_filter(
    filter_matrix,
    prediction,
    *,
    identifier: str,
    version: str,
    causal: bool,
    parameters: dict[str, Any] | None = None,
    noise_covariance=None,
    group_delay: float = 0.0,
    note: str = "",
) -> FilteredPrediction:
    """Put a prediction through ``F``, and record which ``F`` that was.

    When ``noise_covariance`` is given it is the *unfiltered* ``R``; the result
    carries ``F R F^T``, because that is what the filtered prediction must be
    compared against.
    """
    F = np.asarray(filter_matrix, dtype=float)
    if F.ndim != 2:
        raise ValueError("the filter must be a matrix over samples")
    if not np.all(np.isfinite(F)):
        raise ValueError("the filter matrix must be finite")
    values = np.asarray(prediction, dtype=float)
    if values.ndim != 1 or values.size != F.shape[1]:
        raise ValueError("the prediction must be one sample per filter input")
    filtered_R = (
        None if noise_covariance is None
        else filtered_noise_covariance(F, noise_covariance)
    )
    return FilteredPrediction(
        values=F @ values,
        identifier=identifier,
        version=version,
        operator_digest=operator_digest(F),
        causal=bool(causal),
        parameters=dict(parameters or {}),
        noise_covariance=filtered_R,
        group_delay=float(group_delay),
        note=note,
    )


@dataclass(frozen=True)
class ObservationModel:
    """``H`` and ``R``: what the instrument reports, and how well."""

    mode: str
    matrix: Array
    noise_covariance: Array
    outputs: tuple[str, ...]
    calibration_id: str = "uncalibrated"
    reconstruction_version: str = "none"
    temporal_filter: TemporalFilter | None = None
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation_mode(self.mode)
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 2:
            raise ValueError("H must be (m, 2) in the (transverse, heading) basis")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("H must be finite")
        if len(self.outputs) != matrix.shape[0]:
            raise ValueError("outputs must name one entry per row of H")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        noise = _validated_covariance(self.noise_covariance, "R", matrix.shape[0])
        noise.setflags(write=False)
        object.__setattr__(self, "noise_covariance", noise)

    # -- constructors for the two shapes that come up ---------------------
    @classmethod
    def transverse_only(
        cls,
        measurement_sigma: float,
        *,
        mode: str = "ambient-euclidean-chord",
        calibration_id: str = "uncalibrated",
        reconstruction_version: str = "none",
        note: str = "",
    ) -> ObservationModel:
        """A system that reports transverse deviation and nothing else."""
        sigma = float(measurement_sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("measurement_sigma must be finite and positive")
        return cls(
            mode=mode,
            matrix=np.array([[1.0, 0.0]]),
            noise_covariance=np.array([[sigma**2]]),
            outputs=("transverse",),
            calibration_id=calibration_id,
            reconstruction_version=reconstruction_version,
            note=note,
        )

    @classmethod
    def full_pose(
        cls,
        transverse_sigma: float,
        heading_sigma: float,
        *,
        mode: str = "ambient-euclidean-chord",
        calibration_id: str = "uncalibrated",
        reconstruction_version: str = "none",
        note: str = "",
    ) -> ObservationModel:
        """A system that tracks both transverse deviation and orientation."""
        for name, value in (("transverse_sigma", transverse_sigma),
                            ("heading_sigma", heading_sigma)):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        return cls(
            mode=mode,
            matrix=np.eye(2),
            noise_covariance=np.diag([float(transverse_sigma) ** 2, float(heading_sigma) ** 2]),
            outputs=STATE_COMPONENTS,
            calibration_id=calibration_id,
            reconstruction_version=reconstruction_version,
            note=note,
        )

    # -- prediction --------------------------------------------------------
    def _checked_record(self, source: Any) -> TransferRecord:
        record = to_transfer_record(source)
        if record.observation_mode != self.mode:
            raise ValueError(
                f"the record is in {record.observation_mode!r} but this observation "
                f"model reports {self.mode!r}; convert one before predicting"
            )
        return record

    def predict(self, source: Any, initial_state) -> Array:
        """Noise-free output ``H Phi dz0`` along the path, shape ``(n, m)``."""
        record = self._checked_record(source)
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape != (2,):
            raise ValueError("initial_state must be (transverse, heading)")
        propagated = np.stack(
            record.transfer_map().propagate(float(initial[0]), float(initial[1])), axis=-1
        )
        return propagated @ self.matrix.T

    def covariance(self, source: Any, initial_covariance) -> Array:
        """``H Phi C0 Phi^T H^T + R`` along the path, shape ``(n, m, m)``."""
        record = self._checked_record(source)
        initial = _validated_covariance(initial_covariance, "C0", 2)
        propagated = record.propagate_covariance(initial)
        return self.matrix @ propagated @ self.matrix.T + self.noise_covariance

    def resolvability(self, source: Any, initial_covariance) -> Array:
        """``rho(s)``: predicted signal over measurement noise, per output.

        Below one, the instrument cannot distinguish the starting-pose errors
        the tolerance admits: the two are the same measurement. Unlike a
        threshold on ``|b|`` this is dimensionless and does not move when the
        part is rescaled or the angle unit is changed.

        It is a property of the whole chain -- surface, tolerance and sensor --
        and therefore *not* a geometric focus. A tight ``C_0`` drives it low
        with no focus present; a sharp ``R`` keeps it high beside a real one.
        Use :meth:`TransferRecord.focus_events` for the geometry.
        """
        record = self._checked_record(source)
        initial = _validated_covariance(initial_covariance, "C0", 2)
        signal = self.matrix @ record.propagate_covariance(initial) @ self.matrix.T
        signal_sigma = np.sqrt(np.clip(np.einsum("...ii->...i", signal), 0.0, None))
        noise_sigma = np.sqrt(np.diag(self.noise_covariance))
        return signal_sigma / noise_sigma

    def unresolvable_span(
        self, source: Any, initial_covariance, *, threshold: float = 1.0
    ) -> dict[str, Any]:
        """How much of the path the instrument cannot resolve, and where.

        ``threshold`` is a signal-to-noise ratio, so 1 means "the predicted
        spread is the size of the noise". Anything above that is a bar the
        instrument's protocol declares, not one this module knows.
        """
        record = self._checked_record(source)
        rho = self.resolvability(record, initial_covariance)
        worst = np.min(rho, axis=1) if rho.ndim > 1 else rho
        below = worst < float(threshold)
        grid = record.arclength
        span = float(np.trapezoid(below.astype(float), grid)) if below.any() else 0.0
        return {
            "threshold": float(threshold),
            "min_resolvability": float(np.min(worst)),
            "min_at_arclength": float(grid[int(np.argmin(worst))]),
            "unresolvable_length": span,
            "fraction_unresolvable": span / float(grid[-1] - grid[0]),
            "resolvable_everywhere": not bool(below.any()),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matrix"] = self.matrix.tolist()
        payload["noise_covariance"] = self.noise_covariance.tolist()
        payload["outputs"] = list(self.outputs)
        payload["temporal_filter"] = (
            self.temporal_filter.to_dict() if self.temporal_filter else None
        )
        return payload
