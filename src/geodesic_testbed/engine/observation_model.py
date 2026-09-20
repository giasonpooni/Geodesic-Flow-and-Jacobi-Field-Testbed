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

**This is also what a focus should mean.** The repository previously called a
path "clear of a focus" when ``|b(s)|`` stayed above 0.25, which is a number
with units of length per radian: it is specific to a unit-radius torus, and
scaling the same physical part or switching from radians to degrees changes the
verdict. The observable statement has none of those problems:

.. math::

    \\rho(s) = \\frac{\\sqrt{\\operatorname{diag}(H\\Phi C_0 \\Phi^\\mathsf{T} H^\\mathsf{T})}}
                    {\\sqrt{\\operatorname{diag} R}} .

With ``H = [1, 0]`` and a starting uncertainty in heading alone this is exactly
``|b(s)| sigma_alpha / sigma_measurement``. A focus is then where ``rho`` falls
below one: the place at which the instrument cannot tell two different starting
headings apart. That is a fact about a surface, a tolerance and a scanner
together, and it is dimensionless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .observation import mode as observation_mode
from .record import TransferRecord, to_transfer_record

Array = np.ndarray

#: The state the transfer map propagates, in order.
STATE_COMPONENTS = ("transverse", "heading")


def _validated_covariance(matrix, name: str, size: int) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must be {size}x{size}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(matrix, matrix.T, atol=0.0, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    tolerance = 1e-12 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if eigenvalues[0] < -tolerance:
        raise ValueError(
            f"{name} must be positive semidefinite; smallest eigenvalue is {eigenvalues[0]!r}"
        )
    return matrix


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


def filtered_noise_covariance(filter_matrix, noise_covariance) -> Array:
    """``F R F^T`` for an explicit linear filter over samples.

    ``R`` may be a scalar variance, a per-sample vector, or a full covariance.
    The result is dense: a filter correlates samples that were independent, and
    pretending otherwise is the error this function exists to make visible.
    """
    F = np.asarray(filter_matrix, dtype=float)
    if F.ndim != 2:
        raise ValueError("the filter must be a matrix over samples")
    R = np.asarray(noise_covariance, dtype=float)
    if R.ndim == 0:
        R = np.eye(F.shape[1]) * float(R)
    elif R.ndim == 1:
        R = np.diag(R)
    if R.shape != (F.shape[1], F.shape[1]):
        raise ValueError("R must be square and match the filter's input length")
    return F @ R @ F.T


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
        the tolerance admits: the two are the same measurement. That is the
        observable meaning of a focus, and unlike a threshold on ``|b|`` it is
        dimensionless and does not move when the part is rescaled or the angle
        unit is changed.
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
        spread is the size of the noise" and 3 is the usual bar for calling
        something measured.
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
