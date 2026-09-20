"""One record that both halves of this project speak.

The engine produces ``TransferMap`` and ``PathEnvelope``; the application layer
grew up around ``JacobiTrace``. They carry the same mathematics and were
different concrete types, which meant a surface path could only reach
``assess_manufacturing`` through an informal conversion that dropped everything
the conversion could not name -- units, resolution, what the numbers are a
measurement *of*.

:class:`TransferRecord` is that missing type. It is immutable, self-describing
and serialisable, and it carries the things a consumer downstream cannot
reconstruct and must not guess:

``arclength, gaussian_curvature, a, a_rate, b, b_rate``
    the transfer map itself, sampled.
``frame``
    what the transverse direction is measured against.
``units``
    length and angle. A tolerance in millimetres against a record in metres is
    a three-orders-of-magnitude error that no type check would catch.
``source_digest``
    a hash of the surface and path that produced it, so a prediction can be
    tied to the geometry it was computed from.
``resolution``
    method, sample count and step, because "rk4" alone does not say whether a
    focus is located to 1e-3 or 1e-12.
``validity``
    the perturbation range over which the linear map is declared to hold, and
    on what basis. Unknown is a legal and frequently correct answer.
``observation_mode``
    the quantity a comparison against this record would be in.

It is also the natural interchange boundary outward: a mesh project can emit
one of these instead of this repository growing a mesh solver, and an evidence
system downstream can consume one without knowing how it was produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .observation import DEFAULT_MODE
from .observation import mode as observation_mode
from .transfer import FocusEvent, TransferMap

Array = np.ndarray

RECORD_SCHEMA = "path-transfer-record-v1"


@dataclass(frozen=True)
class Units:
    """Units the record's numbers are in. There is no default that is safe."""

    length: str = "declared length unit"
    angle: str = "radian"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Resolution:
    """How the record was computed, in enough detail to judge what it resolves."""

    method: str
    samples: int
    max_step: float
    uniform: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FirstOrderValidity:
    """The perturbation range over which the linear map is declared to hold.

    ``basis`` says how the range was arrived at, and is the field that makes
    the others readable: a bound measured against an exact finite separation
    means something different from one a caller asserted.
    """

    basis: str
    relative_tolerance: float | None = None
    max_lateral: float | None = None
    max_heading: float | None = None

    @classmethod
    def not_established(cls, why: str) -> FirstOrderValidity:
        return cls(basis=f"not-established: {why}")

    @property
    def established(self) -> bool:
        return not self.basis.startswith("not-established")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"established": self.established}


@runtime_checkable
class SupportsTransferRecord(Protocol):
    """Anything that can present itself as a :class:`TransferRecord`."""

    def as_transfer_record(self) -> TransferRecord: ...


@dataclass(frozen=True)
class TransferRecord:
    """A sampled starting-pose transfer map, with everything needed to use it."""

    arclength: Array
    gaussian_curvature: Array
    a: Array
    a_rate: Array
    b: Array
    b_rate: Array
    frame: str = "transverse-to-gamma, parallel-transported"
    units: Units = field(default_factory=Units)
    source_digest: str = ""
    resolution: Resolution = field(
        default_factory=lambda: Resolution("unspecified", 0, float("nan"), False)
    )
    validity: FirstOrderValidity = field(
        default_factory=lambda: FirstOrderValidity.not_established("no declaration supplied")
    )
    observation_mode: str = DEFAULT_MODE
    domain: str = "constant-curvature"

    def __post_init__(self) -> None:
        arrays = ("arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate")
        size = np.asarray(self.arclength).size
        if size < 2:
            raise ValueError("a transfer record needs at least two samples")
        for name in arrays:
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (size,):
                raise ValueError(f"{name} must be a vector on the arclength grid")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if np.any(np.diff(self.arclength) <= 0.0):
            raise ValueError("arclength must be strictly increasing")
        observation_mode(self.observation_mode)

    # -- the map ----------------------------------------------------------
    def transfer_map(self) -> TransferMap:
        return TransferMap(
            arc_length=self.arclength, a=self.a, a_rate=self.a_rate,
            b=self.b, b_rate=self.b_rate,
        )

    @property
    def wronskian_drift(self) -> Array:
        return self.transfer_map().wronskian_drift

    def separation(self, initial_offset: float, initial_angle: float) -> Array:
        """Signed first-order transverse separation, ``a*offset + b*angle``."""
        return self.a * float(initial_offset) + self.b * float(initial_angle)

    def heading_change(self, initial_offset: float, initial_angle: float) -> Array:
        """Signed first-order heading error, ``a'*offset + b'*angle``."""
        return self.a_rate * float(initial_offset) + self.b_rate * float(initial_angle)

    # -- bounds over a tolerance box --------------------------------------
    def cross_track_error(self, max_lateral: float, max_heading: float) -> Array:
        """``e_perp(s) = |a| dp + |b| da``: worst transverse deviation over the box."""
        return np.abs(self.a) * abs(float(max_lateral)) + np.abs(self.b) * abs(
            float(max_heading)
        )

    def heading_error(self, max_lateral: float, max_heading: float) -> Array:
        """``e_alpha(s) = |a'| dp + |b'| da``: worst downstream heading error."""
        return np.abs(self.a_rate) * abs(float(max_lateral)) + np.abs(self.b_rate) * abs(
            float(max_heading)
        )

    def rss_cross_track_error(self, max_lateral: float, max_heading: float) -> Array:
        return np.hypot(
            np.abs(self.a) * abs(float(max_lateral)), np.abs(self.b) * abs(float(max_heading))
        )

    def propagate_covariance(self, covariance) -> Array:
        return self.transfer_map().propagate_covariance(covariance)

    # -- conditioning ------------------------------------------------------
    def focus_events(self, *, component: str = "b") -> list[FocusEvent]:
        return self.transfer_map().focus_events(component=component)

    def clearance_from_focus(self, *, component: str = "b") -> float:
        return self.transfer_map().clearance_from_focus(component=component)

    def scaled_transfer(self, max_lateral: float, max_heading: float) -> Array:
        """``Phi`` made dimensionless by the tolerance box, as an ``(n, 2, 2)`` stack.

        ``a`` and ``b`` do not share units -- ``b`` is a length per radian --
        so the singular values of ``Phi`` are not comparable between paths and
        ranking by them is meaningless. Conjugating by ``S = diag(dp, da)``
        gives ``S^-1 Phi S``, whose entries are all pure ratios of "error out"
        to "error allowed in", and whose norm is a usable score.
        """
        lateral, heading = abs(float(max_lateral)), abs(float(max_heading))
        if lateral <= 0.0 or heading <= 0.0:
            raise ValueError("both tolerance scales must be positive to make Phi dimensionless")
        scale = np.array([lateral, heading])
        phi = self.transfer_map().matrices()
        return (phi * scale) / scale[:, None]

    def scaled_singular_values(self, max_lateral: float, max_heading: float) -> Array:
        """Singular values of the dimensionless transfer, largest first, per sample.

        Their product is ``|det Phi| = 1`` at every arc length, because
        conjugation does not change a determinant. So the two are reciprocal:
        whatever the flow contracts in one direction of the tolerance box, it
        expands by exactly the same factor in the conjugate direction. The
        larger is therefore never below 1, and a path cannot be robust to a
        starting pose error in every direction at once.
        """
        return np.linalg.svd(
            self.scaled_transfer(max_lateral, max_heading), compute_uv=False
        )

    def amplification_score(self, max_lateral: float, max_heading: float) -> float:
        """Worst dimensionless gain of the tolerance box anywhere along the path.

        One number, and a fair one: dividing by the tolerances first makes the
        lateral and heading entries pure ratios, so paths can be compared. The
        raw singular values of ``Phi`` cannot be -- ``b`` is a length per
        radian and ``a`` is dimensionless, so their relative size depends on
        the unit of angle.
        """
        return float(np.max(self.scaled_singular_values(max_lateral, max_heading)))

    # -- serialisation -----------------------------------------------------
    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "frame": self.frame,
            "units": self.units.to_dict(),
            "source_digest": self.source_digest,
            "resolution": self.resolution.to_dict(),
            "validity": self.validity.to_dict(),
            "observation_mode": self.observation_mode,
            "domain": self.domain,
            "samples": int(self.arclength.size),
        }
        if include_samples:
            payload |= {
                name: np.asarray(getattr(self, name)).tolist()
                for name in ("arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate")
            }
        return payload

    def as_transfer_record(self) -> TransferRecord:
        return self


def digest(description: dict[str, Any]) -> str:
    """Stable digest of whatever produced a record.

    Deliberately a hash of a *description* rather than of the samples: two runs
    at different resolutions of the same surface and path should be recognisable
    as the same geometry, and the resolution is recorded separately.
    """
    canonical = json.dumps(description, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def to_transfer_record(source: Any) -> TransferRecord:
    """Normalise anything that can describe a transfer map into a record."""
    if isinstance(source, TransferRecord):
        return source
    converter = getattr(source, "as_transfer_record", None)
    if callable(converter):
        return converter()
    raise TypeError(
        f"{type(source).__name__} cannot be used as a transfer record; it needs an "
        "as_transfer_record() method"
    )
