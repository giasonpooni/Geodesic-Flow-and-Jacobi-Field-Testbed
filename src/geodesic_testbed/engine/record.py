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
``covariance``
    the starting-pose covariance the record is to be propagated with, when one
    is declared. Not declaring one is legal; inventing one is not.
``provenance``
    who produced the record, at what version, and from which upstream
    artefacts.
``calibration``
    opaque calibration identifiers, carried and never interpreted here.

It is also the natural interchange boundary outward: a mesh project can emit
one of these instead of this repository growing a mesh solver, and an evidence
system downstream can consume one without knowing how it was produced. The
vocabulary of that boundary -- units, frames, covariance, provenance,
calibration identifiers -- lives in :mod:`geodesic_testbed.engine.contract`,
and :mod:`geodesic_testbed.boundary` is the single import that presents it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .contract import (
    BOUNDARY_CONTRACT,
    DEFAULT_FRAME,
    CalibrationBinding,
    Provenance,
    StartingCovariance,
    Units,
    UpstreamArtefact,
)
from .contract import frame as declared_frame
from .observation import DEFAULT_MODE, require_available
from .transfer import FocusEvent, TransferMap

Array = np.ndarray

RECORD_SCHEMA = "path-transfer-record-v2"

#: Schemas :meth:`TransferRecord.from_dict` will read. ``v1`` predates the
#: covariance, provenance and calibration fields; a ``v1`` payload is accepted
#: and those fields come back undeclared, which is what they in fact were.
SUPPORTED_RECORD_SCHEMAS: tuple[str, ...] = (
    "path-transfer-record-v1",
    "path-transfer-record-v2",
)

__all__ = [
    "BOUNDARY_CONTRACT",
    "RECORD_SCHEMA",
    "SUPPORTED_RECORD_SCHEMAS",
    "CalibrationBinding",
    "FirstOrderValidity",
    "Provenance",
    "Resolution",
    "StartingCovariance",
    "SupportsTransferRecord",
    "TransferRecord",
    "Units",
    "UpstreamArtefact",
    "digest",
    "to_transfer_record",
]


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
    frame: str = DEFAULT_FRAME
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
    covariance: StartingCovariance = field(default_factory=StartingCovariance.not_declared)
    provenance: Provenance = field(default_factory=Provenance)
    calibration: CalibrationBinding = field(default_factory=CalibrationBinding.unbound)

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
        # The mode and the domain are checked *together*. Each is fine alone
        # and the pair is what fails: a surface record tagged with the
        # intrinsic distance names two real things and claims a quantity
        # nothing here computes -- in the one field a downstream comparison
        # trusts to decide whether two numbers are comparable at all.
        require_available(self.observation_mode, self.domain)
        declared_frame(self.frame)
        if self.covariance.declared:
            if self.covariance.frame != self.frame:
                raise ValueError(
                    f"the starting covariance is in frame {self.covariance.frame!r} but "
                    f"the record is in {self.frame!r}; a covariance in a different frame "
                    "differs by a rotation neither of them records"
                )
            if self.covariance.units != self.units:
                raise ValueError(
                    f"the starting covariance is in {self.covariance.units.to_dict()} but "
                    f"the record is in {self.units.to_dict()}"
                )

    # -- the arclength grid -----------------------------------------------
    @property
    def grid(self) -> dict[str, Any]:
        """The sample grid, summarised. The samples themselves stay authoritative.

        A start/stop/count triple is not a substitute for the vector: a
        non-uniform grid is legal here, and a consumer that resampled one onto
        a uniform grid of the same extent would be holding a different record.
        """
        steps = np.diff(self.arclength)
        return {
            "start": float(self.arclength[0]),
            "end": float(self.arclength[-1]),
            "samples": int(self.arclength.size),
            "min_step": float(np.min(steps)),
            "max_step": float(np.max(steps)),
            "uniform": bool(np.allclose(steps, steps[0], rtol=1e-12, atol=0.0)),
        }

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
        """``Phi C Phi^T`` for a caller-supplied starting covariance."""
        return self.transfer_map().propagate_covariance(covariance)

    def propagate_declared_covariance(self) -> Array:
        """``Phi C0 Phi^T`` for the covariance *this record declares*.

        Raises when none is declared. That is the point: a consumer needing a
        distributional answer must not receive one computed from a covariance
        nobody stated, and the failure has to happen here rather than downstream
        where the substituted number would already look like a result.
        """
        return self.transfer_map().propagate_covariance(self.covariance.require())

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

    @property
    def determinant(self) -> Array:
        """``det Phi = a b' - a' b``, which is 1 at every arc length, exactly."""
        return self.transfer_map().determinant

    def scaled_determinant(self, max_lateral: float, max_heading: float) -> Array:
        """``det(S^-1 Phi S)``, formed directly from the scaled entries.

        The same invariant as :attr:`determinant`, checked after the
        conjugation that makes the entries comparable -- and formed as a 2x2
        determinant rather than as a product of singular values.

        That distinction is the whole point. ``sigma_1 sigma_2 = |det Phi|`` is
        true, and testing the invariant through it is a bad way to test it: the
        SVD of a badly conditioned scaled transfer returns ``sigma_1`` with a
        relative error of order ``eps``, so with a tolerance box whose aspect
        ratio is ``10^4`` the product lands near ``1`` only to about ``10^-12``
        and the "invariant" check is really a check on the conditioning of the
        box. The 2x2 determinant has no such dependence: it is two products and
        a subtraction, and it holds to machine precision on every box.
        """
        phi = self.scaled_transfer(max_lateral, max_heading)
        return phi[..., 0, 0] * phi[..., 1, 1] - phi[..., 0, 1] * phi[..., 1, 0]

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

    # -- amending the contract fields --------------------------------------
    def with_covariance(self, covariance: StartingCovariance) -> TransferRecord:
        """The same record, now declaring a starting covariance.

        Separate from construction because ``C0`` is usually known later and
        elsewhere: the runtime computes ``Phi`` from geometry alone, and what
        the starting pose error actually is belongs to whoever set the part up.
        """
        return replace(self, covariance=covariance)

    def with_calibration(self, calibration: CalibrationBinding) -> TransferRecord:
        """The same record, now carrying calibration identifiers.

        Carrying, not using. The identifiers are opaque here; they exist so a
        downstream comparison can establish that a prediction and a measurement
        came from the same instrument state.
        """
        return replace(self, calibration=calibration)

    def with_provenance(self, provenance: Provenance) -> TransferRecord:
        """The same record, with its provenance replaced."""
        return replace(self, provenance=provenance)

    # -- serialisation -----------------------------------------------------
    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "contract": BOUNDARY_CONTRACT,
            "frame": self.frame,
            "units": self.units.to_dict(),
            "source_digest": self.source_digest,
            "resolution": self.resolution.to_dict(),
            "validity": self.validity.to_dict(),
            "observation_mode": self.observation_mode,
            "domain": self.domain,
            "samples": int(self.arclength.size),
            "grid": self.grid,
            "covariance": self.covariance.to_dict(),
            "provenance": self.provenance.to_dict(),
            "calibration": self.calibration.to_dict(),
        }
        if include_samples:
            payload |= {
                name: np.asarray(getattr(self, name)).tolist()
                for name in ("arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate")
            }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransferRecord:
        """Rebuild a record from :meth:`to_dict`, refusing an unknown schema.

        The round trip is what makes this a boundary rather than an internal
        type: a record has to survive being written to a file by one process
        and read by another that shares no code with it. A payload without
        samples cannot be rebuilt and says so, rather than coming back as a
        record with the metadata right and the numbers missing.
        """
        schema = payload.get("schema")
        if schema not in SUPPORTED_RECORD_SCHEMAS:
            raise ValueError(
                f"unknown record schema {schema!r}; this reader understands "
                f"{SUPPORTED_RECORD_SCHEMAS}"
            )
        arrays = ("arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate")
        missing = [name for name in arrays if name not in payload]
        if missing:
            raise ValueError(
                "this payload carries no samples "
                f"(missing {', '.join(missing)}); it was written with "
                "include_samples=False and is a summary, not a record"
            )
        resolution = payload.get("resolution") or {}
        validity = payload.get("validity") or {}
        return cls(
            **{name: np.asarray(payload[name], dtype=float) for name in arrays},
            frame=str(payload.get("frame", DEFAULT_FRAME)),
            units=Units.from_dict(payload.get("units")),
            source_digest=str(payload.get("source_digest", "")),
            resolution=Resolution(
                method=str(resolution.get("method", "unspecified")),
                samples=int(resolution.get("samples", 0)),
                max_step=float(resolution.get("max_step", float("nan"))),
                uniform=bool(resolution.get("uniform", False)),
            ),
            validity=FirstOrderValidity(
                basis=str(validity.get("basis", "not-established: no declaration supplied")),
                relative_tolerance=validity.get("relative_tolerance"),
                max_lateral=validity.get("max_lateral"),
                max_heading=validity.get("max_heading"),
            ),
            observation_mode=str(payload.get("observation_mode", DEFAULT_MODE)),
            domain=str(payload.get("domain", "constant-curvature")),
            covariance=StartingCovariance.from_dict(payload.get("covariance")),
            provenance=Provenance.from_dict(payload.get("provenance")),
            calibration=CalibrationBinding.from_dict(payload.get("calibration")),
        )

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
