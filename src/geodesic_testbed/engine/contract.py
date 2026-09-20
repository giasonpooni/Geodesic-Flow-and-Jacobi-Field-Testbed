"""The vocabulary both sides of the boundary speak.

This runtime is a computational substrate, not part of an instrument. It takes
geometry and a path and reports how a starting-pose error propagates; an
instrument workbench downstream takes that report and reasons about
calibration, sensing, filtering, physical trials and operational decisions.
The two are adjacent, and the whole of what passes between them is a
:class:`~geodesic_testbed.engine.record.TransferRecord`::

    geometry/path artefact
            |
            v
    geodesic sensitivity runtime      <- this repository
            |
            v
    transfer / path-sensitivity record   <- the contract
            |
            v
    instrument calibration + measurement tooling

A boundary is only worth having if it is narrow and complete. Narrow, because
anything that leaks across becomes a dependency neither side declared: solver
internals, mesh processing, filter design, hardware behaviour and route policy
each belong to exactly one side and none of them crosses. Complete, because a
consumer must not have to *infer* anything about the numbers it receives. The
fields below are the seven things that cannot be reconstructed from the samples
and must therefore travel with them:

``units``
    length and angle. A tolerance in millimetres against a record in metres is
    a thousandfold error that no type check catches.
``frame``
    what the transverse direction and the heading are measured against. Two
    correct records in different frames disagree by a rotation that neither of
    them can see.
``arclength grid``
    where the samples sit. Carried as the record's own strictly increasing
    ``arclength`` vector rather than as a start/stop/count triple, because a
    non-uniform grid is legal and a resampled one is a different record.
``covariance``
    the starting-pose covariance ``C0`` the record is to be propagated with.
    Optional in the sense that a record may decline to declare one -- and then
    a consumer that needs it must fail rather than invent one.
``provenance``
    who produced the record, at what version, and from which upstream
    artefacts. A prediction that cannot be traced to the geometry it came from
    cannot be re-derived when that geometry turns out to be wrong.
``calibration IDs``
    opaque identifiers, carried and never interpreted here. This repository
    owns no calibration; what it owes the instrument is the ability to check
    that a record and a measurement were produced under the *same* one.
``observation mode``
    the quantity a comparison against the record would be in. An in-surface
    distance and an ambient chord differ at the order the campaign is trying to
    resolve, so an untagged comparison is not evidence.

Everything in this module is immutable, serialisable and free of numerical
machinery. It is deliberately dull: a contract that computes nothing is a
contract that cannot drift from what it promises.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

Array = np.ndarray

#: Name and version of the contract itself, distinct from the schema of any one
#: record. It moves when the *shape* of the boundary changes -- a field added,
#: removed or reinterpreted -- and stays put when a record's contents change.
BOUNDARY_CONTRACT = "path-sensitivity-boundary-v1"


# -- units ----------------------------------------------------------------


@dataclass(frozen=True)
class Units:
    """Units the record's numbers are in. There is no default that is safe."""

    length: str = "declared length unit"
    angle: str = "radian"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> Units:
        if not payload:
            return cls()
        return cls(
            length=str(payload.get("length", "declared length unit")),
            angle=str(payload.get("angle", "radian")),
        )


# -- frames ---------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """A named, versioned definition of what the two state components mean.

    The transfer map acts on a two-component starting-pose error. Neither
    component means anything without a statement of what it is measured
    against, and the statement is not recoverable from the numbers: a record
    in a frame that rotates along the path and one in a parallel-transported
    frame differ by exactly the holonomy the record was computed to describe.
    """

    identifier: str
    version: int
    transverse: str
    heading: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: The frames this runtime emits. A record naming anything else is refused at
#: construction: an unknown frame is an unusable record, not a permissive one.
FRAMES: dict[str, Frame] = {
    frame.identifier: frame
    for frame in (
        Frame(
            identifier="transverse-to-gamma, parallel-transported",
            version=1,
            transverse=(
                "signed displacement along the unit normal to gamma' in the tangent "
                "plane, parallel-transported along gamma"
            ),
            heading=(
                "signed angle from gamma', positive towards the transverse "
                "direction, in the same transported frame"
            ),
            note=(
                "The frame the Jacobi equation is written in. Parallel transport is "
                "what removes the first-derivative term, which is in turn what makes "
                "det Phi = 1 an invariant rather than an approximation, so this frame "
                "is not a presentational choice."
            ),
        ),
    )
}

DEFAULT_FRAME = "transverse-to-gamma, parallel-transported"


def frame(identifier: str) -> Frame:
    """Look up a frame, refusing anything unregistered."""
    try:
        return FRAMES[identifier]
    except KeyError as exc:  # pragma: no cover - guard
        raise KeyError(f"unknown frame {identifier!r}; have {sorted(FRAMES)}") from exc


def frame_catalogue() -> list[dict[str, Any]]:
    """Every frame, for a report header."""
    return [value.to_dict() for value in FRAMES.values()]


# -- starting covariance ---------------------------------------------------

#: How a declared covariance was arrived at. ``not-declared`` is the default
#: and is a legal, frequently correct answer: a record computed from an
#: analytic surface has no starting covariance of its own.
COVARIANCE_BASES: tuple[str, ...] = (
    "not-declared",
    "declared-tolerance-box",
    "measured",
    "assumed",
    "propagated-from-upstream",
)


def validated_covariance(matrix: Any, name: str = "covariance", size: int = 2) -> Array:
    """A covariance, checked for shape, finiteness, symmetry and definiteness.

    One implementation, because every caller needs the same four checks and
    each one of them catches a distinct mistake that the others let through. A
    wrong *shape* is the mistake worth naming: a ``(2,)`` of variances and a
    ``(3, 3)`` from a pose with an extra component both arrive here looking
    plausible, and a broadcast against ``Phi`` would turn either into numbers.
    """
    values = np.asarray(matrix, dtype=float)
    if values.shape != (size, size):
        raise ValueError(
            f"{name} must be {size}x{size} in the (transverse, heading) basis, "
            f"not {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    scale = max(1.0, float(np.max(np.abs(values))))
    if not np.allclose(values, values.T, rtol=0.0, atol=1e-12 * scale):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    floor = -1e-12 * max(1.0, float(np.max(np.abs(symmetric))))
    if float(np.min(eigenvalues)) < floor:
        raise ValueError(
            f"{name} must be positive semi-definite; its smallest eigenvalue is "
            f"{float(np.min(eigenvalues)):.3e}"
        )
    symmetric.setflags(write=False)
    return symmetric


@dataclass(frozen=True)
class StartingCovariance:
    """``C0``: the starting-pose covariance the record is propagated with.

    The runtime can propagate a covariance -- ``Phi C0 Phi^T`` is one line --
    but it cannot *know* one, so ``C0`` is part of the contract rather than of
    the solver. It carries its own frame and units because a covariance in a
    different frame from the record is a silent rotation error, and one in
    different units is a silent scale error; both survive every check that
    looks only at shapes.

    ``basis`` is the field that makes the matrix readable. A covariance
    measured on a bench and one converted from a declared tolerance box under
    an assumed coverage factor are different claims, and a consumer weighing
    evidence has to be able to tell them apart.
    """

    matrix: Array | None = None
    frame: str = DEFAULT_FRAME
    units: Units = field(default_factory=Units)
    basis: str = "not-declared"
    coverage_factor: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.basis not in COVARIANCE_BASES:
            raise ValueError(f"basis must be one of {COVARIANCE_BASES}")
        frame(self.frame)
        if self.matrix is None:
            if self.basis != "not-declared":
                raise ValueError(f"basis {self.basis!r} declares a covariance but none was given")
            return
        if self.basis == "not-declared":
            raise ValueError("a covariance was given but its basis says not-declared")
        object.__setattr__(self, "matrix", validated_covariance(self.matrix, "C0"))
        if self.coverage_factor is not None:
            factor = float(self.coverage_factor)
            if not np.isfinite(factor) or factor <= 0.0:
                raise ValueError("coverage_factor must be finite and positive")
            object.__setattr__(self, "coverage_factor", factor)

    @property
    def declared(self) -> bool:
        return self.matrix is not None

    @classmethod
    def not_declared(cls, why: str = "") -> StartingCovariance:
        return cls(note=why)

    @classmethod
    def from_tolerance_box(
        cls,
        lateral: float,
        heading: float,
        *,
        coverage_factor: float = 3.0,
        frame: str = DEFAULT_FRAME,
        units: Units | None = None,
        correlation: float = 0.0,
        note: str = "",
    ) -> StartingCovariance:
        """Read a deterministic tolerance box as a covariance, saying how.

        A box is a bound and a covariance is a distribution; turning one into
        the other requires a coverage convention, and the convention is exactly
        the sort of number that must not be chosen silently. It is recorded
        here, so a downstream resolvability figure can be read back as "at
        ``k`` sigma" rather than as an unqualified claim about the hardware.
        """
        k = float(coverage_factor)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("coverage_factor must be finite and positive")
        sigma_lateral, sigma_heading = abs(float(lateral)) / k, abs(float(heading)) / k
        rho = float(correlation)
        if not -1.0 <= rho <= 1.0:
            raise ValueError("correlation must lie in [-1, 1]")
        cross = rho * sigma_lateral * sigma_heading
        return cls(
            matrix=np.array(
                [[sigma_lateral**2, cross], [cross, sigma_heading**2]], dtype=float
            ),
            frame=frame,
            units=units or Units(),
            basis="declared-tolerance-box",
            coverage_factor=k,
            note=note,
        )

    def require(self) -> Array:
        """The matrix, or an error naming what is missing.

        A consumer that needs ``C0`` and silently substitutes one produces a
        number that looks like a result. The house rule is that a declared
        quantity is never skipped: missing evidence is an error, not a pass.
        """
        if self.matrix is None:
            detail = f": {self.note}" if self.note else ""
            raise ValueError(
                "this record declares no starting covariance, so it cannot be "
                f"propagated distributionally{detail}"
            )
        return self.matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared,
            "matrix": None if self.matrix is None else self.matrix.tolist(),
            "frame": self.frame,
            "units": self.units.to_dict(),
            "basis": self.basis,
            "coverage_factor": self.coverage_factor,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> StartingCovariance:
        if not payload:
            return cls.not_declared()
        matrix = payload.get("matrix")
        return cls(
            matrix=None if matrix is None else np.asarray(matrix, dtype=float),
            frame=str(payload.get("frame", DEFAULT_FRAME)),
            units=Units.from_dict(payload.get("units")),
            basis=str(payload.get("basis", "not-declared")),
            coverage_factor=payload.get("coverage_factor"),
            note=str(payload.get("note", "")),
        )


# -- provenance ------------------------------------------------------------


@dataclass(frozen=True)
class UpstreamArtefact:
    """One input this record was computed from, named the way its owner names it.

    The mesh work this runtime deliberately does not do lives in another
    repository, and the path it produces arrives here as a versioned artefact.
    Recording its identity is what keeps the boundary honest in the *inbound*
    direction: a sensitivity record is only as good as the path it was computed
    along, and "which path" has to be answerable later.
    """

    kind: str
    identifier: str
    version: str = ""
    digest: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UpstreamArtefact:
        return cls(
            kind=str(payload.get("kind", "")),
            identifier=str(payload.get("identifier", "")),
            version=str(payload.get("version", "")),
            digest=str(payload.get("digest", "")),
            note=str(payload.get("note", "")),
        )


PRODUCER = "curved-surface-geodesic-sensitivity-runtime"

#: The single source of this runtime's version. ``geodesic_testbed.__version__``
#: and the distribution metadata both read it, and a test holds the three
#: together: provenance that names a version the package does not have is worse
#: than provenance that names none.
RUNTIME_VERSION = "0.2.0"


@dataclass(frozen=True)
class Provenance:
    """Who computed the record, at what version, from what.

    There is no timestamp and no host. Both would be useful and both would make
    two runs of the same computation differ, which would in turn make the
    committed reports unreproducible -- and a report that cannot be regenerated
    and compared is the thing this repository is least willing to ship. A
    caller who needs wall-clock provenance puts it in ``extra``, knowingly.
    """

    producer: str = PRODUCER
    producer_version: str = RUNTIME_VERSION
    upstream: tuple[UpstreamArtefact, ...] = ()
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.producer:
            raise ValueError("provenance must name a producer")
        object.__setattr__(self, "upstream", tuple(self.upstream))

    def with_upstream(self, *artefacts: UpstreamArtefact) -> Provenance:
        return Provenance(
            producer=self.producer,
            producer_version=self.producer_version,
            upstream=self.upstream + tuple(artefacts),
            note=self.note,
            extra=dict(self.extra),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "producer_version": self.producer_version,
            "upstream": [item.to_dict() for item in self.upstream],
            "note": self.note,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> Provenance:
        if not payload:
            return cls()
        return cls(
            producer=str(payload.get("producer", PRODUCER)),
            producer_version=str(payload.get("producer_version", "")),
            upstream=tuple(
                UpstreamArtefact.from_dict(item) for item in payload.get("upstream", ())
            ),
            note=str(payload.get("note", "")),
            extra=dict(payload.get("extra", {})),
        )


# -- calibration binding ---------------------------------------------------


@dataclass(frozen=True)
class CalibrationBinding:
    """Calibration identifiers, carried across the boundary and never read here.

    No calibration exists in this repository and none ever should: calibration
    is a property of an instrument, and an instrument is the thing on the other
    side of this boundary. What the runtime owes downstream is not a
    calibration model but the ability to *check* one -- that the record a
    prediction came from and the measurement it is compared against were
    produced under the same calibration, the same registration and the same
    reconstruction version.

    So these are opaque strings. Nothing here parses them, and nothing here
    treats an unbound record as worse than a bound one: a record computed from
    an analytic surface is correctly unbound, and forcing a placeholder
    identifier into it would be the failure mode this field exists to prevent.
    """

    calibration_ids: tuple[str, ...] = ()
    registration_id: str = ""
    reconstruction_version: str = ""
    instrument_id: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        ids = tuple(str(item) for item in self.calibration_ids)
        if any(not item for item in ids):
            raise ValueError("a calibration identifier may not be empty")
        object.__setattr__(self, "calibration_ids", ids)

    @property
    def bound(self) -> bool:
        return bool(self.calibration_ids)

    @classmethod
    def unbound(cls, why: str = "") -> CalibrationBinding:
        return cls(note=why)

    def agrees_with(self, other: CalibrationBinding) -> bool:
        """Whether two bindings describe the same instrument state.

        Two unbound records do *not* agree: absence of a calibration is not
        evidence of a shared one, and the case this method exists to catch is
        precisely the one where a comparison would otherwise pass by default.
        """
        if not (self.bound and other.bound):
            return False
        return (
            set(self.calibration_ids) == set(other.calibration_ids)
            and self.registration_id == other.registration_id
            and self.reconstruction_version == other.reconstruction_version
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bound": self.bound,
            "calibration_ids": list(self.calibration_ids),
            "registration_id": self.registration_id,
            "reconstruction_version": self.reconstruction_version,
            "instrument_id": self.instrument_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> CalibrationBinding:
        if not payload:
            return cls.unbound()
        return cls(
            calibration_ids=tuple(payload.get("calibration_ids", ())),
            registration_id=str(payload.get("registration_id", "")),
            reconstruction_version=str(payload.get("reconstruction_version", "")),
            instrument_id=str(payload.get("instrument_id", "")),
            note=str(payload.get("note", "")),
        )
