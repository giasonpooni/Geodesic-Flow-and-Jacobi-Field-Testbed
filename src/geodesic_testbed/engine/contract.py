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


# -- the path itself -------------------------------------------------------

#: What kind of curve the transfer map was computed along.
#:
#: This is not decorative. ``j'' + K j = 0`` has no first-derivative term
#: *because* the curve is a geodesic; along a curve with geodesic curvature the
#: variation equation is a different equation, ``det Phi = 1`` is no longer the
#: free invariant it is here, and every guarantee in this repository is about
#: the geodesic case. A record that says ``non-geodesic`` is telling a consumer
#: that its transfer map was computed under an assumption the path does not
#: satisfy, and that is worth one field.
PATH_TYPES: tuple[str, ...] = ("geodesic", "non-geodesic")

#: What an upstream artefact is, so that "which geometry" has a vocabulary
#: rather than a free-text convention that drifts between producers.
ARTEFACT_KINDS: tuple[str, ...] = (
    "analytic-surface",
    "cad-model",
    "as-built-scan",
    "mesh",
    "path-artefact",
    "other",
)

#: Where a geometry uncertainty came from. ``analytic`` is not a synonym for
#: ``not-declared``: a surface given by a formula has *zero* geometry
#: uncertainty, which is a stronger statement than not knowing.
GEOMETRY_UNCERTAINTY_BASES: tuple[str, ...] = (
    "not-declared",
    "analytic",
    "as-built-scan",
    "cad-nominal",
    "assumed",
)


@dataclass(frozen=True)
class GeometryUncertainty:
    """How well the surface and the path along it are actually known.

    Separate from the starting-pose covariance, and combining with it rather
    than replacing it: ``C0`` says where the tool started relative to the
    nominal path, and this says how well the nominal path itself is known. A
    campaign that propagates only the first is reporting a bound on one of two
    terms and calling it the total.

    Nothing here propagates these yet, which is exactly why they are carried:
    the number a downstream uncertainty budget needs must survive the crossing
    even while the crossing is the only thing that happens to it.
    """

    position: float | None = None
    normal: float | None = None
    curvature: float | None = None
    basis: str = "not-declared"
    note: str = ""

    def __post_init__(self) -> None:
        if self.basis not in GEOMETRY_UNCERTAINTY_BASES:
            raise ValueError(f"basis must be one of {GEOMETRY_UNCERTAINTY_BASES}")
        for name in ("position", "normal", "curvature"):
            value = getattr(self, name)
            if value is None:
                continue
            number = float(value)
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} uncertainty must be finite and nonnegative")
            object.__setattr__(self, name, number)
        if self.basis == "not-declared" and any(
            getattr(self, name) is not None for name in ("position", "normal", "curvature")
        ):
            raise ValueError("an uncertainty was given but its basis says not-declared")

    @property
    def declared(self) -> bool:
        return self.basis != "not-declared"

    @classmethod
    def not_declared(cls, why: str = "") -> GeometryUncertainty:
        return cls(note=why)

    @classmethod
    def analytic(cls, why: str = "surface given in closed form") -> GeometryUncertainty:
        """Zero, and meant: a formula has no reconstruction error."""
        return cls(position=0.0, normal=0.0, curvature=0.0, basis="analytic", note=why)

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared,
            "position": self.position,
            "normal": self.normal,
            "curvature": self.curvature,
            "basis": self.basis,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> GeometryUncertainty:
        if not payload:
            return cls.not_declared()
        return cls(
            position=payload.get("position"),
            normal=payload.get("normal"),
            curvature=payload.get("curvature"),
            basis=str(payload.get("basis", "not-declared")),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class ChartValidity:
    """How much of the requested path the parameterisation could actually carry.

    A record never holds an invalid sample -- integration stops at the chart
    edge and the envelope is truncated there -- so this is not a mask over the
    samples in hand. It is the other half of the story: that the path in the
    record is *shorter than the one that was asked for*, and why.

    A consumer asking whether a route covers a part has to be able to tell a
    route that ends because it is finished from one that ends because the
    parameterisation ran out. Both look like a path of some length.
    """

    samples: int
    requested_samples: int
    truncated: bool = False
    reason: str = ""
    truncated_at: float | None = None
    requested_length: float | None = None
    min_conditioning: float | None = None
    declared_domain: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.samples) < 2:
            raise ValueError("a chart record needs at least two valid samples")
        if int(self.requested_samples) < int(self.samples):
            raise ValueError("more samples are valid than were requested")
        if self.truncated and not self.reason:
            raise ValueError("a truncated path must say why it was truncated")

    @property
    def complete(self) -> bool:
        """Whether the path in the record is the path that was requested."""
        return not self.truncated

    def mask(self) -> Array:
        """The valid prefix as a boolean mask, for a consumer that wants one."""
        return np.arange(int(self.requested_samples)) < int(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": int(self.samples),
            "requested_samples": int(self.requested_samples),
            "complete": self.complete,
            "truncated": bool(self.truncated),
            "reason": self.reason,
            "truncated_at": self.truncated_at,
            "requested_length": self.requested_length,
            "min_conditioning": self.min_conditioning,
            "declared_domain": dict(self.declared_domain),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ChartValidity | None:
        if not payload:
            return None
        return cls(
            samples=int(payload["samples"]),
            requested_samples=int(payload["requested_samples"]),
            truncated=bool(payload.get("truncated", False)),
            reason=str(payload.get("reason", "")),
            truncated_at=payload.get("truncated_at"),
            requested_length=payload.get("requested_length"),
            min_conditioning=payload.get("min_conditioning"),
            declared_domain=dict(payload.get("declared_domain", {})),
        )


def _unit_vectors(values: Any, name: str, size: int) -> Array:
    array = np.asarray(values, dtype=float)
    if array.shape != (size, 3):
        raise ValueError(f"{name} must be ({size}, 3) ambient vectors, not {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(array, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{name} must be unit vectors; the worst norm is "
            f"{float(np.max(np.abs(norms - 1.0))):.3e} from 1"
        )
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PathGeometry:
    """Where the path is, how it is oriented, and how the surface bends there.

    The transfer map alone says how an error grows; it does not say where the
    error is. A consumer that has to point an instrument at the path, register
    a measurement to it, or turn a transverse deviation into a coordinate,
    needs the path itself -- and cannot recompute it, because recomputing it
    would mean carrying this runtime's solver.

    The frame is carried as *vectors*, not only as a name. ``frame`` on the
    record says the transverse direction is parallel-transported; these three
    columns are what that direction actually is at every sample, so a consumer
    can check the claim instead of trusting it. ``transverse`` is the direction
    ``a`` and ``b`` are measured along, and ``normal x tangent`` is how it is
    built, which fixes its sign -- a frame that is right in every respect but
    orientation flips the sign of every heading error it carries.

    Two normal curvatures, because they answer different questions and the
    literature calls both of them ``kappa_n``. ``normal_curvature_along`` is
    how the surface bends in the direction of travel. It is
    ``normal_curvature_transverse`` that sets how far an ambient chord falls
    short of an in-surface separation -- the ``kappa_n^2 sn_K(s)^2`` term -- so
    it is the one a comparison against reconstructed 3-D points needs.
    """

    position: Array
    tangent: Array
    transverse: Array
    surface_normal: Array
    normal_curvature_along: Array
    normal_curvature_transverse: Array
    coordinate_frame: str = "surface-parameterisation-ambient"
    datum_frame: str = "not-declared"
    geodesic_curvature: Array | None = None
    uncertainty: GeometryUncertainty = field(default_factory=GeometryUncertainty.not_declared)

    def __post_init__(self) -> None:
        size = np.asarray(self.position).shape[0]
        for name in ("position",):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (size, 3):
                raise ValueError(f"{name} must be ({size}, 3) ambient points")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("tangent", "transverse", "surface_normal"):
            object.__setattr__(self, name, _unit_vectors(getattr(self, name), name, size))
        for name in ("normal_curvature_along", "normal_curvature_transverse"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (size,):
                raise ValueError(f"{name} must be one value per sample")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.geodesic_curvature is not None:
            array = np.asarray(self.geodesic_curvature, dtype=float)
            if array.shape != (size,) or not np.all(np.isfinite(array)):
                raise ValueError("geodesic_curvature must be one finite value per sample")
            array.setflags(write=False)
            object.__setattr__(self, "geodesic_curvature", array)
        # The triad is what makes the frame checkable, so it is checked here.
        # A transverse direction that has drifted out of the tangent plane is
        # measuring something other than an in-surface deviation.
        for left, right in (
            ("tangent", "surface_normal"),
            ("tangent", "transverse"),
            ("transverse", "surface_normal"),
        ):
            products = np.einsum(
                "ij,ij->i", getattr(self, left), getattr(self, right)
            )
            if not np.allclose(products, 0.0, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"{left} and {right} must be orthogonal at every sample; the "
                    f"worst inner product is {float(np.max(np.abs(products))):.3e}"
                )

    @property
    def samples(self) -> int:
        return int(self.position.shape[0])

    def orientation_residual(self) -> float:
        """How far ``normal x tangent`` is from the carried transverse direction.

        Zero for a right-handed Darboux frame. A consumer can call this rather
        than assume the handedness, which is the point of carrying the vectors.
        """
        expected = np.cross(self.surface_normal, self.tangent)
        return float(np.max(np.linalg.norm(expected - self.transverse, axis=-1)))

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "samples": self.samples,
            "coordinate_frame": self.coordinate_frame,
            "datum_frame": self.datum_frame,
            "uncertainty": self.uncertainty.to_dict(),
            "orientation_residual": self.orientation_residual(),
            "has_geodesic_curvature": self.geodesic_curvature is not None,
        }
        if include_samples:
            payload |= {
                name: np.asarray(getattr(self, name)).tolist()
                for name in (
                    "position",
                    "tangent",
                    "transverse",
                    "surface_normal",
                    "normal_curvature_along",
                    "normal_curvature_transverse",
                )
            }
            payload["geodesic_curvature"] = (
                None
                if self.geodesic_curvature is None
                else np.asarray(self.geodesic_curvature).tolist()
            )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> PathGeometry | None:
        if not payload:
            return None
        required = (
            "position",
            "tangent",
            "transverse",
            "surface_normal",
            "normal_curvature_along",
            "normal_curvature_transverse",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "this geometry payload carries no samples "
                f"(missing {', '.join(missing)}); it is a summary, not a geometry"
            )
        geodesic = payload.get("geodesic_curvature")
        return cls(
            **{name: np.asarray(payload[name], dtype=float) for name in required},
            coordinate_frame=str(
                payload.get("coordinate_frame", "surface-parameterisation-ambient")
            ),
            datum_frame=str(payload.get("datum_frame", "not-declared")),
            geodesic_curvature=(
                None if geodesic is None else np.asarray(geodesic, dtype=float)
            ),
            uncertainty=GeometryUncertainty.from_dict(payload.get("uncertainty")),
        )


# -- how well it was solved ------------------------------------------------


@dataclass(frozen=True)
class ConvergenceEstimate:
    """What the numerics cost, per quantity, from a step-doubling comparison.

    A single "the method is fourth order" is a statement about the limit, not
    about this run. What a consumer needs is the size of the error *here*, and
    it needs it separately per quantity, because they do not converge together:
    a focus location is a root of ``b``, so its error is the error in ``b``
    divided by a slope that is small precisely where the focus is, and a
    propagated covariance is quadratic in ``Phi`` so its relative error is
    roughly twice the transfer map's.

    Every entry is an absolute error estimate in the record's own units, from
    ``run(h)`` against ``run(h/2)`` Richardson-extrapolated at the method's
    order. ``not-established`` is the honest default and stays the default:
    the estimate costs a second integration, and a producer that did not pay
    for it must not appear to have.
    """

    basis: str = "not-established"
    order: int | None = None
    refinement: int | None = None
    position: float | None = None
    transfer: float | None = None
    curvature: float | None = None
    focus: float | None = None
    covariance: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("position", "transfer", "curvature", "focus", "covariance"):
            value = getattr(self, name)
            if value is None:
                continue
            number = float(value)
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"the {name} error estimate must be finite and nonnegative")
            object.__setattr__(self, name, number)
        if not self.established and any(
            getattr(self, name) is not None
            for name in ("position", "transfer", "curvature", "focus", "covariance")
        ):
            raise ValueError("an error estimate was given but the basis says not-established")

    @property
    def established(self) -> bool:
        return not self.basis.startswith("not-established")

    @classmethod
    def not_established(cls, why: str = "") -> ConvergenceEstimate:
        return cls(basis=f"not-established: {why}" if why else "not-established", note=why)

    @property
    def worst(self) -> float | None:
        """The largest of the declared estimates, or ``None`` if none are."""
        values = [
            getattr(self, name)
            for name in ("position", "transfer", "curvature", "focus", "covariance")
            if getattr(self, name) is not None
        ]
        return max(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"established": self.established, "worst": self.worst}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ConvergenceEstimate:
        if not payload:
            return cls.not_established()
        return cls(
            basis=str(payload.get("basis", "not-established")),
            order=payload.get("order"),
            refinement=payload.get("refinement"),
            position=payload.get("position"),
            transfer=payload.get("transfer"),
            curvature=payload.get("curvature"),
            focus=payload.get("focus"),
            covariance=payload.get("covariance"),
            note=str(payload.get("note", "")),
        )
