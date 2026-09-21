# SPDX-License-Identifier: MPL-2.0
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
    PATH_TYPES,
    CalibrationBinding,
    ChartValidity,
    ConvergenceEstimate,
    GeometryUncertainty,
    PathGeometry,
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
    "PATH_TYPES",
    "RECORD_SCHEMA",
    "SUPPORTED_RECORD_SCHEMAS",
    "CalibrationBinding",
    "ChartValidity",
    "ConvergenceEstimate",
    "PERTURBATION_DIRECTIONS",
    "VALIDITY_REFERENCES",
    "ProbeFit",
    "ValidityEnvelope",
    "GeometryUncertainty",
    "PathGeometry",
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
    """How the record was computed, in enough detail to judge what it resolves.

    The method and the step say what was run; ``convergence`` says what it
    cost. Both are needed and neither substitutes for the other: "rk4 at
    h = 0.005" is a recipe, and a consumer deciding whether a focus at
    ``s = 3.1416`` is located well enough to plan against needs the error, not
    the recipe.
    """

    method: str
    samples: int
    max_step: float
    uniform: bool
    convergence: ConvergenceEstimate = field(
        default_factory=lambda: ConvergenceEstimate.not_established(
            "no step-doubling comparison was run"
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"convergence": self.convergence.to_dict()}


#: How a validity envelope was arrived at. The prefix ``not-established``
#: marks the honest answer rather than a missing one.
VALIDITY_REFERENCES: tuple[str, ...] = (
    "closed-form",
    "geodesic-flow-central-difference",
    "declared-by-caller",
    "none",
)

#: The two columns of ``Phi``, which is what a perturbation direction is.
PERTURBATION_DIRECTIONS: tuple[str, ...] = ("lateral", "heading")


@dataclass(frozen=True)
class ProbeFit:
    """How one column's bound was arrived at, in enough detail to re-derive it.

    The bound is the output of an adaptive fit: probes whose local log-log
    slope has left the quadratic regime are dropped, and ``C`` is fitted on
    what remains. Which probes those were is a decision the fit made, and a
    decision nobody can see is not auditable -- two ladders could give the same
    bound from entirely different evidence. So the window, the slopes, the
    coefficient and every rejected probe with its reason are carried.

    ``held_out`` is the part that can falsify the bound rather than describe
    it. The linearisation is re-probed at fractions of the fitted bound that
    took no part in the fit; at ``1.2`` the relative error must *exceed* the
    declared tolerance, or the bound is not where the linearisation fails.
    """

    direction: str
    coefficient: float
    fitted_probes: tuple[float, ...]
    observed_slopes: tuple[float, ...]
    rejected_probes: tuple[tuple[float, str], ...] = ()
    bound: float | None = None
    probe_limited: bool = False
    held_out: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.direction not in PERTURBATION_DIRECTIONS:
            raise ValueError(f"direction must be one of {PERTURBATION_DIRECTIONS}")
        object.__setattr__(self, "fitted_probes", tuple(float(v) for v in self.fitted_probes))
        object.__setattr__(self, "observed_slopes", tuple(float(v) for v in self.observed_slopes))
        object.__setattr__(
            self,
            "rejected_probes",
            tuple((float(value), str(reason)) for value, reason in self.rejected_probes),
        )
        object.__setattr__(
            self, "held_out", tuple((float(a), float(b)) for a, b in self.held_out)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "coefficient": self.coefficient,
            "bound": self.bound,
            "probe_limited": self.probe_limited,
            "fitted_probes": list(self.fitted_probes),
            "observed_slopes": list(self.observed_slopes),
            "rejected_probes": [
                {"magnitude": value, "reason": reason} for value, reason in self.rejected_probes
            ],
            "held_out": [
                {"fraction": fraction, "relative_error": error}
                for fraction, error in self.held_out
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProbeFit:
        return cls(
            direction=str(payload["direction"]),
            coefficient=float(payload["coefficient"]),
            fitted_probes=tuple(payload.get("fitted_probes", ())),
            observed_slopes=tuple(payload.get("observed_slopes", ())),
            rejected_probes=tuple(
                (item["magnitude"], item["reason"]) for item in payload.get("rejected_probes", ())
            ),
            bound=payload.get("bound"),
            probe_limited=bool(payload.get("probe_limited", False)),
            held_out=tuple(
                (item["fraction"], item["relative_error"]) for item in payload.get("held_out", ())
            ),
        )


@dataclass(frozen=True)
class ValidityEnvelope:
    """How far from the nominal path the linear map is declared to hold.

    ``Phi dz0`` is the first term of a series, and the second term is the same
    order as the effect most campaigns here are trying to resolve. So the
    question "over what range of starting errors is this map the answer" is
    not a caveat, it is a number -- and a number with no statement of how it
    was obtained is not usable: a bound measured against an independently
    flowed finite separation means something entirely different from one a
    caller asserted.

    Four things make the range readable and each is carried:

    *which perturbation* -- ``directions`` says whether the heading column, the
    lateral column or both were exercised. A bound established by perturbing
    only the heading says nothing about ``a``, and the two focus in different
    places;

    *against what* -- ``reference`` and ``reference_digest`` name the nonlinear
    computation the linear map was compared with. The geodesic flow itself is
    the reference on a parametric surface: it never touches the Jacobi
    equation, so it is a genuinely independent measurement of the same
    quantity;

    *to what error, chosen how* -- ``relative_tolerance`` with
    ``tolerance_basis``. A tolerance picked after seeing the residuals is not
    a tolerance;

    *how well it was computed* -- ``convergence``. An envelope whose own
    numerics are not resolved is a bound on the solver, not on the
    linearisation.

    ``pointwise_error`` and ``route_error`` are kept apart. The first is the
    worst relative departure at any single arc length; the second is over the
    whole route. A path can be linear everywhere and still accumulate, and a
    campaign that plans to a route-level figure while measuring pointwise is
    comparing two different numbers.
    """

    basis: str
    observation_mode: str = DEFAULT_MODE
    relative_tolerance: float | None = None
    tolerance_basis: str = "not-declared"
    max_lateral: float | None = None
    max_heading: float | None = None
    directions: tuple[str, ...] = ()
    probe_magnitudes: tuple[float, ...] = ()
    pointwise_error: float | None = None
    route_error: float | None = None
    #: The directions whose bound is the end of the probe ladder rather than
    #: where the linearisation fails. Per direction and not a single flag,
    #: because on an intrinsically flat surface the lateral column is exact --
    #: the fitted quadratic coefficient is roundoff and the extrapolated bound
    #: is meaningless -- while the heading column is measured normally. What
    #: was established for the first is that it held out to the largest
    #: perturbation tested, and a boolean over the whole envelope would have
    #: to say that about both.
    probe_limited_directions: tuple[str, ...] = ()
    #: One per exercised direction: the fit that produced that column's bound,
    #: the probes it used, the ones it rejected and why, and the held-out
    #: re-probes that can falsify it.
    fits: tuple[ProbeFit, ...] = ()
    reference: str = "none"
    #: The integrator and step count the nonlinear reference was flowed with.
    #: Declared because the bound depends on them: the probe and the transfer
    #: map it is compared against share an integrator, so its truncation error
    #: is common mode, and it has to sit well below the linearisation error
    #: being measured. With a second-order method at the same steps it does
    #: not, and the fitted bound stops converging -- it moves by ~2% and
    #: non-monotonically in the step count, which is the tell.
    reference_method: str = ""
    reference_samples: int | None = None
    reference_digest: str = ""
    convergence: ConvergenceEstimate = field(
        default_factory=lambda: ConvergenceEstimate.not_established(
            "no step-doubling comparison was run on the validity probe"
        )
    )
    note: str = ""

    def __post_init__(self) -> None:
        if self.reference not in VALIDITY_REFERENCES:
            raise ValueError(f"reference must be one of {VALIDITY_REFERENCES}")
        for direction in self.directions:
            if direction not in PERTURBATION_DIRECTIONS:
                raise ValueError(f"directions must be drawn from {PERTURBATION_DIRECTIONS}")
        for direction in self.probe_limited_directions:
            if direction not in self.directions:
                raise ValueError(
                    f"{direction!r} is marked probe-limited but was never exercised"
                )
        for name in ("relative_tolerance", "max_lateral", "max_heading",
                     "pointwise_error", "route_error"):
            value = getattr(self, name)
            if value is None:
                continue
            number = float(value)
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, number)
        if self.established:
            if not self.directions:
                raise ValueError(
                    "an established envelope must say which perturbation directions "
                    "were exercised; one that perturbs only the heading measures b "
                    "and says nothing about a"
                )
            if self.relative_tolerance is None:
                raise ValueError(
                    "an established envelope must carry the error tolerance it was "
                    "established against; a range with no error is not a range"
                )
            if self.tolerance_basis in ("", "not-declared"):
                raise ValueError(
                    "an established envelope must say how its tolerance was chosen; "
                    "a tolerance picked after seeing the residuals is not a tolerance"
                )
        object.__setattr__(self, "directions", tuple(self.directions))
        object.__setattr__(
            self, "probe_limited_directions", tuple(self.probe_limited_directions)
        )
        object.__setattr__(self, "fits", tuple(self.fits))
        for fit in self.fits:
            if fit.direction not in self.directions:
                raise ValueError(
                    f"a fit is carried for {fit.direction!r}, which was never exercised"
                )
        object.__setattr__(
            self, "probe_magnitudes", tuple(float(value) for value in self.probe_magnitudes)
        )

    @property
    def probe_limited(self) -> bool:
        """Whether any direction's bound is the ladder's end rather than a limit."""
        return bool(self.probe_limited_directions)

    def bound_is_measured(self, direction: str) -> bool:
        """Whether this direction's bound is where the linearisation actually fails."""
        return direction in self.directions and direction not in self.probe_limited_directions

    @classmethod
    def not_established(cls, why: str) -> ValidityEnvelope:
        return cls(basis=f"not-established: {why}")

    @property
    def established(self) -> bool:
        return not self.basis.startswith("not-established")

    def admits(self, lateral: float, heading: float) -> bool:
        """Whether a starting-pose error is inside the declared envelope.

        An undeclared bound does not admit anything. That is deliberate: the
        alternative reading -- an absent bound as no limit -- turns "we never
        checked" into "it always holds", which is the one substitution this
        field exists to prevent.
        """
        if not self.established:
            return False
        for name, value, bound in (
            ("lateral", abs(float(lateral)), self.max_lateral),
            ("heading", abs(float(heading)), self.max_heading),
        ):
            if value == 0.0:
                continue
            if name not in self.directions or bound is None or value > bound:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "established": self.established,
            "observation_mode": self.observation_mode,
            "relative_tolerance": self.relative_tolerance,
            "tolerance_basis": self.tolerance_basis,
            "max_lateral": self.max_lateral,
            "max_heading": self.max_heading,
            "directions": list(self.directions),
            "probe_magnitudes": list(self.probe_magnitudes),
            "pointwise_error": self.pointwise_error,
            "route_error": self.route_error,
            "probe_limited": self.probe_limited,
            "probe_limited_directions": list(self.probe_limited_directions),
            "fits": [fit.to_dict() for fit in self.fits],
            "reference": self.reference,
            "reference_method": self.reference_method,
            "reference_samples": self.reference_samples,
            "reference_digest": self.reference_digest,
            "convergence": self.convergence.to_dict(),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ValidityEnvelope:
        if not payload:
            return cls.not_established("no declaration supplied")
        return cls(
            basis=str(payload.get("basis", "not-established: no declaration supplied")),
            observation_mode=str(payload.get("observation_mode", DEFAULT_MODE)),
            relative_tolerance=payload.get("relative_tolerance"),
            tolerance_basis=str(payload.get("tolerance_basis", "not-declared")),
            max_lateral=payload.get("max_lateral"),
            max_heading=payload.get("max_heading"),
            directions=tuple(payload.get("directions", ())),
            probe_magnitudes=tuple(payload.get("probe_magnitudes", ())),
            pointwise_error=payload.get("pointwise_error"),
            route_error=payload.get("route_error"),
            probe_limited_directions=tuple(payload.get("probe_limited_directions", ())),
            fits=tuple(ProbeFit.from_dict(item) for item in payload.get("fits", ())),
            reference=str(payload.get("reference", "none")),
            reference_method=str(payload.get("reference_method", "")),
            reference_samples=payload.get("reference_samples"),
            reference_digest=str(payload.get("reference_digest", "")),
            convergence=ConvergenceEstimate.from_dict(payload.get("convergence")),
            note=str(payload.get("note", "")),
        )


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
    validity: ValidityEnvelope = field(
        default_factory=lambda: ValidityEnvelope.not_established("no declaration supplied")
    )
    observation_mode: str = DEFAULT_MODE
    domain: str = "constant-curvature"
    covariance: StartingCovariance = field(default_factory=StartingCovariance.not_declared)
    provenance: Provenance = field(default_factory=Provenance)
    calibration: CalibrationBinding = field(default_factory=CalibrationBinding.unbound)
    #: The path itself: positions, the frame as vectors, and the two normal
    #: curvatures. ``None`` where there is no embedding to sample -- a declared
    #: curvature profile is not a surface and has no points.
    geometry: PathGeometry | None = None
    #: What kind of curve this is. The transfer map's equation assumes a
    #: geodesic, so a consumer is entitled to be told.
    path_type: str = "geodesic"
    path_type_basis: str = "declared by the producer"
    #: How much of the requested path the chart could carry. ``None`` where the
    #: producer has no chart -- again, a declared curvature profile.
    chart: ChartValidity | None = None

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
        if self.path_type not in PATH_TYPES:
            raise ValueError(f"path_type must be one of {PATH_TYPES}")
        if self.geometry is not None and self.geometry.samples != size:
            raise ValueError(
                f"the geometry carries {self.geometry.samples} samples but the "
                f"transfer map has {size}; they are the same path or they are not "
                "the same record"
            )
        if self.chart is not None and int(self.chart.samples) != size:
            raise ValueError(
                f"the chart record says {self.chart.samples} samples are valid but "
                f"the record carries {size}; a record holds only valid samples"
            )
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

    def with_geometry_uncertainty(
        self, uncertainty: GeometryUncertainty
    ) -> TransferRecord:
        """The same record, declaring how well its own geometry is known.

        Usually filled in later and elsewhere: how well the surface is known is
        a property of the scan or the CAD model it came from, and the runtime
        that flowed a path along it is not the thing that measured it.
        """
        if self.geometry is None:
            raise ValueError(
                "this record carries no geometry, so there is nothing for a "
                "geometry uncertainty to describe"
            )
        return replace(
            self, geometry=replace(self.geometry, uncertainty=uncertainty)
        )

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
            "path_type": self.path_type,
            "path_type_basis": self.path_type_basis,
            "chart": None if self.chart is None else self.chart.to_dict(),
            "geometry": (
                None
                if self.geometry is None
                else self.geometry.to_dict(include_samples=include_samples)
            ),
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
                convergence=ConvergenceEstimate.from_dict(resolution.get("convergence")),
            ),
            validity=ValidityEnvelope.from_dict(validity),
            observation_mode=str(payload.get("observation_mode", DEFAULT_MODE)),
            domain=str(payload.get("domain", "constant-curvature")),
            covariance=StartingCovariance.from_dict(payload.get("covariance")),
            provenance=Provenance.from_dict(payload.get("provenance")),
            calibration=CalibrationBinding.from_dict(payload.get("calibration")),
            geometry=PathGeometry.from_dict(payload.get("geometry")),
            path_type=str(payload.get("path_type", "geodesic")),
            path_type_basis=str(payload.get("path_type_basis", "declared by the producer")),
            chart=ChartValidity.from_dict(payload.get("chart")),
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
