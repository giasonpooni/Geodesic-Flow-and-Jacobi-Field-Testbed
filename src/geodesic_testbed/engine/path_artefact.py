"""``path-geometry-v1``: the versioned path this runtime consumes.

Mesh work -- triangulated surfaces, discrete curvature estimators, mesh path
convergence -- belongs upstream, in the Intrinsic Surface Geodesics Testbed.
What crosses into this runtime is the *result* of that work: a sampled path
with everything needed to integrate the Jacobi equation along it and nothing
that would require tracing a mesh to obtain.

This module is the inbound half of the boundary ``docs/BOUNDARY.md`` states.
:class:`~geodesic_testbed.engine.record.TransferRecord` is what leaves; this is
what arrives, and the two are deliberately not the same shape. A record carries
a transfer map and no chart, because a consumer cannot be warned by a field it
was not given. An artefact carries a chart and no transfer map, because
producing one is this runtime's whole job.

Nine things have to cross, and each is here because it cannot be recovered on
this side:

``arclength``
    strictly increasing, and the grid every other array is on. Not derivable
    from the positions: a producer that resampled its path knows the arc length
    it resampled along, and recomputing it from chords would lose exactly the
    curvature-dependent difference this runtime exists to compute.
``position`` and ``tangent``
    where the path is and which way it points, in the producer's coordinate
    frame.
``transverse``
    the direction ``a`` and ``b`` are measured along, parallel-transported by
    the producer. Carried rather than reconstructed, because reconstructing it
    means transporting along the path, which needs the connection, which needs
    the surface -- and the surface is what stayed upstream.
``gaussian_curvature``
    the one coefficient of the Jacobi equation.
``units`` and the frame identifiers
    there is no safe default for either.
``digests``
    which surface, which path, which producer run.
``sampling`` and ``curvature_interpolation``
    the curvature *between* samples is not known, and which interpolant fills
    it in changes the answer. A producer that sampled a rapidly varying ``K``
    every 5 mm and a consumer that assumed piecewise-constant curvature
    disagree by a term of the same order as the effects measured here, so the
    choice is declared rather than made quietly on arrival.
``uncertainty``
    how well the surface and the path are actually known, with its basis.
``validity`` and ``upstream_status``
    how much of the requested path the producer could actually produce, and
    whether the run that produced it succeeded. A failed upstream artefact is
    refused here rather than integrated: a path that is wrong in a way its
    producer already detected is worse than no path, because it looks like one.

The normal curvatures are optional and *jointly* required with the mean
curvature. A producer that has them supplies all three and Euler's theorem is
checked on arrival; a producer that has none supplies none and the record says
so. Supplying two of the three would put a number into the contract that
nothing here can check, and a number the record carries is a number something
checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .contract import (
    FRAMES,
    ChartValidity,
    GeometryUncertainty,
    Units,
    UpstreamArtefact,
)

Array = np.ndarray

#: The schema this module reads and writes.
PATH_GEOMETRY_SCHEMA = "path-geometry-v1"

#: How the producer chose where the samples land. This is a statement about
#: the *grid*, not about the interpolation between its points.
SAMPLING_POLICIES: tuple[str, ...] = (
    "uniform-arclength",
    "nonuniform-arclength",
    "curvature-adaptive",
    "declared-by-producer",
)

#: How curvature is to be read between the samples. Every one of these is a
#: different function, and the transfer map they produce differs by a term of
#: the order this runtime measures.
CURVATURE_INTERPOLATIONS: tuple[str, ...] = (
    "pchip-monotone",
    "linear",
    "piecewise-constant",
)

#: The sentinel a missing ``curvature_interpolation`` carries. A dataclass
#: field must have *some* default to sit after one that does, and a string
#: default would be indistinguishable from a producer that chose it -- which is
#: the whole failure this guards. Identity against this object is not.
MISSING_INTERPOLATION: str = "<<curvature_interpolation not declared>>"

#: Whether the upstream run that produced this artefact succeeded.
#: ``degraded`` is carried through into provenance; ``failed`` is refused.
UPSTREAM_STATUS: tuple[str, ...] = ("ok", "degraded", "failed")

#: The domain an imported artefact lives in, for the observation-mode table.
ARTEFACT_DOMAIN = "imported-path-artefact"

#: A frame and a triad are held to this. It is far above the canonical floor
#: and far below anything a producer that actually transported a frame would
#: fail, so it separates "transported" from "interpolated and not renormalised".
ORTHONORMALITY_TOLERANCE = 1e-9

#: Euler's theorem is an identity, so the tolerance is a numerical one. A
#: producer estimating normal curvatures from a mesh will not do better than
#: its own discretisation, and is expected to declare that in ``uncertainty``
#: rather than to be held to roundoff here.
EULER_TOLERANCE = 1e-7


def _finite_vector(values: Any, name: str, size: int) -> Array:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must be one finite value per sample, not {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _finite_points(values: Any, name: str, size: int) -> Array:
    array = np.asarray(values, dtype=float)
    if array.shape != (size, 3):
        raise ValueError(f"{name} must be ({size}, 3) ambient vectors, not {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _unit_rows(values: Any, name: str, size: int) -> Array:
    array = _finite_points(values, name, size)
    norms = np.linalg.norm(array, axis=-1)
    drift = float(np.max(np.abs(norms - 1.0)))
    if drift > ORTHONORMALITY_TOLERANCE:
        raise ValueError(
            f"{name} must be unit vectors; the worst norm departs from 1 by {drift:.3e}, "
            f"above {ORTHONORMALITY_TOLERANCE:.0e}. Renormalise upstream: a tangent that "
            "is not unit means the arclength grid is not arclength."
        )
    return array


@dataclass(frozen=True)
class SamplingPolicy:
    """Where the samples are and why they are there."""

    policy: str = "declared-by-producer"
    requested_spacing: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.policy not in SAMPLING_POLICIES:
            raise ValueError(f"sampling policy must be one of {SAMPLING_POLICIES}")
        if self.requested_spacing is not None:
            spacing = float(self.requested_spacing)
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError("requested_spacing must be finite and positive")
            object.__setattr__(self, "requested_spacing", spacing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "requested_spacing": self.requested_spacing,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SamplingPolicy:
        if not payload:
            return cls()
        return cls(
            policy=str(payload.get("policy", "declared-by-producer")),
            requested_spacing=payload.get("requested_spacing"),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class PathGeometryArtefact:
    """One sampled path, as the upstream testbed hands it over.

    Construction validates; it does not repair. A tangent that is not unit, a
    triad that is not orthogonal, an arclength grid that steps backwards and a
    length unit nobody declared are each refused here, where the artefact can
    still be pointed at, rather than absorbed into a transfer map where they
    become a wrong answer with no symptom.
    """

    identifier: str
    arclength: Array
    position: Array
    tangent: Array
    transverse: Array
    gaussian_curvature: Array
    units: Units
    surface_digest: str
    path_digest: str
    schema: str = PATH_GEOMETRY_SCHEMA
    version: str = "1"
    producer: str = ""
    frame: str = "transverse-to-gamma, parallel-transported"
    coordinate_frame: str = "producer-declared-ambient"
    datum_frame: str = "not-declared"
    path_type: str = "geodesic"
    path_type_basis: str = "declared by the producer"
    sampling: SamplingPolicy = field(default_factory=SamplingPolicy)
    #: Required. There is no default, and ``from_dict`` will not supply one:
    #: the field changes the answer by a factor of 580 at a sampling a producer
    #: might reasonably choose, so a missing one is a refusal rather than a
    #: silently selected monotone cubic. ``MISSING_INTERPOLATION`` is the value
    #: an artefact constructed without it carries into the check below.
    curvature_interpolation: str = MISSING_INTERPOLATION
    uncertainty: GeometryUncertainty = field(default_factory=GeometryUncertainty.not_declared)
    validity: ChartValidity | None = None
    upstream_status: str = "ok"
    upstream_note: str = ""
    #: Jointly required with ``mean_curvature``: all three or none.
    normal_curvature_along: Array | None = None
    normal_curvature_transverse: Array | None = None
    mean_curvature: Array | None = None
    geodesic_curvature: Array | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != PATH_GEOMETRY_SCHEMA:
            raise ValueError(
                f"this reader understands {PATH_GEOMETRY_SCHEMA!r}, not {self.schema!r}"
            )
        if not self.identifier:
            raise ValueError("an artefact must name itself")
        arclength = np.asarray(self.arclength, dtype=float)
        if arclength.ndim != 1 or arclength.size < 2:
            raise ValueError("arclength must be a vector with at least two samples")
        if not np.all(np.isfinite(arclength)):
            raise ValueError("arclength must be finite")
        if np.any(np.diff(arclength) <= 0.0):
            raise ValueError(
                "arclength must be strictly increasing; a repeated or reversed sample "
                "makes every interpolant here ill-posed"
            )
        arclength.setflags(write=False)
        object.__setattr__(self, "arclength", arclength)
        size = arclength.size

        object.__setattr__(self, "position", _finite_points(self.position, "position", size))
        object.__setattr__(self, "tangent", _unit_rows(self.tangent, "tangent", size))
        object.__setattr__(self, "transverse", _unit_rows(self.transverse, "transverse", size))
        object.__setattr__(
            self,
            "gaussian_curvature",
            _finite_vector(self.gaussian_curvature, "gaussian_curvature", size),
        )

        products = np.einsum("ij,ij->i", self.tangent, self.transverse)
        worst = float(np.max(np.abs(products)))
        if worst > ORTHONORMALITY_TOLERANCE:
            raise ValueError(
                f"tangent and transverse must be orthogonal at every sample; the worst "
                f"inner product is {worst:.3e}. A transverse direction that has drifted "
                "out of the tangent plane is not measuring an in-surface deviation."
            )

        declared = [
            name
            for name in ("normal_curvature_along", "normal_curvature_transverse", "mean_curvature")
            if getattr(self, name) is not None
        ]
        if declared and len(declared) != 3:
            raise ValueError(
                "normal_curvature_along, normal_curvature_transverse and mean_curvature "
                f"are declared together or not at all; this artefact declares {declared}. "
                "Two of the three would put a number in the contract that Euler's theorem "
                "cannot be checked against."
            )
        for name in declared:
            object.__setattr__(self, name, _finite_vector(getattr(self, name), name, size))
        if declared:
            residual = np.abs(
                self.normal_curvature_along
                + self.normal_curvature_transverse
                - 2.0 * self.mean_curvature
            )
            scale = np.maximum(1.0, np.abs(2.0 * self.mean_curvature))
            worst_euler = float(np.max(residual / scale))
            if worst_euler > EULER_TOLERANCE:
                raise ValueError(
                    "Euler's theorem fails on this artefact: kappa_n(along) + "
                    f"kappa_n(across) departs from 2H by {worst_euler:.3e} relative, above "
                    f"{EULER_TOLERANCE:.0e}. The two directions are orthogonal by "
                    "construction, so the sum is an identity, not an approximation."
                )
        if self.geodesic_curvature is not None:
            object.__setattr__(
                self,
                "geodesic_curvature",
                _finite_vector(self.geodesic_curvature, "geodesic_curvature", size),
            )

        if self.frame not in FRAMES:
            raise ValueError(f"frame must be one of {sorted(FRAMES)}, not {self.frame!r}")
        if self.units.length == Units().length:
            raise ValueError(
                "an inbound artefact must declare its length unit; "
                f"{Units().length!r} is the placeholder, not a unit"
            )
        if not self.units.angle:
            raise ValueError("an inbound artefact must declare its angle unit")
        if self.curvature_interpolation is MISSING_INTERPOLATION:
            raise ValueError(
                "curvature_interpolation is required. The curvature between the "
                "producer's samples is not known, and which curve fills it in changes "
                f"the transfer map: on the saddle {CURVATURE_INTERPOLATIONS[0]!r} and "
                f"{CURVATURE_INTERPOLATIONS[1]!r} differ by a factor of 580 at the "
                "coarsest sampling in the declared ladder. Defaulting it would make "
                "that a choice nobody recorded."
            )
        if self.curvature_interpolation not in CURVATURE_INTERPOLATIONS:
            raise ValueError(
                f"curvature_interpolation must be one of {CURVATURE_INTERPOLATIONS}, "
                f"not {self.curvature_interpolation!r}"
            )
        if self.upstream_status not in UPSTREAM_STATUS:
            raise ValueError(f"upstream_status must be one of {UPSTREAM_STATUS}")
        if self.upstream_status != "ok" and not self.upstream_note:
            raise ValueError(
                "an artefact that is not 'ok' must say what went wrong; a status with "
                "no reason cannot be acted on"
            )
        if not self.surface_digest or not self.path_digest:
            raise ValueError(
                "an artefact must carry a surface digest and a path digest. Without them "
                "'which surface was this?' has no answer later, and a sensitivity record "
                "is only as good as the path it was computed along."
            )
        if self.validity is not None and int(self.validity.samples) != size:
            raise ValueError(
                f"validity says {self.validity.samples} valid samples; the artefact "
                f"carries {size}. An artefact never holds an invalid sample."
            )
        object.__setattr__(self, "extra", dict(self.extra))

    # -- what it is --------------------------------------------------------

    @property
    def samples(self) -> int:
        return int(self.arclength.size)

    @property
    def length(self) -> float:
        return float(self.arclength[-1] - self.arclength[0])

    @property
    def uniform(self) -> bool:
        steps = np.diff(self.arclength)
        return bool(np.allclose(steps, steps[0], rtol=1e-12, atol=0.0))

    @property
    def max_step(self) -> float:
        return float(np.max(np.diff(self.arclength)))

    @property
    def carries_normal_curvature(self) -> bool:
        return self.normal_curvature_along is not None

    def surface_normal(self) -> Array:
        """``tangent x transverse``, which is the normal the triad implies.

        Derived rather than carried. The producer has already fixed the sign by
        choosing the transverse direction -- ``transverse = normal x tangent``
        -- so a separately supplied normal would be a fourth number in a triad
        with three degrees of freedom, and the case worth worrying about is the
        one where it disagrees.
        """
        return np.cross(self.tangent, self.transverse)

    def require_usable(self) -> PathGeometryArtefact:
        """The artefact, or an error if its producer already knows it is wrong."""
        if self.upstream_status == "failed":
            raise ValueError(
                f"artefact {self.identifier!r} reports upstream_status='failed': "
                f"{self.upstream_note}. Integrating it would produce a transfer map "
                "with no symptom of the failure its producer already detected."
            )
        return self

    def as_upstream(self) -> UpstreamArtefact:
        """How this artefact names itself in a record's provenance."""
        return UpstreamArtefact(
            kind="path-geometry",
            identifier=self.identifier,
            version=self.version,
            digest=self.path_digest,
            note=(
                f"schema={self.schema}; surface={self.surface_digest}; "
                f"sampling={self.sampling.policy}; "
                f"curvature_interpolation={self.curvature_interpolation}; "
                f"upstream_status={self.upstream_status}"
            ),
        )

    # -- curvature between the samples ------------------------------------

    def curvature_interpolant(self):
        """``K(s)`` under the interpolation policy the artefact declares.

        The policy is the artefact's, not this runtime's. That is the whole
        point: the values between samples are the producer's claim about its
        own surface, and a consumer that picked its own interpolant would be
        answering a different question with the same numbers.
        """
        grid = self.arclength
        values = self.gaussian_curvature
        if self.curvature_interpolation == "linear":
            def curvature_at(s: float) -> float:
                return float(np.interp(s, grid, values))

            return curvature_at
        if self.curvature_interpolation == "piecewise-constant":
            def curvature_at(s: float) -> float:
                index = int(np.clip(np.searchsorted(grid, s, side="right") - 1, 0, grid.size - 1))
                return float(values[index])

            return curvature_at
        slopes = _pchip_slopes(grid, values)

        def curvature_at(s: float) -> float:
            return float(_hermite_at(grid, values, slopes, s))

        return curvature_at

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "identifier": self.identifier,
            "version": self.version,
            "producer": self.producer,
            "samples": self.samples,
            "length": self.length,
            "uniform": self.uniform,
            "frame": self.frame,
            "coordinate_frame": self.coordinate_frame,
            "datum_frame": self.datum_frame,
            "path_type": self.path_type,
            "path_type_basis": self.path_type_basis,
            "units": self.units.to_dict(),
            "surface_digest": self.surface_digest,
            "path_digest": self.path_digest,
            "sampling": self.sampling.to_dict(),
            "curvature_interpolation": self.curvature_interpolation,
            "uncertainty": self.uncertainty.to_dict(),
            "validity": None if self.validity is None else self.validity.to_dict(),
            "upstream_status": self.upstream_status,
            "upstream_note": self.upstream_note,
            "arclength": self.arclength.tolist(),
            "position": self.position.tolist(),
            "tangent": self.tangent.tolist(),
            "transverse": self.transverse.tolist(),
            "gaussian_curvature": self.gaussian_curvature.tolist(),
            "extra": dict(self.extra),
        }
        for name in (
            "normal_curvature_along",
            "normal_curvature_transverse",
            "mean_curvature",
            "geodesic_curvature",
        ):
            value = getattr(self, name)
            payload[name] = None if value is None else np.asarray(value).tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PathGeometryArtefact:
        schema = str(payload.get("schema", ""))
        if schema != PATH_GEOMETRY_SCHEMA:
            raise ValueError(
                f"this reader understands {PATH_GEOMETRY_SCHEMA!r}, not {schema!r}"
            )
        optional = {
            name: (None if payload.get(name) is None else np.asarray(payload[name], dtype=float))
            for name in (
                "normal_curvature_along",
                "normal_curvature_transverse",
                "mean_curvature",
                "geodesic_curvature",
            )
        }
        return cls(
            identifier=str(payload["identifier"]),
            arclength=np.asarray(payload["arclength"], dtype=float),
            position=np.asarray(payload["position"], dtype=float),
            tangent=np.asarray(payload["tangent"], dtype=float),
            transverse=np.asarray(payload["transverse"], dtype=float),
            gaussian_curvature=np.asarray(payload["gaussian_curvature"], dtype=float),
            units=Units.from_dict(payload.get("units")),
            surface_digest=str(payload.get("surface_digest", "")),
            path_digest=str(payload.get("path_digest", "")),
            version=str(payload.get("version", "1")),
            producer=str(payload.get("producer", "")),
            frame=str(payload.get("frame", "transverse-to-gamma, parallel-transported")),
            coordinate_frame=str(payload.get("coordinate_frame", "producer-declared-ambient")),
            datum_frame=str(payload.get("datum_frame", "not-declared")),
            path_type=str(payload.get("path_type", "geodesic")),
            path_type_basis=str(payload.get("path_type_basis", "declared by the producer")),
            sampling=SamplingPolicy.from_dict(payload.get("sampling")),
            # No default here either. A serialised artefact that lost the field
            # must fail to reopen; supplying one would reintroduce exactly the
            # silent selection the constructor refuses.
            curvature_interpolation=(
                str(payload["curvature_interpolation"])
                if "curvature_interpolation" in payload
                else MISSING_INTERPOLATION
            ),
            uncertainty=GeometryUncertainty.from_dict(payload.get("uncertainty")),
            validity=ChartValidity.from_dict(payload.get("validity")),
            upstream_status=str(payload.get("upstream_status", "ok")),
            upstream_note=str(payload.get("upstream_note", "")),
            extra=dict(payload.get("extra", {})),
            **optional,
        )

    # -- the transformations a consumer is entitled to apply ---------------

    def converted_to(self, length_unit: str, scale: float) -> PathGeometryArtefact:
        """The same path in a different length unit.

        ``scale`` is how many new units there are in one old one: 1000 from
        metre to millimetre. Arc length and position scale with it, curvature
        with its inverse square, the normal curvatures with its inverse, and
        the frame vectors not at all -- which is the point of doing this here
        rather than leaving each consumer to remember the exponents.
        """
        factor = float(scale)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("a unit conversion scale must be finite and positive")
        curvatures = {
            name: (None if getattr(self, name) is None else getattr(self, name) / factor)
            for name in (
                "normal_curvature_along",
                "normal_curvature_transverse",
                "mean_curvature",
                "geodesic_curvature",
            )
        }
        sampling = replace(
            self.sampling,
            requested_spacing=(
                None
                if self.sampling.requested_spacing is None
                else self.sampling.requested_spacing * factor
            ),
        )
        uncertainty = self.uncertainty
        if uncertainty.declared:
            uncertainty = GeometryUncertainty(
                position=None if uncertainty.position is None else uncertainty.position * factor,
                normal=None if uncertainty.normal is None else uncertainty.normal * factor,
                curvature=(
                    None
                    if uncertainty.curvature is None
                    else uncertainty.curvature / factor**2
                ),
                basis=uncertainty.basis,
                note=uncertainty.note,
            )
        return replace(
            self,
            arclength=self.arclength * factor,
            position=self.position * factor,
            gaussian_curvature=self.gaussian_curvature / factor**2,
            units=Units(length=length_unit, angle=self.units.angle),
            sampling=sampling,
            uncertainty=uncertainty,
            **curvatures,
        )

    def with_reversed_transverse(self) -> PathGeometryArtefact:
        """The same path with the transverse direction, and so the normal, flipped.

        Gaussian curvature is intrinsic and does not move; the normal
        curvatures and the mean curvature are measured against the normal and
        all three change sign. The transfer map is unchanged -- it depends on
        ``K`` alone -- which is exactly why this is worth being able to do: a
        producer whose frame convention is the other one yields the same
        sensitivity, and a consumer's lateral displacement points the other way.
        """
        flipped = {
            name: (None if getattr(self, name) is None else -getattr(self, name))
            for name in (
                "normal_curvature_along",
                "normal_curvature_transverse",
                "mean_curvature",
                "geodesic_curvature",
            )
        }
        return replace(self, transverse=-self.transverse, **flipped)


# -- shape-preserving cubic interpolation ---------------------------------


def _pchip_slopes(grid: Array, values: Array) -> Array:
    """Fritsch-Carlson slopes: a cubic through the data that adds no extrema.

    A natural cubic spline through a curvature profile that is flat and then
    bends can overshoot into the wrong sign of ``K``, which turns a flat region
    of the surface into a focusing one and moves a conjugate point that is not
    there. Monotone interpolation costs the same and cannot do that.
    """
    steps = np.diff(grid)
    secants = np.diff(values) / steps
    slopes = np.zeros_like(values)
    if values.size == 2:
        slopes[:] = secants[0]
        return slopes
    inner_left, inner_right = secants[:-1], secants[1:]
    step_left, step_right = steps[:-1], steps[1:]
    same_sign = inner_left * inner_right > 0.0
    weight_left = 2.0 * step_right + step_left
    weight_right = step_right + 2.0 * step_left
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (weight_left + weight_right) / (
            weight_left / np.where(same_sign, inner_left, 1.0)
            + weight_right / np.where(same_sign, inner_right, 1.0)
        )
    slopes[1:-1] = np.where(same_sign, harmonic, 0.0)
    slopes[0] = _end_slope(steps[0], steps[1], secants[0], secants[1])
    slopes[-1] = _end_slope(steps[-1], steps[-2], secants[-1], secants[-2])
    return slopes


def _end_slope(h_near: float, h_far: float, s_near: float, s_far: float) -> float:
    """One-sided three-point slope, clipped so the end adds no extremum either."""
    slope = ((2.0 * h_near + h_far) * s_near - h_near * s_far) / (h_near + h_far)
    if slope * s_near <= 0.0:
        return 0.0
    if abs(slope) > 3.0 * abs(s_near):
        return 3.0 * s_near
    return float(slope)


def _hermite_at(grid: Array, values: Array, slopes: Array, s: float) -> float:
    """Cubic Hermite value at ``s``, held flat outside the grid.

    Extrapolating a curvature profile past the end of the path it was sampled
    on would be inventing surface, so the ends are held rather than continued.
    """
    if s <= grid[0]:
        return float(values[0])
    if s >= grid[-1]:
        return float(values[-1])
    index = int(np.searchsorted(grid, s, side="right") - 1)
    index = min(index, grid.size - 2)
    step = grid[index + 1] - grid[index]
    t = (s - grid[index]) / step
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return float(
        h00 * values[index]
        + h10 * step * slopes[index]
        + h01 * values[index + 1]
        + h11 * step * slopes[index + 1]
    )
