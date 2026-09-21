"""Solve along a ``path-geometry-v1`` artefact, and produce one from a surface.

This is the adapter the boundary was designed around: a path computed
elsewhere arrives as a :class:`~geodesic_testbed.engine.path_artefact.
PathGeometryArtefact`, and a :class:`~geodesic_testbed.engine.record.
TransferRecord` leaves. Between them is the same integrator and the same
transfer map every other producer here uses. There is no second solver, and
there is no mesh: the artefact's curvature samples and its declared
interpolation policy are the whole of the input.

What the adapter adds over calling the solver directly is the part that is
easy to get wrong and impossible to detect afterwards -- that the record's
units, frame, digests, chart validity, path type and geometry are the
artefact's and not this runtime's defaults. A record built by hand from an
imported path and defaulted everywhere else would claim an analytic surface it
never saw.

:func:`artefact_from_envelope` runs the other way, turning a path this runtime
computed on a parametric surface into the artefact an upstream producer would
have sent. It exists so the round trip is testable: an artefact made from the
sphere has to reproduce the sphere's closed form through the adapter, which is
the anchor that keeps this path honest. It is not a mesh exporter and it does
not estimate curvature -- it reads what the surface already computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .contract import (
    ConvergenceEstimate,
    GeometryUncertainty,
    PathGeometry,
    Provenance,
    StartingCovariance,
    Units,
)
from .integrators import integrate_on_grid
from .path_artefact import (
    ARTEFACT_DOMAIN,
    PATH_GEOMETRY_SCHEMA,
    PathGeometryArtefact,
    SamplingPolicy,
)
from .record import Resolution, TransferRecord, ValidityEnvelope
from .transfer import TransferMap, transfer_from_trajectory, transfer_rhs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .envelope import PathEnvelope

Array = np.ndarray

#: ``Phi(0) = I``: ``a`` starts at one with zero rate, ``b`` at zero with unit
#: rate. This is the initial condition that makes ``Phi`` the transfer map
#: rather than one particular Jacobi field.
IDENTITY_STATE: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)

#: Modes a record from an imported artefact may be tagged with, and what each
#: one additionally requires of the artefact itself. The domain table in
#: ``observation.py`` says what this repository can compute *here*; this says
#: what a particular artefact supplied the evidence for.
MODE_REQUIRES_NORMAL_CURVATURE = ("ambient-euclidean-chord",)


def transfer_map_from_artefact(
    artefact: PathGeometryArtefact, *, method: str = "rk4"
) -> TransferMap:
    """``Phi(s)`` along the artefact's own grid, from its own curvature.

    One step per interval between the samples the producer chose. A finer
    internal grid would be a different curvature profile -- the interpolant
    evaluated where the producer never claimed anything -- and would make the
    record's arclength grid no longer the grid the answer was computed on.
    """
    artefact.require_usable()
    rhs = transfer_rhs(artefact.curvature_interpolant())
    trajectory = integrate_on_grid(
        rhs, np.asarray(IDENTITY_STATE, dtype=float), artefact.arclength, method=method
    )
    return transfer_from_trajectory(artefact.arclength, trajectory)


def transfer_record_from_artefact(
    artefact: PathGeometryArtefact,
    *,
    method: str = "rk4",
    observation_mode: str = "first-order-tangent-separation",
    covariance: StartingCovariance | None = None,
    convergence: ConvergenceEstimate | None = None,
    validity: ValidityEnvelope | None = None,
    note: str = "",
) -> TransferRecord:
    """The record a consumer gets from an imported path.

    The validity envelope is ``not-established`` and stays that way unless a
    caller supplies one, because establishing it means flowing neighbouring
    paths and the surface they would be flowed on stayed upstream. A producer
    that measured the envelope on its own side passes it in; nobody here can
    invent it.

    The domain is fixed at ``imported-path-artefact``. It is not a parametric
    surface -- there is no chart and no map to differentiate -- and it is not a
    declared curvature profile either, because the artefact carries positions
    and a frame. Calling it either of its neighbours would let a consumer ask
    for a quantity that neighbour supports and this one does not.
    """
    artefact.require_usable()
    if observation_mode in MODE_REQUIRES_NORMAL_CURVATURE and not artefact.carries_normal_curvature:
        raise ValueError(
            f"observation mode {observation_mode!r} needs the transverse normal curvature "
            "to turn an in-surface separation into a chord, and this artefact declares "
            "none. Tag the record 'first-order-tangent-separation', which is what was "
            "actually computed, or supply the curvatures upstream."
        )
    transfer = transfer_map_from_artefact(artefact, method=method)

    geometry: PathGeometry | None = None
    if artefact.carries_normal_curvature:
        geometry = PathGeometry(
            position=artefact.position,
            tangent=artefact.tangent,
            transverse=artefact.transverse,
            surface_normal=artefact.surface_normal(),
            normal_curvature_along=artefact.normal_curvature_along,
            normal_curvature_transverse=artefact.normal_curvature_transverse,
            coordinate_frame=artefact.coordinate_frame,
            datum_frame=artefact.datum_frame,
            geodesic_curvature=artefact.geodesic_curvature,
            uncertainty=artefact.uncertainty,
        )

    reason = note or (
        "computed from an imported path-geometry artefact; the curvature between "
        f"samples is the producer's {artefact.curvature_interpolation} interpolant"
    )
    if artefact.upstream_status != "ok":
        reason = f"{reason}. Upstream status {artefact.upstream_status!r}: {artefact.upstream_note}"
    provenance = Provenance(note=reason).with_upstream(artefact.as_upstream())

    resolution = Resolution(
        method=method,
        samples=artefact.samples,
        max_step=artefact.max_step,
        uniform=artefact.uniform,
        convergence=(
            convergence
            if convergence is not None
            else ConvergenceEstimate.not_established(
                "an imported artefact fixes its own grid, so halving the step would "
                "solve a different curvature profile rather than the same one more "
                "finely; refine upstream and compare two artefacts"
            )
        ),
    )
    return TransferRecord(
        arclength=artefact.arclength,
        gaussian_curvature=artefact.gaussian_curvature,
        a=transfer.a,
        a_rate=transfer.a_rate,
        b=transfer.b,
        b_rate=transfer.b_rate,
        frame=artefact.frame,
        units=artefact.units,
        source_digest=artefact.path_digest,
        resolution=resolution,
        validity=validity
        if validity is not None
        else ValidityEnvelope.not_established(
            "an imported artefact carries one path. Establishing where the linear "
            "map stops holding means flowing neighbouring paths and central"
            "-differencing them, and the surface those would be flowed on stayed "
            "upstream -- so the range is unknown here, which is a true statement "
            "and not a missing feature"
        ),
        observation_mode=observation_mode,
        domain=ARTEFACT_DOMAIN,
        covariance=covariance if covariance is not None else StartingCovariance.not_declared(),
        provenance=provenance,
        geometry=geometry,
        path_type=artefact.path_type,
        path_type_basis=artefact.path_type_basis,
        chart=artefact.validity,
    )


def artefact_from_envelope(
    envelope: PathEnvelope,
    *,
    identifier: str,
    surface_digest: str,
    path_digest: str,
    units: Units,
    curvature_interpolation: str,
    sampling: SamplingPolicy | None = None,
    uncertainty: GeometryUncertainty | None = None,
    producer: str = "geodesic-testbed/parametric-surface",
    extra: dict[str, Any] | None = None,
) -> PathGeometryArtefact:
    """Present an envelope's path as the artefact an upstream would send.

    ``curvature_interpolation`` has no default here for the same reason the
    artefact has none: this function stands in for an upstream producer, and a
    producer that did not state which curve fills ``K`` between its samples has
    not said enough. Defaulting it here would put the silent choice back one
    layer up.

    Everything else is read out of what the surface already computed. Nothing
    is estimated and nothing is smoothed: this is a change of container, so
    that the adapter can be anchored against the same closed forms the rest of
    the repository is anchored against. The mean curvature is formed as the
    half-sum of the two normal curvatures, which is Euler's theorem used
    forwards -- and the artefact checks it backwards on arrival, so a producer
    that got it wrong is caught at the boundary rather than believed.
    """
    geometry = envelope.path_geometry()
    chart = envelope.chart_record()
    mean = 0.5 * (geometry.normal_curvature_along + geometry.normal_curvature_transverse)
    return PathGeometryArtefact(
        identifier=identifier,
        arclength=envelope.arc_length,
        position=geometry.position,
        tangent=geometry.tangent,
        transverse=geometry.transverse,
        gaussian_curvature=envelope.curvature,
        units=units,
        surface_digest=surface_digest,
        path_digest=path_digest,
        schema=PATH_GEOMETRY_SCHEMA,
        producer=producer,
        coordinate_frame=geometry.coordinate_frame,
        datum_frame=geometry.datum_frame,
        sampling=sampling
        if sampling is not None
        else SamplingPolicy(
            policy="uniform-arclength" if _uniform(envelope.arc_length) else "nonuniform-arclength",
            note="the grid the surface path was integrated on",
        ),
        curvature_interpolation=curvature_interpolation,
        uncertainty=uncertainty if uncertainty is not None else geometry.uncertainty,
        validity=chart,
        normal_curvature_along=geometry.normal_curvature_along,
        normal_curvature_transverse=geometry.normal_curvature_transverse,
        mean_curvature=mean,
        geodesic_curvature=geometry.geodesic_curvature,
        extra=dict(extra or {}),
    )


def _uniform(grid: Array) -> bool:
    steps = np.diff(np.asarray(grid, dtype=float))
    return bool(np.allclose(steps, steps[0], rtol=1e-12, atol=0.0))
