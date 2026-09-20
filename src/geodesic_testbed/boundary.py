"""The boundary this runtime presents, and the one-way rule that keeps it one.

This repository is a parallel computational substrate, not a module inside an
instrument workbench. It answers one question -- given a surface and a path,
how does a starting-pose error propagate -- and it answers it from geometry
alone. Calibration, sensing, filtering, uncertainty budgets, physical trials,
observability and operational decisions are a different kind of work, they
belong to a different system, and the two are adjacent::

    geometry / path artefact
            |
            v
    geodesic sensitivity runtime            <- this repository
            |
            v
    transfer / path-sensitivity record      <- the shared contract
            |
            v
    instrument calibration + measurement tooling

What is shared is the record and nothing else. Importing this module gets the
whole of it: the vocabulary from :mod:`geodesic_testbed.engine.contract`, the
record itself, and the two functions that put one on disk and read it back.

What is *not* shared, in either direction:

solver internals
    integrators, charts, Christoffel symbols, the eight-component system, step
    control. A consumer that depended on any of these would be pinned to this
    runtime's numerics; what it is entitled to is ``Phi`` sampled on a declared
    grid, with the resolution that produced it stated.
mesh processing
    triangulated surfaces, discrete curvature estimators, mesh path
    convergence. They belong upstream, in the Intrinsic Surface Geodesics
    Testbed, and arrive here as a versioned path artefact named in
    :class:`~geodesic_testbed.engine.contract.UpstreamArtefact`.
filters
    a filter is part of an observation instrument, not of a solver. Smoothing
    inside ``engine/flows.py`` or ``engine/transfer.py`` would tune the model
    to the data through the model's own machinery. A filter changes the
    observation model, and that is where it is declared.
hardware behaviour
    servo error, material mechanics, tow compaction, weld-pool behaviour,
    probability of detection. Nothing here models any of it, and the record
    carries calibration identifiers precisely so that it does not have to.
route policy
    what counts as resolved, what margin is acceptable, which route is chosen.
    The runtime reports amplification, resolvability and clearance; a threshold
    that decides comes from an instrument protocol, never from arithmetic here.

The rule has a direction, and the direction is checkable. :data:`SUBSTRATE`
names the modules that compute the answer; :data:`INSTRUMENT_FACING` names the
modules that exist to meet an instrument on the other side of the record. The
first may not import the second. ``tests/test_boundary.py`` reads the imports
out of the source and enforces it, so the separation is a property of the code
rather than an intention in a document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine.contract import (
    ARTEFACT_KINDS,
    BOUNDARY_CONTRACT,
    COVARIANCE_BASES,
    DEFAULT_FRAME,
    FRAMES,
    GEOMETRY_UNCERTAINTY_BASES,
    PATH_TYPES,
    PRODUCER,
    RUNTIME_VERSION,
    CalibrationBinding,
    ChartValidity,
    ConvergenceEstimate,
    Frame,
    GeometryUncertainty,
    PathGeometry,
    Provenance,
    StartingCovariance,
    Units,
    UpstreamArtefact,
    frame,
    frame_catalogue,
    validated_covariance,
)
from .engine.observation import DOMAINS, MODES
from .engine.observation import catalogue as observation_catalogue
from .engine.observation import mode as observation_mode
from .engine.record import (
    RECORD_SCHEMA,
    SUPPORTED_RECORD_SCHEMAS,
    FirstOrderValidity,
    Resolution,
    SupportsTransferRecord,
    TransferRecord,
    digest,
    to_transfer_record,
)

#: The modules that compute the answer. Geometry in, transfer map out; between
#: them nothing that knows an instrument exists.
SUBSTRATE: tuple[str, ...] = (
    "geodesic_testbed.engine.spaceforms",
    "geodesic_testbed.engine.surfaces",
    "geodesic_testbed.engine.integrators",
    "geodesic_testbed.engine.flows",
    "geodesic_testbed.engine.transfer",
    "geodesic_testbed.engine.envelope",
    "geodesic_testbed.engine.analysis",
    "geodesic_testbed.jacobi",
)

#: The contract itself: the vocabulary, the record, and the canonical form an
#: artefact is written in. It sits between the two sides and may import
#: neither.
CONTRACT: tuple[str, ...] = (
    "geodesic_testbed.engine.contract",
    "geodesic_testbed.engine.canonical",
    "geodesic_testbed.engine.observation",
    "geodesic_testbed.engine.record",
)

#: The modules that exist to meet an instrument. They consume records; nothing
#: in :data:`SUBSTRATE` consumes them.
INSTRUMENT_FACING: tuple[str, ...] = (
    "geodesic_testbed.engine.observation_model",
    "geodesic_testbed.engine.measurement",
    "geodesic_testbed.engine.tracking",
    "geodesic_testbed.engine.routing",
)

#: The application contracts: they take a record and answer a manufacturing or
#: inspection question from it. They speak the record and not the instrument,
#: so they may not import :data:`INSTRUMENT_FACING` either -- a tolerance
#: assessment that reached for an observation model would be deciding with a
#: threshold that belongs to a protocol.
CONSUMERS: tuple[str, ...] = (
    "geodesic_testbed.applications",
    "geodesic_testbed.tolerances",
    "geodesic_testbed.reports",
)

#: The harness: the experiment stages, the figures, the entry points, and this
#: module. It sits above everything and may import anything, which is exactly
#: why it is named -- a layer that is allowed to reach everywhere has to be
#: listed, or every module could quietly claim to be it.
HARNESS: tuple[str, ...] = (
    "geodesic_testbed.boundary",
    "geodesic_testbed.engine.experiment",
    "geodesic_testbed.engine.experiment_surfaces",
    "geodesic_testbed.engine.figure",
    "geodesic_testbed.engine.figure_surfaces",
    "geodesic_testbed.engine.cli",
)

#: Every layer, in dependency order. The completeness of this list is itself a
#: declared check: a module in none of them would be a module the one-way rule
#: does not reach.
LAYERS: dict[str, tuple[str, ...]] = {
    "contract": CONTRACT,
    "substrate": SUBSTRATE,
    "instrument-facing": INSTRUMENT_FACING,
    "consumers": CONSUMERS,
    "harness": HARNESS,
}


def write_record(record: TransferRecord, path: str | Path, *, indent: int = 2) -> Path:
    """Write a record as JSON, samples included.

    Samples are always included: a payload without them is a summary, and a
    summary written to the path a consumer expects a record at is the failure
    this function exists to make impossible.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record.to_dict(include_samples=True), indent=indent) + "\n",
        encoding="utf-8",
    )
    return destination


def read_record(path: str | Path) -> TransferRecord:
    """Read a record written by :func:`write_record`, refusing an unknown schema."""
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return TransferRecord.from_dict(payload)


__all__ = [
    "ARTEFACT_KINDS",
    "BOUNDARY_CONTRACT",
    "GEOMETRY_UNCERTAINTY_BASES",
    "PATH_TYPES",
    "ChartValidity",
    "ConvergenceEstimate",
    "GeometryUncertainty",
    "PathGeometry",
    "validated_covariance",
    "CONSUMERS",
    "CONTRACT",
    "HARNESS",
    "LAYERS",
    "COVARIANCE_BASES",
    "DEFAULT_FRAME",
    "DOMAINS",
    "FRAMES",
    "INSTRUMENT_FACING",
    "MODES",
    "PRODUCER",
    "RECORD_SCHEMA",
    "RUNTIME_VERSION",
    "SUBSTRATE",
    "SUPPORTED_RECORD_SCHEMAS",
    "CalibrationBinding",
    "FirstOrderValidity",
    "Frame",
    "Provenance",
    "Resolution",
    "StartingCovariance",
    "SupportsTransferRecord",
    "TransferRecord",
    "Units",
    "UpstreamArtefact",
    "digest",
    "frame",
    "frame_catalogue",
    "observation_catalogue",
    "observation_mode",
    "read_record",
    "to_transfer_record",
    "write_record",
]
