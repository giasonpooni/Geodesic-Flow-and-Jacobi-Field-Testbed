"""Curved-surface geodesic sensitivity: transfer maps and tolerance contracts.

This is a computational substrate that an instrument may consume, not a module
inside one. The whole of what it offers outward is the transfer record;
:mod:`geodesic_testbed.boundary` is that contract on its own, and
``docs/BOUNDARY.md`` says what stays on each side of it.
"""

from .applications import (
    InspectionAssessment,
    InspectionSpec,
    ManufacturingAssessment,
    ManufacturingSpec,
    assess_inspection,
    assess_manufacturing,
)
from .boundary import read_record, write_record
from .engine.contract import (
    BOUNDARY_CONTRACT,
    DEFAULT_FRAME,
    RUNTIME_VERSION,
    CalibrationBinding,
    Frame,
    Provenance,
    StartingCovariance,
    UpstreamArtefact,
)
from .engine.measurement import (
    MEASUREMENT_SCHEMA,
    MeasurementRecord,
    Perturbation,
    Uncertainty,
    compare,
    prediction_binding,
)
from .engine.observation_model import (
    FilteredPrediction,
    ObservationModel,
    TemporalFilter,
    apply_filter,
    filtered_noise_covariance,
    operator_digest,
)
from .engine.record import (
    RECORD_SCHEMA,
    SUPPORTED_RECORD_SCHEMAS,
    FirstOrderValidity,
    Resolution,
    SupportsTransferRecord,
    TransferRecord,
    Units,
    to_transfer_record,
)
from .engine.routing import (
    ConstraintMargin,
    CoverageSpec,
    RouteAssessment,
    RouteConstraints,
    assess_route,
    rank_routes,
)
from .engine.surfaces import Chart
from .engine.tracking import AcquisitionSpec, TrackingOutcome, evaluate_tracking
from .engine.transfer import FocusEvent, TransferMap
from .jacobi import (
    JacobiTrace,
    constant_curvature_trace,
    finite_angular_separation,
    integrate_jacobi,
)
from .tolerances import PathTolerance

__all__ = [
    "BOUNDARY_CONTRACT",
    "DEFAULT_FRAME",
    "MEASUREMENT_SCHEMA",
    "RECORD_SCHEMA",
    "RUNTIME_VERSION",
    "SUPPORTED_RECORD_SCHEMAS",
    "AcquisitionSpec",
    "CalibrationBinding",
    "Frame",
    "Provenance",
    "StartingCovariance",
    "UpstreamArtefact",
    "prediction_binding",
    "read_record",
    "write_record",
    "Chart",
    "ConstraintMargin",
    "CoverageSpec",
    "FirstOrderValidity",
    "FocusEvent",
    "MeasurementRecord",
    "ObservationModel",
    "TemporalFilter",
    "TrackingOutcome",
    "Perturbation",
    "Resolution",
    "RouteAssessment",
    "RouteConstraints",
    "SupportsTransferRecord",
    "TransferMap",
    "TransferRecord",
    "Uncertainty",
    "Units",
    "assess_route",
    "compare",
    "evaluate_tracking",
    "FilteredPrediction",
    "apply_filter",
    "filtered_noise_covariance",
    "operator_digest",
    "rank_routes",
    "to_transfer_record",
    "InspectionAssessment",
    "InspectionSpec",
    "JacobiTrace",
    "ManufacturingAssessment",
    "ManufacturingSpec",
    "PathTolerance",
    "assess_inspection",
    "assess_manufacturing",
    "constant_curvature_trace",
    "finite_angular_separation",
    "integrate_jacobi",
]

__version__ = RUNTIME_VERSION
