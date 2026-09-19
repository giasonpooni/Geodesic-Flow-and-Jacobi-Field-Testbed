"""Application contracts for manufacturing courses and inspection tracks.

These outputs are deterministic first-order bounds. They are not defect
probabilities and do not include machine tracking, material deformation, or
sensor-detection models unless those effects are included in the supplied
starting tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .jacobi import JacobiTrace
from .tolerances import PathTolerance

Array = NDArray[np.float64]


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class ManufacturingSpec:
    course_width: float
    initial_spacing: float
    initial_heading_delta: float = 0.0
    path_tolerance: PathTolerance = PathTolerance()
    relative_tolerance_factor: float = 2.0

    def __post_init__(self) -> None:
        _positive(self.course_width, "course_width")
        _positive(self.initial_spacing, "initial_spacing")
        _finite(self.initial_heading_delta, "initial_heading_delta")
        if not np.isfinite(self.relative_tolerance_factor) or self.relative_tolerance_factor < 0.0:
            raise ValueError("relative_tolerance_factor must be finite and nonnegative")


@dataclass(frozen=True)
class ManufacturingAssessment:
    arclength: Array
    nominal_spacing: Array
    minimum_spacing: Array
    maximum_spacing: Array
    maximum_possible_gap: Array
    maximum_possible_overlap: Array

    @property
    def max_gap(self) -> float:
        return float(np.max(self.maximum_possible_gap))

    @property
    def max_overlap(self) -> float:
        return float(np.max(self.maximum_possible_overlap))

    @property
    def worst_gap_location(self) -> float:
        return float(self.arclength[int(np.argmax(self.maximum_possible_gap))])

    @property
    def worst_overlap_location(self) -> float:
        return float(self.arclength[int(np.argmax(self.maximum_possible_overlap))])

    def summary(self) -> dict[str, float | str]:
        return {
            "claim_scope": "first-order-deterministic-bound",
            "max_gap": self.max_gap,
            "worst_gap_arclength": self.worst_gap_location,
            "max_overlap": self.max_overlap,
            "worst_overlap_arclength": self.worst_overlap_location,
        }


def assess_manufacturing(trace: JacobiTrace, spec: ManufacturingSpec) -> ManufacturingAssessment:
    """Bound gap and overlap between two neighbouring manufactured courses."""
    nominal = np.abs(trace.separation(spec.initial_spacing, spec.initial_heading_delta))
    relative_error = (
        spec.relative_tolerance_factor * spec.path_tolerance.envelope(trace)
    )
    minimum = np.maximum(0.0, nominal - relative_error)
    maximum = nominal + relative_error
    gap = np.maximum(0.0, maximum - spec.course_width)
    overlap = np.maximum(0.0, spec.course_width - minimum)
    return ManufacturingAssessment(
        arclength=trace.arclength,
        nominal_spacing=nominal,
        minimum_spacing=minimum,
        maximum_spacing=maximum,
        maximum_possible_gap=gap,
        maximum_possible_overlap=overlap,
    )


@dataclass(frozen=True)
class InspectionSpec:
    swath_width: float
    initial_track_spacing: float
    max_cross_track_error: float
    initial_heading_delta: float = 0.0
    path_tolerance: PathTolerance = PathTolerance()
    relative_tolerance_factor: float = 2.0

    def __post_init__(self) -> None:
        _positive(self.swath_width, "swath_width")
        _positive(self.initial_track_spacing, "initial_track_spacing")
        _positive(self.max_cross_track_error, "max_cross_track_error")
        _finite(self.initial_heading_delta, "initial_heading_delta")
        if not np.isfinite(self.relative_tolerance_factor) or self.relative_tolerance_factor < 0.0:
            raise ValueError("relative_tolerance_factor must be finite and nonnegative")


@dataclass(frozen=True)
class InspectionAssessment:
    arclength: Array
    nominal_spacing: Array
    maximum_spacing: Array
    per_path_uncertainty: Array
    maximum_possible_coverage_gap: Array
    coverage_margin: Array
    reliable: bool

    @property
    def max_coverage_gap(self) -> float:
        return float(np.max(self.maximum_possible_coverage_gap))

    @property
    def worst_cross_track_error(self) -> float:
        return float(np.max(self.per_path_uncertainty))

    def summary(self) -> dict[str, float | bool | str]:
        return {
            "claim_scope": "first-order-deterministic-bound",
            "reliable": self.reliable,
            "max_coverage_gap": self.max_coverage_gap,
            "worst_cross_track_error": self.worst_cross_track_error,
            "minimum_coverage_margin": float(np.min(self.coverage_margin)),
        }


def assess_inspection(trace: JacobiTrace, spec: InspectionSpec) -> InspectionAssessment:
    """Assess track coverage and cross-track reliability under pose bounds."""
    nominal = np.abs(trace.separation(spec.initial_track_spacing, spec.initial_heading_delta))
    path_uncertainty = spec.path_tolerance.envelope(trace)
    maximum_spacing = nominal + spec.relative_tolerance_factor * path_uncertainty
    gap = np.maximum(0.0, maximum_spacing - spec.swath_width)
    margin = spec.swath_width - maximum_spacing
    reliable = bool(
        np.all(gap == 0.0)
        and np.max(path_uncertainty) <= spec.max_cross_track_error
    )
    return InspectionAssessment(
        arclength=trace.arclength,
        nominal_spacing=nominal,
        maximum_spacing=maximum_spacing,
        per_path_uncertainty=path_uncertainty,
        maximum_possible_coverage_gap=gap,
        coverage_margin=margin,
        reliable=reliable,
    )
