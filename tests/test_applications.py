# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np

from geodesic_testbed import (
    InspectionSpec,
    ManufacturingSpec,
    PathTolerance,
    assess_inspection,
    assess_manufacturing,
    constant_curvature_trace,
)


def test_flat_exact_pitch_has_no_gap_or_overlap_without_tolerance() -> None:
    trace = constant_curvature_trace(np.linspace(0.0, 1.0, 101), 0.0)
    result = assess_manufacturing(
        trace,
        ManufacturingSpec(course_width=0.1, initial_spacing=0.1),
    )
    assert result.max_gap == 0.0
    assert result.max_overlap == 0.0


def test_curvature_separates_gap_and_overlap_failure_modes() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    spec = ManufacturingSpec(course_width=0.1, initial_spacing=0.1)
    sphere = assess_manufacturing(constant_curvature_trace(grid, 1.0), spec)
    hyperbolic = assess_manufacturing(constant_curvature_trace(grid, -1.0), spec)
    assert sphere.max_overlap > 0.0
    assert sphere.max_gap == 0.0
    assert hyperbolic.max_gap > 0.0
    assert hyperbolic.max_overlap == 0.0


def test_starting_pose_tolerance_expands_manufacturing_bounds() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    trace = constant_curvature_trace(grid, 0.0)
    nominal = assess_manufacturing(
        trace,
        ManufacturingSpec(course_width=0.1, initial_spacing=0.1),
    )
    bounded = assess_manufacturing(
        trace,
        ManufacturingSpec(
            course_width=0.1,
            initial_spacing=0.1,
            path_tolerance=PathTolerance(lateral=0.001, heading=0.002),
        ),
    )
    assert bounded.max_gap > nominal.max_gap
    assert bounded.max_overlap > nominal.max_overlap


def test_inspection_reliability_combines_coverage_and_path_error() -> None:
    grid = np.linspace(0.0, 1.25, 126)
    tolerance = PathTolerance(lateral=0.001, heading=0.001)
    spec = InspectionSpec(
        swath_width=0.12,
        initial_track_spacing=0.10,
        max_cross_track_error=0.005,
        path_tolerance=tolerance,
    )
    sphere = assess_inspection(constant_curvature_trace(grid, 1.0), spec)
    hyperbolic = assess_inspection(constant_curvature_trace(grid, -1.0), spec)
    assert sphere.reliable
    assert sphere.max_coverage_gap == 0.0
    assert not hyperbolic.reliable
    assert hyperbolic.max_coverage_gap > 0.0
