"""Deterministic reference report for the first tested application slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .applications import (
    InspectionSpec,
    ManufacturingSpec,
    assess_inspection,
    assess_manufacturing,
)
from .engine.canonical import jsonable
from .jacobi import constant_curvature_trace, finite_angular_separation, integrate_jacobi
from .tolerances import PathTolerance


def build_reference_report() -> dict[str, Any]:
    """Build the reproducible plane/sphere/hyperbolic reference report."""
    grid = np.linspace(0.0, 1.25, 126)
    validation: list[dict[str, Any]] = []
    for curvature, name in ((0.0, "plane"), (1.0, "sphere"), (-1.0, "hyperbolic-plane")):
        exact = constant_curvature_trace(grid, curvature)
        numerical = integrate_jacobi(grid, curvature)
        delta = 1.0e-3
        finite = finite_angular_separation(grid, curvature, delta)
        first_order = np.abs(exact.angle_basis) * delta
        validation.append(
            {
                "model": name,
                "curvature": curvature,
                "max_position_basis_error": float(
                    np.max(np.abs(numerical.position_basis - exact.position_basis))
                ),
                "max_angle_basis_error": float(
                    np.max(np.abs(numerical.angle_basis - exact.angle_basis))
                ),
                "max_finite_vs_first_order_error": float(
                    np.max(np.abs(finite - first_order))
                ),
            }
        )

    tolerance = PathTolerance(lateral=0.001, heading=0.001)
    manufacturing_spec = ManufacturingSpec(
        course_width=0.100,
        initial_spacing=0.100,
        path_tolerance=tolerance,
    )
    inspection_spec = InspectionSpec(
        swath_width=0.120,
        initial_track_spacing=0.100,
        max_cross_track_error=0.005,
        path_tolerance=tolerance,
    )
    applications: dict[str, Any] = {}
    for curvature, name in ((1.0, "positive-curvature"), (-1.0, "negative-curvature")):
        trace = constant_curvature_trace(grid, curvature)
        applications[name] = {
            "manufacturing": assess_manufacturing(trace, manufacturing_spec).summary(),
            "inspection": assess_inspection(trace, inspection_spec).summary(),
        }

    return {
        "schema": "geodesic-sensitivity-report-v1",
        "revision": 2,
        "revision_note": (
            "Same schema. Revision 2 evaluates the finite ray separation as "
            "sn_K(d/2) = sn_K(s) sin(delta/2) instead of through arccos/arccosh, "
            "which keeps full precision at small delta; the two "
            "max_finite_vs_first_order_error values move by about 2 per cent and "
            "are now free of the reference's own rounding."
        ),
        "claim_scope": "first-order-computational-analysis",
        "units": {
            "arclength": "declared length unit",
            "curvature": "inverse declared length unit squared",
            "heading": "radian",
        },
        "reference_grid": {
            "start": float(grid[0]),
            "stop": float(grid[-1]),
            "samples": int(grid.size),
        },
        "numerical_validation": validation,
        "application_examples": applications,
        "limitations": [
            "Jacobi fields are first-order variations, not exact finite path distances.",
            "Gap, overlap, and coverage outputs are deterministic bounds, not probabilities.",
            "Machine tracking, material deformation, and sensor detection require separate models.",
            "A geodesic segment is not thereby established as globally shortest.",
        ],
    }


def format_reference_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Constant-curvature reference report",
        "",
        f"Schema: `{report['schema']}`.",
        "",
        "Claim scope: **first-order computational analysis**.",
        "",
        "## Numerical validation",
        "",
        "| model | K | max error in a(s) | max error in b(s) | finite/first-order gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["numerical_validation"]:
        lines.append(
            f"| {row['model']} | {row['curvature']:.0f} | "
            f"{row['max_position_basis_error']:.3e} | "
            f"{row['max_angle_basis_error']:.3e} | "
            f"{row['max_finite_vs_first_order_error']:.3e} |"
        )
    lines.extend(["", "## Application examples", ""])
    for name, values in report["application_examples"].items():
        manufacturing = values["manufacturing"]
        inspection = values["inspection"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Maximum possible manufacturing gap: `{manufacturing['max_gap']:.6g}`.",
                "- Maximum possible manufacturing overlap: "
                f"`{manufacturing['max_overlap']:.6g}`.",
                "- Inspection coverage reliable in the declared example: "
                f"`{inspection['reliable']}`.",
                "- Maximum possible inspection coverage gap: "
                f"`{inspection['max_coverage_gap']:.6g}`.",
                "",
            ]
        )
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_reference_report(output_directory: Path) -> tuple[Path, Path]:
    report = build_reference_report()
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "reference-report.json"
    markdown_path = output_directory / "reference-report.md"
    # Canonicalised for the same reason the experiment reports are: this file
    # is committed and compared, and full repr precision makes two identical
    # computations differ on the last digit of a reduction.
    json_path.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(format_reference_markdown(report), encoding="utf-8")
    return json_path, markdown_path
