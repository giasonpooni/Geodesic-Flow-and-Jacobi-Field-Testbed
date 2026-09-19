"""The stage-two report: every check passes, and it says what it is anchored to."""

from __future__ import annotations

import json
import math

import pytest

from geojac.experiment_surfaces import REPORT_SCHEMA, default_cases


def test_every_declared_check_passes(surface_report: dict) -> None:
    failed = [check["id"] for check in surface_report["checks"] if not check["passed"]]
    assert failed == []
    assert surface_report["summary"]["all_passed"] is True
    assert surface_report["summary"]["n_checks"] >= 40


def test_it_declares_what_it_depends_on(surface_report: dict) -> None:
    assert surface_report["schema"] == REPORT_SCHEMA
    assert surface_report["depends_on"] == "geodesic-jacobi-report-v1"
    assert "anchored" in surface_report["claim_scope"]
    assert [case["case"] for case in surface_report["cases"]] == [
        case.key for case in default_cases()
    ]


def test_report_is_strict_json(surface_report: dict) -> None:
    text = json.dumps(surface_report, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == surface_report


def test_the_closed_form_cases_are_recovered(surface_report: dict) -> None:
    rows = {row["case"]: row for row in surface_report["results"]["anchored_to_closed_forms"]}
    assert set(rows) == {"plate", "rolled-sheet", "spherical-cap", "pseudosphere"}
    for row in rows.values():
        assert row["max_jacobi_error"] <= row["tolerance"]
        assert row["max_speed_drift"] < 1e-9


def test_self_convergence_is_order_four_where_there_is_anything_to_converge(
    surface_report: dict,
) -> None:
    for row in surface_report["results"]["self_convergence"]:
        if row["regime"] == "exact-to-roundoff":
            assert row["case"] in {"plate", "rolled-sheet"}
            continue
        assert row["fitted_order_jacobi"] == pytest.approx(4.0, abs=0.2)
        assert row["fitted_order_position"] == pytest.approx(4.0, abs=0.2)


def test_the_chord_excess_is_zero_exactly_where_theory_says_so(
    surface_report: dict,
) -> None:
    rows = {row["case"]: row for row in surface_report["results"]["two_routes"]}
    for case in ("plate", "spherical-cap"):
        assert rows[case]["fitted_coefficient"] == pytest.approx(1.0 / 6.0, rel=1e-2)
        assert rows[case]["chord_closed_form_max_relative_error"] < 1e-9
    for case in ("rolled-sheet", "pseudosphere", "saddle", "torus"):
        assert rows[case]["chord_excess_coefficient"] > 0.0
        assert rows[case]["fitted_exponent"] == pytest.approx(2.0, abs=0.05)


def test_the_envelopes_flag_conditioning(surface_report: dict) -> None:
    rows = {row["case"]: row for row in surface_report["results"]["envelopes"]}
    assert rows["spherical-cap"]["focus_points"][0] == pytest.approx(math.pi, abs=1e-8)
    assert rows["spherical-cap"]["well_conditioned"] is False
    for case in ("plate", "rolled-sheet", "pseudosphere", "saddle", "torus"):
        assert rows[case]["well_conditioned"] is True
    assert rows["saddle"]["curvature_along_path"]["varies"] is True
    assert rows["pseudosphere"]["curvature_along_path"]["varies"] is False


def test_the_heading_scan_produces_a_ranked_decision(surface_report: dict) -> None:
    scan = surface_report["results"]["heading_scan"]
    ranked = [row["max_abs_jacobi_field"] for row in scan["headings"]]
    assert ranked == sorted(ranked)
    assert scan["most_tolerant"] is not None
    assert scan["sensitivity_ratio"] > 1.1
    assert scan["sensitivity_ratio"] == pytest.approx(
        scan["least_tolerant"]["max_abs_jacobi_field"]
        / scan["most_tolerant"]["max_abs_jacobi_field"]
    )
