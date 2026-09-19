"""The report as an artefact: every check passes, and it is really JSON."""

from __future__ import annotations

import json
import math

import pytest

from geojac.experiment import (
    REPORT_SCHEMA,
    ExperimentConfig,
    content_hash,
    run_experiment,
    write_report,
)
from geojac.spaceforms import SpaceForm

EXPECTED_ORDERS = {"euler": 1, "midpoint": 2, "rk4": 4}


def test_every_declared_check_passes(report: dict) -> None:
    failed = [check["id"] for check in report["checks"] if not check["passed"]]
    assert failed == []
    assert report["summary"]["all_passed"] is True
    assert report["summary"]["n_checks"] == len(report["checks"]) > 50


def test_schema_and_scope_are_stated(report: dict) -> None:
    assert report["schema"] == REPORT_SCHEMA
    assert report["claim_scope"] == "numerical-verification-against-closed-form-solutions"
    assert sorted(report["curvatures"]) == [-1.0, 0.0, 1.0]


def test_report_is_strict_json_with_no_nan(report: dict) -> None:
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text) == report
    assert "NaN" not in text and "Infinity" not in text


def test_content_hash_is_reproducible(report: dict) -> None:
    again = run_experiment(ExperimentConfig())
    assert again["content_hash"] == report["content_hash"]
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_report_round_trips_through_a_file(report: dict, tmp_path) -> None:
    path = tmp_path / "nested" / "report-v1.json"
    write_report(report, path)
    assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("key", ["jacobi_ode_convergence", "geodesic_flow_convergence"])
def test_measured_orders_match_the_formal_ones(report: dict, key: str) -> None:
    for row in report["results"][key]:
        if row["curvature"] == 0.0:
            assert row["regime"] == "exact-to-roundoff"
            assert row["max_error_over_levels"] < 1e-13
            continue
        assert row["fitted_order"] == pytest.approx(
            EXPECTED_ORDERS[row["integrator"]], abs=0.15
        )
        assert row["fit"]["r_squared"] > 0.999
        assert row["fit"]["n_used"] >= 5


def test_first_order_error_is_quadratic_with_the_coefficient_theory_predicts(
    report: dict,
) -> None:
    for row in report["results"]["first_order_validity"]:
        form = SpaceForm(row["curvature"])
        theory = float(form.relative_deviation_coefficient(row["arc_length"]))
        assert row["fitted_exponent"] == pytest.approx(2.0, abs=0.05)
        assert row["fitted_coefficient"] == pytest.approx(theory, rel=1e-2)
        assert row["theory_coefficient"] == pytest.approx(theory, rel=1e-12)


def test_the_conjugate_point_is_found_and_costs_what_it_should(report: dict) -> None:
    conjugate = report["results"]["conjugate_point"]
    assert conjugate["jacobi_first_zero"] == pytest.approx(math.pi, abs=1e-9)
    assert conjugate["max_distance_error"] < 1e-9
    before, at, after = (
        conjugate["probes"][0],
        conjugate["probes"][1],
        conjugate["probes"][2],
    )
    assert before["minimising"] and at["minimising"]
    assert not after["minimising"]
    assert after["length_excess"] == pytest.approx(math.pi, abs=1e-8)


def test_sensitivity_readout_is_internally_consistent(report: dict) -> None:
    for row in report["results"]["path_sensitivity"]:
        form = SpaceForm(row["curvature"])
        for entry in row["by_arc_length"]:
            for budget in entry["tolerance_budget"]:
                separation = form.first_order_separation(
                    entry["arc_length"], budget["max_initial_angle"]
                )
                assert float(separation) == pytest.approx(
                    budget["transverse_tolerance"], rel=1e-12
                )
                assert budget["max_initial_angle_degrees"] == pytest.approx(
                    math.degrees(budget["max_initial_angle"]), rel=1e-12
                )
        assert row["bench_predictions"]["max_flow_vs_closed_form_relative_error"] < 1e-10


def test_the_saddle_amplifies_and_the_sphere_damps(report: dict) -> None:
    amplification = {
        row["curvature_label"]: {
            entry["arc_length"]: entry["amplification_vs_flat"]
            for entry in row["by_arc_length"]
        }
        for row in report["results"]["path_sensitivity"]
    }
    assert amplification["K=0"][2.0] == pytest.approx(1.0)
    assert amplification["K=+1"][2.0] < 0.6
    assert amplification["K=-1"][2.0] > 1.8
