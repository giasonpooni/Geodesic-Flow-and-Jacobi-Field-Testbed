"""The stage-two report: every check passes, and it says what it is anchored to."""

from __future__ import annotations

import json
import math

import pytest

from geodesic_testbed.engine.experiment import REPORT_SCHEMA as STAGE_ONE_SCHEMA
from geodesic_testbed.engine.experiment_surfaces import (
    REPORT_SCHEMA,
    SUPERSEDES,
    default_cases,
)

#: Runs a full experiment stage or a perturbation sweep. See the ``numerical``
#: marker in pyproject.toml: CI runs this file once, on one interpreter, rather
#: than once per version of an interpreter that cannot change the answer.
pytestmark = pytest.mark.numerical


def test_every_declared_check_passes(surface_report: dict) -> None:
    failed = [check["id"] for check in surface_report["checks"] if not check["passed"]]
    assert failed == []
    assert surface_report["summary"]["all_passed"] is True
    assert surface_report["summary"]["n_checks"] >= 40


def test_it_declares_what_it_depends_on(surface_report: dict) -> None:
    assert surface_report["schema"] == REPORT_SCHEMA
    assert surface_report["supersedes"] == SUPERSEDES
    # Pinned to the constant, not to a literal, so the two stages cannot drift
    # apart the next time either schema is versioned.
    assert surface_report["depends_on"] == STAGE_ONE_SCHEMA
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
    ranked = [row["max_forward_amplification"] for row in scan["headings"]]
    assert ranked == sorted(ranked)
    assert scan["ranking_scalar"] == "minimum-forward-angular-error-amplification"
    assert scan["amplification_ratio"] > 1.1
    assert scan["amplification_ratio"] == pytest.approx(
        scan["highest_amplification"]["max_forward_amplification"]
        / scan["lowest_amplification"]["max_forward_amplification"]
    )


def test_the_scan_does_not_call_low_amplification_robustness(surface_report: dict) -> None:
    """The ranking scalar is reported; the decision comes from declared limits."""
    scan = surface_report["results"]["heading_scan"]
    assert "robust" not in scan["ranking_scalar"]
    assert scan["lowest_amplification"]["passes_a_focus"] is True

    declared = scan["route_selection"]
    tracked = scan["route_selection_with_tracking"]
    assert declared["recommended"] is not None
    assert tracked["recommended"] is not None
    # The instrument's schedule, not a threshold on |b|, moves the answer.
    assert declared["recommended"]["label"] != tracked["recommended"]["label"]
    assert tracked["n_feasible"] <= declared["n_feasible"]
    assert len(tracked["outcomes"]) >= 2, "the schedule must discriminate"
    assert tracked["recommended"]["tracking"]["outcome"] == "TRACKED"


def test_resolvability_does_not_move_when_the_part_is_rescaled(
    surface_report: dict,
) -> None:
    invariance = surface_report["results"]["heading_scan"][
        "route_selection_with_tracking"
    ]["scale_invariance"]
    assert invariance["max_relative_difference"] < 1e-9


def test_every_path_stays_inside_its_declared_chart(surface_report: dict) -> None:
    for row in surface_report["results"]["envelopes"]:
        assert row["chart"]["valid"] is True
        assert row["chart"]["min_conditioning"] > 0.0
        assert row["chart"]["declared_domain"]["reference_length"] > 0.0


def test_the_transfer_map_keeps_its_invariant_everywhere(surface_report: dict) -> None:
    for row in surface_report["results"]["envelopes"]:
        assert row["max_wronskian_drift"] < 1e-9
    for row in surface_report["results"]["anchored_to_closed_forms"]:
        assert row["max_lateral_error"] <= row["tolerance"]


def test_the_lateral_column_is_validated_by_moving_the_start(surface_report: dict) -> None:
    rows = {row["case"]: row for row in surface_report["results"]["lateral_route"]}
    assert rows["plate"]["regime"] == "exact-to-roundoff"
    for case in ("rolled-sheet", "spherical-cap", "pseudosphere", "saddle", "torus"):
        assert rows[case]["fitted_exponent"] == pytest.approx(2.0, abs=0.05)
        assert rows[case]["observation_mode"] == "ambient-euclidean-chord"


def test_observation_modes_are_declared(surface_report: dict) -> None:
    modes = {mode["identifier"]: mode for mode in surface_report["observation_modes"]}
    assert modes["intrinsic-surface-distance"]["support"]["constant-curvature"] == "exact"
    assert (
        modes["intrinsic-surface-distance"]["support"]["parametric-surface"]
        == "unavailable"
    )
    assert modes["ambient-euclidean-chord"]["support"]["parametric-surface"] == "numerical"
    for identifier in ("scanner-reconstructed-chord", "camera-image-residual"):
        assert set(modes[identifier]["support"].values()) == {"unavailable"}
