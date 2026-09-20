"""The committed artefacts are part of the repository's claim, so they are tested.

A stale ``validation/report-v1.json`` would be worse than none at all: it would
look like evidence.  These tests pin its schema, its verdict and the headline
numbers, so a regression in the library cannot leave a passing report behind.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from geodesic_testbed.engine.canonical import canonical_float
from geodesic_testbed.engine.experiment import REPORT_SCHEMA, SUPERSEDES
from geodesic_testbed.engine.spaceforms import SpaceForm

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "validation" / "report-v1.json"
FIGURE_PATH = ROOT / "figures" / "jacobi-testbed-v1.png"

EXPECTED_ORDERS = {"euler": 1, "midpoint": 2, "rk4": 4}


def test_the_committed_report_exists_and_passed(committed: dict) -> None:
    assert committed["schema"] == REPORT_SCHEMA
    assert committed["supersedes"] == SUPERSEDES
    assert committed["summary"]["all_passed"] is True
    assert committed["summary"]["n_failed"] == 0
    assert committed["summary"]["n_checks"] >= 100


def test_the_committed_figure_exists(committed: dict) -> None:
    assert FIGURE_PATH.exists()
    assert FIGURE_PATH.stat().st_size > 50_000


def test_committed_convergence_orders(committed: dict) -> None:
    for key in ("jacobi_ode_convergence", "geodesic_flow_convergence"):
        for row in committed["results"][key]:
            if row["curvature"] == 0.0:
                assert row["regime"] == "exact-to-roundoff"
                continue
            assert row["fitted_order"] == pytest.approx(
                EXPECTED_ORDERS[row["integrator"]], abs=0.15
            )


def test_committed_first_order_coefficients_match_theory(committed: dict) -> None:
    for row in committed["results"]["first_order_validity"]:
        form = SpaceForm(row["curvature"])
        assert row["fitted_coefficient"] == pytest.approx(
            float(form.relative_deviation_coefficient(row["arc_length"])), rel=1e-2
        )


def test_committed_conjugate_point(committed: dict) -> None:
    assert committed["results"]["conjugate_point"]["jacobi_first_zero"] == pytest.approx(
        math.pi, abs=1e-9
    )


def test_the_committed_report_is_not_stale(committed: dict, report: dict) -> None:
    """Every check the current code declares is present in the committed report.

    Deliberately compared by check *identity*, not by floating-point value: the
    numbers legitimately move a little between numpy builds, while a check that
    appeared or disappeared means the artefact no longer describes this code.
    """
    fresh_ids = [check["id"] for check in report["checks"]]
    committed_ids = [check["id"] for check in committed["checks"]]
    assert committed_ids == fresh_ids
    assert set(committed["results"]) == set(report["results"])
    assert committed["config"] == report["config"]


def test_the_committed_surface_report_exists_and_passed(committed_surface_report: dict) -> None:
    assert committed_surface_report["summary"]["all_passed"] is True
    assert committed_surface_report["summary"]["n_failed"] == 0
    assert (ROOT / "figures" / "surfaces-testbed-v1.png").stat().st_size > 50_000


def test_the_committed_surface_report_is_not_stale(
    committed_surface_report: dict, surface_report: dict
) -> None:
    assert [check["id"] for check in committed_surface_report["checks"]] == [
        check["id"] for check in surface_report["checks"]
    ]
    assert committed_surface_report["config"] == surface_report["config"]


def test_the_committed_report_hash_is_reproducible(committed: dict, report: dict) -> None:
    """Recomputing the report reproduces the committed content hash exactly.

    This is what canonicalising every float to a fixed number of significant
    digits buys, and it is only worth anything if it is asserted. Without it
    the hash drifts on the last two or three digits of a BLAS reduction, an
    ``svd`` or a ``trapezoid`` -- quantities that did not change -- and a
    "content hash" that changes when the content did not is not an identity.

    Compared by *identity*, unlike the check-by-check comparison above: the
    numbers may legitimately move between numpy builds, and the claim here is
    precisely that they may not move by more than the canonicalisation hides.
    """
    assert report["content_hash"] == committed["content_hash"]


def test_the_committed_surface_report_hash_is_reproducible(
    committed_surface_report: dict, surface_report: dict
) -> None:
    assert surface_report["content_hash"] == committed_surface_report["content_hash"]


def test_every_float_in_a_committed_report_is_canonical(committed: dict) -> None:
    """No value in the artefact carries a digit the canonical form would drop."""

    def walk(node) -> None:
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, float):
            assert node == canonical_float(node), node

    walk({key: value for key, value in committed.items() if key != "environment"})
