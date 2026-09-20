"""The committed artefacts are part of the repository's claim, so they are tested.

A stale ``validation/report-v1.json`` would be worse than none at all: it would
look like evidence.  These tests pin its schema, its verdict and the headline
numbers, so a regression in the library cannot leave a passing report behind.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from geodesic_testbed.engine.canonical import canonical_float, content_hash
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


#: Below this a value is a *residual* -- a difference of nearly equal numbers,
#: a Wronskian drift, an Euler-identity leftover -- and its last digits belong
#: to the platform's ``sin``, ``cosh`` and BLAS rather than to this code. Two
#: numpy builds disagree about such a value by a hundred per cent and both are
#: right, so comparing them by value says nothing. Its *verdict* against its
#: own declared threshold is compared instead, which is the thing the artefact
#: actually claims.
AGREEMENT_FLOOR = 1.0e-6

#: How far a value above that floor may move. A genuine regression moves such
#: a number by far more; a different libm moves it by far less.
AGREEMENT_RTOL = 1.0e-6


def _disagreements(fresh, committed, path: str = "") -> list[tuple[str, float, float]]:
    """Every value above the floor that moved further than the tolerance."""
    if isinstance(committed, dict) and isinstance(fresh, dict):
        found = []
        for key in committed:
            if key in fresh:
                found += _disagreements(fresh[key], committed[key], f"{path}/{key}")
        return found
    if isinstance(committed, list) and isinstance(fresh, list):
        found = []
        for index, (left, right) in enumerate(zip(fresh, committed, strict=False)):
            found += _disagreements(left, right, f"{path}[{index}]")
        return found
    if isinstance(committed, float) and isinstance(fresh, float):
        scale = max(abs(fresh), abs(committed))
        if scale < AGREEMENT_FLOOR:
            return []
        if abs(fresh - committed) <= AGREEMENT_RTOL * scale:
            return []
        return [(path, fresh, committed)]
    return []


def _assert_agrees(fresh: dict, committed: dict, what: str) -> None:
    volatile = {"environment", "content_hash"}
    moved = _disagreements(
        {k: v for k, v in fresh.items() if k not in volatile},
        {k: v for k, v in committed.items() if k not in volatile},
    )
    worst = sorted(
        moved, key=lambda item: abs(item[1] - item[2]) / abs(item[2]), reverse=True
    )[:10]
    assert not moved, (
        f"{len(moved)} value(s) above {AGREEMENT_FLOOR:g} in the {what} report "
        f"moved further than rtol={AGREEMENT_RTOL:g}. Worst:\n"
        + "\n".join(
            f"  {path}: {value!r} vs committed {old!r}" for path, value, old in worst
        )
    )


def _assert_same_verdicts(fresh: dict, committed: dict, what: str) -> None:
    """Every check reaches the same conclusion, by identifier.

    The claim that survives a change of numpy build. A residual may sit at
    8e-14 here and 9e-14 there against a threshold of 1e-13 and the artefact
    means the same thing in both places -- which is precisely what a threshold
    is for, and precisely what a comparison of the values themselves cannot
    express.
    """
    fresh_verdicts = {check["id"]: check["passed"] for check in fresh["checks"]}
    old_verdicts = {check["id"]: check["passed"] for check in committed["checks"]}
    assert set(fresh_verdicts) == set(old_verdicts), (
        f"the {what} report declares different checks than the committed one"
    )
    flipped = {
        identifier: (old_verdicts[identifier], verdict)
        for identifier, verdict in fresh_verdicts.items()
        if verdict != old_verdicts[identifier]
    }
    assert not flipped, f"{what}: checks changed verdict: {flipped}"


def test_the_committed_report_agrees_with_a_fresh_one(
    committed: dict, report: dict
) -> None:
    """Every value large enough for the comparison to mean something."""
    _assert_agrees(report, committed, "constant-curvature")


def test_the_committed_surface_report_agrees_with_a_fresh_one(
    committed_surface_report: dict, surface_report: dict
) -> None:
    _assert_agrees(surface_report, committed_surface_report, "surfaces")


def test_every_committed_check_reaches_the_same_verdict(
    committed: dict, report: dict,
    committed_surface_report: dict, surface_report: dict,
) -> None:
    """What the artefact claims, and the claim that crosses a numpy build."""
    _assert_same_verdicts(report, committed, "constant-curvature")
    _assert_same_verdicts(surface_report, committed_surface_report, "surfaces")


def test_the_content_hash_identifies_the_report_it_was_computed_from(
    committed: dict, committed_surface_report: dict
) -> None:
    """The hash is recomputable from the artefact, and is not a decoration.

    What it is *not* is a cross-platform identity. It answers "has anything in
    this report changed since it was last written here", which is the question
    ``tools/e2e.py`` asks a hundred times over on one machine, and it cannot
    answer "do two numpy builds agree" -- they do not, in the last digits, and
    nothing can make them. Four builds of numpy produced four hashes here.
    """
    for document in (committed, committed_surface_report):
        core = {
            key: value
            for key, value in document.items()
            if key not in ("environment", "summary", "content_hash")
        }
        assert document["content_hash"] == content_hash(core)
        assert len(document["content_hash"]) == 64


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


def test_the_agreement_comparison_catches_a_value_that_really_moved(
    committed: dict,
) -> None:
    """A tolerance nothing can fail is decoration, so this makes one fail.

    Three cases, and the third is the point: a value above the floor nudged by
    more than the tolerance must be reported, a value below the floor is not
    compared at all, and a nudge in the last digits of a real value -- which is
    what a different libm does -- must not be reported.
    """
    import copy

    def moved(mutate) -> list[str]:
        mutant = copy.deepcopy(committed)
        mutate(mutant)
        return [path for path, _, _ in _disagreements(mutant, committed)]

    large = next(
        check for check in committed["checks"]
        if check["value"] is not None and abs(check["value"]) > AGREEMENT_FLOOR
    )
    small = next(
        check for check in committed["checks"]
        if check["value"] is not None and 0.0 < abs(check["value"]) < AGREEMENT_FLOOR
    )

    def bump_a_real_value(document: dict) -> None:
        for check in document["checks"]:
            if check["id"] == large["id"]:
                check["value"] *= 1.0 + 1.0e-3

    def bump_a_residual(document: dict) -> None:
        for check in document["checks"]:
            if check["id"] == small["id"]:
                check["value"] *= 1000.0

    def nudge_the_last_digits(document: dict) -> None:
        for check in document["checks"]:
            if isinstance(check["value"], float):
                check["value"] *= 1.0 + 1.0e-13

    assert moved(bump_a_real_value), "a 1e-3 relative move went unreported"
    assert not moved(bump_a_residual), (
        "a residual below the floor was compared by value; its last digits "
        "belong to the platform and comparing them would make the test flaky"
    )
    assert not moved(nudge_the_last_digits), (
        "a 1e-13 relative nudge was reported; that is what a different libm "
        "does, and reporting it would make the comparison unusable"
    )


def test_a_flipped_verdict_is_what_the_comparison_must_catch(
    committed: dict,
) -> None:
    """The residuals are not compared by value, so this is what protects them."""
    import copy

    mutant = copy.deepcopy(committed)
    mutant["checks"][0]["passed"] = not mutant["checks"][0]["passed"]
    with pytest.raises(AssertionError, match="changed verdict"):
        _assert_same_verdicts(mutant, committed, "mutated")
