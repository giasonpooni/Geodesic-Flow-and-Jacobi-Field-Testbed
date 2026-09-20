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


def _assert_same_verdicts(fresh: dict, committed: dict, what: str) -> None:
    """Every check reaches the same conclusion, by identifier.

    This is the claim that survives a change of numpy build, and it took
    measuring to find that out. The obvious alternative -- compare the values
    themselves to some tolerance -- was tried and does not work, because of
    what this artefact *is*: a report about numerical error, most of whose
    numbers are therefore numerical error, and numerical error is exactly the
    thing two builds of ``libm`` disagree about. Measured across Python
    3.11/3.12/3.13 on a GitHub runner against this machine:

    * a fitted convergence order moved by 3e-6 relative -- small, and already
      past any tolerance tight enough to be worth having;
    * ``first_order_validity/coefficient_relative_error`` moved by **9%**. It
      is a relative error, so it is a difference of nearly equal numbers, and
      the cancellation amplifies the last bits into the leading ones;
    * ``jet_step_sensitivity`` at a differencing step of 1e-6 moved by a factor
      of **2.8**. That level is deep in the cancellation regime on purpose --
      it is there to show where the floor is -- so the value is noise and both
      answers are right.

    A tolerance loose enough to pass all three would be loose enough to pass a
    regression, which is the definition of a useless test.

    Verdicts do survive, because that is what a threshold is for: a residual
    may sit at 8e-14 here and 9e-14 there against a limit of 1e-13 and the
    artefact means the same thing in both places. And the coverage is complete
    rather than lucky -- AGENTS.md requires every quantitative claim in the
    README or the docs to correspond to a declared check with a threshold, so
    a regression large enough to matter is a regression that flips a verdict.
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


def test_a_flipped_verdict_is_what_the_comparison_must_catch(
    committed: dict,
) -> None:
    """A comparison nothing can fail is decoration, so this makes one fail."""
    import copy

    mutant = copy.deepcopy(committed)
    mutant["checks"][0]["passed"] = not mutant["checks"][0]["passed"]
    with pytest.raises(AssertionError, match="changed verdict"):
        _assert_same_verdicts(mutant, committed, "mutated")


def test_a_check_that_disappeared_is_what_the_comparison_must_also_catch(
    committed: dict,
) -> None:
    """Verdict identity is only meaningful over the same set of checks."""
    import copy

    mutant = copy.deepcopy(committed)
    mutant["checks"] = mutant["checks"][:-1]
    with pytest.raises(AssertionError, match="different checks"):
        _assert_same_verdicts(mutant, committed, "mutated")


def test_a_regression_large_enough_to_matter_flips_a_verdict(
    committed: dict,
) -> None:
    """Why verdict identity is enough, rather than merely all that is possible.

    Every quantitative claim here has a declared check with a threshold, so a
    value cannot move enough to matter without crossing one. Demonstrated
    rather than asserted: push each check's value past its own threshold and
    the verdict flips, every time.
    """
    for check in committed["checks"]:
        value, threshold, comparison = (
            check["value"], check["threshold"], check["comparison"]
        )
        if value is None:
            continue
        broken = threshold * 10.0 + 1.0 if comparison == "<=" else threshold - 1.0
        passes = broken <= threshold if comparison == "<=" else broken >= threshold
        assert not passes, (
            f"{check['id']}: a value of {broken!r} still satisfies "
            f"{comparison} {threshold!r}, so this check cannot detect a regression"
        )
