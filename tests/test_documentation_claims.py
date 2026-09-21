# SPDX-License-Identifier: MPL-2.0
"""The numbers in the prose are the numbers in the reports.

A document that states a count, a schema or a bound is making a quantitative
claim, and this repository's rule is that every quantitative claim carries a
declared check.  Nothing was checking the prose itself, and three claims had
drifted: ``docs/EXPERIMENT.md`` said 126 checks against 143, ``docs/SURFACES.md``
said schema v2 and 65 checks against v3 and 167, and ``docs/INSTRUMENT.md`` said
235 across two stages against 310.  Each was true when it was written.

So the counts and the schema identifiers are read out of the committed reports
rather than remembered, and each bound a document quotes is held against every
value in the family of checks it describes -- which makes a quoted bound a
declared check in the same sense as any other, and makes a regression past it
fail here as well as in the report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REPORTS = {
    "constant-curvature": ROOT / "validation" / "report-v1.json",
    "surfaces": ROOT / "validation" / "report-v2-surfaces.json",
}

#: Which report a document's unqualified check count refers to.  A count that
#: says "across two stages" refers to both and overrides this.
DOCUMENT_STAGE = {
    "docs/EXPERIMENT.md": "constant-curvature",
    "docs/SURFACES.md": "surfaces",
}

SCHEMA_IN_PROSE = re.compile(r"geodesic-jacobi-(?:report|surfaces)-v\d+")
COUNT_IN_PROSE = re.compile(r"(\d+) declared checks(?: (across two stages))?")

#: ``(document, quoted text, bound, comparison, check-id prefixes)``.  The bound
#: must appear in the document verbatim, and must bound every value in those
#: families in the stated direction.  ``"<="`` means the prose quotes a ceiling
#: ("held to 6e-15"); ``">="`` means it quotes a floor ("converges at order
#: 3.8"), where the claim is that no measurement falls below it.
QUOTED_BOUNDS = (
    ("docs/INSTRUMENT.md", "4e-4", 4e-4, "<=", ("variation-coefficient",)),
    ("docs/INSTRUMENT.md", "3e-15", 3e-15, "<=", ("conjugate-point/jacobi-zero",)),
    ("docs/INSTRUMENT.md", "6e-15", 6e-15, "<=", ("transfer-determinant",)),
    ("docs/INSTRUMENT.md", "5e-13", 5e-13, "<=", ("surface-anchor",)),
    (
        "docs/BOUNDARY.md",
        "order 3.8",
        3.8,
        ">=",
        ("imported-path-interpolation-order/pchip-monotone",),
    ),
    (
        "docs/BOUNDARY.md",
        "1.9997",
        1.999,
        ">=",
        ("imported-path-interpolation-order/linear",),
    ),
    ("README.md", "1.9997", 1.999, ">=", ("imported-path-interpolation-order/linear",)),
    ("docs/BOUNDARY.md", "580", 580.0, ">=", ("imported-path-interpolation-matters",)),
    ("README.md", "580", 580.0, ">=", ("imported-path-interpolation-matters",)),
    ("README.md", "4.4e-13", 4.4e-13, "<=", ("imported-path-anchor",)),
    ("README.md", "2.5e-14", 2.5e-14, "<=", ("imported-path-determinant",)),
    # The validity envelope: a ceiling on how far the measured bound is from the
    # closed form where one exists, and on the self-consistency residual where
    # one does not.
    (
        "docs/BOUNDARY.md",
        "0.72%",
        0.0072,
        "<=",
        ("validity-envelope-matches-the-chord-form",),
    ),
    ("README.md", "0.72%", 0.0072, "<=", ("validity-envelope-matches-the-chord-form",)),
    (
        "docs/BOUNDARY.md",
        "1.3%",
        0.013,
        "<=",
        ("validity-envelope-bound-is-self-consistent",),
    ),
    ("README.md", "1.3%", 0.013, "<=", ("validity-envelope-bound-is-self-consistent",)),
    # The cylinder's envelope is tighter than the plate's; the prose says
    # 15.7%, so the ratio must be at or below 0.8432.
    (
        "docs/BOUNDARY.md",
        "15.7% tighter",
        0.8432,
        "<=",
        ("validity-envelope-chord-term-is-visible",),
    ),
    # A correlated R carries 6.8% of an independent one's information.
    (
        "docs/BOUNDARY.md",
        "6.8% of the information",
        0.068,
        "<=",
        ("gramian-correlated-noise-carries-less",),
    ),
    ("README.md", "6.8%", 0.068, "<=", ("gramian-correlated-noise-carries-less",)),
)


def _load(stage: str) -> dict:
    return json.loads(REPORTS[stage].read_text())


def _documents() -> list[Path]:
    return sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _values_for(prefixes: tuple[str, ...]) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for stage in REPORTS:
        for check in _load(stage)["checks"]:
            if check["id"].startswith(prefixes) and isinstance(check["value"], (int, float)):
                found.append((check["id"], float(check["value"])))
    return found


@pytest.mark.parametrize("document", _documents(), ids=_relative)
def test_every_schema_named_in_prose_is_a_schema_a_report_declares(document: Path) -> None:
    named = set(SCHEMA_IN_PROSE.findall(document.read_text()))
    if not named:
        pytest.skip("names no report schema")
    declared = {_load(stage)["schema"] for stage in REPORTS}
    assert named <= declared, (
        f"{_relative(document)} names {sorted(named - declared)}, "
        f"but the committed reports declare {sorted(declared)}"
    )


@pytest.mark.parametrize("document", _documents(), ids=_relative)
def test_every_check_count_in_prose_is_the_count_in_the_report(document: Path) -> None:
    relative = _relative(document)
    claims = COUNT_IN_PROSE.findall(document.read_text())
    if not claims:
        pytest.skip("states no check count")
    totals = {stage: len(_load(stage)["checks"]) for stage in REPORTS}
    for count, both_stages in claims:
        if both_stages:
            expected = sum(totals.values())
            scope = "across two stages"
        else:
            stage = DOCUMENT_STAGE.get(relative)
            assert stage is not None, (
                f"{relative} states an unqualified count of {count} declared checks but no "
                "stage is declared for it in DOCUMENT_STAGE; say 'across two stages' or add it"
            )
            expected = totals[stage]
            scope = stage
        assert int(count) == expected, (
            f"{relative} says {count} declared checks {scope}; the committed report has "
            f"{expected}. Regenerate the artefacts and update the prose together."
        )


@pytest.mark.parametrize("document", _documents(), ids=_relative)
def test_a_prose_claim_of_no_failures_is_true(document: Path) -> None:
    if "0 failed" not in document.read_text():
        pytest.skip("claims nothing about failures")
    failed = {
        stage: [c["id"] for c in _load(stage)["checks"] if not c["passed"]] for stage in REPORTS
    }
    assert not any(failed.values()), f"{_relative(document)} says 0 failed; failures: {failed}"


@pytest.mark.parametrize(
    ("document", "quoted", "bound", "comparison", "prefixes"),
    QUOTED_BOUNDS,
    ids=[f"{d}:{q}" for d, q, _, _, _ in QUOTED_BOUNDS],
)
def test_a_bound_quoted_in_prose_bounds_the_checks_it_describes(
    document: str, quoted: str, bound: float, comparison: str, prefixes: tuple[str, ...]
) -> None:
    text = (ROOT / document).read_text()
    assert quoted in text, f"{document} no longer quotes {quoted}; this table is stale"
    values = _values_for(prefixes)
    assert values, f"no check in either report has an id starting with {prefixes}"
    worst_id, worst = _worst(values, comparison)
    if comparison == "<=":
        assert worst <= bound, (
            f"{document} quotes {quoted} for {prefixes}, but {worst_id} measures {worst:.6g}"
        )
    else:
        assert worst >= bound, (
            f"{document} quotes {quoted} for {prefixes}, but {worst_id} measures {worst:.6g}"
        )


def test_the_bounds_table_is_tight_enough_to_catch_a_regression() -> None:
    """A bound ten times the worst measurement would pass while meaning nothing.

    The point of quoting a figure in prose is that it is close to what was
    measured.  Each quoted bound must sit within one decade of the worst value
    in the family it describes, or it is decoration rather than a claim.
    """
    for document, quoted, bound, comparison, prefixes in QUOTED_BOUNDS:
        worst_id, worst = _worst(_values_for(prefixes), comparison)
        slack = bound <= 10.0 * worst if comparison == "<=" else bound >= worst / 10.0
        assert slack, (
            f"{document} quotes {quoted} for {prefixes}, but the worst value is {worst:.6g} "
            f"({worst_id}); a bound that loose does not constrain anything"
        )


def _worst(values: list[tuple[str, float]], comparison: str) -> tuple[str, float]:
    """The measurement closest to failing the quoted bound."""
    pick = max if comparison == "<=" else min
    return pick(values, key=lambda pair: pair[1])
