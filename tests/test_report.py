# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import json

from geodesic_testbed.reports import build_reference_report, write_reference_report


def test_reference_report_has_explicit_claim_scope() -> None:
    report = build_reference_report()
    assert report["schema"] == "geodesic-sensitivity-report-v1"
    assert report["claim_scope"] == "first-order-computational-analysis"
    assert len(report["numerical_validation"]) == 3
    assert report["application_examples"]["positive-curvature"]["inspection"]["reliable"]
    assert not report["application_examples"]["negative-curvature"]["inspection"]["reliable"]


def test_reference_report_round_trips(tmp_path) -> None:
    json_path, markdown_path = write_reference_report(tmp_path)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema"] == "geodesic-sensitivity-report-v1"
    assert "Numerical validation" in markdown_path.read_text(encoding="utf-8")
