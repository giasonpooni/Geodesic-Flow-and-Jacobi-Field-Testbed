"""Write the first constant-curvature validation and application report.

Writes into ``results/`` by default, which is where the tracked copy lives and
what the README documents. ``--out`` sends it somewhere else instead, so that a
verification run can rebuild the report and compare it with the tracked one
without overwriting the thing it is checking -- and so that two such runs can
happen at once without racing for the same file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geodesic_testbed.reports import write_reference_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results",
        help="directory to write into (default: the tracked results/)",
    )
    arguments = parser.parse_args()
    json_path, markdown_path = write_reference_report(arguments.out)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
