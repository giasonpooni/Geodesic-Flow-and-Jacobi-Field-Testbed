"""Command line entry point: run the experiment, write the report, draw the figure."""

from __future__ import annotations

import argparse
import pathlib

from .experiment import ExperimentConfig, run_experiment, write_report

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMMITTED_REPORT = ROOT / "validation" / "report-v1.json"
COMMITTED_FIGURE = ROOT / "figures" / "jacobi-testbed-v1.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geojac-experiment",
        description="Run the geodesic-flow and Jacobi-field experiment.",
    )
    parser.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    parser.add_argument(
        "--update-committed",
        action="store_true",
        help="also refresh validation/report-v1.json and figures/jacobi-testbed-v1.png",
    )
    parser.add_argument("--no-figure", action="store_true", help="skip the figure")
    arguments = parser.parse_args(argv)

    report = run_experiment(ExperimentConfig())
    out = pathlib.Path(arguments.out)
    report_path = out / "report-v1.json"
    write_report(report, report_path)
    print(f"report  -> {report_path}")

    if not arguments.no_figure:
        from .figure import write_figure

        figure_path = out / "jacobi-testbed-v1.png"
        write_figure(report, figure_path)
        print(f"figure  -> {figure_path}")

    if arguments.update_committed:
        write_report(report, COMMITTED_REPORT)
        print(f"report  -> {COMMITTED_REPORT}")
        if not arguments.no_figure:
            from .figure import write_figure

            write_figure(report, COMMITTED_FIGURE)
            print(f"figure  -> {COMMITTED_FIGURE}")

    summary = report["summary"]
    print(
        f"\ncontent hash {report['content_hash']}\n"
        f"{summary['n_checks'] - summary['n_failed']}/{summary['n_checks']} checks passed"
    )
    for check in report["checks"]:
        if not check["passed"]:
            print(
                f"  FAILED {check['id']}: {check['value']} "
                f"{check['comparison']} {check['threshold']}"
            )
    return 0 if summary["all_passed"] else 1
