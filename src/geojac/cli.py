"""Command line entry point: run the experiments, write the reports, draw the figures."""

from __future__ import annotations

import argparse
import pathlib

from .experiment import ExperimentConfig, run_experiment, write_report
from .experiment_surfaces import SurfaceConfig, run_surface_experiment

ROOT = pathlib.Path(__file__).resolve().parents[2]

STAGES = {
    "constant-curvature": {
        "report": "report-v1.json",
        "figure": "jacobi-testbed-v1.png",
        "committed_report": ROOT / "validation" / "report-v1.json",
        "committed_figure": ROOT / "figures" / "jacobi-testbed-v1.png",
    },
    "surfaces": {
        "report": "report-v2-surfaces.json",
        "figure": "surfaces-testbed-v1.png",
        "committed_report": ROOT / "validation" / "report-v2-surfaces.json",
        "committed_figure": ROOT / "figures" / "surfaces-testbed-v1.png",
    },
}


def _run(stage: str) -> dict:
    if stage == "constant-curvature":
        return run_experiment(ExperimentConfig())
    return run_surface_experiment(SurfaceConfig())


def _draw(stage: str, report: dict, path) -> None:
    if stage == "constant-curvature":
        from .figure import write_figure
    else:
        from .figure_surfaces import write_figure
    write_figure(report, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geojac-experiment",
        description="Run the geodesic-flow and Jacobi-field experiments.",
    )
    parser.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    parser.add_argument(
        "--stage",
        choices=(*STAGES, "all"),
        default="all",
        help="which experiment to run (default: all)",
    )
    parser.add_argument(
        "--update-committed",
        action="store_true",
        help="also refresh the tracked reports and figures",
    )
    parser.add_argument("--no-figure", action="store_true", help="skip the figures")
    arguments = parser.parse_args(argv)

    stages = tuple(STAGES) if arguments.stage == "all" else (arguments.stage,)
    out = pathlib.Path(arguments.out)
    failed = 0
    for stage in stages:
        paths = STAGES[stage]
        report = _run(stage)
        report_path = out / paths["report"]
        write_report(report, report_path)
        print(f"{stage}: report  -> {report_path}")
        if not arguments.no_figure:
            figure_path = out / paths["figure"]
            _draw(stage, report, figure_path)
            print(f"{stage}: figure  -> {figure_path}")
        if arguments.update_committed:
            write_report(report, paths["committed_report"])
            print(f"{stage}: report  -> {paths['committed_report']}")
            if not arguments.no_figure:
                _draw(stage, report, paths["committed_figure"])
                print(f"{stage}: figure  -> {paths['committed_figure']}")

        summary = report["summary"]
        failed += summary["n_failed"]
        print(
            f"{stage}: content hash {report['content_hash']}\n"
            f"{stage}: {summary['n_checks'] - summary['n_failed']}/{summary['n_checks']} "
            "checks passed"
        )
        for check in report["checks"]:
            if not check["passed"]:
                print(
                    f"  FAILED {check['id']}: {check['value']} "
                    f"{check['comparison']} {check['threshold']}"
                )
    return 0 if failed == 0 else 1
