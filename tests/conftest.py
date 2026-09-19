"""Shared fixtures.

The experiment is run once per test session and shared, so that the suite can
compare the freshly computed report with the committed one without paying for
the sweeps twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geodesic_testbed.engine.experiment import ExperimentConfig, run_experiment
from geodesic_testbed.engine.experiment_surfaces import SurfaceConfig, run_surface_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def report() -> dict:
    """A report computed now, from the code in the working tree."""
    return run_experiment(ExperimentConfig())


@pytest.fixture(scope="session")
def committed() -> dict:
    """The report checked into ``validation/``."""
    return json.loads((ROOT / "validation" / "report-v1.json").read_text())


@pytest.fixture(scope="session")
def surface_report() -> dict:
    """The stage-two report, computed now from the code in the working tree."""
    return run_surface_experiment(SurfaceConfig())


@pytest.fixture(scope="session")
def committed_surface_report() -> dict:
    """The stage-two report checked into ``validation/``."""
    return json.loads((ROOT / "validation" / "report-v2-surfaces.json").read_text())
