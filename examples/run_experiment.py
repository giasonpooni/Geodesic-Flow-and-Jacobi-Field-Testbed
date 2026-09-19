"""Run the experiment without installing the package.

    PYTHONPATH=src python examples/run_experiment.py
    PYTHONPATH=src python examples/run_experiment.py --update-committed

Exits non-zero if any declared check fails, so the same command gates CI.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from geojac.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
