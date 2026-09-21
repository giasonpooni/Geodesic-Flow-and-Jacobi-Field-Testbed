# SPDX-License-Identifier: MPL-2.0
"""Write a ``path-geometry-v1`` artefact, then replay it with nothing else loaded.

This is the boundary exercised end to end, in the inbound direction. The first
half stands in for the upstream testbed: it computes a path on a surface and
hands it over as a file. The second half is what a consumer actually has --
a file and this runtime, and no surface, no chart and no solver from the
producer.

Run it with ``--check`` and it asserts the two halves agree, which is the
statement worth making: a record replayed from the artefact is the record that
would have been produced with the surface in hand.

    python examples/replay_path_artefact.py --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geodesic_testbed.boundary import (
    PathGeometryArtefact,
    Units,
    artefact_from_envelope,
    read_artefact,
    transfer_record_from_artefact,
    write_artefact,
    write_record,
)
from geodesic_testbed.engine.envelope import integrate_paths
from geodesic_testbed.engine.surfaces import hyperbolic_paraboloid


def produce(destination: Path) -> Path:
    """The upstream half: compute a path and hand it over as a file."""
    surface = hyperbolic_paraboloid(0.6)
    envelope = integrate_paths(
        surface, u0=0.15, v0=-0.1, headings=[0.65], length=1.5, n_steps=600
    )[0]
    artefact = artefact_from_envelope(
        envelope,
        identifier="saddle-coupon-seed-route",
        surface_digest="surface:hyperbolic-paraboloid(scale=0.6)",
        path_digest="path:seed-route-600-steps",
        units=Units(length="metre", angle="radian"),
        curvature_interpolation="pchip-monotone",
    )
    return write_artefact(artefact, destination / "path-geometry-v1.json")


def consume(source: Path, destination: Path) -> Path:
    """The downstream half: a file, and a transfer record out of it."""
    artefact = read_artefact(source)
    record = transfer_record_from_artefact(artefact)
    return write_record(record, destination / "transfer-record-from-artefact.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results", help="where to write both files")
    parser.add_argument(
        "--check",
        action="store_true",
        help="also assert the replay agrees with the in-process computation",
    )
    arguments = parser.parse_args()
    destination = Path(arguments.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    artefact_path = produce(destination)
    record_path = consume(artefact_path, destination)
    print(artefact_path)
    print(record_path)

    if arguments.check:
        reopened = PathGeometryArtefact.from_dict(
            json.loads(artefact_path.read_text(encoding="utf-8"))
        )
        replayed = transfer_record_from_artefact(reopened)
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        for component in ("a", "a_rate", "b", "b_rate"):
            written = np.asarray(payload[component], dtype=float)
            assert np.array_equal(written, getattr(replayed, component)), component
        print(f"replay agrees on {replayed.arclength.size} samples, det Phi - 1 <= ", end="")
        print(f"{float(np.max(np.abs(replayed.determinant - 1.0))):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
