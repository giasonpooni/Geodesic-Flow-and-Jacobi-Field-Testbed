# SPDX-License-Identifier: MPL-2.0
"""Emit one path-sensitivity record: everything this runtime hands downstream.

The record below is the whole of the interface. An instrument workbench that
consumes it needs no other file from this repository and no knowledge of how
the numbers were produced -- which is the point, because the alternative is a
downstream system pinned to this solver's internals.

Run it to see what crosses::

    uv run python examples/emit_boundary_record.py --out out

The three contract fields that geometry alone cannot supply are filled in here
by the *caller*, not by the solver, and each one says on what basis:

* the starting covariance, converted from a declared tolerance box at a stated
  coverage factor, so a downstream resolvability figure reads as "at 3 sigma"
  rather than as an unqualified claim;
* the upstream artefact the path came from, because a sensitivity record is
  only as good as the path it was computed along;
* the calibration identifiers, left unbound -- this record was computed from an
  analytic torus, no instrument took part, and writing a placeholder identifier
  into it would be exactly the failure the field exists to prevent.

What the *runtime* fills in, because only it can: the path itself. Positions,
the Darboux frame as vectors rather than as a name, both normal curvatures, how
much of the requested path the chart could carry, whether the curve is a
geodesic, and -- for the price of a second integration -- what the numerics
cost, per quantity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geodesic_testbed.boundary import (
    Provenance,
    StartingCovariance,
    Units,
    UpstreamArtefact,
    read_record,
    write_record,
)
from geodesic_testbed.engine.envelope import estimate_convergence, integrate_path
from geodesic_testbed.engine.surfaces import torus

#: Millimetres and radians, declared once. A record whose units are a guess is
#: a record that can be wrong by a factor of a thousand while passing every
#: check that looks at shapes.
UNITS = Units(length="millimetre", angle="radian")


def build_record(*, length: float = 240.0, steps: int = 4000):
    """A torus path in millimetres, presented as a transfer record."""
    envelope = integrate_path(
        torus(200.0, 60.0), u0=0.3, v0=0.2, heading=0.6, length=length, n_steps=steps
    )
    return envelope.as_transfer_record(
        units=UNITS,
        # Paid for on purpose: a second integration at half the step, so the
        # record states what its own numbers cost instead of naming the recipe
        # and leaving the consumer to guess.
        convergence=estimate_convergence(envelope),
        datum_frame="coupon-datum-A",
        covariance=StartingCovariance.from_tolerance_box(
            0.100,
            0.0035,
            coverage_factor=3.0,
            units=UNITS,
            note="process tolerance box read as a 3-sigma pose distribution",
        ),
        upstream=(
            UpstreamArtefact(
                kind="analytic-surface",
                identifier="torus(R=200 mm, r=60 mm)",
                version="closed-form",
                note="no mesh took part; a meshed part would name its path artefact here",
            ),
        ),
        provenance=Provenance(note="boundary example: one path, one record"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out", help="directory to write the record into")
    args = parser.parse_args()

    record = build_record()
    path = write_record(record, Path(args.out) / "path-sensitivity-record.json")

    restored = read_record(path)
    summary = restored.to_dict(include_samples=False)
    print(path)
    print(f"  schema           {summary['schema']} under {summary['contract']}")
    print(f"  frame            {summary['frame']}")
    print(f"  units            {summary['units']['length']} / {summary['units']['angle']}")
    print(
        "  grid             {samples} samples over [{start:g}, {end:g}], "
        "uniform={uniform}".format(**summary["grid"])
    )
    print(f"  observation mode {summary['observation_mode']} on {summary['domain']}")
    print(f"  covariance       {summary['covariance']['basis']}")
    print(f"  provenance       {summary['provenance']['producer']} "
          f"{summary['provenance']['producer_version']}")
    print(f"  calibration      {'bound' if summary['calibration']['bound'] else 'unbound'}: "
          f"{summary['calibration']['note'] or summary['calibration']['calibration_ids']}")
    geometry = summary["geometry"]
    convergence = summary["resolution"]["convergence"]
    print(f"  geometry         {geometry['samples']} points, frame in "
          f"{geometry['coordinate_frame']}, datum {geometry['datum_frame']}")
    print(f"  geometry sigma   {geometry['uncertainty']['basis']}")
    print(f"  path type        {summary['path_type']}")
    print("  chart            {samples} of {requested_samples} samples, "
          "complete={complete}".format(**summary["chart"]))
    print(f"  error budget     position {convergence['position']:.2e}, "
          f"transfer {convergence['transfer']:.2e}, focus "
          f"{'n/a' if convergence['focus'] is None else format(convergence['focus'], '.2e')}")
    print(f"  amplification    {restored.amplification_score(0.100, 0.0035):.3f} "
          "(dimensionless, over the declared tolerance box)")


if __name__ == "__main__":
    main()
