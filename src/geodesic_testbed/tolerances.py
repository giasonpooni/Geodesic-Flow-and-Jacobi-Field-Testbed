"""Tolerance propagation through Jacobi fundamental solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .engine.record import Units, to_transfer_record

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PathTolerance:
    """Bounds on one path's starting pose in the transverse plane.

    ``units`` is optional and, when given, is checked against the units of the
    transfer record it is applied to. A tolerance in millimetres against a
    record in metres is a thousandfold error that nothing else here would
    catch.
    """

    lateral: float = 0.0
    heading: float = 0.0
    units: Units | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.lateral) or self.lateral < 0.0:
            raise ValueError("lateral tolerance must be finite and nonnegative")
        if not np.isfinite(self.heading) or self.heading < 0.0:
            raise ValueError("heading tolerance must be finite and nonnegative")

    def envelope(self, source, *, combination: str = "worst-case") -> Array:
        """Worst transverse deviation this tolerance allows along the path.

        ``source`` is anything that can present a transfer record: a
        :class:`~geodesic_testbed.jacobi.JacobiTrace` from a declared curvature
        profile, a ``PathEnvelope`` from a real parametric surface, or a record
        itself. They go through identical arithmetic.
        """
        record = to_transfer_record(source)
        if self.units is not None and self.units != record.units:
            raise ValueError(
                f"tolerance is in {self.units.to_dict()} but the record is in "
                f"{record.units.to_dict()}"
            )
        if combination == "worst-case":
            return record.cross_track_error(self.lateral, self.heading)
        if combination == "rss":
            return record.rss_cross_track_error(self.lateral, self.heading)
        raise ValueError("combination must be 'worst-case' or 'rss'")

    def heading_envelope(self, source) -> Array:
        """Worst downstream *heading* error this tolerance allows, ``e_alpha(s)``.

        The other half of the transfer map, and the half a cross-track bound
        alone cannot see. Because ``det Phi = 1`` the flow cannot shrink both
        at once, so a path chosen to keep this small has generally pushed the
        error into the transverse direction, and the reverse.
        """
        return to_transfer_record(source).heading_error(self.lateral, self.heading)
