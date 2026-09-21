# SPDX-License-Identifier: MPL-2.0
"""Numerical engine: geodesic flow, Jacobi transfer maps, verification reports.

This subpackage is the verified numerical core. The application contracts that
consume it -- tolerances, manufacturing and inspection assessments, reports --
live one level up in :mod:`geodesic_testbed`, and are the stable public API.
Import from here when you need the solver itself: space forms, parametric
surfaces, integrators, the sensitivity envelope, or the experiment reports.
"""

from __future__ import annotations

__all__: list[str] = []
