"""Acquisition and retention: when the instrument has the path, and when it loses it.

A minimum on resolvability taken over a whole route is unsatisfiable, because
``b(0) = 0`` makes ``rho(0) = 0`` on every candidate. Taking it only after the
first sample that clears the threshold fixes that and introduces a worse
failure: a route whose ``rho`` first clears the threshold at its final sample
passes vacuously, having been tracked for no distance at all.

What a sensor actually promises is a schedule, so that is what is declared
here:

* acquire when ``rho`` holds at or above ``acquire_threshold`` for a declared
  window, and no later than ``max_acquisition_distance``;
* retain while ``rho`` stays at or above a *lower* ``hold_threshold``;
* tolerate excursions below it for at most ``max_loss_distance``;
* require at least ``min_tracked_distance`` of continuous tracking.

The two thresholds differ on purpose. A single one chatters between ACQUIRED
and TRACK_LOST whenever noise moves ``rho`` across it; hysteresis is what stops
that, and it is a property of the sensor's detection performance, not of the
geometry. Every threshold here comes from the instrument, and this module only
applies them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

Array = np.ndarray

#: What happened to the track along one route.
TRACKED = "TRACKED"
NEVER_ACQUIRED = "NEVER_ACQUIRED"
LATE_ACQUISITION = "LATE_ACQUISITION"
TRACK_LOST = "TRACK_LOST"
TOO_SHORT = "INSUFFICIENT_TRACKED_DISTANCE"
OUTCOMES = (TRACKED, NEVER_ACQUIRED, LATE_ACQUISITION, TRACK_LOST, TOO_SHORT)


@dataclass(frozen=True)
class AcquisitionSpec:
    """The sensor's detection schedule, in resolvability and arc length."""

    acquire_threshold: float
    hold_threshold: float
    acquisition_window: float = 0.0
    max_acquisition_distance: float | None = None
    min_tracked_distance: float | None = None
    max_loss_distance: float = 0.0
    processing: str = "offline"
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("acquire_threshold", "hold_threshold"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.acquire_threshold <= self.hold_threshold:
            raise ValueError(
                "acquire_threshold must exceed hold_threshold; equal thresholds "
                "chatter between ACQUIRED and TRACK_LOST on noise"
            )
        for name in ("acquisition_window", "max_loss_distance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.processing not in ("causal", "offline"):
            raise ValueError("processing must be 'causal' or 'offline'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackingOutcome:
    """Whether a route was acquired, how long it was held, and why it ended."""

    outcome: str
    acquired: bool
    acquisition_arclength: float | None
    acquisition_latency: float | None
    tracked_distance: float
    min_resolvability_while_tracked: float | None
    max_resolvability: float
    loss_intervals: list[tuple[float, float]]
    longest_loss: float
    processing: str
    detail: str

    @property
    def satisfied(self) -> bool:
        return self.outcome == TRACKED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loss_intervals"] = [list(interval) for interval in self.loss_intervals]
        payload["satisfied"] = self.satisfied
        return payload


def _runs(mask: Array) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a boolean mask, as half-open index pairs."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2], strict=True))


def evaluate_tracking(
    arclength: Array, resolvability: Array, spec: AcquisitionSpec
) -> TrackingOutcome:
    """Apply a sensor's acquisition schedule to a resolvability profile."""
    grid = np.asarray(arclength, dtype=float)
    rho = np.asarray(resolvability, dtype=float)
    if grid.shape != rho.shape:
        raise ValueError("arclength and resolvability must have the same shape")
    start = float(grid[0])
    end = float(grid[-1])
    maximum = float(np.max(rho))

    def outcome(name: str, detail: str, **overrides: Any) -> TrackingOutcome:
        base: dict[str, Any] = {
            "outcome": name,
            "acquired": name not in (NEVER_ACQUIRED,),
            "acquisition_arclength": None,
            "acquisition_latency": None,
            "tracked_distance": 0.0,
            "min_resolvability_while_tracked": None,
            "max_resolvability": maximum,
            "loss_intervals": [],
            "longest_loss": 0.0,
            "processing": spec.processing,
            "detail": detail,
        }
        return TrackingOutcome(**(base | overrides))

    # Acquisition: the first sustained run at or above the higher threshold.
    above = rho >= spec.acquire_threshold
    acquisition_index: int | None = None
    for first, last in _runs(above):
        if float(grid[last - 1] - grid[first]) >= spec.acquisition_window:
            acquisition_index = int(first)
            break
    if acquisition_index is None:
        return outcome(
            NEVER_ACQUIRED,
            f"rho never held at or above {spec.acquire_threshold:g} for "
            f"{spec.acquisition_window:g}; it peaked at {maximum:g}",
        )

    acquired_at = float(grid[acquisition_index])
    latency = acquired_at - start
    if (
        spec.max_acquisition_distance is not None
        and latency > float(spec.max_acquisition_distance)
    ):
        return outcome(
            LATE_ACQUISITION,
            f"acquired only after {latency:g}, beyond the declared "
            f"{float(spec.max_acquisition_distance):g}",
            acquisition_arclength=acquired_at,
            acquisition_latency=latency,
        )

    # Retention: excursions below the lower threshold, after acquisition.
    tail_grid = grid[acquisition_index:]
    tail_rho = rho[acquisition_index:]
    below = tail_rho < spec.hold_threshold
    intervals: list[tuple[float, float]] = []
    fatal: float | None = None
    for first, last in _runs(below):
        begin = float(tail_grid[first])
        finish = float(tail_grid[min(last, tail_grid.size - 1)])
        intervals.append((begin, finish))
        if finish - begin > spec.max_loss_distance and fatal is None:
            fatal = begin
    longest = max((end_ - start_ for start_, end_ in intervals), default=0.0)

    if fatal is not None:
        tracked = fatal - acquired_at
        return outcome(
            TRACK_LOST,
            f"rho stayed below {spec.hold_threshold:g} for longer than the "
            f"tolerated {spec.max_loss_distance:g}, first at {fatal:g}",
            acquisition_arclength=acquired_at,
            acquisition_latency=latency,
            tracked_distance=tracked,
            min_resolvability_while_tracked=float(np.min(tail_rho)),
            loss_intervals=intervals,
            longest_loss=longest,
        )

    tracked = end - acquired_at
    if (
        spec.min_tracked_distance is not None
        and tracked < float(spec.min_tracked_distance)
    ):
        return outcome(
            TOO_SHORT,
            f"tracked for {tracked:g}, short of the declared "
            f"{float(spec.min_tracked_distance):g}; acquiring at the very end of a "
            f"route is not tracking it",
            acquisition_arclength=acquired_at,
            acquisition_latency=latency,
            tracked_distance=tracked,
            min_resolvability_while_tracked=float(np.min(tail_rho)),
            loss_intervals=intervals,
            longest_loss=longest,
        )

    return outcome(
        TRACKED,
        f"acquired at {acquired_at:g} and held for {tracked:g}",
        acquisition_arclength=acquired_at,
        acquisition_latency=latency,
        tracked_distance=tracked,
        min_resolvability_while_tracked=float(np.min(tail_rho)),
        loss_intervals=intervals,
        longest_loss=longest,
    )
