# SPDX-License-Identifier: MPL-2.0
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

**When an event happens, and when it can be known.** A sustained window is
recognised only at its end. Looking back over a finished run, the condition
first held at the window's start; a live instrument cannot say so until the
window completes, because until then the run might still be cut short. The two
answers differ by the window length, and the difference is not bookkeeping: it
is how far the tool travels before the sensor admits it has the path.

Both are therefore recorded, for acquisition and for loss alike:

* ``acquisition_window_started_at`` / ``loss_started_at`` -- retrospective,
  where the condition actually began;
* ``acquisition_declared_at`` / ``track_loss_declared_at`` -- where a causal
  instrument could first know it.

``processing`` selects which one drives latency, the maximum-acquisition
distance and the tracked span. ``causal`` uses the declaration; ``offline``
may use the retrospective start, because an offline analysis has the whole
record in hand. An offline schedule reported as a real-time result is the
error this distinction exists to prevent.

**Events are located between samples.** Every threshold crossing here is
linearly interpolated, for the same reason the focus locations are
Hermite-refined: an event snapped to the sample that follows it is known only
to the sample spacing, and the rounding is one-sided, so every latency comes
out biased upward by half a step. A loss is *declared* exactly one tolerated
length after the excursion began -- when a causal instrument's timer expires --
rather than at whichever sample came next.

**What "while tracked" means.** ``min_resolvability_while_tracked`` is taken
over the span the instrument actually held the path: from acquisition to the
end of the route, or to the point where the track was lost. It is never taken
over the samples past a loss. Those samples are below the hold threshold by
construction, so including them would report the depth of the failure as a
property of the tracked stretch and would make the figure useless for the one
thing it is for -- saying how much margin the instrument had while it was
working. It is ``None`` when there was no tracked span at all.
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
        for name in ("max_acquisition_distance", "min_tracked_distance"):
            declared = getattr(self, name)
            if declared is None:
                continue
            value = float(declared)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative when declared")
        if self.processing not in ("causal", "offline"):
            raise ValueError("processing must be 'causal' or 'offline'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackingOutcome:
    """Whether a route was acquired, how long it was held, and why it ended."""

    outcome: str
    acquired: bool
    #: The timing this ``processing`` mode acts on: the declaration for
    #: ``causal``, the retrospective window start for ``offline``.
    acquisition_arclength: float | None
    acquisition_latency: float | None
    #: Where the sustained above-threshold run began, known only in hindsight.
    acquisition_window_started_at: float | None
    #: Where that run completed the declared window -- the earliest a live
    #: instrument could assert it had the path.
    acquisition_declared_at: float | None
    tracked_distance: float
    min_resolvability_while_tracked: float | None
    max_resolvability: float
    loss_intervals: list[tuple[float, float]]
    longest_loss: float
    #: Where the fatal excursion below the hold threshold began.
    loss_started_at: float | None
    #: Where that excursion first exceeded ``max_loss_distance`` -- the
    #: earliest a live instrument could declare the track lost.
    track_loss_declared_at: float | None
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


def _crossing(grid: Array, values: Array, threshold: float, index: int) -> float:
    """Where ``rho`` crossed ``threshold`` between samples ``index-1`` and ``index``.

    Linear interpolation, and the reason to bother is the same one that made
    the focus locations Hermite-refined: an event reported at a sample boundary
    is located only to the sample spacing, and a schedule that says "acquire
    within 1.0 of path length" is being judged against a number known to 0.005.
    Rounding an event to the sample that happened to follow it also biases
    every latency upward by half a step, systematically, in the direction that
    makes a route look worse than it is.

    The bracket is the interval where the sign of ``rho - threshold`` changes,
    so the root is inside it and the interpolation cannot leave it.
    """
    if index <= 0 or index >= grid.size:
        return float(grid[max(0, min(index, grid.size - 1))])
    before, after = float(values[index - 1]) - threshold, float(values[index]) - threshold
    if before == after:
        return float(grid[index])
    weight = before / (before - after)
    weight = min(max(weight, 0.0), 1.0)
    return float(grid[index - 1] + weight * (grid[index] - grid[index - 1]))


def evaluate_tracking(
    arclength: Array, resolvability: Array, spec: AcquisitionSpec
) -> TrackingOutcome:
    """Apply a sensor's acquisition schedule to a resolvability profile.

    Events are located twice: where the condition began, and where a causal
    instrument could first know it had. ``spec.processing`` decides which of
    the two drives latency, the maximum-acquisition distance and the tracked
    span.
    """
    grid = np.asarray(arclength, dtype=float)
    rho = np.asarray(resolvability, dtype=float)
    if grid.ndim != 1:
        raise ValueError("arclength must be one-dimensional")
    if grid.shape != rho.shape:
        raise ValueError("arclength and resolvability must have the same shape")
    if grid.size < 2:
        raise ValueError("a tracking grid needs at least two samples")
    if not np.all(np.isfinite(grid)):
        raise ValueError("arclength must be finite")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("arclength must be strictly increasing")
    if not np.all(np.isfinite(rho)):
        raise ValueError("resolvability must be finite")
    causal = spec.processing == "causal"
    start = float(grid[0])
    end = float(grid[-1])
    maximum = float(np.max(rho))

    def outcome(name: str, detail: str, **overrides: Any) -> TrackingOutcome:
        base: dict[str, Any] = {
            "outcome": name,
            "acquired": name not in (NEVER_ACQUIRED,),
            "acquisition_arclength": None,
            "acquisition_latency": None,
            "acquisition_window_started_at": None,
            "acquisition_declared_at": None,
            "tracked_distance": 0.0,
            "min_resolvability_while_tracked": None,
            "max_resolvability": maximum,
            "loss_intervals": [],
            "longest_loss": 0.0,
            "loss_started_at": None,
            "track_loss_declared_at": None,
            "processing": spec.processing,
            "detail": detail,
        }
        return TrackingOutcome(**(base | overrides))

    # Acquisition: the first run at or above the higher threshold that lasts
    # the declared window. The run begins at ``first``; the window completes at
    # the first sample ``window_index`` far enough past it, and that is the
    # earliest a causal instrument can assert anything.
    above = rho >= spec.acquire_threshold
    window_start_index: int | None = None
    declared_index: int | None = None
    for first, last in _runs(above):
        reached = np.flatnonzero(
            grid[first:last] - grid[first] >= spec.acquisition_window
        )
        if reached.size:
            window_start_index = int(first)
            declared_index = int(first + reached[0])
            break
    if window_start_index is None or declared_index is None:
        return outcome(
            NEVER_ACQUIRED,
            f"rho never held at or above {spec.acquire_threshold:g} for "
            f"{spec.acquisition_window:g}; it peaked at {maximum:g}",
        )

    # Both events are located between samples. The run *began* where rho
    # crossed the acquire threshold, not at the first sample that happened to
    # be above it; the window then completes one window length later.
    window_started_at = _crossing(
        grid, rho, spec.acquire_threshold, window_start_index
    )
    declared_at = max(
        float(grid[declared_index]), window_started_at + spec.acquisition_window
    )
    acquired_at = declared_at if causal else window_started_at
    acquisition_index = declared_index if causal else window_start_index
    latency = acquired_at - start
    timings: dict[str, Any] = {
        "acquisition_arclength": acquired_at,
        "acquisition_latency": latency,
        "acquisition_window_started_at": window_started_at,
        "acquisition_declared_at": declared_at,
    }
    if (
        spec.max_acquisition_distance is not None
        and latency > float(spec.max_acquisition_distance)
    ):
        return outcome(
            LATE_ACQUISITION,
            f"{'declared' if causal else 'acquired'} only after {latency:g}, "
            f"beyond the declared {float(spec.max_acquisition_distance):g}"
            + (
                f" (the run began at {window_started_at:g}, but a causal "
                f"instrument knows only at {declared_at:g})"
                if causal and declared_at > window_started_at
                else ""
            ),
            **timings,
        )

    # Retention: excursions below the lower threshold, after acquisition. Each
    # one is fatal from the sample at which it first exceeds the tolerated
    # length -- again, where it began is hindsight.
    tail_grid = grid[acquisition_index:]
    tail_rho = rho[acquisition_index:]
    below = tail_rho < spec.hold_threshold
    intervals: list[tuple[float, float]] = []
    loss_started_at: float | None = None
    loss_declared_at: float | None = None
    for first, last in _runs(below):
        begin = _crossing(tail_grid, tail_rho, spec.hold_threshold, first)
        finish = (
            _crossing(tail_grid, tail_rho, spec.hold_threshold, last)
            if last < tail_grid.size
            else float(tail_grid[-1])
        )
        intervals.append((begin, finish))
        if loss_started_at is not None:
            continue
        if finish - begin > spec.max_loss_distance:
            loss_started_at = begin
            # The declaration is exactly one tolerated length after the
            # excursion began: that is when a causal instrument's timer
            # expires, and it has nothing to do with where a sample fell.
            loss_declared_at = min(begin + spec.max_loss_distance, finish)
    longest = max((end_ - start_ for start_, end_ in intervals), default=0.0)

    if loss_started_at is not None:
        # Tracked distance runs to where resolvability actually failed, not to
        # where the failure was admitted: the samples in between are degraded
        # whether or not the instrument had noticed yet.
        tracked = loss_started_at - acquired_at
        # ... and the minimum "while tracked" must run over the same span. The
        # samples past the loss are exactly the ones the instrument was *not*
        # tracking through, and they are also the smallest, so including them
        # reports the depth of the failure as though it were a property of the
        # tracked stretch -- a number that is both wrong and unusable, since it
        # is always below the hold threshold by construction.
        tracked_span = tail_grid <= loss_started_at
        min_while_tracked = (
            float(np.min(tail_rho[tracked_span])) if bool(tracked_span.any()) else None
        )
        return outcome(
            TRACK_LOST,
            f"rho stayed below {spec.hold_threshold:g} for longer than the "
            f"tolerated {spec.max_loss_distance:g}, from {loss_started_at:g}"
            f" and detectable at {loss_declared_at:g}",
            tracked_distance=tracked,
            min_resolvability_while_tracked=min_while_tracked,
            loss_intervals=intervals,
            longest_loss=longest,
            loss_started_at=loss_started_at,
            track_loss_declared_at=loss_declared_at,
            **timings,
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
            tracked_distance=tracked,
            min_resolvability_while_tracked=float(np.min(tail_rho)),
            loss_intervals=intervals,
            longest_loss=longest,
            **timings,
        )

    return outcome(
        TRACKED,
        f"acquired at {acquired_at:g} and held for {tracked:g}",
        tracked_distance=tracked,
        min_resolvability_while_tracked=float(np.min(tail_rho)),
        loss_intervals=intervals,
        longest_loss=longest,
        **timings,
    )
