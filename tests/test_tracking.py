# SPDX-License-Identifier: MPL-2.0
"""The acquisition schedule, and the difference between an event and knowing it."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed import AcquisitionSpec
from geodesic_testbed.engine.tracking import evaluate_tracking

GRID = np.linspace(0.0, 10.0, 1001)


def _profile(rise: float = 2.0, collapse: float | None = 7.0) -> np.ndarray:
    """rho low, then high from ``rise``, then low again from ``collapse``."""
    rho = np.full(GRID.shape, 8.0)
    rho[np.less(GRID, rise)] = 0.5
    if collapse is not None:
        rho[np.greater(GRID, collapse)] = 0.2
    return rho


def _spec(**overrides) -> AcquisitionSpec:
    fields = {
        "acquire_threshold": 5.0,
        "hold_threshold": 3.0,
        "acquisition_window": 1.0,
        "max_acquisition_distance": 2.5,
        "max_loss_distance": 0.5,
    }
    return AcquisitionSpec(**(fields | overrides))


def test_acquisition_records_both_when_it_began_and_when_it_could_be_known() -> None:
    outcome = evaluate_tracking(GRID, _profile(), _spec(processing="offline"))
    # rho crosses at s = 2 and the window is 1.0, so a live instrument cannot
    # assert anything until s = 3.
    assert outcome.acquisition_window_started_at == pytest.approx(2.0, abs=0.02)
    assert outcome.acquisition_declared_at == pytest.approx(3.0, abs=0.02)


def test_a_causal_instrument_cannot_acquire_before_the_window_completes() -> None:
    """The same profile, read two ways, gives two different verdicts."""
    offline = evaluate_tracking(GRID, _profile(), _spec(processing="offline"))
    causal = evaluate_tracking(GRID, _profile(), _spec(processing="causal"))

    # Offline may localise the acquisition to where the run began.
    assert offline.acquisition_latency == pytest.approx(2.0, abs=0.02)
    # A causal instrument pays the whole window as latency...
    assert causal.acquisition_latency == pytest.approx(3.0, abs=0.02)
    # ... and that is what the declared 2.5 limit is applied to, so the route
    # the offline analysis calls acquired is one the instrument acquires late.
    assert offline.outcome != "LATE_ACQUISITION"
    assert causal.outcome == "LATE_ACQUISITION"
    assert causal.acquisition_window_started_at == pytest.approx(2.0, abs=0.02)


def test_track_loss_records_both_when_it_began_and_when_it_was_detectable() -> None:
    outcome = evaluate_tracking(GRID, _profile(), _spec(processing="offline"))
    assert outcome.outcome == "TRACK_LOST"
    # rho collapses just past s = 7 and the tolerated excursion is 0.5.
    assert outcome.loss_started_at == pytest.approx(7.0, abs=0.02)
    assert outcome.track_loss_declared_at == pytest.approx(7.5, abs=0.03)
    assert outcome.track_loss_declared_at > outcome.loss_started_at

    # Tracked distance runs to where resolvability actually failed: the samples
    # between the two are degraded whether or not the sensor had noticed.
    assert outcome.tracked_distance == pytest.approx(
        outcome.loss_started_at - outcome.acquisition_arclength, abs=0.02
    )


def test_an_excursion_inside_the_tolerance_is_not_a_loss() -> None:
    rho = np.full(GRID.shape, 8.0)
    rho[(GRID > 5.0) & (GRID < 5.2)] = 0.2
    rho[GRID < 0.5] = 0.5
    outcome = evaluate_tracking(GRID, rho, _spec(acquisition_window=0.1))
    assert outcome.outcome == "TRACKED"
    assert outcome.loss_started_at is None
    assert outcome.longest_loss == pytest.approx(0.2, abs=0.03)


def test_a_run_that_never_completes_the_window_is_never_acquired() -> None:
    rho = np.full(GRID.shape, 0.5)
    rho[(GRID > 4.0) & (GRID < 4.5)] = 8.0
    outcome = evaluate_tracking(GRID, rho, _spec(acquisition_window=1.0))
    assert outcome.outcome == "NEVER_ACQUIRED"
    assert outcome.acquisition_declared_at is None


def test_a_degenerate_grid_is_refused() -> None:
    rho = np.full(GRID.shape, 8.0)
    for grid, message in (
        (np.array([0.0]), "at least two samples"),
        (np.array([0.0, 1.0, 1.0, 2.0]), "strictly increasing"),
        (np.array([0.0, 2.0, 1.0, 3.0]), "strictly increasing"),
        (np.array([0.0, np.nan, 1.0, 2.0]), "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            evaluate_tracking(grid, np.full(grid.shape, 8.0), _spec())
    broken = rho.copy()
    broken[GRID > 5.0] = np.inf
    with pytest.raises(ValueError, match="resolvability must be finite"):
        evaluate_tracking(GRID, broken, _spec())
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_tracking(GRID.reshape(-1, 1), rho.reshape(-1, 1), _spec())


def test_a_schedule_with_a_negative_distance_is_refused() -> None:
    with pytest.raises(ValueError, match="max_acquisition_distance"):
        _spec(max_acquisition_distance=-1.0)
    with pytest.raises(ValueError, match="min_tracked_distance"):
        _spec(min_tracked_distance=float("nan"))
