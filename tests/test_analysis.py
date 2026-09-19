"""Fitting helpers: they are load-bearing, so they get their own tests."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed.engine.analysis import fit_power_law, relative_error, successive_orders


def test_recovers_a_known_power_law() -> None:
    x = np.logspace(-4, 0, 21)
    fit = fit_power_law(x, 3.5 * x**2.25)
    assert fit.exponent == pytest.approx(2.25, abs=1e-9)
    assert fit.prefactor == pytest.approx(3.5, rel=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.n_used == 21


def test_a_roundoff_floor_is_excluded_instead_of_flattening_the_slope() -> None:
    x = np.logspace(-6, 0, 25)
    clean = 2.0 * x**4
    noisy = np.maximum(clean, 1e-15)
    unfiltered = fit_power_law(x, noisy)
    filtered = fit_power_law(x, noisy, y_floor=1e-13)
    assert unfiltered.exponent < 3.0
    assert filtered.exponent == pytest.approx(4.0, abs=1e-9)
    assert filtered.dropped_below_floor > 0
    assert filtered.n_used + filtered.dropped_below_floor == filtered.n_total


def test_a_ceiling_drops_the_samples_where_higher_orders_still_matter() -> None:
    x = np.logspace(-6, 0, 25)
    fit = fit_power_law(x, x**2 + x**4, y_ceiling=1e-4)
    assert fit.exponent == pytest.approx(2.0, abs=1e-3)
    assert fit.dropped_above_ceiling > 0


def test_too_few_usable_samples_gives_a_nan_fit_rather_than_a_wrong_one() -> None:
    fit = fit_power_law([1.0, 2.0], [0.0, 0.0], y_floor=1e-3)
    assert fit.n_used == 0
    assert np.isnan(fit.exponent)
    assert fit.x_min is None


def test_invert_finds_where_the_law_reaches_a_value() -> None:
    fit = fit_power_law(np.logspace(-4, 0, 9), 4.0 * np.logspace(-4, 0, 9) ** 2)
    assert fit.invert(4.0 * 0.01**2) == pytest.approx(0.01, rel=1e-9)


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError):
        fit_power_law([1.0, 2.0], [1.0])


def test_successive_orders_reads_off_each_refinement() -> None:
    x = np.array([0.2, 0.1, 0.05])
    orders = successive_orders(x, 3.0 * x**2)
    assert orders == pytest.approx([2.0, 2.0])
    assert successive_orders([0.2, 0.1], [0.0, 1.0]) == [None]


def test_relative_error_falls_back_to_absolute_at_a_zero_reference() -> None:
    assert relative_error([2.0], [1.0])[0] == pytest.approx(1.0)
    assert relative_error([0.25], [0.0])[0] == pytest.approx(0.25)
