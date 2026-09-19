"""Observation modes: the quantity a comparison is in, named and versioned."""

from __future__ import annotations

import pytest

from geodesic_testbed.engine.observation import (
    DEFAULT_MODE,
    MODES,
    catalogue,
    implemented_modes,
    mode,
)


def test_the_four_modes_are_declared_in_order_of_distance_from_the_model() -> None:
    assert list(MODES) == [
        "intrinsic-surface-distance",
        "ambient-euclidean-chord",
        "scanner-reconstructed-chord",
        "camera-image-residual",
    ]


def test_only_the_two_the_repository_can_compute_are_marked_implemented() -> None:
    assert implemented_modes() == ("intrinsic-surface-distance", "ambient-euclidean-chord")
    for identifier in ("scanner-reconstructed-chord", "camera-image-residual"):
        assert MODES[identifier].implemented is False
        assert MODES[identifier].version == 0


def test_implemented_modes_are_versioned_and_described() -> None:
    for identifier in implemented_modes():
        entry = mode(identifier)
        assert entry.version >= 1
        assert entry.quantity and entry.note


def test_the_default_is_the_one_the_model_predicts() -> None:
    assert DEFAULT_MODE == "intrinsic-surface-distance"
    assert mode(DEFAULT_MODE).implemented is True


def test_the_catalogue_round_trips_for_a_report() -> None:
    entries = catalogue()
    assert {entry["identifier"] for entry in entries} == set(MODES)
    assert all(set(entry) == {"identifier", "version", "quantity", "implemented", "note"}
               for entry in entries)


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(KeyError):
        mode("tape-measure")
