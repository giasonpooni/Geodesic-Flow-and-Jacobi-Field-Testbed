"""Observation modes: what a comparison is in, and where it can be produced."""

from __future__ import annotations

import pytest

from geodesic_testbed.engine.observation import (
    DEFAULT_MODE,
    DOMAINS,
    MODES,
    SUPPORT,
    available_modes,
    catalogue,
    mode,
)


def test_the_four_modes_are_declared_in_order_of_distance_from_the_model() -> None:
    assert list(MODES) == [
        "intrinsic-surface-distance",
        "ambient-euclidean-chord",
        "scanner-reconstructed-chord",
        "camera-image-residual",
    ]


def test_support_is_recorded_per_domain_not_globally() -> None:
    """The distinction a single flag could not carry."""
    intrinsic = mode("intrinsic-surface-distance")
    assert intrinsic.support_in("constant-curvature") == "exact"
    assert intrinsic.support_in("parametric-surface") == "unavailable"
    assert intrinsic.is_available("constant-curvature")
    assert not intrinsic.is_available("parametric-surface")


def test_what_each_domain_can_actually_produce() -> None:
    assert available_modes("constant-curvature") == (
        "intrinsic-surface-distance",
        "ambient-euclidean-chord",
    )
    assert available_modes("parametric-surface") == ("ambient-euclidean-chord",)
    assert available_modes("physical-instrument") == ()


def test_the_instrument_modes_are_declared_and_unimplemented() -> None:
    for identifier in ("scanner-reconstructed-chord", "camera-image-residual"):
        entry = mode(identifier)
        assert entry.version == 0
        assert not any(entry.is_available(domain) for domain in DOMAINS)


def test_implemented_modes_are_versioned_and_described() -> None:
    for identifier in ("intrinsic-surface-distance", "ambient-euclidean-chord"):
        entry = mode(identifier)
        assert entry.version >= 1
        assert entry.quantity and entry.note


def test_the_default_is_the_one_the_model_predicts() -> None:
    assert DEFAULT_MODE == "intrinsic-surface-distance"
    assert mode(DEFAULT_MODE).is_available("constant-curvature")


def test_the_catalogue_round_trips_for_a_report() -> None:
    entries = catalogue()
    assert {entry["identifier"] for entry in entries} == set(MODES)
    for entry in entries:
        assert set(entry) == {"identifier", "version", "quantity", "support", "note"}
        assert set(entry["support"]) == set(DOMAINS)
        assert set(entry["support"].values()) <= set(SUPPORT)


def test_an_unknown_mode_or_domain_is_refused() -> None:
    with pytest.raises(KeyError):
        mode("tape-measure")
    with pytest.raises(KeyError):
        available_modes("wind-tunnel")
    with pytest.raises(KeyError):
        mode("ambient-euclidean-chord").support_in("wind-tunnel")
