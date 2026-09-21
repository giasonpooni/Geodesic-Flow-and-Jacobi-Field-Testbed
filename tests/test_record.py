# SPDX-License-Identifier: MPL-2.0
"""The public transfer record: one type both halves of the project speak."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed import constant_curvature_trace
from geodesic_testbed.engine.envelope import integrate_path
from geodesic_testbed.engine.record import (
    RECORD_SCHEMA,
    SupportsTransferRecord,
    TransferRecord,
    Units,
    ValidityEnvelope,
    digest,
    to_transfer_record,
)
from geodesic_testbed.engine.surfaces import sphere, torus

GRID = np.linspace(0.0, 1.25, 126)


def _trace_record():
    return constant_curvature_trace(GRID, -1.0).as_transfer_record()


def _surface_record():
    return integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=400
    ).as_transfer_record()


def test_both_producers_satisfy_the_protocol() -> None:
    for record in (_trace_record(), _surface_record()):
        assert isinstance(record, SupportsTransferRecord)
        assert to_transfer_record(record) is record
        assert record.to_dict(include_samples=False)["schema"] == RECORD_SCHEMA


def test_a_record_reproduces_the_map_it_came_from() -> None:
    envelope = integrate_path(
        sphere(1.0), u0=np.pi / 2, v0=0.0, heading=0.6, length=1.5, n_steps=300
    )
    record = envelope.as_transfer_record()
    assert np.array_equal(record.a, envelope.lateral_basis)
    assert np.array_equal(record.b, envelope.jacobi_field)
    assert np.array_equal(record.arclength, envelope.arc_length)


def test_the_constant_curvature_record_declares_a_real_validity_bound() -> None:
    """sqrt(24 tol)/max|cn_K| is computed, not asserted."""
    record = constant_curvature_trace(GRID, 1.0).as_transfer_record(relative_tolerance=1e-6)
    assert record.validity.established
    worst = float(np.max(np.abs(np.cos(GRID))))
    assert record.validity.max_heading == pytest.approx(np.sqrt(24e-6) / worst)


def test_a_varying_profile_refuses_to_invent_a_validity_bound() -> None:
    record = _surface_record()
    assert not record.validity.established
    assert record.validity.max_heading is None
    assert "no closed form" in record.validity.basis


def test_the_surface_record_does_not_claim_an_intrinsic_distance() -> None:
    """Nothing here solves the boundary-value problem, so nothing claims it."""
    assert _surface_record().observation_mode == "ambient-euclidean-chord"
    assert _trace_record().observation_mode == "intrinsic-surface-distance"


def test_digests_track_geometry_not_resolution() -> None:
    coarse = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=100
    ).as_transfer_record()
    fine = _surface_record()
    assert coarse.source_digest == fine.source_digest
    assert coarse.resolution.samples != fine.resolution.samples
    other = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.9, length=2.0, n_steps=400
    ).as_transfer_record()
    assert other.source_digest != fine.source_digest
    assert digest({"a": 1}) != digest({"a": 2})


def test_the_two_error_bounds_are_the_two_columns() -> None:
    record = _trace_record()
    lateral, heading = 2e-3, 5e-4
    assert np.allclose(
        record.cross_track_error(lateral, heading),
        np.abs(record.a) * lateral + np.abs(record.b) * heading,
    )
    assert np.allclose(
        record.heading_error(lateral, heading),
        np.abs(record.a_rate) * lateral + np.abs(record.b_rate) * heading,
    )


def test_the_scaled_transfer_is_dimensionless_and_reciprocal() -> None:
    """det Phi = 1 survives conjugation, so no path contracts everything."""
    for record in (_trace_record(), _surface_record()):
        for lateral, heading in ((1e-3, 1e-3), (1e-1, 1e-6)):
            singular = record.scaled_singular_values(lateral, heading)
            assert np.allclose(singular[:, 0] * singular[:, 1], 1.0, atol=1e-11)
            assert np.min(singular[:, 0]) >= 1.0 - 1e-12
            assert record.amplification_score(lateral, heading) == pytest.approx(
                float(np.max(singular[:, 0]))
            )


def test_scaling_the_tolerance_box_is_what_makes_paths_comparable() -> None:
    """Doubling both tolerances cannot change a dimensionless score."""
    record = _surface_record()
    assert record.amplification_score(2e-3, 2e-3) == pytest.approx(
        record.amplification_score(1e-3, 1e-3)
    )
    with pytest.raises(ValueError):
        record.scaled_transfer(0.0, 1e-3)


def test_a_malformed_record_is_refused() -> None:
    good = _trace_record()
    for broken, _reason in (
        ({"a": good.a[:-1]}, "ragged"),
        ({"arclength": good.arclength[::-1]}, "decreasing"),
        ({"b": np.where(np.arange(good.b.size) == 3, np.nan, good.b)}, "nan"),
    ):
        fields = {
            "arclength": good.arclength,
            "gaussian_curvature": good.gaussian_curvature,
            "a": good.a,
            "a_rate": good.a_rate,
            "b": good.b,
            "b_rate": good.b_rate,
        } | broken
        with pytest.raises(ValueError):
            TransferRecord(**fields)
    with pytest.raises(KeyError):
        TransferRecord(
            arclength=good.arclength, gaussian_curvature=good.gaussian_curvature,
            a=good.a, a_rate=good.a_rate, b=good.b, b_rate=good.b_rate,
            observation_mode="tape-measure",
        )


def test_units_are_carried_rather_than_assumed() -> None:
    record = _trace_record()
    assert record.units == Units()
    millimetres = constant_curvature_trace(GRID, 1.0).as_transfer_record(
        units=Units(length="mm")
    )
    assert millimetres.units.length == "mm"
    assert millimetres.to_dict(include_samples=False)["units"]["length"] == "mm"


def test_not_established_validity_says_so_in_both_places() -> None:
    validity = ValidityEnvelope.not_established("no reference")
    assert not validity.established
    assert validity.to_dict()["established"] is False


def test_something_that_is_not_a_record_is_refused() -> None:
    with pytest.raises(TypeError):
        to_transfer_record(object())
