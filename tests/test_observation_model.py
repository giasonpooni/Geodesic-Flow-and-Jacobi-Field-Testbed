"""H, R and rho: the transfer map as something an instrument could falsify."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed import ObservationModel
from geodesic_testbed.engine.envelope import integrate_path
from geodesic_testbed.engine.surfaces import sphere, torus

SIGMA_ALPHA = np.deg2rad(0.1)
SIGMA_MEASUREMENT = 25e-6 / 0.3
HEADING_ONLY = np.diag([0.0, SIGMA_ALPHA**2])


def _sphere_path(radius: float = 1.0, length: float = 4.0):
    return integrate_path(
        sphere(radius), u0=np.pi / 2, v0=0.0, heading=np.pi / 2,
        length=length, n_steps=2000,
    )


def test_rho_is_exactly_the_quantity_it_claims_to_be() -> None:
    """rho = |b| sigma_alpha / sigma_measurement, for a heading-only uncertainty."""
    envelope = _sphere_path()
    model = ObservationModel.transverse_only(SIGMA_MEASUREMENT)
    rho = model.resolvability(envelope, HEADING_ONLY)
    assert np.allclose(
        rho[:, 0], np.abs(envelope.jacobi_field) * SIGMA_ALPHA / SIGMA_MEASUREMENT
    )


def test_the_output_covariance_floors_at_the_metrology_noise() -> None:
    """At a focus the signal vanishes and Cov(y) is R: the instrument sees noise."""
    envelope = _sphere_path()
    model = ObservationModel.transverse_only(SIGMA_MEASUREMENT)
    covariance = model.covariance(envelope, HEADING_ONLY)
    at_focus = int(np.argmin(np.abs(envelope.jacobi_field[1:])) + 1)
    assert envelope.arc_length[at_focus] == pytest.approx(np.pi, abs=1e-2)
    assert covariance[at_focus, 0, 0] == pytest.approx(SIGMA_MEASUREMENT**2, rel=1e-3)
    assert np.all(covariance[:, 0, 0] >= SIGMA_MEASUREMENT**2 - 1e-18)


def test_a_focus_is_where_the_instrument_stops_distinguishing_headings() -> None:
    envelope = _sphere_path()
    model = ObservationModel.transverse_only(SIGMA_MEASUREMENT)
    span = model.unresolvable_span(envelope, HEADING_ONLY, threshold=3.0)
    assert not span["resolvable_everywhere"]
    assert span["min_at_arclength"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < span["fraction_unresolvable"] < 0.5


def test_rho_does_not_move_when_the_same_situation_is_drawn_twice_as_big() -> None:
    """The property a threshold on |b| cannot have: |b| has units."""
    profiles = []
    for radius in (1.0, 2.0):
        model = ObservationModel.transverse_only(SIGMA_MEASUREMENT * radius)
        envelope = _sphere_path(radius=radius, length=3.0 * radius)
        profiles.append(model.resolvability(envelope, HEADING_ONLY)[:, 0])
    assert np.allclose(profiles[0], profiles[1], rtol=1e-9, atol=1e-9)
    # and the dimensionful quantity really does move, so the test has teeth
    assert not np.allclose(
        _sphere_path(1.0, 3.0).jacobi_field, _sphere_path(2.0, 6.0).jacobi_field
    )


def test_a_better_instrument_resolves_more_of_the_path() -> None:
    envelope = _sphere_path()
    coarse = ObservationModel.transverse_only(SIGMA_MEASUREMENT)
    fine = ObservationModel.transverse_only(SIGMA_MEASUREMENT / 10.0)
    assert (
        fine.unresolvable_span(envelope, HEADING_ONLY, threshold=3.0)["fraction_unresolvable"]
        < coarse.unresolvable_span(envelope, HEADING_ONLY, threshold=3.0)[
            "fraction_unresolvable"
        ]
    )


def test_a_full_pose_instrument_reports_both_components() -> None:
    envelope = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=400
    )
    model = ObservationModel.full_pose(SIGMA_MEASUREMENT, 1e-4)
    covariance = model.covariance(envelope, np.diag([1e-8, SIGMA_ALPHA**2]))
    assert covariance.shape == (401, 2, 2)
    assert np.allclose(covariance, np.swapaxes(covariance, -1, -2))
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert model.resolvability(envelope, np.diag([1e-8, SIGMA_ALPHA**2])).shape == (401, 2)


def test_prediction_is_linear_in_the_starting_pose() -> None:
    envelope = _sphere_path()
    model = ObservationModel.transverse_only(SIGMA_MEASUREMENT)
    single = model.predict(envelope, (1e-3, 2e-3))
    assert np.allclose(model.predict(envelope, (2e-3, 4e-3)), 2.0 * single)
    assert np.allclose(single[:, 0], envelope.lateral_basis * 1e-3 + envelope.jacobi_field * 2e-3)


def test_a_mode_mismatch_is_refused_rather_than_compared() -> None:
    envelope = _sphere_path()  # ambient-euclidean-chord
    model = ObservationModel.transverse_only(
        SIGMA_MEASUREMENT, mode="intrinsic-surface-distance"
    )
    with pytest.raises(ValueError, match="convert one before predicting"):
        model.resolvability(envelope, HEADING_ONLY)


def test_a_bad_instrument_or_prior_is_refused() -> None:
    envelope = _sphere_path()
    with pytest.raises(ValueError):
        ObservationModel.transverse_only(0.0)
    with pytest.raises(ValueError):
        ObservationModel.transverse_only(float("nan"))
    model = ObservationModel.transverse_only(SIGMA_MEASUREMENT)
    for bad in (np.array([[1.0, 2.0], [2.0, 1.0]]), np.eye(3), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            model.covariance(envelope, bad)
    with pytest.raises(ValueError):
        ObservationModel(
            mode="ambient-euclidean-chord", matrix=np.eye(2),
            noise_covariance=np.eye(2), outputs=("only-one",),
        )
