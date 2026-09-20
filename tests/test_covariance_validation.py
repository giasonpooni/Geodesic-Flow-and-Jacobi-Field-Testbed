"""Adversarial covariance cases shared by the native observation/filter paths."""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed import ObservationModel, constant_curvature_trace
from geodesic_testbed.engine.observation_model import (
    FilteredPrediction,
    filtered_noise_covariance,
)
from geodesic_testbed.engine.transfer import (
    TransferMap,
    _validated_covariance,
    constant_curvature_transfer,
)


def _model(matrix):
    return ObservationModel(
        mode="intrinsic-surface-distance", matrix=np.eye(2),
        noise_covariance=matrix, outputs=("transverse", "heading"),
    )


def _prediction(matrix):
    return FilteredPrediction(
        values=np.zeros(2), identifier="synthetic-identity", version="1",
        operator_digest="synthetic:identity", causal=True, noise_covariance=matrix,
    )


def _transfer(matrix):
    return constant_curvature_transfer(np.array([0.0]), 0.0).propagate_covariance(matrix)


def _filter(matrix):
    return filtered_noise_covariance(np.eye(2), matrix)


GATES = [_transfer, _model, _filter, _prediction]


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("matrix", [
    [[-1e-30, 0], [0, 1]],
    [[1e-30, 2e-30], [2e-30, 1e-30]],
    [[1e-300, 2], [2, 1e300]],
    [[0, 1e-300], [1e-300, 1]],
    [[0, 0], [1e-300, 1]],
    [[1e-300, 0.1], [0.9, 1e300]],
    [[1, 1 + 1.5e-12], [1 + 0.6e-12, 1]],
    [[True, 0], [0, 1]],
    [["1", 0], [0, 1]],
    [[1 + 0j, 0], [0, 1]],
    [[float("nan"), 0], [0, 1]],
    [[float("inf"), 0], [0, 1]],
    [[10**1000, 0], [0, 1]],
    [[5e-324, 1e308], [1e308, 5e-324]],
])
def test_all_covariance_paths_refuse_the_same_invalid_claim(gate, matrix):
    with pytest.raises(ValueError):
        gate(matrix)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("matrix", [
    [[0, 0], [0, 0]],
    [[0, 0], [0, 1e-300]],
    [[1e-300, 0.5], [0.5, 1e300]],
    [[1e-300, 1], [1, 1e300]],
    [[1e-14, 1e-7], [1e-7, 1]],
    [[5e-324, 0], [0, 5e-324]],
])
def test_singular_and_extreme_mixed_unit_covariance_remain_valid(gate, matrix):
    gate(matrix)


@pytest.mark.parametrize("variance", [-1e-30, True, "0.1", float("nan"), 10**1000])
def test_scalar_and_vector_filter_noise_cannot_bypass_the_matrix_gate(variance):
    with pytest.raises(ValueError):
        filtered_noise_covariance(np.eye(2), variance)
    with pytest.raises(ValueError):
        filtered_noise_covariance(np.eye(2), [variance, 1.0])


def test_empty_covariance_and_empty_filter_are_refused_cleanly():
    for covariance in ([], np.zeros((0, 0))):
        with pytest.raises(ValueError, match="non-empty"):
            _validated_covariance(covariance, size=None)
    with pytest.raises(ValueError):
        filtered_noise_covariance(np.zeros((0, 2)), np.eye(2))


def test_normalization_checks_both_triangles_without_averaging_or_mutation():
    matrix = np.array([[1, 1 + 1.5e-12], [1 + 0.6e-12, 1]])
    original = matrix.copy()
    with pytest.raises(ValueError, match="positive semidefinite"):
        _validated_covariance(matrix)
    assert np.array_equal(matrix, original)


def test_nonconvergence_fails_closed(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic nonconvergence")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    with pytest.raises(ValueError, match="did not converge"):
        _validated_covariance(np.eye(2))


@pytest.mark.parametrize("sigma", [True, "0.1", 1e-300, 1e200, 10**1000])
def test_sigma_constructors_refuse_boolean_overflow_and_erased_variance(sigma):
    with pytest.raises(ValueError):
        ObservationModel.transverse_only(sigma)
    with pytest.raises(ValueError):
        ObservationModel.full_pose(1.0, sigma)


def test_filter_and_transfer_covariance_overflow_are_explicit_refusals():
    with pytest.raises(ValueError, match="finite"):
        filtered_noise_covariance(np.array([[1e308]]), [[1e308]])
    transfer = TransferMap(
        arc_length=np.array([0.0]), a=np.array([1e308]), a_rate=np.array([0.0]),
        b=np.array([0.0]), b_rate=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="finite"):
        transfer.propagate_covariance(np.eye(2))


def test_filter_cannot_underflow_nonzero_uncertainty_into_false_certainty():
    with pytest.raises(ValueError, match="floating-point range"):
        filtered_noise_covariance([[1e-200]], 1.0)


def test_extended_precision_variance_cannot_be_erased_by_float64_conversion():
    if np.finfo(np.longdouble).tiny >= np.finfo(float).tiny:
        pytest.skip("extended precision is unavailable on this platform")
    matrix = np.array([[np.longdouble("1e-400"), 0], [0, 1]], dtype=np.longdouble)
    with pytest.raises(ValueError):
        _validated_covariance(matrix)


def test_negative_computed_variance_is_not_clipped_after_projection():
    # A near-PSD matrix within the declared numerical tolerance can become
    # negative along a cancellation direction. That is a refusal, not repair.
    with pytest.raises(ValueError, match="negative computed variance"):
        filtered_noise_covariance([[1, -1]], [[1, 1 + 0.5e-12], [1 + 0.5e-12, 1]])


def test_input_arrays_are_copied_and_mutated_model_covariance_is_rechecked():
    matrix = np.eye(2)
    model, prediction = _model(matrix), _prediction(matrix)
    assert matrix.flags.writeable
    matrix[0, 0] = -1
    assert model.noise_covariance[0, 0] == 1
    assert prediction.noise_covariance[0, 0] == 1
    model.noise_covariance.setflags(write=True)
    model.noise_covariance[0, 0] = -1e-30
    record = constant_curvature_trace(np.array([0.0, 1.0]), 0.0).as_transfer_record()
    with pytest.raises(ValueError, match="negative variance"):
        model.covariance(record, np.eye(2))


def test_singular_noise_is_valid_but_zero_noise_resolvability_is_undefined():
    model = _model(np.diag([0.0, 1.0]))
    record = constant_curvature_trace(np.array([0.0, 1.0]), 0.0).as_transfer_record()
    assert model.covariance(record, np.eye(2)).shape == (2, 2, 2)
    with pytest.raises(ValueError, match="positive noise variance"):
        model.resolvability(record, np.eye(2))


def test_seeded_covariances_match_dimensionless_numpy_oracle_under_permutation():
    generator = np.random.default_rng(20260920)
    for trial in range(200):
        size = int(generator.choice([1, 2, 3, 5, 8, 16]))
        factor = generator.normal(size=(size, int(generator.integers(1, size + 1))))
        covariance = factor @ factor.T
        roots = np.sqrt(np.diag(covariance))
        correlation = covariance / roots[:, None] / roots[None, :]
        if trial % 3 == 0 and size > 1:
            raw = generator.uniform(-0.99, 0.99, size=(size, size))
            correlation = (raw + raw.T) * 0.5
            np.fill_diagonal(correlation, 1.0)
        scale = 10.0 ** generator.uniform(-150, 150, size=size)
        matrix = correlation * scale[:, None] * scale[None, :]
        expected = np.linalg.eigvalsh(correlation)[0] >= -1e-12
        for permutation in (np.arange(size), generator.permutation(size)):
            permuted = matrix[np.ix_(permutation, permutation)]
            if expected:
                assert np.array_equal(_validated_covariance(permuted, size=size), permuted)
            else:
                with pytest.raises(ValueError, match="positive semidefinite"):
                    _validated_covariance(permuted, size=size)
