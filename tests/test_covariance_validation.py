# SPDX-License-Identifier: MPL-2.0
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
    _validated_covariance_stack,
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


@pytest.mark.parametrize("epsilon", [1e-14, 1e-13])
def test_filter_refuses_output_when_tolerated_input_asymmetry_is_amplified(epsilon):
    matrix = np.array([[1, 1 + epsilon], [1 - epsilon, 1]])
    original = matrix.copy()
    _validated_covariance(matrix)  # The input itself is within the declared tolerance.
    with pytest.raises(ValueError, match="(zero variance|symmetric)"):
        filtered_noise_covariance([[1, -1], [0, 1]], matrix)
    assert np.array_equal(matrix, original)


def test_transfer_refuses_any_invalid_covariance_in_a_sampled_output_stack():
    transfer = TransferMap(
        arc_length=np.array([0.0, 1.0]), a=np.array([1.0, 1.0]),
        a_rate=np.array([0.0, 0.0]), b=np.array([0.0, -1.0]),
        b_rate=np.array([1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="zero variance"):
        transfer.propagate_covariance([[1, 1 + 1e-14], [1 - 1e-14, 1]])


def test_exact_singular_cancellation_and_mixed_scale_output_stay_eligible():
    expected = np.array([[0, 0], [0, 1]])
    actual = filtered_noise_covariance([[1, -1], [0, 1]], [[1, 1], [1, 1]])
    assert np.array_equal(actual, expected)
    mixed = np.array([[1e-300, 1], [1, 1e300]])
    assert np.array_equal(filtered_noise_covariance(np.eye(2), mixed), mixed)
    stack = np.stack([expected, np.zeros((2, 2)), mixed])
    assert np.array_equal(_validated_covariance_stack(stack, "synthetic stack"), stack)


def test_nonzero_declared_variance_cannot_disappear_through_float_cancellation():
    matrix = [[4.0, 19.4], [19.4, 94.08999999999999]]
    _validated_covariance(matrix)
    with pytest.raises(ValueError, match="nonzero declared variance collapsed to zero"):
        filtered_noise_covariance([[9.7, -2.0]], matrix)


def test_false_zero_diagnostic_covers_broadcast_batches_and_exact_zero_operators():
    from geodesic_testbed.engine.transfer import _covariance_product

    covariance = np.array([[4.0, 19.4], [19.4, 94.08999999999999]])
    operators = np.array([[[0.0, 0.0]], [[9.7, -2.0]]])
    with pytest.raises(ValueError, match="nonzero declared variance collapsed to zero"):
        _covariance_product(operators, covariance, "synthetic stack")
    result = _covariance_product(np.zeros((2, 3, 2)), covariance, "zero operator stack")
    assert np.array_equal(result, np.zeros((2, 3, 3)))


@pytest.mark.parametrize("stack", [
    np.empty((0, 2, 2)), np.zeros((2, 0, 0)), np.zeros((2, 2, 3)),
])
def test_computed_covariance_stack_requires_nonempty_square_matrices(stack):
    with pytest.raises(ValueError, match="non-empty square"):
        _validated_covariance_stack(stack, "computed covariance")


def test_observation_covariance_validates_the_final_sum(monkeypatch):
    import geodesic_testbed.engine.observation_model as observation_module

    record = constant_curvature_trace(np.array([0.0, 1.0]), 0.0).as_transfer_record()
    model = _model(np.diag([0.0, 1.0]))
    # Isolate the final-addition boundary from the separately checked product.
    # This injected finite result must not be returned merely because adding
    # valid measurement noise did not overflow.
    monkeypatch.setattr(
        observation_module, "_covariance_product",
        lambda *args: np.array([[[0.0, 1e-14], [-1e-14, 0.0]]] * 2),
    )
    with pytest.raises(ValueError, match="zero variance"):
        model.covariance(record, np.eye(2))


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
