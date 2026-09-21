"""The four terms of ``Sigma_y``, and the cases a scalar residual cannot see.

The tests that matter here are the ones where a wrong answer looks fine. A
shared calibration error recorded as per-sample noise gives the right diagonal
and the wrong everything else; a covariance inflated twice over passes any
one-sided test ever written; a chi-square that lands in the middle of its band
can still have every residual in the wrong place. Each of those has a test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geodesic_testbed.engine.contract import StartingCovariance, Units
from geodesic_testbed.engine.envelope import integrate_paths
from geodesic_testbed.engine.observation_model import ObservationModel
from geodesic_testbed.engine.output_covariance import (
    BLOCKS,
    NoiseModel,
    OutputCovariance,
    SharedParameters,
    assemble,
    chi_square_cdf,
    chi_square_quantile,
    registration_parameter,
    stacked_operator,
)
from geodesic_testbed.engine.surfaces import sphere

UNITS = Units(length="metre", angle="radian")
TOLERANCE = np.diag([2e-4**2, np.deg2rad(0.1) ** 2])


def _record(n_steps: int = 60, length: float = 1.2):
    envelope = integrate_paths(
        sphere(1.0), u0=np.pi / 2, v0=0.0, headings=[0.6], length=length, n_steps=n_steps
    )[0]
    return envelope.as_transfer_record(
        units=UNITS,
        observation_mode="ambient-euclidean-chord",
        covariance=StartingCovariance(
            matrix=TOLERANCE,
            basis="declared-tolerance-box",
            units=UNITS,
            coverage_factor=3.0,
            note="a declared fixturing tolerance",
        ),
    )


def _model(sigma: float = 5e-5) -> ObservationModel:
    return ObservationModel.transverse_only(sigma, mode="ambient-euclidean-chord")


def _noise(model: ObservationModel, **overrides) -> NoiseModel:
    return NoiseModel.from_observation_model(
        model, basis=overrides.pop("basis", "bench characterisation against a nominal"), **overrides
    )


# -- the operator ----------------------------------------------------------


def test_the_stacked_operator_is_the_per_sample_one_written_out() -> None:
    record = _record()
    model = _model()
    operator = stacked_operator(record, model)
    assert operator.shape == (record.arclength.size * len(model.outputs), 2)

    phi = record.transfer_map().matrices()
    for index in (0, 7, record.arclength.size - 1):
        rows = slice(index * len(model.outputs), (index + 1) * len(model.outputs))
        assert np.allclose(operator[rows], model.matrix @ phi[index])


# -- the four blocks -------------------------------------------------------


def test_the_total_is_the_sum_of_the_blocks_it_names() -> None:
    record = _record()
    model = _model()
    total = assemble(record, model, noise=_noise(model))
    assert set(total.blocks) <= set(BLOCKS)
    assert np.allclose(total.total, sum(total.blocks.values()))
    assert total.degrees_of_freedom == record.arclength.size
    assert abs(sum(total.shares().values()) - 1.0) < 1e-12


def test_the_starting_pose_block_is_what_the_runtime_already_computed() -> None:
    """The stacked form has to agree with the per-sample one, or it is a new answer."""
    record = _record()
    model = _model()
    total = assemble(record, model, noise=_noise(model))
    stacked = total.blocks["starting-pose"]
    per_sample = model.matrix @ record.propagate_covariance(TOLERANCE) @ model.matrix.T
    for index in range(record.arclength.size):
        assert np.allclose(stacked[index, index], per_sample[index, 0, 0])


def test_a_shared_parameter_puts_variance_off_the_diagonal() -> None:
    """The whole point. One unknown offset, felt at every sample, correlates them.

    An independent per-sample term with the same diagonal would average away
    along the path; this does not, and the difference is visible only in the
    blocks a per-sample stack discards.
    """
    record = _record()
    model = _model()
    parameters = registration_parameter(
        record, model, 2e-4, basis="declared registration repeatability"
    )
    block = parameters.contribution()
    assert parameters.stacked_jacobian().shape == (record.arclength.size, 1)

    off_diagonal = block[5, 20]
    assert abs(off_diagonal) > 0.0
    assert np.isclose(off_diagonal, parameters.cross_sample_block(5, 20)[0, 0])

    correlation = off_diagonal / math.sqrt(block[5, 5] * block[20, 20])
    assert abs(correlation) > 0.999, (
        "a single shared unknown makes the samples nearly perfectly correlated; "
        f"this one reads {correlation:.6f}"
    )
    assert np.linalg.matrix_rank(block, tol=1e-12 * float(np.max(np.abs(block)))) == 1


def test_a_shared_parameter_and_an_independent_one_differ_only_off_the_diagonal() -> None:
    """Which is exactly why keeping only the diagonal loses the distinction.

    Build an independent term with the *same* per-sample variance and ask the
    two of them about two residuals. A common-mode excursion -- every sample
    displaced together, which is what a registration offset does -- is cheap
    against the correlated covariance and expensive against the diagonal one.
    A differential excursion, alternating sample to sample, is the other way
    round by four orders of magnitude.

    Neither covariance can tell them apart on the diagonal, and the whole
    difference between a bias and noise is in the blocks a per-sample stack
    throws away.
    """
    record = _record()
    model = _model()
    shared = registration_parameter(record, model, 2e-4, basis="declared repeatability")
    block = shared.contribution()
    independent = np.diag(np.diag(block))
    assert np.allclose(np.diag(block), np.diag(independent))

    # A small isotropic floor so both are invertible; a rank-one covariance is
    # singular by construction, and correctly so.
    floor = 1e-12 * float(np.max(np.diag(block))) * np.eye(block.shape[0])
    correlated = OutputCovariance(
        arclength=record.arclength,
        outputs=model.outputs,
        blocks={"shared-parameters": block, "observation-noise": floor},
    )
    diagonal = OutputCovariance(
        arclength=record.arclength,
        outputs=model.outputs,
        blocks={"shared-parameters": independent, "observation-noise": floor},
    )

    common_mode = shared.stacked_jacobian()[:, 0] * 1.0
    alternating = np.sqrt(np.diag(block)) * (-1.0) ** np.arange(block.shape[0])

    moved = int(np.count_nonzero(shared.stacked_jacobian()[:, 0]))
    assert moved == record.arclength.size - 1, "only s = 0 should be exempt"

    common_correlated = correlated.nis(common_mode)["statistic"]
    common_diagonal = diagonal.nis(common_mode)["statistic"]
    assert common_correlated == pytest.approx(1.0 / 2e-4**2, rel=1e-6)
    assert common_diagonal / common_correlated == pytest.approx(moved, rel=1e-6), (
        "the overcharge for a common-mode excursion is the number of samples the "
        f"parameter moves; {common_diagonal / common_correlated:.6f} against {moved}"
    )

    differential_correlated = correlated.nis(alternating)["statistic"]
    differential_diagonal = diagonal.nis(alternating)["statistic"]
    assert differential_correlated > 1e4 * differential_diagonal, (
        "and sample-to-sample scatter is what a shared offset cannot produce, so "
        "the correlated covariance charges for it and the diagonal one does not: "
        f"{differential_correlated:.3g} against {differential_diagonal:.3g}"
    )


def test_the_numerical_term_is_kept_apart_from_the_sensor_noise() -> None:
    record = _record()
    model = _model()
    floor = np.full(record.arclength.size, (1e-9) ** 2)
    total = assemble(
        record, model, noise=_noise(model),
        numerical=floor, numerical_basis="deterministic-bound",
    )
    assert "numerical" in total.blocks
    assert np.allclose(np.diag(total.blocks["numerical"]), floor)
    assert total.shares()["numerical"] > 0.0


def test_a_numerical_term_must_say_which_kind_of_quantity_it_is() -> None:
    """The arithmetic is identical; the meaning is not.

    A Richardson estimate of a truncation error is deterministic -- the error
    is whatever it is, and there is no sampling story behind it. Adding it to
    a covariance is the only way to combine it with the other terms, and that
    is exactly why it has to be labelled: only a probabilistic term makes the
    resulting chi-square a calibrated tail.
    """
    record = _record(n_steps=20)
    model = _model()
    floor = np.full(record.arclength.size, (1e-9) ** 2)
    with pytest.raises(ValueError, match="deterministic bound or a probabilistic"):
        assemble(record, model, noise=_noise(model), numerical=floor)

    bounded = assemble(
        record, model, noise=_noise(model),
        numerical=floor, numerical_basis="deterministic-bound",
    )
    statistic = bounded.nis(np.zeros(bounded.degrees_of_freedom))
    assert statistic["calibrated"] is False
    assert "conservative by an unknown amount" in statistic["note"]

    modelled = assemble(
        record, model, noise=_noise(model),
        numerical=floor, numerical_basis="probabilistic",
    )
    assert modelled.nis(np.zeros(modelled.degrees_of_freedom))["calibrated"] is True

    # With no numerical block at all there is nothing to qualify.
    plain = assemble(record, model, noise=_noise(model))
    assert plain.nis(np.zeros(plain.degrees_of_freedom))["calibrated"] is True


# -- the noise model -------------------------------------------------------


def test_a_noise_model_can_be_stationary_per_sample_or_correlated() -> None:
    record = _record()
    n = record.arclength.size
    stationary = NoiseModel.stationary(
        np.array([[4e-10]]), ("transverse",), basis="bench characterisation"
    )
    assert stationary.stacked(n).shape == (n, n)
    assert np.allclose(np.diag(stationary.stacked(n)), 4e-10)

    varying = NoiseModel(
        blocks=np.array([[[value]] for value in np.linspace(1e-10, 9e-10, n)]),
        structure="per-sample",
        outputs=("transverse",),
        basis="characterised per station",
    )
    assert varying.samples == n
    spread = np.diag(varying.stacked(n))
    assert spread.max() / spread.min() == pytest.approx(9.0, rel=1e-12), (
        "a per-sample R is not a stationary one; this is the check that would "
        "fail if stacked() had quietly repeated the first block"
    )

    lag = np.exp(-np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / 6.0)
    correlated = NoiseModel(
        blocks=4e-10 * lag,
        structure="correlated",
        outputs=("transverse",),
        basis="a declared filter group delay",
    )
    assert correlated.samples == n
    assert correlated.stacked(n)[0, 1] > 0.0


def test_a_noise_model_on_the_wrong_grid_is_refused() -> None:
    """An R paired with the wrong arc lengths is worse than no R."""
    record = _record()
    short = NoiseModel(
        blocks=np.full((7, 1, 1), 4e-10),
        structure="per-sample",
        outputs=("transverse",),
        basis="characterised on a shorter run",
    )
    with pytest.raises(ValueError, match="characterised on 7 samples"):
        assemble(record, _model(), noise=short)


def test_a_noise_model_must_say_how_it_was_arrived_at() -> None:
    with pytest.raises(ValueError, match="how it was arrived at"):
        NoiseModel.stationary(np.array([[1e-9]]), ("transverse",), basis="")


# -- the thing that looks like caution and is not --------------------------


def test_an_uncertainty_declared_twice_is_refused_rather_than_added_twice() -> None:
    record = _record()
    model = _model()
    calibrated = _noise(
        model,
        basis="characterised on the bench against a traceable artefact",
        accounts_for=("calibration-transform",),
    )
    parameters = SharedParameters(
        names=("calibration-scale",),
        kinds=("calibration-transform",),
        covariance=np.array([[1e-8]]),
        jacobian=np.ones((record.arclength.size, 1, 1)),
        basis="a certificate",
    )
    with pytest.raises(ValueError, match="counted twice"):
        assemble(record, model, parameters=parameters, noise=calibrated)

    uncalibrated = _noise(model, basis="characterised against a nominal")
    total = assemble(record, model, parameters=parameters, noise=uncalibrated)
    assert "shared-parameters" in total.blocks


def test_a_missing_starting_covariance_is_an_error_and_not_a_substitution() -> None:
    envelope = integrate_paths(
        sphere(1.0), u0=np.pi / 2, v0=0.0, headings=[0.6], length=1.0, n_steps=40
    )[0]
    record = envelope.as_transfer_record(
        units=UNITS, observation_mode="ambient-euclidean-chord"
    )
    assert not record.covariance.declared
    model = _model()
    with pytest.raises(ValueError, match="no starting covariance"):
        assemble(record, model, noise=_noise(model))

    total = assemble(record, model, starting_covariance=TOLERANCE, noise=_noise(model))
    assert total.degrees_of_freedom == record.arclength.size


def test_a_covariance_assembled_across_two_observation_modes_is_refused() -> None:
    record = _record()
    other = ObservationModel.transverse_only(5e-5, mode="first-order-tangent-separation")
    with pytest.raises(ValueError, match="two different quantities"):
        assemble(record, other, noise=_noise(other))


# -- the statistics --------------------------------------------------------


def test_a_residual_drawn_from_the_declared_covariance_lands_in_its_band() -> None:
    record = _record(n_steps=40)
    model = _model()
    total = assemble(record, model, noise=_noise(model))

    generator = np.random.default_rng(20260920)
    factor = np.linalg.cholesky(total.total)
    verdicts = []
    for _ in range(200):
        residual = factor @ generator.standard_normal(total.degrees_of_freedom)
        verdicts.append(total.accepts(residual, coverage=0.95)["accepted"])
    rate = float(np.mean(verdicts))
    assert 0.90 < rate < 0.99, f"acceptance rate {rate:.3f} at a declared coverage of 0.95"


def test_an_inflated_covariance_is_rejected_by_the_lower_limit() -> None:
    """The failure a one-sided test was never going to catch.

    A budget four times too large covers every residual it will ever see. Only
    the lower limit makes an overstated uncertainty falsifiable, and overstating
    is the direction a budget is usually wrong in.
    """
    record = _record(n_steps=40)
    model = _model()
    honest = assemble(record, model, noise=_noise(model))

    generator = np.random.default_rng(7)
    factor = np.linalg.cholesky(honest.total)
    residual = factor @ generator.standard_normal(honest.degrees_of_freedom)
    assert honest.accepts(residual)["accepted"]

    inflated = OutputCovariance(
        arclength=record.arclength,
        outputs=model.outputs,
        blocks={name: 16.0 * block for name, block in honest.blocks.items()},
    )
    outcome = inflated.accepts(residual)
    assert not outcome["accepted"]
    assert outcome["verdict"] == "lower-tail-inconsistent"


def test_a_residual_too_large_is_rejected_by_the_upper_limit() -> None:
    record = _record(n_steps=40)
    model = _model()
    total = assemble(record, model, noise=_noise(model))
    generator = np.random.default_rng(11)
    factor = np.linalg.cholesky(total.total)
    residual = 4.0 * (factor @ generator.standard_normal(total.degrees_of_freedom))
    outcome = total.accepts(residual)
    assert not outcome["accepted"]
    assert outcome["verdict"] == "upper-tail-inconsistent"


def test_interval_coverage_catches_what_the_chi_square_averages_away() -> None:
    """A few residuals far out and the rest far in sum to the right total.

    NIS asks one question of the whole vector. Interval coverage asks ``n m``
    questions of its components, and a covariance can pass the first while
    being badly wrong sample by sample.
    """
    record = _record(n_steps=40)
    model = _model()
    total = assemble(record, model, noise=_noise(model))
    dof = total.degrees_of_freedom

    whitened = np.zeros(dof)
    heavy = dof // 8
    whitened[:heavy] = math.sqrt(dof / heavy)
    residual = np.linalg.cholesky(total.total) @ whitened

    statistic = total.nis(residual)
    assert abs(statistic["reduced"] - 1.0) < 1e-9, "by construction the chi-square is perfect"

    coverage = total.interval_coverage(residual, sigmas=1.0)
    assert coverage["observed"] > 0.8
    assert coverage["expected"] == pytest.approx(0.6826894921, abs=1e-9)
    assert coverage["difference"] > 0.1, (
        "the components are nothing like standard normal and the chi-square could "
        "not tell, which is the whole reason both are reported"
    )


def test_a_singular_total_says_what_is_missing_rather_than_failing_obscurely() -> None:
    record = _record(n_steps=20)
    model = _model()
    shared = registration_parameter(record, model, 2e-4, basis="declared repeatability")
    systematic_only = OutputCovariance(
        arclength=record.arclength,
        outputs=model.outputs,
        blocks={"shared-parameters": shared.contribution()},
    )
    with pytest.raises(ValueError, match="purely systematic"):
        systematic_only.whiten(np.zeros(record.arclength.size))


def test_whitening_refuses_a_residual_of_the_wrong_length() -> None:
    record = _record(n_steps=20)
    model = _model()
    total = assemble(record, model, noise=_noise(model))
    with pytest.raises(ValueError, match="entries and the stacked vector"):
        total.whiten(np.zeros(3))


# -- the chi-square itself, because the band is a declared limit ------------


@pytest.mark.parametrize(
    ("dof", "probability", "expected"),
    (
        (1, 0.95, 3.841458820694124),
        (2, 0.95, 5.991464547107979),
        (5, 0.05, 1.1454762260617692),
        (10, 0.975, 20.483177350806727),
        (40, 0.025, 24.433039163736567),
        (100, 0.5, 99.33412923598282),
    ),
)
def test_the_chi_square_quantile_matches_published_values(
    dof: int, probability: float, expected: float
) -> None:
    """Written out rather than approximated, so it is checked against tables.

    An acceptance band is a declared limit. A limit computed from a fit whose
    error nobody bounded is the thing this repository refuses everywhere else.
    """
    assert chi_square_quantile(probability, dof) == pytest.approx(expected, rel=1e-9)


def test_the_cumulative_distribution_and_its_inverse_agree() -> None:
    for dof in (1, 3, 17, 250):
        for probability in (0.01, 0.25, 0.5, 0.9, 0.999):
            quantile = chi_square_quantile(probability, dof)
            assert chi_square_cdf(quantile, dof) == pytest.approx(probability, abs=1e-10)


def _closed_form_cdf_for_even_dof(x: float, dof: int) -> float:
    """``1 - e^{-x/2} sum_{k < dof/2} (x/2)^k / k!``, valid only for even ``dof``.

    A second formula rather than a second tolerance. Round-tripping the
    quantile through its own CDF proves the two are inverses and nothing about
    whether either is the chi-square, which is exactly the failure that let a
    wrong table constant into this file in the first place.
    """
    half = 0.5 * x
    term = 1.0
    total = 1.0
    for k in range(1, dof // 2):
        term *= half / k
        total += term
    return 1.0 - math.exp(-half) * total


@pytest.mark.parametrize("dof", (2, 4, 10, 40, 100))
def test_the_series_agrees_with_the_closed_form_wherever_one_exists(dof: int) -> None:
    for x in (0.5, 1.0, float(dof), 2.0 * dof, 4.0 * dof):
        assert chi_square_cdf(x, dof) == pytest.approx(
            _closed_form_cdf_for_even_dof(x, dof), abs=1e-12
        )


# -- fail-closed, in every entry point -------------------------------------

#: The two matrices a review found accepted. Neither is a covariance; the
#: first has two stored triangles that disagree, and the second a negative
#: variance small enough to sit inside an eigenvalue floor scaled to the
#: largest entry. Both used to be repaired into something plausible.
NOT_COVARIANCES = (
    ("asymmetric", np.array([[1.0, 0.2], [0.1, 1.0]]), "symmetric"),
    ("negative variance", np.diag([1.0, -1e-12]), "negative variance"),
    ("nonfinite", np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
    ("indefinite", np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
)


@pytest.mark.parametrize(("label", "matrix", "message"), NOT_COVARIANCES)
def test_a_noise_model_refuses_what_is_not_a_covariance(
    label: str, matrix: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NoiseModel.stationary(matrix, ("a", "b"), basis="a bench characterisation")


@pytest.mark.parametrize(("label", "matrix", "message"), NOT_COVARIANCES)
def test_a_shared_parameter_block_refuses_what_is_not_a_covariance(
    label: str, matrix: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SharedParameters(
            names=("one", "two"),
            kinds=("other", "other"),
            covariance=matrix,
            jacobian=np.ones((4, 1, 2)),
            basis="a certificate",
        )


@pytest.mark.parametrize(("label", "matrix", "message"), NOT_COVARIANCES)
def test_the_numerical_block_refuses_what_is_not_a_covariance(
    label: str, matrix: np.ndarray, message: str
) -> None:
    record = _record(n_steps=1)
    model = ObservationModel.full_pose(2e-4, 1e-3, mode="ambient-euclidean-chord")
    noise = NoiseModel.from_observation_model(model, basis="bench characterisation")
    with pytest.raises(ValueError, match=message):
        assemble(record, model, noise=noise, numerical=matrix)


def test_an_admitted_covariance_is_returned_unaltered() -> None:
    """Repair is erasure. The values that went in are the values that come out."""
    supplied = np.array([[4.0, 1.0], [1.0, 9.0]])
    model = NoiseModel.stationary(supplied, ("a", "b"), basis="bench characterisation")
    assert np.array_equal(model.blocks, supplied)


def test_a_singular_but_valid_covariance_is_admitted_without_a_floor() -> None:
    """A budget of purely systematic terms is singular, and that is correct.

    No jitter is added to make it invertible; whitening against it fails later
    with a message that says what is missing, which is the honest order.
    """
    rank_one = np.array([[1.0, 1.0], [1.0, 1.0]])
    model = NoiseModel.stationary(rank_one, ("a", "b"), basis="one shared unknown")
    assert np.array_equal(model.blocks, rank_one)
    assert np.linalg.matrix_rank(model.blocks) == 1


def test_the_assembled_total_is_validated_in_its_own_coordinates() -> None:
    """Admitting every operand is not enough; a congruence can amplify it.

    Here the total is built from blocks that are each valid, and the sum is
    checked again rather than assumed. The test is that the gate runs: a
    hand-built total with a negative diagonal is refused on read.
    """
    record = _record(n_steps=20)
    model = _model()
    honest = assemble(record, model, noise=_noise(model))
    assert honest.total.shape == (record.arclength.size,) * 2

    broken = OutputCovariance(
        arclength=record.arclength,
        outputs=model.outputs,
        blocks={"observation-noise": -np.eye(record.arclength.size)},
    )
    with pytest.raises(ValueError, match="negative variance"):
        _ = broken.total
