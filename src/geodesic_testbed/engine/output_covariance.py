r"""The full covariance of a stacked prediction, as named blocks that add.

A comparison is only as good as the covariance it is judged against, and the
covariance a first-order propagation produces on its own is not the one the
comparison needs. Stacking the observations at ``n`` arc lengths into one
vector of length ``n m``, the honest statement is

.. code-block:: text

    Sigma_y = A C0 A^T  +  J_theta C_theta J_theta^T  +  R  +  Sigma_num

with ``A_i = H_i Phi_i`` the rows for sample ``i``. Four terms, and they are
here because dropping any one of them makes a comparison pass or fail for the
wrong reason:

``A C0 A^T``
    the starting pose, propagated. This is what the runtime already did, and
    it is the *only* term whose off-diagonal structure comes from the geometry.

``J_theta C_theta J_theta^T``
    everything the samples share. One fitted surface, one fixture, one
    calibration transform, one registration offset: each is a single unknown
    that every sample felt, so

    .. code-block:: text

        Sigma_ij^(theta) = J_theta,i C_theta J_theta,j^T

    is non-zero for ``i != j``. A budget that assigned each sample its own
    independent variance for a shared calibration error would have the samples
    averaging it away, which is exactly what a bias does not do. The
    cross-sample blocks are not a refinement; they are the difference between
    a bias and noise.

``R``
    the instrument's own noise, at whatever resolution it was characterised:
    one stationary block, a block per sample, or a full correlated matrix. A
    filter correlates arc lengths, so the third is not exotic.

``Sigma_num``
    the solver's own error, kept separate rather than folded into ``R``. It is
    not measurement noise and it does not shrink when the instrument improves,
    so a total that hides it inside ``R`` will keep passing a comparison that
    the numerics, not the sensor, is limiting.

**This module owns the meaning of the terms and not the algebra of
covariances.** It assembles the operands -- ``A``, ``J_theta``, the declared
``C_theta``, ``R`` and ``Sigma_num`` -- names what each one is, and adds them.
Jacobian machinery, Monte Carlo and SPD geometry belong to the covariance
tooling downstream; growing a second general covariance engine here is how two
implementations of the same thing start to disagree.

**Nothing is counted twice.** A calibration uncertainty can legitimately live
in ``C_theta`` *or* inside a characterised ``R``, and putting it in both is not
conservative -- it is wrong in a way that looks like caution. A
:class:`NoiseModel` declares which sources it already accounts for, and
assembling a total that also carries them as parameters is refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import Units, validated_covariance
from .observation_model import ObservationModel
from .record import TransferRecord, to_transfer_record
from .uncertainty import CONTRIBUTIONS

Array = np.ndarray

#: The four terms, in the order they are written above.
BLOCKS: tuple[str, ...] = (
    "starting-pose",
    "shared-parameters",
    "observation-noise",
    "numerical",
)

#: At what resolution an instrument's noise was actually characterised.
#: ``stationary``  -- one ``(m, m)`` block, repeated. The usual starting point.
#: ``per-sample``  -- ``(n, m, m)``: the noise varies along the path but
#:                    successive samples are independent.
#: ``correlated``  -- the full ``(n m, n m)``: samples are not independent. A
#:                    filter produces this, and so does a scanner whose
#:                    registration drifts.
NOISE_STRUCTURES: tuple[str, ...] = ("stationary", "per-sample", "correlated")


def _symmetrised(matrix: Array, name: str) -> Array:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, not {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array = 0.5 * (array + array.T)
    floor = -1e-10 * max(1.0, float(np.max(np.abs(array))))
    if float(np.min(np.linalg.eigvalsh(array))) < floor:
        raise ValueError(f"{name} must be positive semi-definite")
    array.setflags(write=False)
    return array


# -- the instrument's noise ------------------------------------------------


@dataclass(frozen=True)
class NoiseModel:
    """``R``, and a statement of what is already inside it.

    ``accounts_for`` is the field that makes the total addable. An instrument
    characterised on the bench against a traceable artefact has its calibration
    uncertainty inside its reported ``R``; one characterised against a nominal
    has not. The two are the same matrix with different meanings, and only the
    declaration tells them apart.
    """

    blocks: Array
    structure: str
    outputs: tuple[str, ...]
    basis: str
    calibration_id: str = "uncalibrated"
    accounts_for: tuple[str, ...] = ()
    samples: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.structure not in NOISE_STRUCTURES:
            raise ValueError(f"structure must be one of {NOISE_STRUCTURES}")
        if not self.basis:
            raise ValueError(
                "a noise model must say how it was arrived at; an R with no "
                "provenance cannot be argued with, and it is what every residual "
                "in the comparison is divided by"
            )
        for kind in self.accounts_for:
            if kind not in CONTRIBUTIONS:
                raise ValueError(f"accounts_for entries must be in {CONTRIBUTIONS}")
        width = len(self.outputs)
        if width < 1:
            raise ValueError("a noise model must name its outputs")
        array = np.asarray(self.blocks, dtype=float)
        if self.structure == "stationary":
            object.__setattr__(self, "blocks", _symmetrised(array, "R"))
            if self.blocks.shape != (width, width):
                raise ValueError(f"a stationary R must be ({width}, {width})")
        elif self.structure == "per-sample":
            if array.ndim != 3 or array.shape[1:] != (width, width):
                raise ValueError(f"a per-sample R must be (n, {width}, {width})")
            stacked = np.stack([_symmetrised(block, "R") for block in array])
            stacked.setflags(write=False)
            object.__setattr__(self, "blocks", stacked)
            object.__setattr__(self, "samples", int(stacked.shape[0]))
        else:
            matrix = _symmetrised(array, "R")
            if matrix.shape[0] % width:
                raise ValueError(
                    f"a correlated R must be (n*{width}, n*{width}); "
                    f"{matrix.shape[0]} is not a multiple of {width}"
                )
            object.__setattr__(self, "blocks", matrix)
            object.__setattr__(self, "samples", matrix.shape[0] // width)
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "accounts_for", tuple(self.accounts_for))

    @property
    def width(self) -> int:
        return len(self.outputs)

    @classmethod
    def stationary(
        cls,
        matrix,
        outputs: tuple[str, ...],
        *,
        basis: str,
        calibration_id: str = "uncalibrated",
        accounts_for: tuple[str, ...] = (),
        note: str = "",
    ) -> NoiseModel:
        return cls(
            blocks=matrix,
            structure="stationary",
            outputs=outputs,
            basis=basis,
            calibration_id=calibration_id,
            accounts_for=accounts_for,
            note=note,
        )

    @classmethod
    def from_observation_model(
        cls, model: ObservationModel, *, basis: str, accounts_for: tuple[str, ...] = ()
    ) -> NoiseModel:
        """The stationary ``R`` an observation model already carries.

        A convenience with a purpose: it forces the caller to state the basis
        and the double-counting declaration, which the observation model does
        not carry and cannot invent.
        """
        return cls.stationary(
            model.noise_covariance,
            model.outputs,
            basis=basis,
            calibration_id=model.calibration_id,
            accounts_for=accounts_for,
            note=model.note,
        )

    def stacked(self, samples: int) -> Array:
        """``R`` over the whole stacked observation vector, ``(n m, n m)``."""
        n = int(samples)
        width = self.width
        if self.samples is not None and self.samples != n:
            raise ValueError(
                f"this R was characterised on {self.samples} samples and the "
                f"prediction has {n}; an R on the wrong grid pairs noise with the "
                "wrong arc lengths"
            )
        if self.structure == "stationary":
            total = np.zeros((n * width, n * width))
            for index in range(n):
                slot = slice(index * width, (index + 1) * width)
                total[slot, slot] = self.blocks
            return total
        if self.structure == "per-sample":
            total = np.zeros((n * width, n * width))
            for index in range(n):
                slot = slice(index * width, (index + 1) * width)
                total[slot, slot] = self.blocks[index]
            return total
        return np.asarray(self.blocks, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "outputs": list(self.outputs),
            "basis": self.basis,
            "calibration_id": self.calibration_id,
            "accounts_for": list(self.accounts_for),
            "samples": self.samples,
            "note": self.note,
        }


# -- what the samples share ------------------------------------------------


@dataclass(frozen=True)
class SharedParameters:
    """``C_theta`` and ``J_theta``: one unknown each, felt at every sample.

    The Jacobian is ``(n, m, p)`` -- the derivative of each output at each arc
    length with respect to each parameter -- because that is the shape in which
    the cross-sample blocks come out for free. Flattening it to ``(n m, p)``
    and forming ``J C J^T`` produces ``Sigma_ij = J_i C J_j^T`` without anyone
    having to remember to write the off-diagonal terms down.
    """

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    covariance: Array
    jacobian: Array
    basis: str
    frame: str = "transverse-to-gamma, parallel-transported"
    units: Units = field(default_factory=Units)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("a shared-parameter block must name its parameters")
        if len(self.kinds) != len(self.names):
            raise ValueError("every parameter needs a kind, so the total can be audited")
        for kind in self.kinds:
            if kind not in CONTRIBUTIONS:
                raise ValueError(f"parameter kinds must be in {CONTRIBUTIONS}")
        if not self.basis:
            raise ValueError(
                "a shared-parameter block must say how C_theta was arrived at"
            )
        count = len(self.names)
        covariance = validated_covariance(self.covariance, "C_theta", count)
        covariance.setflags(write=False)
        object.__setattr__(self, "covariance", covariance)
        jacobian = np.asarray(self.jacobian, dtype=float)
        if jacobian.ndim != 3 or jacobian.shape[2] != count:
            raise ValueError(
                f"J_theta must be (n, m, {count}); one column per declared parameter"
            )
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("J_theta must be finite")
        jacobian.setflags(write=False)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "kinds", tuple(self.kinds))

    @property
    def samples(self) -> int:
        return int(self.jacobian.shape[0])

    @property
    def width(self) -> int:
        return int(self.jacobian.shape[1])

    def stacked_jacobian(self) -> Array:
        """``(n m, p)``: the Jacobian of the stacked observation vector."""
        return self.jacobian.reshape(self.samples * self.width, len(self.names))

    def contribution(self) -> Array:
        """``J_theta C_theta J_theta^T`` over the stacked vector."""
        jacobian = self.stacked_jacobian()
        return jacobian @ self.covariance @ jacobian.T

    def cross_sample_block(self, i: int, j: int) -> Array:
        """``Sigma_ij = J_i C J_j^T``: what samples ``i`` and ``j`` share."""
        return self.jacobian[i] @ self.covariance @ self.jacobian[j].T

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "kinds": list(self.kinds),
            "samples": self.samples,
            "outputs": self.width,
            "basis": self.basis,
            "frame": self.frame,
            "units": self.units.to_dict(),
            "sigma": [float(value) for value in np.sqrt(np.diag(self.covariance))],
            "note": self.note,
        }


def registration_parameter(
    source: Any,
    model: ObservationModel,
    sigma: float,
    *,
    basis: str,
    units: Units | None = None,
) -> SharedParameters:
    """One unknown registration offset ``ds``, shared by every sample.

    ``dy_i = H Phi'(s_i) dz0 ds``, so the Jacobian is read straight off the
    record, which already carries ``a'`` and ``b'``. It is the cleanest example
    of the shape this module exists for: a single scalar, one column, and every
    off-diagonal block non-zero.
    """
    record = to_transfer_record(source)
    value = float(sigma)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("a registration sigma must be finite and positive")
    rates = np.stack([record.a_rate, record.b_rate], axis=-1)
    jacobian = (rates @ model.matrix.T)[:, :, None]
    return SharedParameters(
        names=("path-registration-offset",),
        kinds=("path-registration",),
        covariance=np.array([[value**2]]),
        jacobian=jacobian,
        basis=basis,
        units=units or record.units,
        note=(
            "a single arclength offset between the prediction and the "
            "measurement, felt identically at every sample"
        ),
    )


@dataclass(frozen=True)
class LatentTruth:
    """A starting pose established *independently of the comparison*.

    NEES asks how far the estimate is from the truth in units of the claimed
    covariance, and it needs a truth. Most campaigns do not have one: the
    starting pose is what the trial is trying to measure, and scoring an
    estimate against a value derived from the same measurement is a statistic
    about arithmetic rather than about the instrument.

    So this carries the *provenance* as a required field and nothing here will
    construct one by default. A coupon fixtured against a calibrated datum and
    measured on a separate CMM has one; a run whose pose was read off the same
    scan it is being compared against does not.
    """

    value: Array
    basis: str
    established_by: str
    independent_of_comparison: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        values = np.asarray(self.value, dtype=float).reshape(-1)
        if values.size != 2:
            raise ValueError("a latent starting pose is (transverse, heading)")
        if not np.all(np.isfinite(values)):
            raise ValueError("a latent starting pose must be finite")
        if not self.basis or not self.established_by:
            raise ValueError(
                "a latent truth must say what established it and on what basis; "
                "without that, NEES is a statistic about arithmetic"
            )
        if not self.independent_of_comparison:
            raise ValueError(
                "a truth derived from the measurement it is being compared against "
                "is not a truth. Report NIS instead: it asks about the innovation, "
                "which is a quantity the comparison can actually see."
            )
        values.setflags(write=False)
        object.__setattr__(self, "value", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value.tolist(),
            "basis": self.basis,
            "established_by": self.established_by,
            "note": self.note,
        }


# -- the total -------------------------------------------------------------


@dataclass(frozen=True)
class OutputCovariance:
    """``Sigma_y`` over the stacked observation vector, kept as named blocks.

    The blocks are kept rather than only their sum, because the breakdown is
    what a campaign acts on: a total says whether the comparison can work, and
    the breakdown says which of the four to spend money on.
    """

    arclength: Array
    outputs: tuple[str, ...]
    blocks: dict[str, Array]
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        size = self.samples * self.width
        for name, block in self.blocks.items():
            if name not in BLOCKS:
                raise ValueError(f"block names must be in {BLOCKS}, not {name!r}")
            array = np.asarray(block, dtype=float)
            if array.shape != (size, size):
                raise ValueError(
                    f"the {name!r} block is {array.shape} and the stacked vector is "
                    f"({size}, {size}); every term is over the same stack"
                )
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "blocks", dict(self.blocks))

    @property
    def samples(self) -> int:
        return int(np.asarray(self.arclength).size)

    @property
    def width(self) -> int:
        return len(self.outputs)

    @property
    def total(self) -> Array:
        size = self.samples * self.width
        return sum(self.blocks.values(), start=np.zeros((size, size)))

    @property
    def degrees_of_freedom(self) -> int:
        """One per scalar residual. Stated rather than inferred at use."""
        return self.samples * self.width

    def shares(self) -> dict[str, float]:
        """Each block's share of the total variance, by trace.

        The trace and not the determinant: a block that is singular by
        construction -- every purely systematic one is -- has determinant zero
        and a perfectly real contribution.
        """
        traces = {name: float(np.trace(block)) for name, block in self.blocks.items()}
        denominator = sum(traces.values())
        if denominator <= 0.0:
            raise ValueError("a total covariance with no variance in it is not a total")
        return {name: value / denominator for name, value in traces.items()}

    def cholesky(self, *, jitter: float = 0.0) -> Array:
        total = self.total
        if jitter:
            total = total + float(jitter) * np.eye(total.shape[0])
        try:
            return np.linalg.cholesky(total)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Sigma_y is singular, so no residual can be whitened against it. A "
                "budget of purely systematic terms is singular and that is correct; "
                "it means the comparison needs an independent term -- the "
                "instrument's own noise -- before it can be judged at all."
            ) from exc

    def whiten(self, residual) -> Array:
        """``L^-1 r``: the residual in units of its own uncertainty.

        A whitened residual is the only form in which residuals from different
        instruments, different path lengths and different units are comparable,
        which is why a scalar maximum is not a substitute for it.
        """
        values = np.asarray(residual, dtype=float).reshape(-1)
        if values.size != self.degrees_of_freedom:
            raise ValueError(
                f"the residual has {values.size} entries and the stacked vector has "
                f"{self.degrees_of_freedom}"
            )
        return np.linalg.solve(self.cholesky(), values)

    def nis(self, residual) -> dict[str, Any]:
        """Normalised innovation squared, with the degrees of freedom stated.

        ``r^T Sigma_y^-1 r``. This is a consistency statistic about the
        *innovation* -- a quantity the comparison can actually see -- and it is
        the one to use when there is no independent truth. See :meth:`nees`
        for the other case, which most campaigns do not have.
        """
        whitened = self.whiten(residual)
        statistic = float(whitened @ whitened)
        dof = self.degrees_of_freedom
        return {
            "statistic": statistic,
            "degrees_of_freedom": dof,
            "reduced": statistic / dof,
            "probability_less_than": chi_square_cdf(statistic, dof),
            "shares": self.shares(),
        }

    def nees(self, estimate, truth: LatentTruth, starting_covariance) -> dict[str, Any]:
        """Normalised estimation error squared, against an independent truth.

        Two degrees of freedom, not ``n m``: this is a statement about the
        starting pose, which is what was estimated, and not about the stacked
        observation vector. Reporting it with the wrong dof is how a consistent
        filter is declared inconsistent.

        There is no default truth and no way to omit one. :class:`LatentTruth`
        refuses to be built from the comparison's own measurement, which is the
        case this method exists to keep out.
        """
        error = np.asarray(estimate, dtype=float).reshape(-1) - truth.value
        if error.size != 2:
            raise ValueError("the estimate is (transverse, heading), like the truth")
        covariance = validated_covariance(starting_covariance, "P", 2)
        statistic = float(error @ np.linalg.solve(covariance, error))
        return {
            "statistic": statistic,
            "degrees_of_freedom": 2,
            "reduced": 0.5 * statistic,
            "probability_less_than": chi_square_cdf(statistic, 2),
            "truth": truth.to_dict(),
        }

    def acceptance_band(self, *, coverage: float = 0.95) -> dict[str, Any]:
        """Two-sided limits on the NIS, so an inflated covariance cannot pass.

        One-sided is the mistake worth designing against. A campaign that only
        asks "is the residual smaller than the uncertainty" rewards an
        uncertainty that was overstated: a budget twice too large covers
        everything and is never rejected. The lower limit is what makes a
        covariance falsifiable in the direction it is usually wrong.
        """
        level = float(coverage)
        if not 0.0 < level < 1.0:
            raise ValueError("coverage must be strictly between 0 and 1")
        dof = self.degrees_of_freedom
        tail = 0.5 * (1.0 - level)
        return {
            "coverage": level,
            "degrees_of_freedom": dof,
            "lower": chi_square_quantile(tail, dof),
            "upper": chi_square_quantile(1.0 - tail, dof),
            "reduced_lower": chi_square_quantile(tail, dof) / dof,
            "reduced_upper": chi_square_quantile(1.0 - tail, dof) / dof,
        }

    def accepts(self, residual, *, coverage: float = 0.95) -> dict[str, Any]:
        """Whether the NIS lands inside the two-sided band, and which way if not."""
        statistic = self.nis(residual)
        band = self.acceptance_band(coverage=coverage)
        value = statistic["statistic"]
        verdict = "consistent"
        if value < band["lower"]:
            verdict = "covariance-too-large"
        elif value > band["upper"]:
            verdict = "residual-too-large"
        return statistic | {"band": band, "verdict": verdict, "accepted": verdict == "consistent"}

    def interval_coverage(self, residual, *, sigmas: float = 1.0) -> dict[str, Any]:
        """What fraction of the whitened residuals land inside ``+/- sigmas``.

        The complement of the chi-square: NIS asks one question of the whole
        vector, and this asks ``n m`` questions of its components. A covariance
        can pass the first while failing this badly -- a few samples far out and
        the rest far in sum to the right total.
        """
        whitened = self.whiten(residual)
        inside = float(np.mean(np.abs(whitened) <= float(sigmas)))
        expected = math.erf(float(sigmas) / math.sqrt(2.0))
        return {
            "sigmas": float(sigmas),
            "observed": inside,
            "expected": expected,
            "difference": inside - expected,
            "samples": int(whitened.size),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "outputs": list(self.outputs),
            "degrees_of_freedom": self.degrees_of_freedom,
            "blocks": sorted(self.blocks),
            "shares": self.shares(),
            "note": self.note,
            "extra": dict(self.extra),
        }


def stacked_operator(source: Any, model: ObservationModel) -> Array:
    """``A``: the ``(n m, 2)`` map from a starting-pose error to every output."""
    record: TransferRecord = to_transfer_record(source)
    matrices = record.transfer_map().matrices()
    return (model.matrix @ matrices).reshape(-1, 2)


def assemble(
    source: Any,
    model: ObservationModel,
    *,
    starting_covariance=None,
    parameters: SharedParameters | None = None,
    noise: NoiseModel,
    numerical=None,
    note: str = "",
) -> OutputCovariance:
    """``Sigma_y = A C0 A^T + J C J^T + R + Sigma_num``, as four named blocks.

    ``starting_covariance`` is taken from the record when it declares one. It
    is never substituted: a record whose covariance says undeclared and a call
    that supplies none is an error, because a total assembled without the
    starting pose would be smaller than the truth in exactly the term this
    runtime exists to compute.
    """
    record = to_transfer_record(source)
    if record.observation_mode != model.mode:
        raise ValueError(
            f"the record is in {record.observation_mode!r} and the observation model "
            f"reports {model.mode!r}; a covariance assembled across two modes is a "
            "sum of two different quantities"
        )
    samples = record.arclength.size
    if noise.width != len(model.outputs):
        raise ValueError(
            f"R is {noise.width} wide and the observation model reports "
            f"{len(model.outputs)} outputs"
        )

    if starting_covariance is None:
        if not record.covariance.declared:
            raise ValueError(
                "no starting covariance: the record declares none and none was "
                "supplied. Substituting a C0 here is the one thing this module must "
                "not do -- the total would be short by the term the whole runtime is "
                "about, and nothing downstream could tell."
            )
        initial = record.covariance.require()
    else:
        initial = validated_covariance(starting_covariance, "C0", 2)

    if parameters is not None:
        overlap = sorted(set(parameters.kinds) & set(noise.accounts_for))
        if overlap:
            raise ValueError(
                f"{overlap} would be counted twice: R declares it already accounts "
                f"for {list(noise.accounts_for)}, and C_theta carries it as a shared "
                "parameter. Counting an uncertainty twice is not conservative, it is "
                "wrong in the direction that looks like caution."
            )
        if parameters.samples != samples or parameters.width != len(model.outputs):
            raise ValueError(
                f"J_theta is ({parameters.samples}, {parameters.width}, p) and the "
                f"prediction is ({samples}, {len(model.outputs)})"
            )

    operator = stacked_operator(record, model)
    size = samples * len(model.outputs)
    blocks: dict[str, Array] = {
        "starting-pose": operator @ initial @ operator.T,
        "observation-noise": noise.stacked(samples),
    }
    if parameters is not None:
        blocks["shared-parameters"] = parameters.contribution()
    if numerical is not None:
        block = np.asarray(numerical, dtype=float)
        if block.shape == (size,):
            block = np.diag(block)
        blocks["numerical"] = _symmetrised(block, "Sigma_num")
        if blocks["numerical"].shape != (size, size):
            raise ValueError(f"Sigma_num must be ({size}, {size}) or a vector of {size}")

    return OutputCovariance(
        arclength=record.arclength,
        outputs=model.outputs,
        blocks=blocks,
        note=note,
        extra={
            "mode": model.mode,
            "noise": noise.to_dict(),
            "parameters": None if parameters is None else parameters.to_dict(),
            "starting_covariance_declared": bool(record.covariance.declared),
        },
    )


# -- chi-square, exactly, because the band is a declared limit -------------


def _lower_incomplete_gamma_ratio(a: float, x: float) -> float:
    """``P(a, x)``, by series below the crossover and continued fraction above.

    Written out rather than approximated. The acceptance band is a declared
    limit, and a limit computed from a fit whose error nobody bounded is the
    kind of thing this repository refuses elsewhere.
    """
    if x < 0.0 or a <= 0.0:
        raise ValueError("the incomplete gamma ratio needs a > 0 and x >= 0")
    if x == 0.0:
        return 0.0
    log_gamma = math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        index = a
        for _ in range(10_000):
            index += 1.0
            term *= x / index
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - log_gamma)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, 10_000):
        an = -index * (index - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - log_gamma) * h


def chi_square_cdf(statistic: float, degrees_of_freedom: int) -> float:
    """``P(X <= statistic)`` for a chi-square with ``k`` degrees of freedom."""
    dof = int(degrees_of_freedom)
    if dof < 1:
        raise ValueError("degrees of freedom must be at least one")
    value = float(statistic)
    if value <= 0.0:
        return 0.0
    return _lower_incomplete_gamma_ratio(0.5 * dof, 0.5 * value)


def chi_square_quantile(probability: float, degrees_of_freedom: int) -> float:
    """The inverse of :func:`chi_square_cdf`, by bracketed bisection."""
    target = float(probability)
    if not 0.0 < target < 1.0:
        raise ValueError("a quantile probability must be strictly between 0 and 1")
    dof = int(degrees_of_freedom)
    low, high = 0.0, max(2.0 * dof, 10.0)
    while chi_square_cdf(high, dof) < target:
        high *= 2.0
        if high > 1e12:  # pragma: no cover - guard
            raise ValueError("the quantile did not bracket")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if chi_square_cdf(middle, dof) < target:
            low = middle
        else:
            high = middle
        if high - low < 1e-12 * max(1.0, high):
            break
    return 0.5 * (low + high)
