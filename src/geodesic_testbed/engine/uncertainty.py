r"""Everything else that is uncertain, and what each of it costs downstream.

The starting pose is one term in an uncertainty budget and it is rarely the
largest. A campaign that propagates ``C0`` and calls the result the total is
reporting a bound on one contribution; the surface it flowed along was fitted
to a scan, the part was fixtured against a datum, the metrology frame was
calibrated, the prediction was registered to the measurement at *some* arc
length, and the sensor's noise is not white. Each of those is an error in the
same quantity, and each has a different shape.

Shape is the point. These terms do not combine by adding variances, because
most of them are *systematic*: one unknown surface fit, one unknown datum
offset, one unknown registration shift, each wrong in the same direction at
every sample of the path. A systematic term contributes a rank-one covariance
``sigma^2 v v^T`` -- perfectly correlated along the path -- and a budget that
records it as a per-sample variance has thrown away exactly the structure that
distinguishes a bias from noise. The difference is not academic: averaging
along the path suppresses the second and does nothing at all to the first.

Two of the terms are computable here and nowhere else, because they need the
transfer map itself:

**A curvature error.** If the fitted surface has ``K + dK`` where the part has
``K``, the variation equation picks up a source,

.. code-block:: text

    dj'' + K dj = -dK j     =>     dj(s) = -int_0^s G(s,t) dK(t) j(t) dt

with ``G(s,t) = b(s) a(t) - a(s) b(t)`` the Green's function of the Jacobi
operator -- and ``G`` is built from the two columns of ``Phi`` with no
denominator, because ``det Phi = 1`` makes the Wronskian exactly one. So the
sensitivity of the prediction to a curvature bias is an integral over the
record's own samples, and needs no re-solve.

**A registration error.** If the prediction's ``s`` and the measurement's ``s``
differ by ``ds``, the comparison is off by ``j'(s) ds`` -- again read straight
off the record, which carries ``a'`` and ``b'``.

Nothing here decides what an acceptable total is. It reports the total, the
breakdown, and which term dominates where, and that is the information an
instrument protocol needs in order to decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import validated_covariance
from .record import TransferRecord, to_transfer_record

Array = np.ndarray

#: The sources a path-sensitivity campaign has to account for. Anything a
#: caller adds outside this vocabulary is ``other``, named, and visible in the
#: breakdown -- an unnamed contribution is the one that never gets argued about.
CONTRIBUTIONS: tuple[str, ...] = (
    "starting-pose",
    "surface-reconstruction",
    "fixture-datum",
    "calibration-transform",
    "path-registration",
    "sensor-noise",
    "other",
)

#: How a contribution varies along the path. The distinction decides how it
#: combines, and getting it wrong is the most common way a budget is wrong by
#: an order of magnitude rather than by a few per cent.
#:
#: ``systematic``  -- one unknown, the same at every sample. Rank one.
#:                    Averaging along the path does not reduce it.
#: ``independent`` -- a fresh draw per sample. Diagonal. Averaging helps.
#: ``correlated``  -- neither: a declared correlation length.
STRUCTURES: tuple[str, ...] = ("systematic", "independent", "correlated")


@dataclass(frozen=True)
class Contribution:
    """One named source of error, in the comparison's own units.

    ``covariance`` is always the full ``(n, n)`` matrix over the path, because
    that is the only representation in which the three structures are the same
    kind of object and can be added. It is built by the constructors below
    rather than by hand.
    """

    kind: str
    name: str
    covariance: Array
    structure: str
    basis: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CONTRIBUTIONS:
            raise ValueError(f"kind must be one of {CONTRIBUTIONS}")
        if self.structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}")
        if not self.basis:
            raise ValueError(
                f"the {self.name!r} contribution must say on what basis it was "
                "arrived at; a variance with no provenance cannot be argued with"
            )
        matrix = np.asarray(self.covariance, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("a contribution's covariance must be square over the path")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"the {self.name!r} covariance must be finite")
        matrix = 0.5 * (matrix + matrix.T)
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-10 * max(
            1.0, float(np.max(np.abs(matrix)))
        ):
            raise ValueError(f"the {self.name!r} covariance must be positive semi-definite")
        matrix.setflags(write=False)
        object.__setattr__(self, "covariance", matrix)

    @property
    def sigma(self) -> Array:
        """Per-sample standard deviation: the diagonal, square-rooted."""
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.covariance, tol=1e-12))

    def to_dict(self) -> dict[str, Any]:
        sigma = self.sigma
        return {
            "kind": self.kind,
            "name": self.name,
            "structure": self.structure,
            "basis": self.basis,
            "rank": self.rank,
            "max_sigma": float(np.max(sigma)),
            "sigma_at_end": float(sigma[-1]),
            "note": self.note,
        }


# -- the constructors ------------------------------------------------------


def _systematic(sensitivity: Array, sigma: float) -> Array:
    """``sigma^2 v v^T``: one unknown, felt the same way at every sample."""
    vector = np.asarray(sensitivity, dtype=float)
    return float(sigma) ** 2 * np.outer(vector, vector)


def starting_pose(
    source: Any, covariance, *, name: str = "starting pose", basis: str = "declared"
) -> Contribution:
    """``[Phi C0 Phi^T]_00``: where the tool began, carried down the path.

    Systematic by construction. The starting pose is drawn once per run and the
    transfer map carries that one draw forward, so the resulting error at two
    arc lengths is perfectly correlated -- which is why this is ``a`` and ``b``
    outer-producted with themselves rather than a per-sample variance.
    """
    record: TransferRecord = to_transfer_record(source)
    initial = validated_covariance(covariance, "C0")
    columns = np.stack([record.a, record.b], axis=-1)
    matrix = columns @ initial @ columns.T
    return Contribution(
        kind="starting-pose",
        name=name,
        covariance=matrix,
        structure="systematic",
        basis=basis,
        note="Phi C0 Phi^T, transverse component; one draw carried down the path",
    )


def curvature_greens_function(source: Any) -> Array:
    """``G(s, t) = b(s) a(t) - a(s) b(t)``, the Jacobi operator's own kernel.

    The full ``(n, n)`` kernel, kept for inspection and for the rank-one check
    that it separates; :func:`curvature_sensitivity` uses the separation rather
    than the matrix. Lower-triangular in effect because the variation is
    causal: a curvature error at ``t`` cannot move the path at an earlier
    ``s``. There is no Wronskian in the denominator because
    ``det Phi = 1`` exactly -- the same invariant the solver never enforces,
    used here as an identity.
    """
    record: TransferRecord = to_transfer_record(source)
    a, b = record.a, record.b
    kernel = np.outer(b, a) - np.outer(a, b)
    causal = record.arclength[None, :] <= record.arclength[:, None]
    return kernel * causal


def cumulative_quadrature(values: Array, grid: Array) -> Array:
    """``F(s) = int_0^s f``, at every sample, to fourth order on a uniform grid.

    The cumulative trapezoid would be the obvious choice and it is second
    order, which on a two-thousand-point path puts a 1e-6 relative floor under
    everything computed with it -- enough to hide whether a sensitivity formula
    is the derivative or merely close to it. A composite Simpson stepping two
    samples at a time, with a fourth-order Adams half-step for the odd indices,
    costs the same ``O(n)`` and leaves the formula as the thing being measured.

    A non-uniform grid falls back to the cumulative trapezoid, because Simpson
    on unequal intervals is a different rule and silently applying it would be
    the error this docstring is about.
    """
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    steps = np.diff(grid)
    result = np.zeros_like(values)
    if values.size < 3 or not np.allclose(steps, steps[0], rtol=1e-12, atol=0.0):
        result[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * steps)
        return result
    h = float(steps[0])
    # Even indices: composite Simpson over each pair of intervals.
    simpson = (h / 3.0) * (values[:-2:2] + 4.0 * values[1:-1:2] + values[2::2])
    result[2::2] = np.cumsum(simpson)
    # Odd indices: a fourth-order half-step from the even sample before them.
    result[1:-1:2] = result[0:-2:2] + (h / 12.0) * (
        5.0 * values[0:-2:2] + 8.0 * values[1:-1:2] - values[2::2]
    )
    if values.size % 2 == 0:
        # An even sample count leaves the last index odd and past the pairs.
        result[-1] = result[-2] + (h / 12.0) * (
            -values[-3] + 8.0 * values[-2] + 5.0 * values[-1]
        )
    return result


def curvature_sensitivity(source: Any, lateral: float, heading: float) -> Array:
    """``dj/d(K bias)``: how a uniform curvature error moves the prediction.

    ``dj(s) = -int_0^s G(s, t) j(t) dt`` for a bias that is the same at every
    arc length, which is the shape a surface *fit* error has: a mis-estimated
    radius is wrong in one direction along the whole path, not resampled at
    each step.

    The kernel separates -- ``G(s, t) = b(s) a(t) - a(s) b(t)`` -- so the
    double integral is two cumulative ones,

    ``dj(s) = -[ b(s) int_0^s a j - a(s) int_0^s b j ]``

    which is ``O(n)`` rather than ``O(n^2)`` and needs no kernel matrix. On a
    four-thousand-sample path that is the difference between two vectors and a
    128 MB array.
    """
    record: TransferRecord = to_transfer_record(source)
    nominal = record.separation(lateral, heading)
    grid = record.arclength
    first = cumulative_quadrature(record.a * nominal, grid)
    second = cumulative_quadrature(record.b * nominal, grid)
    return -(record.b * first - record.a * second)


def surface_reconstruction(
    source: Any,
    lateral: float,
    heading: float,
    *,
    curvature_sigma: float | None = None,
    name: str = "surface reconstruction",
    basis: str = "as-built-scan",
) -> Contribution:
    """What a mis-fitted curvature costs the prediction, through the Jacobi kernel.

    ``curvature_sigma`` defaults to the record's own declared geometry
    uncertainty, which is where it belongs: how well the surface is known is a
    property of the scan, and the record carries it precisely so that it does
    not have to be restated here.
    """
    record: TransferRecord = to_transfer_record(source)
    if curvature_sigma is None:
        if record.geometry is None or not record.geometry.uncertainty.declared:
            raise ValueError(
                "this record declares no geometry uncertainty, so the cost of a "
                "curvature error cannot be computed from it; pass "
                "curvature_sigma= explicitly, or attach a declared uncertainty "
                "with with_geometry_uncertainty()"
            )
        declared = record.geometry.uncertainty.curvature
        if declared is None:
            raise ValueError(
                "the record's geometry uncertainty declares no curvature term"
            )
        curvature_sigma = declared
        basis = record.geometry.uncertainty.basis
    return Contribution(
        kind="surface-reconstruction",
        name=name,
        covariance=_systematic(
            curvature_sensitivity(record, lateral, heading), float(curvature_sigma)
        ),
        structure="systematic",
        basis=basis,
        note=(
            "dj(s) = -int_0^s [b(s)a(t) - a(s)b(t)] dK j(t) dt, for a curvature "
            "bias that is the same along the whole path"
        ),
    )


def path_registration(
    source: Any,
    lateral: float,
    heading: float,
    *,
    sigma: float,
    name: str = "path registration",
    basis: str = "declared",
) -> Contribution:
    """An unknown arc-length offset between the prediction and the measurement.

    ``dj = j'(s) ds``, and ``j'`` is the record's second row. Systematic,
    because a registration offset is one number for the whole comparison -- and
    the term that is largest exactly where the prediction is steepest, which is
    not where the prediction is largest.
    """
    record: TransferRecord = to_transfer_record(source)
    return Contribution(
        kind="path-registration",
        name=name,
        covariance=_systematic(record.heading_change(lateral, heading), sigma),
        structure="systematic",
        basis=basis,
        note="dj = j'(s) ds for one unknown arc-length offset",
    )


def fixture_datum(
    source: Any, *, lateral_sigma: float, heading_sigma: float,
    correlation: float = 0.0, name: str = "fixture and datum",
    basis: str = "declared",
) -> Contribution:
    """A rigid mis-set between the part datum and where the path was planned.

    It enters exactly where the starting pose does -- it *is* a starting-pose
    error, arriving from the fixture rather than from the tool -- so it is
    propagated by the same ``Phi C0 Phi^T`` and is kept as its own line only
    because a budget that cannot separate the fixture from the tool cannot tell
    anyone which to improve.
    """
    rho = float(correlation)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    cross = rho * float(lateral_sigma) * float(heading_sigma)
    matrix = np.array(
        [[float(lateral_sigma) ** 2, cross], [cross, float(heading_sigma) ** 2]]
    )
    contribution = starting_pose(source, matrix, name=name, basis=basis)
    return Contribution(
        kind="fixture-datum",
        name=name,
        covariance=contribution.covariance,
        structure="systematic",
        basis=basis,
        note="a rigid datum offset enters as a starting-pose error and propagates as one",
    )


def calibration_transform(
    source: Any, *, sigma: float, name: str = "calibration transform",
    basis: str = "declared",
) -> Contribution:
    """A constant offset in the metrology frame, unknown and the same throughout.

    The simplest systematic there is -- a rank-one covariance on the constant
    vector -- and the one a per-sample variance most badly misrepresents, since
    its per-sample standard deviation is flat and its *effect* on anything
    averaged along the path is undiminished.
    """
    record: TransferRecord = to_transfer_record(source)
    return Contribution(
        kind="calibration-transform",
        name=name,
        covariance=_systematic(np.ones(record.arclength.size), sigma),
        structure="systematic",
        basis=basis,
        note="one unknown offset in the measurement frame, constant along the path",
    )


def sensor_noise(
    source: Any,
    *,
    sigma: float,
    correlation_length: float = 0.0,
    name: str = "sensor noise",
    basis: str = "declared",
) -> Contribution:
    """The instrument's own noise, with a declared correlation length.

    ``correlation_length = 0`` is white noise and a diagonal covariance.
    Anything else gives ``sigma^2 exp(-|s_i - s_j| / l)``, which is what a real
    scanner has and what makes the difference between ``n`` independent
    measurements and rather fewer. Treating correlated noise as independent
    overstates how much averaging buys, which is the optimistic direction.
    """
    record: TransferRecord = to_transfer_record(source)
    grid = record.arclength
    if float(correlation_length) < 0.0:
        raise ValueError("correlation_length must be nonnegative")
    if float(correlation_length) == 0.0:
        matrix = float(sigma) ** 2 * np.eye(grid.size)
        structure = "independent"
    else:
        separation = np.abs(grid[:, None] - grid[None, :])
        matrix = float(sigma) ** 2 * np.exp(-separation / float(correlation_length))
        structure = "correlated"
    return Contribution(
        kind="sensor-noise",
        name=name,
        covariance=matrix,
        structure=structure,
        basis=basis,
        note=(
            "white"
            if structure == "independent"
            else f"exponential kernel, correlation length {float(correlation_length):g}"
        ),
    )


# -- the budget ------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyBudget:
    """Every declared contribution, the total, and which one dominates where.

    The total is a sum of covariances and not of standard deviations, and it is
    a sum of *matrices* rather than of per-sample variances, because the
    systematic terms carry off-diagonal structure that a variance sum destroys.

    The breakdown is the deliverable. A total on its own says whether a
    campaign will work; the breakdown says what to fix, and in practice the
    answer is rarely the term the campaign was designed around.
    """

    arclength: Array
    contributions: tuple[Contribution, ...]
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.contributions:
            raise ValueError(
                "an uncertainty budget with no contributions is not a budget; if a "
                "campaign genuinely has one source, declare that one"
            )
        size = np.asarray(self.arclength).size
        for contribution in self.contributions:
            if contribution.covariance.shape != (size, size):
                raise ValueError(
                    f"the {contribution.name!r} contribution is on a grid of "
                    f"{contribution.covariance.shape[0]} samples and the budget has "
                    f"{size}; every term must be on the same path"
                )
        names = [contribution.name for contribution in self.contributions]
        if len(names) != len(set(names)):
            raise ValueError("two contributions share a name, so the breakdown is ambiguous")
        object.__setattr__(self, "contributions", tuple(self.contributions))

    @property
    def total(self) -> Array:
        """The full ``(n, n)`` residual covariance the comparison is judged against."""
        return sum(
            (contribution.covariance for contribution in self.contributions),
            start=np.zeros((self.arclength.size,) * 2),
        )

    @property
    def sigma(self) -> Array:
        """Per-sample total standard deviation."""
        return np.sqrt(np.clip(np.diag(self.total), 0.0, None))

    def dominant(self, index: int | None = None) -> Contribution:
        """Whichever term contributes the most variance, at the worst sample by default."""
        where = int(np.argmax(self.sigma)) if index is None else int(index)
        return max(
            self.contributions,
            key=lambda contribution: float(contribution.covariance[where, where]),
        )

    def shares(self, index: int | None = None) -> dict[str, float]:
        """Each term's fraction of the total variance at one sample."""
        where = int(np.argmax(self.sigma)) if index is None else int(index)
        total = float(self.total[where, where])
        if total <= 0.0:
            return {contribution.name: 0.0 for contribution in self.contributions}
        return {
            contribution.name: float(contribution.covariance[where, where] / total)
            for contribution in self.contributions
        }

    @property
    def systematic_fraction(self) -> float:
        """How much of the worst-sample variance averaging along the path cannot touch."""
        where = int(np.argmax(self.sigma))
        total = float(self.total[where, where])
        if total <= 0.0:
            return 0.0
        systematic = sum(
            float(contribution.covariance[where, where])
            for contribution in self.contributions
            if contribution.structure == "systematic"
        )
        return float(systematic / total)

    def is_positive_definite(self) -> bool:
        """Whether the total can whiten a residual at all.

        A budget of purely systematic terms is singular, and correctly so: a
        perfectly correlated error is perfectly predictable, so some direction
        in the residual space has no uncertainty in it. Whitening needs at
        least one term with full rank, which in practice means the sensor's own
        noise -- and a campaign that forgot to declare it finds out here.
        """
        try:
            np.linalg.cholesky(self.total)
        except np.linalg.LinAlgError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        where = int(np.argmax(self.sigma))
        return {
            "samples": int(self.arclength.size),
            "contributions": [c.to_dict() for c in self.contributions],
            "max_sigma": float(np.max(self.sigma)),
            "max_sigma_at_arclength": float(self.arclength[where]),
            "sigma_at_end": float(self.sigma[-1]),
            "dominant_at_worst": self.dominant().name,
            "shares_at_worst": self.shares(),
            "systematic_fraction_at_worst": self.systematic_fraction,
            "positive_definite": self.is_positive_definite(),
            "total_rank": int(np.linalg.matrix_rank(self.total, tol=1e-12)),
            "note": self.note,
            "extra": dict(self.extra),
        }


def budget(source: Any, *contributions: Contribution, note: str = "") -> UncertaintyBudget:
    """Assemble a budget on a record's own arclength grid."""
    record: TransferRecord = to_transfer_record(source)
    return UncertaintyBudget(
        arclength=record.arclength, contributions=contributions, note=note
    )


__all__ = [
    "CONTRIBUTIONS",
    "STRUCTURES",
    "Contribution",
    "UncertaintyBudget",
    "budget",
    "calibration_transform",
    "cumulative_quadrature",
    "curvature_greens_function",
    "curvature_sensitivity",
    "fixture_datum",
    "path_registration",
    "sensor_noise",
    "starting_pose",
    "surface_reconstruction",
]
