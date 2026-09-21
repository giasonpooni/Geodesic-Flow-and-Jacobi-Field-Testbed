# SPDX-License-Identifier: MPL-2.0
r"""From the transfer map to something a sensor could have reported.

A prediction and a measurement are only comparable if they are the same
quantity, and the distance between "what ``Phi`` produces" and "what a scanner
reports" is four transformations, not a label::

    geometric transfer map        Phi(s) dz0        -- a tangent vector
            |   first-order -> finite separation
            v
    intrinsic surface distance    d(s)              -- a distance in the surface
            |   in-surface arc -> ambient chord
            v
    ambient euclidean chord       c(s)              -- what 3-D points give
            |   H(s), then the filter F
            v
    instrument output             y(s), R           -- what the sensor reports
            |
            v
    comparison statistic          whitened residual, chi-square

Every step is second order in the perturbation, which is exactly the order a
first-order model's own failure lives at. Skipping one does not produce a worse
answer; it produces an answer that disagrees with the measurement by the size
of the effect being measured, and the disagreement reads as a model failure.

So each stage is an object that carries **the transformation that produced
it**, not only a name for what it now is. :class:`Prediction` records its
stage, its observation mode, the operation applied, and the whole chain behind
it, and each transformation refuses to run where the quantity it would produce
is not available in the record's domain -- the intrinsic distance on a general
parametric surface being the case that matters, since computing it there is a
boundary-value problem nothing here solves.

The last stage is deliberately not a scalar. A maximum absolute residual
throws away the covariance, and the covariance is where the information is:
two residuals of the same size are different evidence if one is in a direction
the instrument resolves well. :func:`residual_statistics` returns the full
residual covariance, the whitened residual and the chi-square, and leaves the
verdict to whoever declared the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import validated_covariance
from .observation import require_available
from .observation_model import ObservationModel, TemporalFilter, operator_digest
from .record import TransferRecord, to_transfer_record
from .spaceforms import asin_k, cos_k, sin_k

Array = np.ndarray

#: A filtered comparison needs a full residual covariance rather than a
#: per-sample stack, and that matrix is quadratic in the number of scalar
#: residuals. Beyond this it is refused rather than silently approximated --
#: a filtered trial lives on the measurement's grid, which is small, and a
#: request to filter a solver grid is a sign the comparison is in the wrong
#: place.
MAX_CORRELATED_RESIDUALS = 4000

#: The stages of the chain, in order. A transformation may only move forward.
STAGES: tuple[str, ...] = (
    "first-order-tangent",
    "intrinsic-surface-distance",
    "ambient-euclidean-chord",
    "instrument-output",
)

#: The observation mode each stage is in.
STAGE_MODES: dict[str, str] = {
    "first-order-tangent": "first-order-tangent-separation",
    "intrinsic-surface-distance": "intrinsic-surface-distance",
    "ambient-euclidean-chord": "ambient-euclidean-chord",
    "instrument-output": "ambient-euclidean-chord",
}


@dataclass(frozen=True)
class Prediction:
    """One stage of the chain, with the operation that produced it.

    ``values`` is ``(n,)`` for a scalar separation and ``(n, m)`` once an
    observation model has projected it onto ``m`` outputs.

    ``chain`` is the whole history, so a comparison can print what was actually
    done rather than what the final label claims. A prediction that arrived at
    ``ambient-euclidean-chord`` by applying the chord correction and one that
    arrived there by being relabelled would carry the same mode and different
    chains, and only one of them is evidence.
    """

    stage: str
    values: Array
    arclength: Array
    transformation: str
    chain: tuple[str, ...] = ()
    covariance: Array | None = None
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        grid = np.asarray(self.arclength, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("a prediction needs an arclength grid of at least two samples")
        if values.shape[0] != grid.size:
            raise ValueError("values must carry one entry per arclength sample")
        if not np.all(np.isfinite(values)):
            raise ValueError("a prediction must be finite")
        grid.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "arclength", grid)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "chain", tuple(self.chain) or (self.transformation,))

    @property
    def observation_mode(self) -> str:
        return STAGE_MODES[self.stage]

    @property
    def outputs(self) -> int:
        """How many outputs each sample carries. One for a bare separation."""
        return 1 if self.values.ndim == 1 else int(self.values.shape[1])

    def advanced(
        self,
        stage: str,
        values: Array,
        transformation: str,
        *,
        covariance: Array | None = None,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> Prediction:
        """The next stage, with this one's history kept.

        Forward only. A chain that could run backwards would let a chord be
        relabelled an intrinsic distance, which is the single confusion this
        whole module exists to prevent.
        """
        if STAGES.index(stage) <= STAGES.index(self.stage):
            raise ValueError(
                f"{self.stage!r} cannot be transformed to {stage!r}: the chain runs "
                "from the transfer map towards the instrument, and a step back "
                "would be a relabelling"
            )
        return Prediction(
            stage=stage,
            values=values,
            arclength=self.arclength,
            transformation=transformation,
            chain=self.chain + (transformation,),
            covariance=covariance,
            note=note,
            extra=dict(extra or {}),
        )

    def to_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "observation_mode": self.observation_mode,
            "transformation": self.transformation,
            "chain": list(self.chain),
            "samples": int(self.arclength.size),
            "outputs": self.outputs,
            "has_covariance": self.covariance is not None,
            "note": self.note,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        if include_samples:
            payload["arclength"] = self.arclength.tolist()
            payload["values"] = self.values.tolist()
        return payload


# -- stage one: what the transfer map produces ----------------------------


def first_order_prediction(
    source: Any, lateral: float, heading: float
) -> Prediction:
    """``Phi(s) dz0``: the linear image of a starting-pose error.

    The only thing the transfer map produces, and not a distance. It is a
    vector in the tangent space at the nominal point; the separation it
    approximates differs from it at second order in the perturbation.
    """
    record: TransferRecord = to_transfer_record(source)
    require_available("first-order-tangent-separation", record.domain)
    return Prediction(
        stage="first-order-tangent",
        values=record.separation(lateral, heading),
        arclength=record.arclength,
        transformation=f"Phi(s) dz0 with dz0 = ({lateral!r}, {heading!r})",
        extra={"lateral": float(lateral), "heading": float(heading)},
    )


# -- stage two: the finite separation the first order approximates ---------


def intrinsic_from_tangent(
    prediction: Prediction, source: Any, *, heading: float | None = None
) -> Prediction:
    """The exact in-surface distance the first-order prediction approximates.

    Only where the curvature is constant along the path, and there the law of
    cosines

    ``sn_K(d/2) = sn_K(s) sin(eps/2)``

    closes in one line for all three curvatures at once. Where it varies this
    is a boundary-value problem that nothing here solves.

    The precondition is constant curvature and *not* the record's domain. A
    sphere reached through the general parametric machinery has a record whose
    domain says ``parametric-surface`` -- correctly, because that is the
    machinery that produced it -- and the closed form still applies to it,
    because the closed form's hypothesis is about the path and not about how
    the path was computed. The justification is recorded on the prediction
    either way, so the distinction is visible rather than assumed.

    The correction it applies is a shortening: ``d = eps sn_K(s) [1 -
    cn_K(s)^2 eps^2 / 24 + ...]``, second order in ``eps`` and therefore the
    same order as the first-order model's own error.
    """
    record: TransferRecord = to_transfer_record(source)
    curvature = np.asarray(record.gaussian_curvature, dtype=float)
    if float(np.ptp(curvature)) > 1e-12:
        raise ValueError(
            "the exact separation has a closed form only where the curvature is "
            "constant; this record's curvature varies along the path, and the "
            "in-surface distance between two nearby geodesics is then a "
            "boundary-value problem that is not solved here. Use "
            "chord_from_tangent, which applies the part that is computable and "
            "declares the part that is not"
        )
    angle = prediction.extra.get("heading") if heading is None else heading
    if angle is None:
        raise ValueError(
            "the exact separation is a function of the heading perturbation, and "
            "this prediction does not carry one; pass heading= explicitly"
        )
    if prediction.extra.get("lateral"):
        raise ValueError(
            "the closed form is for two geodesics leaving the *same* point at an "
            "angle; a starting lateral offset moves the second one's origin and "
            "is a different two-point problem"
        )
    # The free generalised-trigonometric functions rather than ``SpaceForm``,
    # which is restricted to the three model curvatures. The law of cosines
    # holds for any constant K, and a numerically computed pseudosphere hands
    # over K = -1 + 2e-16, which is constant and is not one of the three.
    curvature_value = float(np.mean(curvature))
    exact = 2.0 * asin_k(
        sin_k(prediction.arclength, curvature_value) * np.sin(float(angle) / 2.0),
        curvature_value,
    )
    return prediction.advanced(
        "intrinsic-surface-distance",
        np.sign(float(angle)) * exact,
        "sn_K(d/2) = sn_K(s) sin(eps/2): exact separation for constant K",
        note=(
            "a shortening of relative size cn_K(s)^2 eps^2 / 24, which is the "
            "same order as the first-order model's own error"
        ),
        extra=dict(prediction.extra)
        | {
            "curvature": curvature_value,
            "max_relative_correction_coefficient": float(
                np.max(cos_k(prediction.arclength, curvature_value) ** 2) / 24.0
            ),
            "justification": (
                "K is constant along this path"
                if _domain_lists_intrinsic(record)
                else "K is constant along this path; the record's domain does not "
                "list this mode, but the closed form's hypothesis is about the "
                "path and not about the machinery that produced it"
            ),
        },
    )


def has_closed_form_separation(source: Any) -> bool:
    """Whether the exact finite separation is available for this record.

    The hypothesis is constant curvature *along the path*, which is a property
    of the record's own samples. The pseudosphere is the case worth naming: it
    reaches this runtime through the general parametric machinery, so its
    domain says ``parametric-surface``, and its curvature is nevertheless
    exactly ``-1`` everywhere, so the law of cosines applies to it in full.
    """
    record = to_transfer_record(source)
    return bool(float(np.ptp(np.asarray(record.gaussian_curvature, dtype=float))) <= 1e-12)


def _domain_lists_intrinsic(record: TransferRecord) -> bool:
    """Whether the record's own domain lists the intrinsic distance as producible."""
    try:
        require_available("intrinsic-surface-distance", record.domain)
    except ValueError:
        return False
    return True


def chord_from_tangent(prediction: Prediction, source: Any) -> Prediction:
    r"""The chord correction alone, where the finite-separation one has no closed form.

    Two second-order corrections separate the first-order tangent prediction
    from an ambient chord, and on a surface whose curvature varies only one of
    them is computable here:

    ``c = j [1 - (cn_K^2 + kappa_n^2 sn_K^2) eps^2 / 24 + ...]``
            \_____/   \______________/
          not computable   computable from the record

    The first term needs the exact finite separation, which is a
    boundary-value problem on a general surface. The second needs only the
    transverse normal curvature and the separation itself, both of which the
    record carries.

    So this applies the second and **declares that it has not applied the
    first**. The omission travels with the prediction, in ``chain``, in
    ``note`` and in ``extra["intrinsic_correction"]``, because a consumer
    comparing at second order has to know which second-order terms are in the
    number it was handed. Where the curvature *is* constant, run
    :func:`intrinsic_from_tangent` first and then
    :func:`chord_from_intrinsic`, and both terms are there.
    """
    record: TransferRecord = to_transfer_record(source)
    chord = _chord(prediction, record)
    return prediction.advanced(
        "ambient-euclidean-chord",
        chord,
        "c = (2/kappa_n) sin(kappa_n j / 2) applied to the first-order separation",
        note=(
            "the chord shortening only. The finite-separation correction, of "
            "relative size cn_K(s)^2 eps^2 / 24, has no closed form where the "
            "curvature varies and has NOT been applied"
        ),
        extra=dict(prediction.extra)
        | {
            "intrinsic_correction": (
                "not-applied: no closed form for the finite separation on a "
                "varying-curvature surface"
            ),
            "max_relative_chord_shortening": _shortening(chord, prediction.values),
        },
    )


def _chord(prediction: Prediction, record: TransferRecord) -> Array:
    """``c = (2/kappa) sin(kappa d / 2)``, stable through ``kappa -> 0``."""
    require_available("ambient-euclidean-chord", record.domain)
    if record.geometry is None:
        raise ValueError(
            "the chord correction needs the transverse normal curvature, and this "
            "record carries no geometry; a declared curvature profile has no "
            "embedding and therefore no chord"
        )
    kappa = np.asarray(record.geometry.normal_curvature_transverse, dtype=float)
    distance = prediction.values
    return distance * np.sinc(0.5 * kappa * distance / np.pi)


def _shortening(chord: Array, distance: Array) -> float:
    return float(
        np.max(np.abs(chord - distance) / np.maximum(np.abs(distance), 1e-300))
    )


# -- stage three: what a pair of reconstructed 3-D points gives -------------


def chord_from_intrinsic(prediction: Prediction, source: Any) -> Prediction:
    """The ambient straight line between the two points, not the arc joining them.

    A geodesic has no geodesic curvature, so its curvature in the ambient space
    *is* its normal curvature, and over a short separation the connecting arc
    is a circular arc of radius ``1 / |kappa_n|``. The chord of such an arc is

    ``c = (2 / kappa) sin(kappa d / 2) = d (1 - (kappa d)^2 / 24 + ...)``

    and the ``kappa_n`` in question is the one **transverse** to the path,
    because the separation between two nearby geodesics is transverse. That is
    the quantity the record carries as ``normal_curvature_transverse``, and it
    is why it is carried: nothing else downstream can compute it.

    A camera does not report this either. It reports image coordinates, and a
    chord appears only after calibration, reconstruction and registration --
    which is the next stage, and a different one.
    """
    record: TransferRecord = to_transfer_record(source)
    chord = _chord(prediction, record)
    kappa = np.asarray(record.geometry.normal_curvature_transverse, dtype=float)
    return prediction.advanced(
        "ambient-euclidean-chord",
        chord,
        "c = (2/kappa_n) sin(kappa_n d / 2): the chord of the connecting arc",
        note=(
            "a second shortening, of relative size (kappa_n d)^2 / 24, with "
            "kappa_n the normal curvature transverse to the path"
        ),
        extra=dict(prediction.extra)
        | {
            "intrinsic_correction": "applied",
            "max_abs_transverse_normal_curvature": float(np.max(np.abs(kappa))),
            "max_relative_chord_shortening": _shortening(chord, prediction.values),
        },
    )


# -- stage four: what the instrument reports -------------------------------


def instrument_from_chord(
    prediction: Prediction,
    observation: ObservationModel,
    source: Any,
    *,
    initial_covariance=None,
    temporal_filter: TemporalFilter | None = None,
    filter_matrix=None,
) -> Prediction:
    """``y = H(s) dz(s)``, with the covariance the comparison has to be made against.

    Two things happen here and they are usually conflated. ``H`` projects the
    two-component state onto what the sensor actually reads, which is a
    different vector with different units. And ``R`` -- plus ``H Phi C0 Phi^T
    H^T`` when a starting covariance is declared -- is the spread that residual
    has to be judged against, which is not the same as the spread of the
    measurement alone.

    A filter is applied as an operator on both sides at once: the prediction
    becomes ``F H Phi dz0`` and the noise becomes ``F R F^T``. Comparing a
    filtered measurement against an unfiltered prediction with the original
    ``R`` understates the uncertainty and correlates samples that are then
    treated as independent, which is the single most reliable way to
    manufacture agreement.

    The filter is supplied as a declaration *and* an operator, both required
    together, and the result carries the operator's digest. A name and a
    version say which filter was meant; the digest says which one ran, and a
    comparison that checks only the name will accept a prediction smoothed by
    a different operator wearing the same label.
    """
    record: TransferRecord = to_transfer_record(source)
    if observation.mode != prediction.observation_mode:
        raise ValueError(
            f"this prediction is in {prediction.observation_mode!r} but the "
            f"observation model reports {observation.mode!r}; transform one before "
            "projecting, rather than relabelling either"
        )
    lateral = prediction.extra.get("lateral", 0.0)
    heading = prediction.extra.get("heading", 0.0)
    state = np.stack(
        [
            prediction.values,
            record.heading_change(float(lateral), float(heading)),
        ],
        axis=-1,
    )
    projected = state @ observation.matrix.T
    noise = np.broadcast_to(
        observation.noise_covariance,
        (prediction.arclength.size,) + observation.noise_covariance.shape,
    ).copy()
    chain = "y = H(s) dz(s)"
    if initial_covariance is not None:
        initial = validated_covariance(initial_covariance, "C0")
        noise = noise + observation.matrix @ record.propagate_covariance(
            initial
        ) @ observation.matrix.T
        chain = "y = H(s) dz(s), Cov(y) = H Phi C0 Phi^T H^T + R"
    digest = None
    if (temporal_filter is None) != (filter_matrix is None):
        raise ValueError(
            "a filter needs both its declaration and its operator: the declaration "
            "says which filter was meant and the operator is what actually ran, "
            "and a comparison that has only one of them cannot tell them apart"
        )
    if temporal_filter is not None:
        matrix = np.asarray(filter_matrix, dtype=float)
        if matrix.shape != (prediction.arclength.size, prediction.arclength.size):
            raise ValueError(
                "a temporal filter must be a square operator on the prediction's "
                f"own grid: expected {prediction.arclength.size} samples, got "
                f"{matrix.shape}"
            )
        samples, outputs = projected.shape
        if samples * outputs > MAX_CORRELATED_RESIDUALS:
            raise ValueError(
                f"filtering {samples} samples of {outputs} outputs would need a "
                f"{samples * outputs} x {samples * outputs} residual covariance. "
                "A filter correlates arc lengths, so the covariance it produces is "
                "not a per-sample stack and cannot be stored as one; run the "
                "comparison on the measurement's own grid, which is where a "
                "filtered trial lives, rather than on the solver's"
            )
        projected = matrix @ projected
        # ``F Sigma F^T`` in full. The filter mixes arc lengths, so what was
        # block-diagonal in ``s`` no longer is, and keeping only the diagonal
        # blocks would be exactly the error this whole module is about: it
        # would let a comparison treat samples as independent that the filter
        # made dependent, which understates the uncertainty.
        noise = np.einsum("ik,kab,jk->iajb", matrix, noise, matrix).reshape(
            samples * outputs, samples * outputs
        )
        noise = 0.5 * (noise + noise.T)
        digest = operator_digest(matrix)
        chain += (
            f", then the declared filter {temporal_filter.identifier!r} "
            f"{temporal_filter.version!r} ({digest})"
        )

    return prediction.advanced(
        "instrument-output",
        projected,
        chain,
        covariance=noise,
        note=(
            "the covariance is what the residual is judged against, and it is not "
            "the measurement noise alone"
        ),
        extra=dict(prediction.extra)
        | {
            "outputs": list(observation.outputs),
            "calibration_id": observation.calibration_id,
            "reconstruction_version": observation.reconstruction_version,
            "filtered": temporal_filter is not None,
            "filter_identifier": None if temporal_filter is None else temporal_filter.identifier,
            "filter_version": None if temporal_filter is None else temporal_filter.version,
            "filter_causal": None if temporal_filter is None else temporal_filter.causal,
            "filter_operator_digest": digest,
        },
    )


# -- the comparison statistic ----------------------------------------------


@dataclass(frozen=True)
class ResidualStatistics:
    """A residual, the covariance it must be judged against, and the whitened form.

    Not a scalar maximum. Two residuals of the same size are different evidence
    when one lies in a direction the instrument resolves well and the other
    does not, and a maximum absolute error cannot tell them apart -- it also
    cannot be compared between instruments, between output vectors of different
    lengths, or against any threshold that is not itself in the measurement's
    units.

    The whitened residual is ``L^-1 r`` with ``L L^T`` the residual covariance,
    so it is dimensionless and its components are uncorrelated and unit
    variance under the declared model. ``chi_square`` is its squared norm
    summed over the samples, with ``degrees_of_freedom`` the number of scalar
    residuals -- so ``reduced_chi_square`` near 1 says the residual is the size
    the declared covariance predicts, above 1 that the model or the covariance
    is wrong, and below 1 that the covariance is too generous.

    What "too far" means is not decided here. It is a property of the
    instrument's protocol, and a constant compiled into this module would be a
    declared limit smuggled into arithmetic.
    """

    residual: Array
    covariance: Array
    whitened: Array
    chi_square: float
    degrees_of_freedom: int
    max_abs_whitened: float
    max_abs_residual: float
    at_arclength: float
    #: Whether the covariance correlated samples -- as a filter's does -- so
    #: that the whitening was of the whole residual at once rather than sample
    #: by sample. A comparison that treats a filtered residual as independent
    #: samples is claiming information the filter removed.
    correlated: bool = False
    note: str = ""

    @property
    def reduced_chi_square(self) -> float:
        return float(self.chi_square / self.degrees_of_freedom)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chi_square": float(self.chi_square),
            "degrees_of_freedom": int(self.degrees_of_freedom),
            "reduced_chi_square": self.reduced_chi_square,
            "max_abs_whitened": float(self.max_abs_whitened),
            "max_abs_residual": float(self.max_abs_residual),
            "at_arclength": float(self.at_arclength),
            "samples": int(np.shape(self.residual)[0]),
            "correlated": bool(self.correlated),
            "note": self.note,
        }


def residual_statistics(
    measured, prediction: Prediction, *, covariance=None
) -> ResidualStatistics:
    """Whiten a residual against the covariance the comparison declares.

    ``covariance`` defaults to the one the prediction carries, which is the
    point of carrying it: the instrument stage computed ``H Phi C0 Phi^T H^T +
    R`` and the comparison must not be free to substitute something else by
    accident. Passing one explicitly is allowed and is then on the record, in
    the returned ``note``.
    """
    values = np.asarray(measured, dtype=float)
    if values.shape != prediction.values.shape:
        raise ValueError(
            f"the measurement has shape {values.shape} and the prediction "
            f"{prediction.values.shape}; a comparison needs them on the same grid "
            "and with the same outputs"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("the measurement must be finite")

    supplied = covariance is not None
    matrices = prediction.covariance if covariance is None else covariance
    if matrices is None:
        raise ValueError(
            "this prediction carries no covariance and none was supplied; a "
            "residual with no declared spread is a number, not evidence"
        )
    residual = values - prediction.values
    flat = residual if residual.ndim > 1 else residual[:, None]
    samples, outputs = flat.shape
    matrices = np.asarray(matrices, dtype=float)

    correlated = matrices.ndim == 2 and matrices.shape == (samples * outputs,) * 2
    if correlated:
        # A filter mixed the arc lengths, so the whole thing whitens at once.
        validated_covariance(matrices, "residual covariance", size=samples * outputs)
        factor = np.linalg.cholesky(matrices)
        whitened = np.linalg.solve(factor, flat.reshape(-1)).reshape(samples, outputs)
    else:
        if matrices.ndim == 0:
            matrices = np.full((samples, 1, 1), float(matrices))
        elif matrices.ndim == 1:
            matrices = matrices[:, None, None]
        elif matrices.ndim == 2:
            matrices = np.broadcast_to(matrices, (samples,) + matrices.shape).copy()
        if matrices.shape != (samples, outputs, outputs):
            raise ValueError(
                f"the residual covariance must be ({samples}, {outputs}, "
                f"{outputs}) for independent samples or "
                f"({samples * outputs}, {samples * outputs}) for correlated ones, "
                f"not {matrices.shape}"
            )
        for index, matrix in enumerate(matrices):
            validated_covariance(
                matrix, f"residual covariance at sample {index}", size=outputs
            )
        # Cholesky, not an inverse: the factor is what whitening means, and
        # solving against it is both cheaper and better conditioned than
        # forming Sigma^-1.
        factors = np.linalg.cholesky(matrices)
        whitened = np.linalg.solve(factors, flat[..., None])[..., 0]

    chi_square = float(np.sum(whitened**2))
    worst = int(np.argmax(np.max(np.abs(whitened), axis=-1)))
    return ResidualStatistics(
        residual=residual,
        covariance=matrices,
        whitened=whitened if residual.ndim > 1 else whitened[:, 0],
        chi_square=chi_square,
        degrees_of_freedom=int(whitened.size),
        max_abs_whitened=float(np.max(np.abs(whitened))),
        max_abs_residual=float(np.max(np.abs(residual))),
        at_arclength=float(prediction.arclength[worst]),
        correlated=correlated,
        note=(
            "whitened against a covariance supplied by the caller, not the one "
            "the prediction carries"
            if supplied
            else f"whitened against the covariance from: {prediction.transformation}"
        ),
    )


def predict_chord(
    source: Any, *, heading: float, to_stage: str = "ambient-euclidean-chord"
) -> Prediction:
    """Run the chain from the transfer map to ``to_stage`` in one call.

    A convenience, and a deliberately narrow one: it takes a pure heading
    perturbation, because that is the case the closed-form intrinsic separation
    covers. Anything else assembles the stages itself and sees each
    transformation it is applying.
    """
    if to_stage not in STAGES[: STAGES.index("instrument-output")]:
        raise ValueError(
            "predict_chord stops at the chord; the instrument stage needs an "
            "observation model, so call instrument_from_chord explicitly"
        )
    prediction = first_order_prediction(source, 0.0, heading)
    if to_stage == "first-order-tangent":
        return prediction
    prediction = intrinsic_from_tangent(prediction, source)
    if to_stage == "intrinsic-surface-distance":
        return prediction
    return chord_from_intrinsic(prediction, source)


__all__ = [
    "STAGES",
    "chord_from_tangent",
    "has_closed_form_separation",
    "STAGE_MODES",
    "Prediction",
    "ResidualStatistics",
    "chord_from_intrinsic",
    "first_order_prediction",
    "instrument_from_chord",
    "intrinsic_from_tangent",
    "predict_chord",
    "residual_statistics",
]
