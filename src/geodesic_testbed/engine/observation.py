"""What the instrument actually reports, named and versioned.

A prediction is only comparable with a measurement if both are the same
quantity. Three different quantities appear in this repository and they differ
at second order in the perturbation -- the same order as the first-order
model's own failure -- so keeping them apart is not pedantry:

``intrinsic-surface-distance``
    the distance between two points measured *in the surface*. This is what a
    Jacobi field predicts and what a tape head travelling on the part would
    experience.

``ambient-euclidean-chord``
    the straight-line distance through space between the same two points. It
    is shorter, by a factor that depends on the normal curvature transverse to
    the path, and it is what a reconstructed 3-D coordinate pair gives.

``scanner-reconstructed-chord``
    the ambient chord as a metrology system reports it, after calibration,
    registration and surface fitting. It is the ambient chord plus an
    instrument error model. Not implemented: it needs a real instrument to
    characterise.

``camera-image-residual``
    the residual in image coordinates, before any reconstruction. Also not
    implemented, and further from the model than the others.

Every recorded comparison names its mode. Predicting the chord and comparing
it with an in-surface distance -- or the reverse -- produces a discrepancy of
exactly the size the experiment is trying to resolve, and would be read as a
model failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ObservationMode:
    """A named, versioned definition of the quantity a comparison is in."""

    identifier: str
    version: int
    quantity: str
    implemented: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODES: dict[str, ObservationMode] = {
    mode.identifier: mode
    for mode in (
        ObservationMode(
            identifier="intrinsic-surface-distance",
            version=1,
            quantity="Riemannian distance between two points, measured in the surface",
            implemented=True,
            note=(
                "What the Jacobi field predicts. Available in closed form on the "
                "constant-curvature model spaces; on a general surface it would "
                "require solving a boundary-value problem and is not computed here."
            ),
        ),
        ObservationMode(
            identifier="ambient-euclidean-chord",
            version=1,
            quantity="straight-line distance in R^3 between two points of the surface",
            implemented=True,
            note=(
                "What a pair of reconstructed 3-D coordinates gives. Differs from "
                "the intrinsic distance at second order in the separation, with a "
                "coefficient set by the transverse normal curvature."
            ),
        ),
        ObservationMode(
            identifier="scanner-reconstructed-chord",
            version=0,
            quantity="ambient chord as reported by a calibrated metrology system",
            implemented=False,
            note=(
                "The ambient chord plus calibration, registration and fitting "
                "error. Needs a characterised instrument; nothing here models it."
            ),
        ),
        ObservationMode(
            identifier="camera-image-residual",
            version=0,
            quantity="residual in image coordinates, before reconstruction",
            implemented=False,
            note="Furthest from the model of the four. Not modelled here.",
        ),
    )
}

DEFAULT_MODE = "intrinsic-surface-distance"


def mode(identifier: str) -> ObservationMode:
    try:
        return MODES[identifier]
    except KeyError as exc:  # pragma: no cover - guard
        raise KeyError(f"unknown observation mode {identifier!r}; have {sorted(MODES)}") from exc


def implemented_modes() -> tuple[str, ...]:
    return tuple(key for key, value in MODES.items() if value.implemented)


def catalogue() -> list[dict[str, Any]]:
    """Every mode, for the report header."""
    return [value.to_dict() for value in MODES.values()]
