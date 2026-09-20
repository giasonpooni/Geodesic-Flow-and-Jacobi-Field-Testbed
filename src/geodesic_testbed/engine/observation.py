"""What the instrument actually reports, named, versioned, and scoped by domain.

A prediction is only comparable with a measurement if both are the same
quantity. Four appear around this repository, and the first two differ at
second order in the perturbation -- the same order as the first-order model's
own failure -- so keeping them apart is not pedantry:

``intrinsic-surface-distance``
    the distance between two points measured *in the surface*. This is what a
    Jacobi field predicts and what a tool travelling on the part experiences.

``ambient-euclidean-chord``
    the straight-line distance through space between the same two points. It
    is shorter, by a factor set by the normal curvature transverse to the path,
    and it is what a reconstructed pair of 3-D coordinates gives.

``scanner-reconstructed-chord``
    the ambient chord as a metrology system reports it, after calibration,
    registration and surface fitting -- the chord plus an instrument error
    model.

``camera-image-residual``
    the residual in image coordinates, before any reconstruction. A camera does
    not measure a chord; it measures image coordinates, and a chord appears
    only once those have been calibrated, reconstructed and registered.

**Implementation is recorded per domain, not globally.** A mode can be exact in
one setting and unavailable in another: the intrinsic distance has a closed
form on a constant-curvature model space, but on a general parametric surface
computing it means solving a boundary-value problem, which this repository does
not do. A single ``implemented`` flag would have to lie about one of those.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

#: The settings a mode can be asked for. They are ordered by how far each is
#: from the mathematics: a model space, a curvature profile someone declared,
#: a real surface, a real instrument.
#:
#: ``declared-curvature-profile`` is the setting a
#: :class:`~geodesic_testbed.jacobi.JacobiTrace` on a varying ``K(s)`` lives
#: in, and it is genuinely its own: there is no closed form, because ``K``
#: varies, and there is no embedding either, because a curvature profile is
#: not a surface. Folding it into either neighbour would have to claim one of
#: those, and the modes divide on exactly that line.
DOMAINS: tuple[str, ...] = (
    "constant-curvature",
    "declared-curvature-profile",
    "parametric-surface",
    "physical-instrument",
)

#: How well a mode is supported in one domain.
#: ``exact``       -- available in closed form.
#: ``numerical``   -- computed, to the solver's accuracy.
#: ``unavailable`` -- would require machinery this repository does not have.
SUPPORT: tuple[str, ...] = ("exact", "numerical", "unavailable")


@dataclass(frozen=True)
class ObservationMode:
    """A named, versioned definition of the quantity a comparison is in."""

    identifier: str
    version: int
    quantity: str
    support: dict[str, str]
    note: str

    def __post_init__(self) -> None:
        if set(self.support) != set(DOMAINS):
            raise ValueError(f"support must cover exactly {DOMAINS}")
        for domain, level in self.support.items():
            if level not in SUPPORT:
                raise ValueError(f"support[{domain!r}] must be one of {SUPPORT}")

    def support_in(self, domain: str) -> str:
        try:
            return self.support[domain]
        except KeyError as exc:  # pragma: no cover - guard
            raise KeyError(f"unknown domain {domain!r}; have {DOMAINS}") from exc

    def is_available(self, domain: str) -> bool:
        """True where this repository can actually produce the quantity."""
        return self.support_in(domain) != "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODES: dict[str, ObservationMode] = {
    mode.identifier: mode
    for mode in (
        ObservationMode(
            identifier="intrinsic-surface-distance",
            version=1,
            quantity="Riemannian distance between two points, measured in the surface",
            support={
                "constant-curvature": "exact",
                "declared-curvature-profile": "numerical",
                "parametric-surface": "unavailable",
                "physical-instrument": "unavailable",
            },
            note=(
                "What the Jacobi field predicts. Closed form on the model spaces "
                "(sn_K(d/2) = sn_K(s) sin(eps/2)); on a general parametric surface "
                "it is a boundary-value problem that is not solved here, and no "
                "instrument reports it directly."
            ),
        ),
        ObservationMode(
            identifier="ambient-euclidean-chord",
            version=1,
            quantity="straight-line distance in R^3 between two points of the surface",
            support={
                "constant-curvature": "exact",
                # A curvature profile is not an embedding, so there is no chord
                # to measure between two of its points.
                "declared-curvature-profile": "unavailable",
                "parametric-surface": "numerical",
                "physical-instrument": "unavailable",
            },
            note=(
                "What a reconstructed pair of 3-D coordinates gives. Differs from "
                "the intrinsic distance at second order in the separation, with a "
                "coefficient set by the transverse normal curvature. An instrument "
                "reports it only after calibration and reconstruction, which is a "
                "separate mode."
            ),
        ),
        ObservationMode(
            identifier="scanner-reconstructed-chord",
            version=0,
            quantity="ambient chord as reported by a calibrated metrology system",
            support={
                "constant-curvature": "unavailable",
                "declared-curvature-profile": "unavailable",
                "parametric-surface": "unavailable",
                "physical-instrument": "unavailable",
            },
            note=(
                "The ambient chord plus calibration, registration and fitting "
                "error. Needs a characterised instrument; nothing here models it."
            ),
        ),
        ObservationMode(
            identifier="camera-image-residual",
            version=0,
            quantity="residual in image coordinates, before reconstruction",
            support={
                "constant-curvature": "unavailable",
                "declared-curvature-profile": "unavailable",
                "parametric-surface": "unavailable",
                "physical-instrument": "unavailable",
            },
            note=(
                "Furthest from the model of the four, and the only one a camera "
                "actually produces. Not modelled here."
            ),
        ),
    )
}

DEFAULT_MODE = "intrinsic-surface-distance"


def mode(identifier: str) -> ObservationMode:
    try:
        return MODES[identifier]
    except KeyError as exc:  # pragma: no cover - guard
        raise KeyError(f"unknown observation mode {identifier!r}; have {sorted(MODES)}") from exc


def available_modes(domain: str) -> tuple[str, ...]:
    """Modes this repository can produce in ``domain``."""
    if domain not in DOMAINS:
        raise KeyError(f"unknown domain {domain!r}; have {DOMAINS}")
    return tuple(key for key, value in MODES.items() if value.is_available(domain))


def require_domain(domain: str) -> str:
    """The domain, or an error naming the ones that exist."""
    if domain not in DOMAINS:
        raise KeyError(f"unknown domain {domain!r}; have {DOMAINS}")
    return domain


def require_available(identifier: str, domain: str) -> ObservationMode:
    """The mode, if this repository can actually produce it in ``domain``.

    The combination is checked rather than each half, because each half is
    fine on its own and the pair is what fails. A record on a parametric
    surface tagged ``intrinsic-surface-distance`` names a real mode and a real
    domain and claims a quantity that would take a boundary-value solver
    nothing here has -- and it claims it in the one field a downstream
    comparison trusts to decide whether two numbers are comparable.
    """
    require_domain(domain)
    declared = mode(identifier)
    if not declared.is_available(domain):
        raise ValueError(
            f"observation mode {identifier!r} is {declared.support_in(domain)} on "
            f"{domain!r}; available there: {available_modes(domain) or '(none)'}. "
            f"{declared.note}"
        )
    return declared


def catalogue() -> list[dict[str, Any]]:
    """Every mode, for the report header."""
    return [value.to_dict() for value in MODES.values()]
