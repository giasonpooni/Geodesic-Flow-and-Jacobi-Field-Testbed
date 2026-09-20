"""The perturbation range the linear map holds to, measured against the flow.

``Phi dz0`` is the first term of a series whose second term is the same order
as the effect most campaigns here are trying to resolve, so "over what range
is this the answer" is a number and not a caveat. These tests hold the measured
number to the closed forms where one exists, show it moving for a reason where
one does not, and show it refusing to be invented where it cannot be measured
at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from geodesic_testbed.engine.contract import Units
from geodesic_testbed.engine.envelope import (
    VALIDITY_PROBES,
    integrate_paths,
    measure_validity_envelope,
)
from geodesic_testbed.engine.imported_path import (
    artefact_from_envelope,
    transfer_record_from_artefact,
)
from geodesic_testbed.engine.record import ValidityEnvelope
from geodesic_testbed.engine.surfaces import (
    cylinder,
    hyperbolic_paraboloid,
    plane,
    pseudosphere,
    sphere,
)

TOLERANCE = 1e-3
BASIS = "declared before the probe was run, from the pilot's 25 um metrology floor"


def _measure(surface, *, u0, v0, heading=0.6, length=1.5, n_steps=800, **overrides):
    return measure_validity_envelope(
        surface,
        u0=u0,
        v0=v0,
        heading=heading,
        length=length,
        n_steps=n_steps,
        relative_tolerance=overrides.pop("relative_tolerance", TOLERANCE),
        tolerance_basis=overrides.pop("tolerance_basis", BASIS),
        source_digest="surface:closed-form",
        **overrides,
    )


# -- anchored to the closed forms -----------------------------------------


@pytest.mark.parametrize(
    ("name", "surface", "start"),
    (("plate", plane(), (0.0, 0.0)), ("spherical-cap", sphere(1.0), (np.pi / 2, 0.0))),
)
def test_the_measured_envelope_reproduces_the_closed_form(
    name: str, surface, start: tuple[float, float]
) -> None:
    """``sqrt(24 tol)`` where the chord and the intrinsic separation coincide.

    The relative error of the first-order prediction against an ambient chord
    goes as ``(cn_K(s)^2 + kappa_n^2 sn_K(s)^2) eps^2 / 24``. On the plate the
    normal curvature is zero and ``cn_K = 1``; on the unit sphere every
    direction is a principal direction with ``kappa_n = 1``, so
    ``cos^2 + sin^2`` is one. Both coefficients are ``1/24`` and both bounds
    are the same number, arrived at completely differently.
    """
    envelope = _measure(surface, u0=start[0], v0=start[1])
    expected = float(np.sqrt(24.0 * TOLERANCE))
    assert envelope.established
    assert envelope.max_heading == pytest.approx(expected, rel=1e-3), name
    assert envelope.reference == "geodesic-flow-central-difference"
    assert set(envelope.directions) == {"lateral", "heading"}


def test_the_rolled_cylinder_is_tighter_than_the_plate_by_exactly_the_chord_term() -> None:
    """Same intrinsic curvature, different envelope, and the difference is a chord.

    A cylinder is intrinsically flat, so its transfer map is the plate's to
    1e-13. Its validity envelope is not: the measurement is an ambient chord,
    and the cylinder has a transverse normal curvature the plate does not, so
    the second-order term is ``(1 + kappa_n^2 s^2)/24`` instead of ``1/24``.

    This is the observation mode made into a number. Two paths with identical
    sensitivity admit different perturbations, and a campaign that carried the
    plate's envelope onto the cylinder would be over by ten per cent.
    """
    rolled = cylinder(1.0)
    envelope = _measure(rolled, u0=0.0, v0=0.0)
    plate = _measure(plane(), u0=0.0, v0=0.0)

    path = integrate_paths(
        rolled, u0=0.0, v0=0.0, headings=[0.6], length=1.5, n_steps=800
    )[0]
    geometry = path.path_geometry()
    coefficient = 1.0 + geometry.normal_curvature_transverse**2 * path.arc_length**2
    predicted = float(np.sqrt(24.0 * TOLERANCE / coefficient.max()))

    assert envelope.max_heading == pytest.approx(predicted, rel=5e-3)
    assert envelope.max_heading < 0.95 * plate.max_heading, (
        "the cylinder's envelope has to be visibly tighter, or the chord term "
        "is not being measured at all"
    )


def test_the_transfer_maps_agree_where_the_envelopes_do_not() -> None:
    """The half of the cylinder story that makes the other half surprising."""
    rolled = integrate_paths(
        cylinder(1.0), u0=0.0, v0=0.0, headings=[0.6], length=1.5, n_steps=800
    )[0].transfer_map
    flat = integrate_paths(
        plane(), u0=0.0, v0=0.0, headings=[0.6], length=1.5, n_steps=800
    )[0].transfer_map
    assert float(np.max(np.abs(rolled.b - flat.b))) < 1e-13
    assert float(np.max(np.abs(rolled.a - flat.a))) < 1e-13


def test_a_hyperbolic_surface_admits_less_than_a_flat_one() -> None:
    """``cn_K = cosh`` grows, so the second-order term does too."""
    hyperbolic = _measure(pseudosphere(), u0=1.2, v0=0.0)
    flat = _measure(plane(), u0=0.0, v0=0.0)
    assert hyperbolic.max_heading < 0.5 * flat.max_heading
    assert hyperbolic.max_lateral < 0.5 * flat.max_lateral


# -- what it refuses to claim ---------------------------------------------


def test_an_exact_column_reports_the_ladder_and_not_an_extrapolation() -> None:
    """On a plane a lateral offset gives a parallel line; ``a`` is exactly 1.

    The fitted quadratic coefficient is then roundoff and the extrapolated
    bound is hundreds of radians. What was established is that the
    linearisation held out to the largest perturbation tested, so that is what
    is reported, and ``probe_limited`` says which kind of statement it is.
    """
    envelope = _measure(plane(), u0=0.0, v0=0.0)
    assert envelope.probe_limited
    assert envelope.max_lateral == pytest.approx(max(VALIDITY_PROBES))
    assert "clipped to the end of the ladder" in envelope.note


def test_an_imported_artefact_leaves_the_envelope_not_established() -> None:
    """It carries one path, and the surface the neighbours live on stayed upstream."""
    path = integrate_paths(
        sphere(1.0), u0=np.pi / 2, v0=0.0, headings=[0.4], length=1.0, n_steps=200
    )[0]
    artefact = artefact_from_envelope(
        path,
        identifier="imported",
        surface_digest="surface:x",
        path_digest="path:x",
        units=Units(length="metre", angle="radian"),
    )
    record = transfer_record_from_artefact(artefact)
    assert not record.validity.established
    assert "neighbouring paths" in record.validity.basis
    assert record.validity.admits(0.0, 1e-9) is False


def test_a_producer_that_measured_the_envelope_upstream_may_supply_one() -> None:
    path = integrate_paths(
        sphere(1.0), u0=np.pi / 2, v0=0.0, headings=[0.4], length=1.0, n_steps=200
    )[0]
    artefact = artefact_from_envelope(
        path,
        identifier="imported",
        surface_digest="surface:x",
        path_digest="path:x",
        units=Units(length="metre", angle="radian"),
    )
    declared = ValidityEnvelope(
        basis="measured upstream against the mesh flow",
        relative_tolerance=1e-3,
        tolerance_basis="the producer's declared metrology floor",
        max_heading=0.12,
        max_lateral=0.09,
        directions=("lateral", "heading"),
        reference="declared-by-caller",
    )
    record = transfer_record_from_artefact(artefact, validity=declared)
    assert record.validity.established
    assert record.validity.admits(0.05, 0.1)
    assert not record.validity.admits(0.05, 0.2)


def test_an_undeclared_envelope_admits_nothing() -> None:
    """"We never checked" must not read as "it always holds"."""
    envelope = ValidityEnvelope.not_established("no probe was run")
    assert not envelope.admits(1e-12, 0.0)
    assert not envelope.admits(0.0, 1e-12)
    assert not envelope.admits(0.0, 0.0), (
        "an envelope that was never established does not admit even the nominal "
        "path, because the statement it would be making is about a range it has none of"
    )


def test_a_bound_established_in_one_direction_does_not_cover_the_other() -> None:
    """A probe that perturbs only the heading measures ``b`` and nothing else."""
    heading_only = ValidityEnvelope(
        basis="closed-form: relative error is cn_K(s)^2 eps^2 / 24",
        relative_tolerance=1e-3,
        tolerance_basis="declared",
        max_heading=0.15,
        directions=("heading",),
        reference="closed-form",
    )
    assert heading_only.admits(0.0, 0.1)
    assert not heading_only.admits(0.01, 0.1), "lateral was never exercised"


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"directions": ()}, "at least one column"),
        ({"directions": ("sideways",)}, "drawn from"),
        ({"relative_tolerance": 0.0}, "finite and positive"),
        ({"tolerance_basis": "not-declared"}, "how its tolerance was chosen"),
        ({"tolerance_basis": ""}, "how its tolerance was chosen"),
        ({"probes": (0.01, 0.02)}, "at least three probes"),
    ),
)
def test_the_probe_refuses_a_measurement_that_would_not_mean_anything(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _measure(plane(), u0=0.0, v0=0.0, n_steps=100, **overrides)


def test_an_established_envelope_must_carry_what_makes_it_readable() -> None:
    for overrides, message in (
        ({"directions": ()}, "which perturbation directions"),
        ({"relative_tolerance": None}, "error tolerance"),
        ({"tolerance_basis": "not-declared"}, "how its tolerance was chosen"),
    ):
        fields = {
            "basis": "measured",
            "relative_tolerance": 1e-3,
            "tolerance_basis": "declared",
            "directions": ("heading",),
            "max_heading": 0.1,
        }
        fields.update(overrides)
        with pytest.raises(ValueError, match=message):
            ValidityEnvelope(**fields)


def test_the_envelope_survives_a_round_trip_through_the_record() -> None:
    envelope = _measure(hyperbolic_paraboloid(1.0), u0=0.3, v0=0.2, n_steps=400)
    payload = envelope.to_dict()
    reopened = ValidityEnvelope.from_dict(payload)
    assert reopened.max_heading == pytest.approx(envelope.max_heading)
    assert reopened.max_lateral == pytest.approx(envelope.max_lateral)
    assert reopened.directions == envelope.directions
    assert reopened.probe_limited == envelope.probe_limited
    assert reopened.tolerance_basis == envelope.tolerance_basis
    assert reopened.reference == envelope.reference


def test_a_tighter_tolerance_gives_a_smaller_envelope_at_the_expected_rate() -> None:
    """``eps ~ sqrt(tolerance)``, so a hundredfold tolerance is a tenfold range."""
    loose = _measure(sphere(1.0), u0=np.pi / 2, v0=0.0, relative_tolerance=1e-2)
    tight = _measure(sphere(1.0), u0=np.pi / 2, v0=0.0, relative_tolerance=1e-4)
    assert loose.max_heading / tight.max_heading == pytest.approx(10.0, rel=0.02)
