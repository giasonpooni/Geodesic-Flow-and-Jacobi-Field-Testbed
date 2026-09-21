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

#: Runs a full experiment stage or a perturbation sweep. See the ``numerical``
#: marker in pyproject.toml: CI runs this file once, on one interpreter, rather
#: than once per version of an interpreter that cannot change the answer.
pytestmark = pytest.mark.numerical

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
        curvature_interpolation="pchip-monotone",
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
        curvature_interpolation="pchip-monotone",
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


# -- the fit is auditable, and the bound is falsifiable ---------------------


def test_the_fit_records_which_probes_it_used_and_which_it_threw_away() -> None:
    """An adaptive selection nobody can inspect is not a measurement.

    Two ladders can produce the same bound from entirely different evidence.
    The window, the observed slopes, the coefficient and every rejected probe
    with its reason are carried so that the second case is distinguishable
    from the first.
    """
    envelope = _measure(sphere(1.0), u0=np.pi / 2, v0=0.0)
    assert {fit.direction for fit in envelope.fits} == set(envelope.directions)
    for fit in envelope.fits:
        assert fit.coefficient > 0.0
        assert fit.fitted_probes, "a fit with no probes is not a fit"
        assert all(abs(slope - 2.0) < 0.5 for slope in fit.observed_slopes)
        assert set(fit.fitted_probes) <= set(envelope.probe_magnitudes)
        rejected = {magnitude for magnitude, _ in fit.rejected_probes}
        assert rejected.isdisjoint(fit.fitted_probes)
        assert all(reason for _, reason in fit.rejected_probes)


def test_the_bound_is_re_probed_where_the_fit_did_not_look() -> None:
    """The check that can say the bound is wrong, rather than describe it.

    The relative error goes as ``C eps^2``, so at a fraction ``f`` of the
    bound it must come out at ``f^2`` times the tolerance. At ``1.2`` that is
    ``1.44`` -- above the tolerance, which is what makes the bound a place the
    linearisation fails rather than wherever the ladder stopped.
    """
    envelope = _measure(sphere(1.0), u0=np.pi / 2, v0=0.0)
    tolerance = envelope.relative_tolerance
    for fit in envelope.fits:
        held = dict(fit.held_out)
        assert set(held) == {0.8, 1.0, 1.2}
        assert held[1.0] == pytest.approx(tolerance, rel=0.02)
        assert held[0.8] == pytest.approx(0.64 * tolerance, rel=0.02)
        assert held[1.2] == pytest.approx(1.44 * tolerance, rel=0.02)
        assert held[1.2] > tolerance, (
            "a bound the linearisation survives past is not where it fails"
        )
        assert held[0.8] < tolerance


def test_a_probe_limited_column_says_so_instead_of_claiming_a_failure_point() -> None:
    """On a plane the lateral column is exact; there is no failure point to find."""
    envelope = _measure(plane(), u0=0.0, v0=0.0)
    lateral = next(fit for fit in envelope.fits if fit.direction == "lateral")
    assert lateral.probe_limited
    assert lateral.bound == pytest.approx(max(VALIDITY_PROBES))
    assert all(error == pytest.approx(0.0, abs=1e-12) for _, error in lateral.held_out), (
        "an exact column cannot exceed the tolerance anywhere, which is why its "
        "bound is reported as the ladder's end rather than fitted"
    )

    heading = next(fit for fit in envelope.fits if fit.direction == "heading")
    assert not heading.probe_limited, "the heading column on a plane is measured normally"


def test_the_bound_converges_under_step_refinement_with_the_declared_method() -> None:
    """RK4's bound is stable to a part in ten thousand as the step halves.

    That stability is what makes the bound a statement about the
    linearisation. It is not automatic: the probe and the transfer map it is
    compared against share the surface model, the geodesic right-hand side and
    the integrator, so the integrator's truncation error is common mode. The
    bound is only meaningful while that error sits well below the
    linearisation error being measured.
    """
    bounds = [
        _measure(sphere(1.0), u0=np.pi / 2, v0=0.0, n_steps=n).max_heading
        for n in (400, 800, 1600)
    ]
    for coarse, fine in zip(bounds, bounds[1:], strict=False):
        assert abs(coarse - fine) / fine < 1e-4, f"{coarse:.8f} -> {fine:.8f}"


def test_a_lower_order_integrator_is_not_good_enough_and_shows_it() -> None:
    """The common-mode sensitivity, exposed rather than assumed away.

    A second-order method at the same step count has truncation error
    comparable to the linearisation error at the small end of the ladder. The
    fit's adaptive window then keeps different probes on different runs and
    the bound stops converging: it moves by about two per cent and does so
    non-monotonically in the step count, which is the signature of noise in
    the fit rather than of a real trend.

    This is why the envelope declares ``reference_method`` and
    ``reference_samples``. A bound quoted without them is a bound whose
    numerics cannot be judged.
    """
    reference = _measure(sphere(1.0), u0=np.pi / 2, v0=0.0, n_steps=1600)
    assert reference.reference_method == "rk4"
    assert reference.reference_samples == 1600

    midpoint = [
        _measure(
            sphere(1.0), u0=np.pi / 2, v0=0.0, n_steps=n, method="midpoint"
        ).max_heading
        for n in (400, 800, 1600)
    ]
    drift = abs(midpoint[-1] - reference.max_heading) / reference.max_heading
    assert drift > 0.005, (
        "if a second-order method reproduced the bound, the declaration would "
        f"be pointless; it differs by {drift:.3%}"
    )
    steps = [abs(b - a) for a, b in zip(midpoint, midpoint[1:], strict=False)]
    assert steps[-1] > steps[0], (
        "the midpoint sequence should fail to settle, which is what distinguishes "
        f"a noisy fit from a converging one; got {midpoint}"
    )


def test_the_envelope_carries_its_fits_through_a_round_trip() -> None:
    envelope = _measure(hyperbolic_paraboloid(1.0), u0=0.3, v0=0.2, n_steps=400)
    reopened = ValidityEnvelope.from_dict(envelope.to_dict())
    assert len(reopened.fits) == len(envelope.fits)
    for before, after in zip(envelope.fits, reopened.fits, strict=True):
        assert after.direction == before.direction
        assert after.coefficient == pytest.approx(before.coefficient)
        assert after.fitted_probes == before.fitted_probes
        assert after.rejected_probes == before.rejected_probes
        assert after.held_out == before.held_out


def test_a_fit_for_a_direction_that_was_never_exercised_is_refused() -> None:
    from geodesic_testbed.engine.record import ProbeFit

    with pytest.raises(ValueError, match="never exercised"):
        ValidityEnvelope(
            basis="measured",
            relative_tolerance=1e-3,
            tolerance_basis="declared",
            directions=("heading",),
            max_heading=0.1,
            fits=(ProbeFit(direction="lateral", coefficient=1.0,
                           fitted_probes=(0.1,), observed_slopes=(2.0,)),),
        )
