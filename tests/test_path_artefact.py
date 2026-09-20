"""Acceptance tests for ``path-geometry-v1`` and the adapter that solves along it.

The adapter is the one place a path this runtime did not compute becomes a
transfer record, so it is the one place where a wrong answer would arrive
wearing the same clothes as a right one. Four kinds of test here:

*anchors* -- an artefact built from the plate, the rolled cylinder, the sphere
and the pseudosphere has to reproduce the closed forms the rest of the
repository is held to, through the adapter rather than around it;

*invariances* -- a rigid transform, a reparameterisation, a change of length
unit and a reversed frame each change something about the artefact and must
change the expected thing about the record, which for the first two is nothing
at all;

*convergence* -- the declared curvature interpolation is a choice with an
order, and the order is measured rather than asserted;

*refusals* -- the artefact validates on construction, so every field that can
be wrong is shown being wrong and being refused.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from geodesic_testbed.engine.contract import GeometryUncertainty, Units
from geodesic_testbed.engine.envelope import integrate_paths
from geodesic_testbed.engine.imported_path import (
    artefact_from_envelope,
    transfer_map_from_artefact,
    transfer_record_from_artefact,
)
from geodesic_testbed.engine.observation import available_modes
from geodesic_testbed.engine.path_artefact import (
    PATH_GEOMETRY_SCHEMA,
    PathGeometryArtefact,
    SamplingPolicy,
)
from geodesic_testbed.engine.surfaces import (
    cylinder,
    hyperbolic_paraboloid,
    plane,
    pseudosphere,
    sphere,
)

METRE = Units(length="metre", angle="radian")


def _envelope(surface, *, u0, v0, heading, length, n_steps):
    return integrate_paths(
        surface, u0=u0, v0=v0, headings=[heading], length=length, n_steps=n_steps
    )[0]


def _artefact(envelope, **overrides) -> PathGeometryArtefact:
    return artefact_from_envelope(
        envelope,
        identifier=overrides.pop("identifier", "test-path"),
        surface_digest=overrides.pop("surface_digest", "surface:deadbeef"),
        path_digest=overrides.pop("path_digest", "path:cafe1234"),
        units=overrides.pop("units", METRE),
        **overrides,
    )


# -- the anchors -----------------------------------------------------------

#: ``(name, surface, start, heading, K)``. The cylinder is the one that matters:
#: it is visibly curved, its geodesics are helices, and it is intrinsically flat,
#: so an artefact made from it must produce the plate's transfer map exactly --
#: from a completely different parameterisation, different positions and a
#: different frame.
ANCHORS = (
    ("plate", plane(), (0.3, -0.2), 0.7, 0.0),
    ("rolled-cylinder", cylinder(0.8), (0.4, 0.25), 0.9, 0.0),
    ("sphere", sphere(1.0), (0.5, 0.1), 0.3, 1.0),
    ("pseudosphere", pseudosphere(), (1.1, 0.2), 0.4, -1.0),
)


@pytest.mark.parametrize(("name", "surface", "start", "heading", "curvature"), ANCHORS)
def test_an_artefact_reproduces_the_closed_form_of_its_surface(
    name: str, surface, start: tuple[float, float], heading: float, curvature: float
) -> None:
    """Geometry in, through the artefact, and the model space still comes out."""
    from geodesic_testbed.engine.transfer import constant_curvature_transfer

    envelope = _envelope(
        surface, u0=start[0], v0=start[1], heading=heading, length=1.2, n_steps=1200
    )
    artefact = _artefact(envelope, identifier=name)
    assert np.allclose(artefact.gaussian_curvature, curvature, rtol=0.0, atol=1e-9)

    produced = transfer_map_from_artefact(artefact)
    closed = constant_curvature_transfer(artefact.arclength, curvature)
    for component in ("a", "a_rate", "b", "b_rate"):
        worst = float(np.max(np.abs(getattr(produced, component) - getattr(closed, component))))
        assert worst < 1e-10, f"{name}: {component} departs from the closed form by {worst:.3e}"


def test_the_rolled_cylinder_gives_the_plate_transfer_map_through_the_boundary() -> None:
    """The claim worth buying hardware to test, restated across the adapter.

    Two artefacts with different positions, different tangents, different
    transverse directions and different normal curvatures produce the same
    transfer map, because the only thing the Jacobi equation reads is ``K``.
    """
    plate = _artefact(_envelope(plane(), u0=0.0, v0=0.0, heading=0.4, length=1.5, n_steps=600))
    rolled = _artefact(
        _envelope(cylinder(0.7), u0=0.2, v0=0.1, heading=1.1, length=1.5, n_steps=600)
    )
    assert not np.allclose(plate.position, rolled.position)
    assert not np.allclose(
        plate.normal_curvature_transverse, rolled.normal_curvature_transverse
    )
    plate_map = transfer_map_from_artefact(plate)
    rolled_map = transfer_map_from_artefact(rolled)
    for component in ("a", "b", "a_rate", "b_rate"):
        worst = float(
            np.max(np.abs(getattr(plate_map, component) - getattr(rolled_map, component)))
        )
        assert worst < 1e-13, f"{component} differs by {worst:.3e}"


# -- the invariances -------------------------------------------------------


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return (
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def test_a_rigid_transform_of_the_artefact_leaves_the_transfer_map_untouched() -> None:
    """And exactly untouched, not nearly: the solver never reads a position.

    This is worth asserting bit for bit rather than to a tolerance, because a
    tolerance would also pass an adapter that had quietly started differencing
    the positions to recover a tangent.
    """
    from dataclasses import replace

    envelope = _envelope(
        hyperbolic_paraboloid(0.6), u0=0.1, v0=-0.2, heading=0.7, length=1.4, n_steps=500
    )
    artefact = _artefact(envelope)
    rotation = _rotation(np.array([0.3, -0.5, 0.8]), 0.9)
    offset = np.array([12.0, -3.5, 0.25])
    moved = replace(
        artefact,
        position=artefact.position @ rotation.T + offset,
        tangent=artefact.tangent @ rotation.T,
        transverse=artefact.transverse @ rotation.T,
    )
    original = transfer_map_from_artefact(artefact)
    transformed = transfer_map_from_artefact(moved)
    for component in ("a", "a_rate", "b", "b_rate"):
        assert np.array_equal(
            getattr(original, component), getattr(transformed, component)
        ), component


def test_the_same_geodesic_from_a_different_parameterisation_gives_the_same_map() -> None:
    """A sphere geodesic is a great circle wherever it starts.

    Two artefacts from two different starts and two different headings describe
    the same intrinsic path up to where it begins, so their transfer maps agree
    -- which is the parameterisation invariance restated as something the
    adapter can be held to.
    """
    first = _artefact(_envelope(sphere(1.0), u0=0.4, v0=0.0, heading=0.2, length=1.3, n_steps=900))
    second = _artefact(
        _envelope(sphere(1.0), u0=1.2, v0=2.4, heading=1.9, length=1.3, n_steps=900)
    )
    a = transfer_map_from_artefact(first)
    b = transfer_map_from_artefact(second)
    assert float(np.max(np.abs(a.b - b.b))) < 1e-11
    assert float(np.max(np.abs(a.a - b.a))) < 1e-11


def test_resampling_the_same_path_onto_a_different_grid_gives_the_same_map() -> None:
    """A producer that resampled has not changed the surface.

    The comparison is at the arc lengths the coarse grid keeps, and the
    tolerance is the coarse grid's own interpolation error rather than
    roundoff: resampling genuinely loses information about ``K`` between the
    samples that survive, and pretending otherwise would be the silent choice
    this artefact exists to prevent.
    """
    envelope = _envelope(
        hyperbolic_paraboloid(0.5), u0=0.2, v0=0.1, heading=0.6, length=1.6, n_steps=1600
    )
    fine = _artefact(envelope)
    fine_map = transfer_map_from_artefact(fine)

    strides = (32, 16, 8, 4)
    residuals = []
    for stride in strides:
        keep = np.arange(0, fine.samples, stride)
        coarse = _resampled(fine, keep)
        assert coarse.samples < fine.samples
        coarse_map = transfer_map_from_artefact(coarse)
        residuals.append(float(np.max(np.abs(fine_map.b[keep] - coarse_map.b))))

    order, _ = np.polyfit(np.log(np.asarray(strides, dtype=float)), np.log(residuals), 1)
    assert order > 3.5, f"resampling residual fell at order {order:.2f}, not the interpolant's"
    # The order is the claim; this is the floor under it. An adapter that had
    # started reading something other than the curvature would disagree by
    # O(1), not by a part in ten billion of a b that is of order one.
    assert residuals[-1] < 1e-9, f"resampling moved b by {residuals[-1]:.3e} at stride 4"


def _resampled(artefact: PathGeometryArtefact, keep: np.ndarray) -> PathGeometryArtefact:
    from dataclasses import replace

    return replace(
        artefact,
        arclength=artefact.arclength[keep],
        position=artefact.position[keep],
        tangent=artefact.tangent[keep],
        transverse=artefact.transverse[keep],
        gaussian_curvature=artefact.gaussian_curvature[keep],
        normal_curvature_along=artefact.normal_curvature_along[keep],
        normal_curvature_transverse=artefact.normal_curvature_transverse[keep],
        mean_curvature=artefact.mean_curvature[keep],
        validity=None,
        sampling=SamplingPolicy(policy="nonuniform-arclength", note="resampled for a test"),
    )


def test_a_change_of_length_unit_moves_every_quantity_by_its_own_exponent() -> None:
    """``a`` is dimensionless, ``b`` is a length, ``K`` is an inverse area.

    Getting this wrong is not a rounding error: metres against millimetres puts
    ``K`` out by a factor of a million, which is the difference between a plate
    and a conjugate point inside the coupon.
    """
    artefact = _artefact(
        _envelope(sphere(1.0), u0=0.5, v0=0.1, heading=0.3, length=1.2, n_steps=800)
    )
    scale = 1000.0
    millimetres = artefact.converted_to("millimetre", scale)

    assert millimetres.units.length == "millimetre"
    assert np.allclose(millimetres.arclength, artefact.arclength * scale, rtol=1e-14, atol=0.0)
    assert np.allclose(
        millimetres.gaussian_curvature,
        artefact.gaussian_curvature / scale**2,
        rtol=1e-14,
        atol=0.0,
    )
    assert np.allclose(
        millimetres.normal_curvature_transverse,
        artefact.normal_curvature_transverse / scale,
        rtol=1e-14,
        atol=0.0,
    )

    metres = transfer_map_from_artefact(artefact)
    scaled = transfer_map_from_artefact(millimetres)
    assert float(np.max(np.abs(scaled.a - metres.a))) < 1e-11
    assert float(np.max(np.abs(scaled.b - metres.b * scale))) < 1e-8
    assert float(np.max(np.abs(scaled.b_rate - metres.b_rate))) < 1e-11


def test_reversing_the_transverse_direction_flips_the_curvatures_and_not_the_map() -> None:
    """The frame convention is the producer's; the sensitivity is the surface's.

    ``K`` is intrinsic and does not move. The normal curvatures and the mean
    curvature are measured against the normal, which flips with the transverse
    direction, so all three change sign -- and Euler's theorem still holds,
    which is what the artefact re-checks on construction.
    """
    artefact = _artefact(
        _envelope(hyperbolic_paraboloid(0.7), u0=0.1, v0=0.2, heading=0.5, length=1.2, n_steps=600)
    )
    flipped = artefact.with_reversed_transverse()

    assert np.array_equal(flipped.gaussian_curvature, artefact.gaussian_curvature)
    assert np.allclose(flipped.transverse, -artefact.transverse)
    assert np.allclose(flipped.surface_normal(), -artefact.surface_normal())
    for name in ("normal_curvature_along", "normal_curvature_transverse", "mean_curvature"):
        assert np.allclose(getattr(flipped, name), -getattr(artefact, name))

    original = transfer_map_from_artefact(artefact)
    reversed_map = transfer_map_from_artefact(flipped)
    for component in ("a", "a_rate", "b", "b_rate"):
        assert np.array_equal(
            getattr(original, component), getattr(reversed_map, component)
        ), component


# -- the declared interpolation --------------------------------------------


@pytest.mark.parametrize(
    ("policy", "expected_order"), (("pchip-monotone", 3.5), ("linear", 1.8))
)
def test_the_declared_curvature_interpolation_converges_at_its_own_order(
    policy: str, expected_order: float
) -> None:
    """Which interpolant fills in ``K`` between samples is not a free choice.

    Measured against a path solved on the surface at 6400 steps, the monotone
    cubic converges at fourth order and the piecewise-linear one at second. At
    a hundred samples that is a factor of four hundred in the answer, from a
    field a careless producer would leave at its default -- which is why the
    policy is a required part of the artefact rather than a convention.
    """
    surface = hyperbolic_paraboloid(0.6)
    reference = _envelope(
        surface, u0=0.1, v0=-0.2, heading=0.7, length=1.5, n_steps=6400
    ).transfer_map
    target = float(reference.b[-1])

    counts = (100, 200, 400, 800)
    errors = []
    for n_steps in counts:
        artefact = _artefact(
            _envelope(surface, u0=0.1, v0=-0.2, heading=0.7, length=1.5, n_steps=n_steps),
            curvature_interpolation=policy,
        )
        errors.append(abs(float(transfer_map_from_artefact(artefact).b[-1]) - target))

    order, _ = np.polyfit(np.log(1.0 / np.asarray(counts, dtype=float)), np.log(errors), 1)
    assert order > expected_order, f"{policy} converged at order {order:.2f}"


def test_the_two_interpolants_actually_disagree_at_a_sampling_a_producer_might_use() -> None:
    """Otherwise the declared policy would be bookkeeping rather than a choice."""
    surface = hyperbolic_paraboloid(0.6)
    envelope = _envelope(surface, u0=0.1, v0=-0.2, heading=0.7, length=1.5, n_steps=100)
    maps = {
        policy: transfer_map_from_artefact(_artefact(envelope, curvature_interpolation=policy))
        for policy in ("pchip-monotone", "linear", "piecewise-constant")
    }
    cubic = float(maps["pchip-monotone"].b[-1])
    linear = float(maps["linear"].b[-1])
    constant = float(maps["piecewise-constant"].b[-1])
    assert abs(linear - cubic) / abs(cubic) > 1e-6, (
        "the two interpolants agree to better than a part in a million here, so "
        "declaring which one was used would be bookkeeping rather than a choice"
    )
    assert abs(constant - cubic) > abs(linear - cubic)


def test_the_monotone_interpolant_does_not_invent_a_sign_change_in_the_curvature() -> None:
    """A cubic that overshoots turns a flat region into a focusing one.

    The profile is flat, then bends: a natural spline rings at the corner and
    dips ``K`` below zero, which a Jacobi solver reads as a patch of hyperbolic
    surface that is not there. The monotone interpolant cannot.
    """
    grid = np.linspace(0.0, 1.0, 11)
    curvature = np.where(grid < 0.5, 0.0, (grid - 0.5) * 4.0)
    artefact = _flat_artefact(grid, curvature)
    sampled = np.array(
        [artefact.curvature_interpolant()(s) for s in np.linspace(0.0, 1.0, 501)]
    )
    assert sampled.min() >= -1e-15, f"the interpolant dipped to {sampled.min():.3e}"
    assert sampled.max() <= curvature.max() + 1e-15


def _flat_artefact(grid: np.ndarray, curvature: np.ndarray) -> PathGeometryArtefact:
    n = grid.size
    return PathGeometryArtefact(
        identifier="synthetic",
        arclength=grid,
        position=np.column_stack([grid, np.zeros(n), np.zeros(n)]),
        tangent=np.tile([1.0, 0.0, 0.0], (n, 1)),
        transverse=np.tile([0.0, 1.0, 0.0], (n, 1)),
        gaussian_curvature=curvature,
        units=METRE,
        surface_digest="surface:synthetic",
        path_digest="path:synthetic",
    )


# -- the record it produces ------------------------------------------------


def test_the_record_carries_the_artefact_and_not_this_runtime_s_defaults() -> None:
    envelope = _envelope(sphere(1.0), u0=0.5, v0=0.1, heading=0.3, length=1.2, n_steps=400)
    artefact = _artefact(
        envelope,
        identifier="sphere-cap-run-7",
        surface_digest="surface:sha256:aa",
        path_digest="path:sha256:bb",
        units=Units(length="millimetre", angle="radian"),
        uncertainty=GeometryUncertainty(
            position=0.012, normal=0.004, curvature=1e-5, basis="as-built-scan", note="from a scan"
        ),
    )
    record = transfer_record_from_artefact(artefact)

    assert record.domain == "imported-path-artefact"
    assert record.units.length == "millimetre"
    assert record.source_digest == "path:sha256:bb"
    assert record.frame == artefact.frame
    assert record.geometry is not None
    assert record.geometry.uncertainty.basis == "as-built-scan"
    assert not record.covariance.declared
    assert not record.resolution.convergence.established

    upstream = record.provenance.upstream
    assert len(upstream) == 1
    assert upstream[0].kind == "path-geometry"
    assert upstream[0].identifier == "sphere-cap-run-7"
    assert upstream[0].digest == "path:sha256:bb"
    assert PATH_GEOMETRY_SCHEMA in upstream[0].note
    assert "pchip-monotone" in upstream[0].note


def test_a_record_from_an_artefact_holds_the_determinant_invariant() -> None:
    artefact = _artefact(
        _envelope(pseudosphere(), u0=1.1, v0=0.2, heading=0.4, length=1.2, n_steps=800)
    )
    record = transfer_record_from_artefact(artefact)
    worst = float(np.max(np.abs(record.determinant - 1.0)))
    assert worst < 1e-12, f"det Phi departs from 1 by {worst:.3e}"


def test_the_saved_artefact_replays_offline_to_the_same_record() -> None:
    """The round trip is the deliverable: an artefact is a file, not an object."""
    artefact = _artefact(
        _envelope(hyperbolic_paraboloid(0.5), u0=0.2, v0=0.1, heading=0.6, length=1.3, n_steps=500)
    )
    reopened = PathGeometryArtefact.from_dict(json.loads(json.dumps(artefact.to_dict())))

    assert reopened.identifier == artefact.identifier
    assert reopened.curvature_interpolation == artefact.curvature_interpolation
    assert reopened.units.to_dict() == artefact.units.to_dict()
    assert reopened.sampling.to_dict() == artefact.sampling.to_dict()
    assert np.array_equal(reopened.arclength, artefact.arclength)
    assert np.array_equal(reopened.gaussian_curvature, artefact.gaussian_curvature)

    first = transfer_record_from_artefact(artefact)
    second = transfer_record_from_artefact(reopened)
    for component in ("a", "a_rate", "b", "b_rate"):
        assert np.array_equal(getattr(first, component), getattr(second, component)), component


# -- the refusals ----------------------------------------------------------


def _fields(**overrides):
    grid = np.linspace(0.0, 1.0, 9)
    n = grid.size
    base = {
        "identifier": "refusal",
        "arclength": grid,
        "position": np.column_stack([grid, np.zeros(n), np.zeros(n)]),
        "tangent": np.tile([1.0, 0.0, 0.0], (n, 1)),
        "transverse": np.tile([0.0, 1.0, 0.0], (n, 1)),
        "gaussian_curvature": np.zeros(n),
        "units": METRE,
        "surface_digest": "surface:x",
        "path_digest": "path:x",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("what", "overrides", "message"),
    (
        ("no surface digest", {"surface_digest": ""}, "digest"),
        ("no path digest", {"path_digest": ""}, "digest"),
        ("no identifier", {"identifier": ""}, "name itself"),
        ("undeclared length unit", {"units": Units()}, "length unit"),
        ("no angle unit", {"units": Units(length="metre", angle="")}, "angle unit"),
        (
            "a tangent that is not unit",
            {"tangent": np.tile([1.0, 0.0, 0.3], (9, 1))},
            "unit vectors",
        ),
        (
            "a grid that steps backwards",
            {"arclength": np.array([0.0, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])},
            "strictly increasing",
        ),
        (
            "a repeated arc length",
            {"arclength": np.array([0.0, 0.1, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])},
            "strictly increasing",
        ),
        (
            "a triad that is not orthogonal",
            {"transverse": np.tile([1.0, 0.0, 0.0], (9, 1))},
            "orthogonal",
        ),
        ("a curvature that is not finite", {"gaussian_curvature": np.full(9, np.nan)}, "finite"),
        ("an unknown frame", {"frame": "whatever-the-producer-felt-like"}, "frame must be"),
        (
            "an unknown interpolation policy",
            {"curvature_interpolation": "spline"},
            "curvature_interpolation",
        ),
        ("an unknown sampling policy", {"sampling": None}, ""),
        ("an unknown upstream status", {"upstream_status": "probably-fine"}, "upstream_status"),
        (
            "a degraded status with no reason",
            {"upstream_status": "degraded"},
            "say what went wrong",
        ),
        (
            "two of the three curvatures",
            {
                "normal_curvature_along": np.zeros(9),
                "normal_curvature_transverse": np.zeros(9),
            },
            "declared together",
        ),
        (
            "a mean curvature Euler's theorem refuses",
            {
                "normal_curvature_along": np.full(9, 2.0),
                "normal_curvature_transverse": np.full(9, 3.0),
                "mean_curvature": np.full(9, 9.0),
            },
            "Euler",
        ),
    ),
)
def test_the_artefact_refuses_what_it_cannot_be_trusted_about(
    what: str, overrides: dict, message: str
) -> None:
    if what == "an unknown sampling policy":
        with pytest.raises(ValueError, match="sampling policy"):
            SamplingPolicy(policy="wherever")
        return
    with pytest.raises(ValueError, match=message):
        PathGeometryArtefact(**_fields(**overrides))


def test_an_artefact_its_producer_already_failed_is_not_integrated() -> None:
    """A path that is wrong in a way upstream detected looks exactly like one that is not."""
    artefact = PathGeometryArtefact(
        **_fields(upstream_status="failed", upstream_note="the mesh trace left the patch")
    )
    with pytest.raises(ValueError, match="upstream_status='failed'"):
        transfer_map_from_artefact(artefact)
    with pytest.raises(ValueError, match="upstream_status='failed'"):
        transfer_record_from_artefact(artefact)


def test_a_degraded_artefact_is_integrated_and_says_so_in_the_record() -> None:
    artefact = PathGeometryArtefact(
        **_fields(
            upstream_status="degraded",
            upstream_note="curvature estimated on a coarser patch near the boundary",
        )
    )
    record = transfer_record_from_artefact(artefact)
    assert "degraded" in record.provenance.note
    assert "coarser patch" in record.provenance.note


def test_an_observation_mode_this_domain_cannot_produce_is_refused() -> None:
    """The intrinsic distance needs a boundary-value solver; an artefact is not one."""
    assert "intrinsic-surface-distance" not in available_modes("imported-path-artefact")
    artefact = PathGeometryArtefact(**_fields())
    with pytest.raises(ValueError, match="intrinsic-surface-distance"):
        transfer_record_from_artefact(artefact, observation_mode="intrinsic-surface-distance")


def test_a_chord_mode_is_refused_when_the_artefact_declared_no_normal_curvature() -> None:
    """The domain supports the chord; this artefact did not supply the evidence.

    Two different questions, and the failure modes differ: the first would be a
    claim about the repository, the second is a claim about one file.
    """
    assert "ambient-euclidean-chord" in available_modes("imported-path-artefact")
    artefact = PathGeometryArtefact(**_fields())
    assert not artefact.carries_normal_curvature
    with pytest.raises(ValueError, match="transverse normal curvature"):
        transfer_record_from_artefact(artefact, observation_mode="ambient-euclidean-chord")

    carrying = _artefact(
        _envelope(sphere(1.0), u0=0.5, v0=0.1, heading=0.3, length=1.0, n_steps=200)
    )
    record = transfer_record_from_artefact(carrying, observation_mode="ambient-euclidean-chord")
    assert record.observation_mode == "ambient-euclidean-chord"


def test_a_record_with_no_geometry_is_what_an_artefact_without_curvatures_produces() -> None:
    """Undeclared rather than filled in: there is nothing to compute it from."""
    record = transfer_record_from_artefact(PathGeometryArtefact(**_fields()))
    assert record.geometry is None


def test_the_reader_refuses_a_schema_it_does_not_understand() -> None:
    payload = PathGeometryArtefact(**_fields()).to_dict()
    payload["schema"] = "path-geometry-v2"
    with pytest.raises(ValueError, match="path-geometry-v1"):
        PathGeometryArtefact.from_dict(payload)
