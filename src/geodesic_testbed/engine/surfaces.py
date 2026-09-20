"""Parametric surfaces, where the curvature varies along the path.

The constant-curvature testbed in :mod:`geodesic_testbed.engine.spaceforms` is the calibration
stage: there the Jacobi equation has a closed-form solution to check against.
A real workpiece does not. This module takes the same two objects -- the
geodesic flow and the Jacobi equation -- onto an arbitrary parametric surface
``r(u, v)``, where ``K`` is a function of position and the Jacobi equation

.. math::

    j''(s) + K(\\gamma(s))\\, j(s) = 0

has to be integrated along the path together with the path itself.

Everything is intrinsic. The state is ``(u, v, u', v', j, j')`` and the
geodesic is advanced by its Christoffel symbols, so the trajectory cannot drift
off the surface the way an ambient formulation can -- it is *defined* in the
surface's own coordinates. What can drift is the unit-speed condition, and that
is reported rather than enforced.

A surface may supply its analytic first and second derivatives; if it does not,
they are taken by central differences of ``r`` alone, so a surface can be added
by writing one function. The cost of that convenience is measured rather than
assumed: :mod:`geodesic_testbed.engine.experiment_surfaces` compares the two paths on the same
surfaces and reports the difference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

Array = np.ndarray

# (a, a', b, b') at s = 0: the identity transfer map.
TRANSFER_INITIAL_STATE = (1.0, 0.0, 0.0, 1.0)

INFINITY = float("inf")


@dataclass(frozen=True)
class Chart:
    """The parameter region in which a surface's chart is actually a chart.

    A parameterisation is not the surface. Latitude and longitude cover a
    sphere everywhere except the poles, where the longitude direction collapses
    and the Christoffel symbols diverge; the pseudosphere's ``u`` must stay
    positive and bounded. A path that leaves the region does not stop being a
    path -- it stops being *this chart's* path -- and the distinction has to be
    reported rather than left to appear as NaNs or, worse, as a plausible
    envelope computed from a degenerate metric.
    """

    u_min: float = -INFINITY
    u_max: float = INFINITY
    v_min: float = -INFINITY
    v_max: float = INFINITY
    u_period: float | None = None
    v_period: float | None = None
    #: Declared coordinate scales and a reference length, so that a collapsing
    #: coordinate direction can be detected without smuggling in the units the
    #: part happens to be drawn in.
    u_scale: float = 1.0
    v_scale: float = 1.0
    reference_length: float = 1.0
    note: str = ""

    def margin(self, u, v) -> Array:
        """Distance to the nearest non-periodic boundary; ``inf`` when unbounded.

        Negative outside. Periodic coordinates are unbounded by construction and
        contribute nothing.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        margins = [np.full(np.broadcast(u, v).shape, INFINITY)]
        if self.u_period is None:
            margins.extend([u - self.u_min, self.u_max - u])
        if self.v_period is None:
            margins.extend([v - self.v_min, self.v_max - v])
        return np.min(np.broadcast_arrays(*margins), axis=0)

    def contains(self, u, v) -> Array:
        return self.margin(u, v) >= 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "u": [self.u_min, self.u_max],
            "v": [self.v_min, self.v_max],
            "u_period": self.u_period,
            "v_period": self.v_period,
            "u_scale": self.u_scale,
            "v_scale": self.v_scale,
            "reference_length": self.reference_length,
            "note": self.note,
        }


@dataclass(frozen=True)
class SurfaceJet:
    """A surface's position and its first two derivatives at a parameter point."""

    r: Array
    ru: Array
    rv: Array
    ruu: Array
    ruv: Array
    rvv: Array


@dataclass(frozen=True)
class ParametricSurface:
    """A surface ``r(u, v)`` in ``R^3``, with intrinsic geodesic and Jacobi flow."""

    name: str
    position: Callable[[Any, Any], Array]
    jet: Callable[[Any, Any], SurfaceJet] | None = None
    exact_curvature: Callable[[Any, Any], Array] | None = None
    #: Step for the finite-difference fallback, *relative* to the parameter
    #: scale. An absolute step is a scale defect: the same 1e-4 that is right
    #: for a unit torus is far too small on a part parameterised in millimetres
    #: and far too large on one in metres, and a second derivative amplifies
    #: both failures.
    fd_relative_step: float = 1e-4
    #: The parameter magnitude below which the relative step stops shrinking.
    fd_floor: float = 1.0
    chart: Chart = field(default_factory=Chart)
    #: A chart whose dimensionless conditioning falls below this is degenerate
    #: -- a coordinate direction is collapsing or the coordinate curves are
    #: becoming parallel -- and nothing computed from it can be trusted.
    chart_conditioning_floor: float = 1e-6
    description: str = ""

    # -- derivatives -------------------------------------------------------
    def jet_at(self, u, v) -> SurfaceJet:
        if self.jet is not None:
            return self.jet(u, v)
        return self._finite_difference_jet(u, v)

    def fd_step_at(self, u, v) -> float:
        """Scale-aware differencing step for the fallback jet."""
        scale = max(
            float(self.fd_floor),
            float(np.max(np.abs(np.asarray(u, dtype=float)))),
            float(np.max(np.abs(np.asarray(v, dtype=float)))),
        )
        return float(self.fd_relative_step) * scale

    def _finite_difference_jet(self, u, v) -> SurfaceJet:
        h = self.fd_step_at(u, v)
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        r = self.position(u, v)
        up, um = self.position(u + h, v), self.position(u - h, v)
        vp, vm = self.position(u, v + h), self.position(u, v - h)
        upvp = self.position(u + h, v + h)
        upvm = self.position(u + h, v - h)
        umvp = self.position(u - h, v + h)
        umvm = self.position(u - h, v - h)
        return SurfaceJet(
            r=r,
            ru=(up - um) / (2.0 * h),
            rv=(vp - vm) / (2.0 * h),
            ruu=(up - 2.0 * r + um) / h**2,
            ruv=(upvp - upvm - umvp + umvm) / (4.0 * h**2),
            rvv=(vp - 2.0 * r + vm) / h**2,
        )

    def derivative_convergence(self, u, v) -> dict[str, float]:
        """How much the fallback jet's curvature moves when the step is doubled.

        A Richardson-style estimate of the finite-difference error, so a surface
        supplied as a black box can say how far its curvature can be trusted
        rather than having it assumed. Meaningless -- and returned as zero --
        for a surface that supplies analytic derivatives.
        """
        if self.jet is not None:
            return {"relative_change": 0.0, "step": 0.0, "source": "analytic"}
        step = self.fd_step_at(u, v)
        coarse = replace(self, fd_relative_step=2.0 * float(self.fd_relative_step))
        fine = float(np.max(np.abs(self.gaussian_curvature(u, v))))
        doubled = float(np.max(np.abs(coarse.gaussian_curvature(u, v))))
        scale = max(fine, 1e-300)
        return {
            "relative_change": float(abs(doubled - fine) / scale),
            "step": step,
            "source": "finite-difference",
        }

    # -- chart validity ----------------------------------------------------
    def chart_orthogonality(self, u, v) -> Array:
        """``sigma_2/sigma_1`` of the *direction-normalised* surface Jacobian.

        One where the coordinate curves meet at a right angle, zero where they
        become parallel. Because each column is normalised first, this depends
        only on the angle between the coordinate directions, so it is unchanged
        by rescaling ``u`` and ``v`` independently -- which
        ``sqrt(EG - F^2)/max(E, G)`` is not: that quantity confuses an
        anisotropic chart with a degenerate one, and on a torus it reads 0.34
        for a chart that is perfectly regular.
        """
        E, F, G = self.first_fundamental_form(u, v)
        cosine = np.abs(F) / np.sqrt(np.maximum(E * G, 1e-300))
        cosine = np.clip(cosine, 0.0, 1.0)
        return np.sqrt((1.0 - cosine) / (1.0 + cosine))

    def chart_scale_ratio(self, u, v) -> Array:
        """Shortest coordinate direction against the chart's declared reference.

        The other way a chart fails: a coordinate direction collapsing, as
        longitude does at a sphere's pole. It needs declared scales, because
        "short" is only meaningful against something.
        """
        jet = self.jet_at(u, v)
        lengths = np.stack(
            [
                np.sqrt(_dot(jet.ru, jet.ru)) * float(self.chart.u_scale),
                np.sqrt(_dot(jet.rv, jet.rv)) * float(self.chart.v_scale),
            ],
            axis=0,
        )
        return np.min(lengths, axis=0) / float(self.chart.reference_length)

    def chart_conditioning(self, u, v) -> Array:
        """The worse of the two failure modes, as one dimensionless number."""
        return np.minimum(self.chart_orthogonality(u, v), self.chart_scale_ratio(u, v))

    def chart_validity(self, u, v) -> dict[str, Array]:
        """Per-sample chart diagnostics: inside the domain, and well conditioned."""
        margin = self.chart.margin(u, v)
        orthogonality = self.chart_orthogonality(u, v)
        scale_ratio = self.chart_scale_ratio(u, v)
        conditioning = np.minimum(orthogonality, scale_ratio)
        E, F, G = self.first_fundamental_form(u, v)
        return {
            "in_domain": margin >= 0.0,
            "domain_margin": margin,
            "orthogonality": orthogonality,
            "scale_ratio": scale_ratio,
            "conditioning": conditioning,
            "well_conditioned": conditioning >= self.chart_conditioning_floor,
            "metric_determinant": E * G - F * F,
        }

    def require_valid_chart(self, u, v, *, where: str = "point") -> None:
        """Refuse a point the chart cannot represent, instead of returning NaNs."""
        validity = self.chart_validity(u, v)
        if not np.all(validity["in_domain"]):
            raise ValueError(
                f"{where} lies outside the declared domain of {self.name}: "
                f"{self.chart.to_dict()}"
            )
        if not np.all(validity["well_conditioned"]):
            worst = float(np.min(validity["conditioning"]))
            orthogonality = float(np.min(validity["orthogonality"]))
            scale_ratio = float(np.min(validity["scale_ratio"]))
            cause = (
                "the coordinate curves are becoming parallel"
                if orthogonality <= scale_ratio
                else "a coordinate direction is collapsing"
            )
            raise ValueError(
                f"{where} is in a degenerate part of the chart of {self.name}: "
                f"conditioning {worst:.3e} is below the floor "
                f"{self.chart_conditioning_floor:.3e} ({cause}; orthogonality "
                f"{orthogonality:.3e}, scale ratio {scale_ratio:.3e})"
            )

    # -- fundamental forms -------------------------------------------------
    # Everything below is derived from a single jet.  The flow evaluates the
    # jet once per stage and reuses it for the Christoffel symbols and for the
    # curvature; recomputing it for each would triple the cost of a path.
    def first_fundamental_form(self, u, v) -> tuple[Array, Array, Array]:
        return first_fundamental_form(self.jet_at(u, v))

    def unit_normal(self, u, v) -> Array:
        return unit_normal(self.jet_at(u, v))

    def second_fundamental_form(self, u, v) -> tuple[Array, Array, Array]:
        return second_fundamental_form(self.jet_at(u, v))

    def gaussian_curvature(self, u, v) -> Array:
        """``K = (LN - M^2) / (EG - F^2)``, from the jet."""
        return gaussian_curvature(self.jet_at(u, v))

    def mean_curvature(self, u, v) -> Array:
        """``H``, the average of the principal curvatures, from the jet."""
        return mean_curvature(self.jet_at(u, v))

    def christoffel(self, u, v) -> tuple[Array, Array, Array, Array, Array, Array]:
        """``(G1_11, G1_12, G1_22, G2_11, G2_12, G2_22)`` at ``(u, v)``."""
        return christoffel(self.jet_at(u, v))

    # -- tangent vectors ---------------------------------------------------
    def speed(self, u, v, du, dv) -> Array:
        E, F, G = self.first_fundamental_form(u, v)
        return np.sqrt(np.clip(E * du * du + 2.0 * F * du * dv + G * dv * dv, 0.0, None))

    def unit_direction(self, u, v, heading: float) -> tuple[float, float]:
        """Unit tangent at ``(u, v)`` at angle ``heading`` from the ``u`` direction.

        The frame is the Gram-Schmidt orthonormalisation of ``(r_u, r_v)``, so
        ``heading`` is a genuine angle in the surface's own metric and not an
        artefact of the parameterisation.
        """
        E, F, G = self.first_fundamental_form(u, v)
        root_E = np.sqrt(E)
        W = np.sqrt(E * G - F * F)
        du = np.cos(heading) / root_E - np.sin(heading) * F / (root_E * W)
        dv = np.sin(heading) * root_E / W
        return float(du), float(dv)

    # -- the coupled flow --------------------------------------------------
    def geodesic_transfer_rhs(self) -> Callable[[float, Array], Array]:
        """RHS of the state ``(u, v, u', v', a, a', b, b')``.

        The geodesic and *both* columns of the transfer map are advanced
        together, because the Jacobi equation needs ``K`` at the moving point:
        on a surface of varying curvature the path and its sensitivity cannot
        be separated. ``a`` propagates an initial lateral offset and ``b`` an
        initial heading error; they cost one shared curvature evaluation.
        """

        def rhs(_s: float, y: Array) -> Array:
            u, v, du, dv = (y[..., index] for index in range(4))
            a, a_rate, b, b_rate = (y[..., index] for index in range(4, 8))
            jet = self.jet_at(u, v)
            g1_11, g1_12, g1_22, g2_11, g2_12, g2_22 = christoffel(jet)
            ddu = -(g1_11 * du * du + 2.0 * g1_12 * du * dv + g1_22 * dv * dv)
            ddv = -(g2_11 * du * du + 2.0 * g2_12 * du * dv + g2_22 * dv * dv)
            curvature = gaussian_curvature(jet)
            return np.stack(
                [du, dv, ddu, ddv, a_rate, -curvature * a, b_rate, -curvature * b],
                axis=-1,
            )

        return rhs

    def initial_state(self, u0: float, v0: float, heading: float) -> Array:
        du, dv = self.unit_direction(u0, v0, heading)
        return np.array([u0, v0, du, dv, *TRANSFER_INITIAL_STATE])

    def state_from_tangent(self, u0: float, v0: float, du: float, dv: float) -> Array:
        """Same as :meth:`initial_state` but from a tangent given in coordinates.

        Used when the direction comes from somewhere other than a heading angle
        -- for instance parallel-transported along a perpendicular geodesic, as
        the lateral-offset variation requires.
        """
        return np.array([u0, v0, float(du), float(dv), *TRANSFER_INITIAL_STATE])

    def perpendicular_direction(self, u, v, du, dv) -> tuple[float, float]:
        """Unit tangent orthogonal to ``(du, dv)`` in the surface metric.

        ``N = (-(F du + G dv), E du + F dv) / sqrt(EG - F^2)`` is orthogonal to
        ``(du, dv)`` for any first fundamental form, and is unit whenever the
        given tangent is. Rotating by a heading angle would give a perpendicular
        too, but only at a point whose frame is already known; this works at a
        point reached by flowing, where it is not.
        """
        E, F, G = self.first_fundamental_form(u, v)
        W = np.sqrt(E * G - F * F)
        return (float(-(F * du + G * dv) / W), float((E * du + F * dv) / W))

    def embed(self, u, v) -> Array:
        return self.position(np.asarray(u, dtype=float), np.asarray(v, dtype=float))


# ---------------------------------------------------------------------------
# jet -> geometry.  Free functions, so a caller holding a jet pays for it once,
# which is what makes an integration step affordable.
# ---------------------------------------------------------------------------
def first_fundamental_form(jet: SurfaceJet) -> tuple[Array, Array, Array]:
    return _dot(jet.ru, jet.ru), _dot(jet.ru, jet.rv), _dot(jet.rv, jet.rv)


def _cross(a: Array, b: Array) -> Array:
    """``np.cross`` for the three-vectors used here, without its dispatch cost."""
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def unit_normal(jet: SurfaceJet) -> Array:
    normal = _cross(jet.ru, jet.rv)
    return normal / np.sqrt(_dot(normal, normal))[..., None]


def second_fundamental_form(jet: SurfaceJet) -> tuple[Array, Array, Array]:
    normal = unit_normal(jet)
    return _dot(jet.ruu, normal), _dot(jet.ruv, normal), _dot(jet.rvv, normal)


def mean_curvature(jet: SurfaceJet) -> Array:
    """``H = (EN - 2FM + GL) / (2(EG - F^2))``, the average of the principal curvatures.

    Carried for one reason: Euler's theorem says the normal curvatures in any
    two orthogonal tangent directions sum to ``2H``, independently of which
    pair. That makes the two normal curvatures a record now carries checkable
    against a quantity computed a different way, on every surface, with no
    closed form needed -- which is the standard everything else here is held
    to.
    """
    E, F, G = first_fundamental_form(jet)
    L, M, N = second_fundamental_form(jet)
    return (E * N - 2.0 * F * M + G * L) / (2.0 * (E * G - F * F))


def gaussian_curvature(jet: SurfaceJet) -> Array:
    """``K = (LN - M^2) / (EG - F^2)``, without normalising the surface normal.

    With the unnormalised normal ``n = r_u x r_v`` the second-form entries pick
    up a common factor ``|n| = sqrt(EG - F^2)``, so ``K`` collapses to
    ``(L~ N~ - M~^2) / (EG - F^2)^2`` and the square root disappears.  It is the
    same number, with two fewer operations per stage of every step.
    """
    E, F, G = first_fundamental_form(jet)
    normal = _cross(jet.ru, jet.rv)
    scaled_L = _dot(jet.ruu, normal)
    scaled_M = _dot(jet.ruv, normal)
    scaled_N = _dot(jet.rvv, normal)
    determinant = E * G - F * F
    return (scaled_L * scaled_N - scaled_M * scaled_M) / (determinant * determinant)


def christoffel(jet: SurfaceJet) -> tuple[Array, Array, Array, Array, Array, Array]:
    """``(G1_11, G1_12, G1_22, G2_11, G2_12, G2_22)``.

    The derivatives of ``E``, ``F`` and ``G`` are assembled from the same jet
    rather than differentiated a second time, so a surface that supplies
    analytic derivatives gets analytic Christoffel symbols.
    """
    E, F, G = first_fundamental_form(jet)
    E_u, E_v = 2.0 * _dot(jet.ru, jet.ruu), 2.0 * _dot(jet.ru, jet.ruv)
    F_u = _dot(jet.ruu, jet.rv) + _dot(jet.ru, jet.ruv)
    F_v = _dot(jet.ruv, jet.rv) + _dot(jet.ru, jet.rvv)
    G_u, G_v = 2.0 * _dot(jet.rv, jet.ruv), 2.0 * _dot(jet.rv, jet.rvv)

    # Christoffel symbols of the first kind.
    c11_1, c11_2 = 0.5 * E_u, F_u - 0.5 * E_v
    c12_1, c12_2 = 0.5 * E_v, 0.5 * G_u
    c22_1, c22_2 = F_v - 0.5 * G_u, 0.5 * G_v

    determinant = E * G - F * F
    g11, g12, g22 = G / determinant, -F / determinant, E / determinant
    return (
        g11 * c11_1 + g12 * c11_2,
        g11 * c12_1 + g12 * c12_2,
        g11 * c22_1 + g12 * c22_2,
        g12 * c11_1 + g22 * c11_2,
        g12 * c12_1 + g22 * c12_2,
        g12 * c22_1 + g22 * c22_2,
    )


def _dot(a: Array, b: Array) -> Array:
    return np.sum(a * b, axis=-1)


def _stack(*components) -> Array:
    return np.stack(np.broadcast_arrays(*components), axis=-1)


# ---------------------------------------------------------------------------
# a small catalogue, chosen so that each one tests something different
# ---------------------------------------------------------------------------
def plane() -> ParametricSurface:
    """K = 0 everywhere. The calibrated flat case, reached the general way."""

    def position(u, v):
        return _stack(u, v, np.zeros_like(np.asarray(u, dtype=float) * 1.0))

    def jet(u, v):
        zero = np.zeros_like(np.asarray(u, dtype=float) * 1.0)
        one = np.ones_like(zero)
        return SurfaceJet(
            r=_stack(u, v, zero),
            ru=_stack(one, zero, zero),
            rv=_stack(zero, one, zero),
            ruu=_stack(zero, zero, zero),
            ruv=_stack(zero, zero, zero),
            rvv=_stack(zero, zero, zero),
        )

    return ParametricSurface(
        name="plane",
        position=position,
        jet=jet,
        exact_curvature=lambda u, v: np.zeros_like(np.asarray(u, dtype=float) * 1.0),
        chart=Chart(note="global: the identity chart covers the plane"),
        description="flat plate",
    )


def sphere(radius: float = 1.0) -> ParametricSurface:
    """K = 1/radius^2. Parameterised by colatitude and longitude."""

    def position(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return _stack(
            radius * np.sin(u) * np.cos(v),
            radius * np.sin(u) * np.sin(v),
            radius * np.cos(u) * np.ones_like(v),
        )

    def jet(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        su, cu, sv, cv = np.sin(u), np.cos(u), np.sin(v), np.cos(v)
        zero = np.zeros_like(su * sv)
        return SurfaceJet(
            r=_stack(radius * su * cv, radius * su * sv, radius * cu + zero),
            ru=_stack(radius * cu * cv, radius * cu * sv, -radius * su + zero),
            rv=_stack(-radius * su * sv, radius * su * cv, zero),
            ruu=_stack(-radius * su * cv, -radius * su * sv, -radius * cu + zero),
            ruv=_stack(-radius * cu * sv, radius * cu * cv, zero),
            rvv=_stack(-radius * su * cv, -radius * su * sv, zero),
        )

    return ParametricSurface(
        name=f"sphere(R={radius:g})",
        position=position,
        jet=jet,
        exact_curvature=lambda u, v: np.full_like(
            np.asarray(u, dtype=float) * np.asarray(v, dtype=float) * 1.0, 1.0 / radius**2
        ),
        chart=Chart(
            u_min=0.0,
            u_max=float(np.pi),
            v_period=2.0 * float(np.pi),
            reference_length=radius,
            note="colatitude and longitude; the poles u = 0, pi are chart "
            "singularities where the longitude direction collapses",
        ),
        description="spherical cap",
    )


def cylinder(radius: float = 1.0) -> ParametricSurface:
    """K = 0 although the surface is visibly curved.

    The instrument keys on intrinsic curvature, and a rolled flat sheet has
    none: path sensitivity on a cylinder is exactly what it was on the sheet.
    """

    def position(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return _stack(radius * np.cos(u) + 0.0 * v, radius * np.sin(u) + 0.0 * v, v + 0.0 * u)

    def jet(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        zero = np.zeros_like(cu * v)
        one = np.ones_like(zero)
        return SurfaceJet(
            r=_stack(radius * cu + zero, radius * su + zero, v + zero),
            ru=_stack(-radius * su + zero, radius * cu + zero, zero),
            rv=_stack(zero, zero, one),
            ruu=_stack(-radius * cu + zero, -radius * su + zero, zero),
            ruv=_stack(zero, zero, zero),
            rvv=_stack(zero, zero, zero),
        )

    return ParametricSurface(
        name=f"cylinder(R={radius:g})",
        position=position,
        jet=jet,
        exact_curvature=lambda u, v: np.zeros_like(
            np.asarray(u, dtype=float) * np.asarray(v, dtype=float) * 1.0
        ),
        chart=Chart(
            u_period=2.0 * float(np.pi),
            reference_length=radius,
            note="u wraps around the axis; v runs along it without bound",
        ),
        description="rolled sheet: extrinsically curved, intrinsically flat",
    )


def pseudosphere() -> ParametricSurface:
    """K = -1 everywhere: the tractricoid, an embedded constant-negative surface.

    This is what a physical ``K < 0`` coupon would have to be. A
    hyperbolic-paraboloid saddle is *not* of constant curvature, which is why
    it appears separately below.  Valid for ``u > 0``; the surface has a cusp
    on the circle ``u = 0``.
    """

    def position(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        sech = 1.0 / np.cosh(u)
        return _stack(sech * np.cos(v), sech * np.sin(v), u - np.tanh(u) + 0.0 * v)

    def jet(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        S, T = 1.0 / np.cosh(u), np.tanh(u)
        cv, sv = np.cos(v), np.sin(v)
        zero = np.zeros_like(S * cv)
        return SurfaceJet(
            r=_stack(S * cv, S * sv, u - T + zero),
            ru=_stack(-S * T * cv, -S * T * sv, T * T + zero),
            rv=_stack(-S * sv, S * cv, zero),
            ruu=_stack((S * T * T - S**3) * cv, (S * T * T - S**3) * sv, 2.0 * T * S * S + zero),
            ruv=_stack(S * T * sv, -S * T * cv, zero),
            rvv=_stack(-S * cv, -S * sv, zero),
        )

    return ParametricSurface(
        name="pseudosphere",
        position=position,
        jet=jet,
        exact_curvature=lambda u, v: np.full_like(
            np.asarray(u, dtype=float) * np.asarray(v, dtype=float) * 1.0, -1.0
        ),
        chart=Chart(
            u_min=0.05,
            u_max=6.0,
            v_period=2.0 * float(np.pi),
            note="u > 0 strictly: the cusp circle u = 0 is an edge of the "
            "surface, and the chart also degenerates as u grows and the tube "
            "closes on the axis",
        ),
        description="tractricoid, constant K = -1",
    )


def hyperbolic_paraboloid(scale: float = 1.0) -> ParametricSurface:
    """A saddle: ``z = (u^2 - v^2) / (2a)``, with ``K`` varying across it."""

    def position(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return _stack(u + 0.0 * v, v + 0.0 * u, (u * u - v * v) / (2.0 * scale))

    def jet(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        zero = np.zeros_like(u * v)
        one = np.ones_like(zero)
        return SurfaceJet(
            r=_stack(u + zero, v + zero, (u * u - v * v) / (2.0 * scale)),
            ru=_stack(one, zero, u / scale + zero),
            rv=_stack(zero, one, -v / scale + zero),
            ruu=_stack(zero, zero, one / scale),
            ruv=_stack(zero, zero, zero),
            rvv=_stack(zero, zero, -one / scale),
        )

    def curvature(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return -1.0 / (scale**2 * (1.0 + (u * u + v * v) / scale**2) ** 2)

    return ParametricSurface(
        name=f"hyperbolic-paraboloid(a={scale:g})",
        position=position,
        jet=jet,
        exact_curvature=curvature,
        chart=Chart(note="global: a graph over the whole (u, v) plane"),
        description="saddle coupon: K < 0 but not constant",
    )


def torus(major: float = 2.0, minor: float = 1.0) -> ParametricSurface:
    """``K = cos(u) / (r (R + r cos u))``: positive outside, negative inside, zero on top."""

    def position(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        ring = major + minor * np.cos(u)
        return _stack(ring * np.cos(v), ring * np.sin(v), minor * np.sin(u) + 0.0 * v)

    def jet(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su, cv, sv = np.cos(u), np.sin(u), np.cos(v), np.sin(v)
        ring = major + minor * cu
        zero = np.zeros_like(ring * cv)
        return SurfaceJet(
            r=_stack(ring * cv, ring * sv, minor * su + zero),
            ru=_stack(-minor * su * cv, -minor * su * sv, minor * cu + zero),
            rv=_stack(-ring * sv, ring * cv, zero),
            ruu=_stack(-minor * cu * cv, -minor * cu * sv, -minor * su + zero),
            ruv=_stack(minor * su * sv, -minor * su * cv, zero),
            rvv=_stack(-ring * cv, -ring * sv, zero),
        )

    def curvature(u, v):
        u = np.asarray(u, dtype=float)
        return np.cos(u) / (minor * (major + minor * np.cos(u))) + 0.0 * np.asarray(
            v, dtype=float
        )

    return ParametricSurface(
        name=f"torus(R={major:g}, r={minor:g})",
        position=position,
        jet=jet,
        exact_curvature=curvature,
        chart=Chart(
            u_period=2.0 * float(np.pi),
            v_period=2.0 * float(np.pi),
            reference_length=minor,
            note="both angles wrap; the chart is regular everywhere for R > r",
        ),
        description="curvature of both signs on one part",
    )


CATALOGUE: dict[str, Callable[[], ParametricSurface]] = {
    "plane": plane,
    "cylinder": lambda: cylinder(1.0),
    "sphere": lambda: sphere(1.0),
    "pseudosphere": pseudosphere,
    "saddle": lambda: hyperbolic_paraboloid(1.0),
    "torus": lambda: torus(2.0, 1.0),
}


def built_in(name: str) -> ParametricSurface:
    try:
        return CATALOGUE[name]()
    except KeyError as exc:  # pragma: no cover - guard
        raise KeyError(f"unknown surface {name!r}; have {sorted(CATALOGUE)}") from exc


def darboux_frame(
    surface: ParametricSurface, u, v, du, dv
) -> dict[str, Array]:
    """The path's own frame in ambient space, and how the surface bends in it.

    Returns the three unit vectors -- ``tangent`` along the curve,
    ``surface_normal``, and ``transverse = normal x tangent`` -- together with
    the normal curvature in the direction of travel and in the transverse
    direction.

    The two normal curvatures answer different questions and are both called
    ``kappa_n``. ``along`` is the bending a tool travelling the path feels.
    ``transverse`` is the one that sets how far an ambient chord falls short of
    an in-surface separation, so it is the one a comparison against
    reconstructed 3-D points needs -- and the one this repository's own
    ``(cn_K^2 + kappa_n^2 sn_K^2)/6`` coefficient is written in terms of.

    ``du, dv`` are the parameter velocities; they are normalised here, so a
    caller need not have kept unit speed exactly.
    """
    jet = surface.jet_at(u, v)
    E, F, G = first_fundamental_form(jet)
    du = np.asarray(du, dtype=float)
    dv = np.asarray(dv, dtype=float)
    speed = np.sqrt(np.clip(E * du * du + 2.0 * F * du * dv + G * dv * dv, 0.0, None))
    if np.any(speed <= 0.0):
        raise ValueError("a path with zero speed has no tangent direction")
    du, dv = du / speed, dv / speed

    normal = unit_normal(jet)
    tangent = jet.ru * du[..., None] + jet.rv * dv[..., None]
    tangent = tangent / np.sqrt(_dot(tangent, tangent))[..., None]
    transverse = _cross(normal, tangent)

    # The transverse direction in parameter coordinates, from the same
    # orthonormalisation the heading uses, so that the ambient vector above and
    # the parameter velocities below describe one direction and not two.
    W = np.sqrt(np.clip(E * G - F * F, 0.0, None))
    tu, tv = -(F * du + G * dv) / W, (E * du + F * dv) / W

    L, M, N = second_fundamental_form(jet)
    return {
        "tangent": tangent,
        "surface_normal": normal,
        "transverse": transverse,
        "normal_curvature_along": L * du * du + 2.0 * M * du * dv + N * dv * dv,
        "normal_curvature_transverse": L * tu * tu + 2.0 * M * tu * tv + N * tv * tv,
    }
