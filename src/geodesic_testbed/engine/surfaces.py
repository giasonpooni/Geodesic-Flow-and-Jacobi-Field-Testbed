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
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray

# (a, a', b, b') at s = 0: the identity transfer map.
TRANSFER_INITIAL_STATE = (1.0, 0.0, 0.0, 1.0)


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
    fd_step: float = 1e-4
    description: str = ""

    # -- derivatives -------------------------------------------------------
    def jet_at(self, u, v) -> SurfaceJet:
        if self.jet is not None:
            return self.jet(u, v)
        return self._finite_difference_jet(u, v)

    def _finite_difference_jet(self, u, v) -> SurfaceJet:
        h = self.fd_step
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
