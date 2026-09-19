"""Constant-curvature space forms carried by a single ambient chart.

All three model surfaces used by this testbed -- the Euclidean plane
(``K = 0``), the unit sphere (``K = +1``) and the hyperbolic plane
(``K = -1``) -- are represented in ``R^3`` equipped with the symmetric
bilinear form

.. math::

    \\langle x, y \\rangle_K = x_0 y_0 + x_1 y_1 + K\\, x_2 y_2 .

With that one convention:

* ``K = +1`` gives the Euclidean form and the surface ``<x,x> = 1`` (sphere);
* ``K = -1`` gives the Minkowski form and the upper sheet ``<x,x> = -1``
  (hyperboloid model of ``H^2``);
* ``K = 0`` gives a degenerate form that ignores ``x_2``; the surface is the
  affine plane ``x_2 = 1`` and the form restricts to the Euclidean metric of
  ``R^2`` on it.

Only ``K in {0, +1, -1}`` is supported, and nothing is lost by that: rescaling
the metric by ``lambda^2`` rescales curvature by ``lambda^-2``, so every
constant-curvature surface is one of these three up to a change of unit.

The payoff of the shared chart is that the geodesic flow, the exponential map,
the distance function and the Jacobi reference solution all become one formula
in ``K``, built from the generalised trigonometric pair

.. math::

    \\mathrm{sn}_K'' + K\\,\\mathrm{sn}_K = 0,\\quad
    \\mathrm{sn}_K(0) = 0,\\ \\mathrm{sn}_K'(0) = 1,

that is ``sn_K = s, sin s, sinh s`` for ``K = 0, +1, -1``.  The same
``sn_K`` is the solution of the Jacobi equation this testbed measures, which
is exactly why the radial part of the exponential map and the transverse
Jacobi field are the same object.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SUPPORTED_CURVATURES: tuple[float, ...] = (0.0, 1.0, -1.0)

_NAMES = {0.0: "euclidean-plane", 1.0: "unit-sphere", -1.0: "hyperbolic-plane"}
_LABELS = {0.0: "K=0", 1.0: "K=+1", -1.0: "K=-1"}
_SOLUTION_NAMES = {0.0: "s", 1.0: "sin(s)", -1.0: "sinh(s)"}


def sin_k(s, K: float) -> np.ndarray:
    """Generalised sine: solution of ``j'' + K j = 0`` with ``j(0)=0, j'(0)=1``."""
    s = np.asarray(s, dtype=float)
    if K > 0.0:
        return np.sin(s)
    if K < 0.0:
        return np.sinh(s)
    return s * 1.0


def cos_k(s, K: float) -> np.ndarray:
    """Generalised cosine: solution of ``j'' + K j = 0`` with ``j(0)=1, j'(0)=0``."""
    s = np.asarray(s, dtype=float)
    if K > 0.0:
        return np.cos(s)
    if K < 0.0:
        return np.cosh(s)
    return np.ones_like(s)


def asin_k(z, K: float) -> np.ndarray:
    """Inverse of :func:`sin_k`."""
    z = np.asarray(z, dtype=float)
    if K > 0.0:
        return np.arcsin(np.clip(z, -1.0, 1.0))
    if K < 0.0:
        return np.arcsinh(z)
    return z * 1.0


def conjugate_distance(K: float) -> float | None:
    """First conjugate point along a unit-speed geodesic, or ``None`` if there is none.

    It is the first positive zero of ``sn_K``; only the sphere has one.
    """
    return float(np.pi) if K > 0.0 else None


@dataclass(frozen=True)
class SpaceForm:
    """A simply connected surface of constant curvature ``K in {0, +1, -1}``."""

    K: float

    def __post_init__(self) -> None:
        if float(self.K) not in SUPPORTED_CURVATURES:
            raise ValueError(
                f"curvature {self.K!r} is not supported; use one of {SUPPORTED_CURVATURES}"
            )
        object.__setattr__(self, "K", float(self.K))

    # -- identity ---------------------------------------------------------
    @property
    def name(self) -> str:
        return _NAMES[self.K]

    @property
    def label(self) -> str:
        return _LABELS[self.K]

    @property
    def reference_solution_name(self) -> str:
        """Closed-form Jacobi solution for this curvature, as text."""
        return _SOLUTION_NAMES[self.K]

    # -- ambient bilinear form -------------------------------------------
    @property
    def metric_diagonal(self) -> np.ndarray:
        return np.array([1.0, 1.0, self.K])

    def ip(self, u, v) -> np.ndarray:
        """Ambient bilinear form ``<u, v>_K``, broadcast over leading axes."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return np.sum(u * v * self.metric_diagonal, axis=-1)

    # -- distinguished point and frame ------------------------------------
    def base_point(self) -> np.ndarray:
        """``p0 = (0, 0, 1)``: on the sphere, on the hyperboloid, and on the plane."""
        return np.array([0.0, 0.0, 1.0])

    def base_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Orthonormal tangent frame at :meth:`base_point`, identical for all ``K``."""
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    def rotated_direction(self, angle: float) -> np.ndarray:
        """Unit tangent at ``p0`` making angle ``angle`` with the first frame vector."""
        e0, e1 = self.base_frame()
        return np.cos(angle) * e0 + np.sin(angle) * e1

    # -- geometry ----------------------------------------------------------
    def exp(self, p, v, s) -> tuple[np.ndarray, np.ndarray]:
        """Exponential map and its velocity, in closed form.

        ``gamma(s) = cn_K(s) p + sn_K(s) v`` for a unit tangent ``v`` at ``p``.
        The ``sn_K(s)`` factor in front of the direction is the Jacobi field.
        """
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        s = np.asarray(s, dtype=float)
        c = cos_k(s, self.K)[..., None]
        sn = sin_k(s, self.K)[..., None]
        point = c * p + sn * v
        velocity = -self.K * sn * p + c * v
        return point, velocity

    def accel(self, x, v) -> np.ndarray:
        """Ambient acceleration of a geodesic: ``x'' = -K <v, v>_K x``."""
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        return -self.K * self.ip(v, v)[..., None] * x

    def flow_rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """Right-hand side of the first-order geodesic system on ``(x, v) in R^6``."""

        def rhs(_s: float, y: np.ndarray) -> np.ndarray:
            x, v = y[..., :3], y[..., 3:]
            return np.concatenate([v, self.accel(x, v)], axis=-1)

        return rhs

    def flow_state(self, p, v) -> np.ndarray:
        return np.concatenate([np.asarray(p, float), np.asarray(v, float)], axis=-1)

    def distance(self, p, q) -> np.ndarray:
        """Riemannian distance, from the ambient chord and (on the sphere) the ambient sum.

        In every one of the three models the ambient chord satisfies

            <q - p, q - p>_K = 4 sn_K(d/2)^2,

        so ``d = 2 sn_K^{-1}(chord/2)``.  Written this way the distance stays
        accurate for arbitrarily small separations, where the textbook
        ``arccos`` / ``arccosh`` forms lose most of their digits.

        The sphere needs one more ingredient at the other end of its range.
        ``arcsin`` saturates as the two points approach antipodal, so the
        companion identity

            K <q + p, q + p>_K = 4 cn_K(d/2)^2

        is used to get ``d = 2 atan2(chord, sum)``, which is well conditioned
        over the whole of ``[0, pi]``.  On the plane and the hyperbolic plane
        the chord form is already stable everywhere, ``arcsinh`` having no
        such turning point, so it is kept.
        """
        chord = np.sqrt(np.clip(self.chord_squared(p, q), 0.0, None))
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        if self.K > 0.0:
            total = q + p
            cosine_part = np.sqrt(np.clip(self.K * self.ip(total, total), 0.0, None))
            return 2.0 * np.arctan2(chord, cosine_part)
        return 2.0 * asin_k(chord / 2.0, self.K)

    def chord_squared(self, p, q) -> np.ndarray:
        """Unclipped ambient ``<q - p, q - p>_K``.

        For points that really lie on the surface this equals ``4 sn_K(d/2)^2``
        and is non-negative.  For numerically flowed points it can go negative
        on the hyperbolic plane, where the ambient form is indefinite: once two
        trajectories have drifted off the hyperboloid by more than they are
        apart, their difference stops being spacelike and there is no distance
        left to extract.  :meth:`distance` clips, which is the closest legal
        answer; callers that care about the breakdown should look here.
        """
        delta = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
        return self.ip(delta, delta)

    def constraint_residuals(self, x, v) -> dict[str, np.ndarray]:
        """How far a state has drifted off the unit tangent bundle of the surface."""
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        if self.K == 0.0:
            manifold = np.abs(x[..., 2] - 1.0)
            tangency = np.abs(v[..., 2])
        else:
            manifold = np.abs(self.ip(x, x) - 1.0 / self.K)
            tangency = np.abs(self.ip(x, v))
        return {
            "manifold": manifold,
            "tangency": tangency,
            "speed": np.abs(self.ip(v, v) - 1.0),
        }

    # -- exact two-geodesic separation -------------------------------------
    def exact_separation(self, s, epsilon) -> np.ndarray:
        """Exact distance between two unit-speed geodesics leaving ``p0`` at angle ``epsilon``.

        Both geodesics start at ``p0``; one leaves along ``e0``, the other along
        ``cos(eps) e0 + sin(eps) e1``.  Because the two exponential images differ
        only in the direction factor,

            <gamma - gamma_eps, .>_K  =>  sn_K(d/2) = sn_K(s) sin(eps/2),

        which is the law of cosines for all three curvatures at once, and is
        stable down to the smallest representable ``epsilon``.
        """
        s = np.asarray(s, dtype=float)
        epsilon = np.asarray(epsilon, dtype=float)
        return 2.0 * asin_k(sin_k(s, self.K) * np.sin(epsilon / 2.0), self.K)

    def first_order_separation(self, s, epsilon) -> np.ndarray:
        """First-order (Jacobi) prediction ``eps * sn_K(s)`` of the separation."""
        return np.asarray(epsilon, dtype=float) * sin_k(np.asarray(s, dtype=float), self.K)

    def relative_deviation_coefficient(self, s) -> np.ndarray:
        """Leading coefficient ``C(s)`` in ``d = eps sn_K(s) [1 - C(s) eps^2 + O(eps^4)]``.

        Expanding ``sn_K(d/2) = sn_K(s) sin(eps/2)`` gives ``C(s) = cn_K(s)^2 / 24``.
        """
        return cos_k(np.asarray(s, dtype=float), self.K) ** 2 / 24.0

    def epsilon_for_relative_tolerance(self, s, tolerance: float) -> np.ndarray:
        """Largest ``eps`` for which the first-order prediction stays within ``tolerance``.

        From ``C(s) eps^2 = tol``.  Returns ``inf`` where ``C(s) = 0``.
        """
        coefficient = np.asarray(self.relative_deviation_coefficient(s), dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(coefficient > 0.0, np.sqrt(tolerance / coefficient), np.inf)


def all_space_forms() -> tuple[SpaceForm, ...]:
    return tuple(SpaceForm(K) for K in SUPPORTED_CURVATURES)
