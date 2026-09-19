"""The 2x2 transfer map from a starting pose error to a downstream one.

A heading error is only half of how a path can start wrong. The other half is
a lateral offset: the tool, the probe or the tow begins beside the nominal
path rather than pointed away from it. Both propagate through the same
equation,

.. math::

    j''(s) + K(\\gamma(s))\\, j(s) = 0,

so the two fundamental solutions

.. math::

    a(0) = 1,\\ a'(0) = 0 \\qquad\\text{(lateral offset)}\\\\
    b(0) = 0,\\ b'(0) = 1 \\qquad\\text{(heading error)}

assemble into

.. math::

    \\Phi(s) = \\begin{bmatrix} a(s) & b(s) \\\\ a'(s) & b'(s) \\end{bmatrix},
    \\qquad
    \\begin{bmatrix} \\delta_\\perp(s) \\\\ \\delta_\\alpha(s) \\end{bmatrix}
    = \\Phi(s)\\,
    \\begin{bmatrix} \\delta_\\perp(0) \\\\ \\delta_\\alpha(0) \\end{bmatrix}.

Everything the single-field version did is the second column of this. What the
first column adds is fixture and datum error, and what the matrix adds beyond
either column is the ability to push a covariance through rather than a scalar.

The matrix carries its own invariant. Since the equation has no first-derivative
term, the Wronskian ``a b' - a' b`` is conserved, and its initial value is 1:

.. math::

    \\det \\Phi(s) = 1 \\quad \\text{for every } s .

That is an exact statement about a quantity the solver never enforces, on every
surface, at every arc length -- so its drift is a free and unusually sharp
measure of how well the transfer map is being integrated. It is checked, not
assumed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray

# Index of each component inside the flow state, counted from the end, so the
# same block can ride along with either a scalar or a parametric flow.
COMPONENTS = ("a", "a_rate", "b", "b_rate")
INITIAL_STATE = (1.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class TransferMap:
    """``Phi(s)`` sampled along one path, with the pose errors it maps."""

    arc_length: Array
    a: Array
    a_rate: Array
    b: Array
    b_rate: Array

    @property
    def determinant(self) -> Array:
        """``det Phi = a b' - a' b``, conserved and equal to 1."""
        return self.a * self.b_rate - self.a_rate * self.b

    @property
    def wronskian_drift(self) -> Array:
        return np.abs(self.determinant - 1.0)

    def matrices(self) -> Array:
        """``Phi(s)`` as an ``(n, 2, 2)`` stack."""
        return np.stack(
            [
                np.stack([self.a, self.b], axis=-1),
                np.stack([self.a_rate, self.b_rate], axis=-1),
            ],
            axis=-2,
        )

    def propagate(self, lateral: float, heading: float) -> tuple[Array, Array]:
        """Map a starting pose error to ``(lateral, heading)`` error along the path."""
        lateral = float(lateral)
        heading = float(heading)
        return (
            self.a * lateral + self.b * heading,
            self.a_rate * lateral + self.b_rate * heading,
        )

    def worst_case_offset(self, lateral: float, heading: float) -> Array:
        """Largest transverse offset reachable from a box of starting errors.

        The two contributions are added in absolute value, which is the exact
        bound over the box ``|delta_perp(0)| <= lateral`` and
        ``|delta_alpha(0)| <= heading`` -- the worst corner is always a corner.
        """
        return np.abs(self.a) * abs(float(lateral)) + np.abs(self.b) * abs(float(heading))

    def rss_offset(self, lateral: float, heading: float) -> Array:
        """Root-sum-square combination, for independent errors rather than a box."""
        return np.hypot(np.abs(self.a) * abs(float(lateral)), np.abs(self.b) * abs(float(heading)))

    def propagate_covariance(self, covariance) -> Array:
        """``Phi C Phi^T`` for a 2x2 starting-pose covariance, sampled along the path.

        Returns an ``(n, 2, 2)`` stack in the same ``(lateral, heading)`` basis.
        A deterministic tolerance box is a bound; this is the distributional
        statement to use when the starting error is characterised statistically
        instead.
        """
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (2, 2):
            raise ValueError("covariance must be 2x2 in the (lateral, heading) basis")
        if not np.allclose(covariance, covariance.T, atol=0.0, rtol=1e-12):
            raise ValueError("covariance must be symmetric")
        phi = self.matrices()
        return phi @ covariance @ np.swapaxes(phi, -1, -2)

    def focus_points(self, *, component: str = "b") -> list[float]:
        """Arc lengths at which a column of ``Phi`` vanishes away from the start.

        ``b`` vanishing is a conjugate point of the heading variation: paths
        that left at different angles meet again. ``a`` vanishing is the
        corresponding focus of the lateral variation.
        """
        values = {"a": self.a, "b": self.b}[component]
        found: list[float] = []
        for index in range(1, len(values) - 1):
            if self.arc_length[index] <= 0.0:
                continue
            left, right = values[index], values[index + 1]
            if left == 0.0:
                found.append(float(self.arc_length[index]))
            elif left * right < 0.0:
                weight = left / (left - right)
                found.append(
                    float(
                        self.arc_length[index]
                        + weight * (self.arc_length[index + 1] - self.arc_length[index])
                    )
                )
        return found


def constant_curvature_transfer(arc_length, curvature: float) -> TransferMap:
    """The closed-form ``Phi(s)`` on a surface of constant curvature.

    ``a = cn_K``, ``b = sn_K``, so ``det Phi = cn_K^2 + K sn_K^2 = 1`` is the
    generalised Pythagorean identity -- the Wronskian and Pythagoras are the
    same statement here.
    """
    from .spaceforms import cos_k, sin_k

    s = np.asarray(arc_length, dtype=float)
    K = float(curvature)
    scale = np.sqrt(abs(K)) if K != 0.0 else 1.0
    scaled = s * scale
    sign = 1.0 if K > 0.0 else (-1.0 if K < 0.0 else 0.0)
    a = cos_k(scaled, sign)
    b = sin_k(scaled, sign) / scale
    a_rate = -K * b
    b_rate = a
    return TransferMap(arc_length=s, a=a, a_rate=a_rate, b=b, b_rate=b_rate)


def transfer_rhs(curvature) -> Callable[[float, Array], Array]:
    """RHS of ``(a, a', b, b')`` for a scalar curvature profile.

    ``curvature`` may be a constant or a callable of arc length.
    """
    if callable(curvature):
        curvature_at = curvature
    else:
        constant = float(curvature)

        def curvature_at(_s: float) -> float:
            return constant

    def rhs(s: float, y: Array) -> Array:
        K = curvature_at(s)
        a, a_rate, b, b_rate = (y[..., index] for index in range(4))
        return np.stack([a_rate, -K * a, b_rate, -K * b], axis=-1)

    return rhs


def transfer_from_trajectory(grid: Array, trajectory: Array) -> TransferMap:
    """Wrap an integrated ``(a, a', b, b')`` trajectory as a :class:`TransferMap`."""
    return TransferMap(
        arc_length=np.asarray(grid, dtype=float),
        a=trajectory[..., 0],
        a_rate=trajectory[..., 1],
        b=trajectory[..., 2],
        b_rate=trajectory[..., 3],
    )
