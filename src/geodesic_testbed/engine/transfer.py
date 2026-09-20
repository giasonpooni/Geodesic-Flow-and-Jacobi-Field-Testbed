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
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

Array = np.ndarray

# Index of each component inside the flow state, counted from the end, so the
# same block can ride along with either a scalar or a parametric flow.
COMPONENTS = ("a", "a_rate", "b", "b_rate")
INITIAL_STATE = (1.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class FocusEvent:
    """A refined zero of one column of ``Phi``, with how well it is located."""

    column: str
    arc_length: float
    location_uncertainty: float
    derivative: float
    bracket: tuple[float, float]
    order: int
    distance_to_end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hermite(t: float, left: float, right: float, left_slope: float,
             right_slope: float, h: float) -> tuple[float, float]:
    """Cubic Hermite value and derivative in ``t`` on a unit interval."""
    t2, t3 = t * t, t * t * t
    value = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * left
        + (t3 - 2.0 * t2 + t) * h * left_slope
        + (-2.0 * t3 + 3.0 * t2) * right
        + (t3 - t2) * h * right_slope
    )
    slope = (
        (6.0 * t2 - 6.0 * t) * left
        + (3.0 * t2 - 4.0 * t + 1.0) * h * left_slope
        + (-6.0 * t2 + 6.0 * t) * right
        + (3.0 * t2 - 2.0 * t) * h * right_slope
    )
    return value, slope


def _refine_root(
    left_s: float,
    right_s: float,
    left: float,
    right: float,
    left_slope: float,
    right_slope: float,
    *,
    iterations: int = 60,
) -> tuple[float, float, float]:
    """Root of the cubic Hermite through one bracket: location, uncertainty, slope.

    Newton, safeguarded by bisection so it cannot leave the bracket even where
    the interpolant has a near-zero derivative. The uncertainty returned is the
    distance the root moved from the linear estimate, which is the size of the
    error the refinement removed and a conservative proxy for what remains.
    """
    h = right_s - left_s
    linear = left / (left - right)
    lower, upper = 0.0, 1.0
    t = min(max(linear, 0.0), 1.0)
    for _ in range(iterations):
        value, slope = _hermite(t, left, right, left_slope, right_slope, h)
        if value == 0.0:
            break
        if value * left < 0.0:
            upper = t
        else:
            lower = t
        candidate = t - value / slope if slope != 0.0 else 0.5 * (lower + upper)
        if not (lower < candidate < upper):
            candidate = 0.5 * (lower + upper)
        if abs(candidate - t) <= 1e-16:
            t = candidate
            break
        t = candidate
    _, slope = _hermite(t, left, right, left_slope, right_slope, h)
    return left_s + t * h, abs(t - linear) * abs(h), slope / h


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
        covariance = _validated_covariance(covariance)
        phi = self.matrices()
        return phi @ covariance @ np.swapaxes(phi, -1, -2)

    def focus_events(self, *, component: str = "b") -> list[FocusEvent]:
        """Every focus of one column, located to better than the sample spacing.

        ``b`` vanishing away from the start is a conjugate point of the heading
        variation: paths that left at different angles meet again. ``a``
        vanishing is the corresponding focus of the lateral variation, and it is
        generally somewhere else.

        A sign change between samples with linear interpolation is accurate
        enough to *report* a focus and too coarse to *constrain a route by* one:
        the location error is first order in the sample spacing, and a route
        clearance of "0.2 away from a focus" is meaningless if the focus itself
        is only known to 0.05. Both the value and the slope are available at
        every sample, so each root is refined on the cubic Hermite interpolant
        through them -- fourth-order accurate instead of first -- by a Newton
        iteration safeguarded with bisection, and the shift from the linear
        estimate is reported as the location uncertainty.
        """
        values, slopes = self._column(component)
        grid = self.arc_length
        # An exact zero is not reliably exact in floating point: sin(pi) comes
        # back as 1.2e-16, so a focus that lands on a sample would be missed by
        # a strict `== 0` and by the sign-change test alike. Anything within a
        # few ulps of zero, measured against the column's own scale, counts.
        scale = float(np.max(np.abs(values))) or 1.0
        negligible = 8.0 * np.finfo(float).eps * scale
        events: list[FocusEvent] = []
        consumed = -1
        for index in range(len(values)):
            if grid[index] < 0.0 or index <= consumed:
                continue
            value = float(values[index])
            # b starts at zero by construction; that is the variation's origin,
            # not a focus.
            if index == 0 and abs(value) <= negligible and component == "b":
                continue
            if abs(value) <= negligible:
                location, uncertainty, derivative = (
                    float(grid[index]),
                    0.0,
                    float(slopes[index]),
                )
                bracket = (float(grid[index]), float(grid[index]))
                # Do not also report the sign change straddling this sample.
                consumed = index
            elif index + 1 < len(values) and value * float(values[index + 1]) < 0.0:
                if abs(float(values[index + 1])) <= negligible:
                    continue  # the next sample is the zero; report it there
                location, uncertainty, derivative = _refine_root(
                    float(grid[index]),
                    float(grid[index + 1]),
                    value,
                    float(values[index + 1]),
                    float(slopes[index]),
                    float(slopes[index + 1]),
                )
                bracket = (float(grid[index]), float(grid[index + 1]))
            else:
                continue
            events.append(
                FocusEvent(
                    column=component,
                    arc_length=location,
                    location_uncertainty=uncertainty,
                    derivative=derivative,
                    bracket=bracket,
                    order=len(events) + 1,
                    distance_to_end=float(grid[-1]) - location,
                )
            )
        return events

    def focus_points(self, *, component: str = "b") -> list[float]:
        """Refined focus locations only, for callers that want the bare numbers."""
        return [event.arc_length for event in self.focus_events(component=component)]

    def _column(self, component: str) -> tuple[Array, Array]:
        try:
            return {"a": (self.a, self.a_rate), "b": (self.b, self.b_rate)}[component]
        except KeyError as exc:  # pragma: no cover - guard
            raise KeyError(f"column must be 'a' or 'b', not {component!r}") from exc

    def clearance_from_focus(self, *, component: str = "b") -> float:
        """Smallest distance from any sample to a focus of this column.

        ``inf`` when the path has none -- which is the useful answer for a route
        constraint, since nothing is being approached.
        """
        events = self.focus_events(component=component)
        if not events:
            return float("inf")
        return float(
            min(
                min(abs(event.arc_length - float(self.arc_length[0])),
                    abs(float(self.arc_length[-1]) - event.arc_length))
                for event in events
            )
        )

    def crossed_first_conjugate_point(self, *, component: str = "b") -> bool:
        """Whether the path continues past the first focus of this column."""
        events = self.focus_events(component=component)
        return bool(events) and events[0].arc_length < float(self.arc_length[-1])


def _validated_covariance(covariance) -> Array:
    """A 2x2 starting-pose covariance, or a refusal that says which property failed.

    Congruence by ``Phi`` preserves indefiniteness as faithfully as it preserves
    anything else, so a matrix that is not a covariance in goes to something
    that is not a covariance out, silently and with plausible-looking numbers.
    Symmetry alone does not catch it: ``[[1, 2], [2, 1]]`` is symmetric and has
    eigenvalues 3 and -1.
    """
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError("covariance must be 2x2 in the (lateral, heading) basis")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    if not np.allclose(covariance, covariance.T, atol=0.0, rtol=1e-12):
        raise ValueError("covariance must be symmetric")
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    # Scale the tolerance to the matrix: an absolute floor would reject a
    # legitimate covariance in micrometres and accept a bad one in metres.
    tolerance = 1e-12 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if eigenvalues[0] < -tolerance:
        raise ValueError(
            "covariance must be positive semidefinite; smallest eigenvalue is "
            f"{eigenvalues[0]!r}"
        )
    return covariance


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
