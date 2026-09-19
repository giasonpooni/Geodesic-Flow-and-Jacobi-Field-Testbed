"""Fixed-step explicit integrators with known, and therefore testable, orders.

Three methods are kept on purpose.  A convergence study with a single
integrator can only show that *something* converges; a study with methods of
order 1, 2 and 4 shows that the measured slopes separate exactly as the theory
says they must, which is what turns the sweep into evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

State = np.ndarray
Rhs = Callable[[float, State], State]


def euler_step(f: Rhs, s: float, y: State, h: float) -> State:
    return y + h * f(s, y)


def midpoint_step(f: Rhs, s: float, y: State, h: float) -> State:
    k1 = f(s, y)
    k2 = f(s + 0.5 * h, y + 0.5 * h * k1)
    return y + h * k2


def rk4_step(f: Rhs, s: float, y: State, h: float) -> State:
    k1 = f(s, y)
    k2 = f(s + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(s + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(s + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass(frozen=True)
class Integrator:
    name: str
    step: Callable[[Rhs, float, State, float], State]
    order: int


INTEGRATORS: dict[str, Integrator] = {
    "euler": Integrator("euler", euler_step, 1),
    "midpoint": Integrator("midpoint", midpoint_step, 2),
    "rk4": Integrator("rk4", rk4_step, 4),
}

DEFAULT_INTEGRATORS: tuple[str, ...] = ("euler", "midpoint", "rk4")


def get_integrator(name: str) -> Integrator:
    try:
        return INTEGRATORS[name]
    except KeyError as exc:  # pragma: no cover - guard
        raise KeyError(f"unknown integrator {name!r}; have {sorted(INTEGRATORS)}") from exc


def integrate(
    rhs: Rhs,
    y0: State,
    *,
    length: float,
    n_steps: int,
    method: str = "rk4",
    s0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``y' = rhs(s, y)`` over ``[s0, s0 + length]`` in ``n_steps`` equal steps.

    ``y0`` may carry leading batch axes, in which case independent
    trajectories are advanced together with the same step sequence.

    Returns the parameter grid (``n_steps + 1`` samples) and the trajectory,
    of shape ``(n_steps + 1,) + y0.shape``.  The grid is rebuilt from the step
    index rather than accumulated, so the abscissae carry no drift of their own.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    integrator = get_integrator(method)
    h = float(length) / float(n_steps)
    y0 = np.asarray(y0, dtype=float)
    trajectory = np.empty((n_steps + 1,) + y0.shape, dtype=float)
    trajectory[0] = y0
    grid = s0 + h * np.arange(n_steps + 1, dtype=float)
    y = y0
    for index in range(n_steps):
        y = integrator.step(rhs, float(grid[index]), y, h)
        trajectory[index + 1] = y
    return grid, trajectory


def step_ladder(n_min: int = 10, n_levels: int = 9, factor: int = 2) -> tuple[int, ...]:
    """Geometric ladder of step counts, e.g. 10, 20, 40, ... ."""
    return tuple(int(n_min * factor**level) for level in range(n_levels))
