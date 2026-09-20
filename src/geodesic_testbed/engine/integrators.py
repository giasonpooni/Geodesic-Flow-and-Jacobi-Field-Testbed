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
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise TypeError("n_steps must be an integer")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    length = float(length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not np.isfinite(s0):
        raise ValueError("s0 must be finite")
    integrator = get_integrator(method)
    h = length / float(n_steps)
    y0 = np.asarray(y0, dtype=float)
    if not np.all(np.isfinite(y0)):
        raise ValueError("the initial state must be finite")
    trajectory = np.empty((n_steps + 1,) + y0.shape, dtype=float)
    trajectory[0] = y0
    grid = s0 + h * np.arange(n_steps + 1, dtype=float)
    y = y0
    for index in range(n_steps):
        y = integrator.step(rhs, float(grid[index]), y, h)
        trajectory[index + 1] = y
    return grid, trajectory


#: A predicate on a batch of states, ``True`` where the state is still usable.
Guard = Callable[[State], np.ndarray]


def integrate_guarded(
    rhs: Rhs,
    y0: State,
    *,
    length: float,
    n_steps: int,
    method: str = "rk4",
    s0: float = 0.0,
    guard: Guard,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`integrate`, stopping each trajectory where ``guard`` says to.

    The reason this is not a flag on :func:`integrate` is that stopping is not
    a variation on integrating: a trajectory that has left the region where its
    equation means anything must not be advanced *at all*, because the next
    step evaluates the right-hand side inside that region. Truncating the
    output afterwards removes the samples and not the arithmetic that produced
    them -- on a chart that degenerates, that arithmetic overflows.

    Trajectories are guarded independently and are advanced together. One that
    stops is frozen at its last usable state, so the batch keeps its shape and
    the right-hand side is only ever evaluated at states that passed the guard;
    the frozen rows are meaningless and the returned counts say where each one
    stopped.

    Returns ``(grid, trajectory, valid_samples)``, where ``valid_samples`` has
    the batch's shape and counts the leading samples of each trajectory that
    are real -- ``n_steps + 1`` for one that never tripped the guard.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise TypeError("n_steps must be an integer")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    length = float(length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not np.isfinite(s0):
        raise ValueError("s0 must be finite")
    integrator = get_integrator(method)
    h = length / float(n_steps)
    y0 = np.asarray(y0, dtype=float)
    if not np.all(np.isfinite(y0)):
        raise ValueError("the initial state must be finite")
    batch = y0.shape[:-1]
    if not np.all(np.asarray(guard(y0), dtype=bool)):
        raise ValueError("the initial state does not pass the guard")

    trajectory = np.empty((n_steps + 1,) + y0.shape, dtype=float)
    trajectory[0] = y0
    grid = s0 + h * np.arange(n_steps + 1, dtype=float)
    valid = np.full(batch, n_steps + 1, dtype=int)
    running = np.ones(batch, dtype=bool)
    y = y0
    for index in range(n_steps):
        if not running.any():
            trajectory[index + 1] = y
            continue
        stepped = integrator.step(rhs, float(grid[index]), y, h)
        usable = np.asarray(guard(stepped), dtype=bool) & np.all(
            np.isfinite(stepped), axis=-1
        )
        stopping = running & ~usable
        if stopping.any():
            # Freeze, so that the next step's right-hand side is evaluated at a
            # state the guard already accepted.
            stepped = np.where(stopping[..., None], y, stepped)
            valid = np.where(stopping, index + 1, valid)
            running = running & ~stopping
        stepped = np.where(running[..., None], stepped, y)
        y = stepped
        trajectory[index + 1] = y
    return grid, trajectory, valid


def integrate_on_grid(
    rhs: Rhs,
    y0: State,
    grid,
    *,
    method: str = "rk4",
) -> np.ndarray:
    """Integrate onto a caller-supplied, possibly non-uniform, arc-length grid.

    :func:`integrate` owns the fixed-step case, which is what a convergence
    study needs. This is for the other case: a caller who has already decided
    where the samples must land -- a CAD parameterisation, a toolpath, a
    measured scan -- and wants one step per interval between them.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a one-dimensional array with at least two samples")
    if not np.all(np.isfinite(grid)):
        raise ValueError("grid must be finite")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("grid must be strictly increasing")
    integrator = get_integrator(method)
    y0 = np.asarray(y0, dtype=float)
    if not np.all(np.isfinite(y0)):
        raise ValueError("the initial state must be finite")
    trajectory = np.empty((grid.size,) + y0.shape, dtype=float)
    trajectory[0] = y0
    y = y0
    for index in range(grid.size - 1):
        h = float(grid[index + 1] - grid[index])
        y = integrator.step(rhs, float(grid[index]), y, h)
        trajectory[index + 1] = y
    return trajectory


def step_ladder(n_min: int = 10, n_levels: int = 9, factor: int = 2) -> tuple[int, ...]:
    """Geometric ladder of step counts, e.g. 10, 20, 40, ... ."""
    return tuple(int(n_min * factor**level) for level in range(n_levels))
