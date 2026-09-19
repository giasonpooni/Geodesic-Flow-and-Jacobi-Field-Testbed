"""The figure: four panels, one story, drawn from the report.

Panels B and C are drawn from the report's own numbers, so the figure cannot
drift from the machine-readable artefact.  Panels A and D redraw a couple of
dense curves from the library, which is deterministic and cheap.

Colour carries one thing only -- the curvature -- and keeps the same hue in
every panel.  The integrator, where it appears, is encoded by marker and line
style rather than by a second set of hues.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .jacobi import integrate_jacobi, jacobi_reference
from .spaceforms import all_space_forms

# Categorical slots 1-3 of the validated default palette, assigned to entities
# (the three curvatures) and never reassigned between panels.
CURVATURE_COLOURS = {"K=0": "#2a78d6", "K=+1": "#eb6834", "K=-1": "#1baf7a"}
INTEGRATOR_MARKERS = {"euler": "o", "midpoint": "s", "rk4": "D"}
INTEGRATOR_DASHES = {"euler": (1, 0), "midpoint": (1, 0), "rk4": (1, 0)}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e6e5e0"

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK_SOFT,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.0,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "lines.linewidth": 1.6,
}


def _style_axes(ax) -> None:
    ax.grid(True, which="major", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _panel_title(ax, title: str, subtitle: str) -> None:
    ax.text(0.0, 1.155, title, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=INK, va="bottom")
    ax.text(0.0, 1.120, subtitle, transform=ax.transAxes, fontsize=8.2,
            color=INK_SOFT, va="top", linespacing=1.6)


def _soft_legend(ax, **kwargs):
    legend = ax.legend(frameon=True, facecolor=SURFACE, edgecolor="none",
                       framealpha=0.92, **kwargs)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    return legend


def _panel_fields(ax, *, arc_length: float = 2.0, n_steps: int = 20) -> None:
    """Panel A: the three reference solutions, with the rk4 solver on top of them."""
    dense = np.linspace(0.0, arc_length, 400)
    worst = 0.0
    for form in all_space_forms():
        colour = CURVATURE_COLOURS[form.label]
        ax.plot(
            dense,
            jacobi_reference(dense, form.K),
            color=colour,
            linewidth=1.8,
            zorder=3,
            label=f"{form.label}   j(s) = {form.reference_solution_name}",
        )
        grid, j, _ = integrate_jacobi(
            form.K, length=arc_length, n_steps=n_steps, method="rk4"
        )
        worst = max(worst, float(np.max(np.abs(j - jacobi_reference(grid, form.K)))))
        ax.plot(
            grid,
            j,
            linestyle="none",
            marker="o",
            markersize=4.4,
            markerfacecolor=SURFACE,
            markeredgecolor=colour,
            markeredgewidth=1.2,
            zorder=4,
        )
    ax.set_xlabel("arc length  s")
    ax.set_ylabel("Jacobi field  j(s)")
    ax.set_xlim(0.0, arc_length)
    ax.set_ylim(0.0, 3.9)
    ax.text(
        0.035,
        0.955,
        f"rings: rk4, h = {arc_length / n_steps:g}\nworst error {worst:.1e}",
        transform=ax.transAxes,
        fontsize=8.0,
        color=INK_SOFT,
        va="top",
        linespacing=1.5,
    )
    _soft_legend(ax, loc="lower right", handlelength=1.8)
    _panel_title(
        ax,
        "A · The three reference solutions",
        "j'' + K j = 0,  j(0) = 0,  j'(0) = 1",
    )


def _panel_convergence(ax, report: dict[str, Any]) -> None:
    """Panel B: measured order of accuracy against the formal order."""
    rows = [
        row
        for row in report["results"]["jacobi_ode_convergence"]
        if row["regime"] == "converging"
    ]
    for row in rows:
        colour = CURVATURE_COLOURS[row["curvature_label"]]
        steps = [level["h"] for level in row["levels"]]
        errors = [level["max_error"] for level in row["levels"]]
        ax.plot(
            steps,
            errors,
            color=colour,
            linewidth=1.2,
            marker=INTEGRATOR_MARKERS[row["integrator"]],
            markersize=4.2,
            markeredgecolor=SURFACE,
            markeredgewidth=0.7,
            zorder=3,
        )
    anchor_h = 0.2
    for order, anchor_error in ((1, 3e-1), (2, 1e-2), (4, 2e-5)):
        guide = np.array([anchor_h, anchor_h / 300.0])
        ax.plot(
            guide,
            anchor_error * (guide / anchor_h) ** order,
            color=INK_MUTED,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            zorder=2,
        )
        ax.text(
            0.275,
            anchor_error,
            f"h^{order}",
            fontsize=8.0,
            color=INK_MUTED,
            ha="right",
            va="center",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("step size  h")
    ax.set_ylabel("max |j_numerical - j_exact|  on  s in [0, 2]")
    ax.set_xlim(0.36, 5.5e-4)
    ax.set_ylim(1e-16, 3.0)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=CURVATURE_COLOURS[label], linewidth=1.6, label=label)
        for label in ("K=+1", "K=-1")
    ] + [
        Line2D(
            [],
            [],
            color=INK_SOFT,
            linewidth=0,
            marker=INTEGRATOR_MARKERS[name],
            markersize=4.6,
            label=f"{name} (order {order})",
        )
        for name, order in (("euler", 1), ("midpoint", 2), ("rk4", 4))
    ]
    _soft_legend(ax, handles=handles, loc="lower left", ncols=2,
                 handlelength=1.6, columnspacing=1.2)
    fitted = {
        (row["curvature_label"], row["integrator"]): row["fitted_order"] for row in rows
    }
    summary = "   ".join(
        f"{name} {fitted[('K=+1', name)]:.2f} / {fitted[('K=-1', name)]:.2f}"
        for name in ("euler", "midpoint", "rk4")
    )
    _panel_title(
        ax,
        "B · Measured order of accuracy",
        f"fitted order, K=+1 / K=-1:   {summary}\n"
        "K = 0 is omitted: every method reproduces j(s) = s exactly, so there is "
        "no order to measure",
    )


def _panel_validity(ax, report: dict[str, Any], *, arc_length: float = 1.0,
                    tolerance: float = 1e-6) -> None:
    """Panel C: the window in which the first-order prediction is usable."""
    rows = [
        row
        for row in report["results"]["first_order_validity"]
        if row["arc_length"] == arc_length
    ]
    epsilons = np.array([sample["epsilon"] for sample in rows[0]["samples"]])
    for row in rows:
        colour = CURVATURE_COLOURS[row["curvature_label"]]
        model = np.array([s["relative_deviation_model"] for s in row["samples"]])
        noise = np.array(
            [s["mixed_resolution_vs_exact_relative"] for s in row["samples"]]
        )
        ax.plot(
            epsilons,
            row["theory_coefficient"] * epsilons**2,
            color=colour,
            linewidth=1.5,
            zorder=3,
        )
        ax.plot(
            epsilons[::2],
            model[::2],
            linestyle="none",
            marker="o",
            markersize=4.2,
            markerfacecolor=SURFACE,
            markeredgecolor=colour,
            markeredgewidth=1.1,
            zorder=4,
        )
        ax.plot(
            epsilons,
            np.maximum(noise, 1e-17),
            color=colour,
            linewidth=1.2,
            linestyle=(0, (1.5, 2)),
            zorder=3,
        )
    bench = (np.deg2rad(0.25), np.deg2rad(2.0))
    ax.axvspan(*bench, color="#f1eee8", zorder=0)
    ax.text(
        float(np.sqrt(bench[0] * bench[1])),
        4e2,
        "0.25° – 2°\nbench range",
        fontsize=8.0,
        color=INK_SOFT,
        ha="center",
        va="top",
        linespacing=1.5,
    )
    ax.axhline(tolerance, color=INK_MUTED, linewidth=0.9, zorder=2)
    ax.text(
        1.3e-8,
        tolerance * 2.2,
        f"{tolerance:g} relative accuracy",
        fontsize=8.0,
        color=INK_MUTED,
    )
    marked = next(row for row in rows if row["curvature_label"] == "K=+1")
    star = next(
        entry["epsilon_star_measured"]
        for entry in marked["epsilon_star"]
        if entry["tolerance"] == tolerance
    )
    ax.plot(
        [star],
        [tolerance],
        marker="o",
        markersize=6.0,
        color=CURVATURE_COLOURS["K=+1"],
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        zorder=6,
    )
    ax.annotate(
        f"K=+1 breaks down\nat eps = {star:.2e}",
        xy=(star, tolerance),
        xytext=(star * 0.025, tolerance * 4e3),
        fontsize=8.0,
        color=INK_SOFT,
        linespacing=1.5,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("initial perturbation  eps  (radians)")
    ax.set_ylabel("relative error of the first-order prediction")
    ax.set_ylim(1e-17, 1e3)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=CURVATURE_COLOURS[label], linewidth=1.6, label=label)
        for label in ("K=0", "K=+1", "K=-1")
    ] + [
        Line2D([], [], color=INK_SOFT, linewidth=1.5,
               label="model: cn_K(s)^2 eps^2 / 24"),
        Line2D([], [], color=INK_SOFT, linewidth=1.2, linestyle=(0, (1.5, 2)),
               label="flow noise (mismatched h)"),
    ]
    _soft_legend(ax, handles=handles, loc="lower right", ncols=1,
                 handlelength=1.8, columnspacing=1.2,
                 bbox_to_anchor=(1.0, 0.015))
    _panel_title(
        ax,
        "C · Where the first-order variation stops working",
        f"s = {arc_length:g}; the prediction is trustworthy only between the flow-noise\n"
        "wall (dotted) and the eps^2 model wall (solid)",
    )


def _panel_conjugate(ax, report: dict[str, Any]) -> None:
    """Panel D: the zero of the Jacobi field is where minimality is lost."""
    conjugate = report["results"]["conjugate_point"]
    colour = CURVATURE_COLOURS["K=+1"]
    arc = np.array([point["arc_length"] for point in conjugate["curve"]])
    field = np.array([point["jacobi_field"] for point in conjugate["curve"]])
    distance = np.array(
        [point["flowed_distance_to_start"] for point in conjugate["curve"]]
    )
    ax.axvspan(np.pi, 2.0 * np.pi, color="#f1eee8", zorder=0)
    ax.plot(arc, arc, color=INK_MUTED, linewidth=1.0, zorder=2)
    ax.plot(arc, distance, color=colour, linewidth=1.8, zorder=4)
    ax.plot(arc, field, color=colour, linewidth=1.4, linestyle=(0, (5, 2.5)), zorder=3)
    ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=1)
    ax.plot([np.pi], [0.0], marker="o", markersize=6.0, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6)
    ax.annotate(
        "conjugate point\nj(pi) = 0",
        xy=(np.pi, 0.0),
        xytext=(np.pi - 1.75, -1.45),
        fontsize=8.2,
        color=INK_SOFT,
        linespacing=1.5,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    ax.text(4.45, 4.22, "arc length  s", fontsize=8.2, color=INK_MUTED, rotation=31)
    ax.text(5.42, 2.25, "distance back\nto the start", fontsize=8.2, color=colour,
            linespacing=1.5, ha="center")
    ax.text(1.62, 1.12, "j(s) = sin s", fontsize=8.2, color=colour)
    ax.text(
        np.pi + 0.12,
        7.35,
        "past here the geodesic still solves the geodesic\n"
        "equation, but a shorter path to the same point exists",
        fontsize=8.0,
        color=INK_SOFT,
        va="top",
        linespacing=1.5,
    )
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(-1.9, 7.6)
    ax.set_xticks([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
    ax.set_xlabel("arc length  s  along the great circle")
    ax.set_ylabel("length")
    _panel_title(
        ax,
        "D · A zero of the Jacobi field, and what it costs",
        f"K = +1, rk4;  measured zero at s = {conjugate['jacobi_first_zero']:.12f}",
    )


def build_figure(report: dict[str, Any]):
    """Assemble the four-panel figure and return the matplotlib ``Figure``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
        for ax in axes.flat:
            _style_axes(ax)
        _panel_fields(axes[0, 0], arc_length=report["config"]["arc_length"])
        _panel_convergence(axes[0, 1], report)
        _panel_validity(axes[1, 0], report)
        _panel_conjugate(axes[1, 1], report)

        summary = report["summary"]
        figure.suptitle(
            "Geodesic flow and Jacobi fields on the three constant-curvature surfaces",
            x=0.012,
            y=0.983,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
            color=INK,
        )
        figure.text(
            0.012,
            0.947,
            "Numerical geodesics and the Jacobi equation j'' + K j = 0, verified "
            "against s, sin s and sinh s",
            ha="left",
            fontsize=10.0,
            color=INK_SOFT,
        )
        figure.text(
            0.012,
            0.012,
            f"{summary['n_checks']} declared checks, {summary['n_failed']} failed  ·  "
            f"report {report['schema']}  ·  content hash {report['content_hash'][:16]}  ·  "
            f"numpy {report['environment']['numpy']}, python {report['environment']['python']}",
            ha="left",
            fontsize=7.8,
            color=INK_MUTED,
        )
        figure.tight_layout(rect=(0.0, 0.028, 1.0, 0.922))
        figure.subplots_adjust(hspace=0.46, wspace=0.22)
    return figure


def write_figure(report: dict[str, Any], path, dpi: int = 200) -> None:
    import pathlib

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(report)
    figure.savefig(path, dpi=dpi, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(figure)
