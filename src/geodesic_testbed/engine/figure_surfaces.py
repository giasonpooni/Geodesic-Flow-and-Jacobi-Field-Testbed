"""The stage-two figure: the same instrument, on surfaces that vary.

Drawn from the stage-two report. Colour carries the surface, and the first
three slots are deliberately the ones figure one gave to ``K = 0``, ``K = +1``
and ``K = -1``: the plate, the spherical cap and the pseudosphere are the same
three curvatures, reached by the general machinery instead of a closed form.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .figure import _RC, GRID, INK, INK_MUTED, INK_SOFT, SURFACE, _panel_title, _style_axes

CASE_COLOURS = {
    "plate": "#2a78d6",
    "spherical-cap": "#eb6834",
    "pseudosphere": "#1baf7a",
    "rolled-sheet": "#eda100",
    "saddle": "#e87ba4",
    "torus": "#008300",
}
CASE_ORDER = ("plate", "rolled-sheet", "spherical-cap", "pseudosphere", "saddle", "torus")


def _curves(report: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for row in report["results"]["envelopes"]:
        curve = row["curve"]
        out[row["case"]] = {
            "s": np.array([point["arc_length"] for point in curve]),
            "K": np.array([point["gaussian_curvature"] for point in curve]),
            "j": np.array([point["jacobi_field"] for point in curve]),
        }
    return out


def _panel_curvature(ax, report: dict[str, Any]) -> None:
    curves = _curves(report)
    for case in CASE_ORDER:
        # The plate and the rolled sheet are both flat and lie on top of each
        # other; the sheet is dashed so it stays visible under the plate rather
        # than being silently hidden by it.
        style = (0, (5, 3)) if case == "rolled-sheet" else "-"
        ax.plot(curves[case]["s"], curves[case]["K"], color=CASE_COLOURS[case],
                linewidth=1.7, linestyle=style, zorder=3)
    ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=1)
    labels = {
        "plate": (0.42, -0.14, "plate, K = 0"),
        "rolled-sheet": (2.20, 0.12, "rolled sheet, K = 0 (dashed, coincides)"),
        "spherical-cap": (2.55, 1.06, "spherical cap, K = +1"),
        "pseudosphere": (0.95, -0.93, "pseudosphere, K = -1"),
        "saddle": (1.05, -0.62, "saddle"),
        "torus": (1.05, 0.33, "torus"),
    }
    for case, (x, y, text) in labels.items():
        ax.text(x, y, text, fontsize=8.2, color=CASE_COLOURS[case])
    ax.set_xlabel("arc length  s  along the nominal path")
    ax.set_ylabel("Gaussian curvature  K(s)")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-1.25, 1.3)
    _panel_title(
        ax,
        "A · Curvature is now a function of position",
        "three cases kept constant on purpose, to anchor the other three;\n"
        "each path runs as far as its surface allows, not to a common length",
    )


def _panel_fields(ax, report: dict[str, Any]) -> None:
    curves = _curves(report)
    anchors = {
        row["case"]: row["reference"] for row in report["results"]["anchored_to_closed_forms"]
    }
    closed = {"s": lambda x: x, "sin(s)": np.sin, "sinh(s)": np.sinh}
    for case in CASE_ORDER:
        data = curves[case]
        ax.plot(data["s"], data["j"], color=CASE_COLOURS[case], linewidth=1.7, zorder=3)
        if case in anchors:
            ax.plot(
                data["s"][::8],
                closed[anchors[case]](data["s"][::8]),
                linestyle="none",
                marker="o",
                markersize=4.0,
                markerfacecolor=SURFACE,
                markeredgecolor=INK_SOFT,
                markeredgewidth=0.9,
                zorder=4,
            )
    ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=1)
    ax.plot([np.pi], [0.0], marker="o", markersize=6.0,
            color=CASE_COLOURS["spherical-cap"], markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=6)
    ax.annotate(
        "focus at s = pi:\nneighbouring paths meet",
        xy=(np.pi, 0.0),
        xytext=(2.25, -1.25),
        fontsize=8.0,
        color=INK_SOFT,
        linespacing=1.5,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    for case, (x, y, text, align) in {
        "pseudosphere": (1.46, 2.22, "pseudosphere", "right"),
        "saddle": (2.06, 2.24, "saddle", "left"),
        "plate": (2.06, 1.94, "plate, rolled sheet", "left"),
        "torus": (2.06, 1.66, "torus", "left"),
        "spherical-cap": (3.35, 0.34, "spherical cap", "left"),
    }.items():
        ax.text(x, y, text, fontsize=8.2, color=CASE_COLOURS[case], ha=align)
    ax.text(
        0.035,
        0.955,
        "rings: the closed form the general\nsolver has to reproduce",
        transform=ax.transAxes,
        fontsize=8.0,
        color=INK_SOFT,
        va="top",
        linespacing=1.5,
    )
    ax.set_xlabel("arc length  s")
    ax.set_ylabel("heading column  b(s)")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-1.6, 3.2)
    _panel_title(
        ax,
        "B · How far an aiming error is carried",
        "b'' + K(gamma(s)) b = 0, integrated along the path together with the path\n"
        "and with the lateral column a; det Phi = a b' - a' b holds to 2e-14",
    )


def _panel_two_routes(ax, report: dict[str, Any]) -> None:
    rows = {row["case"]: row for row in report["results"]["two_routes"]}
    epsilons = np.array([s["epsilon"] for s in rows["plate"]["samples"]])
    for case in CASE_ORDER:
        row = rows[case]
        values = np.array([s["relative_difference"] for s in row["samples"]])
        ax.plot(
            epsilons,
            row["fitted_coefficient"] * epsilons**2,
            color=CASE_COLOURS[case],
            linewidth=1.4,
            zorder=3,
        )
        ax.plot(
            epsilons,
            values,
            linestyle="none",
            marker="o",
            markersize=4.2,
            markerfacecolor=SURFACE,
            markeredgecolor=CASE_COLOURS[case],
            markeredgewidth=1.1,
            zorder=4,
        )
    orders = [
        row["fitted_order_jacobi"]
        for row in report["results"]["self_convergence"]
        if row["fitted_order_jacobi"] is not None
    ]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("central-difference step in heading,  eps  (radians)")
    ax.set_ylabel("relative difference between the two routes")
    ax.set_ylim(1e-10, 1e-1)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=CASE_COLOURS[case], linewidth=1.6,
               label=f"{case}   {rows[case]['fitted_coefficient']:.4f}")
        for case in CASE_ORDER
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor=SURFACE,
        edgecolor="none",
        framealpha=0.92,
        handlelength=1.6,
        title="fitted coefficient of eps^2",
    )
    legend.get_title().set_color(INK_SOFT)
    legend.get_title().set_fontsize(8.0)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    _panel_title(
        ax,
        "C · Two independent routes to the same field",
        "exactly 1/6 where the chord is the variation (plate, sphere); the excess\n"
        "elsewhere is bending in space.  rk4 self-convergence "
        f"{min(orders):.2f}-{max(orders):.2f}.",
    )


def _panel_decision(ax, report: dict[str, Any]) -> None:
    """Panel D: low forward amplification is not the same thing as robustness."""
    scan = report["results"]["heading_scan"]
    rows = sorted(scan["headings"], key=lambda row: row["heading_degrees"])
    headings = np.array([row["heading_degrees"] for row in rows])
    worst = np.array([row["max_forward_amplification"] for row in rows])
    focus = np.array([row["passes_a_focus"] for row in rows])
    colour = CASE_COLOURS["torus"]

    ax.plot(headings, worst, color=colour, linewidth=1.6, zorder=3)
    ax.plot(headings[~focus], worst[~focus], linestyle="none", marker="o", markersize=4.6,
            markerfacecolor=SURFACE, markeredgecolor=colour, markeredgewidth=1.3, zorder=4)
    ax.plot(headings[focus], worst[focus], linestyle="none", marker="o", markersize=4.6,
            color=colour, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)

    lowest = scan["lowest_amplification"]
    clear = scan["lowest_amplification_clear_of_a_focus"]
    highest = scan["highest_amplification"]
    annotations = [
        (highest, "highest amplification", (0.22, 0.86)),
        (lowest, "lowest amplification,\nbut passes a focus", (0.04, 0.16)),
    ]
    if clear is not None:
        annotations.append((clear, "lowest clear of a focus", (0.58, 0.46)))
    for row, text, target in annotations:
        ax.plot([row["heading_degrees"]], [row["max_forward_amplification"]], marker="o",
                markersize=7.5, color=colour, markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=6)
        ax.annotate(
            f"{text}\n{row['heading_degrees']:.0f}°, "
            f"max|b| = {row['max_forward_amplification']:.2f}",
            xy=(row["heading_degrees"], row["max_forward_amplification"]),
            xytext=target,
            textcoords="axes fraction",
            fontsize=8.2,
            color=INK_SOFT,
            linespacing=1.5,
            arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
        )

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5.0, markerfacecolor=SURFACE,
               markeredgecolor=colour, markeredgewidth=1.3, label="clear of any focus"),
        Line2D([], [], linestyle="none", marker="o", markersize=5.0, color=colour,
               markeredgecolor=SURFACE, markeredgewidth=0.8,
               label=f"passes a focus ({scan['n_headings_passing_a_focus']} of "
                     f"{scan['n_headings']})"),
    ]
    legend = ax.legend(handles=handles, loc="upper right", frameon=True, facecolor=SURFACE,
                       edgecolor="none", framealpha=0.92, handlelength=1.2, ncols=1,
                       columnspacing=1.2)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    ax.set_xlabel("starting heading (degrees from the u direction)")
    ax.set_ylabel("max |b(s)| anywhere along the path")
    ax.set_xlim(-6.0, 186.0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yscale("log")
    ax.set_ylim(min(worst) * 0.66, max(worst) * 4.2)
    _panel_title(
        ax,
        "D \u00b7 The decision, and the trap in it",
        f"{scan['n_headings']} headings on the {scan['surface']}, ranked by forward "
        "amplification alone;\nevery one of the best few buys its score with a focus",
    )


def build_figure(report: dict[str, Any]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
        for ax in axes.flat:
            _style_axes(ax)
        _panel_curvature(axes[0, 0], report)
        _panel_fields(axes[0, 1], report)
        _panel_two_routes(axes[1, 0], report)
        _panel_decision(axes[1, 1], report)

        summary = report["summary"]
        figure.suptitle(
            "Curvature-aware path sensitivity on surfaces where the curvature varies",
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
            "The geodesic flow and j'' + K j = 0 on parametric surfaces, anchored to "
            "the constant-curvature stage",
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
        figure.subplots_adjust(hspace=0.50, wspace=0.22)
    return figure


def write_figure(report: dict[str, Any], path, dpi: int = 200) -> None:
    import pathlib

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(report)
    figure.savefig(path, dpi=dpi, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(figure)
