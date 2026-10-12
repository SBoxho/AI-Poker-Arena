"""One look for every research notebook: matplotlib style, palette and a few plotting helpers.

Static matplotlib PNGs are the default output of the notebooks because GitHub renders them (it does not
run Plotly's JavaScript). The style is dark (the app's "poker felt" surface, green accent) with a
categorical palette that was validated for the dark surface (lightness band, adjacent-colour separation
for colour-vision deficiencies, 3:1 contrast). Notebooks never rely on colour alone: the helpers add
marker shape / fill / direct value labels as a second channel.

    from poker_arena.explain import notebook_plots as nbp

    nbp.apply_style()                              # once per notebook
    fig, ax = plt.subplots()
    nbp.forest_plot(ax, labels, {"evolved": (est, lo, hi)}, x_label="BB/100")
    nbp.savefig("nb03_example")                    # -> artifacts/figures/nb03_example.png (and still shown inline)

Helper API: :func:`apply_style`, :func:`savefig`, :func:`bar_with_ci`, :func:`forest_plot`,
:func:`title`, :func:`footnote`, plus :func:`use_plotly` / :func:`style_plotly` for the optional
interactive charts. Colour constants: :data:`SERIES` (fixed order), :data:`ACCENT`, :data:`SURFACE`,
:data:`TEXT`, :data:`MUTED`, :data:`GRID`, colour maps :data:`SEQUENTIAL` and :data:`DIVERGING`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from poker_arena.config import FIGURES_DIR

# --------------------------------------------------------------------------------------
# Palette (validated for the dark surface; categorical hues are used in this fixed order)
# --------------------------------------------------------------------------------------
SURFACE = "#10201a"  # figure and axes background: the app's dark "felt"
TEXT = "#e9f1ec"
MUTED = "#9bb0a4"
GRID = "#263b31"
SERIES: tuple[str, ...] = ("#199e70", "#3987e5", "#d95926", "#9085e9", "#c98500", "#d55181")  # green, blue, orange, violet, yellow, magenta
ACCENT = SERIES[0]
WARM, COOL = "#e66767", "#3987e5"  # the two poles of the diverging map
NEUTRAL = "#2b3a33"  # diverging midpoint (a grey, never a hue)

SEQUENTIAL = LinearSegmentedColormap.from_list("felt_sequential", ["#17352a", ACCENT, "#9be8c6"])
DIVERGING = LinearSegmentedColormap.from_list("felt_diverging", [WARM, NEUTRAL, COOL])

INLINE_DPI = 100  # figures embedded in notebooks: moderate, keeps .ipynb files small
SAVE_DPI = 140  # PNG files in artifacts/figures/
DEFAULT_FIGSIZE = (9.6, 4.4)
BAR_THICKNESS = 0.55
MARKER_SIZE = 8.0
_ROW_SPREAD = 0.72  # vertical room shared by the series of one forest-plot row


def apply_style() -> None:
    """Set the dark felt style (colours, thin grid, left-aligned titles) for every following figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": INLINE_DPI,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.titlecolor": TEXT,
            "axes.titlelocation": "left",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.prop_cycle": cycler(color=list(SERIES)),
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.labelcolor": MUTED,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 10,
        }
    )


def savefig(name: str, fig: Figure | None = None, *, directory: Path | None = None, dpi: int = SAVE_DPI) -> Path:
    """Write the figure to ``artifacts/figures/<name>.png`` and return the path (the figure stays open,
    so the notebook still shows it inline)."""
    if Path(name).name != name or name.endswith(".png"):
        raise ValueError(f"name must be a bare file stem such as 'nb03_fitness', got {name!r}")
    target_dir = Path(directory) if directory is not None else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    figure = fig if fig is not None else plt.gcf()
    path = target_dir / f"{name}.png"
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor(), bbox_inches="tight")
    return path


def title(ax: Axes, text: str, subtitle: str | None = None) -> None:
    """Left-aligned bold title with an optional muted one-line subtitle (the 'so what' of the panel)."""
    ax.set_title(text, loc="left", pad=20 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom", ha="left")


def footnote(fig: Figure, text: str) -> None:
    """Small muted source / caveat line under the figure."""
    fig.text(0.01, -0.01, text, color=MUTED, fontsize=8, ha="left", va="top")


def _excludes(low: float, high: float, reference: float) -> bool:
    return bool(low > reference or high < reference)


def bar_with_ci(
    ax: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    ci_low: Sequence[float],
    ci_high: Sequence[float],
    *,
    colors: Sequence[str] | str | None = None,
    horizontal: bool = True,
    reference: float = 0.0,
    value_format: str = "{:+.0f}",
    annotate: bool = True,
) -> Axes:
    """Bars from ``reference`` with confidence-interval whiskers.

    A *solid* bar means the interval excludes ``reference``; an interval that contains it is drawn as a
    translucent outlined bar, so "not distinguishable from zero" is visible without relying on colour.
    """
    values_a, low_a, high_a = (np.asarray(x, dtype=float) for x in (values, ci_low, ci_high))
    positions = np.arange(len(labels))
    palette = [colors] * len(labels) if isinstance(colors, str) else list(colors) if colors else [ACCENT] * len(labels)
    for i, (value, low, high) in enumerate(zip(values_a, low_a, high_a, strict=True)):
        solid = _excludes(low, high, reference)
        style = {"color": palette[i], "alpha": 1.0} if solid else {"facecolor": palette[i], "alpha": 0.28, "edgecolor": palette[i], "linewidth": 1.2}
        if horizontal:
            ax.barh(positions[i], value - reference, left=reference, height=BAR_THICKNESS, **style)
            ax.plot([low, high], [positions[i]] * 2, color=TEXT, linewidth=1.4, solid_capstyle="butt")
        else:
            ax.bar(positions[i], value - reference, bottom=reference, width=BAR_THICKNESS, **style)
            ax.plot([positions[i]] * 2, [low, high], color=TEXT, linewidth=1.4, solid_capstyle="butt")
    if horizontal:
        ax.axvline(reference, color=MUTED, linewidth=1.0)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    else:
        ax.axhline(reference, color=MUTED, linewidth=1.0)
        ax.set_xticks(positions, labels)
        ax.grid(axis="x", visible=False)
    if annotate:  # the value sits beyond the upper whisker end (never over the axis labels)
        for i, (value, high) in enumerate(zip(values_a, high_a, strict=True)):
            end = max(high, value, reference)
            if horizontal:
                ax.annotate(value_format.format(value), (end, positions[i]), xytext=(4, 0), textcoords="offset points",
                            ha="left", va="center", color=MUTED, fontsize=8)  # fmt: skip
            else:
                ax.annotate(value_format.format(value), (positions[i], end), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", color=MUTED, fontsize=8)  # fmt: skip
    return ax


def forest_plot(
    ax: Axes,
    labels: Sequence[str],
    series: Mapping[str, tuple[Sequence[float], Sequence[float], Sequence[float]]],
    *,
    reference: float = 0.0,
    colors: Mapping[str, str] | None = None,
    x_label: str | None = None,
    legend: bool = True,
) -> Axes:
    """Point estimates with confidence intervals: one row per label, one marker per series in the row.

    ``series[name] = (estimates, ci_low, ci_high)``, each aligned with ``labels``. A filled marker means the
    interval excludes ``reference`` (usually 0); a hollow marker means it contains it.
    """
    names = list(series)
    n_rows = len(labels)
    offsets = np.zeros(len(names)) if len(names) == 1 else np.linspace(-_ROW_SPREAD / 2, _ROW_SPREAD / 2, len(names))
    ax.axvline(reference, color=MUTED, linewidth=1.0)
    for k, name in enumerate(names):
        estimate, low, high = (np.asarray(x, dtype=float) for x in series[name])
        color = (colors or {}).get(name, SERIES[k % len(SERIES)])
        y = np.arange(n_rows) + offsets[k]
        ax.hlines(y, low, high, color=color, linewidth=1.8, zorder=2)
        for yi, e, lo, hi in zip(y, estimate, low, high, strict=True):
            filled = _excludes(lo, hi, reference)
            ax.plot(e, yi, "o", markersize=MARKER_SIZE, markerfacecolor=color if filled else SURFACE,
                    markeredgecolor=color, markeredgewidth=1.8, zorder=3)  # fmt: skip
    ax.set_yticks(np.arange(n_rows), labels)
    ax.set_ylim(n_rows - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    if x_label:
        ax.set_xlabel(x_label)
    if legend:
        handles = [Line2D([0], [0], color=(colors or {}).get(n, SERIES[k % len(SERIES)]), marker="o", markersize=MARKER_SIZE - 1,
                          markerfacecolor=(colors or {}).get(n, SERIES[k % len(SERIES)]), linewidth=1.8, label=n)
                   for k, n in enumerate(names)]  # fmt: skip
        handles.append(Line2D([0], [0], color=MUTED, marker="o", markersize=MARKER_SIZE - 1, markerfacecolor=SURFACE,
                              linewidth=1.8, label="hollow: 95% CI contains 0"))  # fmt: skip
        ax.legend(handles=handles, loc="best", ncols=1 if len(handles) <= 3 else 2)
    return ax


# --------------------------------------------------------------------------------------
# Optional interactive charts (Plotly) - every key message must also exist as a static PNG
# --------------------------------------------------------------------------------------
def use_plotly() -> None:
    """Render Plotly figures inline in a notebook (``notebook`` renderer: no network needed, but the
    Plotly bundle is embedded once in the .ipynb, ~3.5 MB, and GitHub does not run it)."""
    import plotly.io as pio

    pio.renderers.default = "notebook"


def style_plotly(fig: Any, height: int = 380) -> Any:
    """Apply the felt palette to a Plotly figure and return it."""
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=list(SERIES),
        font={"color": TEXT},
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED)
    return fig
