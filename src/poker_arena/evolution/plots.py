"""Static figures of a GA run: fitness curves, gene trajectories, held-out results (matplotlib PNG, light surface).

Colours follow the project's data-viz palette (categorical slots 1-3, recessive grid, thin lines).
Both figures are generated from the run's own history, never from hand-typed numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

from poker_arena.agents.ga_agent import GENE_HIGHS, GENE_LOWS, GENE_NAMES  # noqa: E402
from poker_arena.training.ga import EvolutionResult  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
AGENT_COLORS = (BLUE, ORANGE, AQUA, YELLOW)  # categorical slots 1-4, in fixed order: evolved first
EXPLOITABLE_BOTS = ("random", "call_station", "maniac")  # shown on their own scale: they lose thousands of BB/100
LINE_WIDTH = 2.0
BAND_ALPHA = 0.16
FITNESS_CURVE_FILE = "ga_fitness_curve.png"
GENE_TRAJECTORY_FILE = "ga_gene_trajectories.png"
FINAL_EVALUATION_FILE = "ga_final_evaluation.png"
GENE_PANEL_COLUMNS = 3


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_fitness_curve(
    result: EvolutionResult,
    reference_validation: Mapping[str, float],
    population_validation: Mapping[str, Sequence[float]],
    path: Path,
) -> None:
    """Left: training fitness (best, population mean +- std) and the held-out validation score of each
    generation's best genome. Right: held-out score of EVERY genome of the first vs the last generation.
    The dotted line is the best hand-made preset the population started from (context)."""
    history = result.history
    generations = np.array([r.generation + 1 for r in history])
    best = np.array([r.best_fitness for r in history])
    mean = np.array([r.mean_fitness for r in history])
    std = np.array([r.std_fitness for r in history])
    validated = [(r.generation + 1, r.validation["fitness"]) for r in history if r.validation is not None]

    fig, (ax, ax_pop) = plt.subplots(
        1, 2, figsize=(11.5, 4.8), facecolor=SURFACE, sharey=True, gridspec_kw={"width_ratios": [3.0, 1.2]}
    )
    _style_axes(ax)
    _style_axes(ax_pop)
    ax.fill_between(generations, mean - std, mean + std, color=ORANGE, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(generations, mean, color=ORANGE, linewidth=LINE_WIDTH, label="population mean (band: +-1 std)")
    ax.plot(generations, best, color=BLUE, linewidth=LINE_WIDTH, label="best genome (training fitness)")
    if validated:
        xs, ys = zip(*validated, strict=True)
        ax.plot(xs, ys, color=AQUA, linewidth=LINE_WIDTH, linestyle="--", label="best genome on held-out validation")
    if reference_validation:
        name, value = max(reference_validation.items(), key=lambda item: item[1])
        for axis in (ax, ax_pop):
            axis.axhline(value, color=TEXT_SECONDARY, linewidth=1.2, linestyle=":")
        ax.plot([], [], color=TEXT_SECONDARY, linewidth=1.2, linestyle=":", label=f"best hand-made preset, validation ({name})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("generation", color=TEXT_SECONDARY)
    ax.set_ylabel("saturated BB/100 (higher is better)", color=TEXT_SECONDARY)
    ax.set_title("Fitness per generation", color=TEXT_PRIMARY, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8.5, loc="lower right")

    rng = np.random.default_rng(0)  # cosmetic jitter only
    for position, (label, values) in enumerate(population_validation.items()):
        values = np.asarray(values, dtype=float)
        ax_pop.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=22, color=ORANGE, alpha=0.7, linewidths=0)
        ax_pop.hlines(np.median(values), position - 0.3, position + 0.3, color=TEXT_PRIMARY, linewidth=2.0)
    ax_pop.set_xticks(range(len(population_validation)))
    ax_pop.set_xticklabels([f"{k} population\n(median {np.median(v):.0f})" for k, v in population_validation.items()],
                           color=TEXT_SECONDARY, fontsize=8.5)  # fmt: skip
    ax_pop.set_xlim(-0.6, max(len(population_validation) - 1, 0) + 0.6)
    ax_pop.set_title("All genomes, held-out", color=TEXT_PRIMARY, loc="left", fontsize=12)
    fig.text(
        0.01, 0.005,
        "Training fitness plays fresh deals against a growing hall of fame every generation (trend, not point-by-point comparable);"
        " validation always uses the same held-out deals.",
        color=TEXT_SECONDARY, fontsize=7.5, ha="left", va="bottom",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_gene_trajectories(result: EvolutionResult, path: Path) -> None:
    """One small panel per gene: population mean (+- std) and the generation's best value over time."""
    history = result.history
    generations = np.array([r.generation + 1 for r in history])
    mean = np.array([r.gene_mean for r in history])
    std = np.array([r.gene_std for r in history])
    best = np.array([r.best_vector for r in history])
    rows = int(np.ceil(len(GENE_NAMES) / GENE_PANEL_COLUMNS))

    fig, axes = plt.subplots(rows, GENE_PANEL_COLUMNS, figsize=(11, 1.9 * rows), facecolor=SURFACE, sharex=True)
    for index, ax in enumerate(axes.flat):
        if index >= len(GENE_NAMES):
            ax.set_visible(False)
            continue
        _style_axes(ax)
        ax.fill_between(generations, mean[:, index] - std[:, index], mean[:, index] + std[:, index],
                        color=BLUE, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(generations, mean[:, index], color=BLUE, linewidth=LINE_WIDTH, label="population mean (band: +-1 std)")
        ax.plot(generations, best[:, index], color=ORANGE, linewidth=1.2, label="generation's best genome")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(GENE_LOWS[index], GENE_HIGHS[index])
        ax.set_title(GENE_NAMES[index], color=TEXT_PRIMARY, loc="left", fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, labelcolor=TEXT_SECONDARY, fontsize=9)
    fig.supxlabel("generation", color=TEXT_SECONDARY, fontsize=9, y=0.06)
    fig.suptitle("Gene trajectories (y-axis spans each gene's allowed range)", color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_final_evaluation(evaluation: Mapping[str, Any], path: Path) -> None:
    """Held-out BB/100 with 95% CIs: evolved genome vs reference presets, per opponent (two scales)."""
    results = evaluation["results"]
    opponents = list(evaluation["opponents"])
    groups = [
        ("Exploitable bots", [o for o in opponents if o in EXPLOITABLE_BOTS]),
        ("Solid opponents", [o for o in opponents if o not in EXPLOITABLE_BOTS]),
    ]
    agents = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * max(len(g) for _, g in groups) + 2.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [len(groups[0][1]) + 2, len(groups[1][1]) + 2]})  # fmt: skip
    spread = 0.7 / len(agents)
    for ax, (title, names) in zip(axes, groups, strict=True):
        _style_axes(ax)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.0)
        for a_index, agent in enumerate(agents):
            stats = [results[agent][name] for name in names]
            y = np.arange(len(names)) + (a_index - (len(agents) - 1) / 2) * spread
            x = np.array([s["bb_per_100"] for s in stats])
            err = np.array([[s["bb_per_100"] - s["ci_low"] for s in stats], [s["ci_high"] - s["bb_per_100"] for s in stats]])
            ax.errorbar(x, y, xerr=err, fmt="o", color=AGENT_COLORS[a_index % len(AGENT_COLORS)],
                        markersize=8 if agent == "evolved" else 6, elinewidth=1.6, capsize=0,
                        label=agent)  # fmt: skip
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels([n + ("*" if evaluation["opponents"][n]["in_training_pool"] else "") for n in names],
                           color=TEXT_PRIMARY, fontsize=9)  # fmt: skip
        ax.invert_yaxis()
        ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=11)
        ax.set_xlabel("BB/100 vs this opponent (95% CI)", color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(agents), frameon=False, labelcolor=TEXT_SECONDARY, fontsize=9,
               bbox_to_anchor=(0.5, 0.035))  # fmt: skip
    fig.text(
        0.01, 0.005,
        f"{evaluation['hands_per_opponent']} hands per opponent on identical fresh deals; * = also in the training pool.",
        color=TEXT_SECONDARY, fontsize=7.5, ha="left", va="bottom",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def save_figures(
    result: EvolutionResult,
    evaluation: Mapping[str, Any],
    reference_validation: Mapping[str, float],
    population_validation: Mapping[str, Sequence[float]],
    figures_dir: Path,
) -> list[Path]:
    """Write all figures into ``figures_dir`` and return their paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [figures_dir / FITNESS_CURVE_FILE, figures_dir / GENE_TRAJECTORY_FILE, figures_dir / FINAL_EVALUATION_FILE]
    plot_fitness_curve(result, reference_validation, population_validation, paths[0])
    plot_gene_trajectories(result, paths[1])
    plot_final_evaluation(evaluation, paths[2])
    return paths
