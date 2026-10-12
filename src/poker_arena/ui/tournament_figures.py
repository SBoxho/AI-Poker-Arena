"""Plotly figures and small HTML tables of the AI Championship dashboard (pure: DataFrames in)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from poker_arena.ui.theme import PALETTE, SERIES_COLORS, style_figure
from poker_arena.ui.tournament_data import METRIC_BY_KEY, MatrixMetric, format_ci

PCT = 100.0
NEUTRAL = "#7f938a"
HEATMAP_SCALE = [[0.0, "#d9534f"], [0.5, "#17231d"], [1.0, "#2ecc71"]]
SEQUENTIAL_SCALE = [[0.0, "#17231d"], [1.0, "#2ecc71"]]
MIN_ROW_HEIGHT = 46
STYLE_AXES = (
    ("vpip", "VPIP"),
    ("pfr", "PFR"),
    ("three_bet", "3-bet"),
    ("aggression_freq_postflop", "Postflop aggression"),
    ("fold_freq", "Fold"),
    ("call_freq", "Check / call"),
    ("raise_freq", "Bet / raise"),
    ("showdown_rate", "Showdown rate"),
)
RADAR_AXES = ("vpip", "pfr", "three_bet", "aggression_freq_postflop", "fold_freq", "showdown_rate")
STREET_LABELS = {"preflop": "Preflop", "flop": "Flop", "turn": "Turn", "river": "River", "showdown": "Showdown"}
VERDICT_CLASSES = {"win": "pa-v-win", "lose": "pa-v-lose", "draw": "pa-v-draw", "insufficient": "pa-v-low"}


def agent_colors(agents: Sequence[str]) -> dict[str, str]:
    """A stable colour per agent (by position in the run's agent list)."""
    return {name: SERIES_COLORS[i % len(SERIES_COLORS)] for i, name in enumerate(agents)}


def _sign_color(significant: bool, value: float) -> str:
    if not significant:
        return NEUTRAL
    return PALETTE["accent"] if value > 0 else PALETTE["red"]


# ----------------------------------------------------------------------------- leaderboard
def forest_figure(board: pd.DataFrame, level: float) -> go.Figure:
    """Forest plot: BB/100 of every agent with its confidence interval (best on top)."""
    fig = go.Figure()
    groups = (
        ("Significantly positive (CI above 0)", lambda r: r.significant and r.bb_per_100 > 0, PALETTE["accent"]),
        ("Significantly negative (CI below 0)", lambda r: r.significant and r.bb_per_100 <= 0, PALETTE["red"]),
        ("Not significant (CI includes 0)", lambda r: not r.significant, NEUTRAL),
    )
    for label, test, color in groups:
        rows = board[[bool(test(r)) for r in board.itertuples()]]
        if rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["bb_per_100"],
                y=rows["agent"],
                mode="markers",
                name=label,
                marker={"size": 12, "color": color, "line": {"width": 1, "color": "#0b1210"}},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": rows["ci_high"] - rows["bb_per_100"],
                    "arrayminus": rows["bb_per_100"] - rows["ci_low"],
                    "color": color,
                    "thickness": 2.5,
                    "width": 6,
                },
                customdata=np.column_stack([rows["ci_low"], rows["ci_high"], rows["rank"]]),
                hovertemplate=(
                    "<b>%{y}</b> (rank %{customdata[2]})<br>%{x:+.1f} BB/100<br>"
                    f"{level * PCT:g}% CI [%{{customdata[0]:+.1f}}, %{{customdata[1]:+.1f}}]<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_color=PALETTE["muted"], line_width=1, line_dash="dot")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=list(board["agent"]))
    fig.update_xaxes(title_text=f"BB/100, mean with {level * PCT:g}% confidence interval", zeroline=False)
    return style_figure(fig, height=max(240, 90 + MIN_ROW_HEIGHT * len(board)))


# --------------------------------------------------------------------------------- matchup
def _heatmap_text(metric: MatrixMetric, values: np.ndarray, lo: np.ndarray | None, hi: np.ndarray | None, sig: np.ndarray | None) -> list[list[str]]:
    rows = []
    for i in range(values.shape[0]):
        row = []
        for j in range(values.shape[1]):
            v = values[i, j]
            if math.isnan(v):
                row.append("")
                continue
            text = metric.fmt.format(v)
            if lo is not None and hi is not None and not math.isnan(lo[i, j]):
                text += f"<br>±{(hi[i, j] - lo[i, j]) / 2:.0f}"
                text += " ★" if sig is not None and sig[i, j] else ""
            row.append(text)
        rows.append(row)
    return rows


def matchup_heatmap(
    values: pd.DataFrame,
    metric_key: str,
    *,
    ci_low: pd.DataFrame | None = None,
    ci_high: pd.DataFrame | None = None,
    significant: pd.DataFrame | None = None,
    verdicts: pd.DataFrame | None = None,
    level: float = 0.95,
) -> go.Figure:
    """Heatmap of row agent vs column agent. With CIs, cells show the half-width and ★ if significant."""
    metric = METRIC_BY_KEY[metric_key]
    z = values.to_numpy(dtype=float)
    finite = np.abs(z[np.isfinite(z)])
    limit = float(finite.max()) if finite.size else 1.0
    if metric.kind == "diverging":
        scale, zmin, zmax, zmid = HEATMAP_SCALE, -limit, limit, 0.0
    elif metric.kind == "rate":
        scale, zmin, zmax, zmid = HEATMAP_SCALE, 0.0, 1.0, 0.5
    else:
        scale, zmin, zmax, zmid = SEQUENTIAL_SCALE, 0.0, max(limit, 1e-9), None
    lo = ci_low.to_numpy(dtype=float) if ci_low is not None else None
    hi = ci_high.to_numpy(dtype=float) if ci_high is not None else None
    sig = significant.to_numpy(dtype=bool) if significant is not None else None
    text = _heatmap_text(metric, z, lo, hi, sig)
    hover = [
        [
            "" if lo is None or math.isnan(z[i, j]) else f"{level * PCT:g}% CI {format_ci(lo[i, j], hi[i, j], 1)}"
            for j in range(z.shape[1])
        ]
        for i in range(z.shape[0])
    ]
    verdict_text = [
        [verdicts.iloc[i, j] if verdicts is not None else "" for j in range(z.shape[1])] for i in range(z.shape[0])
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(values.columns), y=list(values.index), text=text, texttemplate="%{text}",
            colorscale=scale, zmin=zmin, zmax=zmax, zmid=zmid, xgap=3, ygap=3, hoverongaps=False,
            customdata=np.dstack([np.array(hover, dtype=object), np.array(verdict_text, dtype=object)]),
            hovertemplate=(
                "<b>%{y}</b> vs <b>%{x}</b><br>" + metric.label + ": %{z:.3f}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
            ),
            colorbar={"title": {"text": metric.label, "side": "right"}, "thickness": 12},
            textfont={"size": 13, "color": "#f2f7f4"},
        )
    )
    fig.update_xaxes(title_text="Opponent (column)", side="top")
    fig.update_yaxes(title_text="Agent (row)", autorange="reversed")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 70, "b": 10})
    return style_figure(fig, height=max(320, 140 + 64 * len(values)))


def verdict_table_html(codes: pd.DataFrame, texts: pd.DataFrame) -> str:
    """Colour-coded verdict grid: row agent against column agent, in words."""
    head = "".join(f"<th scope='col'>{escape(str(c))}</th>" for c in codes.columns)
    body = []
    for row in codes.index:
        cells = []
        for col in codes.columns:
            code = codes.loc[row, col]
            if row == col:
                cells.append("<td class='pa-v pa-v-self'>-</td>")
            else:
                cells.append(f"<td class='pa-v {VERDICT_CLASSES.get(code, '')}'>{escape(str(texts.loc[row, col]))}</td>")
        body.append(f"<tr><th scope='row'>{escape(str(row))}</th>{''.join(cells)}</tr>")
    return (
        "<div class='pa-vwrap'><table class='pa-vtable'><caption class='pa-sr-only'>Pairwise verdicts, row agent "
        f"against column agent</caption><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------------- curves
def cumulative_figure(curves: Mapping[str, pd.DataFrame], *, luck_band: bool, level: float, title_unit: str = "BB") -> go.Figure:
    """Cumulative profit curves; the luck band (what a zero-edge player would do) only for one curve."""
    fig = go.Figure()
    colors = agent_colors(list(curves))
    single = len(curves) == 1
    for label, frame in curves.items():
        color = colors[label]
        if single and luck_band and frame["luck_low"].notna().any():
            fig.add_trace(go.Scatter(x=frame["hand"], y=frame["luck_high"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=frame["hand"], y=frame["luck_low"], mode="lines", line={"width": 0}, fill="tonexty",
                    fillcolor="rgba(155,176,164,0.30)", name=f"Luck band ({level * PCT:g}%): zero-edge player",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=frame["hand"], y=frame["cum_bb"], mode="lines", name=label, line={"width": 2.6, "color": color},
                hovertemplate=f"{label}<br>after %{{x:,}} hands: %{{y:+,.0f}} {title_unit}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color=PALETTE["muted"], line_width=1, line_dash="dot")
    fig.update_xaxes(title_text="Hands played")
    fig.update_yaxes(title_text=f"Cumulative profit ({title_unit})")
    return style_figure(fig, height=420)


# ------------------------------------------------------------------------------------ style
def style_radar(style: pd.DataFrame) -> go.Figure:
    """Radar chart of the main style frequencies, one polygon per agent."""
    labels = dict(STYLE_AXES)
    theta = [labels[k] for k in RADAR_AXES]
    colors = agent_colors(list(style["agent"]))
    fig = go.Figure()
    for row in style.itertuples():
        values = [float(getattr(row, k)) for k in RADAR_AXES]
        fig.add_trace(
            go.Scatterpolar(
                r=[*values, values[0]], theta=[*theta, theta[0]], name=row.agent, fill="toself", opacity=0.55,
                line={"color": colors[row.agent], "width": 2}, hovertemplate="%{theta}: %{r:.0%}<extra>" + row.agent + "</extra>",
            )
        )
    style_figure(fig, height=470)
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"range": [0, 1], "tickformat": ".0%", "gridcolor": "rgba(255,255,255,.12)", "angle": 90},
            "angularaxis": {"gridcolor": "rgba(255,255,255,.12)"},
        },
        margin={"l": 140, "r": 110, "t": 60, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.1, "x": 0},
    )
    return fig


def style_bars(style: pd.DataFrame) -> go.Figure:
    """Grouped bars of all style frequencies (x = metric, colour = agent)."""
    colors = agent_colors(list(style["agent"]))
    fig = go.Figure()
    for row in style.itertuples():
        fig.add_trace(
            go.Bar(
                x=[label for _, label in STYLE_AXES], y=[float(getattr(row, key)) for key, _ in STYLE_AXES],
                name=row.agent, marker_color=colors[row.agent], hovertemplate="%{x}: %{y:.1%}<extra>" + row.agent + "</extra>",
            )
        )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(barmode="group")
    return style_figure(fig, height=400)


# ------------------------------------------------------------------- opponents and streets
def opponent_figure(rows: pd.DataFrame, agent: str, level: float) -> go.Figure:
    """BB/100 of one agent against each opponent, with the confidence interval."""
    ordered = rows.sort_values("bb_per_100", ascending=False)
    colors = [_sign_color(bool(s), v) for s, v in zip(ordered["significant"], ordered["bb_per_100"])]
    fig = go.Figure(
        go.Bar(
            x=ordered["opponent"], y=ordered["bb_per_100"], marker_color=colors,
            error_y={
                "type": "data", "symmetric": False, "array": ordered["ci_high"] - ordered["bb_per_100"],
                "arrayminus": ordered["bb_per_100"] - ordered["ci_low"], "color": "#e9f1ec", "thickness": 1.6,
            },
            customdata=ordered["verdict_text"],
            hovertemplate=f"{agent} vs %{{x}}<br>%{{y:+.1f}} BB/100<br>%{{customdata}}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=PALETTE["muted"], line_width=1)
    fig.update_yaxes(title_text=f"{agent}: BB/100 (±{level * PCT:g}% CI)")
    fig.update_xaxes(title_text="Opponent")
    return style_figure(fig, height=360)


def street_actions_figure(per_street: pd.DataFrame) -> go.Figure:
    """Fold / call / raise frequency per street, one panel per action, bars coloured by agent."""
    agents = list(dict.fromkeys(per_street["agent"]))
    colors = agent_colors(agents)
    streets = list(dict.fromkeys(per_street["street"]))
    panels = (("fold_freq", "Fold"), ("call_freq", "Check / call"), ("raise_freq", "Bet / raise"))
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=[label for _, label in panels])
    for col, (key, _) in enumerate(panels, start=1):
        for agent in agents:
            frame = per_street[per_street["agent"] == agent].set_index("street").reindex(streets)
            fig.add_trace(
                go.Bar(
                    x=[STREET_LABELS.get(s, s) for s in streets], y=frame[key], name=agent, marker_color=colors[agent],
                    showlegend=col == 1, legendgroup=agent, hovertemplate="%{x}: %{y:.1%}<extra>" + agent + "</extra>",
                ),
                row=1, col=col,
            )  # fmt: skip
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(barmode="group")
    style_figure(fig, height=400)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 80, "b": 16}, legend={"orientation": "h", "y": 1.22, "x": 0})
    return fig


def profit_by_end_figure(by_end: pd.DataFrame) -> go.Figure:
    """Where each agent's BB/100 comes from: by how the hands ended (the bars add up to its BB/100)."""
    agents = list(dict.fromkeys(by_end["agent"]))
    colors = agent_colors(agents)
    ends = list(dict.fromkeys(by_end["ended"]))
    fig = go.Figure()
    for agent in agents:
        frame = by_end[by_end["agent"] == agent].set_index("ended").reindex(ends)
        fig.add_trace(
            go.Bar(
                x=[STREET_LABELS.get(e, e) + (" fold" if e != "showdown" else "") for e in ends],
                y=frame["bb_per_100_contribution"], name=agent, marker_color=colors[agent],
                customdata=np.column_stack([frame["share_of_hands"], frame["bb_per_hand"]]),
                hovertemplate="%{x}: %{y:+.1f} BB/100<br>%{customdata[0]:.0%} of hands, %{customdata[1]:+.2f} BB per hand<extra>"
                + agent + "</extra>",
            )
        )
    fig.add_hline(y=0, line_color=PALETTE["muted"], line_width=1)
    fig.update_yaxes(title_text="Contribution to BB/100")
    fig.update_xaxes(title_text="How the hand ended")
    fig.update_layout(barmode="group")
    return style_figure(fig, height=400)


# ---------------------------------------------------------------------------------- latency
def latency_figure(latency: pd.DataFrame) -> go.Figure:
    """Mean / median / 95th percentile decision latency per agent (log scale: agents differ by 100x)."""
    fig = go.Figure()
    for column, label, color in (
        ("median_ms", "Median", SERIES_COLORS[0]), ("mean_ms", "Mean", SERIES_COLORS[1]), ("p95_ms", "95th percentile", SERIES_COLORS[2]),
    ):  # fmt: skip
        fig.add_trace(
            go.Bar(x=latency["agent"], y=latency[column], name=label, marker_color=color,
                   hovertemplate="%{x}: %{y:.3f} ms<extra>" + label + "</extra>")
        )
    fig.update_yaxes(type="log", title_text="Decision latency (ms, log scale)")
    fig.update_layout(barmode="group")
    return style_figure(fig, height=360)


# ------------------------------------------------------------------------------ models
def gene_figure(genes: Sequence[Mapping[str, object]]) -> go.Figure:
    """Where each evolved gene sits inside its allowed range (0 = lower bound, 1 = upper bound)."""
    usable = [g for g in genes if g.get("low") is not None and g.get("high") is not None and g["high"] != g["low"]]
    names = [str(g["gene"]) for g in usable]
    position = [(float(g["value"]) - float(g["low"])) / (float(g["high"]) - float(g["low"])) for g in usable]
    fig = go.Figure(
        go.Bar(
            x=position, y=names, orientation="h", marker_color=PALETTE["accent"],
            customdata=[[g["value"], g["low"], g["high"], g.get("meaning", "")] for g in usable],
            hovertemplate="<b>%{y}</b><br>value %{customdata[0]:.3f} in [%{customdata[1]}, %{customdata[2]}]<br>%{customdata[3]}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="Position inside the allowed range")
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, height=max(260, 60 + 26 * len(usable)))


def importance_figure(features: Sequence[tuple[str, float]], title: str = "Global feature importance") -> go.Figure:
    top = list(features)
    fig = go.Figure(
        go.Bar(x=[v for _, v in top], y=[n for n, _ in top], orientation="h", marker_color=PALETTE["blue"],
               hovertemplate="%{y}: %{x:.4f}<extra></extra>")
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title={"text": title, "font": {"size": 14}})
    return style_figure(fig, height=max(240, 80 + 26 * len(top)))
