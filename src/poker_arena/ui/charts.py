"""Small Plotly charts shared by the game pages."""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from poker_arena.ui.theme import PALETTE, SERIES_COLORS, style_figure


def profit_figure(series_bb: Sequence[tuple[float, float]], labels: Sequence[str], height: int = 240) -> go.Figure:
    """Cumulative profit (in big blinds) of both seats after every hand."""
    hands = list(range(1, len(series_bb) + 1))
    fig = go.Figure()
    for seat, label in enumerate(labels):
        fig.add_trace(
            go.Scatter(
                x=hands,
                y=[point[seat] for point in series_bb],
                name=label,
                mode="lines+markers" if len(hands) < 40 else "lines",
                line={"width": 2.5, "color": SERIES_COLORS[seat]},
                hovertemplate=f"{label}<br>after hand %{{x}}: %{{y:+.1f}} BB<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color=PALETTE["muted"], line_width=1, opacity=0.6)
    fig.update_xaxes(dtick=1 if len(hands) < 12 else None, title_text=None)
    fig.update_yaxes(ticksuffix=" BB")
    return style_figure(fig, height=height)
