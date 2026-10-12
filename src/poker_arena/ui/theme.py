"""Shared look & feel: page configuration, injected CSS and the Plotly template.

The base theme (dark, green accent) is declared in ``.streamlit/config.toml``; this module adds the
component styling Streamlit's theme options cannot express (cards, table, panels, banners).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from poker_arena.ui.cards import CARD_CSS
from poker_arena.ui.table import TABLE_CSS

APP_TITLE = "AI Poker Arena"
APP_ICON = "♠️"

# Palette shared with the CSS custom properties below (and with the Plotly charts).
PALETTE = {
    "bg": "#0b1210",
    "panel": "#141f1a",
    "panel_2": "#1a2a23",
    "border": "#2a3f34",
    "text": "#e9f1ec",
    "muted": "#9bb0a4",
    "accent": "#2ecc71",
    "gold": "#f5c542",
    "red": "#ff6b6b",
    "blue": "#5aa9ff",
}
SERIES_COLORS = (PALETTE["accent"], PALETTE["gold"], PALETTE["blue"], "#c792ea", PALETTE["red"])

APP_CSS = """
:root {
  --pa-bg: #0b1210; --pa-panel: #141f1a; --pa-panel-2: #1a2a23; --pa-border: #2a3f34;
  --pa-text: #e9f1ec; --pa-muted: #9bb0a4; --pa-accent: #2ecc71; --pa-gold: #f5c542;
  --pa-red: #ff6b6b; --pa-blue: #5aa9ff;
}
.block-container { padding-top: 2.4rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -.01em; }
[data-testid="stSidebar"] { border-right: 1px solid var(--pa-border); }
[data-testid="stMetric"] {
  background: var(--pa-panel); border: 1px solid var(--pa-border); border-radius: .8rem; padding: .6rem .9rem;
}
.stButton > button, .stDownloadButton > button { font-weight: 650; }
button[kind="primary"] { box-shadow: 0 .15rem .7rem rgba(46,204,113,.25); }

/* generic panels */
.pa-panel {
  background: var(--pa-panel); border: 1px solid var(--pa-border); border-radius: .9rem;
  padding: .9rem 1.1rem; margin-bottom: .8rem;
}
.pa-panel-title {
  font-size: .78rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase;
  color: var(--pa-muted); margin-bottom: .5rem;
}
.pa-muted { color: var(--pa-muted); }
.pa-small { font-size: .85rem; }
.pa-sr-only {
  position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}

/* status strip above the table */
.pa-status { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: 0 0 .7rem; }
.pa-pill {
  display: inline-flex; align-items: center; gap: .4rem; padding: .25rem .8rem; border-radius: 2rem;
  background: var(--pa-panel-2); border: 1px solid var(--pa-border); font-size: .85rem; font-weight: 600;
}
.pa-pill b { color: #fff; }
.pa-pill--turn { border-color: var(--pa-accent); color: #8dffbb; background: rgba(46,204,113,.12); }
.pa-pill--wait { border-color: var(--pa-gold); color: #ffe08a; background: rgba(245,197,66,.1); }
.pa-pill--end { border-color: #7a8f84; }
.pa-dot { width: .6rem; height: .6rem; border-radius: 50%; background: currentColor; animation: pa-blink 1.2s infinite; }
@keyframes pa-blink { 50% { opacity: .25; } }
@media (prefers-reduced-motion: reduce) { .pa-dot { animation: none; } }

/* hand log */
.pa-log { max-height: 22rem; overflow-y: auto; font-size: .9rem; }
.pa-log-row { padding: .28rem .2rem; border-bottom: 1px solid rgba(255,255,255,.05); }
.pa-log-street {
  margin-top: .3rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; font-size: .75rem;
  color: var(--pa-gold); border-bottom: 1px solid rgba(245,197,66,.3);
}
.pa-log-result { color: #ffe9a8; font-weight: 650; }
.pa-log-info { color: var(--pa-muted); font-size: .82rem; }

/* explanations and feature grid */
.pa-explain {
  border-left: .25rem solid var(--pa-accent); background: rgba(46,204,113,.08);
  padding: .55rem .9rem; border-radius: 0 .6rem .6rem 0; margin: .3rem 0 .6rem; font-size: .95rem;
}
.pa-features { display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); gap: .5rem .9rem; }
.pa-feature { display: flex; justify-content: space-between; gap: .8rem; padding: .3rem .1rem;
  border-bottom: 1px solid rgba(255,255,255,.06); font-size: .88rem; }
.pa-feature span:first-child { color: var(--pa-muted); }
.pa-feature span:last-child { font-weight: 650; text-align: right; }
.pa-feature-group { grid-column: 1 / -1; margin-top: .5rem; font-size: .72rem; font-weight: 800;
  letter-spacing: .12em; text-transform: uppercase; color: var(--pa-gold); }

/* outcome banner (Human vs AI) */
.pa-banner {
  border-radius: 1rem; padding: .9rem 1.2rem; margin: .4rem 0 .8rem; border: 1px solid var(--pa-border);
  display: flex; flex-wrap: wrap; align-items: baseline; gap: .4rem 1.2rem;
}
.pa-banner b { font-size: 1.3rem; }
.pa-banner--win { background: linear-gradient(90deg, rgba(46,204,113,.25), rgba(46,204,113,.06));
  border-color: var(--pa-accent); }
.pa-banner--loss { background: linear-gradient(90deg, rgba(255,107,107,.22), rgba(255,107,107,.05));
  border-color: var(--pa-red); }
.pa-banner--split { background: rgba(90,169,255,.12); border-color: var(--pa-blue); }

/* championship: verdict grid */
.pa-vwrap { overflow-x: auto; }
.pa-vtable { border-collapse: separate; border-spacing: 4px; width: 100%; font-size: .82rem; }
.pa-vtable th { font-weight: 700; padding: .35rem .6rem; text-align: left; color: var(--pa-muted); white-space: nowrap; }
.pa-vtable thead th { text-align: center; }
.pa-v { padding: .45rem .5rem; border-radius: .5rem; text-align: center; min-width: 5.5rem; font-size: .78rem; border: 1px solid transparent; }
.pa-v-win { background: rgba(46,204,113,.18); border-color: rgba(46,204,113,.55); color: #b8ffd3; }
.pa-v-lose { background: rgba(255,107,107,.16); border-color: rgba(255,107,107,.5); color: #ffc2c2; }
.pa-v-draw { background: rgba(155,176,164,.12); border-color: rgba(155,176,164,.35); color: var(--pa-muted); }
.pa-v-low { background: rgba(245,197,66,.12); border-color: rgba(245,197,66,.4); color: #ffe08a; }
.pa-v-self { background: transparent; color: var(--pa-muted); }

/* home page */
.pa-hero {
  border-radius: 1.2rem; padding: 2.2rem 2rem; margin-bottom: 1.2rem; border: 1px solid var(--pa-border);
  background: radial-gradient(ellipse at 15% 20%, rgba(46,204,113,.28), transparent 55%),
              radial-gradient(ellipse at 90% 90%, rgba(245,197,66,.14), transparent 50%), var(--pa-panel);
}
.pa-hero h1 { margin: 0 0 .4rem; font-size: 2.6rem; }
.pa-hero p { margin: 0; max-width: 46rem; font-size: 1.08rem; color: #c9d8cf; }
.pa-modes { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 1rem; margin-bottom: .4rem; }
.pa-mode {
  background: var(--pa-panel); border: 1px solid var(--pa-border); border-radius: 1rem; padding: 1.1rem 1.2rem;
  min-height: 12.5rem;
}
.pa-mode h3 { margin: .1rem 0 .4rem; font-size: 1.15rem; }
.pa-mode .pa-mode-icon { font-size: 1.8rem; }
.pa-mode p { margin: 0; color: #c9d8cf; font-size: .93rem; }
.pa-roster { display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); gap: .8rem; }
.pa-agent { background: var(--pa-panel); border: 1px solid var(--pa-border); border-radius: .9rem; padding: .8rem 1rem; }
.pa-agent h4 { margin: 0; font-size: 1rem; display: flex; justify-content: space-between; gap: .5rem; align-items: center; }
.pa-agent p { margin: .35rem 0 0; font-size: .85rem; color: var(--pa-muted); overflow-wrap: anywhere; }
.pa-agent--off { opacity: .55; border-style: dashed; }
.pa-status-badge { font-size: .68rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase;
  padding: .12rem .55rem; border-radius: 1rem; white-space: nowrap; }
.pa-status-badge--on { background: rgba(46,204,113,.2); color: #8dffbb; }
.pa-status-badge--off { background: rgba(255,255,255,.1); color: #c8d6ce; }
.pa-principles { counter-reset: p; list-style: none; padding: 0 !important; margin: 0 !important; display: grid;
  grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr)); gap: .6rem 1.2rem; }
.pa-principles li { counter-increment: p; padding-left: 2rem; position: relative; font-size: .93rem; color: #d3e0d8; }
.pa-principles li::before { content: counter(p); position: absolute; left: 0; top: .05rem; width: 1.4rem; height: 1.4rem;
  border-radius: 50%; background: rgba(245,197,66,.18); color: var(--pa-gold); font-weight: 800; font-size: .8rem;
  display: grid; place-items: center; }
"""


def build_css() -> str:
    return "\n".join((CARD_CSS, TABLE_CSS, APP_CSS))


def inject_theme() -> None:
    """Add the shared CSS to the current page (style-only HTML takes no space in the layout)."""
    st.html(f"<style>{build_css()}</style>")


def configure_page(title: str, icon: str = APP_ICON) -> None:
    """Call first in every page: sets the browser tab, wide layout and injects the theme."""
    st.set_page_config(page_title=f"{title} - {APP_TITLE}", page_icon=icon, layout="wide")
    inject_theme()


def style_figure(fig: Any, *, height: int = 300) -> Any:
    """Apply the dark transparent Plotly look used across the app."""
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 36, "b": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"], "family": "Segoe UI, system-ui, sans-serif", "size": 13},
        colorway=list(SERIES_COLORS),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        hoverlabel={"bgcolor": PALETTE["panel_2"], "font_color": PALETTE["text"]},
    )
    grid = {"gridcolor": "rgba(255,255,255,.08)", "zerolinecolor": "rgba(255,255,255,.25)", "automargin": True}
    fig.update_xaxes(**grid)
    fig.update_yaxes(**grid)
    return fig
