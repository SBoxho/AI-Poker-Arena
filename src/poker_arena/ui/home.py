"""Landing page: project pitch, the three modes, the agent roster and the research principles."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st

from poker_arena.ui.theme import configure_page
from poker_arena.ui.widgets import cached_roster, refresh_roster, roster_grid_html

PITCH = (
    "Three independently trained AI architectures - an Explainable Boosting Machine, XGBoost with SHAP "
    "and a Genetic Algorithm evolved through gameplay - play heads-up No-Limit Texas Hold'em against "
    "each other and against rule-based baselines. A research and education project about "
    "<b>explainable decision making under uncertainty</b>."
)


@dataclass(frozen=True, slots=True)
class ModeCard:
    icon: str
    title: str
    text: str
    page: str
    cta: str


MODES = (
    ModeCard(
        "🤖",
        "AI vs AI",
        "Watch two agents play hand by hand and inspect every decision with the observation behind it.",
        "pages/1_AI_vs_AI.py",
        "Watch a match",
    ),
    ModeCard(
        "🏆",
        "AI Championship",
        "Tournament results from saved experiments: BB/100 with confidence intervals and matchup matrix.",
        "pages/2_AI_Championship.py",
        "Open the dashboard",
    ),
    ModeCard(
        "🧑‍💻",
        "Human vs AI",
        "Sit at the table yourself. The AI sees only what a real opponent would and explains itself after each hand.",
        "pages/3_Human_vs_AI.py",
        "Play a hand",
    ),
)

PRINCIPLES = (
    "<b>Imitation accuracy is not poker skill.</b> A model that predicts a teacher's moves well can still lose chips.",
    "<b>Poker is noisy.</b> Results are reported as BB/100 with confidence intervals, using duplicate (matched) deals.",
    "<b>Different kinds of explainability:</b> EBM shape functions, XGBoost SHAP values, readable genetic rules.",
    "<b>Equity is not profitability.</b> Winning chances against a random hand ignore betting and position.",
    "<b>Legal is not good.</b> Every agent only picks legal actions; that says nothing about quality.",
    "<b>Same interface for everybody.</b> Agents see an <code>Observation</code> - never the opponent's cards.",
    "<b>Reproducible.</b> Seeds and configurations are saved with every model and tournament.",
    "<b>No unsupported claims.</b> Every number in the app comes from saved experiment artifacts.",
)


def _mode_card_html(card: ModeCard) -> str:
    return (
        f'<div class="pa-mode"><div class="pa-mode-icon">{card.icon}</div>'
        f"<h3>{escape(card.title)}</h3><p>{escape(card.text)}</p></div>"
    )


def render_home() -> None:
    configure_page("Home")
    st.html(f'<div class="pa-hero"><h1>♠ AI Poker Arena</h1><p>{PITCH}</p></div>')

    st.subheader("Choose a mode")
    for card, col in zip(MODES, st.columns(len(MODES), gap="medium")):
        with col:
            st.html(_mode_card_html(card))
            st.page_link(card.page, label=card.cta, icon="➡️", width="stretch")

    st.subheader("Agent roster")
    roster = cached_roster()
    ready = sum(e.available for e in roster.values())
    st.caption(
        f"{ready} of {len(roster)} agents are ready. Trained agents appear automatically once their "
        "model artifacts exist - unavailable ones are greyed out with the reason."
    )
    st.html(roster_grid_html(roster))
    st.button("Re-scan agents", on_click=refresh_roster, key="pa.home.rescan")

    st.subheader("Research principles")
    items = "".join(f"<li>{text}</li>" for text in PRINCIPLES)
    st.html(f'<ol class="pa-principles">{items}</ol>')
