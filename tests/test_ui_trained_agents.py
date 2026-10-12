"""Streamlit ``AppTest`` runs with the TRAINED agents (EBM, XGBoost, GA, teacher).

The baseline-only page tests cannot catch UI problems that only appear with real explanation
payloads (e.g. EBM interaction terms carry a *pair* of feature values, SHAP/EBM payloads include
probability charts, the GA returns rule traces). Skipped automatically when an agent is not trained.
"""

from __future__ import annotations

import pytest
from test_ui_apps import (
    A,
    A_PAGE,
    C,
    C_PAGE,
    MAX_CLICKS,
    ai_cards,
    assert_clean,
    human_buttons,
    mentions,
    open_page,
    page_html,
    start_mode_a,
    start_mode_c,
)

from poker_arena.agents.registry import check_availability
from poker_arena.ui.match import Match

TRAINED = ("ebm", "xgboost", "ga", "teacher")
ACTIONS_TO_STEP = 8


def _require(*kinds: str) -> None:
    for kind in kinds:
        status = check_availability(kind)
        if not status.available:
            pytest.skip(f"{kind} is not available: {status.detail}")


@pytest.mark.parametrize("kind", TRAINED)
def test_mode_a_trained_agent_steps_and_explains(kind: str):
    _require(kind)
    at = start_mode_a((kind, "call_station"), seed=7)
    inspected = 0
    for _ in range(ACTIONS_TO_STEP):
        at.button(key=A + "next_action").click().run()
        assert_clean(at)
        match: Match = at.session_state[A + "match"]
        if match.steps:  # the list restarts with every hand; open the panel on the newest decision
            at.selectbox(key=A + "inspect").set_value(len(match.steps) - 1).run()
            assert_clean(at)  # real explanation payloads (pair-valued EBM terms, SHAP, GA rules) must render
            inspected += 1
    assert inspected, "no decision was ever inspected"


def test_mode_a_every_trained_pair_plays_a_hand():
    for kind in TRAINED:
        _require(kind)
    pairs = [("ebm", "xgboost"), ("xgboost", "ga"), ("ga", "teacher"), ("teacher", "ebm")]
    for kinds in pairs:
        at = start_mode_a(kinds, seed=11)
        at.button(key=A + "next_hand").click().run()
        assert_clean(at)
        assert at.session_state[A + "match"].hands_completed >= 1


@pytest.mark.parametrize("kind", TRAINED)
def test_mode_c_trained_opponent_hides_cards_and_reviews_after_the_hand(kind: str):
    _require(kind)
    at = start_mode_c(kind, seed=3)
    match: Match = at.session_state[C + "match"]
    hidden = ai_cards(match)
    clicks = 0
    while match.hand_active:
        html = page_html(at)
        assert not any(mentions(html, c) for c in hidden), f"{kind}: AI hole cards leaked into the page"
        human_buttons(at)["call"].click().run()
        assert_clean(at)
        clicks += 1
        assert clicks < MAX_CLICKS
    assert at.session_state[C + "match"].hands_completed == 1
    # the post-hand "why did the AI do that?" review renders the opponent's real explanations
    assert_clean(open_page(C_PAGE))
