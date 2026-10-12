"""Engine -> TableView builders, including the anti-cheating guarantees of Mode C."""

from __future__ import annotations

import re

import pytest
from core_helpers import make_deal
from ui_helpers import (
    HUMAN,
    IDENTITIES,
    act_check_call,
    agent_match,
    ai_hole,
    fold_spot_match,
    human_match,
    human_obs,
    play_by_calling,
)

from poker_arena.agents.registry import create_agent
from poker_arena.cards import ALL_CARDS
from poker_arena.config import GameConfig
from poker_arena.engine.actions import Action
from poker_arena.engine.game import HandResult
from poker_arena.engine.observation import Street
from poker_arena.engine.session import HandSession
from poker_arena.ui.cards import RANK_LABELS, SUIT_GLYPHS, card_name, render_card
from poker_arena.ui.table import render_table_html
from poker_arena.ui.views import (
    agent_identity,
    hand_name,
    player_table,
    result_headline,
    showdown_hand_names,
    spectator_table,
    street_label,
)


def _player_html(match, *, result=None) -> str:
    table = player_table(
        human_obs(match), IDENTITIES, viewer_seat=HUMAN, actor_seat=match.actor_seat, result=result
    )
    return render_table_html(table)


def _mentions(html: str, card: str) -> bool:
    """True if the card string appears as a token (not as part of a word such as "Ace")."""
    return re.search(rf"(?<![A-Za-z0-9]){card}(?![A-Za-z0-9])", html) is not None


def _spectator_ids(match):
    return tuple(agent_identity(s, a.name, a.architecture) for s, a in enumerate(match.agents))


# ----------------------------------------------------------------- hidden information (Mode C)
@pytest.mark.parametrize("seed", range(1, 25))
def test_active_hand_html_never_contains_the_ai_hole_cards(seed: int):
    match = human_match(seed)
    while match.hand_active:
        html = _player_html(match)
        for card in ai_hole(match):
            assert not _mentions(html, card)
            assert card_name(card) not in html
            assert render_card(card) not in html
        assert html.count("pa-card--back") == 2
        act_check_call(match)


def test_ai_rank_and_suit_glyphs_are_absent_when_not_otherwise_visible():
    """Preflop, with an AI hand sharing no rank/suit with the human's hand, none of its glyphs appear."""
    checked = 0
    for seed in range(1, 400):
        match = human_match(seed)
        if not match.awaiting_human:
            continue
        mine, theirs = human_obs(match).my_hole, ai_hole(match)
        if {c[0] for c in mine} & {c[0] for c in theirs} or {c[1] for c in mine} & {c[1] for c in theirs}:
            continue
        html = _player_html(match)
        for card in theirs:
            assert f"<b>{RANK_LABELS[card[0]]}</b>" not in html
            assert SUIT_GLYPHS[card[1]] not in html
        checked += 1
        if checked == 5:
            return
    pytest.fail("no suitable deal found")


def test_table_html_is_independent_of_the_hidden_cards():
    """Swap the AI's hole cards for different ones: the human's rendered table must not change."""
    base = make_deal("AsKd", "2c3c", "QhJh9d5s4c")  # seat 0 is the button -> the human holds AsKd
    other = make_deal("AsKd", "7h8h", "QhJh9d5s4c")  # same visible cards, different AI cards

    def render(deal) -> str:
        session = HandSession(GameConfig(), deal, 0, [None, create_agent("call_station")])
        obs = session.observation(HUMAN)
        return render_table_html(player_table(obs, IDENTITIES, viewer_seat=HUMAN, actor_seat=session.actor_seat))

    assert render(base) == render(other)
    assert not _mentions(render(base), "2c") and not _mentions(render(base), "7h")


def test_board_only_shows_revealed_cards():
    match = human_match(3)
    seen_board_sizes = set()
    while match.hand_active:
        obs = human_obs(match)
        html = _player_html(match)
        shown = {card for card in ALL_CARDS if f'data-card="{card}"' in html}
        assert shown == set(obs.my_hole) | set(obs.board)
        seen_board_sizes.add(len(obs.board))
        act_check_call(match)
    assert seen_board_sizes == {0, 3, 4, 5}


def test_showdown_reveals_the_ai_hand_and_names_hands():
    match = human_match(2)
    play_by_calling(match)
    result = match.records[-1].result
    assert result.showdown
    html = _player_html(match, result=result)
    for card in ai_hole(match):
        assert f'data-card="{card}"' in html
    assert "pa-card--back" not in html and "Showdown" in html
    assert set(showdown_hand_names(result)) == {0, 1}


def test_fold_keeps_the_ai_hand_private():
    match = fold_spot_match()
    match.submit(Action.fold(), decision_index=match.decision_index)
    result = match.records[-1].result
    assert not result.showdown and result.shown_hands == {}
    html = _player_html(match, result=result)
    for card in ai_hole(match):
        assert not _mentions(html, card)
    assert "Hand over" in html and "folded" in html.lower()


# ----------------------------------------------------------------------------------- spectator
def test_spectator_cards_follow_the_toggle_and_showdown():
    match = agent_match(4, ("call_station", "call_station"))
    ids = _spectator_ids(match)
    engine = match.session.engine
    kwargs = dict(config=match.config, hand_number=0)
    hidden = render_table_html(spectator_table(engine.spectator_view(), ids, reveal_hole_cards=False, **kwargs))
    assert hidden.count("pa-card--back") == 4
    assert not any(_mentions(hidden, c) for c in engine.deal.cards[:4])
    shown = render_table_html(spectator_table(engine.spectator_view(), ids, reveal_hole_cards=True, **kwargs))
    assert shown.count("pa-card--back") == 0
    assert all(f'data-card="{c}"' in shown for c in (*engine.deal.button_hole, *engine.deal.bb_hole))
    match.play_out()
    final = render_table_html(spectator_table(engine.spectator_view(), ids, reveal_hole_cards=False, **kwargs))
    assert final.count("pa-card--back") == 0  # showdown reveals both hands even with the toggle off
    assert "pa-seat--winner" in final or "split" in final.lower()


def test_spectator_marks_dealer_blinds_and_actor():
    match = agent_match(5)
    table = spectator_table(
        match.session.engine.spectator_view(),
        _spectator_ids(match),
        config=match.config,
        hand_number=0,
        reveal_hole_cards=True,
    )
    button = match.session.engine.button_seat
    assert [s.is_dealer for s in (table.bottom, table.top)] == [button == 0, button == 1]
    assert {table.top.blind, table.bottom.blind} == {"SB", "BB"}
    assert sum(s.is_acting for s in (table.top, table.bottom)) == 1
    assert table.street == "Preflop" and table.info.startswith("Hand 1")


# ------------------------------------------------------------------------------------ helpers
def test_hand_names():
    assert hand_name(("As", "Ad"), ("Ah", "Kc", "2d")) == "Three of a kind"
    assert hand_name(("As", "Ad"), ()) is None


def test_street_label():
    assert street_label(Street.FLOP, complete=False, showdown=False) == "Flop"
    assert street_label(Street.RIVER, complete=True, showdown=True) == "Showdown"
    assert street_label(Street.PREFLOP, complete=True, showdown=False) == "Hand over"


def test_result_headline_variants():
    mentions = [i.mention for i in IDENTITIES]
    match = human_match(2)
    play_by_calling(match)
    result = match.records[-1].result
    text = result_headline(result, mentions, showdown_hand_names(result), you=HUMAN)
    assert ("You win" in text) == (result.winner == HUMAN) and "beats" in text

    fold = HandResult((10, -10), (1010, 990), False, 0, 0, (), {}, Street.PREFLOP)
    assert result_headline(fold, mentions, {}, you=HUMAN) == "You win 10 chips - Opponent (Maniac) folded"
    assert "You folded" in result_headline(
        HandResult((-10, 10), (990, 1010), False, 1, 0, (), {}, Street.PREFLOP), mentions, {}, you=HUMAN
    )
    split = HandResult((0, 0), (1000, 1000), True, None, 0, (), {}, Street.RIVER)
    assert result_headline(split, mentions, {}) == "Split pot"
