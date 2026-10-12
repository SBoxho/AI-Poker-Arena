"""The UI-independent game controller."""

from __future__ import annotations

import pytest
from ui_helpers import AI, HUMAN, act_check_call, agent_match, human_match, human_obs, settle

from poker_arena.agents.registry import create_agent
from poker_arena.config import GameConfig
from poker_arena.engine.actions import Action
from poker_arena.ui.match import (
    AUTOPLAY_BASE_SECONDS,
    HAND_END_DWELL,
    MAX_SPEED,
    MIN_HANDS_FOR_RATE,
    MIN_SPEED,
    Match,
    MatchSettings,
    Pacer,
    TickKind,
    tick_interval,
)


def test_advance_alternates_deal_and_actions():
    match = Match(MatchSettings(GameConfig(), 1), [create_agent("random"), create_agent("maniac")])
    assert match.advance() is TickKind.DEAL and match.hand_active and match.decision_index == 0
    kinds = []
    while match.hand_active:
        kinds.append(match.advance())
    assert set(kinds) == {TickKind.ACTION} and match.hand_complete and match.hands_completed == 1
    assert match.advance() is TickKind.DEAL and match.hand_number == 1


def test_one_tick_is_exactly_one_decision():
    match = agent_match(2)
    before = match.decision_index
    match.advance()
    assert match.decision_index == before + 1


def test_stacks_carry_over_and_profit_is_zero_sum():
    match = agent_match(5, ("random", "maniac"))
    for _ in range(MIN_HANDS_FOR_RATE):
        match.play_out()
        assert sum(match.stacks) == sum(match.session.engine.start_stacks)  # chips are conserved
        match.start_hand()
    assert match.profit_chips[0] == -match.profit_chips[1]
    assert match.profit_series_bb[-1] == (match.profit_bb(0), match.profit_bb(1))
    assert len(match.profit_series_bb) == match.hands_completed == MIN_HANDS_FOR_RATE
    assert match.bb_per_100(0) is not None


def test_rate_is_hidden_until_enough_hands():
    match = agent_match(1)
    match.play_out()
    assert match.bb_per_100(0) is None


def test_button_alternates_every_hand():
    match = agent_match(1)
    buttons = []
    for _ in range(4):
        buttons.append(match.session.engine.button_seat)
        match.play_out()
        match.start_hand()
    assert buttons[0] != buttons[1] and buttons[0] == buttons[2] and buttons[1] == buttons[3]


def test_reset_stacks_each_hand():
    match = agent_match(3, ("maniac", "maniac"), reset_stacks_each_hand=True)
    for _ in range(10):
        match.play_out()
        match.start_hand()
        assert match.session.engine.start_stacks == (1000, 1000)


def test_auto_rebuy_keeps_the_game_going():
    config = GameConfig(starting_stack=20, small_blind=5, big_blind=10)
    match = agent_match(4, ("maniac", "maniac"), config=config, auto_rebuy=True)
    for _ in range(60):
        match.play_out()
        match.start_hand()
        assert all(s > 0 for s in match.session.engine.start_stacks)
    assert match.profit_chips[0] == -match.profit_chips[1]


def test_manual_rebuy_blocks_until_confirmed():
    config = GameConfig(starting_stack=20, small_blind=5, big_blind=10)
    match = agent_match(4, ("maniac", "maniac"), config=config, auto_rebuy=False)
    for _ in range(200):
        match.play_out()
        if match.needs_rebuy:
            break
        match.start_hand()
    else:
        pytest.fail("nobody went bust")
    with pytest.raises(RuntimeError, match="rebuy"):
        match.start_hand()
    match.rebuy()
    assert not match.needs_rebuy and match.stacks == [20, 20]
    match.start_hand()
    assert match.session.engine.start_stacks == (20, 20)


def test_cannot_deal_while_a_hand_is_running():
    match = agent_match(1)
    with pytest.raises(RuntimeError, match="in progress"):
        match.start_hand()


def test_human_submit_is_guarded_against_stale_clicks():
    match = human_match(1, "call_station")
    assert match.awaiting_human
    index = match.decision_index
    assert match.submit(Action.check_call(), decision_index=index + 5) is None  # wrong decision
    assert match.decision_index == index
    assert match.submit(Action.check_call(), decision_index=index) is not None
    assert match.submit(Action.check_call(), decision_index=index) is None  # double click: already applied
    settle(match)


def test_submit_is_rejected_when_it_is_the_ais_turn_or_hand_over():
    match = human_match(1, "call_station")
    while match.hand_active:
        act_check_call(match)
    assert match.submit(Action.check_call(), decision_index=match.decision_index) is None
    with pytest.raises(RuntimeError):
        match.step()  # nobody to act


def test_illegal_human_action_raises_without_changing_the_hand():
    from poker_arena.engine.actions import IllegalActionError

    match = human_match(1, "call_station")
    index = match.decision_index
    with pytest.raises(IllegalActionError):
        match.submit(Action.raise_to(1), decision_index=index)
    assert match.decision_index == index and match.awaiting_human


def test_seat_roles_and_queries():
    match = human_match(2)
    assert match.human_seat == HUMAN and match.agents[AI] is not None
    assert match.awaiting_human and not match.awaiting_agent and match.actor_seat == HUMAN
    assert human_obs(match).legal.can_check or human_obs(match).legal.can_call
    assert agent_match(2).human_seat is None


def test_same_seed_same_game():
    def events(seed: int):
        match = agent_match(seed, ("random", "maniac"))
        for _ in range(8):
            match.play_out()
            match.start_hand()
        return [r.events for r in match.records], match.profit_chips

    assert events(11) == events(11)
    assert events(11) != events(12)


def test_explanations_are_collected_unless_disabled():
    on = agent_match(1, collect_explanations=True)
    on.step()
    assert on.steps[-1].decision.explanation is not None
    off = agent_match(1, collect_explanations=False)
    off.step()
    assert off.steps[-1].decision.explanation is None


def test_play_out_returns_the_finished_record():
    match = agent_match(6)
    record = match.play_out()
    assert record is match.records[-1] and match.hand_complete and match.hands_completed == 1
    second = match.play_out()  # a finished hand: deals and plays a fresh one
    assert second is not record and match.hands_completed == 2 and second.hand_number == 1


def test_hand_history_is_capped():
    from poker_arena.ui import match as module

    match = agent_match(1, ("call_station", "call_station"))
    old = module.HAND_HISTORY_LIMIT
    module.HAND_HISTORY_LIMIT = 3
    try:
        for _ in range(6):
            match.play_out()
            match.start_hand()
    finally:
        module.HAND_HISTORY_LIMIT = old
    assert len(match.records) == 3 and match.hands_completed == 6 and len(match.profit_series_bb) == 6


def test_pacing():
    assert tick_interval(1.0) == AUTOPLAY_BASE_SECONDS
    assert tick_interval(MAX_SPEED) < tick_interval(MIN_SPEED)
    assert tick_interval(0.0) == tick_interval(MIN_SPEED)  # never divides by zero
    pacer = Pacer()
    assert pacer.due(0.0, 1.5)  # the very first tick is immediately due
    pacer.mark(100.0)
    assert not pacer.due(100.2, 1.5) and pacer.due(101.5, 1.5)
    assert not pacer.due(101.5, 1.5, after_hand_end=True)  # the finished hand lingers
    assert pacer.due(100.0 + 1.5 * HAND_END_DWELL, 1.5, after_hand_end=True)
