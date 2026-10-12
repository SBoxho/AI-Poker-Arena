"""Legal bet-sizing helpers for the human's controls."""

from __future__ import annotations

from core_helpers import make_deal

from poker_arena.config import GameConfig
from poker_arena.engine.actions import Action
from poker_arena.engine.session import HandSession
from poker_arena.ui.betting import call_label, default_raise, quick_raises, raise_label


def _session(stacks=(1000, 1000), button=0) -> HandSession:
    deal = make_deal("AsKd", "2c3c", "QhJh9d5s4c")
    return HandSession(GameConfig(), deal, button, [None, None], start_stacks=stacks)


def test_labels_for_check_call_and_raise():
    session = _session()  # seat 0 is the button/SB: facing the big blind
    obs = session.observation(0)
    assert call_label(obs) == "Call 5"
    assert raise_label(obs, 30) == "Raise to 30"
    session.submit(Action.check_call())  # SB completes -> BB may check
    assert call_label(session.observation(1)) == "Check"
    assert raise_label(session.observation(1), 30) == "Raise to 30"


def test_bet_label_when_nothing_to_call_and_all_in_suffix():
    session = _session()
    session.submit(Action.check_call())
    session.submit(Action.check_call())  # flop, seat 1 (BB) acts first
    obs = session.observation(1)
    assert obs.opp_bet == 0
    assert raise_label(obs, 40) == "Bet 40"
    assert raise_label(obs, obs.legal.max_raise_to).endswith("(all-in)")


def test_quick_raises_are_legal_and_ordered():
    obs = _session().observation(0)
    legal = obs.legal
    quick = quick_raises(obs)
    assert [q.key for q in quick] == ["min", "half", "pot", "allin"]
    assert quick[0].amount == legal.min_raise_to and quick[-1].amount == legal.max_raise_to
    assert all(legal.min_raise_to <= q.amount <= legal.max_raise_to for q in quick)
    assert [q.amount for q in quick] == sorted(q.amount for q in quick)
    assert legal.min_raise_to <= default_raise(obs) <= legal.max_raise_to


def test_duplicate_presets_are_disabled_for_short_stacks():
    obs = _session(stacks=(25, 1000)).observation(0)  # raise range is only 20..25
    quick = quick_raises(obs)
    enabled = [q.amount for q in quick if q.enabled]
    assert len(enabled) == len(set(enabled)) and quick[0].enabled
    assert any(not q.enabled for q in quick)  # clamped presets that repeat an earlier amount


def test_all_presets_disabled_when_raising_is_impossible():
    session = _session(stacks=(1000, 30))  # seat 1 (BB) has 30 chips
    session.submit(Action.raise_to(1000))  # seat 0 shoves; seat 1 can only call/fold
    obs = session.observation(1)
    assert not obs.legal.can_raise
    assert all(not q.enabled for q in quick_raises(obs))
    assert call_label(obs).endswith("(all-in)")
