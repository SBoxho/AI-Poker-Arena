"""Agent discovery and per-session agent copies."""

from __future__ import annotations

import pytest

from poker_arena.agents.base import ModelArtifactError
from poker_arena.agents.registry import create_agent
from poker_arena.config import GameConfig
from poker_arena.engine.deals import Deal
from poker_arena.engine.session import HandSession
from poker_arena.ui.roster import discover_roster, spawn_agent
from poker_arena.ui.widgets import agent_card_html, cached_roster, roster_grid_html


def _factory(kind: str):
    if kind == "missing":
        raise ModelArtifactError("model file missing - run scripts/train_ebm.py")
    if kind == "broken":
        raise ImportError("xgboost is not installed")
    if kind == "silent":
        raise RuntimeError()
    return create_agent(kind)


def test_roster_marks_unavailable_agents_with_a_reason_instead_of_failing():
    roster = discover_roster(["random", "missing", "broken", "silent", "maniac"], factory=_factory)
    assert list(roster) == ["random", "missing", "broken", "silent", "maniac"]
    assert roster["random"].available and roster["random"].template is not None
    assert not roster["missing"].available and "train_ebm" in roster["missing"].detail
    assert "xgboost" in roster["broken"].detail
    assert roster["silent"].detail == "RuntimeError"
    assert roster["maniac"].architecture == "Rule-based" and roster["maniac"].supports_explanations


def test_default_roster_contains_the_core_baselines():
    roster = discover_roster()
    assert {"random", "call_station", "maniac"} <= {k for k, e in roster.items() if e.available}
    assert set(roster) >= {"teacher", "ebm", "xgboost", "ga"}  # trained kinds are listed whether or not they load


def test_unavailable_agent_cannot_be_spawned():
    roster = discover_roster(["missing"], factory=_factory)
    with pytest.raises(ValueError, match="not available"):
        spawn_agent(roster["missing"], 1)


def _obs():
    return HandSession(GameConfig(), Deal.from_seed(5), 0, [None, None]).observation(0)


def _first_actions(agent, n: int = 12) -> list[str]:
    obs = _obs()
    return [str(agent.act(obs).action) for _ in range(n)]


def test_spawned_agents_are_independent_copies():
    entry = discover_roster(["random"])["random"]
    a, b = spawn_agent(entry, 1), spawn_agent(entry, 1)
    assert a is not b and a is not entry.template
    _first_actions(a, 5)  # advance one copy only
    assert _first_actions(b) == _first_actions(spawn_agent(entry, 1))  # the other copy is untouched
    assert _first_actions(spawn_agent(entry, 1)) != _first_actions(spawn_agent(entry, 2))


def test_roster_cards_show_availability_and_escape_text():
    roster = discover_roster(["random", "missing"], factory=_factory)
    ok, off = agent_card_html(roster["random"]), agent_card_html(roster["missing"])
    assert "Available" in ok and "pa-agent--off" not in ok
    assert "Unavailable" in off and "pa-agent--off" in off and "train_ebm" in off
    grid = roster_grid_html(roster)
    assert grid.count('class="pa-agent') == 2
    hostile = discover_roster(["x"], factory=lambda k: (_ for _ in ()).throw(ValueError("<img src=x>")))
    assert "&lt;img" in agent_card_html(hostile["x"])


def test_cached_roster_is_shared_between_calls():
    assert cached_roster() is cached_roster()
