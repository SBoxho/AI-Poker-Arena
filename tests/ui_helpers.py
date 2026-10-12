"""Helpers shared by the ``test_ui_*`` modules (plain functions, no fixtures)."""

from __future__ import annotations

from poker_arena.agents.registry import create_agent
from poker_arena.config import GameConfig
from poker_arena.engine.actions import Action
from poker_arena.engine.observation import Observation
from poker_arena.ui.match import Match, MatchSettings
from poker_arena.ui.views import SeatIdentity

HUMAN, AI = 0, 1
IDENTITIES = (
    SeatIdentity("You", "You", "Human"),
    SeatIdentity("Opponent (Maniac)", "Maniac", "Opponent · Rule-based"),
)


def human_match(seed: int = 1, ai_kind: str = "call_station", *, config: GameConfig | None = None, **kw) -> Match:
    """Human in seat 0 against a baseline AI, first hand dealt, AI actions already played."""
    settings = MatchSettings(config or GameConfig(), seed, auto_rebuy=False, **kw)
    match = Match(settings, [None, create_agent(ai_kind)])
    match.start_hand()
    settle(match)
    return match


def agent_match(seed: int = 1, kinds: tuple[str, str] = ("random", "maniac"), *, config: GameConfig | None = None, **kw) -> Match:
    settings = MatchSettings(config or GameConfig(), seed, **kw)
    match = Match(settings, [create_agent(k) for k in kinds])
    match.start_hand()
    return match


def settle(match: Match) -> None:
    """Let the AI act until the human is to act or the hand is over."""
    while match.awaiting_agent:
        match.step()


def human_obs(match: Match) -> Observation:
    return match.session.observation(HUMAN)  # type: ignore[union-attr]


def act_check_call(match: Match) -> None:
    match.submit(Action.check_call(), decision_index=match.decision_index)
    settle(match)


def play_by_calling(match: Match) -> None:
    """Human always checks/calls until the hand ends."""
    while match.hand_active:
        act_check_call(match)


def ai_hole(match: Match) -> tuple[str, str]:
    """The AI's hole cards - read from the engine's deal, i.e. exactly what must NOT leak."""
    engine = match.session.engine  # type: ignore[union-attr]
    return engine.deal.button_hole if engine.button_seat == AI else engine.deal.bb_hole


def play_folding(match: Match) -> None:
    """Human folds whenever folding is legal, otherwise checks."""
    while match.hand_active:
        if match.awaiting_human:
            legal = human_obs(match).legal
            match.submit(Action.fold() if legal.can_fold else Action.check_call(), decision_index=match.decision_index)
        else:
            match.step()


def fold_spot_match(ai_kind: str = "call_station") -> Match:
    """A match whose human can fold right away (i.e. faces a bet) - they then fold and the hand ends."""
    for seed in range(1, 60):
        match = human_match(seed, ai_kind)
        if match.awaiting_human and human_obs(match).legal.can_fold:
            return match
    raise AssertionError("no seed gives the human a fold spot")
