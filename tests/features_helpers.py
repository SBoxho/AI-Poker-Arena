"""Helpers for the feature / teacher / dataset tests: scripted engine states and observations.

Plain functions only (like ``core_helpers``). Tests may use ``HandEngine`` directly; the production code
under test (``features/``, ``agents/``) must not.
"""

from __future__ import annotations

from collections.abc import Iterable

from core_helpers import make_deal, new_engine, play_script

from poker_arena.engine.game import HandEngine
from poker_arena.engine.observation import Observation
from poker_arena.engine.opponent import OpponentStats


def scripted_engine(
    button_hole: str,
    bb_hole: str,
    board: str,
    script: Iterable[str | int] = (),
    button_seat: int = 0,
    stacks: tuple[int, int] | None = None,
) -> HandEngine:
    """Engine for an explicit deal, advanced by ``script`` (``"c"`` call/check, ``"f"`` fold, ints = raise to)."""
    engine = new_engine(make_deal(button_hole, bb_hole, board), button_seat, stacks)
    play_script(engine, script)
    return engine


def acting_observation(engine: HandEngine, stats: OpponentStats | None = None) -> Observation:
    """Observation of the player whose turn it is."""
    seat = engine.actor_seat
    assert seat is not None, "hand is over"
    return engine.observation_for(seat, stats)
