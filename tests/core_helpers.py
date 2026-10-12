"""Shared helpers for the *core* test-suite (engine, actions, agents, information boundary).

Plain functions only (no pytest fixtures / conftest) so that the core tests stay independent of the
other workstreams' test files.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Iterable, Iterator, Sequence

from poker_arena.agents.base import Agent, Decision
from poker_arena.cards import ALL_CARDS
from poker_arena.config import GameConfig, derive_seed
from poker_arena.engine.actions import (
    AbstractAction,
    Action,
    LegalActions,
    abstract_to_action,
)
from poker_arena.engine.deals import Deal
from poker_arena.engine.game import HandEngine
from poker_arena.engine.observation import Observation
from poker_arena.engine.history import HandRecord
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.engine.session import play_hand

DEFAULT_CFG = GameConfig()  # 1000 / 5 / 10
MAX_ACTIONS_PER_HAND = 200  # a heads-up hand can never legitimately need more


# --------------------------------------------------------------------------- deals
def make_deal(
    button_hole: str,
    bb_hole: str,
    board: str,
    burns: str | None = None,
) -> Deal:
    """Explicit :class:`Deal` from compact strings, e.g. ``make_deal("AhKh", "2c3c", "QsJsTd5c2d")``.

    Burn cards default to the first three cards of the deck that are not otherwise used.
    """
    from poker_arena.cards import parse_cards

    btn, bb, brd = parse_cards(button_hole), parse_cards(bb_hole), parse_cards(board)
    assert len(btn) == 2 and len(bb) == 2 and len(brd) == 5
    if burns is None:
        used = set(btn) | set(bb) | set(brd)
        brn = tuple(c for c in ALL_CARDS if c not in used)[:3]
    else:
        brn = parse_cards(burns)
    cards = (*btn, *bb, brn[0], *brd[:3], brn[1], brd[3], brn[2], brd[4])
    return Deal(cards)


def random_deal(rng: random.Random) -> Deal:
    return Deal.from_rng(rng)


# ------------------------------------------------------------------------- stacks
def random_stacks(rng: random.Random) -> tuple[int, int]:
    """Asymmetric / tiny / equal stacks - the nasty cases for chip accounting."""
    kind = rng.random()
    if kind < 0.25:
        s = rng.choice([20, 100, 1000])
        return (s, s)
    if kind < 0.5:
        return (rng.randint(1, 2500), rng.randint(1, 2500))
    if kind < 0.75:
        return (rng.randint(1, 25), rng.randint(1, 400))  # one very short stack
    if kind < 0.9:
        return (rng.randint(1, 400), rng.randint(1, 25))
    return (rng.randint(1, 12), rng.randint(1, 12))  # both below / around the blinds


def dealt_board(engine: HandEngine) -> tuple[str, ...]:
    """Board cards PokerKit has really dealt - ground truth independent of ``HandEngine.board``."""
    return tuple(repr(card) for cards in engine._state.board_cards for card in cards)


# ------------------------------------------------------------------------ actions
def abstract_action_for(engine: HandEngine, abstract: AbstractAction) -> Action:
    """Concrete action for ``abstract`` using the same data an agent would see."""
    seat = engine.actor_seat
    assert seat is not None
    obs = engine.observation_for(seat)
    return abstract_to_action(abstract, obs.pot, obs.my_bet, obs.opp_bet, obs.legal)


def random_action(rng: random.Random, engine: HandEngine) -> Action:
    """A random LEGAL concrete action: half the time via the abstraction, otherwise uniform amounts."""
    legal = engine.legal_actions()
    seat = engine.actor_seat
    assert seat is not None
    if rng.random() < 0.5:
        mask = engine.observation_for(seat).abstract_mask
        choices = [AbstractAction(i) for i in range(len(mask)) if mask[i]]
        return abstract_action_for(engine, rng.choice(choices))
    return random_concrete_action(rng, legal)


def random_concrete_action(rng: random.Random, legal: LegalActions) -> Action:
    options = ["call"]
    if legal.can_fold:
        options.append("fold")
    if legal.can_raise:
        options += ["raise", "raise", "min", "max"]
    pick = rng.choice(options)
    if pick == "fold":
        return Action.fold()
    if pick == "call":
        return Action.check_call()
    if pick == "min":
        return Action.raise_to(legal.min_raise_to)
    if pick == "max":
        return Action.raise_to(legal.max_raise_to)
    return Action.raise_to(rng.randint(legal.min_raise_to, legal.max_raise_to))


def play_script(engine: HandEngine, script: Iterable[str | int]) -> list:
    """Apply ``"f"`` (fold), ``"c"`` (check/call) and ints (raise to N). Returns the events."""
    events = []
    for token in script:
        if token == "f":
            events.append(engine.act(Action.fold()))
        elif token == "c":
            events.append(engine.act(Action.check_call()))
        else:
            assert isinstance(token, int), token
            events.append(engine.act(Action.raise_to(token)))
    return events


def new_engine(
    deal: Deal | None = None,
    button_seat: int = 0,
    stacks: tuple[int, int] | None = None,
    cfg: GameConfig = DEFAULT_CFG,
    seed: int = 1,
) -> HandEngine:
    return HandEngine(cfg, deal or Deal.from_seed(seed), button_seat, stacks)


# ------------------------------------------------------------------ random states
def random_midhand_engines(seed: int, count: int) -> Iterator[HandEngine]:
    """``count`` engines paused at a decision point (random stacks, random action prefix)."""
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        stacks = random_stacks(rng)
        engine = HandEngine(DEFAULT_CFG, random_deal(rng), rng.randint(0, 1), stacks, rng.randint(0, 99))
        prefix = rng.choice([0, 0, 1, 2, 3, 4, 5, 6, 8])
        for _ in range(prefix):
            if engine.is_complete:
                break
            engine.act(random_action(rng, engine))
        if engine.is_complete:
            continue
        produced += 1
        yield engine


# ------------------------------------------------------------------------ matches
def run_match(
    agents: Sequence[Agent],
    n_hands: int,
    seed: int,
    cfg: GameConfig = DEFAULT_CFG,
) -> list[HandRecord]:
    """Minimal deterministic match runner (alternating button, persistent opponent trackers)."""
    trackers = (OpponentTracker(), OpponentTracker())
    for i, agent in enumerate(agents):
        agent.reset(derive_seed("test", seed, "agent", i))
    records = []
    for h in range(n_hands):
        deal = Deal.from_seed(derive_seed("test", seed, "deal", h))
        records.append(play_hand(cfg, deal, h % 2, agents, hand_number=h, trackers=trackers))
    return records


# -------------------------------------------------------------------------- test agents
class HoleCardAgent(Agent):
    """Deterministic, card-aware agent: a pure function of exactly what an observation contains."""

    architecture = "test"

    def __init__(self, name: str = "HoleCard") -> None:
        self.name = name

    def act(self, obs: Observation) -> Decision:
        key = repr((obs.my_hole, obs.board, int(obs.street), obs.pot, obs.my_bet, obs.opp_bet, obs.my_stack))
        mask = obs.abstract_mask
        choices = [AbstractAction(i) for i in range(len(mask)) if mask[i]]
        return Decision.from_abstract(obs, choices[zlib.crc32(key.encode()) % len(choices)])
