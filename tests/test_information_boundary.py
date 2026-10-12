"""THE ANTI-CHEATING SUITE.

Agents (and everything built for them: features, training code, explanations) may only ever see an
:class:`Observation`.  These tests prove from several independent angles that an observation carries
no information the player could not legitimately know:

a. type whitelist - an observation is a tree of plain frozen values (no engine, deck, PokerKit, callables)
b. differential / non-interference - changing only hidden cards never changes what a player sees
c. card universe - every card string that reaches an agent is one it is entitled to know
d. spy agent - object-graph walk + pickle audit of everything handed to ``act`` / ``end_hand``
e. static architecture - agent-side packages cannot even import the hidden-state modules
f. deterministic agents decide identically whatever the hidden cards are
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import enum
import gc
import json
import pickle
import pickletools
import random
import re
import typing
from collections.abc import Iterable
from pathlib import Path

import pytest

from poker_arena.agents.base import Agent, Decision
from poker_arena.agents.baselines import CallStationAgent, ManiacAgent, RandomAgent
from poker_arena.agents.registry import check_availability, create_agent
from poker_arena.cards import ALL_CARDS
from poker_arena.config import REPO_ROOT, derive_seed
from poker_arena.engine.actions import Action, LegalActions
from poker_arena.engine.deals import Deal
from poker_arena.engine.game import HandEngine
from poker_arena.engine.observation import (
    BOARD_CARDS_BY_STREET,
    Observation,
    OpponentStats,
    PublicAction,
    Street,
)
from poker_arena.engine.opponent import OpponentTracker, PublicHandSummary
from poker_arena.engine.session import HandSession, play_hand

from core_helpers import (
    DEFAULT_CFG,
    HoleCardAgent,
    dealt_board,
    random_action,
    random_deal,
    random_midhand_engines,
    random_stacks,
)

CARD_TOKEN = re.compile(r'"([2-9TJQKA][cdhs])"')  # quoted card strings in JSON dumps
CARD_TOKEN_REPR = re.compile(r"'([2-9TJQKA][cdhs])'")  # quoted card strings in reprs
STRICT_PRIMITIVES = (bool, int, float, str, type(None))
ALLOWED_DATACLASS_MODULES = ("poker_arena.engine.observation", "poker_arena.engine.actions", "poker_arena.engine.opponent")
FORBIDDEN_MODULES = (
    "poker_arena.engine.game",
    "poker_arena.engine.history",
    "poker_arena.engine.deals",
    "poker_arena.engine.session",
    "pokerkit",
)
FORBIDDEN_FIELD_WORDS = ("deal", "deck", "burn", "engine", "seed", "rng", "random", "state")


# =========================================================================================
# a. type whitelist
# =========================================================================================
def _check_plain_value(value: object, path: str) -> None:
    """Recursively assert ``value`` is a primitive, a plain tuple, an IntEnum or an allowed frozen dataclass."""
    if isinstance(value, enum.Enum):
        assert type(value).__module__ in ALLOWED_DATACLASS_MODULES, f"{path}: foreign enum {type(value)}"
        assert isinstance(value, int), f"{path}: enum must be a plain IntEnum"
        return
    if type(value) in STRICT_PRIMITIVES:  # exact types: rules out numpy scalars, subclasses, ...
        return
    if type(value) is tuple:
        for i, item in enumerate(value):
            _check_plain_value(item, f"{path}[{i}]")
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        assert cls.__module__ in ALLOWED_DATACLASS_MODULES, f"{path}: dataclass from {cls.__module__}"
        assert cls.__dataclass_params__.frozen, f"{path}: {cls.__name__} must be frozen"  # type: ignore[attr-defined]
        assert not hasattr(value, "__dict__"), f"{path}: {cls.__name__} carries a __dict__ (hidden attributes)"
        declared = {f.name for f in dataclasses.fields(value)}
        assert set(getattr(cls, "__slots__", ())) == declared, f"{path}: slots differ from declared fields"
        for f in dataclasses.fields(value):
            for word in FORBIDDEN_FIELD_WORDS:
                assert word not in f.name, f"{path}.{f.name}: suspicious field name"
            _check_plain_value(getattr(value, f.name), f"{path}.{f.name}")
        return
    raise AssertionError(f"{path}: value of forbidden type {type(value)!r}")


def _check_type_hint(hint: object, path: str) -> None:
    origin = typing.get_origin(hint)
    if origin is tuple:
        for arg in typing.get_args(hint):
            if arg is not Ellipsis:
                _check_type_hint(arg, path)
        return
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        for arg in typing.get_args(hint):
            _check_type_hint(arg, path)
        return
    if hint in STRICT_PRIMITIVES:
        return
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        assert hint.__module__ in ALLOWED_DATACLASS_MODULES, f"{path}: {hint}"
        return
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        assert hint.__module__ in ALLOWED_DATACLASS_MODULES, f"{path}: {hint}"
        for name, sub in typing.get_type_hints(hint).items():
            _check_type_hint(sub, f"{path}.{name}")
        return
    raise AssertionError(f"{path}: annotation {hint!r} is not a plain value type")


def test_observation_class_is_made_of_plain_frozen_types() -> None:
    for name, hint in typing.get_type_hints(Observation).items():
        _check_type_hint(hint, f"Observation.{name}")
    params = Observation.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen
    names = [f.name for f in dataclasses.fields(Observation)]
    assert not [n for n in names if n.startswith(("opp_hole", "opp_card", "opponent_hole", "undealt", "future"))]
    assert names.count("my_hole") == 1 and "board" in names
    assert not hasattr(Observation(*_sample_observation_args()), "__dict__")
    # numpy appears only through a read-only property, never as stored state
    assert isinstance(Observation.__dict__["abstract_mask"], property)
    assert "abstract_mask" not in {f.name for f in dataclasses.fields(Observation)}


def _sample_observation_args() -> tuple:
    return (0, Street.PREFLOP, True, ("As", "Kd"), (), 15, 995, 990, 5, 10, 5, 10, 5, 10,
            LegalActions(True, False, True, 5, True, 20, 1000), (), OpponentStats())


def test_public_summary_class_is_made_of_plain_frozen_types() -> None:
    for name, hint in typing.get_type_hints(PublicHandSummary).items():
        _check_type_hint(hint, f"PublicHandSummary.{name}")
    assert PublicHandSummary.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_every_observation_value_is_plain_over_many_states() -> None:
    n = 0
    for engine in random_midhand_engines(11, 250):
        for seat in (0, 1):
            obs = engine.observation_for(seat)
            _check_plain_value(obs, "obs")
            n += 1
    assert n == 500


def test_observation_is_immutable() -> None:
    obs = HandEngine(DEFAULT_CFG, Deal.from_seed(1), 0).observation_for(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.my_hole = ("As", "Ad")  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        obs.secret = 1  # type: ignore[attr-defined]
    assert isinstance(obs.board, tuple) and isinstance(obs.history, tuple)  # tuples cannot be mutated in place


def test_whitelist_rejects_hostile_values() -> None:
    """Self-test: the checker is not vacuous."""
    engine = HandEngine(DEFAULT_CFG, Deal.from_seed(1), 0)
    deal = Deal.from_seed(1)
    import numpy as np

    for bad in (engine, deal, [1], {"a": 1}, {1}, lambda: 0, np.zeros(2), np.float64(1.0), object(), b"x"):
        with pytest.raises(AssertionError):
            _check_plain_value((1, bad), "x")

    @dataclasses.dataclass
    class Mutable:
        a: int = 1

    @dataclasses.dataclass(frozen=True)
    class Foreign:
        a: int = 1

    for bad in (Mutable(), Foreign()):
        with pytest.raises(AssertionError):
            _check_plain_value(bad, "x")


# =========================================================================================
# b. differential / non-interference
# =========================================================================================
# indices into Deal.cards (button hole 0-1, bb hole 2-3, burn 4, flop 5-7, burn 8, turn 9, burn 10, river 11)
_BURN_IDX = (4, 8, 10)
_BOARD_IDX = (5, 6, 7, 9, 11)


def _hole_idx(deal_button_is_hero: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    """(hero hole indices, opponent hole indices)."""
    return ((0, 1), (2, 3)) if deal_button_is_hero else ((2, 3), (0, 1))


def _passive_random_action(rng: random.Random, engine: HandEngine) -> Action:
    """Mostly calls/checks so that hands regularly reach the turn and river."""
    if rng.random() < 0.55:
        return Action.check_call()
    return random_action(rng, engine)


def dry_run(deal: Deal, button: int, stacks: tuple[int, int], seed: int) -> tuple[list[Action], int]:
    """Play a random hand; return its actions and the board length of the last undecided state."""
    rng = random.Random(seed)
    engine = HandEngine(DEFAULT_CFG, deal, button, stacks)
    actions: list[Action] = []
    known_board = 0
    while not engine.is_complete:
        known_board = len(dealt_board(engine))  # PokerKit's own dealt cards, not the wrapper's view
        action = _passive_random_action(rng, engine)
        engine.act(action)
        actions.append(action)
    return actions, known_board


def hidden_variant(rng: random.Random, deal: Deal, hero: int, button: int, known_board: int, mode: str) -> Deal:
    """A deal that differs from ``deal`` ONLY in cards ``hero`` cannot know at any point of the hand.

    ``mode``: ``opp`` = opponent hole cards, ``future`` = burns + board cards beyond ``known_board``,
    ``both`` = all of them.
    """
    _, opp_idx = _hole_idx(hero == button)
    revealed_idx = [i for n, i in enumerate(_BOARD_IDX) if n < known_board]
    hero_idx, _ = _hole_idx(hero == button)
    known = {deal.cards[i] for i in (*hero_idx, *revealed_idx)}
    hidden = []
    if mode in ("opp", "both"):
        hidden += list(opp_idx)
    if mode in ("future", "both"):
        hidden += [*_BURN_IDX, *(i for n, i in enumerate(_BOARD_IDX) if n >= known_board)]
    kept = {c for i, c in enumerate(deal.cards) if i not in hidden}  # cards that stay where they are
    assert known <= kept
    pool = [c for c in ALL_CARDS if c not in kept]
    while True:
        replacement = rng.sample(pool, len(hidden))
        cards = list(deal.cards)
        for idx, card in zip(hidden, replacement):
            cards[idx] = card
        if cards != list(deal.cards):
            assert {cards[i] for i in (*hero_idx, *revealed_idx)} == known
            return Deal(tuple(cards))


def assert_same_view(ea: HandEngine, eb: HandEngine, hero: int) -> None:
    assert ea.actor_seat == eb.actor_seat and ea.is_complete == eb.is_complete
    oa, ob = ea.observation_for(hero), eb.observation_for(hero)
    assert oa == ob
    assert repr(oa) == repr(ob)
    assert oa.to_dict() == ob.to_dict()
    assert oa.abstract_mask.tolist() == ob.abstract_mask.tolist()
    assert ea.legal_actions() == eb.legal_actions()


def lockstep(deal_a: Deal, deal_b: Deal, button: int, stacks: tuple[int, int], actions: Iterable[Action], hero: int) -> int:
    ea = HandEngine(DEFAULT_CFG, deal_a, button, stacks)
    eb = HandEngine(DEFAULT_CFG, deal_b, button, stacks)
    compared = 0
    for action in actions:
        assert not ea.is_complete and not eb.is_complete
        assert_same_view(ea, eb, hero)
        compared += 1
        ea.act(action)
        eb.act(action)
    assert ea.is_complete and eb.is_complete
    return compared


@pytest.mark.parametrize("hero", [0, 1])
@pytest.mark.parametrize("button", [0, 1])
@pytest.mark.parametrize("mode", ["opp", "future", "both"])
def test_hidden_cards_never_change_a_players_observation(hero: int, button: int, mode: str) -> None:
    rng = random.Random(derive_seed("test", "diff", hero, button, mode))
    reached = dict.fromkeys((0, 3, 4, 5), 0)
    compared = 0
    for i in range(70):
        deal = random_deal(rng)
        stacks = random_stacks(rng) if i % 3 else (1000, 1000)
        actions, known_board = dry_run(deal, button, stacks, seed=rng.randrange(10**9))
        other = hidden_variant(rng, deal, hero, button, known_board, mode)
        compared += lockstep(deal, other, button, stacks, actions, hero)
        reached[known_board] += 1
    assert compared > 70
    assert all(n >= 3 for n in reached.values()), f"hands should reach every street: {reached}"


def test_variant_generator_really_changes_hidden_cards_only() -> None:
    """Self-test of the generator used above."""
    rng = random.Random(3)
    deal = Deal.from_seed(5)
    for hero in (0, 1):
        for button in (0, 1):
            for mode in ("opp", "future", "both"):
                for known in (0, 3, 4, 5):
                    other = hidden_variant(rng, deal, hero, button, known, mode)
                    assert other != deal
                    hero_idx, opp_idx = _hole_idx(hero == button)
                    assert [other.cards[i] for i in hero_idx] == [deal.cards[i] for i in hero_idx]
                    revealed = [i for n, i in enumerate(_BOARD_IDX) if n < known]
                    assert [other.cards[i] for i in revealed] == [deal.cards[i] for i in revealed]
                    if mode == "opp":
                        assert all(other.cards[i] == deal.cards[i] for i in (*_BURN_IDX, *_BOARD_IDX))
                    if mode == "future":
                        assert all(other.cards[i] == deal.cards[i] for i in opp_idx)


def test_changing_the_opponents_cards_does_change_the_opponents_view() -> None:
    """Negative control: the opponent (whose cards we replaced) DOES see a different observation."""
    rng = random.Random(8)
    deal = Deal.from_seed(21)
    other = hidden_variant(rng, deal, hero=0, button=0, known_board=0, mode="opp")
    e1, e2 = HandEngine(DEFAULT_CFG, deal, 0), HandEngine(DEFAULT_CFG, other, 0)
    assert e1.observation_for(0) == e2.observation_for(0)
    assert e1.observation_for(1) != e2.observation_for(1)
    assert e1.observation_for(1).my_hole != e2.observation_for(1).my_hole


@pytest.mark.parametrize("hero", [0, 1])
def test_negative_control_own_hole_cards_are_visible(hero: int) -> None:
    deal = Deal.from_seed(77)
    hero_idx, _ = _hole_idx(hero == 0)  # button = seat 0
    cards = list(deal.cards)
    free = [c for c in ALL_CARDS if c not in cards]
    cards[hero_idx[0]] = free[0]
    changed = Deal(tuple(cards))
    e1, e2 = HandEngine(DEFAULT_CFG, deal, 0), HandEngine(DEFAULT_CFG, changed, 0)
    assert e1.observation_for(hero) != e2.observation_for(hero)
    assert e1.observation_for(hero).my_hole != e2.observation_for(hero).my_hole


@pytest.mark.parametrize("revealed_index", [0, 1, 2, 3, 4])  # flop x3, turn, river
def test_negative_control_a_board_card_matters_exactly_from_the_moment_it_is_dealt(revealed_index: int) -> None:
    """The differential test must be able to fail: changing board card *i* changes the observation
    from the street that reveals it - and not one action earlier."""
    deal = Deal.from_seed(404)
    cards = list(deal.cards)
    board_pos = _BOARD_IDX[revealed_index]
    free = [c for c in ALL_CARDS if c not in cards]
    cards[board_pos] = free[0]
    changed = Deal(tuple(cards))
    needed_len = 3 if revealed_index < 3 else revealed_index + 1  # board length once the card is dealt
    e1, e2 = HandEngine(DEFAULT_CFG, deal, 0), HandEngine(DEFAULT_CFG, changed, 0)
    differed_at = None
    step = 0
    for _ in range(8):  # check-down
        same = e1.observation_for(1) == e2.observation_for(1) and e1.observation_for(0) == e2.observation_for(0)
        if not same and differed_at is None:
            differed_at = len(e1.board)
        if e1.is_complete:
            break
        assert (len(e1.board) < needed_len) == same, f"step {step}: board len {len(e1.board)}"
        e1.act(Action.check_call())
        e2.act(Action.check_call())
        step += 1
    assert differed_at == needed_len


def test_engine_hidden_state_is_only_reachable_through_spectator_view() -> None:
    """Whatever ``observation_for`` returns contains no object shared with the engine's private state."""
    engine = HandEngine(DEFAULT_CFG, Deal.from_seed(9), 0)
    obs = engine.observation_for(0)
    private_ids = {id(engine), id(engine.deal), id(engine._state), id(engine.events), id(engine._board)}
    seen: set[int] = set()
    stack: list[object] = [obs]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (type, enum.Enum)):
            continue
        seen.add(id(obj))
        assert id(obj) not in private_ids
        stack.extend(gc.get_referents(obj))


# =========================================================================================
# c + d. spy agent: card universe, object graph and pickle audits
# =========================================================================================
@dataclasses.dataclass
class Truth:
    """What the (test-only) spy knows about the real hand - never available to a real agent."""

    deal: Deal
    seat: int
    button: int
    board_now: tuple[str, ...]

    @property
    def own_hole(self) -> tuple[str, str]:
        return self.deal.button_hole if self.seat == self.button else self.deal.bb_hole


def audit_object_graph(root: object) -> None:
    """Every object reachable from ``root`` must be a plain value of an allowed class."""
    allowed = (
        *STRICT_PRIMITIVES, tuple, Street, Observation, LegalActions, PublicAction, OpponentStats, PublicHandSummary,
    )
    seen: set[int] = set()
    stack: list[tuple[object, object]] = [(root, None)]
    while stack:
        obj, parent = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, type):
            assert parent is not None and obj is type(parent), f"reference to class/callable {obj!r}"
            continue
        assert type(obj) in allowed, f"reachable object of type {type(obj)!r}"
        assert not callable(obj)
        if isinstance(obj, enum.Enum):
            continue
        stack.extend((child, obj) for child in gc.get_referents(obj))


def pickled_strings(obj: object) -> set[str]:
    """All string operands of the pickle stream (card strings, class names, module names)."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return {arg for _, arg, _ in pickletools.genops(data) if isinstance(arg, str)}


def audit_pickle(obj: object, hidden: set[str]) -> None:
    strings = pickled_strings(obj)
    assert not strings & hidden, f"pickle of {type(obj).__name__} contains hidden cards {strings & hidden}"
    for s in strings:
        assert not any(s == m or s.startswith(m + ".") for m in FORBIDDEN_MODULES), f"pickle references {s!r}"
        assert "HandEngine" not in s and "Deal" not in s and "State" not in s, f"pickle references {s!r}"
    pickle.loads(pickle.dumps(obj))  # and it round-trips as plain data


def card_tokens(obj: dict) -> set[str]:
    return set(CARD_TOKEN.findall(json.dumps(obj)))


def audit_observation(obs: Observation, truth: Truth) -> None:
    """Everything in ``obs`` is something ``truth.seat`` may know at this moment."""
    allowed = set(truth.own_hole) | set(truth.board_now)
    hidden = set(truth.deal.cards) - allowed
    assert obs.my_hole == truth.own_hole and obs.board == truth.board_now
    assert obs.is_button == (truth.seat == truth.button)
    # every card string anywhere (nested dict/list/repr) is own hole or revealed board
    assert card_tokens(obs.to_dict()) <= allowed, "observation mentions a card the player cannot know"
    assert set(CARD_TOKEN_REPR.findall(repr(obs))) <= allowed
    assert not (card_tokens(obs.to_dict()) & hidden)
    _check_plain_value(obs, "obs")
    audit_object_graph(obs)
    audit_pickle(obs, hidden)
    assert len(obs.board) == BOARD_CARDS_BY_STREET[obs.street]


def audit_summary(summary: PublicHandSummary, deal: Deal, seat: int, button: int, result) -> None:
    """The post-hand information: opponent cards only at showdown, never undealt board cards."""
    opp_hole = deal.bb_hole if seat == button else deal.button_hole
    own_hole = deal.button_hole if seat == button else deal.bb_hole
    assert summary.showdown == result.showdown
    assert summary.board == deal.board[: len(summary.board)] == result.board
    assert len(summary.board) == BOARD_CARDS_BY_STREET[result.ended_street]
    if summary.showdown:
        assert summary.opp_hole == opp_hole, "showdown reveals the real opponent cards"
        assert len(summary.board) == 5
    else:
        assert summary.opp_hole is None, "no showdown => opponent cards stay hidden"
    revealed = set(summary.board) | (set(summary.opp_hole) if summary.opp_hole else set())
    hidden = set(deal.cards) - revealed - set(own_hole)
    text = repr(summary)
    assert set(CARD_TOKEN_REPR.findall(text)) <= revealed
    assert not set(CARD_TOKEN_REPR.findall(text)) & hidden
    _check_plain_value(summary, "summary")
    audit_object_graph(summary)
    audit_pickle(summary, hidden | ({*opp_hole} if not summary.showdown else set()))
    assert summary.my_payoff == result.payoffs[seat] and summary.was_button == (seat == button)


class SpyAgent(Agent):
    """Wraps a real agent; audits and records everything the engine hands over."""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.name = inner.name
        self.architecture = inner.architecture
        self.truth: Truth | None = None
        self.final: tuple[Deal, int, int, object] | None = None
        self.observations: list[Observation] = []
        self.summaries: list[PublicHandSummary] = []

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)

    def act(self, obs: Observation) -> Decision:
        assert self.truth is not None
        audit_observation(obs, self.truth)
        self.observations.append(obs)
        return self.inner.act(obs)

    def end_hand(self, summary: PublicHandSummary) -> None:
        assert self.final is not None
        deal, seat, button, result = self.final
        audit_summary(summary, deal, seat, button, result)
        self.summaries.append(summary)
        self.inner.end_hand(summary)


def play_audited_hand(deal: Deal, button: int, spies: list[SpyAgent], stacks, trackers, hand_number: int):
    session = HandSession(DEFAULT_CFG, deal, button, spies, start_stacks=stacks, hand_number=hand_number, trackers=trackers)
    while not session.is_complete:
        seat = session.actor_seat
        assert seat is not None
        spies[seat].truth = Truth(deal, seat, button, dealt_board(session.engine))
        session.step()
    result = session.engine.result()
    for seat in (0, 1):
        spies[seat].final = (deal, seat, button, result)
    return session.finish()


MATCHUPS = [
    ("random-random", lambda: (RandomAgent("R1"), RandomAgent("R2"))),
    ("maniac-station", lambda: (ManiacAgent("M"), CallStationAgent("C"))),
    ("random-maniac", lambda: (RandomAgent("R"), ManiacAgent("M"))),
]


@pytest.mark.parametrize("name,factory", MATCHUPS, ids=[m[0] for m in MATCHUPS])
def test_spy_agents_never_receive_hidden_information(name: str, factory) -> None:
    rng = random.Random(derive_seed("test", "spy", name))
    spies = [SpyAgent(a) for a in factory()]
    for i, s in enumerate(spies):
        s.reset(derive_seed("test", "spy-agent", name, i))
    trackers = (OpponentTracker(), OpponentTracker())
    decisions = hands_with_showdown = hands_without = 0
    for h in range(150):
        stacks = random_stacks(rng) if h % 2 else (1000, 1000)
        record = play_audited_hand(random_deal(rng), h % 2, spies, stacks, trackers, h)
        hands_with_showdown += record.result.showdown
        hands_without += not record.result.showdown
    for s in spies:
        decisions += len(s.observations)
        assert len(s.summaries) == 150
    assert decisions > 250 and hands_with_showdown > 15
    assert hands_without > 15 or name == "maniac-station"  # a calling station never lets anyone fold


def test_card_universe_over_thousands_of_baseline_hands() -> None:
    """Cheap, wide version of the audit: card strings per observation vs. the engine's own truth."""
    rng = random.Random(derive_seed("test", "universe"))
    agents = [RandomAgent("R", seed=1), ManiacAgent("M", seed=2)]
    trackers = (OpponentTracker(), OpponentTracker())
    seen_cards: set[str] = set()
    n_obs = 0
    for h in range(1000):
        deal = random_deal(rng)
        button = h % 2
        stacks = random_stacks(rng) if h % 3 else (1000, 1000)
        session = HandSession(DEFAULT_CFG, deal, button, agents, start_stacks=stacks, hand_number=h, trackers=trackers)
        while not session.is_complete:
            seat = session.actor_seat
            own = deal.button_hole if seat == button else deal.bb_hole
            board = dealt_board(session.engine)
            outcome = session.step()
            tokens = card_tokens(outcome.observation.to_dict())
            assert tokens <= set(own) | set(board), (tokens, own, board)
            assert outcome.observation.board == board and outcome.observation.my_hole == own
            seen_cards |= tokens
            n_obs += 1
        session.finish()
    assert n_obs > 1600 and len(seen_cards) == 52, "every card was seen by *someone*, but only legitimately"


def test_audit_helpers_detect_leaks() -> None:
    """Self-tests: the audits are able to fail."""
    deal = Deal.from_seed(2)
    engine = HandEngine(DEFAULT_CFG, deal, 0)
    obs = engine.observation_for(0)
    truth = Truth(deal, 0, 0, ())
    audit_observation(obs, truth)  # clean

    leaky = dataclasses.replace(obs, my_hole=deal.bb_hole)  # pretend the opponent's cards leaked in
    with pytest.raises(AssertionError):
        audit_observation(leaky, truth)
    leaky = dataclasses.replace(obs, board=deal.board[:3])  # future board cards
    with pytest.raises(AssertionError):
        audit_observation(leaky, truth)
    with pytest.raises(AssertionError):
        audit_pickle((obs, deal.bb_hole), set(deal.bb_hole))
    with pytest.raises(AssertionError):
        audit_pickle((obs, engine), set())
    with pytest.raises(AssertionError):
        audit_object_graph((obs, engine))
    with pytest.raises(AssertionError):
        audit_object_graph((obs, lambda: 0))
    with pytest.raises(AssertionError):
        audit_object_graph((obs, Deal))  # even a bare reference to the class

    class Sneaky:
        __slots__ = ("legal", "secret")

    with pytest.raises(AssertionError):
        audit_object_graph(Sneaky())


def test_summary_only_reveals_opponent_cards_at_showdown() -> None:
    deal = Deal.from_seed(12)
    # fold hand
    s_fold = HandSession(DEFAULT_CFG, deal, 0, [None, None])
    s_fold.submit(Action.raise_to(30))
    s_fold.submit(Action.fold())
    rec = s_fold.finish()
    for seat in (0, 1):
        summary = rec.public_summary(seat)
        assert summary.opp_hole is None and not summary.showdown and summary.board == ()
        assert not set(CARD_TOKEN_REPR.findall(repr(summary)))
    # showdown hand
    s_show = HandSession(DEFAULT_CFG, deal, 0, [None, None])
    s_show.submit(Action.raise_to(1000))
    s_show.submit(Action.check_call())
    rec = s_show.finish()
    assert rec.public_summary(0).opp_hole == deal.bb_hole
    assert rec.public_summary(1).opp_hole == deal.button_hole
    assert rec.public_summary(0).board == deal.board


def test_fold_on_flop_summary_has_exactly_three_board_cards() -> None:
    deal = Deal.from_seed(13)
    session = HandSession(DEFAULT_CFG, deal, 1, [None, None])
    for action in (Action.check_call(), Action.check_call(), Action.raise_to(20), Action.fold()):
        session.submit(action)
    rec = session.finish()
    summary = rec.public_summary(0)
    assert summary.board == deal.board[:3] and not set(summary.board) & set(deal.burns)
    assert summary.opp_hole is None


# =========================================================================================
# e. static architecture test (AST)
# =========================================================================================
GUARDED_PACKAGES = ("agents", "features", "training", "explain")
SRC_ROOT = REPO_ROOT / "src" / "poker_arena"
RULE_EXPLANATION = (
    "ARCHITECTURE RULE (docs/ARCHITECTURE.md section 3): agents, feature code, trainers and explanation code "
    "must NEVER import poker_arena.engine.game / history / deals / session or pokerkit - those modules hold the "
    "hidden state (opponent cards, burns, undealt board). They may only use engine.observation, engine.actions "
    "and engine.opponent. Orchestration code (data/, tournament/, ui/, app/, scripts/) may use HandSession/play_hand."
)


def _is_forbidden(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in FORBIDDEN_MODULES)


def forbidden_imports(source: str, module_name: str, is_package: bool = False) -> list[str]:
    """Names of forbidden modules imported (directly, relatively or via string-based dynamic import)."""
    tree = ast.parse(source)
    package = module_name if is_package else module_name.rpartition(".")[0]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if _is_forbidden(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - (node.level - 1)])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if _is_forbidden(module):
                found.append(module)
            else:  # ``from poker_arena.engine import game`` imports the forbidden submodule
                found += [f"{module}.{a.name}" for a in node.names if _is_forbidden(f"{module}.{a.name}")]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.split(":")[0]
            if " " not in text and _is_forbidden(text) and "." in text:
                found.append(f"{node.value!r} (string / dynamic import)")
    return found


def _module_name(path: Path) -> tuple[str, bool]:
    rel = path.relative_to(SRC_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _python_files(package: str) -> list[Path]:
    return sorted((SRC_ROOT / package).rglob("*.py")) if (SRC_ROOT / package).is_dir() else []


@pytest.mark.parametrize("package", GUARDED_PACKAGES)
def test_agent_side_packages_do_not_import_hidden_state_modules(package: str) -> None:
    files = _python_files(package)
    if not files:
        pytest.skip(f"poker_arena/{package} has no python files (yet)")
    violations = []
    for path in files:
        name, is_pkg = _module_name(path)
        for bad in forbidden_imports(path.read_text(encoding="utf-8"), name, is_pkg):
            violations.append(f"{path.relative_to(REPO_ROOT)}: imports {bad}")
    assert not violations, RULE_EXPLANATION + "\nViolations:\n  " + "\n  ".join(violations)


def test_observation_boundary_modules_do_not_import_hidden_state_modules() -> None:
    """The modules that DEFINE what agents see must themselves be free of engine internals."""
    for rel in ("engine/observation.py", "engine/actions.py", "engine/opponent.py", "cards.py", "config.py"):
        path = SRC_ROOT / rel
        name, is_pkg = _module_name(path)
        bad = forbidden_imports(path.read_text(encoding="utf-8"), name, is_pkg)
        assert not bad, f"{rel} imports {bad}\n{RULE_EXPLANATION}"


def test_core_agent_modules_import_only_the_boundary_modules() -> None:
    """agents/base.py, baselines.py and registry.py: verify what they do import (not just what they avoid)."""
    allowed_engine = {"poker_arena.engine.actions", "poker_arena.engine.observation", "poker_arena.engine.opponent"}
    for rel in ("agents/base.py", "agents/baselines.py", "agents/registry.py"):
        tree = ast.parse((SRC_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("poker_arena.engine"):
                assert node.module in allowed_engine, f"{rel} imports {node.module}"
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("poker_arena.engine") for a in node.names)


@pytest.mark.parametrize(
    "source,module,is_pkg,expected_hits",
    [
        ("from poker_arena.engine.game import HandEngine", "poker_arena.agents.x", False, 1),
        ("import poker_arena.engine.deals", "poker_arena.agents.x", False, 1),
        ("import pokerkit", "poker_arena.features.x", False, 1),
        ("from pokerkit import State", "poker_arena.features.x", False, 1),
        ("from pokerkit.state import State", "poker_arena.features.x", False, 1),
        ("from poker_arena.engine import game", "poker_arena.agents.x", False, 1),
        ("from poker_arena.engine import session, history", "poker_arena.agents.x", False, 2),
        ("from ..engine.game import HandEngine", "poker_arena.agents.x", False, 1),
        ("from ..engine import deals", "poker_arena.agents.x", False, 1),
        ("from .. import engine\nfrom ...engine.session import HandSession", "poker_arena.agents.sub.x", False, 1),
        ("from ..engine.game import HandEngine", "poker_arena.agents", True, 1),
        ("import importlib\nimportlib.import_module('poker_arena.engine.game')", "poker_arena.agents.x", False, 1),
        ("LAZY = 'poker_arena.engine.session:HandSession'", "poker_arena.agents.x", False, 1),
        ("if TYPE_CHECKING:\n    from poker_arena.engine.game import HandEngine", "poker_arena.agents.x", False, 1),
        # allowed
        ("from poker_arena.engine.observation import Observation", "poker_arena.agents.x", False, 0),
        ("from poker_arena.engine.actions import Action\nfrom poker_arena.engine.opponent import OpponentTracker", "poker_arena.agents.x", False, 0),
        ("from ..engine.observation import Observation", "poker_arena.agents.x", False, 0),
        ("from poker_arena.engine import observation, actions", "poker_arena.agents.x", False, 0),
        ("from poker_arena.cards import ALL_CARDS", "poker_arena.agents.x", False, 0),
        ('"""never import poker_arena.engine.game"""', "poker_arena.agents.x", False, 0),
    ],
)
def test_import_checker_self_test(source: str, module: str, is_pkg: bool, expected_hits: int) -> None:
    assert len(forbidden_imports(source, module, is_pkg)) == expected_hits, forbidden_imports(source, module, is_pkg)


# =========================================================================================
# f. deterministic agents are blind to hidden cards
# =========================================================================================
def _decisions(record) -> list[tuple[int, Action]]:
    return [(e.seat, e.action) for e in record.events]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: (CallStationAgent("a"), CallStationAgent("b")),
        lambda: (ManiacAgent("a", seed=5), CallStationAgent("b")),
        lambda: (ManiacAgent("a", seed=5), ManiacAgent("b", seed=6)),
        lambda: (RandomAgent("a", seed=9), RandomAgent("b", seed=10)),
    ],
    ids=["station-station", "maniac-station", "maniac-maniac", "random-random"],
)
def test_card_independent_agents_play_identically_on_any_deals(factory) -> None:
    """These agents never look at cards, so two *completely different* deals give the same action list."""
    rng = random.Random(1)
    for h in range(60):
        deal_a, deal_b = random_deal(rng), random_deal(rng)
        recs = []
        for deal in (deal_a, deal_b):
            agents = list(factory())
            for i, a in enumerate(agents):
                a.reset(100 + i)
            recs.append(play_hand(DEFAULT_CFG, deal, h % 2, agents))
        assert _decisions(recs[0]) == _decisions(recs[1])


@pytest.mark.parametrize("hero", [0, 1])
def test_card_aware_agent_decides_only_from_visible_information(hero: int) -> None:
    """Hero's decisions are identical whenever only hidden cards differ (opponent ignores cards)."""
    rng = random.Random(derive_seed("test", "holecard", hero))
    checked = 0
    for h in range(80):
        button = h % 2
        deal = random_deal(rng)

        def run(d: Deal):
            agents: list[Agent] = [CallStationAgent("c"), CallStationAgent("c")]
            agents[hero] = HoleCardAgent()
            # opponent mixes aggression independent of cards so that several streets are reached
            agents[1 - hero] = ManiacAgent("m", seed=3, raise_probability=0.25)
            agents[1 - hero].reset(3)
            return play_hand(DEFAULT_CFG, d, button, agents)

        base = run(deal)
        known = max([BOARD_CARDS_BY_STREET[e.street] for e in base.events if e.seat == hero] or [0])
        for mode in ("opp", "future", "both"):
            variant = hidden_variant(rng, deal, hero, button, known, mode)
            other = run(variant)
            assert _decisions(other) == _decisions(base), f"hidden cards ({mode}) changed the hero's play"
            checked += 1
    assert checked == 240


@pytest.mark.parametrize("kind", ["teacher", "ebm", "xgboost", "ga"])
@pytest.mark.parametrize("hero", [0, 1])
def test_registered_agents_decide_only_from_visible_information(kind: str, hero: int) -> None:
    """End-to-end anti-cheat for the REAL agents: changing only cards the hero cannot know never changes
    any of its decisions. Kinds that are not built / trained yet are skipped, not failed."""
    availability = check_availability(kind)
    if not availability.available:
        pytest.skip(f"agent kind {kind!r} unavailable: {availability.detail[:120]}")
    template = create_agent(kind)
    rng = random.Random(derive_seed("test", "real-agent", kind, hero))

    def run(deal: Deal, button: int):
        agents: list[Agent] = [template, template]
        hero_agent = copy.copy(template)
        hero_agent.reset(7)
        opponent = ManiacAgent("m", seed=3, raise_probability=0.3)  # ignores cards, so the betting path is fixed
        opponent.reset(3)
        agents[hero], agents[1 - hero] = hero_agent, opponent
        return play_hand(DEFAULT_CFG, deal, button, agents)

    checked = 0
    for h in range(25):
        button = h % 2
        deal = random_deal(rng)
        base = run(deal, button)
        known = max([BOARD_CARDS_BY_STREET[e.street] for e in base.events if e.seat == hero] or [0])
        for mode in ("opp", "future", "both"):
            other = run(hidden_variant(rng, deal, hero, button, known, mode), button)
            assert _decisions(other) == _decisions(base), f"{kind}: hidden cards ({mode}) changed its play"
            checked += 1
    assert checked == 75
