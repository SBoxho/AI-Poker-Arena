"""Head-to-head matches with duplicate deals, seat rotation and parallel sessions.

How one match is played (``a`` = first agent, ``b`` = second agent; seat 0 = a, seat 1 = b): every deal
``d_i`` is played twice with the seats swapped between button and big blind - a *duplicate pair*, stored
as the adjacent hands ``2i`` and ``2i + 1`` - so the luck of the cards cancels and the sum of a pair is a
low-variance measure of skill difference. Two identical **stateless** agents cancel exactly in every pair.
Agents that adapt to opponent statistics do not, if the two plays of a deal can learn from each other:
with the default ``two_table`` protocol each session is played at two independent tables (own agent
copies, own opponent trackers, the same deals, seats complemented) so that nothing about a deal crosses
between its two plays and identical adaptive agents cancel in expectation; the legacy ``single_table``
protocol plays both back-to-back and shows a first-mover artifact (see :mod:`poker_arena.tournament.duplicate`).
Deals are ``Deal.from_seed(derive_seed("eval", seed, match_key, deal_index))`` - the ``"eval"`` seed
namespace can never coincide with the training namespaces - and ``match_key`` depends only on the two
agent labels, so a pairing plays the same cards whatever else is in the tournament.

The match is cut into *sessions* of ``cfg.session_hands`` hands (``session_hands / 2`` deals). A session
starts by creating independent agent copies (one pair per table), calling ``reset(seed)`` on them with
seeds derived from (seed, match, agent label, session, table) and clearing the opponent trackers. Nothing
carries over between sessions,
therefore the sessions can run in any order on any number of processes and **the result is identical
for every ``n_workers``**. Workers (spawned processes, Windows-safe) rebuild agents from the
picklable :class:`~poker_arena.agents.registry.AgentSpec`, so each one loads trained models itself;
agent kinds must be resolvable in a fresh interpreter (built-in or lazily registered kinds - a kind
added with ``register_agent`` at runtime in the parent process is not visible to workers).
"""

from __future__ import annotations

import copy
import heapq
import json
import multiprocessing
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from poker_arena.agents.base import Agent
from poker_arena.agents.registry import KIND_LABELS, AgentSpec, create_agent
from poker_arena.config import derive_seed
from poker_arena.engine.deals import Deal
from poker_arena.engine.history import HandRecord
from poker_arena.engine.observation import Street
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.engine.session import play_hand
from poker_arena.tournament.arrays import (
    ACTION_KINDS,
    KIND_INDEX,
    NUM_STREETS,
    HandArrays,
    HandArraysBuilder,
)
from poker_arena.tournament.config import KeepHandsPolicy, MatchConfig, resolve_n_workers
from poker_arena.tournament.duplicate import play_duplicate_deals
from poker_arena.tournament.stats import pair_sums

#: Sample tags (why a hand was kept for replay).
TAG_FIRST = "first"
TAG_BIGGEST_POT = "biggest_pot"
TAG_BIGGEST_WIN_A = "biggest_win_a"
TAG_BIGGEST_WIN_B = "biggest_win_b"
SAMPLE_TAGS = (TAG_FIRST, TAG_BIGGEST_POT, TAG_BIGGEST_WIN_A, TAG_BIGGEST_WIN_B)

_BETTING_KINDS = ("bet", "raise")


# --------------------------------------------------------------------------------------
# Labels and seeds
# --------------------------------------------------------------------------------------
def agent_label(spec: AgentSpec) -> str:
    """Display name of an agent: its explicit label, else the registry's name for the kind."""
    return spec.label or KIND_LABELS.get(spec.kind, spec.kind)


def labelled(spec: AgentSpec | str) -> AgentSpec:
    """The spec with its label filled in (accepts a bare kind string)."""
    spec = AgentSpec(spec) if isinstance(spec, str) else spec
    return replace(spec, label=agent_label(spec))


def match_key(label_a: str, label_b: str) -> str:
    """Order-independent key of a pairing; the deals of a match depend on nothing else."""
    return "__vs__".join(sorted((label_a, label_b)))


def deal_seed(seed: int, key: str, deal_index: int) -> int:
    """Seed of the ``deal_index``-th deal of a match (``derive_seed`` in the ``"eval"`` namespace)."""
    return derive_seed("eval", seed, key, deal_index)


def agent_seed(seed: int, key: str, label: str, session_index: int, table: int | None = None) -> int:
    """Seed passed to ``agent.reset`` at the start of a session; ``table`` (0 or 1) distinguishes the two
    tables of the ``two_table`` protocol, ``None`` is the single legacy table (seeds of old runs)."""
    if table is None:
        return derive_seed("eval", seed, key, "agent", label, session_index)
    return derive_seed("eval", seed, key, "agent", label, session_index, "table", table)


def bootstrap_seed(seed: int, key: str) -> int:
    """Seed of the bootstrap resampling of a match (so bootstrap CIs are reproducible too)."""
    return derive_seed("eval", seed, key, "bootstrap")


# --------------------------------------------------------------------------------------
# Hand selection for replay
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SampledHand:
    """A hand kept for replay. ``record`` is a ``HandRecord`` while a session runs and its
    ``to_dict()`` form in every result object."""

    hand_index: int  # 0-based index within the match; pair index = hand_index // 2 for duplicate matches
    pot_chips: int
    profit_chips: int  # net chips won by agent A
    tags: tuple[str, ...]
    record: Any


class SampleSelector:
    """Keeps the hands asked for by a :class:`KeepHandsPolicy` using O(policy size) memory.

    Ranking is deterministic and independent of arrival order (ties go to the earlier hand), which
    is what lets per-session selections be merged into the same global selection for any number of
    workers.
    """

    def __init__(self, policy: KeepHandsPolicy) -> None:
        self._capacity = {
            TAG_FIRST: policy.first,
            TAG_BIGGEST_POT: policy.biggest_pots,
            TAG_BIGGEST_WIN_A: policy.biggest_wins_each,
            TAG_BIGGEST_WIN_B: policy.biggest_wins_each,
        }
        self._heaps: dict[str, list[tuple[int, int]]] = {tag: [] for tag in SAMPLE_TAGS}  # (key, -hand_index)
        self._kept: dict[int, SampledHand] = {}

    def offer(self, hand_index: int, pot_chips: int, profit_chips: int, record: Any) -> None:
        keys: dict[str, int | None] = {
            TAG_FIRST: -hand_index if hand_index < self._capacity[TAG_FIRST] else None,
            TAG_BIGGEST_POT: pot_chips,
            TAG_BIGGEST_WIN_A: profit_chips if profit_chips > 0 else None,
            TAG_BIGGEST_WIN_B: -profit_chips if profit_chips < 0 else None,
        }
        changed = False
        for tag, key in keys.items():
            heap, capacity = self._heaps[tag], self._capacity[tag]
            if key is None or capacity == 0:
                continue
            item = (key, -hand_index)
            if len(heap) < capacity:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            else:
                continue
            changed = True
        if changed:
            self._kept[hand_index] = SampledHand(hand_index, pot_chips, profit_chips, (), record)
            self._drop_unreferenced()

    def _drop_unreferenced(self) -> None:
        live = {-neg_index for heap in self._heaps.values() for _, neg_index in heap}
        for hand_index in set(self._kept) - live:
            del self._kept[hand_index]

    def selected(self) -> list[SampledHand]:
        """The kept hands in play order, each with the tags under which it was selected."""
        members = {tag: {-neg_index for _, neg_index in heap} for tag, heap in self._heaps.items()}
        return [
            replace(hand, tags=tuple(tag for tag in SAMPLE_TAGS if hand_index in members[tag]))
            for hand_index, hand in sorted(self._kept.items())
        ]


# --------------------------------------------------------------------------------------
# One session
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionTask:
    """Picklable unit of parallel work: ``num_hands`` hands of one match starting at ``first_hand``."""

    match_index: int
    session_index: int
    spec_a: AgentSpec  # labelled
    spec_b: AgentSpec
    cfg: MatchConfig
    first_hand: int
    num_hands: int

    @property
    def key(self) -> str:
        return match_key(self.spec_a.label or "", self.spec_b.label or "")


@dataclass
class SessionResult:
    match_index: int
    session_index: int
    num_hands: int
    arrays: HandArrays
    samples: list[SampledHand]
    all_hands: list[dict[str, Any]] | None
    seconds: float


class AgentCache:
    """Builds every distinct :class:`AgentSpec` once and hands out independent shallow copies.

    This follows the copy contract of ``agents/base.py``: ``reset(seed)`` rebinds all mutable
    per-match state, so ``copy.copy(prototype)`` + ``reset`` shares only the heavy read-only model.
    The prototype itself never plays. A cache lives as long as one run (one pool / one call), so
    models retrained on disk between runs are always picked up.
    """

    def __init__(self) -> None:
        self._prototypes: dict[str, Agent] = {}

    def fresh(self, spec: AgentSpec) -> Agent:
        key = json.dumps(spec.to_dict(), sort_keys=True, default=repr)
        if key not in self._prototypes:
            agent = create_agent(spec)
            agent.name = agent_label(spec)
            self._prototypes[key] = agent
        return copy.copy(self._prototypes[key])


def _summarise_hand(record: HandRecord) -> dict[str, Any]:
    """Compact per-hand numbers (A is seat 0): counts, latencies, 3-bet opportunity."""
    counts = np.zeros((2, NUM_STREETS, len(ACTION_KINDS)), dtype=np.uint16)
    latency: tuple[list[float], list[float]] = ([], [])
    three_bet, opportunity = [0, 0], [0, 0]
    preflop_raises = 0
    for event, decision in zip(record.events, record.decisions):
        counts[event.seat, event.street, KIND_INDEX[event.kind]] += 1
        latency[event.seat].append(decision.latency_ms)
        if event.street == Street.PREFLOP:
            if preflop_raises == 1:  # facing exactly one raise: the chance to 3-bet (blinds count as the first bet)
                opportunity[event.seat] += 1
                three_bet[event.seat] += event.kind in _BETTING_KINDS
            preflop_raises += event.kind in _BETTING_KINDS
    result = record.result
    return {
        "profit_chips": result.payoffs[0],
        "showdown": result.showdown,
        "ended_street": int(result.ended_street),
        "pot_chips": result.total_pot,
        "action_counts": counts,
        "three_bet": tuple(three_bet),
        "three_bet_opportunity": tuple(opportunity),
        "latency_a_ms": latency[0],
        "latency_b_ms": latency[1],
    }


def _session_hands(task: SessionTask, agents: AgentCache) -> Iterator[tuple[bool, HandRecord]]:
    """The hands of a session in layout order as ``(a_on_button, record)``, agents freshly seeded.

    Duplicate matches use :func:`~poker_arena.tournament.duplicate.play_duplicate_deals` (two tables or
    the legacy single table, per ``cfg.protocol``); without duplicate deals every hand has its own deal and
    the button alternates between A and B at one table.
    """
    cfg, key = task.cfg, task.key
    specs = (task.spec_a, task.spec_b)

    def make_agents(table: int | None) -> tuple[Agent, Agent]:
        pair = (agents.fresh(task.spec_a), agents.fresh(task.spec_b))
        for agent, spec in zip(pair, specs):
            agent.reset(agent_seed(cfg.seed, key, spec.label or "", task.session_index, table))
        return pair

    if cfg.duplicate:
        first_deal = task.first_hand // cfg.hands_per_unit
        deals = [Deal.from_seed(deal_seed(cfg.seed, key, first_deal + p)) for p in range(task.num_hands // 2)]
        for played in play_duplicate_deals(cfg.game, deals, make_agents, cfg.protocol):
            yield played.a_on_button, played.record
        return
    seat_agents, trackers = make_agents(None), (OpponentTracker(), OpponentTracker())
    for local_index in range(task.num_hands):
        hand_index = task.first_hand + local_index
        a_on_button = hand_index % 2 == 0
        deal = Deal.from_seed(deal_seed(cfg.seed, key, hand_index))
        record = play_hand(
            cfg.game, deal, 0 if a_on_button else 1, seat_agents, hand_number=local_index, trackers=trackers
        )  # fmt: skip
        yield a_on_button, record


def play_session(task: SessionTask, agents: AgentCache) -> SessionResult:
    """Play the hands of one session with fresh, freshly seeded agents (see module docstring)."""
    started = time.perf_counter()
    cfg = task.cfg
    builder = HandArraysBuilder()
    selector = SampleSelector(cfg.keep_hands)
    everything: list[dict[str, Any]] | None = [] if cfg.keep_hands.all_hands else None

    for local_index, (a_on_button, record) in enumerate(_session_hands(task, agents)):
        row = _summarise_hand(record)
        builder.add_hand(a_on_button=a_on_button, **row)
        selector.offer(task.first_hand + local_index, row["pot_chips"], row["profit_chips"], record)
        if everything is not None:
            everything.append(record.to_dict())

    samples = [replace(s, record=s.record.to_dict()) for s in selector.selected()]
    return SessionResult(
        task.match_index, task.session_index, task.num_hands, builder.build(), samples, everything,
        time.perf_counter() - started,
    )  # fmt: skip


# --------------------------------------------------------------------------------------
# Running sessions (in-process or in a spawned process pool)
# --------------------------------------------------------------------------------------
_worker_cache: AgentCache | None = None


def _init_worker() -> None:
    global _worker_cache
    _worker_cache = AgentCache()


def _run_in_worker(task: SessionTask) -> SessionResult:
    return play_session(task, _worker_cache if _worker_cache is not None else AgentCache())


def execute_tasks(tasks: Sequence[SessionTask], n_workers: int) -> Iterator[SessionResult]:
    """Run sessions and yield results as they finish (in task order when run in-process).

    ``n_workers <= 1`` (or a single task) runs in this process. Otherwise a spawn-based
    ``ProcessPoolExecutor`` is used (``spawn`` is the only start method on Windows and makes behaviour
    identical on every platform). Closing the generator early cancels the pending work.
    """
    workers = resolve_n_workers(n_workers, len(tasks))
    if workers == 1:
        cache = AgentCache()
        for task in tasks:
            yield play_session(task, cache)
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker
    ) as pool:
        futures = [pool.submit(_run_in_worker, task) for task in tasks]
        try:
            for future in as_completed(futures):
                yield future.result()
        finally:
            for future in futures:
                future.cancel()


def build_tasks(match_index: int, spec_a: AgentSpec, spec_b: AgentSpec, cfg: MatchConfig) -> list[SessionTask]:
    """The session tasks of one match (specs must be labelled)."""
    return [
        SessionTask(match_index, session_index, spec_a, spec_b, cfg, first, count)
        for session_index, (first, count) in enumerate(cfg.session_bounds())
    ]


# --------------------------------------------------------------------------------------
# Match result
# --------------------------------------------------------------------------------------
@dataclass
class MatchResult:
    """Everything recorded about one match. Statistics: ``summarize_match`` in ``summary.py``."""

    spec_a: AgentSpec
    spec_b: AgentSpec
    cfg: MatchConfig
    arrays: HandArrays
    samples: list[SampledHand]
    all_hands: list[dict[str, Any]] | None
    cpu_seconds: float  # summed session play time (excludes process start-up and waiting)
    key: str = field(init=False)

    def __post_init__(self) -> None:
        self.key = match_key(self.label_a, self.label_b)

    @property
    def label_a(self) -> str:
        return self.spec_a.label or ""

    @property
    def label_b(self) -> str:
        return self.spec_b.label or ""

    @property
    def unit_profits_bb(self) -> np.ndarray:
        """Agent A's profit in big blinds per statistical unit: per duplicate pair (sum of the two
        mirrored hands), or per hand when ``duplicate=False``. Agent B's profits are the negatives."""
        profit_bb = self.arrays.profit_chips.astype(np.float64) / self.cfg.game.big_blind
        return pair_sums(profit_bb) if self.cfg.duplicate else profit_bb


class MatchAssembler:
    """Collects the session results of one match (any arrival order) and builds the :class:`MatchResult`."""

    def __init__(self, spec_a: AgentSpec, spec_b: AgentSpec, cfg: MatchConfig) -> None:
        self.spec_a, self.spec_b, self.cfg = spec_a, spec_b, cfg
        self._sessions: dict[int, SessionResult] = {}
        self._selector = SampleSelector(cfg.keep_hands)

    def add(self, result: SessionResult) -> None:
        self._sessions[result.session_index] = result
        for hand in result.samples:
            self._selector.offer(hand.hand_index, hand.pot_chips, hand.profit_chips, hand.record)

    @property
    def complete(self) -> bool:
        return len(self._sessions) == self.cfg.num_sessions

    def build(self) -> MatchResult:
        if not self.complete:
            raise RuntimeError("match is not finished: sessions are missing")
        ordered = [self._sessions[i] for i in range(self.cfg.num_sessions)]
        all_hands = [h for s in ordered for h in s.all_hands or []] if self.cfg.keep_hands.all_hands else None
        return MatchResult(
            self.spec_a, self.spec_b, self.cfg,
            HandArrays.concatenate([s.arrays for s in ordered]),
            self._selector.selected(), all_hands, sum(s.seconds for s in ordered),
        )  # fmt: skip


# --------------------------------------------------------------------------------------
# Progress reporting and the public entry point
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Progress:
    """Snapshot handed to a progress callback after every finished session."""

    hands_done: int
    hands_total: int
    matches_done: int
    matches_total: int
    elapsed_s: float
    last_match: str  # "A vs B" of the session that just finished

    @property
    def fraction(self) -> float:
        return self.hands_done / self.hands_total if self.hands_total else 1.0


ProgressCallback = Callable[[Progress], None]


def run_match(
    spec_a: AgentSpec | str,
    spec_b: AgentSpec | str,
    cfg: MatchConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> MatchResult:
    """Play ``cfg.num_hands`` hands of ``spec_a`` against ``spec_b`` (see the module docstring).

    Results depend only on ``(spec_a, spec_b, cfg)`` and the agents' model artifacts - not on
    ``cfg.n_workers``. Raises ``ModelArtifactError`` if an agent cannot be built.
    """
    cfg = cfg or MatchConfig()
    spec_a, spec_b = labelled(spec_a), labelled(spec_b)
    tasks = build_tasks(0, spec_a, spec_b, cfg)
    assembler = MatchAssembler(spec_a, spec_b, cfg)
    started = time.perf_counter()
    hands_done = 0
    for result in execute_tasks(tasks, cfg.n_workers):
        assembler.add(result)
        hands_done += result.num_hands
        if progress is not None:
            progress(
                Progress(hands_done, cfg.num_hands, int(assembler.complete), 1,
                         time.perf_counter() - started, f"{spec_a.label} vs {spec_b.label}")
            )  # fmt: skip
    return assembler.build()
