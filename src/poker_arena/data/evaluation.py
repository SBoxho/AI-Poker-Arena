"""Duplicate-deal head-to-head evaluation of two agent specs (used to validate the teacher).

Every *pair* plays one :class:`~poker_arena.engine.deals.Deal` twice with the two players swapped
between button and big blind, so the luck of the cards cancels and the BB/100 of A against B has a
much smaller variance than in a plain match. The headline confidence interval is a Student-t interval
over per-pair results (hands inside a pair are strongly dependent, so treating hands as independent
would understate the uncertainty).

**Protocol.** Identical *stateless* agents cancel exactly in every pair under any protocol. Agents that
adapt to opponent statistics (the teacher, the trained models) only cancel if no information about a
deal can cross between its two plays. By default (``protocol="two_table"``, see
:mod:`poker_arena.tournament.duplicate`) every match is therefore played at two independent tables - own
agent copies, own opponent trackers, the same deals, seats complemented - whereas the legacy
``"single_table"`` plays both hands of a deal back-to-back against the same opponent model, which gives
the second mover of every pair an information advantage (roughly -10 to -13 BB/100 for the first-named of two
identical teachers). Keep it only to reproduce old numbers.

Caveat that the result also reports: agents that adapt to *opponent statistics* (the teacher does) make
consecutive pairs of a match dependent - a drift in the tracked statistics shifts every later hand. The
independent unit is then the **match**: pairs are played in fixed-size matches (fresh agents and opponent
trackers each), and a second, more conservative interval is computed over the per-match means. It is
also what makes the result for given seeds the same whatever the number of worker processes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import stats

from poker_arena.agents.base import Agent
from poker_arena.agents.registry import AgentSpec
from poker_arena.config import GameConfig, derive_seed
from poker_arena.engine.deals import Deal
from poker_arena.tournament.config import PROTOCOLS, TWO_TABLE
from poker_arena.tournament.duplicate import play_duplicate_deals
from poker_arena.tournament.match import AgentCache

PAIRS_PER_CHUNK = 50
"""Pairs per independent match (100 hands); fixed so that results do not depend on the worker count."""
HANDS_PER_PAIR = 2
CONFIDENCE_LEVEL = 0.95
EVAL_NAMESPACE = "eval"


@dataclass(frozen=True, slots=True)
class DuplicateResult:
    """BB/100 of agent A against agent B with a pair-level confidence interval."""

    label_a: str
    label_b: str
    n_pairs: int
    n_hands: int
    bb_per_100: float
    ci_low: float  # 95% Student-t interval over pairs
    ci_high: float
    std_error: float
    n_matches: int  # independent matches (PAIRS_PER_CHUNK pairs each, the last one may be shorter)
    match_ci_low: float  # 95% Student-t interval over per-match means (robust to adaptation drift)
    match_ci_high: float

    def to_dict(self) -> dict:
        return {
            "a": self.label_a, "b": self.label_b, "pairs": self.n_pairs, "hands": self.n_hands,
            "bb_per_100": self.bb_per_100, "ci95_pairs": [self.ci_low, self.ci_high],
            "std_error": self.std_error, "matches": self.n_matches,
            "ci95_matches": None if math.isnan(self.match_ci_low) else [self.match_ci_low, self.match_ci_high],
        }

    def __str__(self) -> str:
        return (
            f"{self.label_a} vs {self.label_b}: {self.bb_per_100:+.1f} BB/100 "
            f"pair CI [{self.ci_low:+.1f}, {self.ci_high:+.1f}], match CI "
            f"[{self.match_ci_low:+.1f}, {self.match_ci_high:+.1f}] ({self.n_hands} hands)"
        )


def _play_chunk(
    task: tuple[AgentSpec, AgentSpec, GameConfig, tuple[int | str, ...], int, int, str, str],
) -> list[float]:
    """Play ``count`` pairs starting at pair ``first``; return A's net result per pair (in bb).

    One chunk is one independent match/session: agents are built once and every table gets its own
    copy (``copy.copy`` + ``reset``, the registry's copy contract), with fresh opponent trackers.
    """
    spec_a, spec_b, config, seed_parts, first, count, namespace, protocol = task
    cache = AgentCache()

    def make_agents(table: int | None) -> tuple[Agent, Agent]:
        agents = (cache.fresh(spec_a), cache.fresh(spec_b))
        table_part = () if table is None else ("table", table)  # the legacy single table keeps the old seeds
        for seat, agent in enumerate(agents):
            agent.reset(derive_seed(namespace, *seed_parts, "agent", first, seat, *table_part))
        return agents

    deals = [Deal.from_seed(derive_seed(namespace, *seed_parts, "deal", pair)) for pair in range(first, first + count)]
    first_hand = first if protocol == TWO_TABLE else HANDS_PER_PAIR * first
    net = [0] * count
    for played in play_duplicate_deals(config, deals, make_agents, protocol, first_hand_number=first_hand):
        net[played.deal_position] += played.record.result.payoffs[0]
    return [value / config.big_blind for value in net]


def pair_results(
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    n_pairs: int,
    seed_parts: Sequence[int | str],
    config: GameConfig = GameConfig(),
    n_workers: int = 1,
    namespace: str = EVAL_NAMESPACE,
    protocol: str = TWO_TABLE,
) -> np.ndarray:
    """Net result of A (big blinds, both hands of the pair summed) for each of ``n_pairs`` deals.

    ``namespace`` is the seed namespace of the deals and agent streams: the default ``"eval"`` is the
    tournament's; model selection and tuning must pass a validation namespace so they can never see
    (and thereby overfit to) the evaluation deals. ``protocol`` is ``"two_table"`` (default) or the legacy
    ``"single_table"`` (see the module docstring).
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    parts = tuple(seed_parts)
    tasks = [
        (spec_a, spec_b, config, parts, first, min(PAIRS_PER_CHUNK, n_pairs - first), namespace, protocol)
        for first in range(0, n_pairs, PAIRS_PER_CHUNK)
    ]
    if n_workers <= 1 or len(tasks) == 1:
        chunks = [_play_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
            chunks = list(pool.map(_play_chunk, tasks))
    return np.array([value for chunk in chunks for value in chunk], dtype=np.float64)


def _t_interval(values: np.ndarray) -> tuple[float, float, float]:
    """(mean, standard error, half width of the 95% Student-t interval) of independent values."""
    n = len(values)
    std_error = float(values.std(ddof=1) / math.sqrt(n))
    return float(values.mean()), std_error, float(stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2, df=n - 1)) * std_error


def summarize(label_a: str, label_b: str, pairs: np.ndarray) -> DuplicateResult:
    """BB/100 (per hand) with a pair-level and a match-level Student-t confidence interval."""
    n = len(pairs)
    if n < 2:
        raise ValueError("Need at least two pairs for a confidence interval")
    scale = 100.0 / HANDS_PER_PAIR
    mean, std_error, half = _t_interval(pairs)
    match_means = np.array([chunk.mean() for chunk in np.split(pairs, range(PAIRS_PER_CHUNK, n, PAIRS_PER_CHUNK))])
    if len(match_means) > 1:
        _, _, match_half = _t_interval(match_means)
    else:
        match_half = float("nan")  # a single match cannot estimate between-match variation
    return DuplicateResult(
        label_a, label_b, n, n * HANDS_PER_PAIR,
        mean * scale, (mean - half) * scale, (mean + half) * scale, std_error * scale,
        len(match_means), (mean - match_half) * scale, (mean + match_half) * scale,
    )


def evaluate_duplicate(
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    n_pairs: int,
    seed_parts: Sequence[int | str],
    config: GameConfig = GameConfig(),
    n_workers: int = 1,
    namespace: str = EVAL_NAMESPACE,
    protocol: str = TWO_TABLE,
) -> DuplicateResult:
    """Play ``n_pairs`` duplicate pairs (2 x ``n_pairs`` hands) of A against B and summarise A's BB/100."""
    pairs = pair_results(spec_a, spec_b, n_pairs, seed_parts, config, n_workers, namespace, protocol)
    label_a = spec_a.label or spec_a.kind
    label_b = spec_b.label or spec_b.kind
    return summarize(label_a, label_b, pairs)
