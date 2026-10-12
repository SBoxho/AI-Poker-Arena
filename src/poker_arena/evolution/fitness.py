"""Fitness of a genome = its profit when it plays simulated heads-up matches (the GA's environment).

This is orchestration code (it drives :func:`~poker_arena.engine.session.play_hand`), which is why it
lives outside ``agents/`` and ``training/`` where the architecture forbids engine imports.

Design choices that make the fitness a *low-variance, hard-to-overfit* signal
-----------------------------------------------------------------------------
* **Duplicate pairs at two tables.** Every deal is played twice with the two players swapped between
  button and big blind, so the luck of the cards cancels and seats/positions are balanced by
  construction. The two plays are made at two independent tables (own agents and opponent trackers,
  :mod:`poker_arena.tournament.duplicate`), so an adaptive genome cannot learn how the opponent played a
  deal before it plays the same cards from the other seat. (``protocol="single_table"`` is the legacy
  behaviour: both plays back-to-back against one opponent model.)
* **Common random numbers.** Within one generation every genome plays the *same* deals against the
  *same* opponents (and the same random streams), so differences in fitness are differences in
  strategy rather than in luck.
* **Fresh deals every generation** (seeds derive from the generation index), so a genome cannot
  memorise a fixed set of deals; held-out validation and final evaluation use separate seed
  namespaces/labels.
* **A diverse opponent pool** (random, calling station, maniac, fixed GA styles and a hall of fame of
  earlier champions) and a fitness that mixes the mean profit with the worst-case profit, so
  exploiting weak bots at the price of losing to solid players does not pay.
* **Deterministic and worker-independent.** Each (genome, opponent) match is a pure function of its
  task description; results are collected in task order, so any number of worker processes gives
  bit-identical numbers.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import numpy as np

from poker_arena.agents.base import Agent, ModelArtifactError
from poker_arena.agents.ga_agent import PRESETS, GeneticAgent, Genome
from poker_arena.agents.registry import create_agent
from poker_arena.config import GameConfig, derive_seed
from poker_arena.engine.deals import Deal
from poker_arena.engine.history import HandRecord
from poker_arena.engine.observation import Street
from poker_arena.tournament.config import PROTOCOLS, TWO_TABLE
from poker_arena.tournament.duplicate import play_duplicate_deals
from poker_arena.training.ga import Evaluation

GENOME_KIND = "genome"  # OpponentSpec.kind of a GA-style player defined by explicit genes
#: Agent kinds that may appear in the TRAINING pool. Trained / supervised agents (teacher, ebm, xgboost)
#: are held-out opponents: training against them would make the held-out evaluation meaningless.
TRAINING_REGISTRY_KINDS = ("random", "call_station", "maniac")
TRAIN_NAMESPACE = "ga_fitness"
VALIDATION_NAMESPACE = "ga_validation"
HANDS_PER_PAIR = 2
MAX_POOL_WORKERS = 61  # Windows limit of ProcessPoolExecutor
_RESERVED_CORES = 1
_PREFLOP_VOLUNTARY = ("call", "bet", "raise")
_AGGRESSIVE = ("bet", "raise")


# --------------------------------------------------------------------------------------
# Opponents
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OpponentSpec:
    """Picklable recipe of an opponent: a registry kind, or a GA-style player with explicit genes."""

    label: str
    kind: str
    genome: tuple[float, ...] | None = None


def opponent_from_name(name: str) -> OpponentSpec:
    """``"tag"`` -> the GA preset of that name; any other name is an agent-registry kind (``"random"``...)."""
    if name in PRESETS:
        return OpponentSpec(name, GENOME_KIND, tuple(PRESETS[name].to_vector().tolist()))
    return OpponentSpec(name, name)


def hall_of_fame_opponent(index: int, vector: np.ndarray) -> OpponentSpec:
    return OpponentSpec(f"hof{index}", GENOME_KIND, tuple(float(x) for x in vector))


def build_opponent(spec: OpponentSpec, equity_samples: int) -> Agent:
    """Instantiate the opponent (called inside the worker process)."""
    if spec.kind == GENOME_KIND:
        assert spec.genome is not None
        return GeneticAgent(Genome.from_vector(spec.genome), name=spec.label, equity_samples=equity_samples)
    return create_agent(spec.kind)


def training_pool(names: Sequence[str]) -> tuple[OpponentSpec, ...]:
    """Specs for the training pool; refuses held-out agents (see :data:`TRAINING_REGISTRY_KINDS`)."""
    specs = tuple(opponent_from_name(n) for n in names)
    for spec in specs:
        if spec.kind != GENOME_KIND and spec.kind not in TRAINING_REGISTRY_KINDS:
            raise ValueError(
                f"{spec.label!r} may not be in the GA training pool: only {TRAINING_REGISTRY_KINDS} and "
                f"GA presets {sorted(PRESETS)} are allowed (trained agents are held-out opponents)."
            )
    if not specs:
        raise ValueError("The training opponent pool is empty")
    return specs


def available_opponents(names: Sequence[str]) -> tuple[tuple[OpponentSpec, ...], dict[str, str]]:
    """Specs of the opponents that can actually be built, plus ``{name: reason}`` for those that cannot."""
    available: list[OpponentSpec] = []
    missing: dict[str, str] = {}
    for name in names:
        spec = opponent_from_name(name)
        try:
            build_opponent(spec, equity_samples=2)
        except (ModelArtifactError, KeyError, ImportError) as exc:
            missing[name] = str(exc)
        else:
            available.append(spec)
    return tuple(available), missing


# --------------------------------------------------------------------------------------
# One match (runs inside a worker process)
# --------------------------------------------------------------------------------------
@dataclass
class StyleCounts:
    """Behaviour counters of the evaluated genome (hero) - style profile and degeneracy penalty."""

    hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    actions: int = 0
    aggressive: int = 0  # bets and raises
    calls: int = 0
    folds: int = 0
    all_ins: int = 0
    showdowns: int = 0
    showdowns_won: int = 0

    def add_hand(self, record: HandRecord, seat: int = 0) -> None:
        events = [e for e in record.events if e.seat == seat]
        preflop = [e for e in events if e.street == Street.PREFLOP]
        self.hands += 1
        self.vpip_hands += any(e.kind in _PREFLOP_VOLUNTARY for e in preflop)
        self.pfr_hands += any(e.kind in _AGGRESSIVE for e in preflop)
        self.actions += len(events)
        self.aggressive += sum(e.kind in _AGGRESSIVE for e in events)
        self.calls += sum(e.kind == "call" for e in events)
        self.folds += sum(e.kind == "fold" for e in events)
        self.all_ins += sum(e.all_in for e in events)
        self.showdowns += record.result.showdown
        self.showdowns_won += record.result.showdown and record.result.payoffs[seat] > 0

    def merge(self, other: "StyleCounts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def profile(self) -> dict[str, float]:
        """Poker-style statistics (VPIP/PFR/aggression factor/...) of the evaluated genome."""
        return {
            "vpip": self._ratio(self.vpip_hands, self.hands),
            "pfr": self._ratio(self.pfr_hands, self.hands),
            "aggression_factor": self._ratio(self.aggressive, self.calls),
            "all_in_share": self._ratio(self.all_ins, self.actions),
            "fold_share": self._ratio(self.folds, self.actions),
            "showdown_rate": self._ratio(self.showdowns, self.hands),
            "showdown_win_rate": self._ratio(self.showdowns_won, self.showdowns),
            "hands": float(self.hands),
        }


@dataclass(frozen=True)
class MatchTask:
    """A self-contained, picklable description of one genome-vs-opponent match."""

    game: GameConfig
    hero_genome: tuple[float, ...]
    opponent: OpponentSpec
    equity_samples: int
    namespace: str  # seed namespace: TRAIN_NAMESPACE or VALIDATION_NAMESPACE
    seed_parts: tuple[int | str, ...]  # identifies deals and random streams (master seed, round, opponent)
    n_pairs: int
    protocol: str = TWO_TABLE  # "two_table" (default) or the legacy "single_table"


@dataclass
class MatchResult:
    """Hero's chip profit per duplicate pair (sum of its two mirrored hands) and its behaviour counters."""

    pair_profits: tuple[int, ...]
    counts: StyleCounts = field(default_factory=StyleCounts)


def play_match(task: MatchTask) -> MatchResult:
    """Play ``task.n_pairs`` duplicate pairs of hero (seat 0) against the opponent (seat 1).

    Opponent trackers persist over the table's whole game so adaptive genes see realistic statistics;
    with the default two-table protocol each of the two tables has its own hero, opponent and trackers.
    Everything is derived from ``task`` alone, hence deterministic and independent of the process.
    """
    namespace, parts = task.namespace, task.seed_parts

    def make_agents(table: int | None) -> tuple[Agent, Agent]:
        hero = GeneticAgent(Genome.from_vector(task.hero_genome), name="hero", equity_samples=task.equity_samples)
        opponent = build_opponent(task.opponent, task.equity_samples)
        table_part = () if table is None else ("table", table)  # the legacy single table keeps the old seeds
        hero.reset(derive_seed(namespace, *parts, "hero", *table_part))
        opponent.reset(derive_seed(namespace, *parts, "opponent", *table_part))
        return hero, opponent

    deals = [Deal.from_seed(derive_seed(namespace, *parts, "deal", pair)) for pair in range(task.n_pairs)]
    counts = StyleCounts()
    profits = [0] * task.n_pairs
    for played in play_duplicate_deals(task.game, deals, make_agents, task.protocol):
        profits[played.deal_position] += played.record.result.payoffs[0]
        counts.add_hand(played.record)
    return MatchResult(tuple(profits), counts)


# --------------------------------------------------------------------------------------
# Running many matches
# --------------------------------------------------------------------------------------
def resolve_workers(requested: int, n_tasks: int) -> int:
    """``requested <= 0`` means all cores but one; never more workers than tasks."""
    available = max(1, (os.cpu_count() or 1) - _RESERVED_CORES)
    wanted = available if requested <= 0 else requested
    return max(1, min(wanted, n_tasks, MAX_POOL_WORKERS))


class MatchRunner:
    """Runs match tasks serially (``n_workers == 1``) or on a persistent spawn-based process pool.

    Results always come back in task order, so the output is independent of the worker count.
    """

    def __init__(self, n_workers: int = 1) -> None:
        self.n_workers = resolve_workers(n_workers, MAX_POOL_WORKERS)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "MatchRunner":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def run(self, tasks: Sequence[MatchTask]) -> list[MatchResult]:
        if self.n_workers == 1 or len(tasks) <= 1:
            return [play_match(task) for task in tasks]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_workers, mp_context=multiprocessing.get_context("spawn")
            )
        chunk = max(1, len(tasks) // (4 * self.n_workers))
        return list(self._pool.map(play_match, tasks, chunksize=chunk))


# --------------------------------------------------------------------------------------
# Fitness
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FitnessConfig:
    """How genomes are scored."""

    game: GameConfig
    master_seed: int
    training_opponents: tuple[OpponentSpec, ...]
    hands_per_evaluation: int  # hands a genome plays per generation, shared out over all opponents
    equity_samples: int
    validation_opponents: tuple[OpponentSpec, ...] = ()
    validation_pairs: int = 50  # duplicate pairs per validation opponent
    worst_case_weight: float = 0.25  # fitness = (1 - w) * mean + w * worst opponent
    profit_saturation: float = 300.0  # BB/100 scale above which extra profit against ONE opponent is discounted (0 = off)
    max_all_in_share: float = 0.25  # all-in actions beyond this share of all actions are penalised
    min_vpip: float = 0.15  # playing fewer than this share of hands is penalised
    degeneracy_penalty: float = 200.0  # BB/100 lost per unit of excess/shortfall of the two shares above
    protocol: str = TWO_TABLE  # duplicate protocol of every match ("single_table" = legacy, diagnostics only)

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if self.hands_per_evaluation < HANDS_PER_PAIR:
            raise ValueError("hands_per_evaluation must allow at least one duplicate pair")
        if not 0.0 <= self.worst_case_weight <= 1.0:
            raise ValueError("worst_case_weight must be in [0, 1]")
        if self.profit_saturation < 0.0:
            raise ValueError("profit_saturation must be >= 0")
        if self.equity_samples < 2:
            raise ValueError("equity_samples must be >= 2")


def bb_per_100(pair_profits: Sequence[float], big_blind: int) -> float:
    """Mean profit in big blinds per 100 hands (each pair is two hands)."""
    return 100.0 * float(np.sum(pair_profits)) / (HANDS_PER_PAIR * len(pair_profits) * big_blind)


class PokerFitness:
    """Scores genomes by simulated matches; its bound methods plug into :func:`poker_arena.training.ga.evolve`."""

    def __init__(self, config: FitnessConfig, runner: MatchRunner) -> None:
        self.config = config
        self.runner = runner

    # ---- match plumbing -----------------------------------------------------------
    def _task(
        self, genome: np.ndarray, opponent: OpponentSpec, namespace: str, parts: tuple[int | str, ...], pairs: int
    ) -> MatchTask:
        return MatchTask(
            game=self.config.game,
            hero_genome=tuple(float(x) for x in genome),
            opponent=opponent,
            equity_samples=self.config.equity_samples,
            namespace=namespace,
            seed_parts=parts,
            n_pairs=pairs,
            protocol=self.config.protocol,
        )

    def play_against(
        self,
        genomes: Sequence[np.ndarray],
        opponents: Sequence[OpponentSpec],
        namespace: str,
        round_label: int | str,
        pairs: int,
    ) -> list[list[MatchResult]]:
        """``result[g][o]``: genome ``g`` vs opponent ``o``. Deals/streams depend only on (round, opponent index)."""
        tasks = [
            self._task(genome, opponent, namespace, (self.config.master_seed, round_label, j), pairs)
            for genome in genomes
            for j, opponent in enumerate(opponents)
        ]
        flat = self.runner.run(tasks)
        width = len(opponents)
        return [flat[i * width : (i + 1) * width] for i in range(len(genomes))]

    def _utility(self, bb100: float) -> float:
        """Saturating profit: ``cap * tanh(x / cap)`` is ~x near zero but flattens beyond ``cap``.

        Against calling stations and maniacs a good strategy wins thousands of BB/100, which would drown out
        every difference against solid opponents (and the noise of those huge, rare all-in swings with it).
        Saturation keeps "beat everybody" ahead of "bleed the weakest bot dry".
        """
        cap = self.config.profit_saturation
        return bb100 if cap <= 0.0 else cap * math.tanh(bb100 / cap)

    def _score(self, results: Sequence[MatchResult], opponents: Sequence[OpponentSpec]) -> Evaluation:
        cfg = self.config
        per_opponent = {
            spec.label: bb_per_100(r.pair_profits, cfg.game.big_blind) for spec, r in zip(opponents, results, strict=True)
        }
        utilities = np.array([self._utility(v) for v in per_opponent.values()])
        counts = StyleCounts()
        for result in results:
            counts.merge(result.counts)
        profile = counts.profile()
        penalty = cfg.degeneracy_penalty * (
            max(0.0, profile["all_in_share"] - cfg.max_all_in_share) + max(0.0, cfg.min_vpip - profile["vpip"])
        )
        fitness = (1.0 - cfg.worst_case_weight) * float(utilities.mean()) + cfg.worst_case_weight * float(utilities.min()) - penalty
        details = {
            "raw_mean_bb100": float(np.mean(list(per_opponent.values()))),
            "worst_bb100": float(min(per_opponent.values())),
            "penalty": penalty,
            **per_opponent,
        }
        return Evaluation(fitness, details)

    # ---- GA interface -------------------------------------------------------------
    def evaluate_population(
        self, population: np.ndarray, generation: int, hall_of_fame: Sequence[np.ndarray]
    ) -> list[Evaluation]:
        """Score every genome on the generation's shared deals against the training pool + hall of fame."""
        opponents = (*self.config.training_opponents, *(hall_of_fame_opponent(i, v) for i, v in enumerate(hall_of_fame)))
        pairs = max(1, self.config.hands_per_evaluation // (HANDS_PER_PAIR * len(opponents)))
        results = self.play_against(list(population), opponents, TRAIN_NAMESPACE, generation, pairs)
        return [self._score(per_genome, opponents) for per_genome in results]

    def validate_population(self, genomes: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        """Held-out score of every genome: fixed validation deals/opponents, never used for selection.

        The deals are identical for every call (and every generation), so scores are comparable across
        genomes and across the run. ``fitness`` is the mean saturating utility (the training-fitness scale,
        without the worst-case blend); ``per_opponent`` and ``raw_mean_bb100`` are plain BB/100.
        """
        opponents = self.config.validation_opponents
        results = self.play_against(
            genomes, opponents, VALIDATION_NAMESPACE, "validation", self.config.validation_pairs
        )
        reports = []
        for per_genome in results:
            per_opponent = {
                spec.label: bb_per_100(r.pair_profits, self.config.game.big_blind)
                for spec, r in zip(opponents, per_genome, strict=True)
            }
            reports.append(
                {
                    "fitness": float(np.mean([self._utility(v) for v in per_opponent.values()])),
                    "raw_mean_bb100": float(np.mean(list(per_opponent.values()))),
                    "per_opponent": per_opponent,
                }
            )
        return reports

    def validate(self, genome: np.ndarray, generation: int) -> dict[str, Any]:
        """Validation report of the generation's best genome (``generation`` only labels the call)."""
        (report,) = self.validate_population([genome])
        return report
