"""The duplicate protocols: two independent tables (default) versus the legacy single table.

Background: with ONE table the second hand of every deal is played against an opponent model that already
contains the first hand of the same deal, so two identical *adaptive* agents do not cancel - the first-named
agent (the first mover of every pair) loses about 10-13 BB/100 in A/A matches of the teacher. With TWO tables
(own agent copies and opponent trackers, same deals, seats complemented) nothing about a deal can cross
between its two plays. These tests pin the layout, the isolation, the A/A behaviour and the back-compat.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from poker_arena.agents import registry
from poker_arena.agents.base import Agent, Decision
from poker_arena.agents.baselines import CallStationAgent
from poker_arena.agents.ga_agent import PRESETS
from poker_arena.agents.registry import AgentSpec, check_availability
from poker_arena.config import GameConfig, derive_seed
from poker_arena.data.evaluation import PAIRS_PER_CHUNK, _play_chunk, pair_results
from poker_arena.engine.actions import AbstractAction
from poker_arena.engine.deals import Deal
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.engine.session import play_hand
from poker_arena.evolution.fitness import TRAIN_NAMESPACE, MatchTask, opponent_from_name, play_match
from poker_arena.tournament.config import SINGLE_TABLE, TWO_TABLE, KeepHandsPolicy, MatchConfig
from poker_arena.tournament.duplicate import play_duplicate_deals
from poker_arena.tournament.match import agent_seed, deal_seed, labelled, match_key, run_match
from poker_arena.tournament.summary import unit_samples

GAME = GameConfig()
SPY_KIND = "protocol_spy_test"
SEEDS = range(8)  # fixed seeds of the A/A experiments below
AA_HANDS = 20_000
AA_ADAPTIVE = (AgentSpec("teacher", {}, "Teacher_A"), AgentSpec("teacher", {}, "Teacher_B"))


class SpyAgent(Agent):
    """Checks/calls, and records what its opponent model contained at every decision."""

    name = "spy"
    log: list[tuple[int, int, int]] = []  # (instance id, hand number, hands in the opponent model)

    def act(self, obs) -> Decision:
        SpyAgent.log.append((id(self), obs.hand_number, obs.opponent_stats.hands))
        return Decision.from_abstract(obs, AbstractAction.CHECK_CALL)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> type[SpyAgent]:
    monkeypatch.setitem(registry._FACTORIES, SPY_KIND, SpyAgent)
    SpyAgent.log = []
    return SpyAgent


def hands_seen_per_instance(log: list[tuple[int, int, int]]) -> dict[int, list[int]]:
    """For every spy instance the opponent-model size at the first decision of each of its hands."""
    first: dict[tuple[int, int], int] = {}
    for instance, hand_number, model_hands in log:
        first.setdefault((instance, hand_number), model_hands)
    per_instance: dict[int, list[int]] = {}
    for (instance, _), model_hands in first.items():
        per_instance.setdefault(instance, []).append(model_hands)
    return per_instance


def spy_pair_factory(table):  # noqa: ANN001 - (table: int | None) -> (SpyAgent, CallStationAgent)
    return SpyAgent(), CallStationAgent()


# ----------------------------------------------------------------------------- layout
def test_two_table_layout_alternates_the_button_at_every_table() -> None:
    deals = [Deal.from_seed(i) for i in range(6)]
    hands = list(play_duplicate_deals(GAME, deals, spy_pair_factory, TWO_TABLE))
    assert [h.deal_position for h in hands] == [p for p in range(6) for _ in range(2)]
    assert [h.record.deal for h in hands] == [d for d in deals for _ in range(2)]  # adjacent hands = one deal
    assert [h.a_on_button for h in hands] == [True, False, False, True] * 3  # A opens on the button at even deals only
    assert [h.record.button_seat for h in hands] == [0 if h.a_on_button else 1 for h in hands]
    table0, table1 = hands[0::2], hands[1::2]
    assert [h.a_on_button for h in table0] == [True, False] * 3  # each table is a normal game: A, B, A, B ...
    assert [h.a_on_button for h in table1] == [False, True] * 3  # table 1 is the exact complement
    assert [h.record.hand_number for h in hands] == [p for p in range(6) for _ in range(2)]


def test_single_table_layout_is_the_legacy_one() -> None:
    deals = [Deal.from_seed(i) for i in range(4)]
    hands = list(play_duplicate_deals(GAME, deals, spy_pair_factory, SINGLE_TABLE, first_hand_number=10))
    assert [h.a_on_button for h in hands] == [True, False] * 4  # A is the first mover of every pair
    assert [h.record.hand_number for h in hands] == list(range(10, 18))
    assert [h.deal_position for h in hands] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="protocol"):
        list(play_duplicate_deals(GAME, [Deal.from_seed(0)], spy_pair_factory, "three_table"))
    with pytest.raises(ValueError, match="protocol"):
        MatchConfig(protocol="three_table")


# ----------------------------------------------------------------------------- isolation
def test_tables_have_their_own_agents_and_opponent_statistics(spy) -> None:
    n_deals = 5
    deals = [Deal.from_seed(i) for i in range(n_deals)]
    list(play_duplicate_deals(GAME, deals, spy_pair_factory, TWO_TABLE))
    per_instance = hands_seen_per_instance(spy.log)
    assert len(per_instance) == 2  # one spy per table ...
    for sizes in per_instance.values():
        assert sorted(sizes) == list(range(n_deals))  # ... each only knows the hands of its own table
    # the two plays of every deal meet an opponent model of identical size: nothing crossed over
    assert Counter(size for sizes in per_instance.values() for size in sizes) == Counter({p: 2 for p in range(n_deals)})


def test_single_table_leaks_the_first_play_of_a_deal_into_the_second(spy) -> None:
    n_deals = 5
    list(play_duplicate_deals(GAME, [Deal.from_seed(i) for i in range(n_deals)], spy_pair_factory, SINGLE_TABLE))
    (sizes,) = hands_seen_per_instance(spy.log).values()  # one shared agent
    assert sorted(sizes) == list(range(2 * n_deals))  # the second play of deal p knows the first (size 2p + 1)


def test_a_table_never_reuses_the_trackers_of_the_other(spy) -> None:
    made = []

    def factory(table):
        made.append(table)
        return spy_pair_factory(table)

    list(play_duplicate_deals(GAME, [Deal.from_seed(0)], factory, TWO_TABLE))
    assert made == [0, 1]
    made.clear()
    list(play_duplicate_deals(GAME, [Deal.from_seed(0)], factory, SINGLE_TABLE))
    assert made == [None]


def test_evaluation_chunks_play_two_independent_tables(spy) -> None:
    pairs = 6
    task = (AgentSpec(SPY_KIND), AgentSpec("call_station"), GAME, ("test", "spy"), 0, pairs, "test")
    _play_chunk((*task, TWO_TABLE))
    two_tables = hands_seen_per_instance(spy.log)
    spy.log.clear()
    _play_chunk((*task, SINGLE_TABLE))
    one_table = hands_seen_per_instance(spy.log)
    assert [sorted(s) for s in two_tables.values()] == [list(range(pairs))] * 2
    assert [sorted(s) for s in one_table.values()] == [list(range(2 * pairs))]


def test_ga_fitness_matches_play_two_independent_tables(spy) -> None:
    def task(protocol: str) -> MatchTask:
        return MatchTask(
            game=GAME, hero_genome=tuple(PRESETS["tag"].to_vector().tolist()), opponent=opponent_from_name(SPY_KIND),
            equity_samples=20, namespace=TRAIN_NAMESPACE, seed_parts=(3, 0, 0), n_pairs=5, protocol=protocol,
        )  # fmt: skip

    play_match(task(TWO_TABLE))
    two_tables = hands_seen_per_instance(spy.log)  # the hero may fold before the spy ever acts: subsets only
    assert len(two_tables) == 2 and all(set(sizes) <= set(range(5)) for sizes in two_tables.values())
    spy.log.clear()
    play_match(task(SINGLE_TABLE))
    (one_table,) = hands_seen_per_instance(spy.log).values()
    assert set(one_table) <= set(range(10)) and max(one_table) >= 5  # the opponent model spans both plays


def test_table_seeds_are_distinct_and_the_legacy_seed_is_unchanged() -> None:
    key = match_key("A", "B")
    seeds = {agent_seed(7, key, "A", 3, table) for table in (0, 1)}
    assert len(seeds) == 2 and agent_seed(7, key, "A", 3) not in seeds
    assert agent_seed(7, key, "A", 3) == derive_seed("eval", 7, key, "agent", "A", 3)  # what old runs used


def test_stochastic_agents_draw_different_numbers_at_the_two_tables() -> None:
    """Both tables must not replay the same random stream (that would make them copies of each other)."""
    result = run_match("random", "maniac", MatchConfig(num_hands=60, session_hands=60, seed=3, keep_hands=KeepHandsPolicy(0, 0, 0)))
    table0, table1 = result.arrays.action_counts[0::2], result.arrays.action_counts[1::2]
    assert not np.array_equal(table0, table1)


# ----------------------------------------------------------------------------- cancellation
@pytest.mark.parametrize("protocol", [TWO_TABLE, SINGLE_TABLE])
def test_stateless_agents_cancel_exactly_in_evaluation_pairs(protocol: str) -> None:
    spec = AgentSpec("call_station")
    pairs = pair_results(spec, spec, 30, ("test", "mirror"), GAME, protocol=protocol)
    assert len(pairs) == 30 and np.all(pairs == 0.0)


def test_evaluation_protocols_are_worker_independent_and_differ_for_adaptive_agents() -> None:
    kwargs = dict(spec_a=AgentSpec("teacher"), spec_b=AgentSpec("maniac"), n_pairs=2 * PAIRS_PER_CHUNK + 10,
                  seed_parts=("test", "workers"), config=GAME)  # fmt: skip
    two = pair_results(**kwargs, protocol=TWO_TABLE)
    assert np.array_equal(two, pair_results(**kwargs, n_workers=2, protocol=TWO_TABLE))
    one = pair_results(**kwargs, protocol=SINGLE_TABLE)
    assert np.array_equal(one, pair_results(**kwargs, n_workers=2, protocol=SINGLE_TABLE))
    assert not np.array_equal(one, two)
    with pytest.raises(ValueError, match="protocol"):
        pair_results(**kwargs, protocol="three_table")


def test_adaptive_deterministic_agents_cancel_exactly_with_two_tables() -> None:
    """Table 1 is table 0 with the agents' roles swapped, so identical deterministic agents - adaptive or
    not - mirror each other exactly; the single table breaks this."""
    teacher = AgentSpec("teacher")
    two = pair_results(teacher, teacher, 60, ("test", "adaptive-mirror"), GAME, protocol=TWO_TABLE)
    one = pair_results(teacher, teacher, 60, ("test", "adaptive-mirror"), GAME, protocol=SINGLE_TABLE)
    assert np.all(two == 0.0)
    assert np.any(one != 0.0)


def play_reference_single_table(spec_a, spec_b, cfg: MatchConfig) -> np.ndarray:
    """The pre-two-table algorithm written out independently: shared agents and trackers, A first."""
    a, b = labelled(spec_a), labelled(spec_b)
    key = match_key(a.label, b.label)
    out = []
    for session, (first, count) in enumerate(cfg.session_bounds()):
        agents = (registry.create_agent(a), registry.create_agent(b))
        for agent, spec in zip(agents, (a, b)):
            agent.reset(derive_seed("eval", cfg.seed, key, "agent", spec.label, session))
        trackers = (OpponentTracker(), OpponentTracker())
        for local in range(count):
            hand = first + local
            deal = Deal.from_seed(deal_seed(cfg.seed, key, hand // 2))
            record = play_hand(cfg.game, deal, hand % 2, agents, hand_number=local, trackers=trackers)
            out.append(record.result.payoffs[0])
    return np.array(out)


def test_the_legacy_protocol_reproduces_the_old_algorithm_bit_for_bit() -> None:
    cfg = MatchConfig(num_hands=80, session_hands=40, seed=5, protocol=SINGLE_TABLE, keep_hands=KeepHandsPolicy(0, 0, 0))
    for spec_a, spec_b in (("random", "maniac"), ("teacher", "call_station")):
        result = run_match(spec_a, spec_b, cfg)
        assert np.array_equal(result.arrays.profit_chips, play_reference_single_table(spec_a, spec_b, cfg))
        assert result.arrays.a_on_button.tolist() == [True, False] * 40


# ----------------------------------------------------------------------------- A/A experiments (slow)
def pooled_aa(specs: tuple[AgentSpec, AgentSpec], protocol: str, hands: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Per-session BB/100 and per-pair BB (first-named agent) pooled over the fixed seeds, plus per-seed means."""
    sessions, pairs, per_seed = [], [], []
    for seed in SEEDS:
        cfg = MatchConfig(num_hands=hands, seed=seed, n_workers=0, protocol=protocol, keep_hands=KeepHandsPolicy(0, 0, 0))
        result = run_match(*specs, cfg)
        sessions.append(unit_samples(result.arrays.profit_chips, cfg).session)
        pairs.append(result.unit_profits_bb)
        per_seed.append(float(sessions[-1].mean()))
    return np.concatenate(sessions), np.concatenate(pairs), per_seed


def z_score(values: np.ndarray) -> tuple[float, float, float]:
    mean, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, se, (mean / se if se > 0 else 0.0)


@pytest.mark.slow
def test_identical_adaptive_teachers_cancel_with_two_tables() -> None:
    """A/A with the adaptive teacher: 8 seeds x 20,000 hands (800 sessions); the first-named side must be at 0."""
    sessions, pairs, per_seed = pooled_aa(AA_ADAPTIVE, TWO_TABLE, AA_HANDS)
    mean, se, z = z_score(sessions)
    assert len(sessions) == 800 and abs(mean) <= 2 * se + 1e-9, (mean, se, z)
    assert np.all(pairs == 0.0)  # the teacher is deterministic: the tables mirror each other EXACTLY
    assert all(m == 0.0 for m in per_seed)


@pytest.mark.slow
def test_single_table_shows_the_first_mover_artifact_for_identical_teachers() -> None:
    """Regression pin of the bug: with one shared table the first-named teacher loses (pooled z < -2)."""
    sessions, pairs, per_seed = pooled_aa(AA_ADAPTIVE, SINGLE_TABLE, AA_HANDS)
    mean, se, z = z_score(sessions)
    assert z < -2.0, (mean, se, z)  # measured: -9.5 BB/100, SE 3.8, z = -2.5, 7 of 8 seeds negative
    assert sum(m < 0 for m in per_seed) >= 6
    assert z_score(pairs)[2] < -2.0  # and the (too narrow) pair-level view agrees


@pytest.mark.slow
@pytest.mark.skipif(not check_availability("ga").available, reason="trained GA agent not available")
def test_identical_stochastic_adaptive_agents_are_unbiased_with_two_tables() -> None:
    """A/A with a stochastic AND adaptive agent (GA: bluff coin flips + opponent statistics)."""
    specs = (AgentSpec("ga", {}, "GA_A"), AgentSpec("ga", {}, "GA_B"))
    sessions, _, _ = pooled_aa(specs, TWO_TABLE, 4_000)
    mean, se, z = z_score(sessions)
    assert se > 0 and abs(mean) <= 2 * se, (mean, se, z)  # measured: -5.6 BB/100, SE 5.9, z = -1.0
