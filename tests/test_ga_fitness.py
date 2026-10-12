"""Fitness evaluation by simulated matches: determinism, common random numbers, duplicate pairs, regulariser."""

from __future__ import annotations

import numpy as np
import pytest

from poker_arena.agents.base import Agent
from poker_arena.agents.ga_agent import PRESETS, Genome
from poker_arena.config import GameConfig
from poker_arena.evolution import fitness as fitness_module
from poker_arena.evolution.fitness import (
    TRAIN_NAMESPACE,
    FitnessConfig,
    MatchResult,
    MatchRunner,
    MatchTask,
    PokerFitness,
    StyleCounts,
    available_opponents,
    bb_per_100,
    build_opponent,
    hall_of_fame_opponent,
    opponent_from_name,
    play_match,
    resolve_workers,
    training_pool,
)
from poker_arena.tournament import duplicate as duplicate_module

GAME = GameConfig()
TINY_EQUITY_SAMPLES = 20


def tiny_config(**overrides) -> FitnessConfig:
    base = dict(
        game=GAME,
        master_seed=3,
        training_opponents=training_pool(["random", "call_station", "tag"]),
        hands_per_evaluation=24,
        equity_samples=TINY_EQUITY_SAMPLES,
        validation_opponents=tuple(opponent_from_name(n) for n in ("maniac", "balanced")),
        validation_pairs=3,
    )
    base.update(overrides)
    return FitnessConfig(**base)


def make_task(
    opponent: str = "call_station", parts=(3, 0, 0), pairs: int = 4, genome: Genome | None = None,
    protocol: str = "two_table",
) -> MatchTask:
    return MatchTask(
        game=GAME,
        hero_genome=tuple((genome or PRESETS["tag"]).to_vector().tolist()),
        opponent=opponent_from_name(opponent),
        equity_samples=TINY_EQUITY_SAMPLES,
        namespace=TRAIN_NAMESPACE,
        seed_parts=parts,
        n_pairs=pairs,
        protocol=protocol,
    )


# ---------------------------------------------------------------------------- opponents
def test_opponent_pool_rejects_trained_agents() -> None:
    for name in ("teacher", "ebm", "xgboost", "ga"):
        with pytest.raises(ValueError, match="training pool"):
            training_pool(["random", name])
    with pytest.raises(ValueError, match="empty"):
        training_pool([])
    assert [s.label for s in training_pool(["random", "nit"])] == ["random", "nit"]


def test_opponents_are_built_from_presets_and_registry_kinds() -> None:
    for name in ("random", "call_station", "maniac", "nit", "maniac-ish", "balanced"):
        agent = build_opponent(opponent_from_name(name), TINY_EQUITY_SAMPLES)
        assert isinstance(agent, Agent)
    spec = opponent_from_name("lag")
    assert spec.genome == tuple(PRESETS["lag"].to_vector().tolist())
    hof = hall_of_fame_opponent(2, PRESETS["tag"].to_vector())
    assert hof.label == "hof2" and build_opponent(hof, TINY_EQUITY_SAMPLES).genome == PRESETS["tag"]


def test_unavailable_opponents_are_reported_not_hidden() -> None:
    specs, missing = available_opponents(["random", "definitely_not_an_agent"])
    assert [s.label for s in specs] == ["random"]
    assert "definitely_not_an_agent" in missing


# ------------------------------------------------------------------------------- matches
def test_match_is_deterministic_and_seed_dependent() -> None:
    first, second = play_match(make_task()), play_match(make_task())
    assert first == second
    assert len(first.pair_profits) == 4 and first.counts.hands == 8
    assert play_match(make_task(parts=(3, 1, 0))) != first  # a different generation plays different deals
    assert play_match(make_task(parts=(4, 0, 0))) != first


@pytest.mark.parametrize("protocol", ["two_table", "single_table"])
def test_every_deal_is_played_twice_with_positions_swapped(monkeypatch: pytest.MonkeyPatch, protocol: str) -> None:
    seen: list[tuple[tuple[str, ...], int, int]] = []
    real_play_hand = duplicate_module.play_hand

    def spy(config, deal, button_seat, agents, **kwargs):
        seen.append((deal.cards, button_seat, kwargs["hand_number"]))
        return real_play_hand(config, deal, button_seat, agents, **kwargs)

    monkeypatch.setattr(duplicate_module, "play_hand", spy)
    play_match(make_task(pairs=3, protocol=protocol))
    assert len(seen) == 6
    for pair in range(3):
        (cards_a, button_a, number_a), (cards_b, button_b, number_b) = seen[2 * pair], seen[2 * pair + 1]
        assert cards_a == cards_b  # same cards ...
        if protocol == "single_table":  # ... hero on the button, then in the big blind, one game of 6 hands
            assert (button_a, button_b) == (0, 1) and (number_a, number_b) == (2 * pair, 2 * pair + 1)
        else:  # two tables of 3 hands: the hero opens on the button at even deals only
            assert (button_a, button_b) == ((0, 1) if pair % 2 == 0 else (1, 0)) and number_a == number_b == pair
    assert len({cards for cards, _, _ in seen}) == 3  # different pairs use different deals


def test_style_counts_profile() -> None:
    counts = play_match(make_task(opponent="maniac", pairs=10)).counts
    profile = counts.profile()
    assert counts.hands == 20
    assert 0.0 < profile["vpip"] <= 1.0 and profile["pfr"] <= profile["vpip"]
    assert 0.0 <= profile["all_in_share"] <= 1.0
    merged = StyleCounts()
    merged.merge(counts)
    merged.merge(counts)
    assert merged.hands == 40 and merged.profile()["vpip"] == pytest.approx(profile["vpip"])


def test_bb_per_100_arithmetic() -> None:
    assert bb_per_100([100, -50], big_blind=10) == pytest.approx(125.0)  # 50 chips = 5 BB over 4 hands
    assert bb_per_100([0, 0, 0], 10) == 0.0


# ----------------------------------------------------------------------------- fitness
def test_common_random_numbers_identical_genomes_get_identical_fitness() -> None:
    with MatchRunner(1) as runner:
        fitness = PokerFitness(tiny_config(), runner)
        tag, lag, nit = (PRESETS[n].to_vector() for n in ("tag", "lag", "nit"))
        alone = fitness.evaluate_population(np.array([tag]), generation=2, hall_of_fame=[])
        mixed = fitness.evaluate_population(np.array([lag, tag, nit, tag]), generation=2, hall_of_fame=[])
    assert mixed[1] == mixed[3] == alone[0]  # same deals/opponents for everybody, whatever the other members are


def test_fitness_changes_with_generation_and_includes_hall_of_fame_opponents() -> None:
    with MatchRunner(1) as runner:
        fitness = PokerFitness(tiny_config(), runner)
        population = np.array([PRESETS["tag"].to_vector(), PRESETS["lag"].to_vector()])
        gen0 = fitness.evaluate_population(population, 0, [])
        gen1 = fitness.evaluate_population(population, 1, [])
        with_hof = fitness.evaluate_population(population, 0, [PRESETS["nit"].to_vector()])
    assert gen0 != gen1  # fresh deals every generation
    assert "hof0" in with_hof[0].details and "hof0" not in gen0[0].details
    assert set(gen0[0].details) >= {"random", "call_station", "tag", "raw_mean_bb100", "worst_bb100", "penalty"}


def test_hands_budget_is_shared_over_opponents() -> None:
    with MatchRunner(1) as runner:
        fitness = PokerFitness(tiny_config(hands_per_evaluation=48), runner)
        opponents = fitness.config.training_opponents
        (results,) = fitness.play_against([PRESETS["tag"].to_vector()], opponents, TRAIN_NAMESPACE, 0, pairs=8)
    assert [len(r.pair_profits) for r in results] == [8, 8, 8]
    assert [r.counts.hands for r in results] == [16, 16, 16]


def test_validation_uses_fixed_deals_independent_of_generation_and_training_namespace() -> None:
    with MatchRunner(1) as runner:
        fitness = PokerFitness(tiny_config(), runner)
        genome = PRESETS["tag"].to_vector()
        first, later = fitness.validate(genome, 0), fitness.validate(genome, 9)
    assert first == later
    assert set(first["per_opponent"]) == {"maniac", "balanced"}
    cap = fitness_module.FitnessConfig.profit_saturation  # dataclass default
    assert first["fitness"] == pytest.approx(np.mean([cap * np.tanh(v / cap) for v in first["per_opponent"].values()]))
    assert first["raw_mean_bb100"] == pytest.approx(np.mean(list(first["per_opponent"].values())))


def test_saturating_utility_and_worst_case_blend() -> None:
    cfg = tiny_config(profit_saturation=300.0, worst_case_weight=0.5)
    fitness = PokerFitness(cfg, MatchRunner(1))
    assert fitness._utility(50.0) == pytest.approx(300 * np.tanh(50 / 300))
    assert fitness._utility(1e6) == pytest.approx(300.0)
    assert fitness._utility(-1e6) == pytest.approx(-300.0)
    assert PokerFitness(tiny_config(profit_saturation=0.0), MatchRunner(1))._utility(1234.0) == 1234.0


def _fake_results(profits_per_opponent: list[int], all_in_share: float, vpip: float) -> list[MatchResult]:
    results = []
    for profit in profits_per_opponent:
        counts = StyleCounts(hands=100, vpip_hands=int(100 * vpip), actions=1000, all_ins=int(1000 * all_in_share))
        results.append(MatchResult((profit,) * 5, counts))
    return results


def test_degeneracy_penalty_hits_always_shove_and_never_play_genomes() -> None:
    cfg = tiny_config(profit_saturation=0.0, worst_case_weight=0.0, max_all_in_share=0.25, min_vpip=0.15,
                      degeneracy_penalty=200.0)  # fmt: skip
    fitness = PokerFitness(cfg, MatchRunner(1))
    opponents = cfg.training_opponents
    healthy = fitness._score(_fake_results([40, 40, 40], all_in_share=0.10, vpip=0.5), opponents)
    shover = fitness._score(_fake_results([40, 40, 40], all_in_share=0.65, vpip=0.5), opponents)
    folder = fitness._score(_fake_results([40, 40, 40], all_in_share=0.10, vpip=0.05), opponents)
    assert healthy.details["penalty"] == 0.0
    assert shover.details["penalty"] == pytest.approx(200 * 0.40)
    assert folder.details["penalty"] == pytest.approx(200 * 0.10)
    assert shover.fitness < healthy.fitness and folder.fitness < healthy.fitness
    mean_profit = bb_per_100((40,) * 5, GAME.big_blind)
    assert healthy.fitness == pytest.approx(mean_profit)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        tiny_config(hands_per_evaluation=1)
    with pytest.raises(ValueError):
        tiny_config(worst_case_weight=1.5)
    with pytest.raises(ValueError):
        tiny_config(profit_saturation=-1.0)
    with pytest.raises(ValueError):
        tiny_config(equity_samples=1)


# ----------------------------------------------------------------------------- workers
def test_resolve_workers() -> None:
    assert resolve_workers(1, 100) == 1
    assert resolve_workers(8, 3) == 3
    assert resolve_workers(0, 1000) >= 1


@pytest.mark.slow
def test_results_are_identical_for_any_number_of_workers() -> None:
    tasks = [make_task(opponent=o, parts=(3, g, j), pairs=3) for g in range(2) for j, o in enumerate(("random", "tag", "maniac"))]
    with MatchRunner(1) as serial:
        reference = serial.run(tasks)
    with MatchRunner(2) as pool:
        assert pool.n_workers == 2
        parallel = pool.run(tasks)
        again = pool.run(tasks)  # the persistent pool is reusable
    assert parallel == reference == again


@pytest.mark.slow
def test_population_fitness_is_worker_count_independent() -> None:
    population = np.array([PRESETS[n].to_vector() for n in ("tag", "lag", "nit", "passive")])
    outcomes = []
    for workers in (1, 3):
        with MatchRunner(workers) as runner:
            fitness = PokerFitness(tiny_config(), runner)
            outcomes.append(fitness.evaluate_population(population, 1, [PRESETS["lag"].to_vector()]))
    assert outcomes[0] == outcomes[1]


def test_validate_population_matches_single_validation_and_is_composition_independent() -> None:
    with MatchRunner(1) as runner:
        fitness = PokerFitness(tiny_config(), runner)
        tag, lag = PRESETS["tag"].to_vector(), PRESETS["lag"].to_vector()
        both = fitness.validate_population([tag, lag])
        assert both[0] == fitness.validate(tag, 0)
        assert both[1] == fitness.validate_population([lag])[0]  # same held-out deals whatever else is validated
