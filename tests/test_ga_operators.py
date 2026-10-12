"""GA machinery on analytic fitness functions: selection, crossover, mutation, elitism, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from poker_arena.training.ga import (
    Bounds,
    Evaluation,
    GAConfig,
    HallOfFame,
    blx_crossover,
    evolve,
    gaussian_mutation,
    make_initial_population,
    mutation_scale_at,
    next_generation,
    population_diversity,
    tournament_select,
    uniform_crossover,
)

DIM = 6
BOUNDS = Bounds(tuple(f"g{i}" for i in range(DIM)), np.full(DIM, -1.0), np.full(DIM, 2.0))
TARGET = np.linspace(-0.5, 1.5, DIM)


def sphere(population: np.ndarray, generation: int, hall_of_fame) -> list[Evaluation]:
    """Deterministic analytic fitness: maximal (0) at TARGET, ignores the generation."""
    return [Evaluation(-float(np.sum((v - TARGET) ** 2))) for v in population]


def small_config(**overrides) -> GAConfig:
    base = dict(population_size=16, generations=12, elite_count=2, immigrant_count=1, hall_of_fame_size=2,
                finalist_count=4, final_selection_rounds=2, seed=5)  # fmt: skip
    base.update(overrides)
    return GAConfig(**base)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        GAConfig(population_size=1)
    with pytest.raises(ValueError):
        GAConfig(elite_count=20, population_size=20)
    with pytest.raises(ValueError):
        GAConfig(mutation_rate=1.5)
    with pytest.raises(ValueError):
        GAConfig(crossover_kind="magic")
    assert GAConfig().to_dict()["population_size"] == 40


def test_tournament_selection_prefers_fitter_and_k1_is_uniform() -> None:
    rng = np.random.default_rng(0)
    fitness = np.arange(20, dtype=float)
    winners_big = tournament_select(fitness, 5, 4000, rng)
    winners_one = tournament_select(fitness, 1, 4000, rng)
    assert fitness[winners_big].mean() > fitness[winners_one].mean() + 3
    assert abs(fitness[winners_one].mean() - 9.5) < 0.6


def test_crossover_and_mutation_stay_in_bounds_and_are_seeded() -> None:
    rng = np.random.default_rng(1)
    for _ in range(300):
        a, b = BOUNDS.random(rng, 2)
        for child in (blx_crossover(a, b, 0.5, BOUNDS, rng), uniform_crossover(a, b, rng),
                      gaussian_mutation(a, 1.0, 1.0, BOUNDS, rng)):  # fmt: skip
            assert np.all(child >= BOUNDS.lows) and np.all(child <= BOUNDS.highs)
    x = np.zeros(DIM)
    assert np.array_equal(
        gaussian_mutation(x, 0.5, 0.2, BOUNDS, np.random.default_rng(9)),
        gaussian_mutation(x, 0.5, 0.2, BOUNDS, np.random.default_rng(9)),
    )
    assert np.array_equal(gaussian_mutation(x, 0.0, 0.5, BOUNDS, rng), x)  # rate 0 = no mutation


def test_uniform_crossover_takes_genes_from_parents_only() -> None:
    a, b = np.zeros(DIM), np.ones(DIM)
    child = uniform_crossover(a, b, np.random.default_rng(2))
    assert set(child.tolist()) <= {0.0, 1.0}


def test_blx_alpha_zero_stays_between_parents() -> None:
    rng = np.random.default_rng(3)
    a, b = BOUNDS.random(rng, 2)
    for _ in range(100):
        child = blx_crossover(a, b, 0.0, BOUNDS, rng)
        assert np.all(child >= np.minimum(a, b) - 1e-12) and np.all(child <= np.maximum(a, b) + 1e-12)


def test_mutation_scale_schedule() -> None:
    config = small_config(mutation_scale=0.2, mutation_end_factor=0.5, generations=11)
    assert mutation_scale_at(config, 0) == pytest.approx(0.2)
    assert mutation_scale_at(config, 10) == pytest.approx(0.1)
    assert mutation_scale_at(small_config(mutation_end_factor=1.0), 7) == pytest.approx(small_config().mutation_scale)


def test_initial_population_contains_clipped_seeds_first() -> None:
    rng = np.random.default_rng(4)
    seed = np.full(DIM, 99.0)
    population = make_initial_population(8, [seed], BOUNDS, rng)
    assert population.shape == (8, DIM)
    assert np.array_equal(population[0], BOUNDS.highs)
    with pytest.raises(ValueError):
        make_initial_population(1, [seed, seed], BOUNDS, rng)


def test_next_generation_keeps_elites_and_size() -> None:
    rng = np.random.default_rng(6)
    config = small_config()
    population = BOUNDS.random(rng, config.population_size)
    fitness = sphere(population, 0, [])
    values = np.array([e.fitness for e in fitness])
    offspring = next_generation(population, values, config, 0, BOUNDS, rng)
    assert offspring.shape == population.shape
    best_two = population[np.argsort(-values)[:2]]
    assert np.array_equal(offspring[:2], best_two)
    assert np.all(offspring >= BOUNDS.lows) and np.all(offspring <= BOUNDS.highs)


def test_diversity_is_zero_for_clones_and_positive_for_random() -> None:
    rng = np.random.default_rng(7)
    assert population_diversity(np.tile(BOUNDS.random(rng, 1), (10, 1)), BOUNDS) == pytest.approx(0.0, abs=1e-12)
    assert population_diversity(BOUNDS.random(rng, 50), BOUNDS) > 0.2


def test_hall_of_fame_keeps_recent_distinct_members() -> None:
    hall = HallOfFame(2, BOUNDS)
    a, b, c = np.zeros(DIM), np.ones(DIM), np.full(DIM, 0.5)
    for vector in (a, a, b, c):
        hall.add(vector)
    assert len(hall) == 2
    assert np.array_equal(hall.vectors()[0], b) and np.array_equal(hall.vectors()[1], c)
    hall.vectors()[0][:] = 5  # copies: mutating the returned list does not corrupt the hall
    assert np.array_equal(hall.vectors()[0], b)
    empty = HallOfFame(0, BOUNDS)
    empty.add(a)
    assert len(empty) == 0


def test_evolution_improves_and_best_never_decreases_under_fixed_evaluation() -> None:
    result = evolve(small_config(), sphere, bounds=BOUNDS)
    best = [r.best_fitness for r in result.history]
    assert all(later >= earlier - 1e-12 for earlier, later in zip(best, best[1:]))  # elitism
    assert best[-1] > best[0]
    assert result.best_fitness >= best[-1] - 1e-9
    assert len(result.history) == 12
    assert np.all(result.best_vector >= BOUNDS.lows) and np.all(result.best_vector <= BOUNDS.highs)


def test_evolution_is_deterministic_and_seed_dependent() -> None:
    first = evolve(small_config(), sphere, bounds=BOUNDS)
    second = evolve(small_config(), sphere, bounds=BOUNDS)
    other = evolve(small_config(seed=6), sphere, bounds=BOUNDS)
    assert np.array_equal(first.best_vector, second.best_vector)
    assert first.history[3].mean_fitness == second.history[3].mean_fitness
    assert not np.array_equal(first.best_vector, other.best_vector)


def test_validation_callback_and_generation_log() -> None:
    seen: list[int] = []
    result = evolve(
        small_config(validate_every=4, generations=9),
        sphere,
        bounds=BOUNDS,
        validate=lambda vector, generation: {"fitness": float(generation)},
        on_generation=lambda record: seen.append(record.generation),
    )
    assert seen == list(range(9))
    validated = [r.generation for r in result.history if r.validation is not None]
    assert validated == [0, 4, 8]  # every 4th generation and always the last
    row = result.history[0].to_row(BOUNDS.names)
    assert row["validation_fitness"] == 0.0 and "mean_g0" in row and "best_g5" in row and "std_g2" in row
    assert result.history[1].to_row(BOUNDS.names)["validation_fitness"] != result.history[1].to_row(BOUNDS.names)["validation_fitness"]  # NaN
    assert result.history[0].to_dict(BOUNDS.names)["best_genes"]["g0"] == result.history[0].best_vector[0]


def test_final_selection_uses_fresh_generations_and_hall_of_fame() -> None:
    calls: list[tuple[int, int, int]] = []

    def recording(population: np.ndarray, generation: int, hall_of_fame) -> list[Evaluation]:
        calls.append((generation, len(population), len(hall_of_fame)))
        return sphere(population, generation, hall_of_fame)

    config = small_config(generations=3, finalist_count=4, final_selection_rounds=2)
    evolve(config, recording, bounds=BOUNDS)
    assert [c[0] for c in calls] == [0, 1, 2, 3, 4]  # 3 training generations + 2 fresh-deal rounds
    assert calls[3][1] == calls[4][1] == 4
    assert calls[0][2] == 0 and calls[2][2] >= 1  # the hall of fame fills up


def test_final_selection_corrects_a_noisy_last_generation() -> None:
    """A genome that only looks best on the training deals must lose the fresh-deal re-evaluation."""
    config = small_config(generations=2, population_size=6, elite_count=1, immigrant_count=1, finalist_count=3,
                          final_selection_rounds=1)  # fmt: skip

    def noisy(population: np.ndarray, generation: int, hall_of_fame) -> list[Evaluation]:
        if generation < config.generations:  # training: the first genome is lucky
            return [Evaluation(100.0 if i == 0 else float(-i)) for i in range(len(population))]
        return [Evaluation(float(-i if i else -50.0)) for i in range(len(population))]  # fresh deals disagree

    result = evolve(config, noisy, bounds=BOUNDS)
    lucky = result.finalists[-1][0]
    assert result.best_fitness > -50.0
    assert not np.array_equal(result.best_vector, lucky)
