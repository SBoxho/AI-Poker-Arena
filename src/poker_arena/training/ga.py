"""A small, transparent genetic algorithm for real-valued genomes (no external dependency).

This module is *pure*: it knows nothing about poker rules, games or processes. The fitness of a
population is delegated to an injected :func:`PopulationEvaluator` (see
``poker_arena.evolution.fitness`` for the poker implementation that plays simulated matches), so the
operators here can be unit-tested with a trivial analytic fitness function.

Algorithm (one generation)
--------------------------
1. evaluate the population (the evaluator uses fresh deals every generation, see ``evolution/``);
2. record statistics and, optionally, a held-out validation score of the generation's best genome;
3. push the best genome into the hall of fame (used as an opponent by the evaluator -> mild
   co-evolution);
4. build the next population: **elitism** (best genomes copied unchanged), **tournament selection**
   of parents, **crossover** (BLX-alpha blend or uniform), **Gaussian mutation** (per-gene
   probability, step proportional to the gene's range, optionally shrinking over time), and a few
   **fresh random immigrants** that keep exploring.

Fitness values are only comparable *within* a generation (every generation plays new deals against a
new hall of fame), which is why the final answer is chosen by re-evaluating the finalists on
additional fresh deals instead of trusting a single noisy maximum ("winner's curse").
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from poker_arena.agents.ga_agent import GENE_HIGHS, GENE_LOWS, GENE_NAMES
from poker_arena.config import derive_seed

CROSSOVER_KINDS = ("blx", "uniform")
#: Two genomes closer than this (max normalised gene difference) count as the same hall-of-fame entry.
_SAME_GENOME_TOLERANCE = 1e-9


# --------------------------------------------------------------------------------------
# Configuration and interfaces
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Bounds:
    """Names and box constraints of the genes the GA operates on."""

    names: tuple[str, ...]
    lows: np.ndarray
    highs: np.ndarray

    @classmethod
    def for_genome(cls) -> "Bounds":
        """The bounds of :class:`~poker_arena.agents.ga_agent.Genome`."""
        return cls(GENE_NAMES, GENE_LOWS, GENE_HIGHS)

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def ranges(self) -> np.ndarray:
        return self.highs - self.lows

    def clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.lows, self.highs)

    def random(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """``count`` genomes drawn uniformly inside the box (shape ``(count, size)``)."""
        return rng.uniform(self.lows, self.highs, size=(count, self.size))


@dataclass(frozen=True)
class GAConfig:
    """Hyper-parameters of the evolutionary loop."""

    population_size: int = 40
    generations: int = 35
    mutation_rate: float = 0.20  # probability that a given gene of a child is perturbed
    mutation_scale: float = 0.12  # std-dev of the perturbation as a fraction of the gene's range
    mutation_end_factor: float = 0.4  # mutation_scale is multiplied by this at the last generation (1 = constant)
    crossover_rate: float = 0.8  # probability that a child is a recombination (else a copy of parent A)
    crossover_kind: str = "blx"
    blx_alpha: float = 0.3  # BLX-alpha: children may extend this fraction beyond the parents' interval
    tournament_size: int = 3
    elite_count: int = 2
    immigrant_count: int = 2  # fresh random genomes injected every generation
    hall_of_fame_size: int = 3
    validate_every: int = 1  # run the validation callback every k generations (and always at the end)
    finalist_count: int = 6  # top genomes of the last generation re-evaluated for the final choice
    final_selection_rounds: int = 3  # fresh-deal evaluations of the finalists (0 = trust the last generation)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if not (0.0 <= self.mutation_rate <= 1.0 and 0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("mutation_rate and crossover_rate must be in [0, 1]")
        if self.mutation_scale < 0.0 or self.mutation_end_factor < 0.0 or self.blx_alpha < 0.0:
            raise ValueError("mutation_scale, mutation_end_factor and blx_alpha must be >= 0")
        if self.crossover_kind not in CROSSOVER_KINDS:
            raise ValueError(f"crossover_kind must be one of {CROSSOVER_KINDS}")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        if self.elite_count < 0 or self.immigrant_count < 0 or self.hall_of_fame_size < 0:
            raise ValueError("elite_count, immigrant_count and hall_of_fame_size must be >= 0")
        if self.elite_count + self.immigrant_count >= self.population_size:
            raise ValueError("elite_count + immigrant_count must leave room for offspring")
        if self.validate_every < 1 or self.finalist_count < 1 or self.final_selection_rounds < 0:
            raise ValueError("validate_every and finalist_count must be >= 1; final_selection_rounds >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Evaluation:
    """Fitness of one genome plus the numbers it was computed from (per-opponent profits, penalties...)."""

    fitness: float
    details: dict[str, float] = field(default_factory=dict)


#: ``(population[n, genes], generation, hall_of_fame_vectors) -> n evaluations``. Within one call every
#: genome must face identical conditions (common random numbers); the generation index selects fresh deals.
PopulationEvaluator = Callable[[np.ndarray, int, Sequence[np.ndarray]], list[Evaluation]]
#: ``(best_vector, generation) -> {"fitness": float, ...}`` held-out score of the generation's best genome.
Validator = Callable[[np.ndarray, int], dict[str, Any]]


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------
def tournament_select(fitness: np.ndarray, tournament_size: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``count`` winners; each wins a tournament among ``tournament_size`` random contestants.

    Larger tournaments increase selection pressure (size 1 = random choice).
    """
    contestants = rng.integers(0, len(fitness), size=(count, tournament_size))
    best_in_tournament = np.argmax(fitness[contestants], axis=1)
    return contestants[np.arange(count), best_in_tournament]


def blx_crossover(a: np.ndarray, b: np.ndarray, alpha: float, bounds: Bounds, rng: np.random.Generator) -> np.ndarray:
    """BLX-alpha: every gene is drawn uniformly from the parents' interval widened by ``alpha`` each side."""
    low, high = np.minimum(a, b), np.maximum(a, b)
    spread = alpha * (high - low)
    return bounds.clip(rng.uniform(low - spread, high + spread))


def uniform_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Every gene is copied from parent A or parent B with equal probability."""
    return np.where(rng.random(len(a)) < 0.5, a, b)


def gaussian_mutation(
    x: np.ndarray, rate: float, scale: float, bounds: Bounds, rng: np.random.Generator
) -> np.ndarray:
    """Perturb each gene with probability ``rate`` by N(0, (scale * range)^2), clipped into the bounds."""
    mutate = rng.random(len(x)) < rate
    noise = rng.normal(0.0, 1.0, len(x)) * scale * bounds.ranges
    return bounds.clip(np.where(mutate, x + noise, x))


def mutation_scale_at(config: GAConfig, generation: int) -> float:
    """Mutation step size for ``generation``: linear decay from ``mutation_scale`` to ``* end_factor``."""
    if config.generations <= 1:
        return config.mutation_scale
    progress = min(generation, config.generations - 1) / (config.generations - 1)
    return config.mutation_scale * (1.0 - progress * (1.0 - config.mutation_end_factor))


def population_diversity(population: np.ndarray, bounds: Bounds) -> float:
    """Mean per-gene standard deviation relative to the gene range (0 = clones, ~0.29 = uniform random)."""
    return float(np.mean(np.std(population, axis=0) / bounds.ranges))


def make_initial_population(
    size: int, seeds: Sequence[np.ndarray], bounds: Bounds, rng: np.random.Generator
) -> np.ndarray:
    """Seed genomes first (clipped into bounds), the rest uniformly random."""
    if len(seeds) > size:
        raise ValueError(f"{len(seeds)} seed genomes do not fit into a population of {size}")
    population = bounds.random(rng, size)
    for i, seed in enumerate(seeds):
        population[i] = bounds.clip(np.asarray(seed, dtype=np.float64))
    return population


def next_generation(
    population: np.ndarray,
    fitness: np.ndarray,
    config: GAConfig,
    generation: int,
    bounds: Bounds,
    rng: np.random.Generator,
) -> np.ndarray:
    """Elites + offspring (selection, crossover, mutation) + random immigrants, same size as ``population``."""
    order = np.argsort(-fitness, kind="stable")
    elites = population[order[: config.elite_count]]
    n_children = config.population_size - config.elite_count - config.immigrant_count
    parents_a = population[tournament_select(fitness, config.tournament_size, n_children, rng)]
    parents_b = population[tournament_select(fitness, config.tournament_size, n_children, rng)]
    scale = mutation_scale_at(config, generation)
    children = np.empty((n_children, bounds.size))
    for i in range(n_children):
        if rng.random() < config.crossover_rate:
            if config.crossover_kind == "blx":
                child = blx_crossover(parents_a[i], parents_b[i], config.blx_alpha, bounds, rng)
            else:
                child = uniform_crossover(parents_a[i], parents_b[i], rng)
        else:
            child = parents_a[i]
        children[i] = gaussian_mutation(child, config.mutation_rate, scale, bounds, rng)
    immigrants = bounds.random(rng, config.immigrant_count)
    return np.vstack([elites, children, immigrants])


# --------------------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------------------
class HallOfFame:
    """The most recent distinct generation-best genomes; the evaluator plays them as opponents."""

    def __init__(self, capacity: int, bounds: Bounds) -> None:
        self.capacity = capacity
        self._bounds = bounds
        self._members: list[np.ndarray] = []

    def add(self, vector: np.ndarray) -> None:
        if self.capacity == 0:
            return
        if self._members:
            distance = np.max(np.abs(self._members[-1] - vector) / self._bounds.ranges)
            if distance <= _SAME_GENOME_TOLERANCE:  # an elite that keeps winning occupies one slot
                return
        self._members.append(np.array(vector, dtype=np.float64))
        del self._members[: -self.capacity]

    def vectors(self) -> list[np.ndarray]:
        return [m.copy() for m in self._members]

    def __len__(self) -> int:
        return len(self._members)


@dataclass(frozen=True)
class GenerationRecord:
    """Everything logged about one generation."""

    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    worst_fitness: float
    best_vector: tuple[float, ...]
    best_details: dict[str, float]
    diversity: float
    gene_mean: tuple[float, ...]
    gene_std: tuple[float, ...]
    hall_of_fame_size: int
    validation: dict[str, Any] | None
    elapsed_s: float

    def to_dict(self, gene_names: Sequence[str]) -> dict[str, Any]:
        """Nested, JSON-friendly representation (``history.json``)."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "median_fitness": self.median_fitness,
            "std_fitness": self.std_fitness,
            "worst_fitness": self.worst_fitness,
            "diversity": self.diversity,
            "hall_of_fame_size": self.hall_of_fame_size,
            "elapsed_s": self.elapsed_s,
            "best_genes": dict(zip(gene_names, self.best_vector, strict=True)),
            "gene_mean": dict(zip(gene_names, self.gene_mean, strict=True)),
            "gene_std": dict(zip(gene_names, self.gene_std, strict=True)),
            "best_details": self.best_details,
            "validation": self.validation,
        }

    def to_row(self, gene_names: Sequence[str]) -> dict[str, float]:
        """Flat numeric row (``history.csv``): scalars, then ``mean_<gene>``, ``std_<gene>``, ``best_<gene>``."""
        row: dict[str, float] = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "median_fitness": self.median_fitness,
            "std_fitness": self.std_fitness,
            "worst_fitness": self.worst_fitness,
            "validation_fitness": float("nan") if self.validation is None else self.validation["fitness"],
            "diversity": self.diversity,
            "hall_of_fame_size": self.hall_of_fame_size,
            "elapsed_s": self.elapsed_s,
        }
        for name, mean, std, best in zip(gene_names, self.gene_mean, self.gene_std, self.best_vector, strict=True):
            row[f"mean_{name}"] = mean
            row[f"std_{name}"] = std
            row[f"best_{name}"] = best
        return row


@dataclass(frozen=True)
class EvolutionResult:
    """Outcome of :func:`evolve`."""

    best_vector: np.ndarray
    best_fitness: float  # mean fitness of the chosen genome over the final-selection rounds
    finalists: list[tuple[np.ndarray, float]]  # (genome, final-selection fitness), best first
    history: list[GenerationRecord]
    initial_population: np.ndarray
    final_population: np.ndarray
    final_fitness: np.ndarray
    hall_of_fame: list[np.ndarray]


# --------------------------------------------------------------------------------------
# The evolutionary loop
# --------------------------------------------------------------------------------------
def evolve(
    config: GAConfig,
    evaluate_population: PopulationEvaluator,
    *,
    seed_vectors: Sequence[np.ndarray] = (),
    validate: Validator | None = None,
    bounds: Bounds | None = None,
    on_generation: Callable[[GenerationRecord], None] | None = None,
) -> EvolutionResult:
    """Run the GA. Deterministic given ``config.seed`` and a deterministic ``evaluate_population``.

    ``seed_vectors`` are inserted into the otherwise random initial population (e.g. sensible presets).
    """
    bounds = bounds or Bounds.for_genome()
    rng = np.random.default_rng(derive_seed("ga_fitness", config.seed, "operators"))
    hall = HallOfFame(config.hall_of_fame_size, bounds)
    population = make_initial_population(config.population_size, seed_vectors, bounds, rng)
    initial_population = population.copy()
    history: list[GenerationRecord] = []
    started = time.perf_counter()

    for generation in range(config.generations):
        opponents_hof = hall.vectors()
        evaluations = evaluate_population(population, generation, opponents_hof)
        fitness = np.array([e.fitness for e in evaluations], dtype=np.float64)
        best = int(np.argmax(fitness))
        is_last = generation == config.generations - 1
        validation = validate(population[best], generation) if validate and (
            is_last or generation % config.validate_every == 0
        ) else None
        record = GenerationRecord(
            generation=generation,
            best_fitness=float(fitness[best]),
            mean_fitness=float(fitness.mean()),
            median_fitness=float(np.median(fitness)),
            std_fitness=float(fitness.std()),
            worst_fitness=float(fitness.min()),
            best_vector=tuple(float(x) for x in population[best]),
            best_details=dict(evaluations[best].details),
            diversity=population_diversity(population, bounds),
            gene_mean=tuple(float(x) for x in population.mean(axis=0)),
            gene_std=tuple(float(x) for x in population.std(axis=0)),
            hall_of_fame_size=len(opponents_hof),
            validation=validation,
            elapsed_s=time.perf_counter() - started,
        )
        history.append(record)
        if on_generation is not None:
            on_generation(record)
        if is_last:
            break
        hall.add(population[best])
        population = next_generation(population, fitness, config, generation, bounds, rng)

    finalists = _select_finalists(config, evaluate_population, population, fitness, opponents_hof)
    return EvolutionResult(
        best_vector=finalists[0][0],
        best_fitness=finalists[0][1],
        finalists=finalists,
        history=history,
        initial_population=initial_population,
        final_population=population,
        final_fitness=fitness,
        hall_of_fame=opponents_hof,
    )


def _select_finalists(
    config: GAConfig,
    evaluate_population: PopulationEvaluator,
    population: np.ndarray,
    fitness: np.ndarray,
    hall_of_fame: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, float]]:
    """Top genomes of the last generation, re-ranked on fresh deals to undo the winner's curse."""
    order = np.argsort(-fitness, kind="stable")[: config.finalist_count]
    finalists = population[order]
    if config.final_selection_rounds == 0 or len(finalists) == 1:
        scores = fitness[order]
    else:
        totals = np.zeros(len(finalists))
        for round_index in range(config.final_selection_rounds):
            fresh_generation = config.generations + round_index  # never used for training deals
            totals += [e.fitness for e in evaluate_population(finalists, fresh_generation, hall_of_fame)]
        scores = totals / config.final_selection_rounds
    ranking = np.argsort(-scores, kind="stable")
    return [(finalists[i].copy(), float(scores[i])) for i in ranking]
