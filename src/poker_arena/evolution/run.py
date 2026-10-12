"""End-to-end GA training run: evolve, evaluate on held-out opponents, write every artifact.

``scripts/train_ga.py`` only reads its configuration block and calls :func:`run_ga_training`; all the
work (and all the reproducibility bookkeeping) lives here so notebooks can reuse it.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poker_arena.agents.ga_agent import (
    BEST_GENOME_FILE,
    GA_SCHEMA_VERSION,
    GENE_NAMES,
    GENE_SPECS,
    METADATA_FILE,
    PRESETS,
    GeneticAgent,
    Genome,
)
from poker_arena.config import GameConfig, portable_json_default
from poker_arena.evolution.fitness import (
    HANDS_PER_PAIR,
    VALIDATION_NAMESPACE,
    FitnessConfig,
    MatchRunner,
    OpponentSpec,
    PokerFitness,
    StyleCounts,
    available_opponents,
)
from poker_arena.tournament.stats import scale_to_bb_per_100, t_estimate
from poker_arena.training.ga import Bounds, EvolutionResult, GAConfig, GenerationRecord, evolve

LogFn = Callable[[str], None]


def log_to_console(message: str) -> None:
    """Default logger: flushed immediately so progress is visible when output is redirected to a file."""
    print(message, flush=True)


HISTORY_CSV, HISTORY_JSON = "history.csv", "history.json"
POPULATION_FILE, HALL_OF_FAME_FILE = "population_final.json", "hall_of_fame.json"
CONFIG_FILE = "config.json"
CONFIDENCE = 0.95
FINAL_LABEL = "final"
_LOUD = "!" * 78


@dataclass(frozen=True)
class RunSettings:
    """Everything about a run that is not a GA hyper-parameter or a fitness definition."""

    output_dir: Path
    figures_dir: Path | None
    seed_presets: tuple[str, ...]  # presets inserted into the initial population
    final_opponents: tuple[OpponentSpec, ...]  # held-out opponents of the final evaluation
    final_pairs: int  # duplicate pairs per opponent in the final evaluation
    reference_agents: tuple[str, ...]  # presets evaluated on the same final deals for comparison
    n_workers: int
    skipped_opponents: Mapping[str, str] = field(default_factory=dict)  # name -> why it was left out


@dataclass(frozen=True)
class TrainingOutcome:
    result: EvolutionResult
    best_genome: Genome
    final_evaluation: dict[str, Any]
    reference_validation: dict[str, float]  # validation fitness of the seed presets (context for the curve)
    population_validation: dict[str, list[float]]  # validation fitness of every genome: "initial" and "final"
    timings: dict[str, float]
    output_dir: Path


def resolve_held_out(names: Sequence[str], purpose: str, log: LogFn = log_to_console) -> tuple[tuple[OpponentSpec, ...], dict[str, str]]:
    """Specs of the opponents in ``names`` that can be built; a loud warning for any that cannot.

    The ``teacher`` is built by another workstream and may be missing: the run continues without it
    but says so unmistakably, and the omission is recorded in the metadata.
    """
    specs, missing = available_opponents(names)
    for name, reason in missing.items():
        log(f"\n{_LOUD}\nWARNING: held-out opponent {name!r} is UNAVAILABLE for the {purpose}: {reason}\n"
            f"         The {purpose} runs WITHOUT it.\n{_LOUD}\n")
    return specs, {name: f"{purpose}: {reason}" for name, reason in missing.items()}


# --------------------------------------------------------------------------------------
# Held-out evaluation with confidence intervals
# --------------------------------------------------------------------------------------
def _estimate_dict(pair_profits: Sequence[int], big_blind: int) -> dict[str, float]:
    """BB/100 with a Student-t confidence interval computed on duplicate-pair sums."""
    per_pair_bb100 = scale_to_bb_per_100(np.asarray(pair_profits, dtype=np.float64) / big_blind, HANDS_PER_PAIR)
    estimate = t_estimate(per_pair_bb100, CONFIDENCE)
    return {
        "bb_per_100": estimate.mean,
        "se": estimate.se,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "n_pairs": estimate.n,
    }


def final_evaluation(
    fitness: PokerFitness,
    best: np.ndarray,
    training_labels: Sequence[str],
    settings: RunSettings,
) -> dict[str, Any]:
    """Evolved genome (and reference presets) against the held-out opponents on identical fresh deals."""
    big_blind = fitness.config.game.big_blind
    opponents = settings.final_opponents
    agents: dict[str, np.ndarray] = {"evolved": best}
    for name in settings.reference_agents:
        agents[name] = PRESETS[name].to_vector()
    results = fitness.play_against(
        list(agents.values()), opponents, VALIDATION_NAMESPACE, FINAL_LABEL, settings.final_pairs
    )
    table: dict[str, Any] = {}
    for (agent_name, _), per_opponent in zip(agents.items(), results, strict=True):
        counts = StyleCounts()
        entry: dict[str, Any] = {}
        for spec, match in zip(opponents, per_opponent, strict=True):
            entry[spec.label] = _estimate_dict(match.pair_profits, big_blind)
            counts.merge(match.counts)
        entry["style"] = counts.profile()
        table[agent_name] = entry
    return {
        "unit": "BB/100",
        "confidence": CONFIDENCE,
        "method": "Student-t interval on duplicate-pair sums; identical deals for every agent",
        "pairs_per_opponent": settings.final_pairs,
        "hands_per_opponent": HANDS_PER_PAIR * settings.final_pairs,
        "opponents": {spec.label: {"in_training_pool": spec.label in training_labels} for spec in opponents},
        "skipped_opponents": dict(settings.skipped_opponents),
        "results": table,
    }


def validate_populations(fitness: PokerFitness, result: EvolutionResult) -> dict[str, list[float]]:
    """Held-out validation fitness of EVERY genome of the first and the last generation.

    The best genome of a generation is a noisy maximum; the whole population's distribution shows what
    evolution actually changed (random genomes vs evolved genomes on identical deals).
    """
    if not fitness.config.validation_opponents:
        return {}
    populations = {"initial": result.initial_population, "final": result.final_population}
    return {
        name: [report["fitness"] for report in fitness.validate_population(list(population))]
        for name, population in populations.items()
    }


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------
def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "pandas", "pokerkit", "scipy", "matplotlib", "poker-arena"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, default=portable_json_default) + "\n", encoding="utf-8")


def _vector_dict(vector: np.ndarray) -> dict[str, float]:
    return dict(zip(GENE_NAMES, (float(x) for x in vector), strict=True))


def write_artifacts(
    outcome_dir: Path,
    ga: GAConfig,
    fitness_config: FitnessConfig,
    result: EvolutionResult,
    best: Genome,
    evaluation: dict[str, Any],
    reference_validation: Mapping[str, float],
    population_validation: Mapping[str, Sequence[float]],
    timings: dict[str, float],
    effective_config: Mapping[str, Any],
    n_workers: int,
) -> None:
    """Write best_genome / history / population / hall of fame / config / metadata (all small JSON/CSV)."""
    outcome_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    _write_json(
        outcome_dir / BEST_GENOME_FILE,
        {
            "schema_version": GA_SCHEMA_VERSION,
            "genes": best.to_dict(),
            "final_selection_fitness_bb100": result.best_fitness,
            "strategy": best.strategy_card(),
        },
    )
    pd.DataFrame([r.to_row(GENE_NAMES) for r in result.history]).to_csv(outcome_dir / HISTORY_CSV, index=False)
    _write_json(outcome_dir / HISTORY_JSON, [r.to_dict(GENE_NAMES) for r in result.history])
    order = np.argsort(-result.final_fitness, kind="stable")
    _write_json(
        outcome_dir / POPULATION_FILE,
        {
            "gene_names": list(GENE_NAMES),
            "population": result.final_population[order].tolist(),
            "fitness": result.final_fitness[order].tolist(),
            "finalists": [{"genes": _vector_dict(v), "final_selection_fitness": f} for v, f in result.finalists],
        },
    )
    _write_json(
        outcome_dir / HALL_OF_FAME_FILE,
        {"gene_names": list(GENE_NAMES), "members": [v.tolist() for v in result.hall_of_fame]},
    )
    _write_json(
        outcome_dir / CONFIG_FILE,
        {
            "effective_config": dict(effective_config),
            "ga": ga.to_dict(),
            "n_workers_used": n_workers,
            "seeds": {
                "master_seed": fitness_config.master_seed,
                "namespaces": {
                    "training deals": "ga_fitness / (master_seed, generation, opponent_index)",
                    "validation deals": "ga_validation / (master_seed, 'validation', opponent_index)",
                    "final evaluation deals": "ga_validation / (master_seed, 'final', opponent_index)",
                    "operators": "ga_fitness / (master_seed, 'operators')",
                },
            },
        },
    )
    last = result.history[-1]
    _write_json(
        outcome_dir / METADATA_FILE,
        {
            "schema_version": GA_SCHEMA_VERSION,
            "created_at": created_at,
            "gene_names": list(GENE_NAMES),
            "gene_specs": [spec.to_dict() for spec in GENE_SPECS],
            "equity_samples": fitness_config.equity_samples,
            "game_config": fitness_config.game.to_dict(),
            "training": {
                "generations": ga.generations,
                "population_size": ga.population_size,
                "hands_per_evaluation": fitness_config.hands_per_evaluation,
                "training_opponents": [o.label for o in fitness_config.training_opponents],
                "validation_opponents": [o.label for o in fitness_config.validation_opponents],
                "final_selection_fitness_bb100": result.best_fitness,
                "last_generation_validation": last.validation,
                "reference_validation_fitness": dict(reference_validation),
                "population_validation_fitness": {
                    name: {
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "max": float(np.max(values)),
                        "genomes": [float(v) for v in values],
                    }
                    for name, values in population_validation.items()
                },
                "fitness": {
                    "worst_case_weight": fitness_config.worst_case_weight,
                    "profit_saturation_bb100": fitness_config.profit_saturation,
                    "max_all_in_share": fitness_config.max_all_in_share,
                    "min_vpip": fitness_config.min_vpip,
                    "degeneracy_penalty_bb100": fitness_config.degeneracy_penalty,
                },
            },
            "final_evaluation": evaluation,
            "timings_seconds": timings,
            "library_versions": _library_versions(),
        },
    )


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------
def _log_generation(log: LogFn, total: int) -> Callable[[GenerationRecord], None]:
    def report(record: GenerationRecord) -> None:
        validation = "" if record.validation is None else f" | validation {record.validation['fitness']:7.1f}"
        log(
            f"gen {record.generation + 1:3d}/{total} | best {record.best_fitness:7.1f} | mean {record.mean_fitness:7.1f}"
            f" | median {record.median_fitness:7.1f} | diversity {record.diversity:.3f}{validation}"
            f" | {record.elapsed_s:6.0f}s"
        )

    return report


def run_ga_training(
    ga: GAConfig,
    fitness_config: FitnessConfig,
    settings: RunSettings,
    effective_config: Mapping[str, Any],
    log: LogFn = log_to_console,
) -> TrainingOutcome:
    """Evolve a strategy, evaluate it on held-out opponents and write ``settings.output_dir``."""
    bounds = Bounds.for_genome()
    seeds = [PRESETS[name].to_vector() for name in settings.seed_presets]
    training_labels = [o.label for o in fitness_config.training_opponents]
    started = time.perf_counter()
    with MatchRunner(settings.n_workers) as runner:
        fitness = PokerFitness(fitness_config, runner)
        log(f"GA: population {ga.population_size}, {ga.generations} generations, {fitness_config.hands_per_evaluation} "
            f"hands per genome per generation, {runner.n_workers} worker process(es)")
        log(f"training pool: {training_labels} + hall of fame ({ga.hall_of_fame_size})")
        log(f"validation opponents: {[o.label for o in fitness_config.validation_opponents]}")
        reference_validation = {
            name: fitness.validate(PRESETS[name].to_vector(), 0)["fitness"]
            for name in settings.seed_presets
            if fitness_config.validation_opponents
        }
        result = evolve(
            ga, fitness.evaluate_population, seed_vectors=seeds, validate=fitness.validate,
            bounds=bounds, on_generation=_log_generation(log, ga.generations),
        )
        evolved_at = time.perf_counter()
        log("validating the initial and the final population on the held-out deals ...")
        population_validation = validate_populations(fitness, result)
        best = Genome.from_vector(result.best_vector, clip=True)
        log(f"final held-out evaluation: {settings.final_pairs} duplicate pairs per opponent ...")
        evaluation = final_evaluation(fitness, result.best_vector, training_labels, settings)
        n_workers = runner.n_workers
    finished = time.perf_counter()
    timings = {"evolution": evolved_at - started, "final_evaluation": finished - evolved_at, "total": finished - started}
    write_artifacts(
        settings.output_dir, ga, fitness_config, result, best, evaluation, reference_validation,
        population_validation, timings, effective_config, n_workers,
    )
    if settings.figures_dir is not None:
        from poker_arena.evolution.plots import save_figures

        save_figures(result, evaluation, reference_validation, population_validation, settings.figures_dir)
    log(f"artifacts written to {settings.output_dir} (total {timings['total']:.0f}s)")
    return TrainingOutcome(
        result, best, evaluation, reference_validation, population_validation, timings, settings.output_dir
    )


def format_final_table(evaluation: Mapping[str, Any]) -> list[str]:
    """Human-readable lines: evolved vs every held-out opponent with CIs, then the reference presets."""
    lines = []
    results = evaluation["results"]
    for agent, entry in results.items():
        lines.append(f"{agent}:")
        for opponent, stats in entry.items():
            if opponent == "style":
                continue
            marker = "" if not evaluation["opponents"][opponent]["in_training_pool"] else " (training-pool opponent)"
            lines.append(
                f"  vs {opponent:<12} {stats['bb_per_100']:8.1f} BB/100  95% CI [{stats['ci_low']:8.1f}, "
                f"{stats['ci_high']:8.1f}]{marker}"
            )
    return lines


def format_learning_summary(outcome: TrainingOutcome) -> list[str]:
    """Held-out validation of the initial vs final population (the clearest evidence that evolution worked)."""
    populations = outcome.population_validation
    if not populations:
        return []
    lines = [
        f"{name} population: median {np.median(values):6.1f}, mean {np.mean(values):6.1f}, best {np.max(values):6.1f}"
        for name, values in populations.items()
    ]
    if outcome.reference_validation:
        best_preset, value = max(outcome.reference_validation.items(), key=lambda item: item[1])
        lines.append(f"best hand-made preset ({best_preset}): {value:6.1f}")
    return lines


def refresh_final_evaluation(
    fitness_config: FitnessConfig,
    settings: RunSettings,
    log: LogFn = log_to_console,
) -> dict[str, Any]:
    """Re-run ONLY the held-out evaluation of the saved model and update ``metadata.json`` + its figure.

    Training never depends on the held-out opponents, but the held-out numbers do: the ``teacher`` (built by
    another workstream) may change after training. This refreshes them in about a minute without retraining.
    The game rules and ``equity_samples`` are taken from the saved model, not from the current configuration.
    """
    agent = GeneticAgent.load(settings.output_dir)  # validates the artifacts and gives the evolved genome
    metadata_path = settings.output_dir / METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = replace(
        fitness_config, game=GameConfig.from_dict(metadata["game_config"]), equity_samples=agent.equity_samples
    )
    training_labels = [o.label for o in config.training_opponents]
    started = time.perf_counter()
    with MatchRunner(settings.n_workers) as runner:
        evaluation = final_evaluation(PokerFitness(config, runner), agent.genome.to_vector(), training_labels, settings)
    metadata["final_evaluation"] = evaluation
    metadata["final_evaluation_refreshed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    metadata["timings_seconds"]["final_evaluation_refresh"] = time.perf_counter() - started
    _write_json(metadata_path, metadata)
    if settings.figures_dir is not None:
        from poker_arena.evolution.plots import FINAL_EVALUATION_FILE, plot_final_evaluation

        settings.figures_dir.mkdir(parents=True, exist_ok=True)
        plot_final_evaluation(evaluation, settings.figures_dir / FINAL_EVALUATION_FILE)
    log(f"final evaluation refreshed in {settings.output_dir / METADATA_FILE}")
    return evaluation
