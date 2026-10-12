"""End-to-end GA training on a tiny budget: artifacts, loading, determinism, worker independence, figures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poker_arena.agents.ga_agent import GENE_NAMES, GENE_SPECS, GeneticAgent, Genome
from poker_arena.config import REPO_ROOT, GameConfig
from poker_arena.evolution.fitness import FitnessConfig, opponent_from_name, training_pool
from poker_arena.evolution.plots import FINAL_EVALUATION_FILE, FITNESS_CURVE_FILE, GENE_TRAJECTORY_FILE, save_figures
from poker_arena.evolution.run import (
    RunSettings,
    TrainingOutcome,
    format_final_table,
    refresh_final_evaluation,
    resolve_held_out,
    run_ga_training,
)
from poker_arena.training.ga import GAConfig

SCRIPT = REPO_ROOT / "scripts" / "train_ga.py"
SMOKE_TIME_LIMIT_S = 240


def tiny_run(output_dir: Path, workers: int = 1, figures_dir: Path | None = None) -> TrainingOutcome:
    ga = GAConfig(population_size=6, generations=2, elite_count=1, immigrant_count=1, hall_of_fame_size=2,
                  finalist_count=3, final_selection_rounds=1, seed=17)  # fmt: skip
    fitness = FitnessConfig(
        game=GameConfig(),
        master_seed=17,
        training_opponents=training_pool(["random", "call_station", "nit"]),
        hands_per_evaluation=24,
        equity_samples=20,
        validation_opponents=tuple(opponent_from_name(n) for n in ("maniac", "balanced")),
        validation_pairs=3,
    )
    settings = RunSettings(
        output_dir=output_dir,
        figures_dir=figures_dir,
        seed_presets=("tag", "lag"),
        final_opponents=tuple(opponent_from_name(n) for n in ("random", "call_station", "tag")),
        final_pairs=6,
        reference_agents=("tag",),
        n_workers=workers,
        skipped_opponents={"teacher": "validation: not built yet"},
    )
    return run_ga_training(ga, fitness, settings, effective_config={"RANDOM_SEED": 17}, log=lambda message: None)


def test_tiny_evolution_runs_and_writes_every_artifact(tmp_path: Path) -> None:
    outcome = tiny_run(tmp_path / "ga", figures_dir=tmp_path / "figs")
    out = tmp_path / "ga"
    for name in ("best_genome.json", "history.csv", "history.json", "population_final.json", "hall_of_fame.json",
                 "config.json", "metadata.json"):  # fmt: skip
        assert (out / name).stat().st_size > 0, name
    for figure in (FITNESS_CURVE_FILE, GENE_TRAJECTORY_FILE, FINAL_EVALUATION_FILE):
        assert (tmp_path / "figs" / figure).stat().st_size > 1000

    history = pd.read_csv(out / "history.csv")
    assert len(history) == 2 and history["generation"].tolist() == [0, 1]
    for column in ("best_fitness", "mean_fitness", "median_fitness", "std_fitness", "validation_fitness", "diversity"):
        assert np.isfinite(history[column]).all(), column
    for gene in GENE_NAMES:
        assert {f"mean_{gene}", f"std_{gene}", f"best_{gene}"} <= set(history.columns)
    assert len(json.loads((out / "history.json").read_text())) == 2

    population = json.loads((out / "population_final.json").read_text())
    assert np.array(population["population"]).shape == (6, len(GENE_NAMES))
    assert population["fitness"] == sorted(population["fitness"], reverse=True)

    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["schema_version"] == 1 and metadata["gene_names"] == list(GENE_NAMES)
    assert [s["name"] for s in metadata["gene_specs"]] == [s.name for s in GENE_SPECS]
    assert metadata["equity_samples"] == 20 and metadata["game_config"]["big_blind"] == 10
    assert {"python", "numpy", "pokerkit"} <= set(metadata["library_versions"])
    assert metadata["timings_seconds"]["total"] > 0 and metadata["created_at"]
    final = metadata["final_evaluation"]
    assert set(final["results"]) == {"evolved", "tag"}
    assert final["skipped_opponents"] == {"teacher": "validation: not built yet"}
    population = metadata["training"]["population_validation_fitness"]
    assert set(population) == {"initial", "final"} and len(population["initial"]["genomes"]) == 6
    assert population["final"]["max"] >= population["final"]["median"]
    assert set(metadata["training"]["reference_validation_fitness"]) == {"tag", "lag"}
    for entry in final["results"]["evolved"].values():
        if isinstance(entry, dict) and "bb_per_100" in entry:
            assert entry["ci_low"] <= entry["bb_per_100"] <= entry["ci_high"]
    assert final["opponents"]["random"]["in_training_pool"] and not final["opponents"]["tag"]["in_training_pool"]

    config = json.loads((out / "config.json").read_text())
    assert config["effective_config"] == {"RANDOM_SEED": 17} and config["ga"]["population_size"] == 6

    agent = GeneticAgent.load(out)  # the artifacts are a complete, loadable model
    assert agent.genome == outcome.best_genome
    assert agent.equity_samples == 20
    assert any("evolved" in line or "tag" in line for line in format_final_table(outcome.final_evaluation))


def test_runs_with_the_same_seeds_are_identical(tmp_path: Path) -> None:
    first, second = tiny_run(tmp_path / "a"), tiny_run(tmp_path / "b")
    assert first.best_genome == second.best_genome
    assert [r.best_fitness for r in first.result.history] == [r.best_fitness for r in second.result.history]
    assert first.final_evaluation == second.final_evaluation


@pytest.mark.slow
def test_run_is_independent_of_the_worker_count(tmp_path: Path) -> None:
    serial, parallel = tiny_run(tmp_path / "serial", workers=1), tiny_run(tmp_path / "parallel", workers=2)
    assert serial.best_genome == parallel.best_genome
    assert [r.mean_fitness for r in serial.result.history] == [r.mean_fitness for r in parallel.result.history]
    assert serial.final_evaluation == parallel.final_evaluation


def test_best_genome_is_inside_the_bounds_and_selected_from_the_final_population(tmp_path: Path) -> None:
    outcome = tiny_run(tmp_path)
    assert isinstance(outcome.best_genome, Genome)
    finalists = [tuple(v) for v, _ in outcome.result.finalists]
    assert tuple(outcome.best_genome.to_vector()) in finalists
    scores = [f for _, f in outcome.result.finalists]
    assert scores == sorted(scores, reverse=True)


def test_seed_presets_start_in_the_first_population(tmp_path: Path) -> None:
    outcome = tiny_run(tmp_path)
    # elitism + fixed seeds: the first generation's statistics were computed over a population containing the presets
    assert outcome.result.history[0].diversity > 0


def test_unavailable_held_out_opponents_trigger_a_loud_warning() -> None:
    messages: list[str] = []
    specs, skipped = resolve_held_out(["random", "no_such_agent"], "validation", log=messages.append)
    assert [s.label for s in specs] == ["random"] and "no_such_agent" in skipped
    assert any("WARNING" in m and "no_such_agent" in m and "WITHOUT" in m for m in messages)


def test_figures_can_be_written_from_a_result(tmp_path: Path) -> None:
    outcome = tiny_run(tmp_path / "ga")
    paths = save_figures(
        outcome.result, outcome.final_evaluation, outcome.reference_validation, outcome.population_validation,
        tmp_path / "figures",
    )
    assert [p.name for p in paths] == [FITNESS_CURVE_FILE, GENE_TRAJECTORY_FILE, FINAL_EVALUATION_FILE]
    assert all(p.stat().st_size > 1000 for p in paths)


@pytest.mark.slow
def test_training_script_smoke_run_writes_loadable_artifacts(tmp_path: Path) -> None:
    out, figures = tmp_path / "ga", tmp_path / "figs"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--smoke", "--set", f"OUTPUT_DIR={out}", "--set", f"FIGURES_DIR={figures}",
         "--set", "N_WORKERS=2", "--set", "GA_NUM_GENERATIONS=2", "--set", "FINAL_EVAL_PAIRS=10"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=SMOKE_TIME_LIMIT_S,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "Evolved strategy" in done.stdout and "Held-out evaluation" in done.stdout
    metadata = json.loads((out / "metadata.json").read_text())
    config = json.loads((out / "config.json").read_text())["effective_config"]
    assert config["GA_NUM_GENERATIONS"] == 2 and isinstance(config["RANDOM_SEED"], int)
    assert metadata["training"]["generations"] == 2
    assert GeneticAgent.load(out).genome.to_dict().keys() == set(GENE_NAMES)
    assert (figures / FITNESS_CURVE_FILE).exists()


def test_refresh_replaces_only_the_final_evaluation(tmp_path: Path) -> None:
    outcome = tiny_run(tmp_path / "ga", figures_dir=tmp_path / "figs")
    out = tmp_path / "ga"
    before = json.loads((out / "metadata.json").read_text())
    genome_before = (out / "best_genome.json").read_text()
    fitness = FitnessConfig(
        game=GameConfig(1000, 5, 10), master_seed=17, training_opponents=training_pool(["random"]),
        hands_per_evaluation=24, equity_samples=999,  # the saved model's equity_samples (20) must win
    )  # fmt: skip
    settings = RunSettings(
        output_dir=out, figures_dir=tmp_path / "figs", seed_presets=(),
        final_opponents=tuple(opponent_from_name(n) for n in ("random", "maniac")), final_pairs=4,
        reference_agents=("lag",), n_workers=1,
    )  # fmt: skip
    evaluation = refresh_final_evaluation(fitness, settings, log=lambda message: None)
    after = json.loads((out / "metadata.json").read_text())
    assert after["final_evaluation"] == json.loads(json.dumps(evaluation))
    assert set(after["final_evaluation"]["results"]) == {"evolved", "lag"} and after["final_evaluation"]["pairs_per_opponent"] == 4
    assert before["final_evaluation"] != after["final_evaluation"] and "final_evaluation_refreshed_at" in after
    for key in ("created_at", "training", "gene_specs", "equity_samples", "game_config", "library_versions"):
        assert after[key] == before[key], key
    assert (out / "best_genome.json").read_text() == genome_before
    assert evaluation["results"]["evolved"].keys() >= {"random", "maniac", "style"}
    assert GeneticAgent.load(out).genome == outcome.best_genome
