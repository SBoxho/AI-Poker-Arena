"""Evolve a heads-up poker strategy with a genetic algorithm (fitness = profit in simulated matches).

The population of 18-gene genomes plays duplicate-deal matches against a diverse opponent pool (random,
calling station, maniac, fixed GA styles, a hall of fame of earlier champions); the best genome is then
evaluated on held-out opponents/deals with confidence intervals. Results go to ``artifacts/models/ga/``
(used by ``create_agent("ga")``) and figures to ``artifacts/figures/ga_*.png``.

    .venv/Scripts/python scripts/train_ga.py                      # full run
    .venv/Scripts/python scripts/train_ga.py --smoke              # ~1-2 min end-to-end check (separate output folder)
    .venv/Scripts/python scripts/train_ga.py --set GA_POPULATION_SIZE=60 --set N_WORKERS=8
    .venv/Scripts/python scripts/train_ga.py --evaluate-only      # refresh only the held-out evaluation of the saved model
"""

import argparse

from poker_arena import config as arena_config
from poker_arena.config import GameConfig
from poker_arena.evolution.fitness import FitnessConfig, training_pool
from poker_arena.evolution.run import (
    RunSettings,
    format_final_table,
    format_learning_summary,
    refresh_final_evaluation,
    resolve_held_out,
    run_ga_training,
)
from poker_arena.script_config import add_config_arguments, apply_cli_overrides
from poker_arena.training.ga import GAConfig

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
RANDOM_SEED = 2026  # master seed: operators, training/validation deals and agent streams derive from it
# Game (fixed-depth cash game: both stacks are reset to STARTING_STACK before every hand)
STARTING_STACK = 1000
SMALL_BLIND = 5
BIG_BLIND = 10
# Evolution
GA_POPULATION_SIZE = 40  # genomes per generation
GA_NUM_GENERATIONS = 40
GA_MUTATION_RATE = 0.20  # probability that a gene of an offspring is perturbed
GA_MUTATION_SCALE = 0.12  # std-dev of a perturbation as a fraction of the gene's range
GA_MUTATION_END_FACTOR = 0.4  # mutation scale is multiplied by this at the last generation (1.0 = constant)
GA_CROSSOVER_RATE = 0.8  # probability that an offspring recombines two parents
GA_CROSSOVER_KIND = "blx"  # "blx" (blend crossover) or "uniform"
GA_BLX_ALPHA = 0.3  # blend crossover: how far beyond the parents' interval a child may land
GA_TOURNAMENT_SIZE = 3  # contestants per parent-selection tournament (selection pressure)
GA_ELITE_COUNT = 2  # best genomes copied unchanged into the next generation
GA_IMMIGRANTS = 2  # fresh random genomes injected every generation (exploration)
GA_FINALIST_COUNT = 6  # top genomes of the last generation that are re-evaluated for the final pick
GA_FINAL_SELECTION_ROUNDS = 3  # fresh-deal evaluations of the finalists (0 = trust the last generation)
SEED_PRESETS = ["tag", "lag", "nit"]  # sensible styles inserted into the otherwise random first population
# Fitness (profit in BB/100 from simulated duplicate-deal matches)
GA_GAMES_PER_EVALUATION = 3000  # hands one genome plays per generation, shared out over all opponents
EQUITY_SAMPLES = 120  # Monte-Carlo deals per postflop equity estimate (also used when the agent is deployed)
OPPONENT_POOL = ["random", "call_station", "maniac", "nit", "tag", "lag", "passive", "maniac-ish"]
HALL_OF_FAME_SIZE = 3  # most recent distinct generation-bests that also act as opponents (co-evolution)
WORST_CASE_WEIGHT = 0.25  # fitness = (1 - w) * mean profit + w * profit against the toughest opponent
PROFIT_SATURATION = 300.0  # BB/100 above which profit against ONE opponent counts less (tanh); 0 = plain BB/100
MAX_ALL_IN_SHARE = 0.25  # regulariser: share of all-in actions tolerated before fitness is penalised
MIN_VPIP = 0.15  # regulariser: share of hands played voluntarily preflop below which fitness is penalised
DEGENERACY_PENALTY = 200.0  # BB/100 subtracted per unit of excess/shortfall of the two shares above
# Validation: the generation's best genome on fixed, independent deals against opponents that include some
# NOT in the training pool ("balanced", "bluffer" are never trained against; "teacher" is skipped loudly if unavailable)
VALIDATION_OPPONENTS = ["random", "call_station", "maniac", "balanced", "bluffer", "teacher"]
VALIDATION_PAIRS = 150  # duplicate pairs (2 hands each) per validation opponent
VALIDATE_EVERY = 1  # run the validation every k generations (and always at the end)
# Final held-out evaluation of the chosen genome (the same deals are also used for the reference presets)
FINAL_EVAL_OPPONENTS = ["random", "call_station", "maniac", "teacher", "balanced", "bluffer", "nit", "tag", "lag"]
FINAL_EVAL_PAIRS = 1500  # duplicate pairs per opponent
REFERENCE_PRESETS = ["tag", "lag", "nit"]  # fixed styles evaluated like the evolved genome, for comparison
# Execution and output
N_WORKERS = 0  # worker processes; 0 = all cores but one, 1 = single process (results do not depend on it)
OUTPUT_DIR = arena_config.MODELS_DIR / "ga"
FIGURES_DIR = arena_config.FIGURES_DIR
SMOKE_OVERRIDES = {  # applied by --smoke: tiny run into a throw-away folder (never overwrites the committed model)
    "GA_POPULATION_SIZE": 12,
    "GA_NUM_GENERATIONS": 5,
    "GA_GAMES_PER_EVALUATION": 160,
    "GA_FINALIST_COUNT": 3,
    "GA_FINAL_SELECTION_ROUNDS": 1,
    "HALL_OF_FAME_SIZE": 2,
    "VALIDATION_PAIRS": 8,
    "FINAL_EVAL_PAIRS": 40,
    "OUTPUT_DIR": arena_config.ARTIFACTS_DIR / "tmp" / "ga_smoke",
    "FIGURES_DIR": arena_config.ARTIFACTS_DIR / "tmp" / "ga_smoke_figures",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="do not train: re-run only the held-out evaluation of the model saved in OUTPUT_DIR "
        "(e.g. after the teacher changed) and update its metadata.json and figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(globals(), SMOKE_OVERRIDES)
    game = GameConfig(STARTING_STACK, SMALL_BLIND, BIG_BLIND)
    ga = GAConfig(
        population_size=GA_POPULATION_SIZE,
        generations=GA_NUM_GENERATIONS,
        mutation_rate=GA_MUTATION_RATE,
        mutation_scale=GA_MUTATION_SCALE,
        mutation_end_factor=GA_MUTATION_END_FACTOR,
        crossover_rate=GA_CROSSOVER_RATE,
        crossover_kind=GA_CROSSOVER_KIND,
        blx_alpha=GA_BLX_ALPHA,
        tournament_size=GA_TOURNAMENT_SIZE,
        elite_count=GA_ELITE_COUNT,
        immigrant_count=GA_IMMIGRANTS,
        hall_of_fame_size=HALL_OF_FAME_SIZE,
        validate_every=VALIDATE_EVERY,
        finalist_count=GA_FINALIST_COUNT,
        final_selection_rounds=GA_FINAL_SELECTION_ROUNDS,
        seed=RANDOM_SEED,
    )
    validation, validation_skipped = resolve_held_out(VALIDATION_OPPONENTS, "validation")
    final_opponents, final_skipped = resolve_held_out(FINAL_EVAL_OPPONENTS, "final evaluation")
    fitness = FitnessConfig(
        game=game,
        master_seed=RANDOM_SEED,
        training_opponents=training_pool(OPPONENT_POOL),
        hands_per_evaluation=GA_GAMES_PER_EVALUATION,
        equity_samples=EQUITY_SAMPLES,
        validation_opponents=validation,
        validation_pairs=VALIDATION_PAIRS,
        worst_case_weight=WORST_CASE_WEIGHT,
        profit_saturation=PROFIT_SATURATION,
        max_all_in_share=MAX_ALL_IN_SHARE,
        min_vpip=MIN_VPIP,
        degeneracy_penalty=DEGENERACY_PENALTY,
    )
    settings = RunSettings(
        output_dir=OUTPUT_DIR,
        figures_dir=FIGURES_DIR,
        seed_presets=tuple(SEED_PRESETS),
        final_opponents=final_opponents,
        final_pairs=FINAL_EVAL_PAIRS,
        reference_agents=tuple(REFERENCE_PRESETS),
        n_workers=N_WORKERS,
        skipped_opponents={**validation_skipped, **final_skipped},
    )
    if args.evaluate_only:
        evaluation = refresh_final_evaluation(fitness, settings)
        print("\nHeld-out evaluation (BB/100, 95% CI on duplicate pairs):")
        for line in format_final_table(evaluation):
            print(f"  {line}")
        return
    outcome = run_ga_training(ga, fitness, settings, effective_config=cfg)

    print("\nEvolved strategy:")
    for rule in outcome.best_genome.strategy_card():
        print(f"  - {rule}")
    print("\nLearning signal (held-out validation, saturated BB/100):")
    for line in format_learning_summary(outcome):
        print(f"  {line}")
    print("\nHeld-out evaluation (BB/100, 95% CI on duplicate pairs):")
    for line in format_final_table(outcome.final_evaluation):
        print(f"  {line}")


if __name__ == "__main__":
    main()
