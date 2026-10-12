"""Train the Explainable Boosting Machine (EBM) poker agent on the teacher-labelled decision dataset.

    .venv/Scripts/python scripts/train_ebm.py                  # standard preset (~14 min on 10 cores)
    .venv/Scripts/python scripts/train_ebm.py --smoke          # under a minute end-to-end check (separate output folder)
    .venv/Scripts/python scripts/train_ebm.py --set EBM_INTERACTIONS=0 --set DAGGER_ROUNDS=0

Writes the best DAgger round to ``artifacts/models/ebm/`` (``ebm.joblib``, ``metadata.json``, ``config.json``; used by
``create_agent("ebm")``) and static figures to ``artifacts/figures/ebm_*.png``. Needs the dataset made by
``scripts/generate_dataset.py``. All logic lives in :mod:`poker_arena.training.ebm`; this file holds the configuration.

Imitation accuracy is NOT poker skill: the summary prints agreement with the teacher and duplicate-deal BB/100
(validation seeds, never the tournament's ``eval`` seeds) separately.
"""

from __future__ import annotations

import argparse

from poker_arena import config as arena_config
from poker_arena.config import GameConfig
from poker_arena.script_config import add_config_arguments, apply_cli_overrides
from poker_arena.training.common import DaggerSettings
from poker_arena.training.ebm import EBMParams, EBMTrainSettings, ValidationSettings, train_ebm

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
RANDOM_SEED = 42  # master seed: row subsampling, EBM bagging, DAgger / validation hands derive from it
# Game (must equal the settings the dataset was generated with; checked at start)
STARTING_STACK = 1000
SMALL_BLIND = 5
BIG_BLIND = 10
# Data
DATASET_NAME = "teacher_bc_v2"  # file stem under DATASETS_DIR (made by scripts/generate_dataset.py)
DATASETS_DIR = arena_config.DATASETS_DIR
MAX_TRAIN_ROWS = 150_000  # whole hands are kept until this many rows (None = all); the behaviour-cloning part only
CLASS_WEIGHT_POWER = 0.5  # sample weight = (n / (K * class count)) ** power; 0 = none, 1 = fully balanced
# EBM (InterpretML ExplainableBoostingClassifier; tuned on the validation split, see artifacts/models/ebm/README_training.md)
EBM_MAX_ROUNDS = 300  # boosting rounds cap per stage (one round = one step for every term)
EBM_LEARNING_RATE = 0.1
EBM_INTERACTIONS = 40  # pairwise terms picked by InterpretML's FAST ranking (0 = main effects only, fully "flat")
EBM_OUTER_BAGS = 10  # sub-models averaged into the final shape functions (one per CPU thread is free parallelism)
EBM_INNER_BAGS = 0
EBM_MAX_BINS = 128  # bins per feature for the single-feature shape functions
EBM_MAX_INTERACTION_BINS = 24  # bins per feature inside the pairwise tables (the main driver of the file size)
EBM_MIN_SAMPLES_LEAF = 4
EBM_MAX_LEAVES = 2  # 2 = one cut per boosting step (the classic EBM)
EBM_EARLY_STOPPING_ROUNDS = 50  # stop a stage when the held-out loss has not improved for this many rounds
EBM_VALIDATION_FRACTION = 0.15  # share of the training HANDS each outer bag holds out for early stopping
N_JOBS = 10  # CPU threads for fitting (this script never uses more)
# DAgger: after behaviour cloning (round 0) the EBM plays, the teacher relabels the visited states, the EBM is refit
DAGGER_ROUNDS = 3  # 0 = plain behaviour cloning
DAGGER_HANDS_PER_ROUND = 10_000  # hands the current EBM plays per round (about 3 labelled decisions per hand)
DAGGER_TEACHER_BETA = 0.5  # chance that the teacher (not the EBM) acts per decision in round 1; halves every round
DAGGER_N_WORKERS = 10  # worker processes for the on-policy hands
# Held-out validation of EVERY round (validation seeds; the tournament's "eval" seeds are never touched)
VALIDATION_OPPONENTS = ["teacher", "call_station", "maniac", "random"]
VALIDATION_PAIRS = 500  # duplicate-deal pairs (two hands each) per opponent; 500 = 10 independent matches
VALIDATION_ON_POLICY_HANDS = 4_000  # hands the EBM plays to collect fresh states for the covariate-shift measurement
EVAL_N_WORKERS = 10  # worker processes for the duplicate-deal matches (results do not depend on this number)
IMPORTANCE_SAMPLE_ROWS = 20_000  # validation rows for the global term importance stored in metadata.json
# Output
OUTPUT_DIR = arena_config.MODELS_DIR / "ebm"  # the committed model
WORK_DIR = arena_config.ARTIFACTS_DIR / "tmp" / "ebm_rounds"  # every round's candidate model (git-ignored)
KEEP_ROUND_MODELS = False  # keep WORK_DIR after the best round was copied to OUTPUT_DIR
FIGURES_DIR = arena_config.FIGURES_DIR
SAVE_FIGURES = True
SMOKE_OVERRIDES = {  # applied by --smoke: tiny run into a throw-away folder (never overwrites the committed model)
    "MAX_TRAIN_ROWS": 6_000,
    "EBM_MAX_ROUNDS": 40,
    "EBM_LEARNING_RATE": 0.1,
    "EBM_INTERACTIONS": 6,
    "EBM_OUTER_BAGS": 4,
    "EBM_MAX_BINS": 64,
    "EBM_MAX_INTERACTION_BINS": 16,
    "EBM_EARLY_STOPPING_ROUNDS": 10,
    "DAGGER_ROUNDS": 1,
    "DAGGER_HANDS_PER_ROUND": 300,
    "VALIDATION_PAIRS": 20,
    "VALIDATION_ON_POLICY_HANDS": 150,
    "IMPORTANCE_SAMPLE_ROWS": 2_000,
    "OUTPUT_DIR": arena_config.ARTIFACTS_DIR / "tmp" / "ebm_smoke",
    "WORK_DIR": arena_config.ARTIFACTS_DIR / "tmp" / "ebm_smoke_rounds",
    "FIGURES_DIR": arena_config.ARTIFACTS_DIR / "tmp" / "ebm_smoke_figures",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    return parser.parse_args()


def build_settings() -> EBMTrainSettings:
    return EBMTrainSettings(
        dataset_name=DATASET_NAME,
        game=GameConfig(STARTING_STACK, SMALL_BLIND, BIG_BLIND),
        seed=RANDOM_SEED,
        params=EBMParams(
            max_rounds=EBM_MAX_ROUNDS,
            learning_rate=EBM_LEARNING_RATE,
            interactions=EBM_INTERACTIONS,
            outer_bags=EBM_OUTER_BAGS,
            inner_bags=EBM_INNER_BAGS,
            max_bins=EBM_MAX_BINS,
            max_interaction_bins=EBM_MAX_INTERACTION_BINS,
            min_samples_leaf=EBM_MIN_SAMPLES_LEAF,
            max_leaves=EBM_MAX_LEAVES,
            early_stopping_rounds=EBM_EARLY_STOPPING_ROUNDS,
            validation_fraction=EBM_VALIDATION_FRACTION,
        ),
        max_train_rows=MAX_TRAIN_ROWS,
        class_weight_power=CLASS_WEIGHT_POWER,
        n_jobs=N_JOBS,
        dagger=DaggerSettings(
            rounds=DAGGER_ROUNDS,
            hands_per_round=DAGGER_HANDS_PER_ROUND,
            teacher_beta=DAGGER_TEACHER_BETA,
            n_workers=DAGGER_N_WORKERS,
            seed=RANDOM_SEED,
        ),
        validation=ValidationSettings(
            opponents=tuple(VALIDATION_OPPONENTS),
            pairs=VALIDATION_PAIRS,
            on_policy_hands=VALIDATION_ON_POLICY_HANDS,
            importance_sample_rows=IMPORTANCE_SAMPLE_ROWS,
            n_workers=EVAL_N_WORKERS,
        ),
        output_dir=OUTPUT_DIR,
        work_dir=WORK_DIR,
        figures_dir=FIGURES_DIR if SAVE_FIGURES else None,
        keep_round_models=KEEP_ROUND_MODELS,
        datasets_dir=DATASETS_DIR,
    )


def format_summary(metadata: dict) -> list[str]:
    """Plain-text report: imitation (teacher agreement) and poker performance, kept strictly apart."""
    lines = ["", "IMITATION of the teacher (legal-masked agreement; NOT poker skill)"]
    lines.append(f"  {'round':>5} {'rows':>8} {'fit s':>6} | {'BC-val acc':>10} {'macro-F1':>9} | {'on-policy acc':>13} {'macro-F1':>9}")
    for row in metadata["dagger"]:
        bc, live = row["bc_validation"], row["on_policy_validation"]
        marker = "*" if row["selected"] else " "
        lines.append(
            f"  {row['round']:>4}{marker} {row['n_train_rows']:>8,} {row['fit_seconds']:>6.0f} | "
            f"{bc['masked_accuracy']:>10.4f} {bc['masked_macro_f1']:>9.4f} | "
            f"{live['masked_accuracy']:>13.4f} {live['masked_macro_f1']:>9.4f}"
        )
    opponents = list(metadata["policy_validation"]["opponents"])
    lines += ["", "POKER PERFORMANCE: validation duplicate-deal BB/100 [95% CI over pairs] (* = selected round)"]
    lines.append("  " + " " * 6 + "".join(f"{name:>26}" for name in opponents) + f"{'mean':>10}")
    for row in metadata["dagger"]:
        cells = ""
        for name in opponents:
            result = row["policy_validation"][name]
            low, high = result["ci95_pairs"]
            cells += f"{result['bb_per_100']:>+10.1f} [{low:>+6.0f},{high:>+6.0f}]  "
        marker = "*" if row["selected"] else " "
        lines.append(f"  {row['round']:>4}{marker} {cells}{row['mean_bb_per_100']:>+9.1f}")
    final = metadata["imitation"]
    lines += [
        "",
        f"Selected round {metadata['selected_round']}: {metadata['n_terms']} terms ({metadata['n_interaction_terms']} interactions), "
        f"{metadata['model_file_bytes'] / 1e6:.1f} MB, fit {metadata['training_time_s']:.0f} s "
        f"(whole pipeline {metadata['total_pipeline_time_s']:.0f} s)",
        f"  validation (legal-masked): accuracy {final['masked_accuracy']:.4f}, macro-F1 {final['masked_macro_f1']:.4f}, "
        f"balanced accuracy {final['masked_balanced_accuracy']:.4f}, log-loss {final['log_loss']:.4f}",
        f"  latency: {metadata['inference']['predict_proba_median_ms']:.2f} ms/decision (model), "
        f"{metadata['inference']['term_decomposition_median_ms']:.2f} ms per term decomposition",
        "",
        "Top terms by mean |score| (log-odds, averaged over the five actions; 'x' = interaction):",
    ]
    for name, value in list(metadata["global_importance"].items())[:12]:
        lines.append(f"  {name:40s} {value:.3f}")
    return lines


def main() -> None:
    parse_args()
    cfg = apply_cli_overrides(globals(), SMOKE_OVERRIDES)
    settings = build_settings()
    print(
        f"Training the EBM on '{DATASET_NAME}' (<= {MAX_TRAIN_ROWS} rows, {EBM_INTERACTIONS} interactions, "
        f"{DAGGER_ROUNDS} DAgger rounds, {N_JOBS} threads)"
    )
    outcome = train_ebm(settings, cfg)
    for line in format_summary(outcome.metadata):
        print(line)
    print(f"\nwrote {outcome.output_dir}")


if __name__ == "__main__":
    main()
