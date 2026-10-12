"""Generate the teacher-labelled decision dataset used to train the EBM and XGBoost agents.

    .venv/Scripts/python scripts/generate_dataset.py            # standard preset
    .venv/Scripts/python scripts/generate_dataset.py --smoke    # ~300 hands, well under a minute
    .venv/Scripts/python scripts/generate_dataset.py --set RANDOM_SEED=7 --set DATASET_NAME=bc_seed7

Writes ``artifacts/datasets/<DATASET_NAME>.parquet`` + ``<DATASET_NAME>.meta.json`` (git-ignored,
regenerable). All logic lives in :mod:`poker_arena.data`; this script only holds the configuration.
"""

from __future__ import annotations

import os
import time

import poker_arena.config as arena_config
from poker_arena.config import GameConfig
from poker_arena.data import DatasetConfig, generate_dataset, write_dataset
from poker_arena.data.generation import legal_statistics
from poker_arena.script_config import apply_cli_overrides

# ==========================================
# DATASET GENERATION CONFIGURATION
# ==========================================
RANDOM_SEED = 42  # master seed; train/validation/DAgger hands use disjoint seed namespaces derived from it
# Game
STARTING_STACK = 1000  # chips per player at the start of every hand
SMALL_BLIND = 5
BIG_BLIND = 10
# Size (the standard preset takes ~1 minute on 22 cores, roughly 5-6 on a 4-core laptop; use --smoke for a quick check)
NUM_TRAINING_HANDS = 115_000  # hands played for the train split (about 3.2 labelled decisions per hand)
NUM_VALIDATION_HANDS = 23_000  # hands played for the validation split (disjoint seeds => disjoint hands)
# Behaviour of the learner-controlled seat, sampled per hand (weights are normalised). Labels are ALWAYS
# the canonical teacher's action; the behaviour only decides which states get visited.
BEHAVIOR_MIX = {
    "teacher": 0.30,  # follows the teacher exactly (on-policy states of the teacher)
    "teacher_eps": 0.20,  # teacher, but each decision is uniformly random with probability EXPLORATION_EPSILON
    "random": 0.10,  # uniformly random legal action (wild states: huge bets, odd stack ratios)
    "tight": 0.10,  # teacher style presets: different ranges lead to different board/pot structures
    "loose": 0.10,
    "aggressive": 0.10,
    "passive": 0.10,
}
EXPLORATION_EPSILON = 0.15  # random-action probability of the "teacher_eps" behaviour (and opponent)
# Opponent of the learner seat, sampled once per match (opponent-statistics features need history)
OPPONENT_MIX = {
    "random": 0.14,
    "call_station": 0.14,
    "maniac": 0.14,
    "teacher": 0.18,
    "tight": 0.10,
    "loose": 0.10,
    "aggressive": 0.10,
    "passive": 0.10,
}
MATCH_LENGTHS = (30, 100, 300, 1000)  # hands per match, cycled: short ones show "unknown opponent" states, long ones "known"
SHORT_STACK_PROBABILITY = 0.10  # share of hands that start with random shorter stacks (spr / push-fold coverage)
MIN_SHORT_STACK_BB = 10  # shortest starting stack (in big blinds) of such a hand
# Output and compute
DATASET_NAME = "teacher_bc_v2"  # file stem under OUTPUT_DIR
OUTPUT_DIR = "artifacts/datasets"  # relative paths are resolved against the repository root
N_WORKERS = 0  # worker processes; 0 = all CPU cores. The output does not depend on this number.
SMOKE_OVERRIDES = {
    "NUM_TRAINING_HANDS": 240,
    "NUM_VALIDATION_HANDS": 60,
    "MATCH_LENGTHS": (20, 60),
    "DATASET_NAME": "smoke",
}


def main() -> None:
    cfg = apply_cli_overrides(globals(), SMOKE_OVERRIDES)
    workers = cfg["N_WORKERS"] or os.cpu_count() or 1
    config = DatasetConfig(
        game=GameConfig(STARTING_STACK, SMALL_BLIND, BIG_BLIND),
        seed=RANDOM_SEED,
        behavior_mix=dict(BEHAVIOR_MIX),
        opponent_mix=dict(OPPONENT_MIX),
        exploration_epsilon=EXPLORATION_EPSILON,
        match_lengths=tuple(MATCH_LENGTHS),
        short_stack_probability=SHORT_STACK_PROBABILITY,
        min_short_stack_bb=MIN_SHORT_STACK_BB,
    )
    print(f"Generating '{DATASET_NAME}': {NUM_TRAINING_HANDS} train + {NUM_VALIDATION_HANDS} validation hands, {workers} workers")
    started = time.perf_counter()
    dataset = generate_dataset(config, NUM_TRAINING_HANDS, NUM_VALIDATION_HANDS, n_workers=workers)
    meta = {**dataset.meta, "script_configuration": {k: v for k, v in cfg.items()}}
    output_dir = arena_config.REPO_ROOT / OUTPUT_DIR  # an absolute OUTPUT_DIR wins (pathlib semantics)
    parquet_path, meta_path = write_dataset(dataset.frame, meta, DATASET_NAME, output_dir)
    elapsed = time.perf_counter() - started

    frame = dataset.frame
    print(f"\nrows: {len(frame):,}  ({elapsed:.1f} s total, {len(frame) / elapsed:,.0f} rows/s)")
    for split, info in dataset.meta["splits"].items():
        print(f"  {split:5s}: {info['rows']:>9,} rows from {info['hands']:>7,} hands")
    print("label distribution (teacher actions):")
    for name, entry in dataset.meta["label_distribution"].items():
        print(f"  {name:12s} {entry['count']:>9,}  {entry['fraction']:6.1%}")
    print("legal-action statistics:")
    for key, value in legal_statistics(frame).items():
        print(f"  {key:28s} {value:.3f}")
    print(f"content sha256: {dataset.meta['content_sha256']}")
    print(f"wrote {parquet_path}\n      {meta_path}")


if __name__ == "__main__":
    main()
