"""Measure how strong the rule-based teacher is: duplicate-deal BB/100 against the baselines and its own styles.

    .venv/Scripts/python scripts/evaluate_teacher.py            # standard: 2,000 pairs (4,000 hands) per opponent
    .venv/Scripts/python scripts/evaluate_teacher.py --smoke

Each *pair* plays one deal twice with the seats swapped, so luck of the cards cancels. BB/100 is per
hand; the 95% interval is a Student-t interval over pairs. Imitation accuracy is not poker skill -
this is the number to quote for the teacher's strength. Results are saved as JSON (config, seeds,
parameters, versions) so nothing in the docs or UI has to be typed by hand.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import poker_arena.config as arena_config
from poker_arena.agents import teacher
from poker_arena.agents.registry import AgentSpec
from poker_arena.config import GameConfig
from poker_arena.data.evaluation import evaluate_duplicate
from poker_arena.data.io import library_versions
from poker_arena.script_config import apply_cli_overrides

# ==========================================
# TEACHER EVALUATION CONFIGURATION
# ==========================================
RANDOM_SEED = 2024  # evaluation seed ("eval" namespace: never overlaps the dataset seeds)
# Game
STARTING_STACK = 1000
SMALL_BLIND = 5
BIG_BLIND = 10
# Matches
NUM_PAIRS = 2000  # duplicate pairs per opponent (each pair = 2 hands on the same deal)
OPPONENTS = (  # registry kinds, or teacher style presets as "teacher:<style>"
    "random",
    "call_station",
    "maniac",
    "teacher:tight",
    "teacher:loose",
    "teacher:aggressive",
    "teacher:passive",
    "teacher",  # the canonical teacher against itself (should be ~0)
)
N_WORKERS = 0  # worker processes; 0 = all CPU cores. Results do not depend on this number.
OUTPUT_FILE = "artifacts/teacher/evaluation.json"  # relative to the repository root
SMOKE_OVERRIDES = {
    "NUM_PAIRS": 30,
    "OPPONENTS": ("random", "maniac"),
    "OUTPUT_FILE": "artifacts/tmp/teacher_evaluation_smoke.json",
}


def opponent_spec(name: str) -> AgentSpec:
    kind, _, style = name.partition(":")
    if kind == "teacher" and style:
        return AgentSpec("teacher", {"style": style}, f"teacher_{style}")
    return AgentSpec(kind, {}, name)


def main() -> None:
    cfg = apply_cli_overrides(globals(), SMOKE_OVERRIDES)
    workers = cfg["N_WORKERS"] or os.cpu_count() or 1
    game = GameConfig(STARTING_STACK, SMALL_BLIND, BIG_BLIND)
    hero = AgentSpec("teacher", {}, "teacher")
    results = []
    started = time.perf_counter()
    for name in OPPONENTS:
        result = evaluate_duplicate(hero, opponent_spec(name), NUM_PAIRS, (RANDOM_SEED, name), game, workers)
        results.append(result.to_dict())
        print(result, flush=True)
    payload = {
        "configuration": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
        "teacher_params": teacher.CANONICAL_PARAMS.to_dict(),
        "results": results,
        "libraries": library_versions(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": time.perf_counter() - started,
    }
    path = arena_config.REPO_ROOT / OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
