"""Round-robin tournament between the configured agents (headless, multi-core, reproducible).

Every pair of agents plays one match of duplicate-paired hands (seat rotation + matched deals), the
results are written to ``artifacts/tournaments/<run_id>/`` and summarised below. Edit the
configuration block, or override from the command line:

    .venv/Scripts/python scripts/run_tournament.py                       # full tournament
    .venv/Scripts/python scripts/run_tournament.py --smoke               # ~1 min end-to-end check
    .venv/Scripts/python scripts/run_tournament.py --set NUM_TOURNAMENT_HANDS=50000 --set RUN_NAME=big
    .venv/Scripts/python scripts/run_tournament.py --agents random,call_station,maniac

Why sessions are the unit of the confidence intervals: duplicate dealing cancels card luck within a
pair, but the trained agents and the teacher adapt to opponent statistics that persist during a session
(SESSION_HANDS hands), so the pairs of one session are NOT independent and a pair-level interval can be
far too narrow. Agents are rebuilt and their statistics cleared at every session start, which makes
sessions independent: by default (CI_UNIT = "session") every interval is a t-interval (or bootstrap) over
per-session BB/100 values. The pair-level interval is still stored for comparison. With fewer than 8
sessions per pairing the pair level is used instead and flagged loudly in the report and the result files.

Why every session is played at two tables (PROTOCOL = "two_table"): the two plays of a deal must not be
able to learn from each other. If they were played back-to-back against the same agents and opponent
statistics (PROTOCOL = "single_table", the old behaviour, diagnostics only), the second mover would face
an opponent model that already contains the first play of the SAME deal, and identical adaptive agents
would not cancel (the first-named one loses ~10-13 BB/100). Two independent tables - own agent copies,
own opponent statistics, same deals, seats complemented - remove that leak.
"""

import argparse
import sys
import time
from pathlib import Path

from poker_arena.agents.base import ModelArtifactError
from poker_arena.agents.registry import AgentSpec, create_agent
from poker_arena.config import GameConfig
from poker_arena.script_config import add_config_arguments, apply_cli_overrides
from poker_arena.tournament.config import KeepHandsPolicy, MatchConfig
from poker_arena.tournament.match import Progress
from poker_arena.tournament.results import TournamentResults, load_results
from poker_arena.tournament.tournament import run_tournament

# ==========================================
# TOURNAMENT CONFIGURATION
# ==========================================
RANDOM_SEED = 42  # master seed: deals and agent seeds are derived from it (namespace "eval")
# Game (fixed-depth cash game: both stacks are reset to STARTING_STACK before every hand)
STARTING_STACK = 1000
SMALL_BLIND = 5
BIG_BLIND = 10
# Match length and structure
NUM_TOURNAMENT_HANDS = 20_000  # hands per pairing (rounded up to even; a duplicate pair = 2 hands)
DUPLICATE_DEALS = True  # every deal is played twice with seats swapped; confidence intervals use pair sums
SESSION_HANDS = 200  # agents are re-created and opponent statistics reset every SESSION_HANDS hands
# Statistics
CONFIDENCE_LEVEL = 0.95  # confidence level of every interval
CI_METHOD = "t"  # "t" (t-interval on pair profits) or "bootstrap" (seeded percentile bootstrap)
BOOTSTRAP_RESAMPLES = 2000  # only used when CI_METHOD = "bootstrap"
CI_UNIT = "session"  # independent unit of the intervals: "session" (recommended) or "pair" (see the note above)
PROTOCOL = "two_table"  # how the two plays of a deal are organised: "two_table" (independent tables, default) or the legacy "single_table"
# Execution
N_WORKERS = 0  # worker processes; 0 = all cores but one, 1 = single process (results do not depend on it)
# Agents: (kind, constructor params, label). Labels must be unique. Trained kinds (ebm, xgboost, ga, teacher)
# are skipped with a loud warning while their artifacts / code are missing, unless REQUIRE_ALL_AGENTS.
AGENTS = [
    ("ebm", {}, "EBM"),
    ("xgboost", {}, "XGBoost"),
    ("ga", {}, "GA"),
    ("teacher", {}, "Teacher"),
    ("random", {}, "Random"),
    ("call_station", {}, "CallStation"),
    ("maniac", {}, "Maniac"),
]
REQUIRE_ALL_AGENTS = False  # True = abort instead of skipping an agent that cannot be loaded
# Hands stored for replay (numeric per-hand data is always stored for ALL hands)
KEEP_FIRST_HANDS = 10  # the first N hands of every match
KEEP_BIGGEST_POTS = 5  # the N largest pots of every match
KEEP_BIGGEST_WINS = 5  # for each player, the N hands in which they won the most
KEEP_FULL_HISTORY = False  # also write EVERY hand to a gzip-jsonl file (only sensible for small runs)
# Output
RUN_NAME = None  # appended to the run folder name, e.g. "official" -> 20260101-120000_official
OUT_DIR = None  # folder that holds the run folders; None = artifacts/tournaments
MARK_LATEST = True  # point artifacts/tournaments/latest.json at this run (the run the app shows)
SMOKE_OVERRIDES = {  # applied by --smoke
    "NUM_TOURNAMENT_HANDS": 400,
    "SESSION_HANDS": 40,  # 10 sessions per pairing: enough for session-level intervals
    "RUN_NAME": "smoke",
    "N_WORKERS": 4,  # no need to spawn a process per core for a few seconds of work
    "MARK_LATEST": False,  # a smoke run is never the official run
}

_PROGRESS_BAR_WIDTH = 30
_BANNER = "!" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--agents", help="comma separated labels or kinds to keep (default: all configured agents)")
    parser.add_argument("--out", type=Path, help="folder that receives the run folder (overrides OUT_DIR)")
    return parser.parse_args()


def select_agents(entries: list, only: str | None, require_all: bool) -> list[AgentSpec]:
    """Specs for the configured agents; agents whose model artifacts are unavailable are skipped loudly."""
    wanted = {name.strip().lower() for name in only.split(",")} if only else None
    specs: list[AgentSpec] = []
    for kind, params, label in entries:
        if wanted is not None and kind.lower() not in wanted and label.lower() not in wanted:
            continue
        spec = AgentSpec(kind, dict(params), label)
        try:
            create_agent(spec)
        except ModelArtifactError as exc:
            if require_all:
                raise SystemExit(f"Agent {label!r} ({kind}) cannot be loaded and REQUIRE_ALL_AGENTS is set:\n{exc}") from exc
            print(f"{_BANNER}\nSKIPPING AGENT {label!r} ({kind}) - it cannot be loaded:\n  {exc}\n{_BANNER}", file=sys.stderr)
            continue
        specs.append(spec)
    if len(specs) < 2:
        raise SystemExit(f"Need at least two usable agents, have {[s.label for s in specs]}.")
    return specs


class ProgressBar:
    """One-line progress display on stderr."""

    def __call__(self, p: Progress) -> None:
        filled = int(_PROGRESS_BAR_WIDTH * p.fraction)
        rate = p.hands_done / p.elapsed_s if p.elapsed_s > 0 else 0.0
        bar = "#" * filled + "-" * (_PROGRESS_BAR_WIDTH - filled)
        end = "\n" if p.hands_done >= p.hands_total else ""
        print(
            f"\r[{bar}] {100 * p.fraction:5.1f}%  matches {p.matches_done}/{p.matches_total}  "
            f"{rate:7.0f} hands/s  {p.last_match:<40}",
            end=end, file=sys.stderr, flush=True,
        )  # fmt: skip


def print_report(results: TournamentResults) -> None:
    """Leaderboard, matchup matrix and verdicts of a finished run."""
    unit = results.ci_unit
    ci = f"{100 * results.confidence:.0f}% CI over {unit}s"
    board = results.leaderboard()
    shown = board[["rank", "agent", "bb_per_100", "ci_low", "ci_high", "wins", "losses", "draws"]]
    print(f"\nLEADERBOARD  (BB/100, every opponent weighted equally; {ci})")
    for note in (results.protocol_note, results.ci_unit_note):
        if note:
            print(f"{_BANNER}\nWARNING: {note}\n{_BANNER}")
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\nMATCHUP MATRIX  (row agent's BB/100 against column agent)")
    print(results.matchup_matrix().to_string(float_format=lambda x: f"{x:.1f}", na_rep="-"))
    print("\nVERDICTS  (win/lose only if the CI excludes 0; draw = no significant difference)")
    print(results.verdict_matrix().to_string())
    print(f"\n{results.meta['ranking_note']}")


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(globals(), SMOKE_OVERRIDES)
    specs = select_agents(cfg["AGENTS"], args.agents, cfg["REQUIRE_ALL_AGENTS"])
    match_config = MatchConfig(
        game=GameConfig(cfg["STARTING_STACK"], cfg["SMALL_BLIND"], cfg["BIG_BLIND"]),
        num_hands=cfg["NUM_TOURNAMENT_HANDS"],
        seed=cfg["RANDOM_SEED"],
        duplicate=cfg["DUPLICATE_DEALS"],
        session_hands=cfg["SESSION_HANDS"],
        keep_hands=KeepHandsPolicy(
            first=cfg["KEEP_FIRST_HANDS"],
            biggest_pots=cfg["KEEP_BIGGEST_POTS"],
            biggest_wins_each=cfg["KEEP_BIGGEST_WINS"],
            all_hands=cfg["KEEP_FULL_HISTORY"],
        ),
        n_workers=cfg["N_WORKERS"],
        confidence=cfg["CONFIDENCE_LEVEL"],
        ci_method=cfg["CI_METHOD"],
        bootstrap_resamples=cfg["BOOTSTRAP_RESAMPLES"],
        ci_unit=cfg["CI_UNIT"],
        protocol=cfg["PROTOCOL"],
    )
    out_dir = args.out if args.out is not None else cfg["OUT_DIR"]
    print(f"Tournament: {[s.label for s in specs]}, {match_config.num_hands} hands per pairing")
    started = time.perf_counter()
    run_dir = run_tournament(
        specs, match_config, run_name=cfg["RUN_NAME"], out_dir=out_dir, mark_latest=cfg["MARK_LATEST"],
        progress=ProgressBar(), extra_config={"script": "scripts/run_tournament.py", "config": cfg},
    )  # fmt: skip
    print_report(load_results(run_dir))
    print(f"\nSaved to {run_dir}  ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
