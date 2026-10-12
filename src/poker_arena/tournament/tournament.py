"""Round-robin tournaments: every pair of agents plays one duplicate match; results go to disk.

``run_tournament`` is the one entry point scripts, the Streamlit app and notebooks use to *produce* a
run; ``poker_arena.tournament.results.load_results`` is the one entry point to *read* it. All
sessions of all matches share a single process pool, so cores stay busy until the very end.
"""

from __future__ import annotations

import gzip
import itertools
import json
import platform
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from poker_arena.agents.registry import AgentSpec, create_agent
from poker_arena.config import TOURNAMENTS_DIR
from poker_arena.tournament import storage
from poker_arena.tournament.config import MatchConfig, resolve_n_workers
from poker_arena.tournament.match import (
    MatchAssembler,
    MatchResult,
    Progress,
    ProgressCallback,
    SessionTask,
    build_tasks,
    execute_tasks,
    labelled,
)
from poker_arena.tournament.stats import latency_summary
from poker_arena.tournament.summary import UnitSamples, build_standings, summarize_match, unit_samples

#: Libraries whose versions are recorded in ``config.json`` (those that influence results).
RECORDED_LIBRARIES = (
    "poker-arena", "pokerkit", "numpy", "pandas", "scipy", "scikit-learn", "interpret", "xgboost", "shap", "joblib",
)  # fmt: skip
_RUN_ID_TIME_FORMAT = "%Y%m%d-%H%M%S"
_PARTIAL_SUFFIX = ".partial"


def _library_versions() -> dict[str, str]:
    versions = {}
    for name in RECORDED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _validate_agents(agent_specs: Sequence[AgentSpec | str]) -> list[AgentSpec]:
    specs = [labelled(s) for s in agent_specs]
    if len(specs) < 2:
        raise ValueError("A tournament needs at least two agents")
    labels = [s.label or "" for s in specs]
    slugs = [storage.slugify(label).lower() for label in labels]  # lower: Windows file names ignore case
    if len(set(labels)) != len(labels) or len(set(slugs)) != len(slugs):
        raise ValueError(f"Agent labels must be unique (and distinct as file names); got {labels}. Give each spec its own label.")
    return specs


def _new_run_dir(root: Path, run_name: str | None) -> tuple[str, Path]:
    """A fresh ``<timestamp>[_<name>]`` id whose directory does not exist yet."""
    stamp = datetime.now().astimezone().strftime(_RUN_ID_TIME_FORMAT)
    base = f"{stamp}_{storage.slugify(run_name)}" if run_name else stamp
    run_id, counter = base, 1
    while (root / run_id).exists() or (root / f"{run_id}{_PARTIAL_SUFFIX}").exists():
        counter += 1
        run_id = f"{base}-{counter}"
    return run_id, root / run_id


def _describe_agents(specs: Sequence[AgentSpec]) -> list[dict[str, Any]]:
    """Build every agent once in this process: fails fast on a missing model and records metadata."""
    return [
        {"label": s.label, "slug": storage.slugify(s.label or ""), "spec": s.to_dict(), "describe": create_agent(s).describe()}
        for s in specs
    ]


def _write_match_files(run_dir: Path, result: MatchResult, summary: dict[str, Any]) -> str:
    stem = storage.match_stem(result.label_a, result.label_b)
    storage.write_json(run_dir / storage.MATCHES_DIR / f"{stem}.json", summary)
    result.arrays.save(run_dir / storage.HANDS_DIR / f"{stem}.npz")
    if result.all_hands is not None:
        path = run_dir / storage.FULL_HISTORY_DIR / f"hands_full_{stem}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for hand_index, record in enumerate(result.all_hands):
                fh.write(json.dumps({"hand_index": hand_index, "record": record}) + "\n")
    return stem


def _sample_lines(result: MatchResult) -> list[str]:
    """JSON lines of the sampled hands of a match (``record`` is ``HandRecord.to_dict()``)."""
    bb = result.cfg.game.big_blind
    hands_per_unit = result.cfg.hands_per_unit
    return [
        json.dumps(
            storage.to_jsonable(
                {
                    "match": {"a": result.label_a, "b": result.label_b},
                    "hand_index": hand.hand_index,
                    "pair_index": hand.hand_index // hands_per_unit,
                    "tags": list(hand.tags),
                    "profit_a_bb": hand.profit_chips / bb,
                    "pot_bb": hand.pot_chips / bb,
                    "record": hand.record,
                }
            )
        )
        for hand in result.samples
    ]


def run_tournament(
    agent_specs: Sequence[AgentSpec | str],
    cfg: MatchConfig | None = None,
    run_name: str | None = None,
    out_dir: Path | str | None = None,
    *,
    mark_latest: bool = True,
    progress: ProgressCallback | None = None,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """Play every unordered pair of agents (one duplicate match each) and write the run to disk.

    * ``agent_specs``  two or more specs (or kind strings) with distinct labels. Each pairing is
      played with the alphabetically first label as agent A, so a pairing's cards, seeds and results
      do not depend on the order or the composition of the list.
    * ``cfg``          shared by all matches (``cfg.num_hands`` hands per pairing).
    * ``out_dir``      directory that holds run folders (default ``artifacts/tournaments``).
    * ``mark_latest``  also point ``latest.json`` at this run (official runs); pass False for experiments.
    * ``progress``     called with a :class:`Progress` after every finished session.
    * ``extra_config`` anything JSON-able to store in ``config.json`` (e.g. the script's config block).

    Returns the run directory. The run is written to ``<id>.partial`` and renamed when complete, so a
    reader never sees a half-written run. Raises ``ModelArtifactError`` before any hand is played if
    an agent cannot be built.
    """
    cfg = cfg or MatchConfig()
    specs = _validate_agents(agent_specs)
    descriptions = _describe_agents(specs)
    root = Path(out_dir) if out_dir is not None else TOURNAMENTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = _new_run_dir(root, run_name)
    partial = run_dir.with_name(run_dir.name + _PARTIAL_SUFFIX)
    for sub in (storage.MATCHES_DIR, storage.HANDS_DIR, *([storage.FULL_HISTORY_DIR] if cfg.keep_hands.all_hands else [])):
        (partial / sub).mkdir(parents=True)
    try:
        _run_into(partial, run_id, specs, descriptions, cfg, run_name, progress, extra_config)
        partial.rename(run_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if mark_latest:
        _write_latest(root, run_id)
    return run_dir


def _write_latest(root: Path, run_id: str) -> None:
    tmp = root / f"{storage.LATEST_FILE}.tmp"
    storage.write_json(tmp, {"run_id": run_id, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    tmp.replace(root / storage.LATEST_FILE)


def _run_into(
    run_dir: Path,
    run_id: str,
    specs: list[AgentSpec],
    descriptions: list[dict[str, Any]],
    cfg: MatchConfig,
    run_name: str | None,
    progress: ProgressCallback | None,
    extra_config: dict[str, Any] | None,
) -> None:
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    ordered = sorted(specs, key=lambda s: s.label or "")
    pairings = list(itertools.combinations(ordered, 2))
    tasks: list[SessionTask] = [t for i, (a, b) in enumerate(pairings) for t in build_tasks(i, a, b, cfg)]
    workers_used = resolve_n_workers(cfg.n_workers, len(tasks))
    labels = [s.label or "" for s in specs]

    storage.write_json(
        run_dir / storage.CONFIG_FILE,
        {
            "schema_version": storage.SCHEMA_VERSION,
            "run_id": run_id,
            "run_name": run_name,
            "created_at": started_at.isoformat(timespec="seconds"),
            "match_config": cfg.to_dict(),
            "agents": descriptions,
            "seeds": {
                "namespace": "eval",
                "master_seed": cfg.seed,
                "deals": 'Deal.from_seed(derive_seed("eval", seed, match_key, pair_index))',
                "agents": (
                    'agent.reset(derive_seed("eval", seed, match_key, "agent", label, session_index, "table", t)) '
                    'for table t in {0, 1}; the legacy single_table protocol omits the last two parts'
                ),
                "protocol": cfg.protocol,
                "match_key": 'the two agent labels, sorted, joined by "__vs__"',
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "libraries": _library_versions(),
            },
            "script_config": extra_config,
        },
    )

    assemblers = {i: MatchAssembler(a, b, cfg) for i, (a, b) in enumerate(pairings)}
    samples: dict[tuple[str, str], UnitSamples] = {}
    pairing_index: list[dict[str, str]] = []
    latencies: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    sample_lines: list[str] = []
    cpu_seconds = 0.0
    total_hands = cfg.num_hands * len(pairings)
    hands_done = matches_done = 0

    for result in execute_tasks(tasks, cfg.n_workers):
        assembler = assemblers[result.match_index]
        assembler.add(result)
        hands_done += result.num_hands
        if assembler.complete:
            match = assembler.build()
            del assemblers[result.match_index]
            matches_done += 1
            stem = _write_match_files(run_dir, match, summarize_match(match))
            pairing_index.append({"a": match.label_a, "b": match.label_b, "stem": stem})
            units = unit_samples(match.arrays.profit_chips, cfg)
            samples[match.label_a, match.label_b] = units
            samples[match.label_b, match.label_a] = units.negated()
            latencies[match.label_a].append(match.arrays.latency_a_ms)
            latencies[match.label_b].append(match.arrays.latency_b_ms)
            sample_lines.extend(_sample_lines(match))
            cpu_seconds += match.cpu_seconds
        if progress is not None:
            progress(
                Progress(hands_done, total_hands, matches_done, len(pairings), time.perf_counter() - started,
                         f"{pairings[result.match_index][0].label} vs {pairings[result.match_index][1].label}")
            )  # fmt: skip

    wall = time.perf_counter() - started
    pairing_index.sort(key=lambda p: (p["a"], p["b"]))
    (run_dir / storage.SAMPLES_FILE).write_text("".join(line + "\n" for line in sample_lines), encoding="utf-8")
    storage.write_json(
        run_dir / storage.SUMMARY_FILE,
        {
            "schema_version": storage.SCHEMA_VERSION,
            "run_id": run_id,
            "agents": labels,
            "matches": pairing_index,
            **build_standings(labels, samples, cfg),
            "latency": {label: latency_summary(np.concatenate(parts)) for label, parts in latencies.items()},
            "meta": {
                "created_at": started_at.isoformat(timespec="seconds"),
                "wall_seconds": wall,
                "cpu_seconds": cpu_seconds,
                "total_hands": total_hands,
                "hands_per_second": total_hands / wall if wall > 0 else None,
                "n_workers_used": workers_used,
            },
        },
    )
