"""On-disk layout of a tournament run and JSON helpers shared by the writer and the loader.

``<tournaments dir>/``                       ``artifacts/tournaments`` by default
  ``latest.json``                            ``{"run_id": ..., "updated_at": ...}`` -> newest official run
  ``<run_id>/``
    ``config.json``                          effective configuration, agents, environment, seeds
    ``summary.json``                         leaderboard, matchup matrix, verdict matrix, latency, run meta
    ``sample_hands.jsonl``                   replayable sampled hands (one JSON object per line)
    ``matches/<A>__vs__<B>.json``            every statistic of one pairing (A sorts before B)
    ``hands/<A>__vs__<B>.npz``               compact per-hand arrays (profit curves, style counts)
    ``full_history/hands_full_<A>__vs__<B>.jsonl.gz``   every hand (only with ``keep_hands.all_hands``)

Use :func:`poker_arena.tournament.results.load_results` to read a run; the layout is an implementation
detail of this package.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2  # 2: intervals over sessions (ci_unit), pair-level extras; 1 = pair-level only
CONFIG_FILE = "config.json"
SUMMARY_FILE = "summary.json"
SAMPLES_FILE = "sample_hands.jsonl"
LATEST_FILE = "latest.json"
MATCHES_DIR = "matches"
HANDS_DIR = "hands"
FULL_HISTORY_DIR = "full_history"  # files are named hands_full_*.jsonl.gz, which .gitignore excludes
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]+")


def slugify(label: str) -> str:
    """Filesystem-safe form of an agent label."""
    return _SLUG_UNSAFE.sub("_", label).strip("_") or "agent"


def match_stem(label_a: str, label_b: str) -> str:
    """File stem of a pairing, e.g. ``"EBM__vs__Random"``."""
    return f"{slugify(label_a)}__vs__{slugify(label_b)}"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / path / non-finite values to strict-JSON-safe Python values.

    ``NaN`` and infinities become ``None`` (JSON has no representation for them); readers map
    ``None`` back to ``NaN``.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
