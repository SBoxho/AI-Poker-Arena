"""Pure data helpers of the AI Championship dashboard (no Streamlit, no heavy computation).

The dashboard only *reads* saved tournament runs (``poker_arena.tournament.results``). The few
things computed here are cheap re-views of what was saved:

* re-estimating confidence intervals at another level / for a subset of agents from the stored
  per-session profits - the independent unit of the run's intervals (a t-interval or bootstrap on a
  few hundred numbers),
* the CLI command that (re)creates a run - the app never launches the tournament itself,
* defensive readers of ``artifacts/models/*`` (files may be missing, formats may evolve).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poker_arena.config import MODELS_DIR, TOURNAMENTS_DIR, GameConfig
from poker_arena.tournament.results import TournamentResults
from poker_arena.tournament.stats import DRAW, LOSE, WIN, Estimate, compare_verdict, estimate_mean, pool_groups

ENV_TOURNAMENTS_DIR = "POKER_ARENA_TOURNAMENTS_DIR"  # lets tests / other machines point the page at another folder
ENV_MODELS_DIR = "POKER_ARENA_MODELS_DIR"
RUN_SCRIPT = ".venv/Scripts/python scripts/run_tournament.py"
CI_LEVELS = (0.80, 0.90, 0.95, 0.99)
DEFAULT_LEVEL = 0.95
LEVEL_TOLERANCE = 1e-9
PCT = 100.0


def tournaments_root() -> Path:
    """Folder that holds the run folders (``POKER_ARENA_TOURNAMENTS_DIR`` or ``artifacts/tournaments``)."""
    return Path(os.environ.get(ENV_TOURNAMENTS_DIR) or TOURNAMENTS_DIR)


def models_root() -> Path:
    return Path(os.environ.get(ENV_MODELS_DIR) or MODELS_DIR)


# ------------------------------------------------------------------------------ run facts
def format_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def run_facts(results: TournamentResults) -> list[tuple[str, str]]:
    """Label/value pairs summarising the run's configuration (shown as chips above the tabs)."""
    cfg = results.match_config
    game = cfg["game"]
    big_blind = game["big_blind"]
    n_agents = len(results.agent_names)
    pairings = n_agents * (n_agents - 1) // 2
    meta = results.meta
    unit = _interval_unit_text(results)
    method = "t-interval" if cfg["ci_method"] == "t" else f"bootstrap ({cfg['bootstrap_resamples']:,} resamples)"
    return [
        ("Date", format_date(results.config["created_at"])),
        ("Agents", str(n_agents)),
        ("Hands per pairing", f"{cfg['num_hands']:,}"),
        ("Total hands", f"{cfg['num_hands'] * pairings:,}"),
        ("Seed", str(cfg["seed"])),
        ("Stack", f"{game['starting_stack']:,} ({game['starting_stack'] / big_blind:g} BB)"),
        ("Blinds", f"{game['small_blind']}/{big_blind}"),
        ("Dealing", "duplicate (matched pairs)" if cfg["duplicate"] else "independent deals"),
        ("Statistical unit", unit),
        ("Protocol", _protocol_text(results)),
        ("Interval", f"{cfg['confidence'] * PCT:g}% {method}"),
        ("Compute", f"{meta['wall_seconds']:.0f} s, {meta['hands_per_second']:.0f} hands/s, {meta['n_workers_used']} workers"),
    ]


def _protocol_text(results: TournamentResults) -> str:
    """How the two plays of each deal were organised (the legacy protocol is called out as such)."""
    if not results.match_config["duplicate"]:
        return "independent deals"
    if results.protocol == "two_table":
        return "two-table duplicate"
    return "single-table duplicate (legacy: second-mover artifact)"


def _interval_unit_text(results: TournamentResults) -> str:
    """What one independent sample of the intervals is (shown in the run facts)."""
    cfg = results.match_config
    if results.ci_unit == "session":
        pairs = f", {cfg['session_hands'] // 2} duplicate pairs each" if cfg["duplicate"] else ""
        return f"sessions of {cfg['session_hands']} hands{pairs}"
    return "duplicate pairs" if cfg["duplicate"] else "independent hands"


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    return str(value)


def _quote(text: str) -> str:
    """Double quotes work in bash, PowerShell and cmd alike (labels may contain spaces)."""
    return f'"{text}"' if any(c.isspace() for c in text) else text


def reproduce_command(results: TournamentResults) -> str:
    """CLI that recreates this run (same seed, agents and settings -> identical results)."""
    cfg = results.match_config
    game = cfg["game"]
    overrides = {
        "RANDOM_SEED": cfg["seed"],
        "STARTING_STACK": game["starting_stack"],
        "SMALL_BLIND": game["small_blind"],
        "BIG_BLIND": game["big_blind"],
        "NUM_TOURNAMENT_HANDS": cfg["num_hands"],
        "DUPLICATE_DEALS": cfg["duplicate"],
        "SESSION_HANDS": cfg["session_hands"],
        "CONFIDENCE_LEVEL": cfg["confidence"],
        "CI_METHOD": cfg["ci_method"],
        "BOOTSTRAP_RESAMPLES": cfg["bootstrap_resamples"],
        "CI_UNIT": cfg["ci_unit"],
        "PROTOCOL": cfg["protocol"],
    }
    parts = [RUN_SCRIPT, f"--agents {_quote(','.join(results.agent_names))}"]
    parts += [f"--set {key}={_cli_value(value)}" for key, value in overrides.items()]
    return " ".join(parts)


# ------------------------------------------------------------------ launch-a-tournament helper
@dataclass(frozen=True)
class LaunchSpec:
    """Choices of the "run a new tournament" panel."""

    agents: tuple[str, ...]  # registry kinds or labels understood by ``--agents``
    hands: int = 20_000
    seed: int = 42
    starting_stack: int = 1000
    small_blind: int = 5
    big_blind: int = 10
    workers: int = 0  # 0 = all cores but one
    duplicate: bool = True
    ci_method: str = "t"
    confidence: float = 0.95
    run_name: str = ""
    mark_latest: bool = False  # the official run is generated deliberately, never by an experiment


def validate_launch(spec: LaunchSpec) -> str | None:
    """Error message for an invalid choice, else ``None``."""
    if len(spec.agents) < 2:
        return "Choose at least two agents."
    if spec.hands < 2:
        return "A pairing needs at least 2 hands."
    try:
        GameConfig(spec.starting_stack, spec.small_blind, spec.big_blind)
    except ValueError as exc:
        return str(exc)
    if spec.run_name and not (spec.run_name[0].isalpha() and all(c.isalnum() or c in "-_." for c in spec.run_name)):
        return "The run name must start with a letter and may only contain letters, digits, '-', '_' and '.'."
    return None


def launch_command(spec: LaunchSpec) -> str:
    """The exact command line for ``scripts/run_tournament.py`` (the app itself never runs it)."""
    error = validate_launch(spec)
    if error:
        raise ValueError(error)
    overrides = {
        "NUM_TOURNAMENT_HANDS": spec.hands,
        "RANDOM_SEED": spec.seed,
        "STARTING_STACK": spec.starting_stack,
        "SMALL_BLIND": spec.small_blind,
        "BIG_BLIND": spec.big_blind,
        "DUPLICATE_DEALS": spec.duplicate,
        "CI_METHOD": spec.ci_method,
        "CONFIDENCE_LEVEL": spec.confidence,
        "N_WORKERS": spec.workers,
        "MARK_LATEST": spec.mark_latest,
    }
    if spec.run_name:
        overrides["RUN_NAME"] = spec.run_name
    parts = [RUN_SCRIPT, f"--agents {','.join(spec.agents)}"]
    parts += [f"--set {key}={_cli_value(value)}" for key, value in overrides.items()]
    return " ".join(parts)


def estimate_duration_seconds(spec: LaunchSpec, hands_per_second: float | None) -> float | None:
    """Rough wall time from the throughput of an earlier run (``None`` if unknown)."""
    if not hands_per_second or hands_per_second <= 0:
        return None
    pairings = len(spec.agents) * (len(spec.agents) - 1) // 2
    return spec.hands * pairings / hands_per_second


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


# --------------------------------------------------------------- intervals (stored or re-estimated)
@dataclass(frozen=True)
class IntervalView:
    """Leaderboard / pairwise tables at the displayed interval level, for the chosen agents."""

    level: float
    recomputed: bool  # False = exactly what the tournament stored
    method: str
    leaderboard: pd.DataFrame
    pairwise: pd.DataFrame
    verdict_codes: pd.DataFrame
    verdict_texts: pd.DataFrame

    @property
    def order(self) -> list[str]:
        return list(self.leaderboard["agent"])


def _unit_estimates(
    results: TournamentResults, agents: Sequence[str], level: float, unit: str | None = None
) -> dict[tuple[str, str], Estimate]:
    """BB/100 of every ordered pair of ``agents`` at ``level``, with the interval over independent
    ``unit`` samples (default: the run's own unit, sessions)."""
    cfg = results.match_config
    unit = unit or results.ci_unit
    out: dict[tuple[str, str], Estimate] = {}
    for a in agents:
        for b in agents:
            if a != b:
                out[a, b] = estimate_mean(
                    results.ci_samples(a, b, unit), confidence=level, method=cfg["ci_method"],
                    n_resamples=cfg["bootstrap_resamples"], unit=unit,
                )  # fmt: skip
    return out


def interval_view(results: TournamentResults, agents: Sequence[str], level: float) -> IntervalView:
    """Tables for ``agents`` at confidence ``level``.

    With the stored level and all agents this is the saved leaderboard verbatim. Otherwise the
    intervals are re-estimated from the saved per-session profits (the same independent unit as the
    stored intervals) with the run's own method - same estimator, other level / subset - a cheap
    re-view, not a new simulation.
    """
    cfg = results.match_config
    chosen = [a for a in results.agent_names if a in set(agents)]
    stored = abs(level - results.confidence) < LEVEL_TOLERANCE and len(chosen) == len(results.agent_names)
    if stored:
        return IntervalView(
            level, False, cfg["ci_method"], results.leaderboard(), results.pairwise_table(),
            results.verdict_matrix(), results.verdict_matrix(text=True),
        )  # fmt: skip

    unit = results.ci_unit
    estimates = _unit_estimates(results, chosen, level)
    pair_level = _unit_estimates(results, chosen, level, "pair") if unit == "session" else estimates
    stored_pairs = results.pairwise_table().set_index(["agent", "opponent"])
    pair_rows, verdicts = [], {}
    for (a, b), est in estimates.items():
        verdict = compare_verdict(a, b, est)
        verdicts[a, b] = verdict
        row = stored_pairs.loc[(a, b)].to_dict()
        pair_est = pair_level[a, b]
        row.update(
            agent=a, opponent=b, bb_per_100=est.mean, se=est.se, ci_low=est.ci_low, ci_high=est.ci_high,
            significant=est.significant, verdict=verdict.code, verdict_text=verdict.text,
            pair_se=pair_est.se, pair_ci_low=pair_est.ci_low, pair_ci_high=pair_est.ci_high,
            pair_significant=pair_est.significant,
        )  # fmt: skip
        pair_rows.append(row)
    pairwise = pd.DataFrame(pair_rows, columns=["agent", "opponent", *stored_pairs.columns])

    board_rows = []
    for a in chosen:
        opponents = [b for b in chosen if b != a]
        groups = [results.ci_samples(a, b) for b in opponents]
        est = pool_groups(
            groups, weighting="equal", confidence=level, method=cfg["ci_method"],
            n_resamples=cfg["bootstrap_resamples"], unit=unit,
        )  # fmt: skip
        codes = [verdicts[a, b].code for b in opponents]
        board_rows.append(
            {
                "agent": a, "bb_per_100": est.mean, "se": est.se, "ci_low": est.ci_low, "ci_high": est.ci_high,
                "significant": est.significant, "wins": codes.count(WIN), "losses": codes.count(LOSE),
                "draws": codes.count(DRAW), "insufficient": len(codes) - sum(codes.count(c) for c in (WIN, LOSE, DRAW)),
                "hands": int(pairwise.loc[pairwise["agent"] == a, "hands"].sum()), "opponents": len(opponents),
            }
        )  # fmt: skip
    board = pd.DataFrame(board_rows).sort_values("bb_per_100", ascending=False, ignore_index=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    order = list(board["agent"])
    code_matrix = pd.DataFrame(
        {col: {row: verdicts[row, col].code if row != col else "" for row in order} for col in order},
        index=order, columns=order,
    )  # fmt: skip
    text_matrix = pd.DataFrame(
        {col: {row: verdicts[row, col].text if row != col else "" for row in order} for col in order},
        index=order, columns=order,
    )  # fmt: skip
    for matrix in (code_matrix, text_matrix):
        matrix.index.name, matrix.columns.name = "agent", "opponent"
    return IntervalView(level, True, cfg["ci_method"], board, pairwise, code_matrix, text_matrix)


# ------------------------------------------------------------------------------ display tables
def format_ci(low: float, high: float, digits: int = 0) -> str:
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"


def significance_marker(significant: bool, mean: float) -> str:
    if not significant:
        return "no (CI includes 0)"
    return "yes, positive" if mean > 0 else "yes, negative"


def leaderboard_display(board: pd.DataFrame, level: float) -> pd.DataFrame:
    """Human-readable leaderboard table (strings for the interval and the markers)."""
    return pd.DataFrame(
        {
            "Rank": board["rank"],
            "Agent": board["agent"],
            "BB/100": board["bb_per_100"].round(1),
            f"{level * PCT:g}% CI": [format_ci(lo, hi) for lo, hi in zip(board["ci_low"], board["ci_high"])],
            "Significant": [significance_marker(s, m) for s, m in zip(board["significant"], board["bb_per_100"])],
            "W / D / L": [f"{w} / {d} / {l}" for w, d, l in zip(board["wins"], board["draws"], board["losses"])],
            "Hands": board["hands"],
        }
    )


def matrix_from_pairwise(pairwise: pd.DataFrame, column: str, order: Sequence[str]) -> pd.DataFrame:
    matrix = pairwise.pivot(index="agent", columns="opponent", values=column).reindex(index=order, columns=order)
    matrix.index.name, matrix.columns.name = "agent", "opponent"
    return matrix


@dataclass(frozen=True)
class MatrixMetric:
    key: str
    label: str
    kind: str  # "diverging" (centre 0), "rate" (0..1, centre 0.5), "sequential"
    fmt: str  # format of the cell value


MATRIX_METRICS = (
    MatrixMetric("bb_per_100", "Win rate (BB/100)", "diverging", "{:+.0f}"),
    MatrixMetric("net_bb", "Net profit (BB)", "diverging", "{:+.0f}"),
    MatrixMetric("win_rate", "Hands won (share)", "rate", "{:.0%}"),
    MatrixMetric("showdown_win_rate", "Showdowns won (share)", "rate", "{:.0%}"),
    MatrixMetric("vpip", "VPIP of the row agent (vs the column agent)", "sequential", "{:.0%}"),
)
METRIC_BY_KEY = {m.key: m for m in MATRIX_METRICS}


# ------------------------------------------------------------------------------ model cards
MODEL_SUBDIRS = {"ebm": "ebm", "xgboost": "xgb", "ga": "ga"}
MODEL_TITLES = {"ebm": "Explainable Boosting Machine", "xgboost": "XGBoost (+ SHAP)", "ga": "Genetic Algorithm"}
TRAIN_COMMANDS = {
    "ebm": ".venv/Scripts/python scripts/train_ebm.py",
    "xgboost": ".venv/Scripts/python scripts/train_xgboost.py",
    "ga": ".venv/Scripts/python scripts/train_ga.py",
}
_SIZE_KEYS = (
    "n_train", "num_train", "train_size", "n_samples", "num_samples", "training_samples", "n_rows", "num_rows",
    "num_training_decisions", "n_decisions", "dataset_size", "n_training_decisions", "train_rows",
)  # fmt: skip
_SECTION_KEYS = (
    "training", "training_metrics", "metrics", "validation_metrics", "val_metrics", "imitation", "dataset",
    "data", "training_data",
)  # fmt: skip
_IMITATION_HINTS = ("accuracy", "agreement", "f1", "log loss", "logloss", "auc", "top1", "precision", "recall", "brier")
_IMPORTANCE_KEYS = (
    "global_importance", "feature_importance", "feature_importances", "top_features", "global_explanation",
    "importances",
)  # fmt: skip
_MAX_FLATTEN_DEPTH = 1
_MAX_METRIC_ROWS = 16


@dataclass(frozen=True)
class ModelCard:
    """What is known about one trained model (everything optional - files may be missing)."""

    kind: str
    title: str
    trained: bool
    note: str = ""
    created_at: str | None = None
    facts: tuple[tuple[str, str], ...] = ()
    imitation: tuple[tuple[str, str], ...] = ()  # accuracy-like numbers: how well the model copies its teacher
    training: tuple[tuple[str, str], ...] = ()  # everything else numeric from the training summary
    top_features: tuple[tuple[str, float], ...] = ()
    rules: tuple[str, ...] = ()
    genes: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _flatten_scalars(data: Mapping[str, Any], prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in data.items():
        label = f"{prefix}{key}".replace("_", " ")
        if isinstance(value, bool | int | float | str) and not (isinstance(value, float) and math.isnan(value)):
            rows.append((label, value))
        elif isinstance(value, Mapping) and depth < _MAX_FLATTEN_DEPTH:
            rows.extend(_flatten_scalars(value, f"{label} / ", depth + 1))
    return rows


_TRAIN_SIZE_KEY = re.compile(r"^(n|num)_(train|training)(_\w+)?$")


def _find_size(meta: Mapping[str, Any]) -> int | None:
    """Number of training examples, under any of the usual key names (top level or training sections)."""
    for container in (meta, *(meta[k] for k in _SECTION_KEYS if isinstance(meta.get(k), Mapping))):
        for key, value in container.items():
            is_size = key in _SIZE_KEYS or _TRAIN_SIZE_KEY.match(key)
            if is_size and isinstance(value, int | float) and not isinstance(value, bool):
                return int(value)
    return None


def _parse_importances(raw: Any) -> list[tuple[str, float]]:
    if isinstance(raw, Mapping):
        items: Iterable[Any] = list(raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        items = raw
    else:
        return []
    out: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, Mapping):
            name = item.get("feature") or item.get("name") or item.get("term")
            value = next((item[k] for k in ("importance", "value", "score", "contribution", "mean_abs_shap") if k in item), None)
        elif isinstance(item, Sequence) and not isinstance(item, str) and len(item) >= 2:
            name, value = item[0], item[-1]
        else:
            continue
        if name is not None and isinstance(value, int | float) and not isinstance(value, bool):
            out.append((str(name), float(value)))
    return sorted(out, key=lambda pair: -abs(pair[1]))


def _find_importances(meta: Mapping[str, Any]) -> list[tuple[str, float]]:
    for container in (meta, *(meta[k] for k in _SECTION_KEYS if isinstance(meta.get(k), Mapping))):
        for key in _IMPORTANCE_KEYS:
            if key in container:
                parsed = _parse_importances(container[key])
                if parsed:
                    return parsed
    return []


def _gene_rows(meta: Mapping[str, Any], best: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    specs = {s["name"]: s for s in meta.get("gene_specs", []) if isinstance(s, Mapping) and "name" in s}
    genes = best.get("genes") if isinstance(best.get("genes"), Mapping) else {}
    rows = []
    for name, value in genes.items():
        spec = specs.get(name, {})
        rows.append(
            {
                "gene": name, "value": value, "low": spec.get("low"), "high": spec.get("high"),
                "group": spec.get("group", ""), "meaning": spec.get("description", ""),
            }
        )  # fmt: skip
    return tuple(rows)


def load_model_card(kind: str, models_dir: Path) -> ModelCard:
    """Read ``<models_dir>/<subdir>/metadata.json`` (+ GA ``best_genome.json``) defensively."""
    title = MODEL_TITLES[kind]
    folder = models_dir / MODEL_SUBDIRS[kind]
    meta = _read_json(folder / "metadata.json")
    if meta is None:
        return ModelCard(kind, title, False, note=f"Not trained yet - run `{TRAIN_COMMANDS[kind]}`.")
    best = _read_json(folder / "best_genome.json") or {}
    facts: list[tuple[str, str]] = []
    size = _find_size(meta)
    if size is not None:
        facts.append(("Training examples", f"{size:,}"))
    features = meta.get("feature_names")
    if isinstance(features, Sequence) and not isinstance(features, str):
        facts.append(("Input features", str(len(features))))
    if isinstance(meta.get("seeds"), Mapping):
        facts.append(("Seeds", ", ".join(f"{k}={v}" for k, v in meta["seeds"].items())))
    if "game_config" in meta and isinstance(meta["game_config"], Mapping):
        game = meta["game_config"]
        facts.append(("Game", f"stack {game.get('starting_stack')}, blinds {game.get('small_blind')}/{game.get('big_blind')}"))

    summary: list[tuple[str, Any]] = []
    for key in _SECTION_KEYS:
        if isinstance(meta.get(key), Mapping):
            summary.extend(_flatten_scalars(meta[key]))
    summary = summary[:_MAX_METRIC_ROWS]
    imitation = tuple((k, _fmt_value(v)) for k, v in summary if isinstance(v, int | float) and any(h in k.lower() for h in _IMITATION_HINTS))
    training = tuple((k, _fmt_value(v)) for k, v in summary if (k, _fmt_value(v)) not in imitation)
    if "final_selection_fitness_bb100" in best:
        training += (("fitness vs the training opponents (BB/100)", _fmt_value(best["final_selection_fitness_bb100"])),)
    rules = tuple(str(r) for r in best.get("strategy", [])) if isinstance(best.get("strategy"), Sequence) else ()
    return ModelCard(
        kind=kind,
        title=title,
        trained=True,
        created_at=format_date(str(meta["created_at"])) if "created_at" in meta else None,
        facts=tuple(facts),
        imitation=imitation,
        training=training,
        top_features=tuple(_find_importances(meta)[:15]),
        rules=rules,
        genes=_gene_rows(meta, best),
        raw=meta,
    )


def load_model_cards(models_dir: Path) -> list[ModelCard]:
    return [load_model_card(kind, models_dir) for kind in MODEL_SUBDIRS]
