"""Reading tournament runs: the stable API for the Streamlit app and the notebooks.

    from poker_arena.tournament.results import load_results, list_runs

    res = load_results()                      # newest official run ("latest"); or a run id / folder path
    res.leaderboard()                         # ranked, BB/100 with 95% CI
    res.matchup_matrix()                      # rows = agent, columns = opponent, BB/100
    res.verdict_matrix()                      # "win" / "lose" / "draw" per ordered pair (significance-aware)
    res.cumulative_profit("EBM", "Random")    # profit curve in BB with a luck band

Conventions used by every table
-------------------------------
* ``bb_per_100`` is the win rate in big blinds per 100 hands. ``se`` / ``ci_low`` / ``ci_high`` are its
  standard error and confidence interval over **independent sessions** (one BB/100 value per session,
  t-interval with sessions - 1 degrees of freedom, or a bootstrap over sessions; ``config["match_config"]
  ["ci_method"]`` says which). ``significant`` means the CI excludes 0 with enough data.
* Why sessions are the unit: duplicate pairs are only independent if the agents ignore what they have
  seen of their opponent. The trained agents and the teacher adapt to opponent statistics that persist
  within a session, so the pairs of one session are correlated and a pair-level interval can be much too
  narrow. Every session starts with freshly built agents and cleared statistics, hence sessions are
  independent. The pair-level interval is kept as ``pair_se`` / ``pair_ci_low`` / ``pair_ci_high`` /
  ``pair_significant`` (``session_over_pair_se`` = how much wider the honest interval is). ``ci_unit`` says
  which unit the headline columns use; it is "pair" only when requested (``CI_UNIT = "pair"``) or when a
  match has too few sessions for a session-level interval - never silently: see :attr:`TournamentResults.ci_unit_note`.
  Runs saved before this rule (result schema 1) are re-estimated from their stored per-hand arrays when
  loaded, so they show the same numbers a fresh run would (``meta["recomputed_on_load"]``).
* ``protocol`` says how the two plays of every deal were organised. ``"two_table"`` (current): two
  independent tables per session, so nothing about a deal crosses between its plays and identical
  *adaptive* agents cancel in expectation (stateless ones exactly, pair by pair). ``"single_table"``
  (legacy; also what a run without the field is read as): both plays back-to-back against the same agents
  and opponent statistics, which gives the second mover of every pair an information advantage - a
  first-mover artifact of roughly -10 to -13 BB/100 for the first-named of two identical teachers. Results of
  such a run are only trustworthy for stateless opponents; see :attr:`TournamentResults.protocol_note`
  (and the ``UserWarning`` ``load_results`` raises for them). Hand ``2p`` / ``2p + 1`` are always the two
  plays of deal ``p``, but agent A is on the button in hand ``2p`` only for even ``p`` under
  ``two_table``: use the stored ``a_on_button`` flag, never the index parity.
* ``verdict`` is ``"win"`` / ``"lose"`` / ``"draw"`` / ``"insufficient"`` from the row agent's point of
  view; ``"draw"`` means *no significant difference*, not "equally strong". ``verdict_text`` is the
  sentence to show in a UI. Never describe one agent as better without ``verdict == "win"``.
* Undefined numbers are ``NaN``. Chip amounts are converted to big blinds (BB).
* Frequencies are fractions in ``[0, 1]``.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poker_arena.config import TOURNAMENTS_DIR
from poker_arena.engine.history import HandRecord
from poker_arena.tournament import storage
from poker_arena.tournament.config import MatchConfig
from poker_arena.tournament.stats import (
    Estimate,
    PlayerTally,
    cumulative_with_luck_band,
    pair_sums,
    session_totals,
    sum_tallies,
)
from poker_arena.tournament.summary import (
    UnitSamples,
    apply_estimate_fields,
    build_standings,
    estimate_fields,
    unit_samples,
)

LATEST_NAME = "latest"
SESSION = "session"
SUPPORTED_SCHEMAS = (1, storage.SCHEMA_VERSION)


def _num(value: Any) -> float:
    """``None`` (JSON's NaN) -> NaN."""
    return math.nan if value is None else float(value)


def _estimate_columns(est: Estimate, prefix: str = "", with_mean: bool = True) -> dict[str, Any]:
    columns = {
        f"{prefix}bb_per_100": est.mean,
        f"{prefix}se": est.se,
        f"{prefix}ci_low": est.ci_low,
        f"{prefix}ci_high": est.ci_high,
        f"{prefix}significant": est.significant,
    }
    if not with_mean:
        del columns[f"{prefix}bb_per_100"]
    return columns


# --------------------------------------------------------------------------------------
# Locating runs
# --------------------------------------------------------------------------------------
def resolve_run_dir(run_dir_or_name: str | Path = LATEST_NAME, root: Path | str | None = None) -> Path:
    """Directory of a run given ``"latest"``, a run id (looked up under ``root``) or a folder path."""
    root_dir = Path(root) if root is not None else TOURNAMENTS_DIR
    if str(run_dir_or_name) == LATEST_NAME:
        pointer = root_dir / storage.LATEST_FILE
        if not pointer.exists():
            raise FileNotFoundError(
                f"No official tournament yet ({pointer} is missing). Run scripts/run_tournament.py first."
            )
        run_dir_or_name = storage.read_json(pointer)["run_id"]
    for candidate in (Path(run_dir_or_name), root_dir / str(run_dir_or_name)):
        if (candidate / storage.SUMMARY_FILE).is_file():
            return candidate
    known = ", ".join(list_runs(root_dir)["run_id"]) if root_dir.exists() else "none"
    raise FileNotFoundError(f"No tournament run {str(run_dir_or_name)!r} under {root_dir}. Available: {known or 'none'}")


def list_runs(root: Path | str | None = None) -> pd.DataFrame:
    """All finished runs under ``root`` (default ``artifacts/tournaments``), newest first.

    Columns: run_id, run_name, created_at, agents (comma separated), num_agents, hands_per_match,
    total_hands, seed, duplicate, ci_method, ci_unit (unit of the intervals: session or pair/hand),
    protocol ("two_table" or the legacy "single_table"), is_latest, path.
    """
    root_dir = Path(root) if root is not None else TOURNAMENTS_DIR
    pointer = root_dir / storage.LATEST_FILE
    latest = storage.read_json(pointer)["run_id"] if pointer.exists() else None
    rows = []
    for config_path in sorted(root_dir.glob(f"*/{storage.CONFIG_FILE}")) if root_dir.exists() else []:
        run_dir = config_path.parent
        if not (run_dir / storage.SUMMARY_FILE).is_file():
            continue
        config = storage.read_json(config_path)
        match_config = config["match_config"]
        n_agents = len(config["agents"])
        hands = match_config["num_hands"]
        rows.append(
            {
                "run_id": config["run_id"],
                "run_name": config.get("run_name"),
                "created_at": config["created_at"],
                "agents": ", ".join(a["label"] for a in config["agents"]),
                "num_agents": n_agents,
                "hands_per_match": hands,
                "total_hands": hands * n_agents * (n_agents - 1) // 2,
                "seed": match_config["seed"],
                "duplicate": match_config["duplicate"],
                "ci_method": match_config["ci_method"],
                "ci_unit": MatchConfig.from_dict(match_config).ci_unit_used,
                "protocol": MatchConfig.from_dict(match_config).protocol,
                "is_latest": config["run_id"] == latest,
                "path": str(run_dir),
            }
        )
    columns = ["run_id", "run_name", "created_at", "agents", "num_agents", "hands_per_match", "total_hands",
               "seed", "duplicate", "ci_method", "ci_unit", "protocol", "is_latest", "path"]  # fmt: skip
    return pd.DataFrame(rows, columns=columns).sort_values("created_at", ascending=False, ignore_index=True)


def load_results(run_dir_or_name: str | Path = LATEST_NAME, root: Path | str | None = None) -> "TournamentResults":
    """Open a run: ``"latest"`` (default), a run id, or the path of a run folder.

    Runs made with the legacy ``single_table`` protocol (see :attr:`TournamentResults.protocol_note`) are
    loaded as usual but announced with a ``UserWarning``.
    """
    results = TournamentResults(resolve_run_dir(run_dir_or_name, root))
    if results.protocol_note:
        warnings.warn(f"Tournament run {results.run_id}: {results.protocol_note}", UserWarning, stacklevel=2)
    return results


# --------------------------------------------------------------------------------------
# The results object
# --------------------------------------------------------------------------------------
class TournamentResults:
    """A finished tournament run. All accessors return pandas DataFrames (or plain Python objects)
    and never recompute anything random: they only read what ``run_tournament`` stored."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.config: dict[str, Any] = storage.read_json(self.run_dir / storage.CONFIG_FILE)
        self._summary: dict[str, Any] = storage.read_json(self.run_dir / storage.SUMMARY_FILE)
        version = self._summary["schema_version"]
        if version not in SUPPORTED_SCHEMAS:
            raise ValueError(
                f"{self.run_dir} has result schema {version}, this code reads {SUPPORTED_SCHEMAS}; "
                "re-run the tournament (results are reproducible)."
            )
        self.run_id: str = self.config["run_id"]
        self._cfg = MatchConfig.from_dict(self.config["match_config"])
        self.recomputed_on_load = version < storage.SCHEMA_VERSION
        if self.recomputed_on_load:
            self._refresh_legacy_intervals()

    def __repr__(self) -> str:
        return f"TournamentResults(run_id={self.run_id!r}, agents={self.agent_names})"

    # ---------------------------------------------------------------- basics
    @property
    def agent_names(self) -> list[str]:
        """Agent labels in the order they were given to the tournament."""
        return list(self._summary["agents"])

    @property
    def meta(self) -> dict[str, Any]:
        """Run facts: id, name, created_at, wall/CPU seconds, hands per second, workers, environment
        (python and library versions), schema version."""
        return {
            "run_id": self.run_id,
            "run_name": self.config.get("run_name"),
            "schema_version": self._summary["schema_version"],
            **self._summary["meta"],
            "environment": self.config["environment"],
            "ranking_note": self._summary["ranking_note"],
            "ci_unit": self._summary["ci_unit"],
            "protocol": {"used": self.protocol, "note": self.protocol_note},
            "recomputed_on_load": self.recomputed_on_load,
        }

    @property
    def match_config(self) -> dict[str, Any]:
        """The match settings of the run (``MatchConfig.to_dict()``; defaults filled in for old runs)."""
        return self._cfg.to_dict()

    @property
    def protocol(self) -> str:
        """"two_table" (current) or "single_table" (legacy; the value for runs that predate the field)."""
        return self._cfg.protocol

    @property
    def protocol_note(self) -> str | None:
        """Warning for the legacy single-table protocol (second-mover artifact for adaptive agents), else None."""
        return self._cfg.protocol_note

    @property
    def ci_unit(self) -> str:
        """Unit of the headline intervals: "session", or "pair" ("hand" without duplicate deals) when pairs
        were requested or a match has too few sessions. Everything in the tables uses this unit."""
        return self._summary["ci_unit"]["used"]

    @property
    def ci_unit_note(self) -> str | None:
        """Warning to show whenever the intervals are not session-level (``None`` if they are)."""
        return self._summary["ci_unit"]["note"]

    @property
    def confidence(self) -> float:
        return self._cfg.confidence

    @cached_property
    def _stems(self) -> dict[tuple[str, str], str]:
        return {(m["a"], m["b"]): m["stem"] for m in self._summary["matches"]}

    def _locate(self, a: str, b: str) -> tuple[str, bool]:
        """(file stem, flipped) of the pairing; ``flipped`` if it is stored as (b, a)."""
        if (a, b) in self._stems:
            return self._stems[a, b], False
        if (b, a) in self._stems:
            return self._stems[b, a], True
        raise KeyError(f"No match between {a!r} and {b!r}; agents: {self.agent_names}")

    @cached_property
    def _matches(self) -> dict[str, dict[str, Any]]:
        return {
            stem: storage.read_json(self.run_dir / storage.MATCHES_DIR / f"{stem}.json")
            for stem in self._stems.values()
        }

    def _profit_chips(self, stem: str) -> np.ndarray:
        with np.load(self.run_dir / storage.HANDS_DIR / f"{stem}.npz") as data:
            return data["profit_chips"]

    def _refresh_legacy_intervals(self) -> None:
        """Re-estimate a schema-1 run (intervals over pairs) at the current default unit.

        Only the interval-dependent fields change: they are recomputed from the stored per-hand arrays
        and ``session_hands`` exactly as a new run would compute them (same seeds), so old and new runs
        are comparable. Counts, styles and latencies are taken from the files as saved.
        """
        samples: dict[tuple[str, str], UnitSamples] = {}
        for (a, b), stem in self._stems.items():
            profit = self._profit_chips(stem)
            match = self._matches[stem]
            apply_estimate_fields(match, estimate_fields(profit, self._cfg, match["key"], a, b))
            units = unit_samples(profit, self._cfg)
            samples[a, b], samples[b, a] = units, units.negated()
        self._summary.update(build_standings(self.agent_names, samples, self._cfg))

    def match_summary(self, a: str, b: str) -> dict[str, Any]:
        """The raw stored statistics of the pairing (see ``summarize_match``), as stored (A sorts first)."""
        return self._matches[self._locate(a, b)[0]]

    # ---------------------------------------------------------------- agents, leaderboard
    def agents(self) -> pd.DataFrame:
        """One row per agent: agent (label), kind, params (JSON), architecture, description,
        supports_explanations - what the agents are, as recorded when the run was made."""
        rows = []
        for entry in self.config["agents"]:
            description = entry["describe"]
            rows.append(
                {
                    "agent": entry["label"],
                    "kind": entry["spec"]["kind"],
                    "params": json.dumps(entry["spec"]["params"], sort_keys=True),
                    "architecture": description.get("architecture"),
                    "description": description.get("description"),
                    "supports_explanations": bool(description.get("supports_explanations", False)),
                }
            )
        return pd.DataFrame(rows)

    def leaderboard(self) -> pd.DataFrame:
        """Agents ranked by BB/100 averaged over opponents with every opponent weighted equally.

        Columns: rank, agent, bb_per_100, se, ci_low, ci_high, significant (CI excludes 0), all over
        ``ci_unit`` units (sessions by default); pooled_* (the same with every unit weighted equally);
        pair_se / pair_ci_low / pair_ci_high / pair_significant (the pair-level interval, for comparison;
        it is too narrow for agents that adapt within a session); wins / losses (number of opponents
        beaten by / lost to with a CI that excludes 0), draws (no significant difference), insufficient
        (too little data), hands (played by the agent), opponents, ci_unit, protocol.
        A higher rank is NOT a significant difference from the next one - see :meth:`verdict_matrix`.
        """
        rows = []
        for row in self._summary["leaderboard"]:
            equal, pooled = Estimate.from_dict(row["bb_per_100"]), Estimate.from_dict(row["bb_per_100_pooled"])
            pair_level = Estimate.from_dict(row["bb_per_100_pair"])
            rows.append(
                {
                    "rank": row["rank"],
                    "agent": row["agent"],
                    **_estimate_columns(equal),
                    **_estimate_columns(pooled, "pooled_"),
                    **_estimate_columns(pair_level, "pair_", with_mean=False),
                    **{k: row[k] for k in ("wins", "losses", "draws", "insufficient", "hands", "opponents", "ci_unit")},
                    "protocol": self.protocol,
                }
            )
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- pairwise
    def pairwise_table(self) -> pd.DataFrame:
        """One row per ordered pair (agent, opponent) - each match appears twice, once per side.

        Columns: agent, opponent, hands, units, unit ("pair"/"hand"; the finest unit), net_bb, bb_per_100,
        se, ci_low, ci_high, significant, verdict, verdict_text (all over ``ci_unit`` units - sessions by
        default - of which there are ``ci_units``; ``n_sessions`` is the number of sessions), protocol,
        pair_se / pair_ci_low / pair_ci_high / pair_significant (the same over duplicate pairs),
        session_over_pair_se (session-level SE / pair-level SE: > 1 = pairs of a session are correlated),
        se_ratio (standard-error ratio of duplicate pairs vs independent hands; < 1 = card-luck variance
        reduction), win_rate (+ _ci_low/_ci_high, Wilson),
        showdown_win_rate (+ CI), the style metrics (see :meth:`style_table`) and per-decision latency
        (latency_mean_ms, latency_median_ms, latency_p95_ms).
        """
        rows = []
        for a, b in sorted(self._stems):
            match = self.match_summary(a, b)
            for side in ("a", "b"):
                p = match["players"][side]
                est = Estimate.from_dict(p["bb_per_100"])
                pair_level = Estimate.from_dict(p["bb_per_100_pair"])
                metrics = {k: _num(v) for k, v in p["metrics"].items() if k != "hands"}
                rows.append(
                    {
                        "agent": p["label"],
                        "opponent": p["opponent"],
                        "hands": match["num_hands"],
                        "units": match["num_units"],
                        "unit": match["unit"],
                        "net_bb": p["net_bb"],
                        **_estimate_columns(est),
                        "verdict": p["verdict"]["code"],
                        "verdict_text": p["verdict"]["text"],
                        "ci_unit": match["ci_unit_used"],
                        "ci_units": match["ci_units"],
                        "protocol": self.protocol,
                        "n_sessions": match["num_sessions"],
                        **_estimate_columns(pair_level, "pair_", with_mean=False),
                        "session_over_pair_se": _num(match["variance"]["session_over_pair_se"]),
                        "se_ratio": _num(match["variance"]["se_ratio"]),
                        **metrics,
                        "win_rate_ci_low": _num(p["win_rate_ci"][0]),
                        "win_rate_ci_high": _num(p["win_rate_ci"][1]),
                        "showdown_win_rate_ci_low": _num(p["showdown_win_rate_ci"][0]),
                        "showdown_win_rate_ci_high": _num(p["showdown_win_rate_ci"][1]),
                        "latency_mean_ms": _num(p["latency"]["mean_ms"]),
                        "latency_median_ms": _num(p["latency"]["median_ms"]),
                        "latency_p95_ms": _num(p["latency"]["p95_ms"]),
                    }
                )
        return pd.DataFrame(rows)

    def _ranked_agents(self) -> list[str]:
        return [row["agent"] for row in self._summary["leaderboard"]]

    def matchup_matrix(self, metric: str = "bb_per_100", part: str = "value") -> pd.DataFrame:
        """Matrix with the row agent's result against the column agent (diagonal = NaN), rows and
        columns in leaderboard order.

        ``metric``  any numeric column of :meth:`pairwise_table`, e.g. "bb_per_100", "win_rate",
                    "showdown_win_rate", "vpip", "net_bb".
        ``part``    "value" (default); "se", "ci_low", "ci_high" (for "bb_per_100" the plain column - the
                    session-level interval -, for e.g. "win_rate" the "win_rate_ci_low"/"win_rate_ci_high"
                    columns); "significant" (bool matrix, bb_per_100 only); for bb_per_100 also the
                    pair-level "pair_se", "pair_ci_low", "pair_ci_high", "pair_significant".
        """
        table = self.pairwise_table()
        if part == "value":
            column = metric
        elif part in ("significant", "pair_significant"):
            column = part
            if metric != "bb_per_100":
                raise ValueError(f"part={part!r} is only defined for metric='bb_per_100'")
        else:
            column = part if metric == "bb_per_100" else f"{metric}_{part}"
        if column not in table.columns:
            raise KeyError(f"Unknown matrix quantity {column!r}; choose from {sorted(table.columns)}")
        order = self._ranked_agents()
        matrix = table.pivot(index="agent", columns="opponent", values=column).reindex(index=order, columns=order)
        matrix.index.name, matrix.columns.name = "agent", "opponent"
        return matrix

    def verdict_matrix(self, text: bool = False) -> pd.DataFrame:
        """Significance-aware pairwise verdicts, row agent vs column agent: "win" (row agent
        significantly better), "lose", "draw" (NO significant difference), "insufficient" (too little data);
        "" on the diagonal. ``text=True`` returns the sentences ("A > B (significant)") instead."""
        order = self._ranked_agents()
        key = "text" if text else "code"
        data = {
            col: {row: self._summary["verdicts"][row][col][key] if row != col else "" for row in order} for col in order
        }
        matrix = pd.DataFrame(data, index=order, columns=order)
        matrix.index.name, matrix.columns.name = "agent", "opponent"
        return matrix

    # ---------------------------------------------------------------- per-hand curves
    def _hand_profits_bb(self, a: str, b: str) -> np.ndarray:
        """Agent ``a``'s profit in big blinds in every hand against ``b`` (hand order of the match)."""
        stem, flipped = self._locate(a, b)
        profit_bb = self._profit_chips(stem).astype(np.float64) / self._cfg.game.big_blind
        return -profit_bb if flipped else profit_bb

    def pair_profits(self, a: str, b: str) -> np.ndarray:
        """Agent ``a``'s profit in big blinds per duplicate pair (per hand without duplicate deals)
        against ``b`` - the pair-level samples behind the ``pair_*`` columns. For intervals that are
        valid for adaptive agents use :meth:`session_profits` / :meth:`ci_samples`."""
        hand_bb = self._hand_profits_bb(a, b)
        return pair_sums(hand_bb) if self._cfg.duplicate else hand_bb

    def session_profits(self, a: str, b: str) -> np.ndarray:
        """Agent ``a``'s BB/100 in every session against ``b`` (A's perspective; sessions are consecutive
        blocks of ``session_hands`` hands, rebuilt agents and cleared opponent statistics at each start).

        These are the independent samples of the default intervals: their mean is the BB/100 of the match
        and ``t_estimate`` of them reproduces ``se`` / ``ci_low`` / ``ci_high``. If the last session is
        shorter than the others its value is scaled by its relative length so that the mean stays exactly
        the hand-weighted BB/100; with equally long sessions every value is a plain session BB/100.
        """
        return self.ci_samples(a, b, SESSION)

    def ci_samples(self, a: str, b: str, unit: str | None = None) -> np.ndarray:
        """BB/100-scaled independent-unit samples of ``a`` against ``b``, for re-estimating intervals
        (another confidence level, a bootstrap...): one value per ``unit`` - "session" or "pair" (a hand
        without duplicate deals); default the run's :attr:`ci_unit`. Pass the same ``unit`` to
        ``stats.estimate_mean`` / ``pool_groups`` so that the verdict rules of that unit apply."""
        stem, flipped = self._locate(a, b)
        units = unit_samples(self._profit_chips(stem), self._cfg)
        return (units.negated() if flipped else units).for_unit(unit or self.ci_unit)

    def cumulative_profit(self, a: str, b: str, max_points: int | None = None) -> pd.DataFrame:
        """Cumulative profit of ``a`` against ``b`` in BB after every duplicate pair.

        Columns: hand (hands played), cum_bb, luck_low, luck_high - the band inside which a player with
        NO edge would stay with ``confidence`` probability (``+- t * sd * sqrt(k)`` after ``k`` pairs).
        A curve that leaves the band is evidence of a real edge; one that stays inside is compatible with
        luck. When the intervals are session-level the band uses the variance rate implied by the
        *session* totals (and sessions - 1 degrees of freedom), so its width at the end of the match is
        exactly the confidence interval of the match total; a naive pair-level band would be too narrow
        for agents that adapt within a session. ``max_points`` thins long curves for plotting (the last
        point is always kept).
        """
        hand_bb = self._hand_profits_bb(a, b)
        units = pair_sums(hand_bb) if self._cfg.duplicate else hand_bb
        unit_sd = df = None
        if self.ci_unit == SESSION:
            totals = session_totals(hand_bb, self._cfg.session_hands)
            variance_per_hand = totals.var(ddof=1) / (len(hand_bb) / len(totals))
            unit_sd, df = math.sqrt(variance_per_hand * self._cfg.hands_per_unit), len(totals) - 1
        cumulative, low, high = cumulative_with_luck_band(units, self.confidence, unit_sd=unit_sd, df=df)
        hands = (np.arange(len(units)) + 1) * self._cfg.hands_per_unit
        frame = pd.DataFrame({"hand": hands, "cum_bb": cumulative, "luck_low": low, "luck_high": high})
        if max_points is not None and len(frame) > max_points:
            keep = np.unique(np.linspace(0, len(frame) - 1, max_points).round().astype(int))
            frame = frame.iloc[keep].reset_index(drop=True)
        return frame

    # ---------------------------------------------------------------- style and street tables
    def _tallies(self, vs: str | None) -> dict[str, PlayerTally]:
        """Pooled tally of every agent over all its opponents (or only against ``vs``)."""
        if vs is not None and vs not in self.agent_names:
            raise KeyError(f"Unknown opponent {vs!r}; agents: {self.agent_names}")
        per_agent: dict[str, list[PlayerTally]] = {name: [] for name in self.agent_names}
        for a, b in self._stems:
            match = self.match_summary(a, b)
            for side in ("a", "b"):
                p = match["players"][side]
                if vs is None or p["opponent"] == vs:
                    per_agent[p["label"]].append(PlayerTally.from_dict(p["tally"]))
        return {name: sum_tallies(parts) for name, parts in per_agent.items() if parts}

    def style_table(self, vs: str | None = None) -> pd.DataFrame:
        """Playing style per agent pooled over its matches (or only against opponent ``vs``).

        Columns: agent, hands, vpip, pfr, three_bet (3-bets / 3-bet opportunities),
        aggression_factor ((bets+raises)/calls), aggression_factor_postflop,
        aggression_freq_postflop ((bets+raises)/all postflop actions), fold_freq, call_freq
        (check or call), raise_freq (bet or raise; overall aggression frequency), win_rate (hands won),
        showdown_rate, showdown_win_rate. AF is NaN when the agent never called.
        """
        rows = [{"agent": name, **tally.metrics()} for name, tally in self._tallies(vs).items()]
        return pd.DataFrame(rows).astype({"hands": int})

    def per_street_table(self, vs: str | None = None) -> pd.DataFrame:
        """Action counts and frequencies per agent and street (preflop/flop/turn/river): decisions,
        fold, check, call, bet, raise counts, fold_freq, call_freq, raise_freq, aggression_factor."""
        rows = [{"agent": name, **row} for name, tally in self._tallies(vs).items() for row in tally.per_street()]
        return pd.DataFrame(rows)

    def profit_by_end_table(self, vs: str | None = None) -> pd.DataFrame:
        """Where each agent's profit comes from, by how the hand ended (preflop/flop/turn/river fold, or
        showdown): hands, share_of_hands, profit_bb, bb_per_hand, bb_per_100_contribution (the rows of an
        agent add up to its pooled BB/100)."""
        rows = [{"agent": name, **row} for name, tally in self._tallies(vs).items() for row in tally.profit_by_end()]
        return pd.DataFrame(rows)

    def latency_table(self) -> pd.DataFrame:
        """Decision latency per agent over the whole run (ms): n (decisions), mean_ms, median_ms, p95_ms, max_ms.
        Wall-clock measurements - they vary from run to run and with the load of the machine."""
        rows = [{"agent": name, **{k: _num(v) for k, v in self._summary["latency"][name].items()}} for name in self.agent_names]
        return pd.DataFrame(rows).astype({"n": int})

    # ---------------------------------------------------------------- hands
    @cached_property
    def _samples(self) -> list[dict[str, Any]]:
        path = self.run_dir / storage.SAMPLES_FILE
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def _filter_samples(self, tag: str | None, a: str | None, b: str | None) -> list[dict[str, Any]]:
        wanted = {a, b} - {None}
        return [
            s
            for s in self._samples
            if (tag is None or tag in s["tags"])
            and (not wanted or wanted <= {s["match"]["a"], s["match"]["b"]})
        ]

    def sample_hands_info(self, tag: str | None = None, a: str | None = None, b: str | None = None) -> pd.DataFrame:
        """Metadata of the stored sample hands, in the same order as :meth:`sample_hands`.

        Columns: agent_a, agent_b (the match; A sorts first), hand_index, pair_index, tags (comma
        separated; first, biggest_pot, biggest_win_a, biggest_win_b), profit_a_bb, pot_bb.
        ``tag`` keeps hands carrying that tag; ``a`` / ``b`` keep hands of matches involving those agents.
        """
        rows = [
            {
                "agent_a": s["match"]["a"],
                "agent_b": s["match"]["b"],
                "hand_index": s["hand_index"],
                "pair_index": s["pair_index"],
                "tags": ",".join(s["tags"]),
                "profit_a_bb": s["profit_a_bb"],
                "pot_bb": s["pot_bb"],
            }
            for s in self._filter_samples(tag, a, b)
        ]
        return pd.DataFrame(rows, columns=["agent_a", "agent_b", "hand_index", "pair_index", "tags", "profit_a_bb", "pot_bb"])

    def sample_hands(self, tag: str | None = None, a: str | None = None, b: str | None = None) -> list[HandRecord]:
        """Replayable sample hands (``record.replay()`` / ``record.verify()``); filters as in
        :meth:`sample_hands_info`. Only a sample is stored, never all hands (unless the run was made
        with ``keep_hands.all_hands``, see :meth:`full_history_path`)."""
        return [HandRecord.from_dict(s["record"]) for s in self._filter_samples(tag, a, b)]

    def full_history_path(self, a: str, b: str) -> Path | None:
        """gzip-jsonl file with EVERY hand of the pairing (``{"hand_index", "record"}`` per line), or
        None if the run did not keep full history."""
        stem, _ = self._locate(a, b)
        path = self.run_dir / storage.FULL_HISTORY_DIR / f"hands_full_{stem}.jsonl.gz"
        return path if path.exists() else None

    def pairings(self) -> Sequence[tuple[str, str]]:
        """The stored pairings ``(A, B)`` (A sorts before B)."""
        return sorted(self._stems)
