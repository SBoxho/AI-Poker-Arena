"""From per-hand data to the statistics stored in a run: match summaries and tournament standings.

Both return plain JSON-able dicts (``NaN`` where undefined; the writer converts them to ``null``).

**Which unit the intervals use.** Every interval is computed over *independent units*, in BB/100:

* ``session`` (default, :attr:`MatchConfig.ci_unit`): one value per session. Agents are rebuilt and their
  opponent statistics cleared at every session start, so sessions are independent even when the agents
  adapt to the opponent within a session.
* ``pair`` (the single hand when ``duplicate=False``): one value per duplicate pair. Pairs are *not*
  independent for adaptive agents (the second half of a session is played against an opponent model
  built from the first half), so these intervals can be too narrow. They are always stored as the
  extra ``bb_per_100_pair`` so the two can be compared, and used instead of sessions only when a match
  has fewer than :data:`~poker_arena.tournament.config.MIN_SESSIONS_FOR_CI` sessions (flagged via
  ``ci_unit_used`` / ``ci_unit_note``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from poker_arena.config import derive_seed
from poker_arena.tournament.arrays import PLAYER_A, PLAYER_B
from poker_arena.tournament.config import MIN_SESSIONS_FOR_CI, MatchConfig
from poker_arena.tournament.match import MatchResult, bootstrap_seed, match_key
from poker_arena.tournament.stats import (
    DRAW,
    INSUFFICIENT,
    LOSE,
    UNIT_SESSION,
    WIN,
    Estimate,
    PlayerTally,
    compare_verdict,
    estimate_mean,
    latency_summary,
    pair_sums,
    pool_groups,
    scale_sessions_to_bb_per_100,
    scale_to_bb_per_100,
    session_totals,
    wilson_interval,
)

RANKING_NOTE = (
    "Agents are ranked by their point estimate of BB/100 (every opponent weighted equally). Poker is "
    "noisy: neighbouring ranks are NOT statistically separated unless the pairwise verdict says so."
)
_SIDES = (("a", PLAYER_A), ("b", PLAYER_B))


# --------------------------------------------------------------------------------------
# Independent units
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class UnitSamples:
    """Agent A's profit per independent unit, in BB/100 (the mean of either array is A's BB/100).

    ``fine``    one value per duplicate pair (per hand without duplicate dealing);
    ``session`` one value per session of ``cfg.session_hands`` hands (a shorter last session counts
                proportionally, see :func:`~poker_arena.tournament.stats.scale_sessions_to_bb_per_100`).
    """

    fine: np.ndarray
    session: np.ndarray

    def negated(self) -> "UnitSamples":
        """The same samples from the opponent's side."""
        return UnitSamples(-self.fine, -self.session)

    def for_unit(self, unit: str) -> np.ndarray:
        return self.session if unit == UNIT_SESSION else self.fine


def unit_samples(profit_chips: np.ndarray, cfg: MatchConfig) -> UnitSamples:
    """Both unit levels from A's per-hand profits in chips (hand order of the match)."""
    profit_bb = np.asarray(profit_chips, dtype=np.float64) / cfg.game.big_blind
    fine_bb = pair_sums(profit_bb) if cfg.duplicate else profit_bb
    return UnitSamples(
        scale_to_bb_per_100(fine_bb, cfg.hands_per_unit),
        scale_sessions_to_bb_per_100(session_totals(profit_bb, cfg.session_hands), len(profit_bb)),
    )


def _estimate(samples: np.ndarray, cfg: MatchConfig, key: str, unit: str) -> Estimate:
    return estimate_mean(
        samples,
        confidence=cfg.confidence,
        method=cfg.ci_method,
        n_resamples=cfg.bootstrap_resamples,
        seed=bootstrap_seed(cfg.seed, key),
        unit=unit,
    )


def match_estimate(samples: UnitSamples, cfg: MatchConfig, key: str) -> Estimate:
    """Headline BB/100 of the first agent of a pairing: interval over ``cfg.ci_unit_used`` units."""
    unit = cfg.ci_unit_used
    return _estimate(samples.for_unit(unit), cfg, key, unit)


def pair_level_estimate(samples: UnitSamples, cfg: MatchConfig, key: str) -> Estimate:
    """The same BB/100 with the interval over duplicate pairs (hands without duplicate dealing)."""
    return _estimate(samples.fine, cfg, key, cfg.fine_unit)


def ci_unit_info(cfg: MatchConfig) -> dict[str, Any]:
    """Which unit the intervals use, and the warning to show if that is not the requested session level."""
    return {
        "requested": cfg.ci_unit,
        "used": cfg.ci_unit_used,
        "min_sessions": MIN_SESSIONS_FOR_CI,
        "sessions_per_match": cfg.num_sessions,
        "note": cfg.ci_unit_note,
    }


def protocol_info(cfg: MatchConfig) -> dict[str, Any]:
    """Duplicate protocol of the run and the legacy-protocol warning (``None`` for ``two_table``)."""
    return {"used": cfg.protocol, "note": cfg.protocol_note}


# --------------------------------------------------------------------------------------
# One match
# --------------------------------------------------------------------------------------
def estimate_fields(profit_chips: np.ndarray, cfg: MatchConfig, key: str, label_a: str, label_b: str) -> dict[str, Any]:
    """The interval-dependent part of a match summary: headline and pair-level BB/100 for both sides,
    verdicts, the unit used. Shared by :func:`summarize_match` and the loader's refresh of old runs."""
    samples = unit_samples(profit_chips, cfg)
    headline = match_estimate(samples, cfg, key)
    pair_level = pair_level_estimate(samples, cfg, key)
    labels = {"a": label_a, "b": label_b}
    players = {}
    for side, other in (("a", "b"), ("b", "a")):
        sign = 1 if side == "a" else -1
        est = headline if sign == 1 else headline.negated()
        verdict = compare_verdict(labels[side], labels[other], est)
        players[side] = {
            "bb_per_100": est.to_dict(),
            "bb_per_100_pair": (pair_level if sign == 1 else pair_level.negated()).to_dict(),
            "verdict": {"code": verdict.code, "text": verdict.text},
        }
    pair_se = _plain_se(samples.fine)
    return {
        "ci_unit_used": cfg.ci_unit_used,
        "ci_units": len(samples.for_unit(cfg.ci_unit_used)),
        "num_sessions": cfg.num_sessions,
        "session_over_pair_se": _plain_se(samples.session) / pair_se if pair_se > 0 else math.nan,
        "players": players,
    }


def _plain_se(samples: np.ndarray) -> float:
    """Standard error of the mean of independent samples (NaN for fewer than two)."""
    return float(np.std(samples, ddof=1) / math.sqrt(len(samples))) if len(samples) > 1 else math.nan


def apply_estimate_fields(match: dict[str, Any], fields: dict[str, Any]) -> None:
    """Write :func:`estimate_fields` output into a match summary dict (new or loaded)."""
    match["ci_unit_used"] = fields["ci_unit_used"]
    match["ci_units"] = fields["ci_units"]
    match["num_sessions"] = fields["num_sessions"]
    match["variance"]["session_over_pair_se"] = fields["session_over_pair_se"]
    for side, values in fields["players"].items():
        match["players"][side].update(values)


def _variance_report(result: MatchResult) -> dict[str, float]:
    """How much duplicate dealing reduced the noise of the BB/100 estimate.

    ``se_ratio`` compares the standard error obtained from pair sums with the one that the same
    number of *independent* hands would have (SD of a single hand * sqrt(n)): below 1 the pairing
    helped (e.g. 0.4 = the CI is 60% narrower than with independent deals). It describes card luck only;
    the interval actually reported also accounts for adaptation within sessions (``session_over_pair_se``,
    filled in by :func:`apply_estimate_fields`, is the session-level SE divided by the pair-level SE:
    well above 1 means the pairs of a session are positively correlated).
    """
    hand_sd = float(np.std(result.arrays.profit_chips / result.cfg.game.big_blind, ddof=1))
    unit_sd = float(np.std(result.unit_profits_bb, ddof=1))
    hands_per_unit = result.cfg.hands_per_unit
    independent_unit_sd = hand_sd * math.sqrt(hands_per_unit)
    return {
        "hand_sd_bb": hand_sd,
        "unit_sd_bb": unit_sd,
        "se_ratio": unit_sd / independent_unit_sd if independent_unit_sd > 0 else math.nan,
    }


def summarize_match(result: MatchResult) -> dict[str, Any]:
    """All statistics of one match (the content of ``matches/<A>__vs__<B>.json``).

    Layout: match-level facts, then ``players.a`` / ``players.b``, each seen from that player's side
    (``bb_per_100`` is the headline :class:`~poker_arena.tournament.stats.Estimate` dict over
    ``ci_unit_used`` units, ``bb_per_100_pair`` the pair-level one, ``tally`` the additive counters,
    ``metrics`` the derived rates, plus per-street, profit-by-ending-street and latency).
    """
    cfg, arrays, bb = result.cfg, result.arrays, result.cfg.game.big_blind
    net_bb_a = float(arrays.profit_chips.sum()) / bb
    labels = {"a": result.label_a, "b": result.label_b}

    players: dict[str, Any] = {}
    for side, player in _SIDES:
        other = "b" if side == "a" else "a"
        tally = PlayerTally.from_arrays(arrays, player, bb)
        players[side] = {
            "label": labels[side],
            "opponent": labels[other],
            "net_bb": net_bb_a if side == "a" else -net_bb_a,
            "tally": tally.to_dict(),
            "metrics": tally.metrics(),
            "win_rate_ci": wilson_interval(tally.hands_won, tally.hands, cfg.confidence),
            "showdown_win_rate_ci": wilson_interval(tally.showdowns_won, tally.showdowns, cfg.confidence),
            "per_street": tally.per_street(),
            "profit_by_end": tally.profit_by_end(),
            "latency": latency_summary(arrays.latency_ms(player)),
        }
    match = {
        "a": result.label_a,
        "b": result.label_b,
        "key": result.key,
        "num_hands": len(arrays),
        "num_units": cfg.num_units,
        "unit": cfg.fine_unit,
        "protocol": cfg.protocol,
        "hands_per_unit": cfg.hands_per_unit,
        "big_blind": bb,
        "seed": cfg.seed,
        "variance": _variance_report(result),
        "cpu_seconds": result.cpu_seconds,
        "players": players,
    }
    apply_estimate_fields(match, estimate_fields(arrays.profit_chips, cfg, result.key, result.label_a, result.label_b))
    return match


# --------------------------------------------------------------------------------------
# The round-robin
# --------------------------------------------------------------------------------------
def build_standings(
    agents: Sequence[str], samples: dict[tuple[str, str], UnitSamples], cfg: MatchConfig
) -> dict[str, Any]:
    """Leaderboard, matchup matrix, verdict matrix and the CI-unit note of a round-robin.

    ``samples[(x, y)]`` are x's per-unit profits against y (the negation of ``samples[(y, x)]``). The
    leaderboard gives each agent two pooled BB/100 estimates over its opponents, both over
    ``cfg.ci_unit_used`` units - ``equal`` (every opponent counts the same; the ranking criterion) and
    ``pooled`` (every unit counts the same) - each with n, SE and CI, plus its count of significant wins
    and losses and the pair-level estimate (``bb_per_100_pair``) for comparison.
    """
    unit = cfg.ci_unit_used
    estimates: dict[tuple[str, str], Estimate] = {}
    verdicts: dict[tuple[str, str], Any] = {}
    for x in agents:
        for y in agents:
            if x != y:
                estimates[x, y] = match_estimate(samples[x, y], cfg, match_key(x, y))
                verdicts[x, y] = compare_verdict(x, y, estimates[x, y])

    def pooled(x: str, weighting: str, level: str) -> Estimate:
        return pool_groups(
            [samples[x, y].for_unit(level) for y in agents if y != x],
            weighting=weighting,
            confidence=cfg.confidence,
            method=cfg.ci_method,
            n_resamples=cfg.bootstrap_resamples,
            seed=derive_seed("eval", cfg.seed, "leaderboard", x, weighting),
            unit=level if level == UNIT_SESSION else cfg.fine_unit,
        )

    rows = []
    for x in agents:
        equal, sample = pooled(x, "equal", unit), pooled(x, "sample", unit)
        codes = [verdicts[x, y].code for y in agents if y != x]
        rows.append(
            {
                "agent": x,
                "bb_per_100": equal.to_dict(),
                "bb_per_100_pooled": sample.to_dict(),
                "bb_per_100_pair": pooled(x, "equal", "pair").to_dict(),
                "ci_unit": unit,
                "hands": sum(len(samples[x, y].fine) * cfg.hands_per_unit for y in agents if y != x),
                "opponents": len(agents) - 1,
                "wins": codes.count(WIN),
                "losses": codes.count(LOSE),
                "draws": codes.count(DRAW),
                "insufficient": codes.count(INSUFFICIENT),
            }
        )
    rows.sort(key=lambda r: (-_rank_value(r["bb_per_100"]["mean"]), -_rank_value(r["bb_per_100_pooled"]["mean"]), r["agent"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "ranking_note": RANKING_NOTE,
        "ci_unit": ci_unit_info(cfg),
        "protocol": protocol_info(cfg),
        "leaderboard": rows,
        "matrix": {x: {y: estimates[x, y].to_dict() for y in agents if y != x} for x in agents},
        "verdicts": {x: {y: {"code": verdicts[x, y].code, "text": verdicts[x, y].text} for y in agents if y != x} for x in agents},
    }


def _rank_value(mean: float) -> float:
    """Sort key where undefined means rank last."""
    return -math.inf if math.isnan(mean) else mean
