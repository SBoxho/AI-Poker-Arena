"""Pure statistics for poker tournaments (numpy in, numbers out; no I/O, no engine).

Why this module is careful: poker results are extremely noisy (a few BB/100 is invisible after
thousands of hands). Three rules are built in:

1. **Intervals are computed over independent units.** Duplicate dealing makes the two mirrored hands of a
   deal strongly negatively correlated (card luck cancels; two identical stateless agents cancel exactly,
   adaptive ones only under the two-table protocol - see :mod:`poker_arena.tournament.duplicate`), so a
   *pair* sum has a much smaller variance than two independent hands - never use single hands of a
   duplicate match. But agents that adapt to
   opponent statistics (the trained agents and the teacher) change their play *during* a session, which
   makes the pairs of one session dependent, and a pair-level interval can then be far too narrow
   (:func:`cumulative_with_luck_band` and the examples in ``tests/test_tournament_stats.py``). A
   **session** - a block of hands that starts with freshly built agents and cleared opponent statistics -
   is independent of every other session by construction, so the default unit is the session: one
   BB/100 value per session (:func:`session_totals`, :func:`scale_sessions_to_bb_per_100`), a t-interval
   (df = sessions - 1) or a bootstrap over sessions. Pair-level results stay available for comparison.
2. **Every number carries its uncertainty** (:class:`Estimate`: n, mean, SE, CI) - t-interval or a seeded
   percentile bootstrap.
3. **No claim without a CI that excludes zero.** :func:`compare_verdict` is the only place that turns an
   estimate into words (``"A > B (significant)"``); with too little data or a CI containing 0 it says so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from poker_arena.tournament.arrays import (
    ACTION_KINDS,
    END_CATEGORIES,
    KIND_INDEX,
    NUM_STREETS,
    STREET_NAMES,
    HandArrays,
)
from poker_arena.tournament.config import MIN_SESSIONS_FOR_CI

#: Fewer independent pairs (or hands) than this and no verdict is issued, whatever the CI says (t-intervals
#: on a handful of heavy-tailed poker results are not trustworthy).
MIN_UNITS_FOR_VERDICT = 30
#: The same rule for session-level estimates: each session already averages many pairs, so far fewer
#: units are enough - and fewer than this many sessions never get a session-level interval at all.
MIN_SESSIONS_FOR_VERDICT = MIN_SESSIONS_FOR_CI
UNIT_SESSION = "session"
#: Verdict codes of a comparison from the point of view of the first agent.
WIN, LOSE, DRAW, INSUFFICIENT = "win", "lose", "draw", "insufficient"
#: Cap on elements of one bootstrap index matrix so memory stays bounded for huge matches.
_BOOTSTRAP_CHUNK_ELEMENTS = 4_000_000
_POSTFLOP = slice(1, NUM_STREETS)
_BET_KINDS = (KIND_INDEX["bet"], KIND_INDEX["raise"])


# --------------------------------------------------------------------------------------
# Basic conversions
# --------------------------------------------------------------------------------------
def bb_per_100(total_bb: float, num_hands: int) -> float:
    """Win rate in big blinds per 100 hands (``NaN`` for zero hands)."""
    return 100.0 * total_bb / num_hands if num_hands else math.nan


def pair_sums(per_hand: np.ndarray) -> np.ndarray:
    """Sum of the two mirrored hands of every duplicate pair (hands ``2i`` and ``2i + 1``)."""
    values = np.asarray(per_hand, dtype=np.float64)
    if len(values) % 2:
        raise ValueError("duplicate pairs need an even number of hands")
    return values[0::2] + values[1::2]


def min_units_for_verdict(unit: str) -> int:
    """Smallest number of independent units of kind ``unit`` ("session", "pair" or "hand") for a verdict."""
    return MIN_SESSIONS_FOR_VERDICT if unit == UNIT_SESSION else MIN_UNITS_FOR_VERDICT


def session_totals(per_hand: np.ndarray, session_hands: int) -> np.ndarray:
    """Sum of ``per_hand`` over consecutive sessions of ``session_hands`` hands (the last may be shorter)."""
    values = np.asarray(per_hand, dtype=np.float64)
    if len(values) == 0:
        return values
    return np.add.reduceat(values, np.arange(0, len(values), session_hands))


def scale_sessions_to_bb_per_100(session_profit_bb: np.ndarray, total_hands: int) -> np.ndarray:
    """Per-session profits (BB per session) as BB/100 hands, ready for an interval over sessions.

    For equally long sessions this is simply ``100 * profit / session_hands``. If the last session is
    shorter its value is scaled by its relative length, so that the **mean over sessions is exactly the
    hand-weighted BB/100 of the whole match** (``total_hands`` in all) and short sessions weigh less.
    """
    totals = np.asarray(session_profit_bb, dtype=np.float64)
    return totals * (100.0 * len(totals) / total_hands)


def scale_to_bb_per_100(unit_profits_bb: np.ndarray, hands_per_unit: int) -> np.ndarray:
    """Express per-unit profits (BB per pair, or per hand) as BB/100 hands.

    The mean of the result *is* the BB/100 win rate and its standard error is the BB/100 standard
    error, so every interval computed on it is directly in BB/100.
    """
    return np.asarray(unit_profits_bb, dtype=np.float64) * (100.0 / hands_per_unit)


# --------------------------------------------------------------------------------------
# Estimates with uncertainty
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Estimate:
    """A mean with its uncertainty. All fields are ``NaN`` where undefined (e.g. n < 2).

    ``n`` is the number of independent units of kind ``unit`` ("session", "pair" or "hand") the interval
    was computed over; ``df`` is the t degrees of freedom (``NaN`` for bootstrap). ``excludes_zero`` is
    the pure CI fact; :attr:`significant` additionally requires :func:`min_units_for_verdict` units.
    """

    n: int
    mean: float
    se: float
    ci_low: float
    ci_high: float
    confidence: float
    method: str  # "t" | "bootstrap"
    df: float = field(default=math.nan, compare=False)  # NaN != NaN would break equality of bootstrap estimates
    unit: str = "pair"

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0  # False when NaN

    @property
    def significant(self) -> bool:
        return self.excludes_zero and self.n >= min_units_for_verdict(self.unit)

    def negated(self) -> "Estimate":
        """The same estimate seen from the opponent (``-x``; interval endpoints swap)."""
        return Estimate(
            self.n, -self.mean, self.se, -self.ci_high, -self.ci_low, self.confidence, self.method, self.df, self.unit
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "method": self.method,
            "df": self.df,
            "unit": self.unit,
            "excludes_zero": self.excludes_zero,
            "significant": self.significant,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Estimate":
        def number(key: str) -> float:
            return math.nan if d.get(key) is None else float(d[key])

        return cls(
            int(d["n"]), number("mean"), number("se"), number("ci_low"), number("ci_high"),
            float(d["confidence"]), d["method"], number("df"), d.get("unit", "pair"),
        )  # fmt: skip


def _undefined(n: int, mean: float, confidence: float, method: str, unit: str) -> Estimate:
    return Estimate(n, mean, math.nan, math.nan, math.nan, confidence, method, unit=unit)


def _t_critical(df: float, confidence: float) -> float:
    from scipy import stats as scipy_stats  # imported lazily: keeps spawned tournament workers light

    return float(scipy_stats.t.ppf(0.5 + confidence / 2.0, df))


def t_estimate(samples: Sequence[float] | np.ndarray, confidence: float = 0.95, unit: str = "pair") -> Estimate:
    """Mean with a Student-t confidence interval (the samples must be independent units of kind ``unit``)."""
    x = np.asarray(samples, dtype=np.float64)
    n = len(x)
    if n == 0:
        return _undefined(0, math.nan, confidence, "t", unit)
    mean = float(x.mean())
    if n < 2:
        return _undefined(n, mean, confidence, "t", unit)
    se = float(x.std(ddof=1) / math.sqrt(n))
    half = _t_critical(n - 1, confidence) * se
    return Estimate(n, mean, se, mean - half, mean + half, confidence, "t", float(n - 1), unit)


def bootstrap_estimate(
    groups: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
    unit: str = "pair",
) -> Estimate:
    """Seeded percentile bootstrap of ``sum_k weights[k] * mean(groups[k])``.

    One group is the ordinary bootstrap of a mean; several groups are resampled *independently* (use
    this for the equal-weight average over opponents). The resampled elements are the independent
    units (sessions or pairs, named by ``unit``). Deterministic for a given ``seed``.
    """
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    total_n = sum(len(a) for a in arrays)
    if not arrays:
        return _undefined(0, math.nan, confidence, "bootstrap", unit)
    w = np.full(len(arrays), 1.0 / len(arrays)) if weights is None else np.asarray(weights, dtype=np.float64)
    if any(len(a) < 2 for a in arrays):
        return _undefined(total_n, math.nan, confidence, "bootstrap", unit)
    point = float(sum(wk * a.mean() for wk, a in zip(w, arrays)))
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_resamples)
    for wk, a in zip(w, arrays):
        chunk = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // len(a))
        for start in range(0, n_resamples, chunk):
            size = min(chunk, n_resamples - start)
            boot[start : start + size] += wk * a[rng.integers(0, len(a), size=(size, len(a)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return Estimate(total_n, point, float(boot.std(ddof=1)), float(low), float(high), confidence, "bootstrap", unit=unit)


def _combine_equal_weight(estimates: Sequence[Estimate], confidence: float) -> Estimate:
    """t-interval of the plain average of independent estimates (Welch-Satterthwaite degrees of freedom)."""
    k = len(estimates)
    unit = estimates[0].unit if estimates else "pair"
    if k == 0 or any(math.isnan(e.se) for e in estimates):
        return _undefined(sum(e.n for e in estimates), math.nan, confidence, "t", unit)
    mean = sum(e.mean for e in estimates) / k
    terms = [(e.se / k) ** 2 for e in estimates]  # variance contribution of each component
    var = sum(terms)
    se = math.sqrt(var)
    df = (
        sum(e.df for e in estimates)
        if var == 0.0
        else var**2 / sum(t**2 / e.df for t, e in zip(terms, estimates))
    )
    half = _t_critical(df, confidence) * se
    return Estimate(sum(e.n for e in estimates), mean, se, mean - half, mean + half, confidence, "t", df, unit)


def pool_groups(
    groups: Sequence[np.ndarray],
    *,
    weighting: str = "equal",
    confidence: float = 0.95,
    method: str = "t",
    n_resamples: int = 2000,
    seed: int = 0,
    unit: str = "pair",
) -> Estimate:
    """Estimate the mean over several independent groups of unit profits (e.g. one per opponent).

    ``weighting="equal"``   every group (opponent) counts the same, whatever its size.
    ``weighting="sample"``  every unit counts the same (the groups are simply concatenated).
    A single group reduces to :func:`estimate_mean`.
    """
    if weighting not in ("equal", "sample"):
        raise ValueError("weighting must be 'equal' or 'sample'")
    nonempty = [np.asarray(g, dtype=np.float64) for g in groups if len(g)]
    if weighting == "sample" and nonempty:
        nonempty = [np.concatenate(nonempty)]
    if method == "bootstrap":
        return bootstrap_estimate(nonempty, confidence=confidence, n_resamples=n_resamples, seed=seed, unit=unit)
    if method != "t":
        raise ValueError("method must be 't' or 'bootstrap'")
    if len(nonempty) == 1:
        return t_estimate(nonempty[0], confidence, unit)
    return _combine_equal_weight([t_estimate(g, confidence, unit) for g in nonempty], confidence)


def estimate_mean(
    samples: np.ndarray,
    *,
    confidence: float = 0.95,
    method: str = "t",
    n_resamples: int = 2000,
    seed: int = 0,
    unit: str = "pair",
) -> Estimate:
    """Mean of independent unit profits with a ``method`` ("t" or "bootstrap") confidence interval."""
    return pool_groups(
        [samples], confidence=confidence, method=method, n_resamples=n_resamples, seed=seed, unit=unit
    )


def cumulative_with_luck_band(
    unit_profits: np.ndarray, confidence: float = 0.95, *, unit_sd: float | None = None, df: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running total of unit profits plus the band a *zero-edge* player's total would stay inside.

    The band is ``+- t * s * sqrt(k)`` after ``k`` units: a total that leaves the band is evidence of a
    real edge (or deficit), one inside it is indistinguishable from luck. By default ``s`` is the sample
    standard deviation of the unit profits and ``t`` has ``len - 1`` degrees of freedom (units assumed
    independent). Pass ``unit_sd`` / ``df`` to use another standard deviation per unit - e.g. one derived
    from session totals when the units (pairs) of a session are dependent, which makes the final width
    equal the session-level confidence interval. Returns ``(cumulative, band_low, band_high)``.
    """
    x = np.asarray(unit_profits, dtype=np.float64)
    if len(x) < 2 and unit_sd is None:
        nan = np.full(len(x), math.nan)
        return np.cumsum(x), nan, nan
    sd = float(x.std(ddof=1)) if unit_sd is None else unit_sd
    k = np.arange(1, len(x) + 1)
    half = _t_critical(len(x) - 1 if df is None else df, confidence) * sd * np.sqrt(k)
    return np.cumsum(x), -half, half


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a proportion (well behaved for small n and rates near 0 or 1)."""
    from scipy import stats as scipy_stats

    if n == 0:
        return math.nan, math.nan
    z = float(scipy_stats.norm.ppf(0.5 + confidence / 2.0))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------------------
# Verdicts (the only place where numbers become words)
# --------------------------------------------------------------------------------------
class Verdict(NamedTuple):
    code: str  # WIN | LOSE | DRAW | INSUFFICIENT, from the first agent's point of view
    text: str


def compare_verdict(name_a: str, name_b: str, a_minus_b: Estimate) -> Verdict:
    """Word a comparison of ``a`` against ``b`` from the estimate of (A's result - B's result).

    ``"A > B (significant)"`` only if the CI is entirely above 0 (and there are enough independent units:
    :func:`min_units_for_verdict` of the estimate's own unit - sessions by default),
    ``"A < B (significant)"`` if entirely below, otherwise ``"no significant difference"``. A draw
    is *absence of evidence*, not evidence of equal strength.
    """
    needed = min_units_for_verdict(a_minus_b.unit)
    if math.isnan(a_minus_b.ci_low) or a_minus_b.n < needed:
        return Verdict(INSUFFICIENT, f"insufficient data (fewer than {needed} independent {a_minus_b.unit}s)")
    if a_minus_b.ci_low > 0.0:
        return Verdict(WIN, f"{name_a} > {name_b} (significant)")
    if a_minus_b.ci_high < 0.0:
        return Verdict(LOSE, f"{name_a} < {name_b} (significant)")
    return Verdict(DRAW, "no significant difference")


# --------------------------------------------------------------------------------------
# Playing style and outcome counters (additive, so they pool across matches by plain addition)
# --------------------------------------------------------------------------------------
def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


@dataclass
class PlayerTally:
    """Everything counted about one player in a match; pool matches with ``+`` / :func:`sum_tallies`.

    ``vpip_hands`` / ``pfr_hands`` count hands in which the player voluntarily put chips in
    (call/bet/raise) or raised preflop; ``three_bet`` counts re-raises of an open-raise out of
    ``three_bet_opportunity`` situations. ``action_counts[street, kind]`` follows ``STREET_NAMES`` x
    ``ACTION_KINDS``. ``end_profit_bb`` is the profit split by where the hand ended (``END_CATEGORIES``).
    """

    hands: int = 0
    hands_won: int = 0
    showdowns: int = 0
    showdowns_won: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    three_bet: int = 0
    three_bet_opportunity: int = 0
    action_counts: np.ndarray = field(default_factory=lambda: np.zeros((NUM_STREETS, len(ACTION_KINDS)), np.int64))
    end_hands: np.ndarray = field(default_factory=lambda: np.zeros(len(END_CATEGORIES), np.int64))
    end_profit_bb: np.ndarray = field(default_factory=lambda: np.zeros(len(END_CATEGORIES)))

    _COUNTERS = ("hands", "hands_won", "showdowns", "showdowns_won", "vpip_hands", "pfr_hands", "three_bet", "three_bet_opportunity")
    _VECTORS = ("action_counts", "end_hands", "end_profit_bb")

    @classmethod
    def from_arrays(cls, arrays: HandArrays, player: int, big_blind: int) -> "PlayerTally":
        """Tally of ``player`` (0 = A, 1 = B) over all hands of ``arrays``."""
        profit = (arrays.profit_chips.astype(np.int64) * (1 if player == 0 else -1)) / big_blind
        counts = arrays.action_counts[:, player].astype(np.int64)  # (hands, street, kind)
        preflop = counts[:, 0]
        won = profit > 0
        category = arrays.end_category
        n_categories = len(END_CATEGORIES)
        return cls(
            hands=len(arrays),
            hands_won=int(won.sum()),
            showdowns=int(arrays.showdown.sum()),
            showdowns_won=int((won & arrays.showdown).sum()),
            vpip_hands=int((preflop[:, [KIND_INDEX["call"], *_BET_KINDS]].sum(axis=1) > 0).sum()),
            pfr_hands=int((preflop[:, list(_BET_KINDS)].sum(axis=1) > 0).sum()),
            three_bet=int(arrays.three_bet[:, player].sum()),
            three_bet_opportunity=int(arrays.three_bet_opportunity[:, player].sum()),
            action_counts=counts.sum(axis=0),
            end_hands=np.bincount(category, minlength=n_categories).astype(np.int64),
            end_profit_bb=np.bincount(category, weights=profit, minlength=n_categories),
        )

    def __add__(self, other: "PlayerTally") -> "PlayerTally":
        merged = {name: getattr(self, name) + getattr(other, name) for name in (*self._COUNTERS, *self._VECTORS)}
        return PlayerTally(**merged)

    # ---- serialisation (JSON friendly: nested dicts keyed by street / kind / end category)
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {name: int(getattr(self, name)) for name in self._COUNTERS}
        d["action_counts"] = {
            street: {kind: int(self.action_counts[s, k]) for k, kind in enumerate(ACTION_KINDS)}
            for s, street in enumerate(STREET_NAMES)
        }
        d["end_hands"] = {cat: int(self.end_hands[i]) for i, cat in enumerate(END_CATEGORIES)}
        d["end_profit_bb"] = {cat: float(self.end_profit_bb[i]) for i, cat in enumerate(END_CATEGORIES)}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlayerTally":
        return cls(
            **{name: int(d[name]) for name in cls._COUNTERS},
            action_counts=np.array([[d["action_counts"][s][k] for k in ACTION_KINDS] for s in STREET_NAMES], np.int64),
            end_hands=np.array([d["end_hands"][c] for c in END_CATEGORIES], np.int64),
            end_profit_bb=np.array([d["end_profit_bb"][c] for c in END_CATEGORIES], np.float64),
        )

    # ---- derived metrics
    def metrics(self) -> dict[str, float]:
        """Rates and aggression indicators (``NaN`` where the denominator is zero).

        ``aggression_factor``  = (bets + raises) / calls over all streets (checks do not count);
        ``*_postflop``         the same restricted to flop/turn/river, the convention of poker trackers;
        ``aggression_freq_postflop`` = (bets + raises) / all postflop actions;
        ``fold_freq`` / ``call_freq`` / ``raise_freq`` split ALL decisions into fold, check-or-call,
        bet-or-raise; ``three_bet`` = 3-bets / 3-bet opportunities.
        """
        c = self.action_counts
        calls, aggressive = c[:, KIND_INDEX["call"]], c[:, list(_BET_KINDS)].sum(axis=1)
        decisions = int(c.sum())
        folds = int(c[:, KIND_INDEX["fold"]].sum())
        passive = int(c[:, [KIND_INDEX["check"], KIND_INDEX["call"]]].sum())
        return {
            "hands": float(self.hands),
            "win_rate": _ratio(self.hands_won, self.hands),
            "showdown_rate": _ratio(self.showdowns, self.hands),
            "showdown_win_rate": _ratio(self.showdowns_won, self.showdowns),
            "vpip": _ratio(self.vpip_hands, self.hands),
            "pfr": _ratio(self.pfr_hands, self.hands),
            "three_bet": _ratio(self.three_bet, self.three_bet_opportunity),
            "aggression_factor": _ratio(aggressive.sum(), calls.sum()),
            "aggression_factor_postflop": _ratio(aggressive[_POSTFLOP].sum(), calls[_POSTFLOP].sum()),
            "aggression_freq_postflop": _ratio(aggressive[_POSTFLOP].sum(), c[_POSTFLOP].sum()),
            "fold_freq": _ratio(folds, decisions),
            "call_freq": _ratio(passive, decisions),
            "raise_freq": _ratio(aggressive.sum(), decisions),
        }

    def per_street(self) -> list[dict[str, Any]]:
        """One row per street: raw action counts, their frequencies and the street's aggression factor."""
        rows = []
        for s, street in enumerate(STREET_NAMES):
            c = self.action_counts[s]
            decisions = int(c.sum())
            aggressive = int(c[list(_BET_KINDS)].sum())
            rows.append(
                {
                    "street": street,
                    "decisions": decisions,
                    **{kind: int(c[k]) for k, kind in enumerate(ACTION_KINDS)},
                    "fold_freq": _ratio(c[KIND_INDEX["fold"]], decisions),
                    "call_freq": _ratio(c[KIND_INDEX["check"]] + c[KIND_INDEX["call"]], decisions),
                    "raise_freq": _ratio(aggressive, decisions),
                    "aggression_factor": _ratio(aggressive, c[KIND_INDEX["call"]]),
                }
            )
        return rows

    def profit_by_end(self) -> list[dict[str, Any]]:
        """Profit split by where the hand ended (fold on a street, or showdown).

        ``bb_per_100_contribution`` sums over the rows to the player's overall BB/100, so it shows
        *where* the win rate comes from (e.g. folds preflop vs showdowns).
        """
        total = int(self.end_hands.sum())
        return [
            {
                "ended": category,
                "hands": int(self.end_hands[i]),
                "share_of_hands": _ratio(self.end_hands[i], total),
                "profit_bb": float(self.end_profit_bb[i]),
                "bb_per_hand": _ratio(self.end_profit_bb[i], self.end_hands[i]),
                "bb_per_100_contribution": bb_per_100(float(self.end_profit_bb[i]), total),
            }
            for i, category in enumerate(END_CATEGORIES)
        ]


def sum_tallies(tallies: Sequence[PlayerTally]) -> PlayerTally:
    """Pool tallies of several matches (the identity for an empty sequence)."""
    total = PlayerTally()
    for t in tallies:
        total = total + t
    return total


def latency_summary(latency_ms: np.ndarray) -> dict[str, float]:
    """Per-decision latency: count, mean, median, 95th percentile and maximum, in milliseconds."""
    x = np.asarray(latency_ms, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0, "mean_ms": math.nan, "median_ms": math.nan, "p95_ms": math.nan, "max_ms": math.nan}
    return {
        "n": int(len(x)),
        "mean_ms": float(x.mean()),
        "median_ms": float(np.median(x)),
        "p95_ms": float(np.percentile(x, 95)),
        "max_ms": float(x.max()),
    }
