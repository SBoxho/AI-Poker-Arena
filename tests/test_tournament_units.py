"""Independent units of the confidence intervals: sessions (default) versus duplicate pairs.

The key scenario is an *adaptive* agent: it learns about its opponent during a session, so the pairs
of that session share a common "session effect" and a pair-level interval is too narrow. The synthetic
series below reproduce exactly that structure with a known truth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from poker_arena.config import GameConfig
from poker_arena.tournament.config import MIN_SESSIONS_FOR_CI, MatchConfig
from poker_arena.tournament.stats import (
    DRAW,
    INSUFFICIENT,
    LOSE,
    MIN_SESSIONS_FOR_VERDICT,
    MIN_UNITS_FOR_VERDICT,
    WIN,
    Estimate,
    cumulative_with_luck_band,
    compare_verdict,
    estimate_mean,
    min_units_for_verdict,
    pool_groups,
    scale_sessions_to_bb_per_100,
    session_totals,
    t_estimate,
)
from poker_arena.tournament.summary import (
    UnitSamples,
    build_standings,
    ci_unit_info,
    estimate_fields,
    match_estimate,
    pair_level_estimate,
    unit_samples,
)

BIG_BLIND = 10
PAIRS_PER_SESSION = 25


def cfg_for(sessions: int, **changes) -> MatchConfig:
    """A config with exactly ``sessions`` sessions of ``2 * PAIRS_PER_SESSION`` hands."""
    fields = dict(game=GameConfig(), num_hands=sessions * 2 * PAIRS_PER_SESSION, session_hands=2 * PAIRS_PER_SESSION, seed=3)
    fields.update(changes)
    return MatchConfig(**fields)


def adaptive_profits(rng: np.random.Generator, sessions: int, mean_per_pair: float = 0.5,
                     session_sd: float = 1.0, pair_sd: float = 1.0) -> np.ndarray:  # fmt: skip
    """Per-hand chip profits whose pair sums are ``mean + session effect + noise`` (BB units x 10).

    The second hand of every pair is 0, so a pair's sum is the first hand's profit. All pairs of a
    session share one random effect: this is what learning about the opponent does.
    """
    effect = rng.normal(0.0, session_sd, size=(sessions, 1))
    pairs = mean_per_pair + effect + rng.normal(0.0, pair_sd, size=(sessions, PAIRS_PER_SESSION))
    hands = np.zeros((sessions * PAIRS_PER_SESSION, 2))
    hands[:, 0] = pairs.ravel() * BIG_BLIND
    return hands.ravel()


# ----------------------------------------------------------------------------- config
def test_ci_unit_defaults_to_sessions_and_validates() -> None:
    cfg = cfg_for(20)
    assert cfg.ci_unit == "session" and cfg.ci_unit_used == "session" and cfg.ci_unit_note is None
    assert MatchConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        MatchConfig(ci_unit="week")
    # configs written before the field existed load with the new default
    legacy = cfg.to_dict()
    del legacy["ci_unit"]
    assert MatchConfig.from_dict(legacy).ci_unit == "session"


@pytest.mark.parametrize("sessions", [1, 2, MIN_SESSIONS_FOR_CI - 1])
def test_too_few_sessions_fall_back_to_pairs_with_a_loud_note(sessions: int) -> None:
    cfg = cfg_for(sessions)
    assert cfg.ci_unit == "session" and cfg.ci_unit_used == "pair"
    note = cfg.ci_unit_note
    assert note is not None and str(sessions) in note and str(MIN_SESSIONS_FOR_CI) in note and "NOT independent" in note
    assert ci_unit_info(cfg) == {
        "requested": "session", "used": "pair", "min_sessions": MIN_SESSIONS_FOR_CI,
        "sessions_per_match": sessions, "note": note,
    }  # fmt: skip


def test_enough_sessions_use_sessions_and_pair_can_be_requested() -> None:
    assert cfg_for(MIN_SESSIONS_FOR_CI).ci_unit_used == "session"
    asked_for_pairs = cfg_for(20, ci_unit="pair")
    assert asked_for_pairs.ci_unit_used == "pair" and "requested" in asked_for_pairs.ci_unit_note
    assert cfg_for(2, duplicate=False).ci_unit_used == "hand" and cfg_for(20, duplicate=False).ci_unit_used == "session"


# ----------------------------------------------------------------------------- session arithmetic
def test_session_totals_and_scaling_keep_the_overall_bb_per_100() -> None:
    per_hand = np.arange(1.0, 11.0)  # 10 hands
    assert session_totals(per_hand, 4).tolist() == [10.0, 26.0, 19.0]  # last session has 2 hands
    assert session_totals(per_hand, 5).tolist() == [15.0, 40.0]
    assert session_totals(np.array([]), 4).size == 0
    scaled = scale_sessions_to_bb_per_100(session_totals(per_hand, 4), total_hands=10)
    assert scaled.mean() == pytest.approx(100.0 * per_hand.sum() / 10)  # exactly the hand-weighted BB/100
    even = scale_sessions_to_bb_per_100(session_totals(per_hand, 5), total_hands=10)
    assert even.tolist() == pytest.approx([100 * 15 / 5, 100 * 40 / 5])  # equal sessions: plain session BB/100


def test_unit_samples_means_agree_and_match_the_hand_weighted_win_rate() -> None:
    rng = np.random.default_rng(1)
    cfg = cfg_for(12)
    profit = adaptive_profits(rng, 12)
    samples = unit_samples(profit, cfg)
    overall = 100.0 * profit.sum() / BIG_BLIND / len(profit)
    assert samples.fine.mean() == pytest.approx(overall) and samples.session.mean() == pytest.approx(overall)
    assert len(samples.fine) == 12 * PAIRS_PER_SESSION and len(samples.session) == 12
    assert samples.negated().session.mean() == pytest.approx(-overall)
    assert np.array_equal(samples.for_unit("session"), samples.session)
    assert np.array_equal(samples.for_unit("pair"), samples.fine) and np.array_equal(samples.for_unit("hand"), samples.fine)


def test_shorter_last_session_counts_proportionally() -> None:
    cfg = MatchConfig(num_hands=10, session_hands=4)
    profit = np.array([10, 0, 10, 0, 10, 0, 10, 0, 20, 0], dtype=float) * 1  # chips; big blind 10
    samples = unit_samples(profit, cfg)
    assert len(samples.session) == 3
    assert samples.session.mean() == pytest.approx(100.0 * profit.sum() / BIG_BLIND / 10)


# ----------------------------------------------------------------------------- the adaptive scenario
def test_session_interval_is_wider_than_the_pair_interval_for_adaptive_series() -> None:
    cfg = cfg_for(40)
    samples = unit_samples(adaptive_profits(np.random.default_rng(7), 40), cfg)
    session, pair = estimate_mean(samples.session, unit="session"), estimate_mean(samples.fine)
    assert session.mean == pytest.approx(pair.mean)  # same point estimate
    assert session.se > 2.0 * pair.se  # design effect 1 + (m - 1) * rho = 13 -> SE x 3.6
    assert (session.ci_high - session.ci_low) > 2.0 * (pair.ci_high - pair.ci_low)
    assert session.n == 40 and session.df == 39 and session.unit == "session" and pair.unit == "pair"


def test_without_a_session_effect_both_units_agree() -> None:
    """Non-adaptive agents (independent pairs): the session interval is no wider than chance allows."""
    cfg = cfg_for(80)
    samples = unit_samples(adaptive_profits(np.random.default_rng(8), 80, session_sd=0.0), cfg)
    ratio = estimate_mean(samples.session).se / estimate_mean(samples.fine).se
    assert 0.75 < ratio < 1.3


def test_session_intervals_cover_the_truth_and_pair_intervals_do_not() -> None:
    """Simulation: true mean 0.5 BB per pair (= 25 BB/100). 95% intervals over 600 replications."""
    rng = np.random.default_rng(2024)
    cfg = cfg_for(20)
    truth = 0.5 * 50  # BB/100: mean per pair x (100 hands / 2 hands per pair)
    session_hits = pair_hits = 0
    reps = 600
    for _ in range(reps):
        samples = unit_samples(adaptive_profits(rng, 20), cfg)
        s, p = t_estimate(samples.session, unit="session"), t_estimate(samples.fine)
        session_hits += s.ci_low <= truth <= s.ci_high
        pair_hits += p.ci_low <= truth <= p.ci_high
    assert 0.92 <= session_hits / reps <= 0.98  # honest
    assert pair_hits / reps < 0.75  # the pair-level interval is overconfident for adaptive play


def test_bootstrap_resamples_sessions() -> None:
    cfg = cfg_for(30, ci_method="bootstrap", bootstrap_resamples=400)
    samples = unit_samples(adaptive_profits(np.random.default_rng(9), 30), cfg)
    headline = match_estimate(samples, cfg, "k")
    pair = pair_level_estimate(samples, cfg, "k")
    assert headline.method == pair.method == "bootstrap" and headline.unit == "session" and headline.n == 30
    assert headline.se > 2.0 * pair.se
    assert headline == match_estimate(samples, cfg, "k")  # seeded
    assert headline.ci_low < headline.mean < headline.ci_high


def test_match_estimate_uses_the_configured_unit() -> None:
    samples = unit_samples(adaptive_profits(np.random.default_rng(10), 20), cfg_for(20))
    assert match_estimate(samples, cfg_for(20), "k").unit == "session"
    assert match_estimate(samples, cfg_for(20, ci_unit="pair"), "k").unit == "pair"
    assert match_estimate(samples, cfg_for(20, ci_unit="pair"), "k").se == pytest.approx(
        match_estimate(samples, cfg_for(3), "k").se
    )  # the fallback IS the pair-level interval (only the sample size of the config differs)


# ----------------------------------------------------------------------------- estimates and verdicts
def test_estimate_unit_threads_through_pooling_and_serialisation() -> None:
    groups = [np.random.default_rng(i).normal(i, 1.0, 12) for i in range(3)]
    assert pool_groups(groups, unit="session").unit == "session"
    assert pool_groups(groups, weighting="sample", unit="session").unit == "session"
    assert pool_groups(groups, method="bootstrap", n_resamples=200, unit="session").unit == "session"
    est = t_estimate(groups[0], unit="session")
    assert Estimate.from_dict(est.to_dict()).unit == "session" and est.negated().unit == "session"
    stored_before_units_existed = {k: v for k, v in est.to_dict().items() if k != "unit"}
    assert Estimate.from_dict(stored_before_units_existed).unit == "pair"


def test_minimum_data_rule_is_expressed_in_sessions_for_session_estimates() -> None:
    assert min_units_for_verdict("session") == MIN_SESSIONS_FOR_VERDICT == MIN_SESSIONS_FOR_CI
    assert min_units_for_verdict("pair") == min_units_for_verdict("hand") == MIN_UNITS_FOR_VERDICT
    wide = lambda n, unit: Estimate(n, 10.0, 1.0, 8.0, 12.0, 0.95, "t", n - 1.0, unit)  # noqa: E731
    assert compare_verdict("A", "B", wide(MIN_SESSIONS_FOR_VERDICT, "session")).code == WIN
    assert wide(MIN_SESSIONS_FOR_VERDICT, "session").significant
    short = compare_verdict("A", "B", wide(MIN_SESSIONS_FOR_VERDICT - 1, "session"))
    assert short.code == INSUFFICIENT and str(MIN_SESSIONS_FOR_VERDICT) in short.text and "sessions" in short.text
    assert compare_verdict("A", "B", wide(MIN_SESSIONS_FOR_VERDICT, "pair")).code == INSUFFICIENT  # pairs need 30
    assert compare_verdict("B", "A", wide(MIN_SESSIONS_FOR_VERDICT, "session").negated()).code == LOSE


def test_clear_cases_keep_their_verdict_at_either_level() -> None:
    cfg = cfg_for(20)
    rng = np.random.default_rng(11)
    for mean, expected in ((8.0, WIN), (-8.0, LOSE), (0.0, DRAW)):
        samples = unit_samples(adaptive_profits(rng, 20, mean_per_pair=mean, session_sd=0.3, pair_sd=1.0), cfg)
        for unit_cfg in (cfg, cfg_for(20, ci_unit="pair")):
            assert compare_verdict("A", "B", match_estimate(samples, unit_cfg, "k")).code == expected


def test_estimate_fields_report_both_levels_for_both_sides() -> None:
    cfg = cfg_for(20)
    profit = adaptive_profits(np.random.default_rng(12), 20, mean_per_pair=2.0)
    fields = estimate_fields(profit, cfg, "A__vs__B", "A", "B")
    a, b = fields["players"]["a"], fields["players"]["b"]
    assert fields["ci_unit_used"] == "session" and fields["ci_units"] == 20 and fields["num_sessions"] == 20
    assert a["bb_per_100"]["unit"] == "session" and a["bb_per_100_pair"]["unit"] == "pair"
    assert a["bb_per_100"]["mean"] == pytest.approx(a["bb_per_100_pair"]["mean"]) == pytest.approx(-b["bb_per_100"]["mean"])
    assert a["bb_per_100"]["se"] > a["bb_per_100_pair"]["se"] and fields["session_over_pair_se"] > 2
    assert a["bb_per_100"]["ci_low"] == pytest.approx(-b["bb_per_100"]["ci_high"])
    assert a["verdict"]["code"] == WIN and b["verdict"]["code"] == LOSE
    assert a["verdict"]["text"] == "A > B (significant)"


def test_standings_pool_sessions_and_flag_the_unit() -> None:
    cfg = cfg_for(12)
    rng = np.random.default_rng(13)
    agents = ["A", "B", "C"]
    samples: dict[tuple[str, str], UnitSamples] = {}
    for x, y, mean in (("A", "B", 4.0), ("A", "C", 3.0), ("B", "C", 0.0)):
        units = unit_samples(adaptive_profits(rng, 12, mean_per_pair=mean), cfg)
        samples[x, y], samples[y, x] = units, units.negated()
    standings = build_standings(agents, samples, cfg)
    rows = {r["agent"]: r for r in standings["leaderboard"]}
    assert [r["agent"] for r in standings["leaderboard"]][0] == "A" and rows["A"]["ci_unit"] == "session"
    assert rows["A"]["bb_per_100"]["unit"] == "session" and rows["A"]["bb_per_100"]["n"] == 24  # 2 opponents x 12 sessions
    assert rows["A"]["bb_per_100_pair"]["unit"] == "pair" and rows["A"]["bb_per_100_pair"]["n"] == 2 * 12 * PAIRS_PER_SESSION
    assert rows["A"]["bb_per_100"]["se"] > rows["A"]["bb_per_100_pair"]["se"]
    assert standings["ci_unit"]["used"] == "session" and standings["ci_unit"]["note"] is None

    few = cfg_for(3)
    flagged = build_standings(agents, {k: unit_samples(adaptive_profits(rng, 3), few) for k in samples}, few)
    assert flagged["ci_unit"]["used"] == "pair" and "NOT independent" in flagged["ci_unit"]["note"]
    assert all(r["ci_unit"] == "pair" and r["bb_per_100"]["unit"] == "pair" for r in flagged["leaderboard"])


# ----------------------------------------------------------------------------- the luck band
def test_luck_band_with_explicit_unit_sd_and_df() -> None:
    units = np.random.default_rng(14).normal(0.0, 5.0, 50)
    _, low, high = cumulative_with_luck_band(units, unit_sd=10.0, df=7)
    from scipy import stats

    assert high[-1] == pytest.approx(stats.t.ppf(0.975, 7) * 10.0 * math.sqrt(50)) and np.allclose(low, -high)
    default = cumulative_with_luck_band(units)[2]
    assert default[-1] == pytest.approx(stats.t.ppf(0.975, 49) * units.std(ddof=1) * math.sqrt(50))
