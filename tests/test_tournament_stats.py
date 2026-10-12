"""Unit tests for ``poker_arena.tournament.stats`` and ``arrays`` on synthetic data with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from poker_arena.tournament.arrays import (
    ACTION_KINDS,
    END_CATEGORIES,
    KIND_INDEX,
    STREET_NAMES,
    HandArrays,
    HandArraysBuilder,
)
from poker_arena.tournament.stats import (
    DRAW,
    INSUFFICIENT,
    LOSE,
    MIN_UNITS_FOR_VERDICT,
    WIN,
    Estimate,
    PlayerTally,
    bb_per_100,
    bootstrap_estimate,
    compare_verdict,
    cumulative_with_luck_band,
    estimate_mean,
    latency_summary,
    pair_sums,
    pool_groups,
    scale_to_bb_per_100,
    sum_tallies,
    t_estimate,
    wilson_interval,
)


# ----------------------------------------------------------------------------- conversions
def test_bb_per_100_arithmetic() -> None:
    assert bb_per_100(50.0, 1000) == pytest.approx(5.0)
    assert bb_per_100(-3.0, 200) == pytest.approx(-1.5)
    assert math.isnan(bb_per_100(1.0, 0))


def test_pair_sums_adds_mirrored_hands_and_rejects_odd_length() -> None:
    assert pair_sums(np.array([1, -1, 5, 2, 0, 0])).tolist() == [0.0, 7.0, 0.0]
    with pytest.raises(ValueError):
        pair_sums(np.array([1, 2, 3]))


def test_scaling_makes_the_mean_the_bb_per_100_win_rate() -> None:
    per_hand_bb = np.array([2.0, -1.0, 3.0, -4.0, 0.5, 0.5])  # 3 pairs, 6 hands
    scaled = scale_to_bb_per_100(pair_sums(per_hand_bb), hands_per_unit=2)
    assert scaled.mean() == pytest.approx(bb_per_100(per_hand_bb.sum(), 6))
    # without pairing every hand is its own unit
    assert scale_to_bb_per_100(per_hand_bb, 1).mean() == pytest.approx(bb_per_100(per_hand_bb.sum(), 6))


# ----------------------------------------------------------------------------- t-interval
def test_t_estimate_matches_hand_computation() -> None:
    est = t_estimate([1, 2, 3, 4, 5], confidence=0.95)
    se = math.sqrt(2.5 / 5)
    half = 2.7764451051977987 * se  # t_{0.975, 4}
    assert (est.n, est.mean, est.method, est.df) == (5, 3.0, "t", 4.0)
    assert est.se == pytest.approx(se)
    assert est.ci_low == pytest.approx(3 - half) and est.ci_high == pytest.approx(3 + half)
    reference = scipy_stats.t.interval(0.95, 4, loc=3, scale=se)
    assert (est.ci_low, est.ci_high) == pytest.approx(reference)


def test_t_estimate_higher_confidence_is_wider() -> None:
    x = np.random.default_rng(0).normal(size=50)
    assert t_estimate(x, 0.99).ci_high - t_estimate(x, 0.99).ci_low > t_estimate(x, 0.90).ci_high - t_estimate(x, 0.90).ci_low


def test_t_estimate_coverage_is_close_to_nominal() -> None:
    """Simulation sanity check: ~95% of intervals from N(1, 3^2) samples contain the true mean 1."""
    rng = np.random.default_rng(123)
    reps, n, true_mean = 2000, 40, 1.0
    hits = sum(
        (lambda e: e.ci_low <= true_mean <= e.ci_high)(t_estimate(rng.normal(true_mean, 3.0, n)))
        for _ in range(reps)
    )
    assert 0.93 <= hits / reps <= 0.97


def test_t_estimate_undefined_cases_are_nan_not_errors() -> None:
    empty, single = t_estimate([]), t_estimate([4.0])
    assert empty.n == 0 and math.isnan(empty.mean) and not empty.excludes_zero
    assert single.n == 1 and single.mean == 4.0 and math.isnan(single.se) and math.isnan(single.ci_low)
    assert not single.excludes_zero and not single.significant


def test_zero_variance_gives_a_degenerate_interval() -> None:
    est = t_estimate(np.zeros(100))
    assert (est.mean, est.se, est.ci_low, est.ci_high) == (0.0, 0.0, 0.0, 0.0)
    assert not est.excludes_zero and not est.significant
    constant = t_estimate(np.full(100, 3.0))
    assert constant.excludes_zero and constant.significant


def test_excludes_zero_and_significance_guard_on_sample_size() -> None:
    small = t_estimate(np.full(5, 1.0) + np.arange(5) * 0.001)
    assert small.excludes_zero and not small.significant  # n < MIN_UNITS_FOR_VERDICT
    large = t_estimate(np.random.default_rng(1).normal(2.0, 1.0, MIN_UNITS_FOR_VERDICT))
    assert large.excludes_zero and large.significant
    straddling = t_estimate(np.random.default_rng(2).normal(0.0, 1.0, 100))
    assert straddling.ci_low < 0 < straddling.ci_high and not straddling.significant


def test_estimate_negation_swaps_the_interval_and_round_trips_through_dict() -> None:
    est = t_estimate([1.0, 2.0, 3.0, 5.0])
    neg = est.negated()
    assert (neg.mean, neg.ci_low, neg.ci_high, neg.se) == (-est.mean, -est.ci_high, -est.ci_low, est.se)
    assert Estimate.from_dict(est.to_dict()) == est
    undefined = t_estimate([1.0])
    restored = Estimate.from_dict({**undefined.to_dict(), "se": None, "ci_low": None, "ci_high": None, "df": None})
    assert math.isnan(restored.se) and math.isnan(restored.ci_low)


# ----------------------------------------------------------------------------- bootstrap
def test_bootstrap_is_deterministic_per_seed_and_close_to_the_t_interval() -> None:
    x = np.random.default_rng(5).normal(1.0, 2.0, 400)
    a = bootstrap_estimate([x], n_resamples=1000, seed=11)
    b = bootstrap_estimate([x], n_resamples=1000, seed=11)
    c = bootstrap_estimate([x], n_resamples=1000, seed=12)
    assert a == b and a != c
    t = t_estimate(x)
    assert a.method == "bootstrap" and a.mean == pytest.approx(t.mean)
    assert a.se == pytest.approx(t.se, rel=0.15)
    width = t.ci_high - t.ci_low
    assert abs(a.ci_low - t.ci_low) < 0.15 * width and abs(a.ci_high - t.ci_high) < 0.15 * width


def test_bootstrap_with_weights_estimates_the_weighted_mean() -> None:
    rng = np.random.default_rng(6)
    g1, g2 = rng.normal(0.0, 1.0, 100), rng.normal(10.0, 1.0, 300)
    est = bootstrap_estimate([g1, g2], weights=[0.25, 0.75], n_resamples=500, seed=1)
    assert est.mean == pytest.approx(0.25 * g1.mean() + 0.75 * g2.mean())
    assert est.ci_low < est.mean < est.ci_high


def test_bootstrap_undefined_for_tiny_samples() -> None:
    assert math.isnan(bootstrap_estimate([np.array([1.0])]).ci_low)
    assert math.isnan(bootstrap_estimate([]).mean)


def test_bootstrap_chunking_does_not_change_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory-bounded chunks must draw the same random numbers as one big matrix."""
    import poker_arena.tournament.stats as stats_module

    x = np.random.default_rng(8).normal(size=60)
    whole = bootstrap_estimate([x], n_resamples=200, seed=3)
    monkeypatch.setattr(stats_module, "_BOOTSTRAP_CHUNK_ELEMENTS", 60 * 7)
    assert bootstrap_estimate([x], n_resamples=200, seed=3) == whole


# ----------------------------------------------------------------------------- pooling
def test_equal_weight_pooling_matches_the_analytic_formula() -> None:
    rng = np.random.default_rng(9)
    g1, g2 = rng.normal(1.0, 4.0, 30), rng.normal(5.0, 1.0, 300)
    e1, e2 = t_estimate(g1), t_estimate(g2)
    pooled = pool_groups([g1, g2], weighting="equal")
    var = (e1.se / 2) ** 2 + (e2.se / 2) ** 2
    df = var**2 / (((e1.se / 2) ** 4) / e1.df + ((e2.se / 2) ** 4) / e2.df)
    assert pooled.mean == pytest.approx((g1.mean() + g2.mean()) / 2)
    assert pooled.se == pytest.approx(math.sqrt(var))
    assert pooled.df == pytest.approx(df)
    half = scipy_stats.t.ppf(0.975, df) * math.sqrt(var)
    assert pooled.ci_high - pooled.mean == pytest.approx(half)
    assert pooled.n == 330
    assert min(e1.df, e2.df) <= pooled.df <= e1.df + e2.df  # Welch-Satterthwaite bounds


def test_sample_weighted_pooling_is_the_t_interval_of_the_concatenation() -> None:
    rng = np.random.default_rng(10)
    g1, g2 = rng.normal(0.0, 1.0, 20), rng.normal(2.0, 3.0, 80)
    assert pool_groups([g1, g2], weighting="sample") == t_estimate(np.concatenate([g1, g2]))


def test_equal_and_sample_weighting_agree_for_equal_sized_groups() -> None:
    rng = np.random.default_rng(11)
    groups = [rng.normal(m, 2.0, 50) for m in (0.0, 1.0, 2.0)]
    assert pool_groups(groups, weighting="equal").mean == pytest.approx(pool_groups(groups, weighting="sample").mean)


def test_pooling_one_group_equals_estimate_mean_and_empty_input_is_undefined() -> None:
    x = np.random.default_rng(12).normal(size=40)
    assert pool_groups([x]) == estimate_mean(x)
    assert pool_groups([]).n == 0 and math.isnan(pool_groups([]).mean)
    assert pool_groups([np.array([]), x]) == estimate_mean(x)  # empty groups are ignored
    with pytest.raises(ValueError):
        pool_groups([x], weighting="median")
    with pytest.raises(ValueError):
        pool_groups([x], method="magic")


def test_bootstrap_pooling_equal_weights_groups_independently() -> None:
    rng = np.random.default_rng(13)
    g1, g2 = rng.normal(0.0, 1.0, 40), rng.normal(4.0, 1.0, 400)
    est = pool_groups([g1, g2], weighting="equal", method="bootstrap", n_resamples=300, seed=2)
    assert est.mean == pytest.approx((g1.mean() + g2.mean()) / 2)
    assert est.se > pool_groups([g1, g2], weighting="sample", method="bootstrap", n_resamples=300, seed=2).se * 0.5


def test_estimate_mean_dispatches_on_method() -> None:
    x = np.random.default_rng(14).normal(size=30)
    assert estimate_mean(x, method="t").method == "t"
    assert estimate_mean(x, method="bootstrap", n_resamples=200).method == "bootstrap"


# ----------------------------------------------------------------------------- verdicts
def _estimate(mean: float, se: float, n: int = 200) -> Estimate:
    half = 1.96 * se
    return Estimate(n, mean, se, mean - half, mean + half, 0.95, "t", n - 1.0)


def test_verdict_win_lose_draw_with_exact_wording() -> None:
    assert compare_verdict("A", "B", _estimate(10.0, 2.0)) == (WIN, "A > B (significant)")
    assert compare_verdict("A", "B", _estimate(-10.0, 2.0)) == (LOSE, "A < B (significant)")
    assert compare_verdict("A", "B", _estimate(1.0, 2.0)) == (DRAW, "no significant difference")


def test_verdict_refuses_with_too_little_data_or_undefined_interval() -> None:
    assert compare_verdict("A", "B", _estimate(100.0, 1.0, n=MIN_UNITS_FOR_VERDICT - 1)).code == INSUFFICIENT
    assert compare_verdict("A", "B", t_estimate([1.0])).code == INSUFFICIENT
    assert compare_verdict("A", "B", t_estimate([])).code == INSUFFICIENT


def test_verdict_boundary_when_interval_just_touches_zero() -> None:
    touching = Estimate(100, 2.0, 1.0, 0.0, 4.0, 0.95, "t", 99.0)
    assert compare_verdict("A", "B", touching).code == DRAW  # lower bound must be strictly above 0


def test_verdict_text_never_claims_superiority_without_significance() -> None:
    for mean in (-3.0, -1.0, 0.0, 1.0, 3.0):
        for se in (0.5, 2.0, 10.0):
            verdict = compare_verdict("A", "B", _estimate(mean, se))
            if verdict.code in (DRAW, INSUFFICIENT):
                assert ">" not in verdict.text and "<" not in verdict.text and "better" not in verdict.text


def test_verdict_for_a_swapped_comparison_is_the_mirror_image() -> None:
    est = _estimate(8.0, 2.0)
    assert compare_verdict("A", "B", est).code == WIN and compare_verdict("B", "A", est.negated()).code == LOSE


# ----------------------------------------------------------------------------- proportions, curves
def test_wilson_interval_known_values_and_edges() -> None:
    low, high = wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 10)[0] == 0.0 and 0.0 < wilson_interval(0, 10)[1] < 0.4
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
    narrow = wilson_interval(500, 1000)
    assert narrow[1] - narrow[0] < high - low


def test_cumulative_luck_band() -> None:
    units = np.random.default_rng(15).normal(0.0, 5.0, 100)
    cumulative, low, high = cumulative_with_luck_band(units)
    assert np.allclose(cumulative, np.cumsum(units))
    assert np.allclose(low, -high) and np.all(np.diff(high) > 0)
    k = np.arange(1, 101)
    assert np.allclose(high, high[0] * np.sqrt(k))  # grows with sqrt(k)
    assert high[-1] == pytest.approx(scipy_stats.t.ppf(0.975, 99) * units.std(ddof=1) * 10)
    assert math.isnan(cumulative_with_luck_band(np.array([1.0]))[1][0])


def test_latency_summary() -> None:
    summary = latency_summary(np.arange(1.0, 101.0))
    assert summary["n"] == 100 and summary["mean_ms"] == pytest.approx(50.5)
    assert summary["median_ms"] == pytest.approx(50.5) and summary["p95_ms"] == pytest.approx(95.05)
    assert summary["max_ms"] == 100.0
    assert latency_summary(np.array([]))["n"] == 0 and math.isnan(latency_summary(np.array([]))["mean_ms"])


# ----------------------------------------------------------------------------- tallies
def _counts(player_actions: dict[tuple[int, str], int]) -> np.ndarray:
    counts = np.zeros((len(STREET_NAMES), len(ACTION_KINDS)), dtype=np.uint16)
    for (street, kind), n in player_actions.items():
        counts[street, KIND_INDEX[kind]] = n
    return counts


def synthetic_arrays() -> HandArrays:
    """Four hand-made hands (big blind = 10) whose statistics are worked out by hand in the tests."""
    hands = [
        # profit, A button, showdown, ended street, A actions, B actions, 3bet A, 3bet B, opp A, opp B
        (20, True, False, 0, {(0, "raise"): 1}, {(0, "fold"): 1}, (0, 0), (0, 1)),
        (-100, False, True, 3,
         {(0, "call"): 1, (1, "check"): 1, (1, "call"): 1, (2, "check"): 1, (3, "call"): 1},
         {(0, "raise"): 1, (1, "bet"): 1, (2, "bet"): 1, (3, "bet"): 1}, (0, 0), (1, 0)),
        (40, True, True, 3,
         {(0, "raise"): 1, (1, "bet"): 1, (2, "bet"): 1, (3, "raise"): 1},
         {(0, "call"): 1, (1, "call"): 1, (2, "call"): 1, (3, "call"): 1}, (0, 0), (0, 0)),
        (0, False, True, 3,
         {(0, "check"): 1, (1, "check"): 1, (2, "check"): 1, (3, "check"): 1},
         {(0, "call"): 1, (1, "check"): 1, (2, "check"): 1, (3, "check"): 1}, (0, 0), (0, 0)),
    ]  # fmt: skip
    builder = HandArraysBuilder()
    for profit, a_button, showdown, ended, a_actions, b_actions, (a3, b3), (a_opp, b_opp) in (
        (h[0], h[1], h[2], h[3], h[4], h[5], (h[6][0], h[6][1]), h[7]) for h in hands
    ):
        builder.add_hand(
            profit_chips=profit, a_on_button=a_button, showdown=showdown, ended_street=ended, pot_chips=2 * abs(profit) + 20,
            action_counts=np.stack([_counts(a_actions), _counts(b_actions)]),
            three_bet=(a3, b3), three_bet_opportunity=(a_opp, b_opp), latency_a_ms=[1.0, 2.0], latency_b_ms=[3.0],
        )  # fmt: skip
    return builder.build()


def test_tally_counts_and_metrics_match_hand_calculation_for_player_a() -> None:
    tally = PlayerTally.from_arrays(synthetic_arrays(), player=0, big_blind=10)
    assert (tally.hands, tally.hands_won, tally.showdowns, tally.showdowns_won) == (4, 2, 3, 1)
    m = tally.metrics()
    assert m["win_rate"] == pytest.approx(0.5) and m["showdown_rate"] == pytest.approx(0.75)
    assert m["showdown_win_rate"] == pytest.approx(1 / 3)
    assert m["vpip"] == pytest.approx(0.75)  # raise, call, raise (the big blind's check is not voluntary)
    assert m["pfr"] == pytest.approx(0.5)
    assert m["three_bet"] == pytest.approx(0.0)  # one opportunity (hand 2), not used
    assert m["aggression_factor"] == pytest.approx(5 / 3)  # (2 raises + 2 bets + 1 river raise) / 3 calls
    assert m["aggression_factor_postflop"] == pytest.approx(3 / 2)
    assert m["aggression_freq_postflop"] == pytest.approx(3 / 10)
    assert m["fold_freq"] == 0.0 and m["call_freq"] == pytest.approx(9 / 14) and m["raise_freq"] == pytest.approx(5 / 14)
    assert m["fold_freq"] + m["call_freq"] + m["raise_freq"] == pytest.approx(1.0)


def test_tally_for_player_b_uses_the_opponents_columns_and_negated_profit() -> None:
    tally = PlayerTally.from_arrays(synthetic_arrays(), player=1, big_blind=10)
    assert tally.hands_won == 1 and tally.showdowns_won == 1
    assert tally.three_bet_opportunity == 1 and tally.three_bet == 0
    m = tally.metrics()
    assert m["vpip"] == pytest.approx(0.75) and m["pfr"] == pytest.approx(0.25)
    assert m["fold_freq"] == pytest.approx(1 / 13)  # B acted 13 times, folding once


def test_profit_by_end_adds_up_to_the_overall_win_rate() -> None:
    arrays = synthetic_arrays()
    for player, sign in ((0, 1), (1, -1)):
        tally = PlayerTally.from_arrays(arrays, player, big_blind=10)
        rows = tally.profit_by_end()
        assert [r["ended"] for r in rows] == list(END_CATEGORIES)
        assert sum(r["hands"] for r in rows) == 4
        total_bb = sign * arrays.profit_chips.sum() / 10
        assert sum(r["bb_per_100_contribution"] for r in rows) == pytest.approx(bb_per_100(total_bb, 4))
    a_rows = PlayerTally.from_arrays(arrays, 0, 10).profit_by_end()
    assert a_rows[0]["hands"] == 1 and a_rows[0]["profit_bb"] == pytest.approx(2.0)
    assert a_rows[4]["hands"] == 3 and a_rows[4]["profit_bb"] == pytest.approx(-6.0)


def test_per_street_rows() -> None:
    rows = PlayerTally.from_arrays(synthetic_arrays(), 0, 10).per_street()
    assert [r["street"] for r in rows] == list(STREET_NAMES)
    flop = rows[1]
    assert (flop["decisions"], flop["check"], flop["call"], flop["bet"], flop["raise"]) == (4, 2, 1, 1, 0)
    assert flop["aggression_factor"] == pytest.approx(1.0) and flop["call_freq"] == pytest.approx(0.75)
    assert sum(r["decisions"] for r in rows) == 14


def test_tallies_pool_by_addition_and_survive_a_dict_round_trip() -> None:
    arrays = synthetic_arrays()
    tally = PlayerTally.from_arrays(arrays, 0, 10)
    pooled = sum_tallies([tally, tally])
    assert pooled.hands == 8 and pooled.action_counts.sum() == 28
    assert pooled.metrics()["vpip"] == pytest.approx(tally.metrics()["vpip"])
    assert pooled.metrics()["aggression_factor"] == pytest.approx(tally.metrics()["aggression_factor"])
    restored = PlayerTally.from_dict(tally.to_dict())
    assert restored.metrics() == tally.metrics() and np.array_equal(restored.action_counts, tally.action_counts)
    assert sum_tallies([]).hands == 0 and math.isnan(sum_tallies([]).metrics()["vpip"])


def test_aggression_factor_is_undefined_without_calls() -> None:
    builder = HandArraysBuilder()
    builder.add_hand(
        profit_chips=10, a_on_button=True, showdown=False, ended_street=0, pot_chips=20,
        action_counts=np.stack([_counts({(0, "raise"): 2}), _counts({(0, "fold"): 1})]),
        three_bet=(0, 0), three_bet_opportunity=(0, 0), latency_a_ms=[], latency_b_ms=[],
    )  # fmt: skip
    metrics = PlayerTally.from_arrays(builder.build(), 0, 10).metrics()
    assert math.isnan(metrics["aggression_factor"]) and metrics["raise_freq"] == 1.0


# ----------------------------------------------------------------------------- hand arrays
def test_hand_arrays_round_trip_and_concatenate(tmp_path) -> None:
    arrays = synthetic_arrays()
    path = tmp_path / "hands.npz"
    arrays.save(path)
    restored = HandArrays.load(path)
    for name in ("profit_chips", "a_on_button", "showdown", "ended_street", "pot_chips", "action_counts",
                 "three_bet", "three_bet_opportunity"):  # fmt: skip
        assert np.array_equal(getattr(restored, name), getattr(arrays, name)), name
    assert len(restored.latency_a_ms) == 0  # raw latencies are summarised in the JSON, not stored by default
    arrays.save(path, with_latency=True)
    with_latency = HandArrays.load(path)
    assert np.array_equal(with_latency.latency_a_ms, arrays.latency_a_ms)
    assert np.array_equal(with_latency.latency_b_ms, arrays.latency_b_ms)
    restored = arrays
    both = HandArrays.concatenate([arrays, restored])
    assert len(both) == 8 and len(both.latency_a_ms) == 2 * len(arrays.latency_a_ms)
    assert len(HandArrays.concatenate([])) == 0


def test_hand_arrays_end_category_and_validation() -> None:
    arrays = synthetic_arrays()
    assert arrays.end_category.tolist() == [0, 4, 4, 4]
    with pytest.raises(ValueError):
        HandArrays(
            profit_chips=np.zeros(3), a_on_button=np.zeros(3), showdown=np.zeros(3), ended_street=np.zeros(3),
            pot_chips=np.zeros(3), action_counts=np.zeros((2, 2, 4, 5)), three_bet=np.zeros((3, 2)),
            three_bet_opportunity=np.zeros((3, 2)), latency_a_ms=np.zeros(0), latency_b_ms=np.zeros(0),
        )  # fmt: skip
