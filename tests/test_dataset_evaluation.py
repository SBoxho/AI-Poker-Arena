"""Duplicate-deal evaluation helper used to validate the teacher."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from poker_arena.agents.registry import AgentSpec
from poker_arena.data.evaluation import (
    HANDS_PER_PAIR,
    PAIRS_PER_CHUNK,
    evaluate_duplicate,
    pair_results,
    summarize,
)


def test_summary_matches_a_hand_computed_t_interval() -> None:
    pairs = np.array([4.0, -2.0, 7.0, 1.0, 0.0, 3.0])  # big blinds won by A per pair (two hands each)
    result = summarize("A", "B", pairs)
    mean, sem = pairs.mean(), pairs.std(ddof=1) / math.sqrt(len(pairs))
    half = stats.t.ppf(0.975, df=len(pairs) - 1) * sem
    scale = 100 / HANDS_PER_PAIR
    assert result.bb_per_100 == pytest.approx(mean * scale)
    assert (result.ci_low, result.ci_high) == pytest.approx(((mean - half) * scale, (mean + half) * scale))
    assert result.n_hands == 2 * len(pairs) and result.n_matches == 1
    assert math.isnan(result.match_ci_low)  # one match cannot estimate between-match variation
    with pytest.raises(ValueError):
        summarize("A", "B", pairs[:1])


def test_match_level_interval_uses_per_match_means() -> None:
    rng = np.random.default_rng(0)
    pairs = rng.normal(2.0, 5.0, size=PAIRS_PER_CHUNK * 8 + 7)  # eight full matches and a short one
    result = summarize("A", "B", pairs)
    means = [chunk.mean() for chunk in np.split(pairs, range(PAIRS_PER_CHUNK, len(pairs), PAIRS_PER_CHUNK))]
    half = stats.t.ppf(0.975, df=len(means) - 1) * np.std(means, ddof=1) / math.sqrt(len(means))
    assert result.n_matches == 9
    assert result.match_ci_high - result.match_ci_low == pytest.approx(2 * half * 100 / HANDS_PER_PAIR)
    assert result.to_dict()["ci95_matches"] is not None


def test_results_are_identical_for_any_number_of_workers_and_reproducible() -> None:
    kwargs = {"spec_a": AgentSpec("teacher"), "spec_b": AgentSpec("random"), "n_pairs": 2 * PAIRS_PER_CHUNK + 5,
              "seed_parts": ("test", "workers")}
    serial = pair_results(n_workers=1, **kwargs)
    parallel = pair_results(n_workers=2, **kwargs)
    assert np.array_equal(serial, parallel) and np.array_equal(serial, pair_results(n_workers=1, **kwargs))
    assert not np.array_equal(serial, pair_results(**{**kwargs, "seed_parts": ("test", "other")}))


def test_a_clearly_better_player_has_a_positive_interval() -> None:
    result = evaluate_duplicate(AgentSpec("teacher"), AgentSpec("call_station"), 200, ("test", "teacher-vs-station"))
    assert result.bb_per_100 > 0 and result.ci_low > 0 and result.n_matches == 4
