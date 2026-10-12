"""Quick performance guard-rails (generous bounds) that also print the measured numbers.

Run with ``pytest -s tests/test_equity_speed.py`` to see them.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import numpy as np

from poker_arena.equity.evaluator import evaluate_batch
from poker_arena.equity.monte_carlo import equity_vs_random

# Generous: the real numbers are ~10-20x better, but CI machines are noisy.
MAX_MEDIAN_SECONDS_FLOP_200_SAMPLES = 5e-3
MIN_HANDS_PER_SECOND = 300_000


def median_seconds(function: Callable[[], object], repeats: int) -> float:
    function()  # warm-up
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def test_equity_vs_random_200_samples_on_flop_is_fast() -> None:
    rng = np.random.default_rng(0)
    hole, flop = ["As", "Kd"], ["Qs", "Jd", "2c"]
    median = median_seconds(lambda: equity_vs_random(hole, flop, 200, rng), repeats=200)
    print(f"\nequity_vs_random(200 samples, flop): median {median * 1e3:.3f} ms")
    assert median < MAX_MEDIAN_SECONDS_FLOP_200_SAMPLES


def test_evaluate_batch_throughput() -> None:
    rng = np.random.default_rng(1)
    hands = np.argsort(rng.random((100_000, 52)), axis=1)[:, :7].astype(np.int64)
    median = median_seconds(lambda: evaluate_batch(hands), repeats=7)
    throughput = len(hands) / median
    print(f"\nevaluate_batch: {throughput / 1e6:.2f} M 7-card hands/s")
    assert throughput > MIN_HANDS_PER_SECOND
