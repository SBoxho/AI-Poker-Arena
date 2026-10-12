"""Monte Carlo equity: determinism, unbiasedness vs exact enumeration, validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats

from poker_arena.cards import ALL_CARDS
from poker_arena.equity import monte_carlo
from poker_arena.equity.monte_carlo import (
    equity_vs_random,
    equity_vs_random_with_se,
    exact_equity_vs_random,
)

SE_TOLERANCE = 4.0  # seeded runs are deterministic; 4 standard errors keeps margins wide


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- determinism


def test_same_seed_gives_identical_result() -> None:
    args = (["As", "Kd"], ["Qs", "Jd", "2c"], 3_000)
    assert equity_vs_random(*args, rng(7)) == equity_vs_random(*args, rng(7))
    assert equity_vs_random_with_se(*args, rng(7)) == equity_vs_random_with_se(*args, rng(7))


def test_different_seeds_differ_but_agree_statistically() -> None:
    args = (["As", "Kd"], ["Qs", "Jd", "2c"], 5_000)
    first, second = equity_vs_random(*args, rng(1)), equity_vs_random(*args, rng(2))
    assert first != second
    assert abs(first - second) < 0.05


def test_accepts_ints_strings_and_numpy_ints_identically() -> None:
    strings = equity_vs_random(["As", "Kd"], ["Qs", "Jd", "2c"], 1_000, rng(3))
    ints = [ALL_CARDS.index(c) for c in ["As", "Kd", "Qs", "Jd", "2c"]]
    assert equity_vs_random(ints[:2], ints[2:], 1_000, rng(3)) == strings
    numpy_ints = [np.int64(i) for i in ints]
    assert equity_vs_random(numpy_ints[:2], numpy_ints[2:], 1_000, rng(3)) == strings


def test_chunked_simulation_agrees_with_single_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunking only changes how the RNG stream is consumed, not the estimand."""
    args = (["Ah", "Qh"], [], 10_000)
    whole = equity_vs_random(*args, rng(5))
    monkeypatch.setattr(monte_carlo, "_CHUNK_TRIALS", 1_000)
    chunked = equity_vs_random(*args, rng(5))
    assert abs(whole - chunked) < 0.03


# --------------------------------------------------------------------------- exactness


EXACT_CASES = [
    # (hole, board) - river
    (["As", "Ks"], ["Qs", "Js", "2d", "7h", "9c"]),
    (["7c", "7d"], ["7h", "2s", "2d", "Kc", "Qd"]),
    (["9h", "8h"], ["Th", "7h", "2c", "Kd", "3s"]),
    (["Ac", "2d"], ["Kc", "Qd", "Jh", "Ts", "5c"]),
    # turn
    (["As", "Ks"], ["Qs", "Js", "2d", "7h"]),
    (["6c", "6d"], ["Ah", "Kh", "6s", "2c"]),
    (["Tc", "9c"], ["8d", "7h", "2s", "Kc"]),
]


@pytest.mark.parametrize(("hole", "board"), EXACT_CASES)
def test_monte_carlo_matches_exact_enumeration(hole: list[str], board: list[str]) -> None:
    exact = exact_equity_vs_random(hole, board)
    estimate, std_error = equity_vs_random_with_se(hole, board, 30_000, rng(11))
    assert std_error > 0
    assert abs(estimate - exact) < SE_TOLERANCE * std_error


def test_monte_carlo_matches_exact_enumeration_on_flop() -> None:
    hole, board = ["Ad", "Th"], ["9d", "8d", "2c"]
    exact = exact_equity_vs_random(hole, board)
    estimate, std_error = equity_vs_random_with_se(hole, board, 60_000, rng(12))
    assert abs(estimate - exact) < SE_TOLERANCE * std_error


def test_exact_equity_special_values() -> None:
    assert exact_equity_vs_random(["As", "Ks"], ["Qs", "Js", "Ts", "2d", "3c"]) == 1.0  # royal
    # The board is a royal flush: every hand plays the board and ties.
    assert exact_equity_vs_random(["2c", "3d"], ["Ts", "Js", "Qs", "Ks", "As"]) == 0.5
    # Quad aces with the best kicker available cannot lose or tie.
    assert exact_equity_vs_random(["Ac", "Ad"], ["Ah", "As", "Kd", "2c", "3c"]) == 1.0


def test_exact_river_equity_is_a_fraction_of_990_opponent_hands() -> None:
    equity = exact_equity_vs_random(["7c", "7d"], ["7h", "2s", "2d", "Kc", "Qd"])
    assert (equity * 2 * 990) == pytest.approx(round(equity * 2 * 990))


# --------------------------------------------------------------------------- known values


def test_pocket_aces_preflop_equity() -> None:
    estimate, std_error = equity_vs_random_with_se(["As", "Ah"], [], 100_000, rng(21))
    assert abs(estimate - 0.852) < max(SE_TOLERANCE * std_error, 0.005)


def test_preflop_ordering_of_known_hands() -> None:
    aces = equity_vs_random(["As", "Ah"], [], 40_000, rng(22))
    suited_connector = equity_vs_random(["Js", "Ts"], [], 40_000, rng(23))
    worst = equity_vs_random(["7c", "2d"], [], 40_000, rng(24))
    assert 0.84 < aces < 0.87
    assert 0.56 < suited_connector < 0.59  # JTs ~0.576
    assert 0.32 < worst < 0.37
    assert aces > suited_connector > worst


@pytest.mark.parametrize("board_size", [0, 3, 4, 5])
def test_all_board_sizes_return_probabilities(board_size: int) -> None:
    board = ["Qs", "Jd", "2c", "7h", "9c"][:board_size]
    equity = equity_vs_random(["As", "Kd"], board, 2_000, rng(31))
    assert 0.0 <= equity <= 1.0


def test_known_hand_cannot_lose_when_sampling_is_correct() -> None:
    """Hero has quad aces with a 3 kicker: the opponent can never win or tie. A sampler
    that dealt a known card (e.g. an ace) to the opponent would break this."""
    assert equity_vs_random(["Ac", "Ad"], ["Ah", "As", "3d", "2c", "2d"], 5_000, rng(32)) == 1.0
    assert equity_vs_random(["2c", "3d"], ["Ts", "Js", "Qs", "Ks", "As"], 5_000, rng(33)) == 0.5


def test_standard_error_shrinks_with_more_samples() -> None:
    args = (["Qh", "Jh"], ["9h", "4c", "2d"])
    _, small = equity_vs_random_with_se(*args, 1_000, rng(41))
    _, large = equity_vs_random_with_se(*args, 100_000, rng(41))
    assert large < small / 5
    assert large == pytest.approx(0.0015, abs=0.001)


# --------------------------------------------------------------------------- sampling quality


def captured_hands(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], object]
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``call`` and return all scored (hero hands, opponent hands), each of shape (n, 7)."""
    heroes: list[np.ndarray] = []
    opponents: list[np.ndarray] = []
    original = monte_carlo._evaluate_unchecked

    def spy(cards: np.ndarray) -> np.ndarray:
        half = len(cards) // 2  # the simulator scores hero rows first, then opponent rows
        heroes.append(cards[:half].copy())
        opponents.append(cards[half:].copy())
        return original(cards)

    monkeypatch.setattr(monte_carlo, "_evaluate_unchecked", spy)
    call()
    return np.vstack(heroes), np.vstack(opponents)


@pytest.mark.parametrize("board_size", [0, 3, 4, 5])
def test_sampled_hands_are_valid_and_respect_known_cards(
    monkeypatch: pytest.MonkeyPatch, board_size: int
) -> None:
    hole = [ALL_CARDS.index(c) for c in ("Ah", "Kd")]
    board = [ALL_CARDS.index(c) for c in ("Qs", "Jd", "2c", "7h", "9c")][:board_size]
    n = 4_000
    hero, opponent = captured_hands(monkeypatch, lambda: equity_vs_random(hole, board, n, rng(51)))
    assert hero.shape == opponent.shape == (n, 7)
    for rows in (hero, opponent):
        ordered = np.sort(rows, axis=1)
        assert np.all(ordered[:, 1:] != ordered[:, :-1])  # seven distinct cards per hand
        assert np.array_equal(rows[:, 2 : 2 + board_size], np.tile(board, (n, 1)))
    assert np.array_equal(hero[:, :2], np.tile(hole, (n, 1)))
    assert np.array_equal(hero[:, 2:], opponent[:, 2:])  # same final board for both players
    known = set(hole + board)
    assert not known & set(opponent[:, :2].ravel().tolist())  # opponent never holds known cards
    assert not known & set(hero[:, 2 + board_size :].ravel().tolist())  # nor does the runout


@pytest.mark.parametrize("column", [0, 1, 5, 6])
def test_every_dealt_slot_is_uniform_over_unseen_cards(
    monkeypatch: pytest.MonkeyPatch, column: int
) -> None:
    """Opponent cards and runout cards are uniform over the 50 unseen preflop cards."""
    hole = [ALL_CARDS.index("Ah"), ALL_CARDS.index("Kd")]
    n = 100_000
    _, opponent = captured_hands(monkeypatch, lambda: equity_vs_random(hole, [], n, rng(52)))
    values = opponent[:, column]  # columns 0-1: opponent hole cards, 2-6: board runout
    unseen = np.setdiff1d(np.arange(52), hole)
    observed = np.array([(values == card).sum() for card in unseen])
    assert observed.sum() == n
    assert stats.chisquare(observed).pvalue > 1e-3


def test_opponent_pairs_are_uniform_over_unordered_hands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Joint check: each of the C(50,2)=1225 opponent hole-card pairs is equally likely."""
    hole = [ALL_CARDS.index("Ah"), ALL_CARDS.index("Kd")]
    n = 245_000  # ~200 per pair
    _, opponent = captured_hands(monkeypatch, lambda: equity_vs_random(hole, [], n, rng(53)))
    pairs = np.sort(opponent[:, :2], axis=1)
    codes = pairs[:, 0] * 52 + pairs[:, 1]
    _, counts = np.unique(codes, return_counts=True)
    assert counts.size == 1225
    assert stats.chisquare(counts).pvalue > 1e-3


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("hole", "board"),
    [
        (["As"], []),  # one hole card
        (["As", "Kd", "Qh"], []),  # three hole cards
        (["As", "As"], []),  # duplicate hole cards
        (["As", "Kd"], ["As", "2c", "3d"]),  # board repeats a hole card
        (["As", "Kd"], ["2c", "2c", "3d"]),  # duplicate board cards
        (["As", "Kd"], ["2c"]),  # invalid board lengths
        (["As", "Kd"], ["2c", "3d"]),
        (["As", "Kd"], ["2c", "3d", "4h", "5s", "6c", "7d"]),
        (["As", "Kx"], []),  # invalid card string
        ([52, 1], []),  # card int out of range
        ([-1, 1], []),
    ],
)
def test_invalid_inputs_raise(hole: list[str | int], board: list[str]) -> None:
    with pytest.raises(ValueError):
        equity_vs_random(hole, board, 100, rng())
    with pytest.raises(ValueError):
        equity_vs_random_with_se(hole, board, 100, rng())
    with pytest.raises(ValueError):
        exact_equity_vs_random(hole, board)


@pytest.mark.parametrize("n_samples", [0, -5])
def test_invalid_sample_count_raises(n_samples: int) -> None:
    with pytest.raises(ValueError):
        equity_vs_random(["As", "Kd"], [], n_samples, rng())


def test_standard_error_needs_two_samples() -> None:
    with pytest.raises(ValueError):
        equity_vs_random_with_se(["As", "Kd"], [], 1, rng())


def test_exact_enumeration_needs_a_flop_at_least() -> None:
    with pytest.raises(ValueError):
        exact_equity_vs_random(["As", "Kd"], [])
