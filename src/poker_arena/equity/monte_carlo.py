"""Monte Carlo and exact hand equity against ONE uniformly random opponent hand.

INFORMATION RULE (anti-cheating boundary): every function here receives only the hero's
own hole cards and the public board. The opponent's cards are always *sampled* from the
cards hero cannot see, never read from game state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from poker_arena.cards import NUM_CARDS
from poker_arena.equity._inputs import HOLE_SIZE, parse_hole_board
from poker_arena.equity.evaluator import _evaluate_unchecked

BOARD_SIZE = 5
_HAND_SIZE = HOLE_SIZE + BOARD_SIZE
_CHUNK_TRIALS = 1 << 15  # trials simulated per vectorised batch (bounds memory)
_EXACT_CHUNK_HANDS = 1 << 18  # (board, opponent) combinations scored per batch
_MIN_EXACT_BOARD = 3
_ALL_CARDS = np.arange(NUM_CARDS, dtype=np.int64)


def _unseen_cards(known: Sequence[int]) -> NDArray[np.int64]:
    return np.delete(_ALL_CARDS, list(known))


def _combination_array(items: NDArray[np.int64], size: int) -> NDArray[np.int64]:
    """All ``size``-card combinations of ``items`` as rows (one empty row if ``size == 0``)."""
    rows = math.comb(items.size, size)
    return np.fromiter(
        (card for combo in combinations(items.tolist(), size) for card in combo),
        dtype=np.int64,
        count=rows * size,
    ).reshape(rows, size)


def _simulate_chunk(
    hole: Sequence[int],
    board: Sequence[int],
    unseen: NDArray[np.int64],
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Run ``n_trials`` independent deals; return (hero wins, ties)."""
    missing = BOARD_SIZE - len(board)
    n_draws = HOLE_SIZE + missing  # opponent hole cards, then the rest of the board
    pool = unseen.size

    # Partial Fisher-Yates shuffle, vectorised over trials: draw k picks a uniform slot in
    # [k, pool) and moves the card currently at slot k into the vacated slot. Every trial
    # gets an exact uniform sample without replacement (u < 1 so slots stay in range).
    deck = np.tile(unseen, (n_trials, 1))
    rows = np.arange(n_trials)
    spans = pool - np.arange(n_draws)
    slots = (rng.random((n_draws, n_trials)) * spans[:, None]).astype(np.intp)
    slots += np.arange(n_draws)[:, None]
    draws = np.empty((n_draws, n_trials), dtype=np.int64)
    for k in range(n_draws):
        draws[k] = deck[rows, slots[k]]
        deck[rows, slots[k]] = deck[:, k]

    # Hero rows first, then opponent rows; both players share the same final board.
    hands = np.empty((2 * n_trials, _HAND_SIZE), dtype=np.int64)
    hands[:n_trials, :HOLE_SIZE] = hole
    hands[n_trials:, :HOLE_SIZE] = draws[:HOLE_SIZE].T
    hands[:, HOLE_SIZE : HOLE_SIZE + len(board)] = board
    runout = draws[HOLE_SIZE:].T
    hands[:n_trials, HOLE_SIZE + len(board) :] = runout
    hands[n_trials:, HOLE_SIZE + len(board) :] = runout

    scores = _evaluate_unchecked(hands)
    hero, opponent = scores[:n_trials], scores[n_trials:]
    return int(np.count_nonzero(hero > opponent)), int(np.count_nonzero(hero == opponent))


def _count_outcomes(
    hole: Sequence[int | str],
    board: Sequence[int | str],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """(wins, ties) of hero over ``n_samples`` random deals."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    hole_ints, board_ints = parse_hole_board(hole, board)
    unseen = _unseen_cards(hole_ints + board_ints)
    wins = ties = 0
    for start in range(0, n_samples, _CHUNK_TRIALS):
        chunk_wins, chunk_ties = _simulate_chunk(
            hole_ints, board_ints, unseen, min(_CHUNK_TRIALS, n_samples - start), rng
        )
        wins += chunk_wins
        ties += chunk_ties
    return wins, ties


def equity_vs_random(
    hole: Sequence[int | str],
    board: Sequence[int | str],
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Estimate hero's equity against one uniformly random opponent hand.

    Each trial deals the opponent two cards and completes the board (0, 3 or 4 known
    cards -> 5) uniformly *without replacement* from the cards not in ``hole``/``board``.
    A win counts 1, a tie 0.5. Deterministic for a given seeded ``rng``.

    Args:
        hole: hero's two hole cards (ints 0..51 or strings like ``"Ah"``).
        board: public board cards - 0, 3, 4 or 5 of them.
        n_samples: number of simulated deals (standard error <= 0.5 / sqrt(n_samples)).
        rng: numpy random generator (the only source of randomness).

    Raises:
        ValueError: invalid/duplicate cards, wrong hole/board length or ``n_samples < 1``.
    """
    wins, ties = _count_outcomes(hole, board, n_samples, rng)
    return (wins + 0.5 * ties) / n_samples


def equity_vs_random_with_se(
    hole: Sequence[int | str],
    board: Sequence[int | str],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Like :func:`equity_vs_random` but also returns the standard error of the estimate.

    The standard error is the sample standard deviation of the per-deal outcomes
    (1 / 0.5 / 0) divided by ``sqrt(n_samples)``; it needs ``n_samples >= 2``.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2 to estimate a standard error, got {n_samples}")
    wins, ties = _count_outcomes(hole, board, n_samples, rng)
    equity = (wins + 0.5 * ties) / n_samples
    mean_square = (wins + 0.25 * ties) / n_samples  # outcomes are 0, 0.5 or 1
    variance = max(mean_square - equity**2, 0.0) * n_samples / (n_samples - 1)
    return equity, math.sqrt(variance / n_samples)


def exact_equity_vs_random(hole: Sequence[int | str], board: Sequence[int | str]) -> float:
    """Exact equity against one random opponent hand by full enumeration (flop/turn/river).

    Enumerates every (missing board cards, opponent hand) combination: 990 on the river,
    ~44k on the turn, ~1.1M on the flop. Preflop is intractable - use
    :func:`equity_vs_random` or the precomputed :mod:`poker_arena.equity.preflop` table.

    Raises:
        ValueError: invalid/duplicate cards, wrong lengths or a board with fewer than 3 cards.
    """
    hole_ints, board_ints = parse_hole_board(hole, board)
    if len(board_ints) < _MIN_EXACT_BOARD:
        raise ValueError("Exact enumeration needs a board of 3, 4 or 5 cards")
    unseen = _unseen_cards(hole_ints + board_ints)
    missing = BOARD_SIZE - len(board_ints)

    completions = _combination_array(unseen, missing)
    opponents = _combination_array(unseen, HOLE_SIZE)
    completion_bits = np.bitwise_or.reduce(np.int64(1) << completions, axis=1, initial=0)
    opponent_bits = np.bitwise_or.reduce(np.int64(1) << opponents, axis=1, initial=0)
    disjoint = (completion_bits[:, None] & opponent_bits[None, :]) == 0

    wins = ties = total = 0
    step = max(1, _EXACT_CHUNK_HANDS // len(opponents))
    for start in range(0, len(completions), step):
        completion_idx, opponent_idx = np.nonzero(disjoint[start : start + step])
        completion_idx += start
        shared = np.hstack(
            [np.tile(board_ints, (len(completion_idx), 1)), completions[completion_idx]]
        )
        hero_hands = np.hstack([np.tile(hole_ints, (len(completion_idx), 1)), shared])
        opponent_hands = np.hstack([opponents[opponent_idx], shared])
        hero, opponent = _evaluate_unchecked(hero_hands), _evaluate_unchecked(opponent_hands)
        wins += int(np.count_nonzero(hero > opponent))
        ties += int(np.count_nonzero(hero == opponent))
        total += len(completion_idx)
    return (wins + 0.5 * ties) / total
