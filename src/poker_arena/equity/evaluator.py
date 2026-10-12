"""Fast, numpy-vectorised Texas Hold'em hand evaluator (best 5 of 5, 6 or 7 cards).

Scores are plain integers where **higher is better** and equal hands tie::

    score = category << 20 | r0 << 16 | r1 << 12 | r2 << 8 | r3 << 4 | r4

``category`` is 0 (high card) .. 8 (straight flush) and ``r0..r4`` are the tie-break
ranks (0 = deuce .. 12 = ace) in decreasing order of significance: e.g. for two pair
``r0`` is the higher pair, ``r1`` the lower pair, ``r2`` the kicker. A wheel straight
(A-2-3-4-5) has ``r0 = 3``, i.e. it is the lowest straight.

Algorithm (no Python loop over hands): per hand we build four 13-bit rank masks - ranks
held at least 1, 2, 3 and 4 times - from one packed per-rank counter, then every category
is scored from small 8192-entry lookup tables. The best category wins via ordered
overwrites. Flushes are detected with packed suit counters and scored on the (few) rows
that have one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np
from numpy.typing import NDArray

from poker_arena.equity._inputs import to_int_cards

(
    HIGH_CARD,
    ONE_PAIR,
    TWO_PAIR,
    THREE_OF_A_KIND,
    STRAIGHT,
    FLUSH,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    STRAIGHT_FLUSH,
) = range(9)

CATEGORY_NAMES: tuple[str, ...] = (
    "High card",
    "One pair",
    "Two pair",
    "Three of a kind",
    "Straight",
    "Flush",
    "Full house",
    "Four of a kind",
    "Straight flush",
)
"""``CATEGORY_NAMES[hand_category(score)]`` is the human-readable hand class."""

NUM_RANKS = 13
NUM_SUITS = 4
SUPPORTED_HAND_SIZES = (5, 6, 7)
FLUSH_SIZE = 5

_RANK_SHIFT = 2  # card = rank * 4 + suit
_SUIT_MASK = NUM_SUITS - 1
_NIBBLE = 4  # bits per tie-break rank inside a score
_NUM_SLOTS = 5  # tie-break ranks per score
_CATEGORY_SHIFT = _NUM_SLOTS * _NIBBLE
_SLOT0, _SLOT1, _SLOT2 = (_CATEGORY_SHIFT - (i + 1) * _NIBBLE for i in range(3))  # bit offsets

_MASK_SPACE = 1 << NUM_RANKS  # number of distinct 13-bit rank masks
_FULL_MASK = _MASK_SPACE - 1

# Per-rank / per-suit counters are packed as 3-bit fields (counts 0..7) in one integer.
_COUNT_BITS = 3
_COUNT_FIELD_MAX = (1 << _COUNT_BITS) - 1
_CHUNK_RANKS = 5  # ranks decoded per table lookup
_CHUNK_BITS = _COUNT_BITS * _CHUNK_RANKS
_CHUNK_KEY_MASK = (1 << _CHUNK_BITS) - 1
_LANE_BITS = 16  # one 13-bit rank mask per lane inside a decoded int64
_MAX_RANK_COUNT = 4  # a rank can be held at most 4 times (one per suit)


def _build_straight_windows() -> tuple[tuple[int, int], ...]:
    """(top rank, rank mask) of the 10 five-card straights, lowest (the wheel) first."""
    wheel = (1 << (NUM_RANKS - 1)) | 0b1111
    windows = [(3, wheel)]
    windows += [(top, 0b11111 << (top - 4)) for top in range(4, NUM_RANKS)]
    return tuple(windows)


_STRAIGHT_WINDOWS = _build_straight_windows()
STRAIGHT_WINDOW_MASKS: tuple[int, ...] = tuple(mask for _, mask in _STRAIGHT_WINDOWS)
"""Rank bit-masks of the 10 possible straights (wheel included)."""

_RANKS = np.arange(NUM_RANKS, dtype=np.int64)
_RANK_BIT = np.int64(1) << _RANKS
_CLEAR_RANK = ~_RANK_BIT  # AND-ing a mask with it removes one rank (never adds bits)
_RANK_FIELDS_BITS = _COUNT_BITS * NUM_RANKS
_RANK_FIELDS_MASK = (1 << _RANK_FIELDS_BITS) - 1


def _build_card_weights() -> NDArray[np.int64]:
    """Per-card increment of one packed counter: 13 rank fields, then 4 suit fields above."""
    cards = np.arange(NUM_RANKS * NUM_SUITS, dtype=np.int64)
    rank_increment = np.int64(1) << (_COUNT_BITS * (cards >> _RANK_SHIFT))
    suit_increment = np.int64(1) << (_RANK_FIELDS_BITS + _COUNT_BITS * (cards & _SUIT_MASK))
    return rank_increment | suit_increment


_CARD_WEIGHT = _build_card_weights()


def _build_rank_mask_tables() -> tuple[NDArray[np.int64], ...]:
    """Tables over all 13-bit rank masks.

    Returns the highest rank, the best 3 / 2 / 5 ranks already placed in the score slots
    they fill (kickers after a pair start at slot 1, after trips at slot 2, a flush or
    high card at slot 0) and the top rank of the best straight (-1 if none).
    """
    masks = np.arange(_MASK_SPACE, dtype=np.int64)
    present = (masks[:, None] >> _RANKS) & 1
    # Descending ranks of each mask; absent slots are padded with 0 (only read when valid).
    descending = np.maximum(np.sort(present * (_RANKS + 1), axis=1)[:, ::-1] - 1, 0)

    def pack(count: int, first_slot: int) -> NDArray[np.int64]:
        shifts = _CATEGORY_SHIFT - (first_slot + 1 + np.arange(count)) * _NIBBLE
        return np.sum(descending[:, :count] << shifts, axis=1)

    straight_top = np.full(_MASK_SPACE, -1, dtype=np.int64)
    for top_rank, window in _STRAIGHT_WINDOWS:  # ascending, so the highest straight wins
        straight_top[(masks & window) == window] = top_rank
    return descending[:, 0], pack(3, 1), pack(2, 2), pack(5, 0), straight_top


_HIGH, _PAIR_KICKERS, _TRIPS_KICKERS, _TOP5, _STRAIGHT_TOP = _build_rank_mask_tables()


def _build_count_decoders() -> tuple[NDArray[np.int64], ...]:
    """One table per rank chunk mapping packed counts to the four rank masks.

    Lane ``j`` (bits ``16j..16j+12``) of an entry holds the ranks of the chunk held at
    least ``j + 1`` times, already shifted to the chunk's position in the 13-bit mask.
    """
    keys = np.arange(1 << _CHUNK_BITS, dtype=np.int64)
    tables = []
    for first_rank in range(0, NUM_RANKS, _CHUNK_RANKS):
        lanes = np.zeros(keys.size, dtype=np.int64)
        for offset in range(min(_CHUNK_RANKS, NUM_RANKS - first_rank)):
            count = (keys >> (_COUNT_BITS * offset)) & _COUNT_FIELD_MAX
            for lane in range(_MAX_RANK_COUNT):
                held = (count > lane).astype(np.int64)
                lanes |= held << (_LANE_BITS * lane + first_rank + offset)
        tables.append(lanes)
    return tuple(tables)


_COUNT_DECODERS = _build_count_decoders()


def _build_flush_suit_table() -> NDArray[np.int64]:
    """Packed suit counters -> suit holding >= 5 cards, or -1."""
    keys = np.arange(1 << (_COUNT_BITS * NUM_SUITS), dtype=np.int64)
    flush_suit = np.full(keys.size, -1, dtype=np.int64)
    for suit in range(NUM_SUITS):
        count = (keys >> (_COUNT_BITS * suit)) & _COUNT_FIELD_MAX
        flush_suit[count >= FLUSH_SIZE] = suit
    return flush_suit


_FLUSH_SUIT = _build_flush_suit_table()


def _base(category: int) -> int:
    return category << _CATEGORY_SHIFT


def _rank_masks(packed_ranks: NDArray[np.int64]) -> tuple[NDArray[np.int64], ...]:
    """Rank masks of ranks held at least 1, 2, 3 and 4 times, from packed rank counts."""
    lanes = (
        _COUNT_DECODERS[0][packed_ranks & _CHUNK_KEY_MASK]
        | _COUNT_DECODERS[1][(packed_ranks >> _CHUNK_BITS) & _CHUNK_KEY_MASK]
        | _COUNT_DECODERS[2][packed_ranks >> (2 * _CHUNK_BITS)]
    )
    return tuple((lanes >> (lane * _LANE_BITS)) & _FULL_MASK for lane in range(_MAX_RANK_COUNT))


def _score_flushes(cards: NDArray[np.int64], suits: NDArray[np.int64]) -> NDArray[np.int64]:
    """Score (flush or straight flush) for rows of ``cards`` that hold >= 5 cards of ``suits``."""
    in_suit = (cards & _SUIT_MASK) == suits[:, None]
    flush_mask = np.sum(_RANK_BIT[cards >> _RANK_SHIFT] * in_suit, axis=1)
    straight_top = _STRAIGHT_TOP[flush_mask]
    return np.where(
        straight_top >= 0,
        _base(STRAIGHT_FLUSH) | (straight_top << _SLOT0),
        _base(FLUSH) | _TOP5[flush_mask],
    )


def _evaluate_unchecked(cards: NDArray[np.int64]) -> NDArray[np.int32]:
    """Core of :func:`evaluate_batch`; ``cards`` must be valid, distinct and int64."""
    counters = _CARD_WEIGHT[cards].sum(axis=1)  # per-rank and per-suit card counts, packed
    held, pairs, trips, quads = _rank_masks(counters & _RANK_FIELDS_MASK)

    # Categories are applied from weakest to strongest; each overwrite is only valid for
    # rows where that category exists, and a stronger category always has a larger score.
    score = _TOP5[held]  # high card (category 0)

    pair = _HIGH[pairs]
    candidate = _base(ONE_PAIR) | (pair << _SLOT0) | _PAIR_KICKERS[held & _CLEAR_RANK[pair]]
    np.copyto(score, candidate, where=pairs != 0)

    low_pairs = pairs & _CLEAR_RANK[pair]
    low_pair = _HIGH[low_pairs]
    two_pair_kicker = _HIGH[held & _CLEAR_RANK[pair] & _CLEAR_RANK[low_pair]]
    candidate = (
        _base(TWO_PAIR) | (pair << _SLOT0) | (low_pair << _SLOT1) | (two_pair_kicker << _SLOT2)
    )
    np.copyto(score, candidate, where=low_pairs != 0)

    trip = _HIGH[trips]
    candidate = (
        _base(THREE_OF_A_KIND) | (trip << _SLOT0) | _TRIPS_KICKERS[held & _CLEAR_RANK[trip]]
    )
    np.copyto(score, candidate, where=trips != 0)

    straight_top = _STRAIGHT_TOP[held]
    np.copyto(score, _base(STRAIGHT) | (straight_top << _SLOT0), where=straight_top >= 0)

    trip_pairs = pairs & _CLEAR_RANK[trip]
    candidate = _base(FULL_HOUSE) | (trip << _SLOT0) | (_HIGH[trip_pairs] << _SLOT1)
    np.copyto(score, candidate, where=(trips != 0) & (trip_pairs != 0))

    quad = _HIGH[quads]
    quad_kicker = _HIGH[held & _CLEAR_RANK[quad]]
    candidate = _base(FOUR_OF_A_KIND) | (quad << _SLOT0) | (quad_kicker << _SLOT1)
    np.copyto(score, candidate, where=quads != 0)

    flush_suit = _FLUSH_SUIT[counters >> _RANK_FIELDS_BITS]
    flush_rows = np.flatnonzero(flush_suit >= 0)
    if flush_rows.size:
        flush_scores = _score_flushes(cards[flush_rows], flush_suit[flush_rows])
        score[flush_rows] = np.maximum(score[flush_rows], flush_scores)

    return score.astype(np.int32)


def evaluate_batch(cards: NDArray[np.integer]) -> NDArray[np.int32]:
    """Score many hands at once: the best 5-card hand out of each row's 5, 6 or 7 cards.

    Args:
        cards: integer array of shape ``(N, k)`` with ``k`` in {5, 6, 7} and values in
            0..51 (``rank * 4 + suit``, see :mod:`poker_arena.cards`). Cards within a
            row must be distinct (not checked - this is the hot path).

    Returns:
        ``int32`` array of shape ``(N,)``. Higher is stronger; equal hands tie. Use
        :func:`hand_category` to recover the hand class.

    Raises:
        ValueError: bad shape, non-integer dtype or card values outside 0..51.
    """
    cards = np.asarray(cards)
    if cards.ndim != 2 or cards.shape[1] not in SUPPORTED_HAND_SIZES:
        raise ValueError(f"cards must have shape (N, k) with k in {SUPPORTED_HAND_SIZES}")
    if not np.issubdtype(cards.dtype, np.integer):
        raise ValueError(f"cards must be an integer array, got {cards.dtype}")
    if cards.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    if cards.min() < 0 or cards.max() >= NUM_RANKS * NUM_SUITS:
        raise ValueError("card values must be in 0..51")
    return _evaluate_unchecked(cards.astype(np.int64, copy=False))


def evaluate(cards: Sequence[int | str]) -> int:
    """Score a single hand of 5-7 distinct cards (ints ``0..51`` or strings like ``"Ah"``)."""
    ints = to_int_cards(cards)
    if len(set(ints)) != len(ints):
        raise ValueError(f"Duplicate cards in {cards!r}")
    return int(evaluate_batch(np.array([ints], dtype=np.int64))[0])


@overload
def hand_category(score: int) -> int: ...
@overload
def hand_category(score: NDArray[np.integer]) -> NDArray[np.integer]: ...
def hand_category(score: int | NDArray[np.integer]) -> int | NDArray[np.integer]:
    """Hand class of a score (scalar or array): 0 high card .. 8 straight flush."""
    return score >> _CATEGORY_SHIFT
