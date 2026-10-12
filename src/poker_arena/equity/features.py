"""Cheap, deterministic, observable-only hand/board descriptors (model + UI features).

Every function takes only the hero's own hole cards and/or the public board - never any
hidden information. Ranks are indices 0..12 (0 = deuce ... 12 = ace), suits 0..3. All
values are plain ``int`` (booleans are 0/1) so they can be dropped straight into a
feature table.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from poker_arena.equity._inputs import parse_board, parse_hole, parse_hole_board
from poker_arena.equity.evaluator import (
    HIGH_CARD,
    NUM_SUITS,
    ONE_PAIR,
    STRAIGHT_WINDOW_MASKS,
    evaluate_batch,
    hand_category,
)

_SUITS_PER_RANK = 4
_FLUSH_DRAW_SIZE = 4
_CARDS_PER_RANK = 4
_FLOP_SIZE, _TURN_SIZE = 3, 4


def _ranks(cards: Sequence[int]) -> list[int]:
    return [card // _SUITS_PER_RANK for card in cards]


def _rank_mask(cards: Sequence[int]) -> int:
    mask = 0
    for rank in _ranks(cards):
        mask |= 1 << rank
    return mask


def hole_card_features(hole: Sequence[int | str]) -> dict[str, int]:
    """Descriptors of the two private cards.

    * ``high_rank`` / ``low_rank``: rank (0 = deuce .. 12 = ace) of the higher / lower card.
    * ``is_pair``: 1 for a pocket pair.
    * ``is_suited``: 1 if both cards share a suit (always 0 for a pair).
    * ``gap``: ``high_rank - low_rank``: 0 pair, 1 connectors (e.g. 98), 2 one-gapper (J9)...
    """
    cards = parse_hole(hole)
    first, second = _ranks(cards)
    return {
        "high_rank": max(first, second),
        "low_rank": min(first, second),
        "is_pair": int(first == second),
        "is_suited": int(cards[0] % _SUITS_PER_RANK == cards[1] % _SUITS_PER_RANK),
        "gap": abs(first - second),
    }


def made_hand_category(hole: Sequence[int | str], board: Sequence[int | str]) -> int:
    """Class of the best 5-card hand made from hole + board (0 high card .. 8 straight flush).

    Categories match :data:`poker_arena.equity.evaluator.CATEGORY_NAMES`. Preflop (empty
    board) no 5-card hand exists, so the value degrades to a pair indicator: 1 (one pair)
    for a pocket pair, otherwise 0 (high card).
    """
    hole_ints, board_ints = parse_hole_board(hole, board)
    if not board_ints:
        first, second = _ranks(hole_ints)
        return ONE_PAIR if first == second else HIGH_CARD
    score = evaluate_batch(np.array([hole_ints + board_ints], dtype=np.int64))[0]
    return int(hand_category(int(score)))


def board_texture(board: Sequence[int | str]) -> dict[str, int]:
    """Descriptors of the public board (0, 3, 4 or 5 cards); all public information.

    * ``n_board_cards``: number of board cards (0 preflop, 3 flop, 4 turn, 5 river).
    * ``board_paired``: 1 if any rank appears at least twice.
    * ``board_max_suit_count``: most cards of a single suit (flush potential; 0 if empty).
    * ``board_straightness``: most distinct board ranks inside any 5-rank straight window
      (ace counts low too); 3 means a straight is two cards away, 5 a straight on board.
    * ``board_high_rank``: rank of the highest board card, or -1 for an empty board.
    """
    board_ints = parse_board(board)
    mask = _rank_mask(board_ints)
    suit_counts = Counter(card % _SUITS_PER_RANK for card in board_ints)
    return {
        "n_board_cards": len(board_ints),
        "board_paired": int(len(set(_ranks(board_ints))) < len(board_ints)),
        "board_max_suit_count": max(suit_counts.values(), default=0),
        "board_straightness": max((mask & w).bit_count() for w in STRAIGHT_WINDOW_MASKS),
        "board_high_rank": max(_ranks(board_ints), default=-1),
    }


def draw_features(hole: Sequence[int | str], board: Sequence[int | str]) -> dict[str, int]:
    """Drawing / relative-strength descriptors of the hero's hand on a flop, turn or river.

    * ``flush_draw``: 1 if exactly four cards of one suit are available (hole + board) and
      at least one of them is a hole card. Only on flop/turn (0 elsewhere: a river flush
      draw has no cards to come, and a made flush is not a draw).
    * ``straight_outs``: number of unseen cards that complete a straight using at least one
      hole card: 0, 4 (gutshot), 8 (open-ended / double gutshot). Only on flop/turn, and 0
      when hero already has a straight.
    * ``overcards``: hole cards (0-2) ranked above every board card (flop/turn/river).
    * ``has_pair_with_board``: 1 if a hole card matches the rank of a board card.

    Before the flop (empty board) every feature is 0.
    """
    hole_ints, board_ints = parse_hole_board(hole, board)
    features = {"flush_draw": 0, "straight_outs": 0, "overcards": 0, "has_pair_with_board": 0}
    if not board_ints:
        return features

    board_ranks = set(_ranks(board_ints))
    hole_ranks = _ranks(hole_ints)
    features["overcards"] = sum(rank > max(board_ranks) for rank in hole_ranks)
    features["has_pair_with_board"] = int(any(rank in board_ranks for rank in hole_ranks))
    if len(board_ints) not in (_FLOP_SIZE, _TURN_SIZE):
        return features

    suit_cards = Counter(card % _SUITS_PER_RANK for card in hole_ints + board_ints)
    suit_hole = Counter(card % _SUITS_PER_RANK for card in hole_ints)
    features["flush_draw"] = int(
        any(
            suit_cards[suit] == _FLUSH_DRAW_SIZE and suit_hole[suit] > 0
            for suit in range(NUM_SUITS)
        )
    )

    held = _rank_mask(hole_ints + board_ints)
    if any(held & window == window for window in STRAIGHT_WINDOW_MASKS):
        return features  # already has a straight
    hole_mask = _rank_mask(hole_ints)
    completing = 0
    for window in STRAIGHT_WINDOW_MASKS:
        missing = window & ~held
        if missing.bit_count() == 1 and window & hole_mask:
            completing |= missing
    features["straight_outs"] = completing.bit_count() * _CARDS_PER_RANK
    return features
