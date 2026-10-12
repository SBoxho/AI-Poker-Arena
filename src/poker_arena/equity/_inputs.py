"""Input normalisation / validation shared by the equity modules (private)."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Sequence

from poker_arena.cards import assert_no_duplicates, card_to_int

HOLE_SIZE = 2
VALID_BOARD_SIZES = (0, 3, 4, 5)
"""Public-board lengths that exist in hold'em: preflop, flop, turn, river."""


def to_int_cards(cards: Iterable[int | str]) -> tuple[int, ...]:
    """Convert strings (``"Ah"``), ints and numpy integers to validated int cards."""
    return tuple(
        card_to_int(card if isinstance(card, str) else operator.index(card)) for card in cards
    )


def parse_hole(hole: Sequence[int | str]) -> tuple[int, ...]:
    """Validate two distinct hole cards; return them as ints."""
    hole_ints = to_int_cards(hole)
    if len(hole_ints) != HOLE_SIZE:
        raise ValueError(f"Expected {HOLE_SIZE} hole cards, got {len(hole_ints)}")
    assert_no_duplicates(hole_ints)
    return hole_ints


def parse_board(board: Sequence[int | str]) -> tuple[int, ...]:
    """Validate a public board (0, 3, 4 or 5 distinct cards); return it as ints."""
    board_ints = to_int_cards(board)
    if len(board_ints) not in VALID_BOARD_SIZES:
        raise ValueError(f"Board must have 0, 3, 4 or 5 cards, got {len(board_ints)}")
    assert_no_duplicates(board_ints)
    return board_ints


def parse_hole_board(
    hole: Sequence[int | str], board: Sequence[int | str]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate a hero hand and public board together (no card shared); return int tuples.

    Raises:
        ValueError: wrong number of hole/board cards, invalid card, or duplicates.
    """
    hole_ints, board_ints = parse_hole(hole), parse_board(board)
    assert_no_duplicates(hole_ints + board_ints)
    return hole_ints, board_ints
