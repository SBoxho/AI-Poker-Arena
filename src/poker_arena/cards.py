"""Card encoding shared by every module.

Two interchangeable representations:

* **string**  ``"Ah"``, ``"Td"``, ``"2c"`` - human readable, JSON friendly, PokerKit compatible.
* **int**     ``rank_index * 4 + suit_index`` in ``0..51`` - used in hot loops / numpy code.

``rank_index``: 0..12 for ``2 3 4 5 6 7 8 9 T J Q K A``.
``suit_index``: 0..3  for ``c d h s``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"
NUM_CARDS = 52

ALL_CARDS: tuple[str, ...] = tuple(f"{r}{s}" for r in RANKS for s in SUITS)
"""All 52 cards as strings, ordered so that ``ALL_CARDS[i]`` is the string of int-card ``i``."""

_STR_TO_INT = {c: i for i, c in enumerate(ALL_CARDS)}


def card_to_int(card: str | int) -> int:
    """``"Ah"`` -> int. Ints are validated and returned unchanged."""
    if isinstance(card, int):
        if not 0 <= card < NUM_CARDS:
            raise ValueError(f"Card int out of range: {card}")
        return card
    try:
        return _STR_TO_INT[card]
    except KeyError:
        raise ValueError(f"Invalid card string: {card!r}") from None


def card_to_str(card: int | str) -> str:
    """int -> ``"Ah"``. Strings are validated and returned unchanged."""
    if isinstance(card, str):
        card_to_int(card)
        return card
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"Card int out of range: {card}")
    return ALL_CARDS[card]


def cards_to_ints(cards: Iterable[str | int]) -> tuple[int, ...]:
    return tuple(card_to_int(c) for c in cards)


def cards_to_strs(cards: Iterable[str | int]) -> tuple[str, ...]:
    return tuple(card_to_str(c) for c in cards)


def rank_of(card: str | int) -> int:
    """Rank index 0..12 (0 = deuce, 12 = ace)."""
    return card_to_int(card) // 4


def suit_of(card: str | int) -> int:
    """Suit index 0..3 (c, d, h, s)."""
    return card_to_int(card) % 4


def parse_cards(text: str) -> tuple[str, ...]:
    """``"AsKd"`` or ``"As Kd"`` -> ``("As", "Kd")``."""
    compact = text.replace(" ", "").replace(",", "")
    if len(compact) % 2:
        raise ValueError(f"Cannot parse cards from {text!r}")
    return cards_to_strs(compact[i : i + 2] for i in range(0, len(compact), 2))


def assert_no_duplicates(cards: Sequence[str | int]) -> None:
    ints = cards_to_ints(cards)
    if len(set(ints)) != len(ints):
        raise ValueError(f"Duplicate cards: {cards_to_strs(ints)}")


def shuffled_deck(rng: random.Random) -> list[str]:
    """A freshly shuffled 52-card deck using the supplied (seeded) RNG."""
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    return deck
