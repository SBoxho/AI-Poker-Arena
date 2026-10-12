"""Seeded, reproducible deals.

A :class:`Deal` is the *complete hidden card information* of one hand: both hole cards, three burn
cards and the five community cards, in dealing order. It lives only inside the engine / hand
record. Agents never see it - they only see :class:`~poker_arena.engine.observation.Observation`.

Cards are tied to **positions** (button / big blind), not to seats. Replaying the same ``Deal``
with the two players swapped between button and big blind gives a *duplicate* (matched) pair of
hands: two identical agents would exactly cancel out, so luck of the deal is removed from
head-to-head comparisons.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from poker_arena.cards import ALL_CARDS, assert_no_duplicates, cards_to_strs

DEAL_SIZE = 12  # 2 + 2 hole cards, 3 burn cards, 5 board cards


@dataclass(frozen=True, slots=True)
class Deal:
    """Hidden cards of one hand, laid out in dealing order.

    ``cards`` = button hole (2) + big-blind hole (2) + burn, flop(3) + burn, turn + burn, river.
    """

    cards: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.cards) != DEAL_SIZE:
            raise ValueError(f"A deal needs exactly {DEAL_SIZE} cards, got {len(self.cards)}")
        object.__setattr__(self, "cards", cards_to_strs(self.cards))
        assert_no_duplicates(self.cards)

    @property
    def button_hole(self) -> tuple[str, str]:
        return (self.cards[0], self.cards[1])

    @property
    def bb_hole(self) -> tuple[str, str]:
        return (self.cards[2], self.cards[3])

    @property
    def burns(self) -> tuple[str, str, str]:
        return (self.cards[4], self.cards[8], self.cards[10])

    @property
    def board(self) -> tuple[str, str, str, str, str]:
        c = self.cards
        return (c[5], c[6], c[7], c[9], c[11])

    @classmethod
    def from_rng(cls, rng: random.Random) -> "Deal":
        return cls(tuple(rng.sample(ALL_CARDS, DEAL_SIZE)))

    @classmethod
    def from_seed(cls, seed: int) -> "Deal":
        return cls.from_rng(random.Random(seed))

    def to_list(self) -> list[str]:
        return list(self.cards)

    @classmethod
    def from_list(cls, cards: list[str]) -> "Deal":
        return cls(tuple(cards))
