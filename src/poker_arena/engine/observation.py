"""The restricted view of the game that a player is allowed to see.

:class:`Observation` is the **only** object handed to an agent (during training-data generation,
simulation, tournaments and live play alike). It is a frozen bundle of plain values - it holds no
reference to the engine, the deck, the opponent's cards or the random seed.

An agent may legitimately know:

* its own hole cards and the *revealed* community cards,
* stacks, bets, pot, blinds, position, the legal actions,
* the public action history of the current hand,
* statistics about the opponent derived from *previously observed* public actions.

Anything else (opponent hole cards, undealt cards, deck order, future board, seeds) is simply not
representable here. ``tests/test_information_boundary.py`` enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from poker_arena.engine.actions import AbstractAction, LegalActions, legal_abstract_mask  # noqa: F401


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


BOARD_CARDS_BY_STREET = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


@dataclass(frozen=True, slots=True)
class PublicAction:
    """One action everybody at the table saw."""

    street: Street
    by_me: bool  # True if the observing player made it (False = the opponent)
    kind: str  # "fold" | "check" | "call" | "bet" | "raise"
    chips: int  # chips this action added to the pot
    street_total: int  # actor's total chips on this street after the action
    all_in: bool
    pot_after: int

    def to_dict(self) -> dict:
        return {
            "street": int(self.street),
            "by_me": self.by_me,
            "kind": self.kind,
            "chips": self.chips,
            "street_total": self.street_total,
            "all_in": self.all_in,
            "pot_after": self.pot_after,
        }


@dataclass(frozen=True, slots=True)
class OpponentStats:
    """Opponent tendencies computed *only* from previously observed public actions.

    All rates are smoothed towards a neutral prior so they are well defined from hand one.
    """

    hands: int = 0
    vpip: float = 0.5  # voluntarily put chips in preflop (calls/raises beyond the forced blind)
    pfr: float = 0.3  # raised preflop
    aggression: float = 0.4  # (bets+raises) / (bets+raises+calls), postflop
    fold_to_bet: float = 0.4  # folds / times facing a bet or raise
    showdown_rate: float = 0.3  # share of hands that reached a showdown

    def to_dict(self) -> dict:
        return {
            "hands": self.hands,
            "vpip": self.vpip,
            "pfr": self.pfr,
            "aggression": self.aggression,
            "fold_to_bet": self.fold_to_bet,
            "showdown_rate": self.showdown_rate,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything the player to act is allowed to know (amounts are chips)."""

    hand_number: int
    street: Street
    is_button: bool  # heads-up: the button is the small blind; acts first preflop, last postflop
    my_hole: tuple[str, str]
    board: tuple[str, ...]  # revealed community cards only (0, 3, 4 or 5 cards)
    pot: int  # all chips in the middle, including bets on the current street
    my_stack: int  # chips behind
    opp_stack: int
    my_bet: int  # chips committed on the current street
    opp_bet: int
    my_committed: int  # chips committed over the whole hand
    opp_committed: int
    small_blind: int
    big_blind: int
    legal: LegalActions
    history: tuple[PublicAction, ...]
    opponent_stats: OpponentStats = field(default_factory=OpponentStats)

    # ---- derived (pure functions of the fields above) ----
    @property
    def to_call(self) -> int:
        return self.legal.call_amount

    @property
    def num_raises(self) -> int:
        """Bets/raises made so far in this hand (by either player)."""
        return sum(1 for a in self.history if a.kind in ("bet", "raise"))

    @property
    def num_raises_this_street(self) -> int:
        return sum(1 for a in self.history if a.street == self.street and a.kind in ("bet", "raise"))

    @property
    def effective_stack(self) -> int:
        """Most that can still be won/lost by either player on this hand beyond the current bets.

        Never negative: when facing a bet larger than the shorter stack (an overbet that will be
        partly returned) there is simply nothing more to put in after calling.
        """
        shorter_total = min(self.my_stack + self.my_committed, self.opp_stack + self.opp_committed)
        return max(0, shorter_total - max(self.my_committed, self.opp_committed))

    @property
    def abstract_mask(self):
        """Boolean mask over :class:`AbstractAction` of legal discretised actions."""
        return legal_abstract_mask(self.pot, self.my_bet, self.opp_bet, self.legal)

    def to_dict(self) -> dict:
        return {
            "hand_number": self.hand_number,
            "street": int(self.street),
            "is_button": self.is_button,
            "my_hole": list(self.my_hole),
            "board": list(self.board),
            "pot": self.pot,
            "my_stack": self.my_stack,
            "opp_stack": self.opp_stack,
            "my_bet": self.my_bet,
            "opp_bet": self.opp_bet,
            "my_committed": self.my_committed,
            "opp_committed": self.opp_committed,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "legal": {
                "can_fold": self.legal.can_fold,
                "can_check": self.legal.can_check,
                "can_call": self.legal.can_call,
                "call_amount": self.legal.call_amount,
                "can_raise": self.legal.can_raise,
                "min_raise_to": self.legal.min_raise_to,
                "max_raise_to": self.legal.max_raise_to,
            },
            "history": [a.to_dict() for a in self.history],
            "opponent_stats": self.opponent_stats.to_dict(),
        }
