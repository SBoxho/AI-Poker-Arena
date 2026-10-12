"""Concrete actions, legal-action description and the discretised action abstraction.

*Concrete* actions are what the poker engine understands:

* ``FOLD``
* ``CHECK_CALL``  (the engine decides whether it is a check or a call)
* ``RAISE_TO(n)`` (a bet or a raise; ``n`` is the actor's total chips committed on the
  *current street* after the action, i.e. PokerKit's "complete bet or raise to")

Learning agents usually pick from a small *abstract* action set (:class:`AbstractAction`).
:func:`legal_abstract_mask` and :func:`abstract_to_action` translate between the two while
respecting minimum raises, stack sizes and all-in edge cases. The engine remains the final
authority on legality - the abstraction can only ever produce actions the engine accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np


class IllegalActionError(ValueError):
    """Raised when an agent / user submits an action that the engine does not allow."""


class ActionKind(StrEnum):
    FOLD = "fold"
    CHECK_CALL = "check_call"
    RAISE_TO = "raise_to"


@dataclass(frozen=True, slots=True)
class Action:
    """A concrete engine action. ``amount`` is only meaningful for ``RAISE_TO``."""

    kind: ActionKind
    amount: int = 0

    @staticmethod
    def fold() -> "Action":
        return Action(ActionKind.FOLD)

    @staticmethod
    def check_call() -> "Action":
        return Action(ActionKind.CHECK_CALL)

    @staticmethod
    def raise_to(amount: int) -> "Action":
        return Action(ActionKind.RAISE_TO, int(amount))

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "amount": self.amount}

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(ActionKind(d["kind"]), int(d.get("amount", 0)))

    def __str__(self) -> str:
        return f"raise to {self.amount}" if self.kind is ActionKind.RAISE_TO else self.kind.value


@dataclass(frozen=True, slots=True)
class LegalActions:
    """What the player to act may legally do (all amounts in chips)."""

    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: int  # chips the actor must add to call (0 when checking is possible)
    can_raise: bool
    min_raise_to: int  # smallest legal total street bet when raising (0 if cannot raise)
    max_raise_to: int  # largest legal total street bet = all-in (0 if cannot raise)


# --------------------------------------------------------------------------------------
# Discretised action abstraction
# --------------------------------------------------------------------------------------
class AbstractAction(IntEnum):
    """Small action set used by the learned / evolved agents."""

    FOLD = 0
    CHECK_CALL = 1
    RAISE_SMALL = 2  # bet/raise ~1/2 pot
    RAISE_POT = 3  # bet/raise ~pot
    ALL_IN = 4


NUM_ABSTRACT_ACTIONS = len(AbstractAction)
ABSTRACT_ACTION_NAMES: tuple[str, ...] = tuple(a.name.lower() for a in AbstractAction)

RAISE_SMALL_POT_FRACTION = 0.5
RAISE_POT_POT_FRACTION = 1.0


def pot_fraction_raise_to(
    fraction: float, pot: int, my_bet: int, opp_bet: int, legal: LegalActions
) -> int:
    """Total street bet for a raise of ``fraction`` x (pot after calling), clamped to legal bounds.

    ``pot`` includes all chips already in the middle (including current street bets). After a call
    the pot is ``pot + to_call``; the raise adds ``fraction`` times that on top of the opponent's
    bet, which reproduces the usual "half pot" / "pot-sized" raise convention (a pot-sized bet on
    an unopened street equals the pot).
    """
    to_call = max(0, opp_bet - my_bet)
    target = opp_bet + round(fraction * (pot + to_call))
    return int(min(max(target, legal.min_raise_to), legal.max_raise_to))


def abstract_raise_amounts(
    pot: int, my_bet: int, opp_bet: int, legal: LegalActions
) -> dict[AbstractAction, int]:
    """Concrete raise-to amounts for each *distinct* legal raise abstraction.

    Duplicates are removed deterministically: if two abstractions map to the same chip amount only
    the more specific one survives (``ALL_IN`` > ``RAISE_SMALL`` > ``RAISE_POT``), so the set of
    legal abstract actions always corresponds to distinct concrete actions.
    """
    if not legal.can_raise:
        return {}
    out: dict[AbstractAction, int] = {}
    small = pot_fraction_raise_to(RAISE_SMALL_POT_FRACTION, pot, my_bet, opp_bet, legal)
    big = pot_fraction_raise_to(RAISE_POT_POT_FRACTION, pot, my_bet, opp_bet, legal)
    allin = legal.max_raise_to
    if small < allin:
        out[AbstractAction.RAISE_SMALL] = small
    if big < allin and big != small:
        out[AbstractAction.RAISE_POT] = big
    out[AbstractAction.ALL_IN] = allin
    return out


def legal_abstract_mask(pot: int, my_bet: int, opp_bet: int, legal: LegalActions) -> np.ndarray:
    """Boolean mask of length :data:`NUM_ABSTRACT_ACTIONS` of the currently legal abstract actions."""
    mask = np.zeros(NUM_ABSTRACT_ACTIONS, dtype=bool)
    mask[AbstractAction.FOLD] = legal.can_fold
    mask[AbstractAction.CHECK_CALL] = legal.can_check or legal.can_call
    for a in abstract_raise_amounts(pot, my_bet, opp_bet, legal):
        mask[a] = True
    return mask


def abstract_to_action(
    abstract: AbstractAction, pot: int, my_bet: int, opp_bet: int, legal: LegalActions
) -> Action:
    """Map an abstract action to a concrete one. Raises :class:`IllegalActionError` if not legal."""
    if abstract is AbstractAction.FOLD:
        if not legal.can_fold:
            raise IllegalActionError("Fold is not legal here (nothing to call).")
        return Action.fold()
    if abstract is AbstractAction.CHECK_CALL:
        if not (legal.can_check or legal.can_call):
            raise IllegalActionError("Neither check nor call is available.")
        return Action.check_call()
    amounts = abstract_raise_amounts(pot, my_bet, opp_bet, legal)
    if abstract not in amounts:
        raise IllegalActionError(f"{abstract.name} is not a legal distinct raise here.")
    return Action.raise_to(amounts[abstract])
