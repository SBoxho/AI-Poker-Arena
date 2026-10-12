"""Heads-up No-Limit Texas Hold'em hand engine built on PokerKit.

PokerKit performs the actual rules enforcement (betting order, min-raise rules, side pots,
showdown evaluation, chip settlement). :class:`HandEngine` adds

* **deterministic dealing** from a pre-set :class:`~poker_arena.engine.deals.Deal`,
* the **information boundary**: all hidden state stays inside this object; agents only ever get an
  :class:`~poker_arena.engine.observation.Observation` via :meth:`HandEngine.observation_for`,
* a seat-based API (seat 0 / seat 1 are persistent players; the button alternates between hands),
* a replayable action log (:class:`ActionEvent`).

Heads-up conventions (standard rules, enforced by PokerKit): the button posts the small blind and
acts *first* preflop and *last* on every later street; the other player posts the big blind.

``spectator_view`` / ``HandResult.shown_hands`` expose hidden cards for the UI and debugging.
**Agents and feature code must never import this module** - ``tests/test_information_boundary.py``
checks that.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from pokerkit import Automation, Mode, NoLimitTexasHoldem
from pokerkit.state import State

from poker_arena.config import GameConfig
from poker_arena.engine.actions import Action, ActionKind, IllegalActionError, LegalActions
from poker_arena.engine.deals import Deal
from poker_arena.engine.observation import (
    BOARD_CARDS_BY_STREET,
    Observation,
    OpponentStats,
    PublicAction,
    Street,
)

_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.RUNOUT_COUNT_SELECTION,
)
# PokerKit seats the big blind at player index 0 and the button / small blind at index 1.
_PK_BB, _PK_BUTTON = 0, 1
_STREET_BY_BOARD_LEN = {n: s for s, n in BOARD_CARDS_BY_STREET.items()}
_NO_ACTION = LegalActions(False, False, False, 0, False, 0, 0)


@dataclass(frozen=True, slots=True)
class ActionEvent:
    """One action in absolute-seat terms (the full, replayable log entry)."""

    seat: int
    street: Street
    action: Action
    kind: str  # fold | check | call | bet | raise
    chips: int
    street_total: int
    all_in: bool
    pot_after: int

    def to_dict(self) -> dict:
        return {
            "seat": self.seat,
            "street": int(self.street),
            "action": self.action.to_dict(),
            "kind": self.kind,
            "chips": self.chips,
            "street_total": self.street_total,
            "all_in": self.all_in,
            "pot_after": self.pot_after,
        }


@dataclass(frozen=True, slots=True)
class HandResult:
    """Outcome of a finished hand. ``payoffs`` are net chips won/lost per seat (sum is zero)."""

    payoffs: tuple[int, int]
    final_stacks: tuple[int, int]
    showdown: bool
    winner: int | None  # seat that won chips; None for an exact split
    total_pot: int  # chips contested: 2 x the smaller contribution (uncalled excess is returned)
    board: tuple[str, ...]
    shown_hands: dict[int, tuple[str, str]]  # seat -> hole cards, only filled at showdown
    ended_street: Street

    def to_dict(self) -> dict:
        return {
            "payoffs": list(self.payoffs),
            "final_stacks": list(self.final_stacks),
            "showdown": self.showdown,
            "winner": self.winner,
            "total_pot": self.total_pot,
            "board": list(self.board),
            "shown_hands": {str(s): list(h) for s, h in self.shown_hands.items()},
            "ended_street": int(self.ended_street),
        }


@dataclass(frozen=True, slots=True)
class SpectatorView:
    """Full table state INCLUDING hidden cards. For UI / debugging only - never for agents."""

    hole_cards: tuple[tuple[str, str], tuple[str, str]]  # by seat
    board: tuple[str, ...]
    pot: int
    stacks: tuple[int, int]
    bets: tuple[int, int]
    street: Street
    button_seat: int
    actor_seat: int | None
    events: tuple[ActionEvent, ...]
    complete: bool
    result: HandResult | None = None


@dataclass
class HandEngine:
    """One hand of heads-up NLHE. Create, then alternate ``observation_for`` / ``act`` until complete."""

    config: GameConfig
    deal: Deal
    button_seat: int
    start_stacks: tuple[int, int] | None = None
    hand_number: int = 0
    events: list[ActionEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.button_seat not in (0, 1):
            raise ValueError("button_seat must be 0 or 1")
        if self.start_stacks is None:
            self.start_stacks = (self.config.starting_stack, self.config.starting_stack)
        if len(self.start_stacks) != 2:
            raise ValueError("start_stacks needs exactly one stack per seat")
        # plain int tuple: callers may pass lists / numpy ints, which would break record equality and JSON
        self.start_stacks = (int(self.start_stacks[0]), int(self.start_stacks[1]))
        if any(s <= 0 for s in self.start_stacks):
            raise ValueError("Both players need a positive stack")
        self._pk_of_seat = {self.button_seat: _PK_BUTTON, 1 - self.button_seat: _PK_BB}
        self._seat_of_pk = {pk: seat for seat, pk in self._pk_of_seat.items()}
        pk_stacks = [self.start_stacks[self._seat_of_pk[pk]] for pk in (_PK_BB, _PK_BUTTON)]
        self._state: State = NoLimitTexasHoldem.create_state(
            _AUTOMATIONS,
            True,  # ante trimming (unused, no antes)
            0,
            (self.config.small_blind, self.config.big_blind),
            self.config.big_blind,
            pk_stacks,
            2,
            mode=Mode.CASH_GAME,
        )
        self._burns_dealt = 0
        self._board: list[str] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._state.deal_hole("".join(self.deal.bb_hole))  # dealee 0 is the big blind
            self._state.deal_hole("".join(self.deal.button_hole))
        # Blinds as actually posted (a short stack posts less). Captured before ``_advance`` because a
        # blind-induced all-in finishes the whole hand inside it, after which stacks include winnings.
        self._posted = (self.committed(0), self.committed(1))
        self._advance()

    # ------------------------------------------------------------------ queries
    @property
    def is_complete(self) -> bool:
        return not self._state.status

    @property
    def actor_seat(self) -> int | None:
        idx = self._state.actor_index
        return None if idx is None else self._seat_of_pk[idx]

    @property
    def board(self) -> tuple[str, ...]:
        """Community cards dealt so far (never the undealt ones)."""
        return tuple(self._board)

    @property
    def street(self) -> Street:
        return _STREET_BY_BOARD_LEN[len(self.board)]

    @property
    def pot(self) -> int:
        return int(self._state.total_pot_amount)

    def stack(self, seat: int) -> int:
        return int(self._state.stacks[self._pk_of_seat[seat]])

    def bet(self, seat: int) -> int:
        return int(self._state.bets[self._pk_of_seat[seat]])

    def committed(self, seat: int) -> int:
        """Chips this seat has put into the pot during this hand (including current street)."""
        return int(self.start_stacks[seat]) - self.stack(seat)

    def is_button(self, seat: int) -> bool:
        return seat == self.button_seat

    def legal_actions(self) -> LegalActions:
        """Legal actions for the player to act (all-False if the hand is over)."""
        st = self._state
        idx = st.actor_index
        if idx is None:
            return _NO_ACTION
        to_call = int(st.checking_or_calling_amount)
        opp_idx = 1 - idx
        # Raising makes sense only if PokerKit allows it *and* the opponent can still call something.
        can_raise = bool(st.can_complete_bet_or_raise_to()) and st.stacks[opp_idx] > 0
        lo = hi = 0
        if can_raise:
            hi = int(st.max_completion_betting_or_raising_to_amount)
            lo = min(int(st.min_completion_betting_or_raising_to_amount), hi)  # short all-in
        return LegalActions(
            can_fold=to_call > 0,  # folding when a free check exists is never allowed
            can_check=to_call == 0,
            can_call=to_call > 0,
            call_amount=to_call,
            can_raise=can_raise,
            min_raise_to=lo,
            max_raise_to=hi,
        )

    # --------------------------------------------------------------- observations
    def observation_for(
        self, seat: int, opponent_stats: OpponentStats | None = None
    ) -> Observation:
        """The restricted view of ``seat``. Contains no information the player could not know."""
        legal = self.legal_actions() if self.actor_seat == seat else _NO_ACTION
        opp = 1 - seat
        history = tuple(
            PublicAction(e.street, e.seat == seat, e.kind, e.chips, e.street_total, e.all_in, e.pot_after)
            for e in self.events
        )
        return Observation(
            hand_number=self.hand_number,
            street=self.street,
            is_button=self.is_button(seat),
            my_hole=self._hole_cards_of(seat),
            board=self.board,
            pot=self.pot,
            my_stack=self.stack(seat),
            opp_stack=self.stack(opp),
            my_bet=self.bet(seat),
            opp_bet=self.bet(opp),
            my_committed=self.committed(seat),
            opp_committed=self.committed(opp),
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            legal=legal,
            history=history,
            opponent_stats=opponent_stats if opponent_stats is not None else OpponentStats(),
        )

    def _hole_cards_of(self, seat: int) -> tuple[str, str]:
        """Hole cards of ``seat`` (private to that seat - used only to build its observation)."""
        return self.deal.button_hole if seat == self.button_seat else self.deal.bb_hole

    # --------------------------------------------------------------------- acting
    def act(self, action: Action) -> ActionEvent:
        """Apply ``action`` for the player to act. Raises :class:`IllegalActionError` if illegal."""
        seat = self.actor_seat
        if seat is None:
            raise IllegalActionError("The hand is over / no player is to act.")
        legal = self.legal_actions()
        st = self._state
        my_bet_before = self.bet(seat)
        opp_bet_before = self.bet(1 - seat)
        stack_before = self.stack(seat)
        pot_before = self.pot
        street = self.street

        if action.kind is ActionKind.FOLD:
            if not legal.can_fold:
                raise IllegalActionError("Cannot fold when a free check is available.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.fold()
            kind, chips, total = "fold", 0, my_bet_before
        elif action.kind is ActionKind.CHECK_CALL:
            chips = legal.call_amount
            st.check_or_call()
            kind = "call" if chips > 0 else "check"
            total = my_bet_before + chips
        elif action.kind is ActionKind.RAISE_TO:
            if not legal.can_raise:
                raise IllegalActionError("Raising is not legal here.")
            if not legal.min_raise_to <= action.amount <= legal.max_raise_to:
                raise IllegalActionError(
                    f"Raise to {action.amount} outside legal range "
                    f"[{legal.min_raise_to}, {legal.max_raise_to}]."
                )
            st.complete_bet_or_raise_to(action.amount)
            total = action.amount
            chips = total - my_bet_before
            kind = "raise" if opp_bet_before > 0 else "bet"
        else:  # pragma: no cover - enum is exhaustive
            raise IllegalActionError(f"Unknown action {action}")

        event = ActionEvent(
            seat=seat,
            street=street,
            action=action,
            kind=kind,
            chips=chips,
            street_total=total,
            all_in=chips > 0 and chips >= stack_before,
            pot_after=pot_before + chips,
        )
        self.events.append(event)
        self._advance()
        return event

    def _advance(self) -> None:
        """Deal burn + board cards whenever betting is closed and the hand is not over."""
        st = self._state
        board = self.deal.board
        while st.status and st.actor_index is None:
            if st.can_burn_card():
                st.burn_card(self.deal.burns[self._burns_dealt])
                self._burns_dealt += 1
            elif st.can_deal_board():
                dealt = len(self._board)
                new = board[dealt : dealt + (3 if dealt == 0 else 1)]
                st.deal_board("".join(new))
                self._board.extend(new)
            else:  # pragma: no cover - defensive; automations cover every other phase
                raise RuntimeError("Engine reached an unexpected state without an actor")

    # -------------------------------------------------------------------- results
    def result(self) -> HandResult:
        if not self.is_complete:
            raise RuntimeError("The hand is not finished yet.")
        st = self._state
        payoffs = tuple(int(st.payoffs[self._pk_of_seat[s]]) for s in (0, 1))
        final = tuple(self.stack(s) for s in (0, 1))
        showdown = not (self.events and self.events[-1].kind == "fold")
        winners = [s for s in (0, 1) if payoffs[s] > 0]
        shown = {s: self._hole_cards_of(s) for s in (0, 1)} if showdown else {}
        # ``committed()`` is meaningless once the pot has been paid out (stacks then include winnings),
        # so the contested pot is rebuilt from the posted blinds and the action log. Any uncalled excess
        # (a bet the opponent could not or did not match) is returned to its owner, hence 2 * the smaller
        # contribution - the usual "total pot" of a hand history.
        contributed = [
            self._posted[s] + sum(e.chips for e in self.events if e.seat == s) for s in (0, 1)
        ]
        return HandResult(
            payoffs=payoffs,  # type: ignore[arg-type]
            final_stacks=final,  # type: ignore[arg-type]
            showdown=showdown,
            winner=winners[0] if len(winners) == 1 else None,
            total_pot=2 * min(contributed),
            board=self.board,
            shown_hands=shown,
            ended_street=self.street,
        )

    # ------------------------------------------------------------------ spectator
    def spectator_view(self) -> SpectatorView:
        """Full state including both hole cards. UI / debugging only - NEVER give to an agent."""
        return SpectatorView(
            hole_cards=(self._hole_cards_of(0), self._hole_cards_of(1)),
            board=self.board,
            pot=self.pot,
            stacks=(self.stack(0), self.stack(1)),
            bets=(self.bet(0), self.bet(1)),
            street=self.street,
            button_seat=self.button_seat,
            actor_seat=self.actor_seat,
            events=tuple(self.events),
            complete=self.is_complete,
            result=self.result() if self.is_complete else None,
        )
