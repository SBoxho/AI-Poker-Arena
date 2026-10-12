"""A continuing heads-up match: the UI-independent game controller behind Modes A and C.

``Match`` owns the things that live across hands (stacks, opponent trackers, running profit) and
the current :class:`~poker_arena.engine.session.HandSession`. It contains no Streamlit code, so the
whole game flow is unit-testable and every browser session simply keeps its own ``Match`` in
``st.session_state``.

Seats may be ``None`` (a human): that seat acts through :meth:`Match.submit`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from poker_arena.agents.base import Agent
from poker_arena.config import GameConfig, derive_seed
from poker_arena.engine.actions import Action
from poker_arena.engine.deals import Deal
from poker_arena.engine.history import HandRecord
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.engine.session import HandSession, StepOutcome

MIN_HANDS_FOR_RATE = 20  # BB/100 over fewer hands is pure noise, so it is not displayed
HAND_HISTORY_LIMIT = 500  # finished hands kept for the history table (profit series is unbounded)
SEATS = (0, 1)


class TickKind(StrEnum):
    DEAL = "deal"  # a new hand was dealt, nobody acted yet
    ACTION = "action"  # exactly one decision was taken


@dataclass(frozen=True, slots=True)
class MatchSettings:
    config: GameConfig
    seed: int
    reset_stacks_each_hand: bool = False  # True: every hand starts with fresh equal stacks
    auto_rebuy: bool = True  # continuous play: a busted player is topped up automatically
    collect_explanations: bool = True  # ask agents to explain every decision (needed for the UI)


class Match:
    """State machine of one table: deal -> decisions -> result -> next hand."""

    def __init__(self, settings: MatchSettings, agents: Sequence[Agent | None]) -> None:
        if len(agents) != len(SEATS):
            raise ValueError("A heads-up match needs exactly two seats")
        self.settings = settings
        self.agents: tuple[Agent | None, Agent | None] = (agents[0], agents[1])
        self.trackers = (OpponentTracker(), OpponentTracker())
        self.stacks = [settings.config.starting_stack] * 2
        self.first_button = derive_seed("ui", settings.seed, "button") % 2
        self.hand_number = -1  # index of the current (or last) hand
        self.session: HandSession | None = None
        self.steps: list[StepOutcome] = []
        self.records: list[HandRecord] = []
        self.hands_completed = 0
        self.profit_chips = [0, 0]
        self.profit_series_bb: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ queries
    @property
    def config(self) -> GameConfig:
        return self.settings.config

    @property
    def human_seat(self) -> int | None:
        return next((s for s in SEATS if self.agents[s] is None), None)

    @property
    def hand_complete(self) -> bool:
        return self.session is not None and self.session.is_complete

    @property
    def hand_active(self) -> bool:
        return self.session is not None and not self.session.is_complete

    @property
    def actor_seat(self) -> int | None:
        return self.session.actor_seat if self.session is not None else None

    @property
    def decision_index(self) -> int:
        """Number of decisions taken in the current hand; identifies "the current decision"."""
        return len(self.steps)

    @property
    def awaiting_agent(self) -> bool:
        """True when an agent (not a human) is to act."""
        seat = self.actor_seat
        return seat is not None and self.agents[seat] is not None

    @property
    def awaiting_human(self) -> bool:
        seat = self.actor_seat
        return seat is not None and self.agents[seat] is None

    @property
    def last_record(self) -> HandRecord | None:
        return self.records[-1] if self.records else None

    @property
    def has_bust(self) -> bool:
        return any(stack <= 0 for stack in self.stacks)

    @property
    def needs_rebuy(self) -> bool:
        """A player is out of chips and the match is not configured to refill automatically."""
        s = self.settings
        return self.has_bust and not (s.auto_rebuy or s.reset_stacks_each_hand)

    def profit_bb(self, seat: int) -> float:
        return self.profit_chips[seat] / self.config.big_blind

    def bb_per_100(self, seat: int) -> float | None:
        """Win rate in big blinds per 100 hands (``None`` until enough hands were played)."""
        if self.hands_completed < MIN_HANDS_FOR_RATE:
            return None
        return self.profit_bb(seat) / self.hands_completed * 100.0

    # --------------------------------------------------------------- hand flow
    def rebuy(self) -> None:
        """Give both players a fresh starting stack (the UI's answer to a bust)."""
        self.stacks = [self.config.starting_stack] * 2

    def start_hand(self) -> HandSession:
        """Deal the next hand (the previous one must be finished)."""
        if self.hand_active:
            raise RuntimeError("The current hand is still in progress.")
        if self.needs_rebuy:
            raise RuntimeError("A player is bust - call rebuy() first.")
        if self.settings.reset_stacks_each_hand:
            self.rebuy()
        else:
            self.stacks = [s if s > 0 else self.config.starting_stack for s in self.stacks]
        self.hand_number += 1
        seed = self.settings.seed
        deal = Deal.from_seed(derive_seed("ui", seed, self.hand_number))
        self.session = HandSession(
            self.config,
            deal,
            (self.first_button + self.hand_number) % 2,
            self.agents,
            start_stacks=(self.stacks[0], self.stacks[1]),
            hand_number=self.hand_number,
            trackers=self.trackers,
            collect_explanations=self.settings.collect_explanations,
        )
        self.steps = []
        return self.session

    def step(self) -> StepOutcome:
        """Let the agent whose turn it is take one decision."""
        if self.session is None or not self.awaiting_agent:
            raise RuntimeError("No agent is waiting to act.")
        outcome = self.session.step()
        self._record(outcome)
        return outcome

    def submit(self, action: Action, *, decision_index: int) -> StepOutcome | None:
        """Apply the human's action - only if it is still the decision the player was looking at.

        Returns ``None`` (and changes nothing) for stale clicks such as a double click.
        """
        if self.session is None or not self.awaiting_human or decision_index != self.decision_index:
            return None
        outcome = self.session.submit(action)
        self._record(outcome)
        return outcome

    def advance(self) -> TickKind:
        """One autoplay tick: deal if no hand is running, otherwise take one decision."""
        if self.session is None or self.session.is_complete:
            self.start_hand()
            return TickKind.DEAL
        self.step()
        return TickKind.ACTION

    def play_out(self) -> HandRecord:
        """Finish the current hand instantly (dealing a new one if none is running)."""
        if self.session is None or self.session.is_complete:
            self.start_hand()
        while self.hand_active:
            self.step()
        return self.records[-1]

    # ---------------------------------------------------------------- internals
    def _record(self, outcome: StepOutcome) -> None:
        self.steps.append(outcome)
        session = self.session
        if session is not None and session.is_complete:
            self._finish_hand(session)

    def _finish_hand(self, session: HandSession) -> None:
        record = session.finish()
        self.records.append(record)
        del self.records[:-HAND_HISTORY_LIMIT]
        self.stacks = list(record.result.final_stacks)
        self.hands_completed += 1
        for seat in SEATS:
            self.profit_chips[seat] += record.result.payoffs[seat]
        self.profit_series_bb.append((self.profit_bb(0), self.profit_bb(1)))


# ------------------------------------------------------------------------------ pacing
AUTOPLAY_BASE_SECONDS = 1.5  # seconds per action at 1x speed
MIN_SPEED, MAX_SPEED, DEFAULT_SPEED = 0.2, 3.0, 1.0
HAND_END_DWELL = 2.0  # the finished hand stays on screen this many ticks before the next deal
TICK_TOLERANCE = 0.9  # a timer firing slightly early still counts as a tick


def tick_interval(speed: float) -> float:
    """Seconds between autoplay ticks for a speed multiplier."""
    return AUTOPLAY_BASE_SECONDS / max(speed, MIN_SPEED)


@dataclass
class Pacer:
    """Decides whether an autoplay tick is due.

    Fragment timers also fire on every full rerun (e.g. when the speed slider moves); gating on
    elapsed time prevents double steps and makes the hand-end pause independent of the timer.
    """

    last_tick: float = float("-inf")

    def due(self, now: float, interval: float, *, after_hand_end: bool = False) -> bool:
        needed = interval * (HAND_END_DWELL if after_hand_end else 1.0) * TICK_TOLERANCE
        return now - self.last_tick >= needed

    def mark(self, now: float) -> None:
        self.last_tick = now
