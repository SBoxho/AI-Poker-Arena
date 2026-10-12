"""Steppable hand sessions that connect agents (or a human) to the engine.

:class:`HandSession` is the single place where an :class:`~poker_arena.agents.base.Agent` meets the
poker engine. It builds the restricted observation, asks the agent, measures decision latency,
applies the action and finally feeds the public result back to both agents and opponent trackers.
Tournaments, dataset generation, GA fitness evaluation and the Streamlit modes all use it, so the
information boundary is identical everywhere.

* ``agents[seat] is None`` marks a seat controlled from outside (a human in the UI): call
  :meth:`HandSession.submit` for that seat.
* :func:`play_hand` runs a whole hand between two agents.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from poker_arena.agents.base import Agent, Decision
from poker_arena.config import GameConfig
from poker_arena.engine.actions import Action, IllegalActionError
from poker_arena.engine.deals import Deal
from poker_arena.engine.game import ActionEvent, HandEngine
from poker_arena.engine.history import DecisionRecord, HandRecord
from poker_arena.engine.observation import Observation
from poker_arena.engine.opponent import OpponentTracker


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Result of one decision: the engine event, the decision metadata and the observation used."""

    event: ActionEvent
    decision: DecisionRecord
    observation: Observation


class HandSession:
    """One hand being played; advance it action by action."""

    def __init__(
        self,
        config: GameConfig,
        deal: Deal,
        button_seat: int,
        agents: Sequence[Agent | None],
        *,
        start_stacks: tuple[int, int] | None = None,
        hand_number: int = 0,
        trackers: Sequence[OpponentTracker] | None = None,
        collect_explanations: bool = False,
    ) -> None:
        if len(agents) != 2:
            raise ValueError("Heads-up engine needs exactly two seats")
        self.agents = tuple(agents)
        self.trackers = tuple(trackers) if trackers is not None else (OpponentTracker(), OpponentTracker())
        self.collect_explanations = collect_explanations
        self.engine = HandEngine(config, deal, button_seat, start_stacks, hand_number)
        self._decisions: list[DecisionRecord] = []
        self._finished: HandRecord | None = None

    # ------------------------------------------------------------------ state
    @property
    def is_complete(self) -> bool:
        return self.engine.is_complete

    @property
    def actor_seat(self) -> int | None:
        return self.engine.actor_seat

    def observation(self, seat: int) -> Observation:
        """Restricted observation of ``seat`` including that seat's tracked opponent statistics."""
        return self.engine.observation_for(seat, self.trackers[seat].stats())

    # ----------------------------------------------------------------- acting
    def step(self) -> StepOutcome:
        """Let the agent whose turn it is act (not valid for human-controlled seats)."""
        seat = self.engine.actor_seat
        if seat is None:
            raise RuntimeError("The hand is complete.")
        agent = self.agents[seat]
        if agent is None:
            raise RuntimeError(f"Seat {seat} is externally controlled; call submit().")
        obs = self.observation(seat)
        start = time.perf_counter()
        decision = agent.act(obs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._apply(seat, obs, decision, latency_ms)

    def submit(self, action: Action) -> StepOutcome:
        """Apply an action for the externally controlled seat that is to act."""
        seat = self.engine.actor_seat
        if seat is None:
            raise RuntimeError("The hand is complete.")
        obs = self.observation(seat)
        return self._apply(seat, obs, Decision(action=action), 0.0)

    def _apply(self, seat: int, obs: Observation, decision: Decision, latency_ms: float) -> StepOutcome:
        explanation = None
        agent = self.agents[seat]
        if self.collect_explanations and agent is not None:
            explained = agent.explain(obs, decision)
            if explained is not None:
                explanation = {"summary": explained.summary, "details": explained.details}
        try:
            event = self.engine.act(decision.action)
        except IllegalActionError as exc:
            who = agent.name if agent is not None else f"seat {seat}"
            raise IllegalActionError(f"{who} attempted an illegal action {decision.action}: {exc}") from exc
        record = DecisionRecord(
            latency_ms=latency_ms,
            abstract=decision.abstract.name if decision.abstract is not None else None,
            explanation=explanation,
        )
        self._decisions.append(record)
        return StepOutcome(event, record, obs)

    # ----------------------------------------------------------------- finish
    def finish(self) -> HandRecord:
        """Close the hand: build the record, update trackers and notify both agents (once)."""
        if self._finished is not None:
            return self._finished
        if not self.engine.is_complete:
            raise RuntimeError("Cannot finish an unfinished hand.")
        eng = self.engine
        names = tuple(a.name if a is not None else "human" for a in self.agents)
        record = HandRecord(
            hand_number=eng.hand_number,
            config=eng.config,
            button_seat=eng.button_seat,
            start_stacks=eng.start_stacks,  # type: ignore[arg-type]
            deal=eng.deal,
            events=tuple(eng.events),
            result=eng.result(),
            agent_names=names,  # type: ignore[arg-type]
            decisions=tuple(self._decisions),
        )
        for seat in (0, 1):
            summary = record.public_summary(seat)
            self.trackers[seat].update(summary)
            agent = self.agents[seat]
            if agent is not None:
                agent.end_hand(summary)
        self._finished = record
        return record

    def play_out(self) -> HandRecord:
        """Run all remaining decisions (all seats must be agent-controlled)."""
        while not self.is_complete:
            self.step()
        return self.finish()


def play_hand(
    config: GameConfig,
    deal: Deal,
    button_seat: int,
    agents: Sequence[Agent],
    *,
    start_stacks: tuple[int, int] | None = None,
    hand_number: int = 0,
    trackers: Sequence[OpponentTracker] | None = None,
    collect_explanations: bool = False,
) -> HandRecord:
    """Play one complete hand between two agents and return its replayable record."""
    session = HandSession(
        config,
        deal,
        button_seat,
        agents,
        start_stacks=start_stacks,
        hand_number=hand_number,
        trackers=trackers,
        collect_explanations=collect_explanations,
    )
    return session.play_out()
