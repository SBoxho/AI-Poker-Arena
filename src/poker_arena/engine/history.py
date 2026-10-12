"""Replayable hand records (full information - for analysis and UI replays, never for agents)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from poker_arena.config import GameConfig
from poker_arena.engine.actions import Action
from poker_arena.engine.deals import Deal
from poker_arena.engine.game import ActionEvent, HandEngine, HandResult, SpectatorView
from poker_arena.engine.observation import PublicAction, Street
from poker_arena.engine.opponent import PublicHandSummary


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Per-action annotation produced by the session (index aligns with ``HandRecord.events``)."""

    # Wall-clock time is not part of a hand's identity: excluding it keeps "same seeds => equal HandRecords".
    latency_ms: float = field(compare=False)
    abstract: str | None = None  # name of the discretised action, if the agent used one
    explanation: dict[str, Any] | None = None  # {"summary": str, "details": {...}} when requested

    def to_dict(self) -> dict:
        return {"latency_ms": self.latency_ms, "abstract": self.abstract, "explanation": self.explanation}


@dataclass(frozen=True)
class HandRecord:
    """Everything needed to reproduce and replay one hand exactly."""

    hand_number: int
    config: GameConfig
    button_seat: int
    start_stacks: tuple[int, int]
    deal: Deal
    events: tuple[ActionEvent, ...]
    result: HandResult
    agent_names: tuple[str, str] = ("seat0", "seat1")
    decisions: tuple[DecisionRecord, ...] = field(default_factory=tuple)

    # ---------------------------------------------------------------- replay
    def replay(self) -> Iterator[SpectatorView]:
        """Re-run the hand through the real engine, yielding the table after every action."""
        engine = HandEngine(self.config, self.deal, self.button_seat, self.start_stacks, self.hand_number)
        yield engine.spectator_view()
        for ev in self.events:
            engine.act(ev.action)
            yield engine.spectator_view()

    def verify(self) -> bool:
        """True iff replaying the actions reproduces the recorded result exactly."""
        last = None
        for last in self.replay():
            pass
        return last is not None and last.result == self.result

    # ---------------------------------------------------------- public views
    def public_summary(self, seat: int) -> PublicHandSummary:
        """The post-hand information available to ``seat`` (no private cards of the opponent
        unless they were shown at showdown)."""
        history = tuple(
            PublicAction(e.street, e.seat == seat, e.kind, e.chips, e.street_total, e.all_in, e.pot_after)
            for e in self.events
        )
        shown = self.result.shown_hands.get(1 - seat) if self.result.showdown else None
        return PublicHandSummary(
            hand_number=self.hand_number,
            was_button=seat == self.button_seat,
            history=history,
            board=self.result.board,
            my_payoff=self.result.payoffs[seat],
            showdown=self.result.showdown,
            opp_hole=shown,
        )

    # --------------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {
            "hand_number": self.hand_number,
            "config": self.config.to_dict(),
            "button_seat": self.button_seat,
            "start_stacks": list(self.start_stacks),
            "deal": self.deal.to_list(),
            "events": [e.to_dict() for e in self.events],
            "result": self.result.to_dict(),
            "agent_names": list(self.agent_names),
            "decisions": [d.to_dict() for d in self.decisions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HandRecord":
        events = tuple(
            ActionEvent(
                seat=e["seat"],
                street=Street(e["street"]),
                action=Action.from_dict(e["action"]),
                kind=e["kind"],
                chips=e["chips"],
                street_total=e["street_total"],
                all_in=e["all_in"],
                pot_after=e["pot_after"],
            )
            for e in d["events"]
        )
        r = d["result"]
        result = HandResult(
            payoffs=tuple(r["payoffs"]),  # type: ignore[arg-type]
            final_stacks=tuple(r["final_stacks"]),  # type: ignore[arg-type]
            showdown=r["showdown"],
            winner=r["winner"],
            total_pot=r["total_pot"],
            board=tuple(r["board"]),
            shown_hands={int(s): tuple(h) for s, h in r["shown_hands"].items()},  # type: ignore[misc]
            ended_street=Street(r["ended_street"]),
        )
        return cls(
            hand_number=d["hand_number"],
            config=GameConfig.from_dict(d["config"]),
            button_seat=d["button_seat"],
            start_stacks=tuple(d["start_stacks"]),  # type: ignore[arg-type]
            deal=Deal.from_list(d["deal"]),
            events=events,
            result=result,
            agent_names=tuple(d.get("agent_names", ("seat0", "seat1"))),  # type: ignore[arg-type]
            decisions=tuple(
                DecisionRecord(x["latency_ms"], x.get("abstract"), x.get("explanation"))
                for x in d.get("decisions", [])
            ),
        )
