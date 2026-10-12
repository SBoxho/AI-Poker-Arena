"""Step-by-step replay frames of a stored hand (post-hoc spectator view, pure)."""

from __future__ import annotations

from dataclasses import dataclass

from poker_arena.engine.history import HandRecord
from poker_arena.ui.history import LogEntry, build_hand_log, describe_action
from poker_arena.ui.table import TableView
from poker_arena.ui.views import SeatIdentity, showdown_hand_names, spectator_table

REPLAY_BADGE = "Replay - both hands shown (post-hoc view)"
DEAL_CAPTION = "Cards dealt, blinds posted"


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    table: TableView
    caption: str  # what just happened
    log: tuple[LogEntry, ...]  # the hand history up to this frame


def replay_identities(record: HandRecord) -> tuple[SeatIdentity, SeatIdentity]:
    """Seat identities from the names stored in the record (seat 0 = first name)."""
    return tuple(  # type: ignore[return-value]
        SeatIdentity(f"Seat {seat + 1} ({name})", name, f"Seat {seat + 1}")
        for seat, name in enumerate(record.agent_names)
    )


def replay_frames(record: HandRecord) -> list[ReplayFrame]:
    """One frame for the deal and one after every action; the last one shows the result."""
    identities = replay_identities(record)
    mentions = [i.mention for i in identities]
    frames: list[ReplayFrame] = []
    for shown, view in enumerate(record.replay()):
        table = spectator_table(
            view,
            identities,
            config=record.config,
            hand_number=record.hand_number,
            reveal_hole_cards=True,
            badge=REPLAY_BADGE,
            badge_tone="warn",
        )
        names = showdown_hand_names(view.result) if view.result is not None and view.result.showdown else None
        log = build_hand_log(
            record.events[:shown],
            mentions,
            config=record.config,
            button_seat=record.button_seat,
            start_stacks=record.start_stacks,
            board=view.board,
            result=view.result,
            hand_names=names,
        )
        if shown == 0:
            caption = DEAL_CAPTION
        else:
            event = record.events[shown - 1]
            caption = f"{mentions[event.seat]} {describe_action(event)}"
        frames.append(ReplayFrame(table, caption, tuple(log)))
    return frames
