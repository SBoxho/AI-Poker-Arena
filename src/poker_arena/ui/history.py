"""Readable hand logs: turn engine events into short English lines (pure, no Streamlit).

Hole cards only appear in a log when the engine itself revealed them at showdown, so a log built
for the Human-vs-AI mode never leaks the opponent's hand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Literal, Protocol

from poker_arena.config import GameConfig
from poker_arena.engine.game import ActionEvent, HandResult
from poker_arena.engine.history import HandRecord
from poker_arena.engine.observation import Street
from poker_arena.ui.cards import card_text_html, cards_text

STREET_NAMES = {Street.PREFLOP: "Preflop", Street.FLOP: "Flop", Street.TURN: "Turn", Street.RIVER: "River"}
_BOARD_CARDS_BEFORE = {Street.PREFLOP: 0, Street.FLOP: 0, Street.TURN: 3, Street.RIVER: 4}
_BOARD_CARDS_AT = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}

EntryKind = Literal["info", "street", "action", "result"]


class ActionLike(Protocol):
    """Shared shape of ``ActionEvent`` and ``PublicAction`` (everything needed to describe an action)."""

    kind: str
    chips: int
    street_total: int
    all_in: bool


def describe_action(action: ActionLike) -> str:
    """``raises to 60``, ``calls 20``, ``checks`` ... (lower case, verb first)."""
    match action.kind:
        case "fold":
            text = "folds"
        case "check":
            text = "checks"
        case "call":
            text = f"calls {action.chips:,}"
        case "bet":
            text = f"bets {action.street_total:,}"
        case "raise":
            text = f"raises to {action.street_total:,}"
        case other:  # pragma: no cover - the engine only emits the kinds above
            text = other
    return f"{text} (all-in)" if action.all_in and action.kind != "fold" else text


def bubble_text(action: ActionLike) -> str:
    """Capitalised variant for the table's speech bubble."""
    text = describe_action(action)
    return text[:1].upper() + text[1:]


@dataclass(frozen=True, slots=True)
class LogEntry:
    kind: EntryKind
    text: str
    cards: tuple[str, ...] = ()  # board cards (street headers) or shown hole cards (results)
    seat: int | None = None


def _street_header(street: Street, board: tuple[str, ...]) -> LogEntry:
    new_cards = board[_BOARD_CARDS_BEFORE[street] : _BOARD_CARDS_AT[street]]
    return LogEntry("street", STREET_NAMES[street], cards=new_cards)


def build_hand_log(
    events: Sequence[ActionEvent],
    mentions: Sequence[str],
    *,
    config: GameConfig,
    button_seat: int,
    start_stacks: tuple[int, int],
    board: tuple[str, ...],
    result: HandResult | None = None,
    hand_names: Mapping[int, str] | None = None,
) -> list[LogEntry]:
    """Chronological log of one hand. ``mentions[seat]`` is how a seat is named (``"Seat 2 (EBM)"``).

    ``board`` is the full board known to the viewer; street headers only show cards of streets
    that were actually reached.
    """
    big_blind_seat = 1 - button_seat
    small = min(config.small_blind, start_stacks[button_seat])
    big = min(config.big_blind, start_stacks[big_blind_seat])
    log = [
        LogEntry(
            "info",
            f"{mentions[button_seat]} is on the button and posts the small blind ({small:,}); "
            f"{mentions[big_blind_seat]} posts the big blind ({big:,}).",
        ),
        _street_header(Street.PREFLOP, board),
    ]
    current = Street.PREFLOP
    for event in events:
        if event.street is not current:
            current = event.street
            log.append(_street_header(current, board))
        log.append(LogEntry("action", f"{mentions[event.seat]} {describe_action(event)}", seat=event.seat))
    if result is not None:
        log.extend(_result_entries(result, mentions, events, board, hand_names or {}, current))
    return log


def _result_entries(
    result: HandResult,
    mentions: Sequence[str],
    events: Sequence[ActionEvent],
    board: tuple[str, ...],
    hand_names: Mapping[int, str],
    last_street: Street,
) -> list[LogEntry]:
    entries: list[LogEntry] = []
    if result.showdown:
        # An all-in run-out deals the remaining streets without any betting.
        entries.extend(_street_header(s, board) for s in Street if last_street < s <= result.ended_street)
        for seat, hole in sorted(result.shown_hands.items()):
            name = hand_names.get(seat)
            suffix = f" - {name}" if name else ""
            entries.append(LogEntry("result", f"{mentions[seat]} shows{suffix}", cards=hole, seat=seat))
    if result.winner is None:
        entries.append(LogEntry("result", "The pot is split."))
    else:
        how = "" if result.showdown else " (opponent folded)"
        won = result.payoffs[result.winner]
        entries.append(LogEntry("result", f"{mentions[result.winner]} wins {won:,} chips{how}", seat=result.winner))
    return entries


def log_from_record(
    record: HandRecord,
    mentions: Sequence[str],
    *,
    hand_names: Mapping[int, str] | None = None,
) -> list[LogEntry]:
    """Convenience wrapper for a finished :class:`HandRecord` (board visible as dealt)."""
    return build_hand_log(
        record.events,
        mentions,
        config=record.config,
        button_seat=record.button_seat,
        start_stacks=record.start_stacks,
        board=record.result.board,
        result=record.result,
        hand_names=hand_names,
    )


def log_entry_text(entry: LogEntry) -> str:
    """Plain-text rendering (tests, copy/paste)."""
    return f"{entry.text} {cards_text(entry.cards)}".strip()


def render_log_html(entries: Sequence[LogEntry], *, newest_first: bool = True) -> str:
    """Scrollable HTML log; newest first keeps the latest action visible without scripting."""
    ordered = list(reversed(entries)) if newest_first else list(entries)
    rows = []
    for entry in ordered:
        cards = " ".join(card_text_html(c) for c in entry.cards)
        rows.append(
            f'<div class="pa-log-row pa-log-{entry.kind}" role="listitem">'
            f'{escape(entry.text)}{" " + cards if cards else ""}</div>'
        )
    return f'<div class="pa-log" role="list" aria-label="Hand history">{"".join(rows)}</div>'


def previous_hands_rows(
    records: Sequence[HandRecord], mentions: Sequence[str], big_blind: int, *, limit: int
) -> list[dict[str, object]]:
    """Table rows (newest first) summarising finished hands for the history expander."""
    rows: list[dict[str, object]] = []
    for record in list(records)[-limit:][::-1]:
        result = record.result
        if result.winner is None:
            outcome = "Split"
        else:
            outcome = f"{mentions[result.winner]} +{result.payoffs[result.winner]:,}"
        rows.append(
            {
                "Hand": record.hand_number + 1,
                "Button": mentions[record.button_seat],
                "Result": outcome,
                "Ended": "Showdown" if result.showdown else f"Fold ({STREET_NAMES[result.ended_street]})",
                f"{mentions[0]} (BB)": round(result.payoffs[0] / big_blind, 2),
                f"{mentions[1]} (BB)": round(result.payoffs[1] / big_blind, 2),
            }
        )
    return rows
