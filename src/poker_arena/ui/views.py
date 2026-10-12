"""Translate engine state into pure :class:`~poker_arena.ui.table.TableView` objects.

Two builders with a deliberate difference in what they are allowed to read:

* :func:`spectator_table` reads a ``SpectatorView`` (full information) - Mode A, where both
  agents are watched and the user may peek at their cards.
* :func:`player_table` reads **only an** ``Observation`` (plus the finished ``HandResult``) - Mode C.
  An observation cannot contain the opponent's hole cards, so the rendered table cannot either;
  the opponent's cards are only drawn once the engine itself revealed them at showdown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from poker_arena.config import GameConfig
from poker_arena.engine.game import HandResult, SpectatorView
from poker_arena.engine.observation import Observation, PublicAction, Street
from poker_arena.ui.history import STREET_NAMES, bubble_text
from poker_arena.ui.table import SeatView, TableView

SHOWDOWN_LABEL = "Showdown"
HAND_OVER_LABEL = "Hand over"
FACE_DOWN: tuple[None, None] = (None, None)


@dataclass(frozen=True, slots=True)
class SeatIdentity:
    """How a seat is named: ``mention`` in logs and sentences, ``name``/``subtitle`` on the plate."""

    mention: str  # "Seat 2 (EBM)" | "You" | "Opponent (EBM)"
    name: str
    subtitle: str


def agent_identity(seat: int, agent_name: str, architecture: str) -> SeatIdentity:
    """Identity for an AI-controlled seat in the spectator mode."""
    title = f"Seat {seat + 1}"
    return SeatIdentity(f"{title} ({agent_name})", agent_name, f"{title} · {architecture}")


def hand_name(hole: Sequence[str], board: Sequence[str]) -> str | None:
    """Poker hand class (``"Two pair"``) from the equity package, or ``None`` if it is unavailable."""
    if len(board) < 3:
        return None
    try:
        from poker_arena.equity.evaluator import CATEGORY_NAMES, evaluate, hand_category
    except ImportError:
        return None
    return CATEGORY_NAMES[hand_category(evaluate([*hole, *board]))]


def showdown_hand_names(result: HandResult) -> dict[int, str]:
    """Hand class of every hand the engine revealed at showdown (empty if nobody showed)."""
    names = {seat: hand_name(hole, result.board) for seat, hole in result.shown_hands.items()}
    return {seat: name for seat, name in names.items() if name}


def street_label(street: Street, *, complete: bool, showdown: bool) -> str:
    if complete:
        return SHOWDOWN_LABEL if showdown else HAND_OVER_LABEL
    return STREET_NAMES[street]


def result_headline(
    result: HandResult,
    mentions: Sequence[str],
    hand_names: Mapping[int, str],
    *,
    you: int | None = None,
) -> str:
    """One sentence for the table banner: who won how much and why."""
    if result.winner is None:
        return "Split pot"
    winner = result.winner
    loser = 1 - winner
    subject = "You win" if winner == you else f"{mentions[winner]} wins"
    head = f"{subject} {result.payoffs[winner]:,} chips"
    if not result.showdown:
        folded = "You folded" if loser == you else f"{mentions[loser]} folded"
        return f"{head} - {folded}"
    if winner in hand_names and loser in hand_names:
        return f"{head} - {hand_names[winner]} beats {hand_names[loser]}"
    return head


def _blind_label(seat: int, button_seat: int) -> str:
    return "SB" if seat == button_seat else "BB"


def _bubble_for(last: PublicAction | None, seat_is_actor: bool) -> str | None:
    return bubble_text(last) if last is not None and seat_is_actor else None


def spectator_table(
    view: SpectatorView,
    identities: Sequence[SeatIdentity],
    *,
    config: GameConfig,
    hand_number: int,
    reveal_hole_cards: bool,
    badge: str | None = None,
    badge_tone: str = "info",
) -> TableView:
    """Table for the AI-vs-AI spectator. Seat 0 is drawn at the bottom, seat 1 at the top."""
    result = view.result
    showdown = result is not None and result.showdown
    hand_names = showdown_hand_names(result) if result is not None and showdown else {}
    last = view.events[-1] if view.events else None
    seats: list[SeatView] = []
    for seat in (0, 1):
        identity = identities[seat]
        hole = view.hole_cards[seat] if reveal_hole_cards or showdown else FACE_DOWN
        seats.append(
            SeatView(
                name=identity.name,
                subtitle=identity.subtitle,
                stack=view.stacks[seat],
                bet=view.bets[seat],
                hole=hole,
                is_dealer=seat == view.button_seat,
                blind=_blind_label(seat, view.button_seat),
                is_acting=view.actor_seat == seat,
                is_winner=result is not None and result.winner == seat,
                is_folded=last is not None and last.kind == "fold" and last.seat == seat,
                bubble=bubble_text(last) if last is not None and last.seat == seat else None,
                hand_name=hand_names.get(seat),
                payoff=result.payoffs[seat] if result is not None else None,
                is_all_in=view.stacks[seat] == 0 and not view.complete,
            )
        )
    mentions = [i.mention for i in identities]
    return TableView(
        top=seats[1],
        bottom=seats[0],
        board=view.board,
        pot=view.pot,
        street=street_label(view.street, complete=view.complete, showdown=showdown),
        big_blind=config.big_blind,
        info=f"Hand {hand_number + 1} · Blinds {config.small_blind}/{config.big_blind}",
        badge=badge,
        badge_tone=badge_tone,
        result_text=result_headline(result, mentions, hand_names) if result is not None else None,
    )


def player_table(
    obs: Observation,
    identities: Sequence[SeatIdentity],
    *,
    viewer_seat: int,
    actor_seat: int | None,
    result: HandResult | None = None,
    badge: str | None = None,
    badge_tone: str = "info",
) -> TableView:
    """Table from one player's point of view, built from the restricted ``Observation`` only.

    The viewer is drawn at the bottom. The opponent's hole cards stay face down unless ``result``
    (the engine's own outcome) lists them as shown at showdown.
    """
    opp_seat = 1 - viewer_seat
    showdown = result is not None and result.showdown
    shown = result.shown_hands if result is not None and showdown else {}
    hand_names = showdown_hand_names(result) if result is not None and showdown else {}
    last = obs.history[-1] if obs.history else None
    button_seat = viewer_seat if obs.is_button else opp_seat

    def seat_view(seat: int, *, mine: bool) -> SeatView:
        identity = identities[seat]
        if mine:
            hole: tuple[str | None, ...] = obs.my_hole
            stack, bet = obs.my_stack, obs.my_bet
        else:
            hole = tuple(shown[seat]) if seat in shown else FACE_DOWN
            stack, bet = obs.opp_stack, obs.opp_bet
        acted_last = last is not None and last.by_me == mine
        return SeatView(
            name=identity.name,
            subtitle=identity.subtitle,
            stack=stack,
            bet=bet,
            hole=hole,
            is_dealer=seat == button_seat,
            blind=_blind_label(seat, button_seat),
            is_acting=actor_seat == seat,
            is_winner=result is not None and result.winner == seat,
            is_folded=acted_last and last is not None and last.kind == "fold",
            bubble=_bubble_for(last, acted_last),
            hand_name=hand_names.get(seat),
            payoff=result.payoffs[seat] if result is not None else None,
            is_all_in=stack == 0 and result is None,
        )

    mentions = [i.mention for i in identities]
    return TableView(
        top=seat_view(opp_seat, mine=False),
        bottom=seat_view(viewer_seat, mine=True),
        board=obs.board,
        pot=obs.pot,
        street=street_label(obs.street, complete=result is not None, showdown=showdown),
        big_blind=obs.big_blind,
        info=f"Hand {obs.hand_number + 1} · Blinds {obs.small_blind}/{obs.big_blind}",
        badge=badge,
        badge_tone=badge_tone,
        result_text=(
            result_headline(result, mentions, hand_names, you=viewer_seat) if result is not None else None
        ),
    )
