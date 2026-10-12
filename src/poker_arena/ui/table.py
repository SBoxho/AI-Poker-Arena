"""Poker table renderer: a pure ``TableView`` data object -> self-contained HTML/CSS.

This module knows nothing about Streamlit or the poker engine. Whatever is *not* in the
``TableView`` cannot be drawn, so the anti-cheating guarantee of the Human-vs-AI mode reduces to
"never put the opponent's hole cards into the ``TableView``" (face-down cards are ``None``).

The table scales with its container (CSS container queries, ``cqw`` units) so it stays readable
from a phone to a wide desktop screen.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from html import escape

from poker_arena.ui.cards import CARD_CSS, card_name, render_card_slot, render_hand

BOARD_SLOTS = 5
AVATAR_HUES = 360


def format_chips(amount: int) -> str:
    return f"{amount:,}"


def format_bb(chips: int, big_blind: int) -> str:
    """``chips`` expressed in big blinds, trimmed (``"100"``, ``"2.5"``)."""
    value = chips / big_blind
    return f"{value:.1f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


@dataclass(frozen=True, slots=True)
class SeatView:
    """Everything the table draws for one seat. ``hole`` entries equal to ``None`` are face down."""

    name: str
    subtitle: str
    stack: int
    bet: int = 0
    hole: tuple[str | None, ...] = ()
    is_dealer: bool = False
    blind: str | None = None  # "SB" | "BB"
    is_acting: bool = False
    is_winner: bool = False
    is_folded: bool = False
    bubble: str | None = None  # last action, e.g. "Raises to 60"
    hand_name: str | None = None  # shown at showdown, e.g. "Two pair"
    payoff: int | None = None  # net chips won/lost, set when the hand is over
    is_all_in: bool = False


@dataclass(frozen=True, slots=True)
class TableView:
    """Complete description of the table at one instant. ``top`` is drawn opposite the viewer."""

    top: SeatView
    bottom: SeatView
    board: tuple[str, ...] = ()
    pot: int = 0
    street: str = "Preflop"
    big_blind: int = 10
    info: str = ""  # top-left caption, e.g. "Hand 12 · Blinds 5/10"
    badge: str | None = None  # top-right caption, e.g. "Your turn"
    badge_tone: str = "info"  # "info" | "turn" | "wait" | "warn"
    result_text: str | None = None  # shown under the board when the hand is over


# ----------------------------------------------------------------------------- rendering
def _avatar(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") % AVATAR_HUES
    initial = escape(name.strip()[:1].upper() or "?")
    return f'<span class="pa-avatar" style="--hue:{hue}" aria-hidden="true">{initial}</span>'


def _signed(amount: int) -> str:
    return f"+{format_chips(amount)}" if amount > 0 else format_chips(amount)


def _tags(seat: SeatView) -> str:
    tags: list[str] = []
    if seat.is_winner:
        payoff = f" {_signed(seat.payoff)}" if seat.payoff else ""
        tags.append(f'<span class="pa-tag pa-tag--win">Winner{payoff}</span>')
    elif seat.payoff is not None and seat.payoff < 0:
        tags.append(f'<span class="pa-tag pa-tag--loss">{_signed(seat.payoff)}</span>')
    if seat.is_acting:
        tags.append('<span class="pa-tag pa-tag--turn">To act</span>')
    if seat.is_folded:
        tags.append('<span class="pa-tag pa-tag--fold">Folded</span>')
    if seat.is_all_in and not seat.is_folded:
        tags.append('<span class="pa-tag pa-tag--allin">All-in</span>')
    if seat.hand_name:
        tags.append(f'<span class="pa-tag pa-tag--hand">{escape(seat.hand_name)}</span>')
    return f'<div class="pa-tags">{"".join(tags)}</div>' if tags else ""


def _seat_html(seat: SeatView, position: str, big_blind: int) -> str:
    classes = ["pa-seat", f"pa-seat--{position}"]
    for flag, css in (
        (seat.is_acting, "pa-seat--acting"),
        (seat.is_winner, "pa-seat--winner"),
        (seat.is_folded, "pa-seat--folded"),
    ):
        if flag:
            classes.append(css)
    cards = render_hand(seat.hole, extra_class="pa-card--win" if seat.is_winner else "")
    blind = f'<span class="pa-tag pa-tag--blind">{escape(seat.blind)}</span>' if seat.blind else ""
    dealer = '<span class="pa-dealer" title="Dealer button">D</span>' if seat.is_dealer else ""
    bubble = f'<div class="pa-bubble" role="status">{escape(seat.bubble)}</div>' if seat.bubble else ""
    return (
        f'<div class="{" ".join(classes)}">'
        f'<div class="pa-hole">{cards}</div>'
        f'<div class="pa-plate">{dealer}{_avatar(seat.name)}'
        '<div class="pa-plate-text">'
        f'<div class="pa-name">{escape(seat.name)}{blind}</div>'
        f'<div class="pa-sub">{escape(seat.subtitle)}</div>'
        f'<div class="pa-stack">{format_chips(seat.stack)}'
        f'<small>{format_bb(seat.stack, big_blind)} BB</small></div>'
        f"{_tags(seat)}</div></div>{bubble}</div>"
    )


def _bet_html(seat: SeatView, position: str) -> str:
    if seat.bet <= 0:
        return ""
    return (
        f'<div class="pa-bet pa-bet--{position}" aria-label="Bet {format_chips(seat.bet)}">'
        f'<span class="pa-chip"></span><b>{format_chips(seat.bet)}</b></div>'
    )


def _board_html(board: tuple[str, ...]) -> str:
    slots = "".join(render_card_slot() for _ in range(BOARD_SLOTS - len(board)))
    return f'<div class="pa-board">{render_hand(board)}{slots}</div>'


def _hole_summary(seat: SeatView) -> str:
    if not seat.hole:
        return "no cards"
    return ", ".join("face-down card" if c is None else card_name(c) for c in seat.hole)


def accessible_summary(view: TableView) -> str:
    """Screen-reader sentence built only from the data the table shows (so it cannot leak)."""
    board = ", ".join(card_name(c) for c in view.board) or "no community cards yet"
    parts = [
        f"{view.street}. Pot {format_chips(view.pot)}. Board: {board}.",
        *(
            f"{seat.name}: stack {format_chips(seat.stack)}, bet {format_chips(seat.bet)}, "
            f"cards: {_hole_summary(seat)}."
            for seat in (view.top, view.bottom)
        ),
    ]
    if view.result_text:
        parts.append(view.result_text)
    return " ".join(parts)


def render_table_html(view: TableView, *, include_css: bool = False) -> str:
    """HTML for the whole table. Set ``include_css`` for a standalone document fragment."""
    badge = (
        f'<div class="pa-badge pa-badge--{escape(view.badge_tone)}">{escape(view.badge)}</div>'
        if view.badge
        else ""
    )
    result = f'<div class="pa-result">{escape(view.result_text)}</div>' if view.result_text else ""
    style = f"<style>{TABLE_CSS}</style>" if include_css else ""
    return (
        f"{style}"
        '<div class="pa-table-wrap">'
        f'<div class="pa-table" role="group" aria-label="Poker table">'
        f'<p class="pa-sr-only">{escape(accessible_summary(view))}</p>'
        '<div class="pa-felt"><div class="pa-felt-inner"></div></div>'
        f'<div class="pa-info">{escape(view.info)}</div>{badge}'
        f"{_seat_html(view.top, 'top', view.big_blind)}"
        f"{_bet_html(view.top, 'top')}"
        '<div class="pa-center">'
        f'<div class="pa-pot"><span class="pa-chip"></span>Pot <b>{format_chips(view.pot)}</b></div>'
        f"{_board_html(view.board)}"
        f'<div class="pa-street">{escape(view.street)}</div>{result}</div>'
        f"{_bet_html(view.bottom, 'bottom')}"
        f"{_seat_html(view.bottom, 'bottom', view.big_blind)}"
        "</div></div>"
    )


def table_document(view: TableView) -> str:
    """Self-contained HTML (card + table CSS included), e.g. for notebooks or exports."""
    return f"<style>{CARD_CSS}</style>{render_table_html(view, include_css=True)}"


TABLE_CSS = """
.pa-table-wrap { container-type: inline-size; width: 100%; max-width: 1000px; margin: 0 auto; }
.pa-table {
  --card-fs: 2.3cqw; --gold: #f5c542; --accent: #2ecc71;
  position: relative; width: 100%; aspect-ratio: 16 / 11; overflow: hidden;
  font-size: max(9.5px, 1.7cqw); line-height: 1.2; color: #e9f1ec;
  font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  border-radius: 1.6em;
  background: radial-gradient(ellipse at 50% 40%, #12201a 0%, #0a120f 100%);
  box-shadow: 0 .6em 2.2em rgba(0,0,0,.5), inset 0 0 0 .1em rgba(255,255,255,.05);
}
.pa-table * { box-sizing: border-box; }
.pa-sr-only {
  position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}
.pa-felt {
  position: absolute; inset: 3.1em 1em; border-radius: 50%;
  background: linear-gradient(180deg, #5b3b24 0%, #3b2616 100%);
  box-shadow: 0 .5em 1.6em rgba(0,0,0,.6), inset 0 .15em .3em rgba(255,255,255,.18),
              inset 0 0 0 .22em #c9a24b33;
  padding: 1.1em;
}
.pa-felt-inner {
  width: 100%; height: 100%; border-radius: 50%;
  background: radial-gradient(ellipse at 50% 38%, #2b9763 0%, #1c7a4d 38%, #115a38 74%, #0c4529 100%);
  box-shadow: inset 0 0 3em rgba(0,0,0,.55), inset 0 0 0 .14em rgba(0,0,0,.35);
}
.pa-info {
  position: absolute; top: .9em; left: 1.2em; z-index: 4; font-size: .95em; color: #a9bdb1;
  background: rgba(0,0,0,.35); padding: .3em .8em; border-radius: 1em;
}
.pa-badge {
  position: absolute; top: .9em; right: 1.2em; z-index: 4; font-size: .95em; font-weight: 700;
  padding: .3em .9em; border-radius: 1em; letter-spacing: .02em; max-width: 15em; text-align: center;
  line-height: 1.2;
}
.pa-badge--info { background: rgba(90,169,255,.18); color: #9fd0ff; border: .08em solid rgba(90,169,255,.5); }
.pa-badge--turn { background: rgba(46,204,113,.22); color: #7dffb0; border: .08em solid rgba(46,204,113,.7); }
.pa-badge--wait { background: rgba(245,197,66,.16); color: #ffe08a; border: .08em solid rgba(245,197,66,.55); }
.pa-badge--warn { background: rgba(229,72,77,.18); color: #ff9da0; border: .08em solid rgba(229,72,77,.6); }

/* seats */
.pa-seat { position: absolute; left: 50%; transform: translateX(-50%); z-index: 3; display: flex;
  align-items: center; gap: .9em; }
.pa-seat--top { top: .9em; }
.pa-seat--bottom { bottom: .9em; }
.pa-hole { display: flex; gap: .35em; min-width: 8.6em; justify-content: flex-end; }
.pa-plate {
  position: relative; display: flex; align-items: center; gap: .8em; min-width: 15.5em;
  padding: .6em 1em .6em .7em; border-radius: 1.3em;
  background: linear-gradient(180deg, #1c2a24 0%, #111b16 100%);
  border: .13em solid #34503f; box-shadow: 0 .3em .9em rgba(0,0,0,.55);
  transition: box-shadow .25s, border-color .25s, opacity .25s;
}
.pa-avatar {
  flex: none; width: 3.1em; height: 3.1em; border-radius: 50%; display: grid; place-items: center;
  font-weight: 800; font-size: 1.15em; color: #fff;
  background: linear-gradient(145deg, hsl(var(--hue) 55% 48%), hsl(var(--hue) 55% 28%));
  box-shadow: inset 0 0 0 .12em rgba(255,255,255,.25);
}
.pa-plate-text { min-width: 0; }
.pa-name { font-weight: 700; font-size: 1.2em; display: flex; align-items: center; gap: .45em; white-space: nowrap; }
.pa-sub { font-size: .82em; color: #a9bdb1; white-space: nowrap; }
.pa-stack { font-weight: 800; font-size: 1.25em; color: #fff; }
.pa-stack small { font-size: .62em; font-weight: 600; color: #a9bdb1; margin-left: .5em; }
.pa-tags { display: flex; flex-wrap: wrap; gap: .3em; margin-top: .3em; }
.pa-tag {
  display: inline-block; font-size: .78em; font-weight: 800; letter-spacing: .04em; text-transform: uppercase;
  padding: .12em .55em; border-radius: .8em; line-height: 1.35; white-space: nowrap;
}
.pa-tag--blind { background: #3a4a42; color: #dbe7df; }
.pa-tag--win { background: linear-gradient(180deg, #ffd96b, #e0a92a); color: #2a1d00; }
.pa-tag--loss { background: rgba(229,72,77,.2); color: #ffa3a6; }
.pa-tag--turn { background: #2ecc71; color: #052a14; }
.pa-tag--fold { background: #3a4a42; color: #c8d6ce; }
.pa-tag--allin { background: #e5484d; color: #fff; }
.pa-tag--hand { background: rgba(255,255,255,.12); color: #e9f1ec; text-transform: none; letter-spacing: 0; }
.pa-dealer {
  position: absolute; top: -.9em; left: -.7em; width: 2.2em; height: 2.2em; border-radius: 50%; z-index: 2;
  display: grid; place-items: center; background: #fff; color: #111; font-weight: 900; font-size: .95em;
  box-shadow: 0 .15em .4em rgba(0,0,0,.6), inset 0 0 0 .14em #d6d6d6;
}
.pa-seat--acting .pa-plate {
  border-color: var(--accent);
  box-shadow: 0 0 0 .22em rgba(46,204,113,.32), 0 0 1.6em rgba(46,204,113,.55);
  animation: pa-pulse 1.6s ease-in-out infinite;
}
.pa-seat--winner .pa-plate {
  border-color: var(--gold); box-shadow: 0 0 0 .22em rgba(245,197,66,.32), 0 0 1.8em rgba(245,197,66,.6);
}
.pa-seat--folded .pa-plate, .pa-seat--folded .pa-hole { opacity: .5; }
.pa-bubble {
  position: absolute; left: calc(100% + .9em); top: 50%; transform: translateY(-50%); white-space: nowrap;
  background: #f3f6f4; color: #0f1a15; font-weight: 700; font-size: 1.05em; padding: .45em .9em;
  border-radius: .9em; box-shadow: 0 .25em .8em rgba(0,0,0,.5); animation: pa-pop .25s ease-out;
}
.pa-bubble::before {
  content: ""; position: absolute; left: -.45em; top: 50%; width: .9em; height: .9em; background: inherit;
  transform: translateY(-50%) rotate(45deg); border-radius: .15em;
}
/* the top seat's bubble opens to the left so the corner badge never collides with it */
.pa-seat--top .pa-bubble { left: auto; right: calc(100% + .9em); }
.pa-seat--top .pa-bubble::before { left: auto; right: -.45em; }

/* centre: pot, board, street */
.pa-center {
  position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); z-index: 2;
  display: flex; flex-direction: column; align-items: center; gap: .7em;
}
.pa-pot {
  display: inline-flex; align-items: center; gap: .55em; padding: .3em 1.1em; border-radius: 2em;
  background: rgba(0,0,0,.4); border: .09em solid rgba(245,197,66,.55); color: #f6dc8c;
  font-size: 1.15em; letter-spacing: .03em;
}
.pa-pot b { color: #fff; font-size: 1.15em; }
.pa-board { display: flex; gap: .55em; }
.pa-street {
  font-size: .85em; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; color: rgba(255,255,255,.62);
}
.pa-result {
  max-width: 34em; text-align: center; font-weight: 700; font-size: 1.05em; padding: .35em 1em;
  border-radius: 1em; background: rgba(0,0,0,.5); border: .09em solid rgba(245,197,66,.6); color: #ffe9a8;
}
.pa-bet {
  position: absolute; left: 50%; transform: translateX(-50%); z-index: 2; display: inline-flex;
  align-items: center; gap: .5em; padding: .22em .9em .22em .35em; border-radius: 2em;
  background: rgba(0,0,0,.42); font-size: 1.1em; font-weight: 700; color: #fff;
}
.pa-bet--top { top: 9em; }
.pa-bet--bottom { bottom: 9em; }
.pa-chip {
  display: inline-block; width: 1.5em; height: 1.5em; border-radius: 50%; flex: none;
  background: repeating-conic-gradient(#fff 0 12deg, #d23c3c 12deg 36deg);
  box-shadow: inset 0 0 0 .3em #d23c3c, inset 0 0 0 .38em rgba(255,255,255,.85), 0 .1em .25em rgba(0,0,0,.6);
}

@keyframes pa-pulse { 50% { box-shadow: 0 0 0 .3em rgba(46,204,113,.18), 0 0 2.2em rgba(46,204,113,.75); } }
@keyframes pa-pop { from { opacity: 0; } }
@media (prefers-reduced-motion: reduce) {
  .pa-seat--acting .pa-plate, .pa-bubble { animation: none; }
}

/* narrow containers (phones): larger relative type, taller table, stacked seat contents */
@container (max-width: 520px) {
  .pa-table { font-size: 2.35cqw; --card-fs: 3.2cqw; aspect-ratio: 1 / 1.2; }
  .pa-felt { inset: 3.4em .4em; }
  .pa-seat { flex-direction: column; gap: .5em; }
  .pa-hole { min-width: 0; justify-content: center; }
  .pa-plate { min-width: 14em; }
  .pa-bet--top { top: 13.6em; }
  .pa-bet--bottom { bottom: 13.6em; }
  .pa-bubble, .pa-seat--top .pa-bubble { left: calc(100% + .5em); right: auto; top: 1em; transform: none; }
  .pa-bubble::before, .pa-seat--top .pa-bubble::before { display: none; }
  .pa-info, .pa-badge { top: .4em; font-size: .85em; }
}
"""
