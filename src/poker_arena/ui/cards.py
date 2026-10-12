"""Playing-card rendering as plain HTML/CSS (no Streamlit dependency, unit-testable).

Card sizes are driven by the CSS custom property ``--card-fs`` (the font size the card is drawn
at): everything inside a card is expressed in ``em`` so a card scales with a single number.

A face-down card deliberately carries no information about the card it stands for - the
anti-cheating tests rely on :func:`render_card_back` being independent of the hidden card.
"""

from __future__ import annotations

from html import escape

from poker_arena.cards import RANKS, card_to_str

SUIT_GLYPHS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
SUIT_NAMES = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}
RANK_LABELS = {r: r for r in RANKS} | {"T": "10"}
RANK_NAMES = {
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
    "T": "Ten",
    "J": "Jack",
    "Q": "Queen",
    "K": "King",
    "A": "Ace",
}

# Standard two-colour deck; the suit glyph (not only the colour) always identifies the suit.
RED_SUITS = frozenset("dh")


def card_name(card: str) -> str:
    """``"Ah"`` -> ``"Ace of hearts"`` (used for accessible labels)."""
    card = card_to_str(card)
    return f"{RANK_NAMES[card[0]]} of {SUIT_NAMES[card[1]]}"


def card_text(card: str) -> str:
    """Compact plain-text form with a suit glyph: ``"Ah"`` -> ``"A♥"``, ``"Td"`` -> ``"10♦"``."""
    card = card_to_str(card)
    return f"{RANK_LABELS[card[0]]}{SUIT_GLYPHS[card[1]]}"


def cards_text(cards: tuple[str, ...] | list[str]) -> str:
    return " ".join(card_text(c) for c in cards)


def card_text_html(card: str) -> str:
    """Inline coloured ``A♥`` snippet for logs and tables."""
    card = card_to_str(card)
    colour = "pa-red" if card[1] in RED_SUITS else "pa-black"
    return f'<span class="pa-ctext {colour}">{escape(card_text(card))}</span>'


def render_card(card: str, *, extra_class: str = "") -> str:
    """A face-up card."""
    card = card_to_str(card)
    rank, suit = card[0], card[1]
    classes = f"pa-card pa-suit-{suit}" + (f" {extra_class}" if extra_class else "")
    glyph = SUIT_GLYPHS[suit]
    return (
        f'<span class="{classes}" role="img" aria-label="{escape(card_name(card))}" data-card="{card}">'
        f'<span class="pa-corner"><b>{RANK_LABELS[rank]}</b><i>{glyph}</i></span>'
        f'<span class="pa-pip">{glyph}</span>'
        "</span>"
    )


def render_card_back(*, extra_class: str = "") -> str:
    """A face-down card (identical for every hidden card)."""
    classes = "pa-card pa-card--back" + (f" {extra_class}" if extra_class else "")
    return f'<span class="{classes}" role="img" aria-label="Face-down card"></span>'


def render_card_slot() -> str:
    """Empty placeholder for a community card that has not been dealt yet."""
    return '<span class="pa-card pa-card--slot" aria-hidden="true"></span>'


def render_hand(cards: tuple[str | None, ...], *, extra_class: str = "") -> str:
    """Render hole cards; ``None`` entries are drawn face down."""
    return "".join(
        render_card(c, extra_class=extra_class) if c is not None else render_card_back(extra_class=extra_class)
        for c in cards
    )


CARD_CSS = """
.pa-card {
  --suit: #17171c;
  position: relative; display: inline-block; flex: none;
  width: 3.1em; height: 4.4em; font-size: var(--card-fs, 16px); line-height: 1;
  border-radius: .38em; color: var(--suit);
  background: linear-gradient(165deg, #ffffff 0%, #eef1f3 100%);
  box-shadow: 0 .12em .35em rgba(0,0,0,.55), inset 0 0 0 .06em rgba(0,0,0,.14);
  font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  user-select: none;
}
.pa-card.pa-suit-h, .pa-card.pa-suit-d { --suit: #c62828; }
.pa-card .pa-corner {
  position: absolute; top: .26em; left: .3em; display: flex; flex-direction: column; align-items: center;
  line-height: 1; font-style: normal;
}
.pa-card .pa-corner b { font-size: 1.12em; font-weight: 800; letter-spacing: -.04em; }
.pa-card .pa-corner i { font-size: .85em; font-style: normal; margin-top: .02em; }
.pa-card .pa-pip {
  position: absolute; right: .22em; bottom: .12em; font-size: 2.15em; line-height: 1;
}
.pa-card--back {
  background:
    radial-gradient(circle at 50% 50%, rgba(255,255,255,.18) 0 .5em, transparent .55em),
    repeating-linear-gradient(45deg, #23479e 0 .3em, #1a3680 .3em .6em);
  box-shadow: 0 .12em .35em rgba(0,0,0,.55), inset 0 0 0 .17em #f4f6f8, inset 0 0 0 .22em rgba(0,0,0,.25);
}
.pa-card--slot {
  background: rgba(255,255,255,.05); box-shadow: inset 0 0 0 .1em rgba(255,255,255,.18);
  border: .1em dashed rgba(255,255,255,.18); box-sizing: border-box;
}
.pa-card--win { box-shadow: 0 0 0 .14em #f5c542, 0 .12em .6em rgba(245,197,66,.55); }
.pa-ctext { font-weight: 700; white-space: nowrap; }
.pa-ctext.pa-red { color: #ff6b6b; }
.pa-ctext.pa-black { color: #e9f1ec; }
"""
