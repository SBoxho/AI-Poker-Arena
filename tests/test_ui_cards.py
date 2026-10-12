"""Card HTML rendering."""

from __future__ import annotations

import pytest

from poker_arena.cards import ALL_CARDS
from poker_arena.ui.cards import (
    SUIT_GLYPHS,
    card_name,
    card_text,
    card_text_html,
    render_card,
    render_card_back,
    render_card_slot,
    render_hand,
)


def test_names_and_text():
    assert card_name("Ah") == "Ace of hearts"
    assert card_name("Td") == "Ten of diamonds"
    assert card_text("Td") == "10♦"
    assert card_text("2c") == "2♣"


@pytest.mark.parametrize("card", ALL_CARDS)
def test_every_card_renders_with_label_and_glyph(card: str):
    html = render_card(card)
    assert f'data-card="{card}"' in html
    assert f'aria-label="{card_name(card)}"' in html
    assert SUIT_GLYPHS[card[1]] in html


def test_two_colour_deck_and_ten_label():
    assert "pa-suit-h" in render_card("Ah") and "pa-suit-d" in render_card("2d")
    assert "pa-suit-s" in render_card("As") and "pa-suit-c" in render_card("2c")
    assert "<b>10</b>" in render_card("Ts")


def test_card_back_reveals_nothing():
    back = render_card_back()
    assert "data-card" not in back and "Face-down" in back
    assert not any(glyph in back for glyph in SUIT_GLYPHS.values())
    assert render_card_slot().count("pa-card--slot") == 1


def test_render_hand_mixes_faces_and_backs():
    html = render_hand(("Ah", None))
    assert html.count("pa-card--back") == 1 and 'data-card="Ah"' in html


def test_inline_text_is_coloured_by_suit():
    assert "pa-red" in card_text_html("Qh") and "pa-black" in card_text_html("Qs")
