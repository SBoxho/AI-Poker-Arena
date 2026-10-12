"""Pure table renderer: data object -> HTML."""

from __future__ import annotations

from poker_arena.ui.table import (
    SeatView,
    TableView,
    accessible_summary,
    format_bb,
    format_chips,
    render_table_html,
    table_document,
)


def _view(**overrides) -> TableView:
    top = SeatView(
        "Maniac", "Seat 2 · Rule-based", 940, bet=60, hole=(None, None), is_dealer=True, blind="SB",
        bubble="Raises to 60",
    )
    bottom = SeatView("You", "Human", 1000, hole=("Ah", "Kd"), blind="BB", is_acting=True)
    base = dict(
        top=top, bottom=bottom, board=("2c", "7d", "Js"), pot=120, street="Flop", big_blind=10,
        info="Hand 3 · Blinds 5/10", badge="Your turn", badge_tone="turn",
    )
    return TableView(**(base | overrides))


def test_formatting_helpers():
    assert format_chips(1234567) == "1,234,567"
    assert format_bb(25, 10) == "2.5" and format_bb(1000, 10) == "100" and format_bb(5, 10) == "0.5"


def test_table_contains_all_visible_elements():
    html = render_table_html(_view())
    for fragment in (
        "Maniac", "You", "Pot <b>120</b>", "Flop", "Raises to 60", "pa-dealer", ">SB<", ">BB<",
        "pa-seat--acting", "To act", "Your turn", "Hand 3", 'data-card="Ah"', 'data-card="Js"', "pa-bet--top",
    ):
        assert fragment in html, fragment
    assert html.count("pa-card--back") == 2  # the hidden opponent hand
    assert html.count("pa-card--slot") == 2  # two undealt community cards
    assert "pa-bet--bottom" not in html  # nothing bet by the bottom seat


def test_names_are_html_escaped():
    seat = SeatView("<script>alert(1)</script>", "x&y", 100)
    html = render_table_html(_view(top=seat))
    assert "<script>" not in html and "&lt;script&gt;" in html and "x&amp;y" in html


def test_winner_folded_and_result_rendering():
    winner = SeatView("A", "", 1200, hole=("Ah", "Ad"), is_winner=True, payoff=200, hand_name="One pair")
    loser = SeatView("B", "", 800, hole=("2c", "3c"), is_folded=True, payoff=-200)
    html = render_table_html(_view(top=winner, bottom=loser, result_text="A wins 200 chips", street="Showdown"))
    for fragment in (
        "pa-seat--winner", "Winner +200", "One pair", "pa-seat--folded", "Folded", "-200",
        "A wins 200 chips", "Showdown", "pa-card--win",
    ):
        assert fragment in html, fragment


def test_all_in_tag_and_missing_optional_parts():
    html = render_table_html(_view(top=SeatView("S", "", 0, is_all_in=True), badge=None))
    assert "All-in" in html and "pa-badge" not in html


def test_standalone_document_embeds_css():
    assert "<style>" not in render_table_html(_view())
    doc = table_document(_view())
    assert doc.count("<style>") == 2 and ".pa-table" in doc and ".pa-card" in doc


def test_accessible_summary_only_uses_visible_information():
    summary = accessible_summary(_view())
    assert "face-down card" in summary
    assert "Ace of hearts" in summary and "Jack of spades" in summary and "Pot 120" in summary
