"""Hand-log formatting."""

from __future__ import annotations

from types import SimpleNamespace

from ui_helpers import agent_match, fold_spot_match, play_folding

from poker_arena.engine.observation import Street
from poker_arena.ui.history import (
    bubble_text,
    build_hand_log,
    describe_action,
    log_entry_text,
    log_from_record,
    previous_hands_rows,
    render_log_html,
)
from poker_arena.ui.views import showdown_hand_names

MENTIONS = ["Seat 1 (EBM)", "Seat 2 (Random)"]


def _action(kind: str, chips: int = 0, total: int = 0, all_in: bool = False):
    return SimpleNamespace(kind=kind, chips=chips, street_total=total, all_in=all_in)


def test_action_descriptions():
    assert describe_action(_action("fold")) == "folds"
    assert describe_action(_action("check")) == "checks"
    assert describe_action(_action("call", chips=1500, total=1500)) == "calls 1,500"
    assert describe_action(_action("bet", chips=40, total=40)) == "bets 40"
    assert describe_action(_action("raise", chips=60, total=60)) == "raises to 60"
    assert describe_action(_action("raise", chips=990, total=1000, all_in=True)) == "raises to 1,000 (all-in)"
    assert describe_action(_action("fold", all_in=True)) == "folds"
    assert bubble_text(_action("raise", total=60)) == "Raises to 60"


def test_log_of_a_finished_showdown_hand():
    match = agent_match(4, ("call_station", "call_station"))
    record = match.play_out()
    log = log_from_record(record, MENTIONS, hand_names=showdown_hand_names(record.result))
    texts = [log_entry_text(e) for e in log]
    assert log[0].kind == "info" and "small blind (5)" in log[0].text and "big blind (10)" in log[0].text
    headers = [e.text for e in log if e.kind == "street"]
    assert headers == ["Preflop", "Flop", "Turn", "River"]
    flop = next(e for e in log if e.text == "Flop")
    assert len(flop.cards) == 3 and len(next(e for e in log if e.text == "River").cards) == 1
    assert sum(e.kind == "action" for e in log) == len(record.events)
    assert any("shows" in t for t in texts) and any("wins" in t or "split" in t for t in texts)
    assert log[-1].kind == "result"


def test_fold_result_mentions_the_fold_and_hides_cards():
    match = fold_spot_match()
    play_folding(match)
    record = match.records[-1]
    assert not record.result.showdown
    log = log_from_record(record, ["You", "Opponent"])
    assert log[-1].kind == "result" and "(opponent folded)" in log[-1].text
    assert all(e.cards == () for e in log if e.kind == "result")  # nobody shows a hand


def test_all_in_runout_lists_the_dealt_streets():
    match = agent_match(7, ("maniac", "maniac"))
    for _ in range(60):  # find a hand that ended all-in before the river
        record = match.play_out()
        last_betting_street = record.events[-1].street
        if record.result.showdown and last_betting_street < Street.RIVER:
            log = log_from_record(record, MENTIONS)
            headers = [e.text for e in log if e.kind == "street"]
            assert headers[-1] == "River" and "Flop" in headers
            return
        match.start_hand()
    raise AssertionError("no all-in run-out hand found")


def test_in_progress_log_has_no_result():
    match = agent_match(2)
    match.step()
    engine = match.session.engine
    log = build_hand_log(
        engine.events, MENTIONS, config=match.config, button_seat=engine.button_seat,
        start_stacks=engine.start_stacks, board=engine.board,
    )
    assert all(e.kind != "result" for e in log) and log[-1].kind == "action"


def test_render_log_html_escapes_and_orders_newest_first():
    match = agent_match(2)
    match.play_out()
    record = match.records[-1]
    log = log_from_record(record, ["<b>x</b>", "y"])
    html = render_log_html(log)
    assert "&lt;b&gt;x&lt;/b&gt;" in html and "<b>x</b>" not in html
    assert html.index("pa-log-result") < html.index("pa-log-info")  # newest first
    chronological = render_log_html(log, newest_first=False)
    assert chronological.index("pa-log-info") < chronological.index("pa-log-result")


def test_previous_hands_rows():
    match = agent_match(3, ("random", "maniac"))
    for _ in range(5):
        match.play_out()
        match.start_hand()
    rows = previous_hands_rows(match.records, MENTIONS, 10, limit=3)
    assert [r["Hand"] for r in rows] == [5, 4, 3]  # newest first, limited
    assert all(set(r) >= {"Result", "Ended", "Button"} for r in rows)
    assert rows[0][f"{MENTIONS[0]} (BB)"] == -rows[0][f"{MENTIONS[1]} (BB)"]
