"""End-to-end Streamlit ``AppTest`` runs of the pages (baseline agents only)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import human_match, play_by_calling

from poker_arena.engine.actions import Action
from poker_arena.ui import human_vs_ai
from poker_arena.ui.match import Match
from poker_arena.ui.roster import discover_roster
from poker_arena.ui.state import SessionNamespace
from poker_arena.ui.widgets import cached_roster

APP_DIR = Path(__file__).resolve().parents[1] / "app"
TIMEOUT_SECONDS = 90
MAX_CLICKS = 80  # a heads-up hand never needs more human clicks than this

A_PAGE, C_PAGE = "pages/1_AI_vs_AI.py", "pages/3_Human_vs_AI.py"
A = "pa.ava."
C = "pa.hva."


def open_page(name: str) -> AppTest:
    return AppTest.from_file(str(APP_DIR / name), default_timeout=TIMEOUT_SECONDS).run()


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def page_html(at: AppTest) -> str:
    return "".join(str(e.value) for e in at.get("html"))


def face_down_cards(html: str) -> int:
    return html.count('aria-label="Face-down card"')


def mentions(html: str, card: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){card}(?![A-Za-z0-9])", html) is not None


# ----------------------------------------------------------------------------------- Home
def test_home_lists_every_agent_and_the_research_principles():
    at = open_page("Home.py")
    assert_clean(at)
    html = page_html(at)
    assert "AI Poker Arena" in html and "Imitation accuracy is not poker skill" in html
    for entry in discover_roster().values():
        assert entry.label in html
    assert "Unavailable" in html or all(e.available for e in discover_roster().values())
    assert len(at.button) >= 1  # re-scan button


# ----------------------------------------------------------------------------------- Mode A
def start_mode_a(kinds: tuple[str, str] = ("call_station", "maniac"), seed: int = 42, at: AppTest | None = None) -> AppTest:
    at = at or open_page(A_PAGE)
    at.selectbox(key=A + "kind0").select(kinds[0])
    at.selectbox(key=A + "kind1").select(kinds[1])
    at.number_input(key=A + "seed").set_value(seed)
    at.button(key=A + "start").click().run()
    assert_clean(at)
    return at


def test_mode_a_shows_a_lobby_before_the_game_starts():
    at = open_page(A_PAGE)
    assert_clean(at)
    assert A + "match" not in at.session_state
    assert any("Start game" in i.value for i in at.info)
    assert at.sidebar.selectbox  # agent pickers live in the sidebar


def test_mode_a_start_step_and_next_hand():
    at = start_mode_a()
    match: Match = at.session_state[A + "match"]
    assert match.hand_number == 0 and match.hand_active
    assert at.session_state[A + "playing"] is True  # autoplay starts with the game

    at.button(key=A + "next_action").click().run()
    assert_clean(at)
    assert at.session_state[A + "playing"] is False  # manual stepping pauses autoplay
    decisions = match.decision_index
    at.button(key=A + "next_action").click().run()
    assert match.decision_index == decisions + 1 or match.hand_complete  # exactly one decision

    at.button(key=A + "next_hand").click().run()
    assert_clean(at)
    assert match.hand_complete and match.hands_completed == 1
    assert "Hand over" in page_html(at)

    at.button(key=A + "next_action").click().run()  # a finished hand: next tick deals the next one
    assert match.hand_number == 1 and match.decision_index == 0 and match.hand_active
    assert at.selectbox(key=A + "inspect") if match.steps else True


def test_mode_a_decision_panel_and_features_after_a_step():
    at = start_mode_a(("random", "maniac"))
    at.button(key=A + "next_action").click().run()
    assert_clean(at)
    html = page_html(at)
    assert "Decision panel" in html and "pa-explain" in html  # baselines explain themselves
    assert at.selectbox(key=A + "inspect").value == -1
    assert any("Observation features" in e.label for e in at.expander)


def test_mode_a_spectator_toggle_hides_and_reveals_hole_cards():
    at = start_mode_a(("call_station", "call_station"))
    match: Match = at.session_state[A + "match"]
    hole = match.session.engine.deal.cards[:4]
    assert all(f'data-card="{c}"' in page_html(at) for c in hole)  # spectator view is on by default
    assert "Spectator view" in page_html(at)
    at.toggle(key=A + "reveal").set_value(False).run()
    assert_clean(at)
    html = page_html(at)
    assert face_down_cards(html) == 4 and not any(mentions(html, c) for c in hole)


def test_mode_a_autoplay_ticks_are_paced_and_pausable():
    at = start_mode_a()
    match: Match = at.session_state[A + "match"]
    after_start = match.decision_index + match.hands_completed * 100
    at.run()  # an immediate rerun (e.g. a widget change) must not take another step
    assert match.decision_index + match.hands_completed * 100 == after_start

    pacer = at.session_state[A + "pacer"]
    pacer.last_tick -= 1000.0  # pretend the timer fired
    at.run()
    assert_clean(at)
    assert match.decision_index + match.hands_completed * 100 > after_start

    at.button(key=A + "toggle").click().run()
    assert at.session_state[A + "playing"] is False
    frozen = (match.decision_index, match.hands_completed)
    at.session_state[A + "pacer"].last_tick -= 1000.0
    at.run()
    assert (match.decision_index, match.hands_completed) == frozen


def test_mode_a_reset_restarts_the_same_game():
    at = start_mode_a(seed=7)
    at.button(key=A + "next_hand").click().run()
    first_events = at.session_state[A + "match"].records[0].events
    assert at.session_state[A + "match"].hands_completed == 1
    at.button(key=A + "reset").click().run()
    assert_clean(at)
    fresh: Match = at.session_state[A + "match"]
    assert fresh.hands_completed == 0 and fresh.hand_number == 0 and at.session_state[A + "playing"] is False
    at.button(key=A + "next_hand").click().run()
    assert fresh.records[0].events == first_events  # same seed + agents -> identical hand


def test_mode_a_continuous_play_scoreboard_and_history():
    at = start_mode_a(("random", "maniac"))
    for _ in range(3):
        at.button(key=A + "next_hand").click().run()
        assert_clean(at)
    match: Match = at.session_state[A + "match"]
    assert match.hands_completed == 3 and len(match.profit_series_bb) == 3
    assert any("Previous hands" in e.label for e in at.expander)
    assert at.metric  # profit metrics for both seats
    assert "3 hands played" in " ".join(c.value for c in at.caption)


# ----------------------------------------------------------------------------------- Mode C
def start_mode_c(opponent: str = "call_station", seed: int = 5, *, stack: int | None = None) -> AppTest:
    at = open_page(C_PAGE)
    at.slider(key=C + "think").set_value(0.0)
    at.selectbox(key=C + "opponent").select(opponent)
    at.number_input(key=C + "seed").set_value(seed)
    if stack is not None:
        at.number_input(key=C + "stack").set_value(stack)
    at.button(key=C + "start").click().run()
    assert_clean(at)
    return at


def human_buttons(at: AppTest) -> dict[str, object]:
    """Enabled action buttons keyed by their prefix ("fold", "call", "raise-btn", "quick-allin" ...)."""
    out = {}
    for button in at.button:
        key = button.key or ""
        if key.startswith(C) and not button.disabled:
            out[key.removeprefix(C).rsplit("-", 2)[0]] = button
    return out


def ai_cards(match: Match) -> tuple[str, str]:
    engine = match.session.engine
    return engine.deal.button_hole if engine.button_seat == 1 else engine.deal.bb_hole


def test_mode_c_lobby_then_play():
    at = open_page(C_PAGE)
    assert_clean(at)
    assert C + "match" not in at.session_state and any("Play" in i.value for i in at.info)
    at = start_mode_c()
    match: Match = at.session_state[C + "match"]
    assert match.human_seat == 0 and match.agents[1] is not None and match.hand_active
    assert match.awaiting_human  # the AI acted if it was first, then waits for the human
    assert {"call", "fold", "raise-btn", "quick-min", "quick-allin"} <= set(
        k.removeprefix(C).rsplit("-", 2)[0] for k in (b.key for b in at.button) if k and k.startswith(C)
    )


def test_mode_c_play_to_the_end_without_leaking_the_ai_hand():
    at = start_mode_c("call_station", seed=5)
    match: Match = at.session_state[C + "match"]
    hidden = ai_cards(match)
    clicks = 0
    while match.hand_active:
        html = page_html(at)
        assert not any(mentions(html, c) for c in hidden), "AI hole cards leaked into the page"
        assert "Your turn" in html
        human_buttons(at)["call"].click().run()
        assert_clean(at)
        clicks += 1
        assert clicks < MAX_CLICKS
    html = page_html(at)
    assert "pa-banner" in html and "Showdown" in html
    assert all(f'data-card="{c}"' in html for c in hidden)  # revealed only because it went to showdown
    assert at.session_state[C + "match"].hands_completed == 1
    assert any(b.key == f"{C}next-0" for b in at.button)

    at.button(key=f"{C}next-0").click().run()  # next hand
    assert_clean(at)
    assert match.hand_number == 1 and match.hand_active


def test_mode_c_actions_use_legal_amounts_and_quick_buttons():
    at = start_mode_c("call_station", seed=5)
    match: Match = at.session_state[C + "match"]
    legal = match.session.observation(0).legal
    slider = at.slider(key=next(b.key for b in at.slider if b.key and b.key.startswith(C + "amount-")))
    assert slider.min == legal.min_raise_to and slider.max == legal.max_raise_to
    quick = [b for b in at.button if (b.key or "").startswith(C + "quick-pot")][0]
    quick.click().run()
    assert_clean(at)
    slider = at.slider(key=next(b.key for b in at.slider if b.key and b.key.startswith(C + "amount-")))
    assert legal.min_raise_to <= slider.value <= legal.max_raise_to
    human_buttons(at)["raise-btn"].click().run()
    assert_clean(at)
    assert any(s.event.kind in ("raise", "bet") and s.event.seat == 0 for s in match.steps)


def test_mode_c_stale_clicks_are_ignored():
    """A double click delivers the same callback twice: only the first may change the game."""
    match = human_match(5)
    ns = SessionNamespace(human_vs_ai.NAMESPACE, {})
    ns.set("match", match)
    index = match.decision_index
    human_vs_ai._on_action(ns, index, Action.check_call())
    after_first = (match.hand_number, match.decision_index, match.hand_active)
    assert after_first[1] > index
    human_vs_ai._on_action(ns, index, Action.check_call())
    assert (match.hand_number, match.decision_index, match.hand_active) == after_first

    play_by_calling(match)
    human_vs_ai._on_next_hand(ns, match.hand_number, rebuy=False)
    dealt = (match.hand_number, match.decision_index)
    assert match.hand_active
    human_vs_ai._on_next_hand(ns, dealt[0] - 1, rebuy=False)  # stale: refers to the finished hand
    assert (match.hand_number, match.decision_index) == dealt
    assert "error" not in ns


def test_mode_c_illegal_action_is_reported_not_raised():
    match = human_match(5)
    ns = SessionNamespace(human_vs_ai.NAMESPACE, {})
    ns.set("match", match)
    human_vs_ai._on_action(ns, match.decision_index, Action.raise_to(1))
    assert "raise" in ns.get("error").lower() and match.awaiting_human


def test_mode_c_fold_ends_the_hand_and_keeps_the_ai_hand_private():
    for seed in range(1, 40):
        at = start_mode_c("call_station", seed=seed)
        match: Match = at.session_state[C + "match"]
        if "fold" in human_buttons(at):
            break
    else:
        pytest.fail("no seed with a fold spot")
    human_buttons(at)["fold"].click().run()
    assert_clean(at)
    assert match.hand_complete and not match.records[-1].result.showdown
    html = page_html(at)
    assert "pa-banner" in html and "folded" in html
    assert not any(mentions(html, c) for c in ai_cards(match))


def test_mode_c_review_of_the_ai_only_after_the_hand():
    at = start_mode_c("random", seed=3)
    match: Match = at.session_state[C + "match"]
    assert not any(t.label == "Why did the AI do that?" for t in at.toggle)  # hidden mid-hand
    while match.hand_active:
        buttons = human_buttons(at)
        buttons.get("call", buttons.get("fold")).click().run()
    review = [t for t in at.toggle if t.label == "Why did the AI do that?"]
    if match.steps and any(s.event.seat == 1 for s in match.steps):
        assert review
        review[0].set_value(True).run()
        assert_clean(at)
        assert at.selectbox(key=f"{C}why-pick-0")
        assert "Opponent" in page_html(at)


def test_mode_c_bust_requires_a_rebuy():
    at = start_mode_c("maniac", seed=1, stack=20)  # 2 big blinds: one all-in decides the game
    match: Match = at.session_state[C + "match"]
    for _ in range(MAX_CLICKS):
        if match.needs_rebuy:
            break
        buttons = human_buttons(at)
        if "quick-allin" in buttons:
            buttons["quick-allin"].click().run()
            buttons = human_buttons(at)
            buttons["raise-btn"].click().run()
        elif "call" in buttons:
            buttons["call"].click().run()
        else:  # hand over: deal the next one
            next(b for b in at.button if (b.key or "").startswith(C + "next-")).click().run()
        assert_clean(at)
    else:
        pytest.fail("nobody went bust")
    labels = [b.label for b in at.button]
    assert "Rebuy and play next hand" in labels and "Play next hand" not in labels
    next(b for b in at.button if b.label == "Rebuy and play next hand").click().run()
    assert_clean(at)
    assert not match.needs_rebuy and match.hand_active and match.session.engine.start_stacks == (20, 20)


# ------------------------------------------------------------------------------- isolation
def test_two_sessions_do_not_share_state_or_agents():
    first = start_mode_a(("random", "maniac"), seed=1)
    second = open_page(A_PAGE)
    assert A + "match" not in second.session_state  # a fresh browser session starts empty
    start_mode_a(("call_station", "call_station"), seed=2, at=second)

    m1: Match = first.session_state[A + "match"]
    m2: Match = second.session_state[A + "match"]
    assert m1 is not m2 and m1.settings.seed == 1 and m2.settings.seed == 2
    assert not {id(a) for a in m1.agents} & {id(a) for a in m2.agents}
    templates = {id(e.template) for e in cached_roster().values() if e.template is not None}
    assert not templates & ({id(a) for a in m1.agents} | {id(a) for a in m2.agents})  # copies, not templates

    before = (m1.decision_index, m1.hands_completed)
    second.button(key=A + "next_hand").click().run()
    assert (m1.decision_index, m1.hands_completed) == before  # acting in one session never moves the other
    assert m2.hands_completed >= 1


def test_mode_a_and_c_state_are_separate_namespaces():
    at_a = start_mode_a()
    at_c = start_mode_c()
    assert A + "match" in at_a.session_state and C + "match" not in at_a.session_state
    assert C + "match" in at_c.session_state and A + "match" not in at_c.session_state
