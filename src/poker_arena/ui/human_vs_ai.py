"""Mode C - play heads-up No-Limit Hold'em against an AI agent.

Information boundary: while a hand is active the table is rendered from
``session.observation(human_seat)`` - the *restricted* view - so the AI's hole cards are never part
of the HTML. They are only drawn if the engine reports them as shown at showdown.

All state lives in ``st.session_state`` (namespace ``hva``). Every action button carries the index
of the decision it belongs to, so stale clicks (double clicks) are ignored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from html import escape

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from poker_arena.config import derive_seed
from poker_arena.engine.actions import Action, IllegalActionError
from poker_arena.engine.observation import Observation
from poker_arena.ui.betting import QuickRaise, call_label, default_raise, quick_raises, raise_label
from poker_arena.ui.explain import render_step_detail
from poker_arena.ui.history import STREET_NAMES, build_hand_log, previous_hands_rows, render_log_html
from poker_arena.ui.match import Match, MatchSettings
from poker_arena.ui.roster import RosterEntry, spawn_agent
from poker_arena.ui.state import SessionNamespace
from poker_arena.ui.table import render_table_html
from poker_arena.ui.theme import configure_page
from poker_arena.ui.views import SeatIdentity, player_table, result_headline, showdown_hand_names
from poker_arena.ui.widgets import (
    GameSettings,
    agent_card_html,
    agent_picker,
    cached_roster,
    game_settings_widgets,
    unavailable_agents_note,
)

NAMESPACE = "hva"
HUMAN_SEAT = 0
AI_SEAT = 1
PREFERRED_OPPONENTS = ("ebm", "teacher", "random")
DEFAULT_THINK_SECONDS, MAX_THINK_SECONDS = 0.8, 3.0
HISTORY_ROWS_SHOWN = 30
GAME_KEYS = ("match", "request", "error")


@dataclass(frozen=True, slots=True)
class StartRequest:
    opponent_kind: str
    settings: GameSettings


def build_match(roster: dict[str, RosterEntry], request: StartRequest) -> Match:
    """New match (human in seat 0, a private AI copy in seat 1) with the first hand dealt."""
    seed = request.settings.seed
    ai = spawn_agent(roster[request.opponent_kind], derive_seed("ui", seed, "agent", AI_SEAT))
    settings = MatchSettings(
        config=request.settings.config,
        seed=seed,
        reset_stacks_each_hand=request.settings.reset_stacks_each_hand,
        auto_rebuy=False,  # busts are handled explicitly with a "Rebuy" button
    )
    match = Match(settings, [None, ai])
    match.start_hand()
    return match


def seat_identities(match: Match) -> tuple[SeatIdentity, SeatIdentity]:
    ai = match.agents[AI_SEAT]
    if ai is None:
        raise ValueError("Seat 2 must be an AI agent")
    you = SeatIdentity("You", "You", "Human")
    opponent = SeatIdentity(f"Opponent ({ai.name})", ai.name, f"Opponent · {ai.architecture}")
    return (you, opponent) if HUMAN_SEAT == 0 else (opponent, you)


# ------------------------------------------------------------------------------ callbacks
def _on_action(ns: SessionNamespace, decision_index: int, action: Action) -> None:
    match: Match = ns.get("match")
    try:
        match.submit(action, decision_index=decision_index)
    except IllegalActionError as exc:
        ns.set("error", str(exc))


def _on_set_raise(raise_key: str, amount: int) -> None:
    st.session_state[raise_key] = amount


def _on_next_hand(ns: SessionNamespace, hand_number: int, *, rebuy: bool) -> None:
    match: Match = ns.get("match")
    if match.hand_number != hand_number or not match.hand_complete:
        return  # stale click: the next hand was already dealt
    if rebuy:
        match.rebuy()
    match.start_hand()
    ns.pop("error")


# ------------------------------------------------------------------------------ sidebar
def _sidebar_setup(ns: SessionNamespace, roster: dict[str, RosterEntry], *, running: bool) -> StartRequest | None:
    with st.sidebar:
        st.header("Game setup")
        kind = agent_picker("Opponent", roster, ns.key("opponent"), _preferred(roster))
        unavailable_agents_note(roster, ns.key("rescan"))
        settings, error = game_settings_widgets(
            ns,
            default_seed=None,
            reset_help="Off: stacks carry over between hands and a bust ends the session until you rebuy.",
        )
        if error:
            st.error(error)
        st.slider(
            "Opponent thinking time (s)",
            0.0,
            MAX_THINK_SECONDS,
            DEFAULT_THINK_SECONDS,
            0.1,
            key=ns.key("think"),
            help="Pause before the AI acts so you can follow the action.",
        )
        pressed = st.button(
            "New game" if running else "Play",
            type="primary",
            width="stretch",
            disabled=settings is None,
            key=ns.key("start"),
        )
    return StartRequest(kind, settings) if pressed and settings is not None else None


def _preferred(roster: dict[str, RosterEntry]) -> str:
    available = [k for k, e in roster.items() if e.available]
    return next((k for k in PREFERRED_OPPONENTS if k in available), available[0])


# --------------------------------------------------------------------------------- views
def _status_html(match: Match, obs: Observation) -> str:
    cfg = match.config
    position = "You are on the button (small blind)" if obs.is_button else "You are in the big blind"
    pills = [
        f'<span class="pa-pill">Hand <b>{match.hand_number + 1}</b></span>',
        f'<span class="pa-pill">Blinds <b>{cfg.small_blind}/{cfg.big_blind}</b></span>',
        f'<span class="pa-pill">{escape(position)}</span>',
    ]
    return f'<div class="pa-status">{"".join(pills)}</div>'


def _turn_badge(match: Match) -> tuple[str | None, str]:
    if match.awaiting_human:
        return "Your turn", "turn"
    if match.awaiting_agent:
        return "Opponent is thinking...", "wait"
    return None, "info"


def _render_action_panel(ns: SessionNamespace, match: Match, obs: Observation) -> None:
    """Fold / check-call / raise controls, all built from ``obs.legal``."""
    legal = obs.legal
    interactive = match.awaiting_human
    index = match.decision_index
    tag = f"{match.hand_number}-{index}"
    raise_key = ns.key(f"amount-{tag}")
    ns.drop_prefix("amount-", keep=f"amount-{tag}")

    with st.container(border=True):
        st.html(f'<div class="pa-panel-title">{"Your action" if interactive else "Waiting for the opponent"}</div>')
        top = st.columns([1, 1.4, 2], vertical_alignment="center")
        slider_box = st.container()
        quick_cols = st.columns(4)

        can_raise = legal.can_raise and interactive
        has_range = can_raise and legal.min_raise_to < legal.max_raise_to
        amount = legal.max_raise_to
        if has_range:
            st.session_state.setdefault(raise_key, default_raise(obs))
            with slider_box:
                amount = st.slider(
                    "Bet / raise to (chips)",
                    legal.min_raise_to,
                    legal.max_raise_to,
                    key=raise_key,
                    help="Total chips you will have bet on this street after raising.",
                )
        elif can_raise:
            slider_box.caption(f"Only one raise size is possible: {amount:,} chips.")

        top[0].button(
            "Fold",
            disabled=not (interactive and legal.can_fold),
            on_click=_on_action,
            args=(ns, index, Action.fold()),
            width="stretch",
            key=ns.key(f"fold-{tag}"),
        )
        top[1].button(
            call_label(obs),
            type="primary",
            disabled=not interactive,
            on_click=_on_action,
            args=(ns, index, Action.check_call()),
            width="stretch",
            key=ns.key(f"call-{tag}"),
        )
        top[2].button(
            raise_label(obs, amount) if legal.can_raise else "Raise",
            disabled=not can_raise,
            on_click=_on_action,
            args=(ns, index, Action.raise_to(amount)),
            width="stretch",
            key=ns.key(f"raise-btn-{tag}"),
        )
        for col, quick in zip(quick_cols, quick_raises(obs)):
            _quick_button(col, quick, raise_key, tag, ns, enabled=quick.enabled and has_range)


def _quick_button(
    col: DeltaGenerator, quick: QuickRaise, raise_key: str, tag: str, ns: SessionNamespace, *, enabled: bool
) -> None:
    col.button(
        quick.label,
        disabled=not enabled,
        on_click=_on_set_raise,
        args=(raise_key, quick.amount),
        width="stretch",
        key=ns.key(f"quick-{quick.key}-{tag}"),
    )


def _render_outcome(ns: SessionNamespace, match: Match, identities: tuple[SeatIdentity, SeatIdentity]) -> None:
    record = match.records[-1]
    result = record.result
    names = showdown_hand_names(result) if result.showdown else {}
    net = result.payoffs[HUMAN_SEAT]
    tone = "win" if net > 0 else "loss" if net < 0 else "split"
    headline = result_headline(result, [i.mention for i in identities], names, you=HUMAN_SEAT)
    st.html(
        f'<div class="pa-banner pa-banner--{tone}"><b>{escape(headline)}</b>'
        f'<span>{net / match.config.big_blind:+.1f} BB for you</span></div>'
    )
    if match.needs_rebuy:
        broke = "You are out of chips." if match.stacks[HUMAN_SEAT] <= 0 else "The opponent is out of chips."
        st.warning(f"{broke} Rebuy to give both players a fresh {match.config.starting_stack:,} chips.")
    label = "Rebuy and play next hand" if match.needs_rebuy else "Play next hand"
    st.button(
        label,
        type="primary",
        on_click=_on_next_hand,
        args=(ns, match.hand_number),
        kwargs={"rebuy": match.needs_rebuy},
        key=ns.key(f"next-{match.hand_number}"),
    )
    _render_ai_review(ns, match, identities, showdown=result.showdown)


def _render_ai_review(
    ns: SessionNamespace, match: Match, identities: tuple[SeatIdentity, SeatIdentity], *, showdown: bool
) -> None:
    """Post-hand explanations of the AI's decisions (only ever called once the hand is over)."""
    ai_steps = [s for s in match.steps if s.event.seat == AI_SEAT]
    if not ai_steps or not st.toggle(
        "Why did the AI do that?",
        key=ns.key("why"),
        help="Explanations of the opponent's decisions - shown only after the hand is over.",
    ):
        return
    choice = st.selectbox(
        "AI decision",
        range(len(ai_steps)),
        format_func=lambda i: f"{STREET_NAMES[ai_steps[i].event.street]}: {ai_steps[i].event.kind}",
        key=ns.key(f"why-pick-{match.hand_number}"),
    )
    render_step_detail(
        ai_steps[choice],
        identities[AI_SEAT].mention,
        key=f"{ns.key('why-detail')}-{match.hand_number}-{choice}",
        include_hole=showdown,  # a folded AI hand stays private
    )


def _render_stats(match: Match) -> None:
    st.html('<div class="pa-panel-title">Your results</div>')
    first, second = st.columns(2)
    rate = match.bb_per_100(HUMAN_SEAT)
    first.metric(
        "Net result",
        f"{match.profit_bb(HUMAN_SEAT):+.1f} BB",
        delta=None if rate is None else f"{rate:+.0f} BB/100",
    )
    second.metric("Hands played", match.hands_completed)
    won = sum(r.result.payoffs[HUMAN_SEAT] > 0 for r in match.records)
    lost = sum(r.result.payoffs[HUMAN_SEAT] < 0 for r in match.records)
    st.caption(
        f"Won {won} / lost {lost} of {match.hands_completed} hands. Poker is high-variance: "
        "a few dozen hands say very little about skill."
    )


def _render_history(match: Match, identities: tuple[SeatIdentity, SeatIdentity], obs: Observation) -> None:
    st.html('<div class="pa-panel-title">Hand history</div>')
    engine = match.session.engine  # type: ignore[union-attr]
    result = match.records[-1].result if match.hand_complete else None
    entries = build_hand_log(
        engine.events,
        [i.mention for i in identities],
        config=match.config,
        button_seat=engine.button_seat,
        start_stacks=engine.start_stacks,  # type: ignore[arg-type]
        board=obs.board,
        result=result,
        hand_names=showdown_hand_names(result) if result is not None and result.showdown else None,
    )
    st.html(render_log_html(entries))
    rows = previous_hands_rows(
        match.records, [i.mention for i in identities], match.config.big_blind, limit=HISTORY_ROWS_SHOWN
    )
    if rows:
        with st.expander(f"Previous hands ({match.hands_completed})"):
            st.dataframe(rows, hide_index=True, width="stretch")


def _render_game(ns: SessionNamespace, match: Match) -> None:
    identities = seat_identities(match)
    obs = match.session.observation(HUMAN_SEAT)  # type: ignore[union-attr]  # the restricted view - the only source for the table
    result = match.records[-1].result if match.hand_complete else None
    badge, tone = _turn_badge(match)
    table = player_table(
        obs,
        identities,
        viewer_seat=HUMAN_SEAT,
        actor_seat=match.actor_seat,
        result=result,
        badge=badge,
        badge_tone=tone,
    )
    if error := ns.get("error"):
        st.error(error)
    st.html(_status_html(match, obs))
    left, right = st.columns([5, 3], gap="large")
    with left:
        st.html(render_table_html(table))
        if match.hand_complete:
            _render_outcome(ns, match, identities)
        else:
            _render_action_panel(ns, match, obs)
    with right:
        _render_stats(match)
        _render_history(match, identities, obs)


def _lobby(roster: dict[str, RosterEntry], ns: SessionNamespace) -> None:
    st.info(
        "Pick an opponent in the sidebar (arrow at the top left on small screens) and press **Play**. "
        "You sit in seat 1; the button alternates every hand. The AI sees exactly what a real opponent "
        "would see - never your cards."
    )
    kind = ns.get("opponent", _preferred(roster))
    st.html(agent_card_html(roster[kind]))


def _think_then_act(ns: SessionNamespace, match: Match) -> None:
    """The AI's turn: wait a moment (so the move is followable), act once, redraw."""
    time.sleep(float(ns.get("think", DEFAULT_THINK_SECONDS)))
    try:
        match.step()
    except Exception as exc:  # noqa: BLE001 - show agent failures instead of crashing the page
        ns.set("error", f"{type(exc).__name__}: {exc}")
        return
    st.rerun()


def render_human_vs_ai_page() -> None:
    configure_page("Human vs AI", "🧑‍💻")
    ns = SessionNamespace(NAMESPACE)
    roster = cached_roster()
    st.title("Human vs AI")
    st.caption("Heads-up No-Limit Hold'em against one of the agents. Its hole cards stay hidden until a showdown.")
    if not any(e.available for e in roster.values()):
        st.error("No agent is available - see the Home page for details.")
        return

    request = _sidebar_setup(ns, roster, running=ns.get("match") is not None)
    if request is not None:
        ns.forget(*GAME_KEYS)
        ns.set("request", request)
        ns.set("match", build_match(roster, request))
        st.rerun()  # redraw the sidebar (its button label depends on a running game)
    match: Match | None = ns.get("match")
    if match is None:
        _lobby(roster, ns)
        return
    _render_game(ns, match)
    if match.awaiting_agent and not ns.get("error"):
        _think_then_act(ns, match)
