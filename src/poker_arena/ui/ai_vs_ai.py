"""Mode A - watch two AI agents play heads-up No-Limit Hold'em.

All state lives in ``st.session_state`` (namespace ``ava``). The live part of the page (status,
table, decision panel, scoreboard, log) is a ``st.fragment`` that re-runs on a timer while autoplay
is on, so widgets and the sidebar are not re-rendered on every tick.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from html import escape

import streamlit as st

from poker_arena.config import derive_seed
from poker_arena.ui.charts import profit_figure
from poker_arena.ui.explain import render_step_detail
from poker_arena.ui.history import (
    STREET_NAMES,
    build_hand_log,
    describe_action,
    previous_hands_rows,
    render_log_html,
)
from poker_arena.ui.match import (
    DEFAULT_SPEED,
    MAX_SPEED,
    MIN_HANDS_FOR_RATE,
    MIN_SPEED,
    Match,
    MatchSettings,
    Pacer,
    TickKind,
    tick_interval,
)
from poker_arena.ui.roster import RosterEntry, spawn_agent
from poker_arena.ui.state import SessionNamespace
from poker_arena.ui.table import render_table_html
from poker_arena.ui.theme import configure_page
from poker_arena.ui.views import SeatIdentity, agent_identity, showdown_hand_names, spectator_table
from poker_arena.ui.widgets import (
    GameSettings,
    agent_card_html,
    agent_picker,
    cached_roster,
    game_settings_widgets,
    unavailable_agents_note,
)

NAMESPACE = "ava"
LATEST = -1  # sentinel of the "inspect decision" selector: follow the newest decision
PREFERRED_SEAT_KINDS = (("ebm", "teacher", "random"), ("xgboost", "ga", "maniac"))
HISTORY_ROWS_SHOWN = 30
GAME_KEYS = ("match", "request", "playing", "pacer", "error")
SPECTATOR_BADGE = "Spectator view - hidden from the agents"


@dataclass(frozen=True, slots=True)
class StartRequest:
    kinds: tuple[str, str]
    settings: GameSettings


def pick_default(roster: dict[str, RosterEntry], preferences: tuple[str, ...]) -> str:
    """First available preferred kind, else the first available agent."""
    available = [k for k, e in roster.items() if e.available]
    return next((k for k in preferences if k in available), available[0])


def build_match(roster: dict[str, RosterEntry], request: StartRequest) -> Match:
    """New match with private agent copies; the first hand is already dealt."""
    seed = request.settings.seed
    agents = [spawn_agent(roster[k], derive_seed("ui", seed, "agent", seat)) for seat, k in enumerate(request.kinds)]
    settings = MatchSettings(
        config=request.settings.config,
        seed=seed,
        reset_stacks_each_hand=request.settings.reset_stacks_each_hand,
        auto_rebuy=True,
    )
    match = Match(settings, agents)
    match.start_hand()
    return match


def seat_identities(match: Match) -> tuple[SeatIdentity, SeatIdentity]:
    first, second = (agent_identity(seat, a.name, a.architecture) for seat, a in enumerate(match.agents) if a)
    return first, second


# ------------------------------------------------------------------------------ actions
def _start_game(ns: SessionNamespace, roster: dict[str, RosterEntry], request: StartRequest, *, autoplay: bool) -> None:
    ns.forget(*GAME_KEYS, "inspect")
    ns.set("request", request)
    ns.set("match", build_match(roster, request))
    ns.set("playing", autoplay)
    ns.set("pacer", Pacer())


def _guarded(ns: SessionNamespace, action: Callable[[], object]) -> None:
    """Run a match mutation; agent errors stop autoplay and are shown instead of crashing the page."""
    try:
        action()
    except Exception as exc:  # noqa: BLE001
        ns.set("error", f"{type(exc).__name__}: {exc}")
        ns.set("playing", False)


def _advance(ns: SessionNamespace) -> None:
    match: Match = ns.get("match")

    def tick() -> None:
        if match.advance() is TickKind.DEAL:
            ns.pop("inspect")

    _guarded(ns, tick)


def _on_next_action(ns: SessionNamespace) -> None:
    ns.set("playing", False)
    _advance(ns)


def _on_next_hand(ns: SessionNamespace) -> None:
    ns.set("playing", False)
    ns.pop("inspect")
    _guarded(ns, ns.get("match").play_out)


def _on_toggle_play(ns: SessionNamespace) -> None:
    ns.set("playing", not ns.get("playing"))
    ns.pop("error")


def _on_reset(ns: SessionNamespace, roster: dict[str, RosterEntry]) -> None:
    _start_game(ns, roster, ns.get("request"), autoplay=False)


# ------------------------------------------------------------------------------ sidebar
def _sidebar_setup(ns: SessionNamespace, roster: dict[str, RosterEntry], *, running: bool) -> StartRequest | None:
    """Agent + game settings. Returns a request when the user presses Start."""
    with st.sidebar:
        st.header("Game setup")
        kinds = (
            agent_picker("Seat 1 agent", roster, ns.key("kind0"), pick_default(roster, PREFERRED_SEAT_KINDS[0])),
            agent_picker("Seat 2 agent", roster, ns.key("kind1"), pick_default(roster, PREFERRED_SEAT_KINDS[1])),
        )
        unavailable_agents_note(roster, ns.key("rescan"))
        settings, error = game_settings_widgets(
            ns,
            default_seed=42,
            reset_help="Off: continuous play, stacks carry over and a busted player is rebought automatically.",
        )
        if error:
            st.error(error)
        pressed = st.button(
            "Restart with these settings" if running else "Start game",
            type="primary",
            width="stretch",
            disabled=settings is None,
            key=ns.key("start"),
        )
    return StartRequest(kinds, settings) if pressed and settings is not None else None


# ------------------------------------------------------------------------------ lobby
def _render_lobby(roster: dict[str, RosterEntry], ns: SessionNamespace) -> None:
    st.info(
        "Choose two agents and the game settings in the sidebar (arrow at the top left on small screens), "
        "then press **Start game**."
    )
    cols = st.columns(2)
    for seat, col in enumerate(cols):
        kind = ns.get(f"kind{seat}", pick_default(roster, PREFERRED_SEAT_KINDS[seat]))
        with col:
            st.markdown(f"**Seat {seat + 1}**")
            st.html(agent_card_html(roster[kind]))


# ------------------------------------------------------------------------------ controls
def _render_controls(ns: SessionNamespace, roster: dict[str, RosterEntry]) -> None:
    playing = bool(ns.get("playing"))
    cols = st.columns([1.15, 1.25, 1.25, 1.0, 2.4, 1.6], vertical_alignment="bottom")
    cols[0].button(
        "⏸ Pause" if playing else "▶ Resume",
        on_click=_on_toggle_play,
        args=(ns,),
        type="primary",
        width="stretch",
        key=ns.key("toggle"),
        help="Autoplay on/off",
    )
    cols[1].button(
        "⏭ Next action",
        on_click=_on_next_action,
        args=(ns,),
        width="stretch",
        key=ns.key("next_action"),
        help="Advance exactly one decision (or deal the next hand)",
    )
    cols[2].button(
        "⏩ Next hand",
        on_click=_on_next_hand,
        args=(ns,),
        width="stretch",
        key=ns.key("next_hand"),
        help="Play out the current hand instantly (or play a whole new hand)",
    )
    cols[3].button(
        "↺ Reset",
        on_click=_on_reset,
        args=(ns, roster),
        width="stretch",
        key=ns.key("reset"),
        help="Restart this game from hand 1 with the same agents and seed",
    )
    cols[4].slider(
        "Speed", MIN_SPEED, MAX_SPEED, DEFAULT_SPEED, 0.1, format="%.1fx", key=ns.key("speed"),
        help="1.0x = one action every 1.5 seconds",
    )
    cols[5].toggle(
        "Show hole cards",
        value=True,
        key=ns.key("reveal"),
        help="Spectator view: both players' cards are visible to you, never to the agents.",
    )


# ------------------------------------------------------------------------------ live view
def _autoplay_tick(ns: SessionNamespace, match: Match) -> None:
    pacer: Pacer = ns.setdefault("pacer", Pacer)
    interval = tick_interval(ns.get("speed", DEFAULT_SPEED))
    now = time.monotonic()
    if pacer.due(now, interval, after_hand_end=match.hand_complete):
        pacer.mark(now)
        _advance(ns)


def _status_html(match: Match, identities: tuple[SeatIdentity, ...], playing: bool) -> str:
    engine = match.session.engine  # type: ignore[union-attr]
    pills = [f'<span class="pa-pill">Hand <b>{match.hand_number + 1}</b></span>']
    if match.hand_complete:
        pills.append('<span class="pa-pill pa-pill--end">Hand over</span>')
    else:
        pills.append(f'<span class="pa-pill">Street <b>{STREET_NAMES[engine.street]}</b></span>')
        actor = engine.actor_seat
        if actor is not None:
            pills.append(
                f'<span class="pa-pill pa-pill--turn"><span class="pa-dot"></span>'
                f"To act <b>{escape(identities[actor].mention)}</b></span>"
            )
    cfg = match.config
    pills.append(f'<span class="pa-pill">Blinds <b>{cfg.small_blind}/{cfg.big_blind}</b></span>')
    pills.append(
        '<span class="pa-pill pa-pill--turn">▶ Autoplay</span>'
        if playing
        else '<span class="pa-pill pa-pill--wait">⏸ Paused</span>'
    )
    return f'<div class="pa-status">{"".join(pills)}</div>'


def _decision_panel(ns: SessionNamespace, match: Match, identities: tuple[SeatIdentity, ...]) -> None:
    st.html('<div class="pa-panel-title">Decision panel</div>')
    steps = match.steps
    if not steps:
        st.caption("No decision has been taken in this hand yet.")
        return
    options = [LATEST, *range(len(steps))]
    if ns.get("inspect") not in options:
        ns.set("inspect", LATEST)
    choice = st.selectbox(
        "Inspect decision",
        options,
        key=ns.key("inspect"),
        format_func=lambda i: (
            "Latest decision"
            if i == LATEST
            else f"#{i + 1} {identities[steps[i].event.seat].mention} {describe_action(steps[i].event)}"
        ),
    )
    index = len(steps) - 1 if choice == LATEST else choice
    step = steps[index]
    render_step_detail(
        step,
        identities[step.event.seat].mention,
        key=f"{ns.key('expl')}-{match.hand_number}-{index}",
    )


def _scoreboard(match: Match, identities: tuple[SeatIdentity, ...]) -> None:
    st.html('<div class="pa-panel-title">Running profit (cumulative BB)</div>')
    cols = st.columns(2)
    for seat, col in enumerate(cols):
        rate = match.bb_per_100(seat)
        col.metric(
            identities[seat].mention,
            f"{match.profit_bb(seat):+.1f} BB",
            delta=None if rate is None else f"{rate:+.0f} BB/100",
            help="Net big blinds won since the game started.",
        )
    if match.profit_series_bb:
        st.plotly_chart(
            profit_figure(match.profit_series_bb, [i.mention for i in identities]),
            key=f"profit-{match.hands_completed}",
            theme=None,
            width="stretch",
        )
    st.caption(
        f"{match.hands_completed} hands played. Short samples are mostly luck: BB/100 is shown after "
        f"{MIN_HANDS_FOR_RATE} hands, and the AI Championship page reports it with confidence intervals."
    )


def _hand_log(match: Match, identities: tuple[SeatIdentity, ...]) -> None:
    st.html('<div class="pa-panel-title">Hand history</div>')
    engine = match.session.engine  # type: ignore[union-attr]
    view = engine.spectator_view()
    entries = build_hand_log(
        engine.events,
        [i.mention for i in identities],
        config=match.config,
        button_seat=engine.button_seat,
        start_stacks=engine.start_stacks,  # type: ignore[arg-type]
        board=view.board,
        result=view.result,
        hand_names=showdown_hand_names(view.result) if view.result and view.result.showdown else None,
    )
    st.html(render_log_html(entries))
    rows = previous_hands_rows(
        match.records, [i.mention for i in identities], match.config.big_blind, limit=HISTORY_ROWS_SHOWN
    )
    if rows:
        with st.expander(f"Previous hands ({match.hands_completed})"):
            st.dataframe(rows, hide_index=True, width="stretch")


def _live_view(ns: SessionNamespace) -> None:
    """Fragment body: advance autoplay (if on) and draw everything that changes."""
    match: Match = ns.get("match")
    playing = bool(ns.get("playing"))
    if playing:
        _autoplay_tick(ns, match)
    if error := ns.get("error"):
        st.error(f"Autoplay stopped: {error}")
    identities = seat_identities(match)
    reveal = bool(ns.get("reveal", True))
    view = match.session.engine.spectator_view()  # type: ignore[union-attr]
    st.html(_status_html(match, identities, bool(ns.get("playing"))))
    left, right = st.columns([5, 3], gap="large")
    with left:
        table = spectator_table(
            view,
            identities,
            config=match.config,
            hand_number=match.hand_number,
            reveal_hole_cards=reveal,
            badge=SPECTATOR_BADGE if reveal else None,
            badge_tone="warn",
        )
        st.html(render_table_html(table))
        _scoreboard(match, identities)
    with right:
        _decision_panel(ns, match, identities)
        _hand_log(match, identities)


def render_ai_vs_ai_page() -> None:
    configure_page("AI vs AI", "🤖")
    ns = SessionNamespace(NAMESPACE)
    roster = cached_roster()
    st.title("AI vs AI")
    st.caption("Watch two agents play heads-up No-Limit Hold'em and inspect every decision.")
    if not any(e.available for e in roster.values()):
        st.error("No agent is available - see the Home page for details.")
        return

    request = _sidebar_setup(ns, roster, running=ns.get("match") is not None)
    if request is not None:
        _start_game(ns, roster, request, autoplay=True)
        st.rerun()  # redraw the sidebar (its button label depends on a running game)
    if ns.get("match") is None:
        _render_lobby(roster, ns)
        return

    _render_controls(ns, roster)
    playing = bool(ns.get("playing"))
    live = st.fragment(_live_view, run_every=tick_interval(ns.get("speed", DEFAULT_SPEED)) if playing else None)
    live(ns)
