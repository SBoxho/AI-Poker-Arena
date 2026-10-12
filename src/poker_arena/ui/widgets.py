"""Streamlit widgets shared by the pages: agent picker, game settings, roster cards."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st

from poker_arena.config import GameConfig
from poker_arena.ui.roster import RosterEntry, discover_roster
from poker_arena.ui.state import SessionNamespace, random_seed

ROSTER_TTL_SECONDS = 30  # trained agents appear shortly after their artifacts are written
DEFAULT_CONFIG = GameConfig()
MIN_STACK, MAX_STACK, STACK_STEP = 20, 1_000_000, 100
MAX_BLIND = 100_000
MAX_SEED = 2**31 - 1


@st.cache_resource(ttl=ROSTER_TTL_SECONDS, show_spinner=False)
def cached_roster() -> dict[str, RosterEntry]:
    """Loaded agent templates, shared (read-only) by all sessions."""
    return discover_roster()


def refresh_roster() -> None:
    cached_roster.clear()


# ----------------------------------------------------------------------------- agents
def agent_card_html(entry: RosterEntry) -> str:
    """Roster card; unavailable agents are greyed out and show the reason."""
    if entry.available:
        badge = '<span class="pa-status-badge pa-status-badge--on">Available</span>'
        explain = "Explains its decisions" if entry.supports_explanations else "No decision explanations"
        body = (
            f'<p><b>{escape(entry.architecture)}</b></p><p>{escape(entry.description)}</p>'
            f'<p class="pa-small">{explain}</p>'
        )
        css = "pa-agent"
    else:
        badge = '<span class="pa-status-badge pa-status-badge--off">Unavailable</span>'
        body = f"<p>{escape(entry.detail)}</p>"
        css = "pa-agent pa-agent--off"
    return f'<div class="{css}"><h4>{escape(entry.label)}{badge}</h4>{body}</div>'


def roster_grid_html(roster: dict[str, RosterEntry]) -> str:
    return f'<div class="pa-roster">{"".join(agent_card_html(e) for e in roster.values())}</div>'


def agent_picker(label: str, roster: dict[str, RosterEntry], key: str, default_kind: str) -> str:
    """Select among *available* agents; returns the chosen kind."""
    kinds = [kind for kind, entry in roster.items() if entry.available]
    index = kinds.index(default_kind) if default_kind in kinds else 0
    return st.selectbox(
        label,
        kinds,
        index=index,
        format_func=lambda k: roster[k].label,
        key=key,
        help="Any available agent can be picked for both seats.",
    )


def unavailable_agents_note(roster: dict[str, RosterEntry], key: str) -> None:
    """Greyed-out list of agents that cannot be played yet, with the reason."""
    missing = {k: e for k, e in roster.items() if not e.available}
    if not missing:
        return
    with st.expander(f"Unavailable agents ({len(missing)})"):
        st.html(roster_grid_html(missing))
        st.button("Re-scan agents", on_click=refresh_roster, key=key)


def describe_selection(entry: RosterEntry) -> None:
    st.html(agent_card_html(entry))


# ------------------------------------------------------------------------------ game
@dataclass(frozen=True, slots=True)
class GameSettings:
    config: GameConfig
    seed: int
    reset_stacks_each_hand: bool


def _reroll_seed(seed_key: str) -> None:
    st.session_state[seed_key] = random_seed()


def game_settings_widgets(
    ns: SessionNamespace, *, default_seed: int | None, reset_help: str
) -> tuple[GameSettings | None, str | None]:
    """Stack / blinds / seed / stack-reset widgets. Returns ``(settings, None)`` or ``(None, error)``.

    ``default_seed=None`` draws a fresh random seed once per browser session.
    """
    seed_key = ns.key("seed")
    ns.setdefault("seed", lambda: random_seed() if default_seed is None else default_seed)
    stack = st.number_input(
        "Starting stack (chips)",
        MIN_STACK,
        MAX_STACK,
        DEFAULT_CONFIG.starting_stack,
        STACK_STEP,
        key=ns.key("stack"),
    )
    col_sb, col_bb = st.columns(2)
    small = col_sb.number_input("Small blind", 1, MAX_BLIND, DEFAULT_CONFIG.small_blind, key=ns.key("sb"))
    big = col_bb.number_input("Big blind", 1, MAX_BLIND, DEFAULT_CONFIG.big_blind, key=ns.key("bb"))
    col_seed, col_roll = st.columns([3, 1], vertical_alignment="bottom")
    seed = col_seed.number_input(
        "Seed", 0, MAX_SEED, step=1, key=seed_key, help="Same seed + same agents = same cards and play."
    )
    col_roll.button("🎲", on_click=_reroll_seed, args=(seed_key,), help="New random seed", key=ns.key("reroll"))
    reset_each_hand = st.checkbox("Reset stacks every hand", value=False, key=ns.key("reset_each"), help=reset_help)
    try:
        config = GameConfig(int(stack), int(small), int(big))
    except ValueError as exc:
        return None, str(exc)
    return GameSettings(config, int(seed), reset_each_hand), None
