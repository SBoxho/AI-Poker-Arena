"""Mode B - AI Championship: a statistics dashboard over saved tournament runs.

The page only READS runs written by ``scripts/run_tournament.py`` (through
``poker_arena.tournament.results``); nothing heavy runs here. The "run a new tournament" panel just
builds the command line. Pure builders live in ``tournament_data`` / ``tournament_figures`` /
``replay``; this module wires them to Streamlit widgets.

Statistical hygiene shown to the user: every number carries its interval, "significant" means the
interval excludes 0, and a better rank is NOT a significant difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from poker_arena.agents.registry import KIND_LABELS, known_kinds
from poker_arena.tournament.results import TournamentResults, list_runs, load_results
from poker_arena.ui.explain import FeatureRow, render_features_html
from poker_arena.ui.history import render_log_html
from poker_arena.ui.replay import ReplayFrame, replay_frames
from poker_arena.ui.state import SessionNamespace
from poker_arena.ui.table import render_table_html
from poker_arena.ui.theme import configure_page
from poker_arena.ui.tournament_data import (
    CI_LEVELS,
    DEFAULT_LEVEL,
    MATRIX_METRICS,
    METRIC_BY_KEY,
    PCT,
    RUN_SCRIPT,
    IntervalView,
    LaunchSpec,
    ModelCard,
    estimate_duration_seconds,
    format_ci,
    format_duration,
    interval_view,
    launch_command,
    leaderboard_display,
    load_model_cards,
    matrix_from_pairwise,
    models_root,
    reproduce_command,
    run_facts,
    tournaments_root,
    validate_launch,
)
from poker_arena.ui.tournament_figures import (
    cumulative_figure,
    forest_figure,
    gene_figure,
    importance_figure,
    latency_figure,
    matchup_heatmap,
    opponent_figure,
    profit_by_end_figure,
    street_actions_figure,
    style_bars,
    style_radar,
    verdict_table_html,
)
from poker_arena.ui.widgets import cached_roster

NAMESPACE = "champ"
MAX_CACHED_RUNS = 6
CURVE_POINTS = 600  # long profit curves are thinned for plotting
ALL_OPPONENTS = "All opponents"
TAB_NAMES = (
    "Leaderboard",
    "Head-to-head",
    "Profit curves",
    "Playing style",
    "By opponent & street",
    "Models",
    "Latency",
    "Interesting hands",
)
HAND_TAGS = {
    "first": "First hands of a match",
    "biggest_pot": "Biggest pots",
    "biggest_win_a": "Biggest wins of agent A",
    "biggest_win_b": "Biggest wins of agent B",
}
STYLE_COLUMNS = {
    "vpip": "VPIP",
    "pfr": "PFR",
    "three_bet": "3-bet",
    "aggression_freq_postflop": "Postflop aggression",
    "fold_freq": "Fold",
    "call_freq": "Check/call",
    "raise_freq": "Bet/raise",
    "showdown_rate": "Showdown rate",
    "showdown_win_rate": "Showdown win rate",
    "win_rate": "Hands won",
}
VARIANCE_NOTE = (
    "Poker is a high-variance game: a few thousand hands can put any agent ahead by luck. Read every number "
    "together with its confidence interval - 'significant' here means the interval excludes 0, and a higher "
    "rank is not a significant difference unless the head-to-head verdict says so. Intervals are computed "
    "over independent sessions (agents are rebuilt at every session start), because agents that adapt to "
    "their opponent make the pairs inside a session dependent."
)
SKILL_NOTE = (
    "**Imitation accuracy is not poker skill.** Accuracy / agreement numbers say how well a model copies its "
    "teacher's decisions on held-out data; BB/100 in a tournament says how much it wins at the table. A model "
    "can imitate very well and still lose money, so the two are shown side by side and must not be mixed up."
)
RUN_COMMANDS = (
    f"{RUN_SCRIPT} --smoke   # ~1 minute end-to-end check (never marked as the official run)\n"
    f"{RUN_SCRIPT}           # full tournament with the configuration block of the script\n"
    f"{RUN_SCRIPT} --agents random,call_station,maniac --set NUM_TOURNAMENT_HANDS=50000"
)


# ---------------------------------------------------------------------------- data access
@st.cache_resource(max_entries=MAX_CACHED_RUNS, show_spinner=False)
def open_run(run_dir: str, signature: float) -> TournamentResults:
    """Open a saved run once per (folder, modification time); runs are read-only."""
    return load_results(Path(run_dir))


@st.cache_data(max_entries=32, show_spinner=False)
def cached_interval_view(run_dir: str, signature: float, agents: tuple[str, ...], level: float) -> IntervalView:
    return interval_view(open_run(run_dir, signature), agents, level)


def run_signature(run_dir: str) -> float:
    path = Path(run_dir) / "summary.json"
    return path.stat().st_mtime if path.exists() else 0.0


def run_label(row: pd.Series) -> str:
    star = " (latest)" if row["is_latest"] else ""
    return f"{row['run_id']}{star} - {row['agents']} - {int(row['hands_per_match']):,} hands/pairing"


def _open_or_stop(run_dir: str) -> TournamentResults:
    try:
        return open_run(run_dir, run_signature(run_dir))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"This run cannot be read: {exc}")
        st.stop()


@dataclass(frozen=True)
class Page:
    """Everything the tab renderers need."""

    results: TournamentResults
    view: IntervalView
    level: float
    metric: str
    ns: SessionNamespace

    @property
    def agents(self) -> tuple[str, ...]:
        return tuple(self.view.order)


# ------------------------------------------------------------------- launch-a-tournament panel
def _agent_option_label(kind: str, available: bool) -> str:
    return KIND_LABELS.get(kind, kind) + ("" if available else " (not available yet - skipped)")


def _launch_spec_from_widgets(ns: SessionNamespace) -> LaunchSpec:
    roster = cached_roster()
    kinds = known_kinds()
    chosen = st.multiselect(
        "Agents",
        kinds,
        default=[k for k in kinds if roster[k].available],
        format_func=lambda k: _agent_option_label(k, roster[k].available),
        key=ns.key("launch_agents"),
    )
    c1, c2, c3 = st.columns(3)
    hands = c1.number_input(
        "Hands per pairing", 2, 10_000_000, 20_000, 1000, key=ns.key("launch_hands"),
        help="Rounded up to an even number (a duplicate pair is two hands).",
    )  # fmt: skip
    seed = c2.number_input("Seed", 0, 2**31 - 1, 42, key=ns.key("launch_seed"))
    workers = c3.number_input(
        "Workers", 0, 61, 0, key=ns.key("launch_workers"), help="0 = all cores but one. Results do not depend on it."
    )
    c4, c5, c6 = st.columns(3)
    stack = c4.number_input("Starting stack", 20, 1_000_000, 1000, 100, key=ns.key("launch_stack"))
    small = c5.number_input("Small blind", 1, 100_000, 5, key=ns.key("launch_sb"))
    big = c6.number_input("Big blind", 1, 100_000, 10, key=ns.key("launch_bb"))
    c7, c8, c9 = st.columns(3)
    method = c7.selectbox("Interval method", ["t", "bootstrap"], key=ns.key("launch_method"))
    confidence = c8.selectbox(
        "Confidence level", CI_LEVELS, index=CI_LEVELS.index(DEFAULT_LEVEL), format_func=lambda c: f"{c * PCT:g}%",
        key=ns.key("launch_conf"),
    )  # fmt: skip
    name = c9.text_input("Run name (optional)", key=ns.key("launch_name"))
    duplicate = st.checkbox(
        "Duplicate (matched) deals", value=True, key=ns.key("launch_dup"),
        help="Every deal is played twice with the seats swapped, at two independent tables; this cancels most card luck.",
    )  # fmt: skip
    official = st.checkbox(
        "Mark as the latest (official) run", value=False, key=ns.key("launch_latest"),
        help="Leave off for experiments: the dashboard's default run follows latest.json.",
    )  # fmt: skip
    return LaunchSpec(
        tuple(chosen), int(hands), int(seed), int(stack), int(small), int(big), int(workers), duplicate, method,
        float(confidence), name.strip(), official,
    )  # fmt: skip


def _render_launcher(ns: SessionNamespace, hands_per_second: float | None, *, expanded: bool) -> None:
    with st.expander("Run a new tournament (command builder)", expanded=expanded):
        st.caption(
            "This page never runs a tournament - heavy simulation belongs in a terminal. Choose the settings, "
            "copy the command and run it from the repository root; the new run then appears in the run selector."
        )
        spec = _launch_spec_from_widgets(ns)
        error = validate_launch(spec)
        if error:
            st.warning(error)
            return
        st.code(launch_command(spec), language="bash", wrap_lines=True)
        seconds = estimate_duration_seconds(spec, hands_per_second)
        if seconds is not None:
            st.caption(
                f"Rough duration: {format_duration(seconds)}, scaled from the throughput of the selected run "
                "(depends on the agents and the number of cores)."
            )


def _render_empty(ns: SessionNamespace, root: Path) -> None:
    st.info("**No tournament results yet.** Run a tournament first - it writes run folders that this page then reads:")
    st.code(RUN_COMMANDS, language="bash")
    st.caption(f"Looking for runs in `{root}`.")
    _render_launcher(ns, None, expanded=True)


# ------------------------------------------------------------------------------- run header
def _render_run_header(results: TournamentResults) -> None:
    agents = "".join(f'<span class="pa-pill pa-pill--turn">{escape(a)}</span>' for a in results.agent_names)
    facts = "".join(
        f'<span class="pa-pill">{escape(label)} <b>{escape(value)}</b></span>' for label, value in run_facts(results)
    )
    st.html(f'<div class="pa-status">{agents}</div><div class="pa-status">{facts}</div>')
    if results.protocol_note:
        st.warning(results.protocol_note)
    if results.ci_unit_note:
        st.warning(results.ci_unit_note)
    with st.expander("Reproduce this run / environment"):
        st.code(reproduce_command(results), language="bash", wrap_lines=True)
        env = results.meta["environment"]
        libraries = ", ".join(f"{k} {v}" for k, v in env["libraries"].items())
        st.caption(f"Python {env['python']} on {env['platform']}. Libraries: {libraries}.")


# --------------------------------------------------------------------------------- tabs
def _recompute_note(page: Page) -> None:
    if page.view.recomputed:
        method = "t-interval" if page.view.method == "t" else "bootstrap"
        st.caption(
            f"Intervals re-estimated at {page.level * PCT:g}% ({method}) from the saved per-{page.results.ci_unit} profits for the "
            "selected agents; the tournament itself stored a different level / set of agents."
        )


def _needs_two_agents(page: Page) -> bool:
    if len(page.view.order) < 2:
        st.warning("Select at least two agents in the sidebar.")
        return True
    return False


def _tab_leaderboard(page: Page) -> None:
    if _needs_two_agents(page):
        return
    board = page.view.leaderboard
    _recompute_note(page)
    st.plotly_chart(forest_figure(board, page.level), key=page.ns.key("forest"), theme=None, width="stretch")
    st.dataframe(leaderboard_display(board, page.level), hide_index=True, width="stretch")
    st.caption(
        "BB/100 = big blinds won per 100 hands, averaged over opponents with every opponent weighted equally. "
        + page.results.meta["ranking_note"]
    )
    st.info(VARIANCE_NOTE)
    with st.expander("Who are the agents?"):
        st.dataframe(page.results.agents().drop(columns=["params"]), hide_index=True, width="stretch")


def _matrix_inputs(page: Page) -> dict[str, pd.DataFrame]:
    """Heatmap inputs: values, plus interval bounds and significance for BB/100."""
    order, pairwise = page.view.order, page.view.pairwise
    if page.metric == "bb_per_100":
        return {
            "values": matrix_from_pairwise(pairwise, "bb_per_100", order),
            "ci_low": matrix_from_pairwise(pairwise, "ci_low", order),
            "ci_high": matrix_from_pairwise(pairwise, "ci_high", order),
            "significant": matrix_from_pairwise(pairwise, "significant", order).fillna(False),
        }
    return {"values": page.results.matchup_matrix(page.metric).reindex(index=order, columns=order)}


def _pairing_details(page: Page) -> None:
    order = page.view.order
    col_a, col_b = st.columns(2)
    agent = col_a.selectbox("Agent", order, key=page.ns.key("pair_a"))
    opponent = col_b.selectbox("Opponent", [o for o in order if o != agent], key=page.ns.key("pair_b"))
    pairwise = page.view.pairwise
    row = pairwise[(pairwise["agent"] == agent) & (pairwise["opponent"] == opponent)].iloc[0]
    cols = st.columns(4)
    interval = f"{page.level * PCT:g}% CI {format_ci(row['ci_low'], row['ci_high'], 1)}"
    cols[0].metric("BB/100", f"{row['bb_per_100']:+.1f}", help=interval)
    cols[1].metric("Interval half-width", f"±{(row['ci_high'] - row['ci_low']) / 2:.1f}")
    cols[2].metric("Hands won", f"{row['win_rate']:.1%}", help="Share of hands won, not weighted by pot size.")
    cols[3].metric("Showdowns won", f"{row['showdown_win_rate']:.1%}")
    ratio = row["se_ratio"]
    variance = "" if pd.isna(ratio) else (
        f" Duplicate dealing brought the standard error to {ratio:.0%} of what independent hands would give."
    )
    st.markdown(f"**Verdict:** {row['verdict_text']}.{variance}")


def _tab_head_to_head(page: Page) -> None:
    if _needs_two_agents(page):
        return
    _recompute_note(page)
    metric = METRIC_BY_KEY[page.metric]
    st.markdown(f"**{metric.label}** - row agent against column agent.")
    inputs = _matrix_inputs(page)
    fig = matchup_heatmap(
        inputs.pop("values"), page.metric, verdicts=page.view.verdict_texts, level=page.level, **inputs
    )
    st.plotly_chart(fig, key=page.ns.key("heatmap"), theme=None, width="stretch")
    if page.metric == "bb_per_100":
        st.caption(
            "Cells: BB/100, ± half-width of the interval, ★ = interval excludes 0. Hover for the exact interval."
        )
    st.markdown("#### Verdicts")
    st.html(verdict_table_html(page.view.verdict_codes, page.view.verdict_texts))
    st.caption(
        "A verdict is 'significant' only when the interval of the difference excludes 0 (with enough independent "
        "sessions). 'No significant difference' is absence of evidence, not proof of equal strength."
    )
    st.info(VARIANCE_NOTE)
    st.markdown("#### Pairing details")
    _pairing_details(page)


def _tab_curves(page: Page) -> None:
    if _needs_two_agents(page):
        return
    order = page.view.order
    col_a, col_b = st.columns(2)
    agent = col_a.selectbox("Agent", order, key=page.ns.key("curve_a"))
    opponents = [o for o in order if o != agent]
    opponent = col_b.selectbox("Opponent", opponents, key=page.ns.key("curve_b"))
    everyone = st.checkbox("Compare against all opponents (no luck band)", key=page.ns.key("curve_all"))
    versus = opponents if everyone else [opponent]
    curves = {f"{agent} vs {o}": page.results.cumulative_profit(agent, o, CURVE_POINTS) for o in versus}
    fig = cumulative_figure(curves, luck_band=not everyone, level=page.results.confidence)
    st.plotly_chart(fig, key=page.ns.key("curves"), theme=None, width="stretch")
    st.caption(
        "Cumulative profit in big blinds after every duplicate pair. The grey band is where a player with NO edge "
        f"would stay with {page.results.confidence * PCT:g}% probability (its width is derived from the "
        f"{page.results.ci_unit}-level variance, like the intervals): a curve that leaves the band is evidence "
        "of a real edge, one that stays inside is compatible with luck."
    )


def _against_selector(page: Page, key: str) -> str | None:
    choice = st.selectbox(
        "Against", [ALL_OPPONENTS, *page.agents], key=page.ns.key(key),
        help="Pool the statistics over all opponents, or look at a single opponent.",
    )  # fmt: skip
    return None if choice == ALL_OPPONENTS else choice


def _tab_style(page: Page) -> None:
    against = _against_selector(page, "style_vs")
    style = page.results.style_table(against)
    style = style[style["agent"].isin(page.agents)].reset_index(drop=True).fillna(0.0)
    if style.empty:
        st.warning("No data for this selection.")
        return
    chart = st.radio("Chart", ["Radar", "Grouped bars"], horizontal=True, key=page.ns.key("style_view"))
    fig = style_radar(style) if chart == "Radar" else style_bars(style)
    st.plotly_chart(fig, key=page.ns.key("style_fig"), theme=None, width="stretch")
    shown = style.rename(columns={**STYLE_COLUMNS, "aggression_factor": "Aggression factor"})
    percent = {c: st.column_config.NumberColumn(c, format="percent") for c in STYLE_COLUMNS.values()}
    st.dataframe(
        shown[["agent", "hands", *STYLE_COLUMNS.values(), "Aggression factor"]],
        hide_index=True, width="stretch", column_config=percent,
    )  # fmt: skip
    st.caption(
        "VPIP = hands where the agent voluntarily put chips in preflop; PFR = preflop raises; postflop aggression = "
        "(bets + raises) / all postflop actions. Style describes behaviour, not strength."
    )


def _tab_opponent_street(page: Page) -> None:
    st.markdown("#### Result by opponent")
    agent = st.selectbox("Agent", page.agents, key=page.ns.key("opp_agent"))
    rows = page.view.pairwise[page.view.pairwise["agent"] == agent]
    if not rows.empty:
        _recompute_note(page)
        st.plotly_chart(opponent_figure(rows, agent, page.level), key=page.ns.key("opp_fig"), theme=None, width="stretch")
        table = rows[["opponent", "bb_per_100", "ci_low", "ci_high", "verdict_text", "hands"]].rename(
            columns={"opponent": "Opponent", "bb_per_100": "BB/100", "ci_low": "CI low", "ci_high": "CI high",
                     "verdict_text": "Verdict", "hands": "Hands"}  # fmt: skip
        )
        signed = {c: st.column_config.NumberColumn(format="%+.1f") for c in ("BB/100", "CI low", "CI high")}
        st.dataframe(table, hide_index=True, width="stretch", column_config=signed)
    st.markdown("#### Behaviour and profit by street")
    against = _against_selector(page, "street_vs")
    per_street = page.results.per_street_table(against)
    by_end = page.results.profit_by_end_table(against)
    per_street = per_street[per_street["agent"].isin(page.agents)]
    by_end = by_end[by_end["agent"].isin(page.agents)]
    if per_street.empty:
        st.warning("No data for this selection.")
        return
    st.plotly_chart(street_actions_figure(per_street), key=page.ns.key("street_fig"), theme=None, width="stretch")
    st.caption("Share of each agent's decisions on every street that were folds, checks/calls and bets/raises.")
    st.plotly_chart(profit_by_end_figure(by_end), key=page.ns.key("end_fig"), theme=None, width="stretch")
    st.caption(
        "Where the win rate comes from: hands that ended with a fold on a street versus hands that went to "
        "showdown. The bars of one agent add up to its BB/100."
    )
    with st.expander("Tables"):
        st.dataframe(per_street, hide_index=True, width="stretch")
        st.dataframe(by_end, hide_index=True, width="stretch")


# --------------------------------------------------------------------------------- models
def _skill_lines(page: Page, kind: str) -> list[str]:
    kinds = page.results.agents().set_index("agent")["kind"]
    board = page.view.leaderboard
    return [
        f"**{r.agent}**: rank {r.rank} of {len(board)}, {r.bb_per_100:+.1f} BB/100 (CI {format_ci(r.ci_low, r.ci_high)}) - "
        + ("significantly different from 0" if r.significant else "not distinguishable from 0")
        for r in board.itertuples()
        if kinds.get(r.agent) == kind
    ]


def _render_model_card(page: Page, card: ModelCard) -> None:
    with st.container(border=True):
        st.markdown(f"#### {card.title}")
        skill = _skill_lines(page, card.kind)
        skill_text = "\n\n".join(skill) if skill else "_This model did not play in the selected run._"
        if not card.trained:
            st.markdown(f"**Not trained yet.** {card.note}")
            st.markdown(f"Tournament skill (selected run): {skill_text}")
            return
        left, right = st.columns(2)
        with left:
            st.markdown("**Training summary**")
            rows = [FeatureRow("Model", k, v) for k, v in card.facts]
            if card.created_at:
                rows.insert(0, FeatureRow("Model", "Trained", card.created_at))
            rows += [FeatureRow("Training", k, v) for k, v in card.training]
            st.html(render_features_html(rows))
        with right:
            st.markdown("**Imitation quality** (copying the teacher on held-out decisions)")
            if card.imitation:
                st.html(render_features_html([FeatureRow("Imitation", k, v) for k, v in card.imitation]))
            else:
                st.caption("Not applicable or not stored for this model.")
            st.markdown("**Tournament skill** (selected run, BB/100 with interval)")
            st.markdown(skill_text)
        if card.top_features:
            fig = importance_figure(card.top_features)
            st.plotly_chart(fig, key=page.ns.key(f"imp_{card.kind}"), theme=None, width="stretch")
        if card.rules:
            st.markdown("**Evolved strategy in words**\n" + "\n".join(f"- {r}" for r in card.rules))
        if card.genes:
            fig = gene_figure(card.genes)
            st.plotly_chart(fig, key=page.ns.key(f"genes_{card.kind}"), theme=None, width="stretch")
            with st.expander("Gene table"):
                st.dataframe(pd.DataFrame(card.genes), hide_index=True, width="stretch")
        with st.expander("Raw metadata"):
            st.json(card.raw, expanded=False)


def _tab_models(page: Page) -> None:
    st.info(SKILL_NOTE)
    for card in load_model_cards(models_root()):
        _render_model_card(page, card)
    st.caption(
        "Rule-based agents (Random, Call Station, Maniac, Equity Teacher) have no trained parameters. Model files "
        "are read from `artifacts/models/<ebm|xgb|ga>/metadata.json`."
    )


# ---------------------------------------------------------------------------- latency / hands
def _tab_latency(page: Page) -> None:
    latency = page.results.latency_table()
    latency = latency[latency["agent"].isin(page.agents)]
    st.plotly_chart(latency_figure(latency), key=page.ns.key("latency_fig"), theme=None, width="stretch")
    names = {"agent": "Agent", "n": "Decisions", "mean_ms": "Mean (ms)", "median_ms": "Median (ms)",
             "p95_ms": "95th pct (ms)", "max_ms": "Max (ms)"}  # fmt: skip
    st.dataframe(latency.rename(columns=names), hide_index=True, width="stretch")
    st.caption(
        "Wall-clock time per decision measured during the run: it varies with the machine and its load, and the "
        "maximum includes first-call warm-up."
    )


def _hand_label(index: int, info: pd.Series) -> str:
    tags = ", ".join(HAND_TAGS.get(t, t) for t in str(info["tags"]).split(",") if t)
    return (
        f"#{index + 1}: {info['agent_a']} vs {info['agent_b']} - hand {info['hand_index']} - "
        f"pot {info['pot_bb']:.0f} BB ({tags})"
    )


def _on_step(step_key: str, delta: int, last: int) -> None:
    st.session_state[step_key] = max(0, min(last, st.session_state.get(step_key, 0) + delta))


def _render_replay(page: Page, frames: list[ReplayFrame], key: str) -> None:
    last = len(frames) - 1
    step_key = page.ns.key(f"step_{key}")
    st.session_state[step_key] = min(st.session_state.get(step_key, 0), last)
    prev_col, slider_col, next_col = st.columns([1, 6, 1], vertical_alignment="center")
    prev_col.button("◀", on_click=_on_step, args=(step_key, -1, last), key=page.ns.key(f"prev_{key}"),
                    help="Previous action", width="stretch")  # fmt: skip
    next_col.button("▶", on_click=_on_step, args=(step_key, 1, last), key=page.ns.key(f"next_{key}"),
                    help="Next action", width="stretch")  # fmt: skip
    with slider_col:
        step = st.slider("Action", 0, last, key=step_key, help="0 = cards dealt; the last step shows the result.")
    frame = frames[step]
    st.html(render_table_html(frame.table))
    st.markdown(f"**Step {step} of {last}:** {frame.caption}")
    st.html(render_log_html(frame.log))


def _tab_hands(page: Page) -> None:
    results = page.results
    col_tag, col_pair = st.columns(2)
    tag = col_tag.selectbox(
        "Kind of hand", [None, *HAND_TAGS], format_func=lambda t: "All stored hands" if t is None else HAND_TAGS[t],
        key=page.ns.key("hand_tag"),
    )  # fmt: skip
    pairings = results.pairings()
    pair_labels = [f"{a} vs {b}" for a, b in pairings]
    pairing = col_pair.selectbox("Pairing", ["All pairings", *pair_labels], key=page.ns.key("hand_pair"))
    a, b = (None, None) if pairing == "All pairings" else pairings[pair_labels.index(pairing)]
    try:
        info = results.sample_hands_info(tag, a, b)
    except FileNotFoundError:
        st.info("This run did not store sample hands.")
        return
    if info.empty:
        st.info("No stored hands match this filter.")
        return
    records = results.sample_hands(tag, a, b)
    scope = f"{tag}_{pairing}"
    choice = st.selectbox(
        "Hand", range(len(info)), format_func=lambda i: _hand_label(i, info.iloc[i]), key=page.ns.key(f"hand_{scope}")
    )
    st.caption(
        "Post-hoc replay: a spectator view of a finished hand, so both players' cards are shown - the agents never "
        "saw them. Only a sample is stored (the first hands and the extremes of every match)."
    )
    _render_replay(page, replay_frames(records[choice]), f"{scope}_{choice}")


# --------------------------------------------------------------------------------- sidebar
@dataclass(frozen=True)
class Selection:
    run_dir: str
    level: float
    agents: tuple[str, ...]
    metric: str


def _sidebar(ns: SessionNamespace, runs: pd.DataFrame) -> Selection:
    """Run selector and display settings."""
    labels = {row.run_id: run_label(row) for _, row in runs.iterrows()}
    paths = dict(zip(runs["run_id"], runs["path"], strict=True))
    latest = runs.loc[runs["is_latest"], "run_id"]
    with st.sidebar:
        st.header("Championship")
        run_id = st.selectbox(
            "Tournament run", list(labels), index=list(labels).index(latest.iloc[0]) if len(latest) else 0,
            format_func=labels.get, key=ns.key("run"),
            help="Newest first. The run marked 'latest' is the official one.",
        )  # fmt: skip
        results = _open_or_stop(paths[run_id])
        stored = results.confidence
        levels = sorted({*CI_LEVELS, stored})
        level = st.selectbox(
            "Interval level", levels, index=levels.index(stored), key=ns.key("level"),
            format_func=lambda c: f"{c * PCT:g}%" + (" (as stored)" if c == stored else ""),
            help="Other levels are re-estimated from the saved per-session profits (cheap, no new simulation).",
        )  # fmt: skip
        agents = st.multiselect(
            "Agents shown", results.agent_names, default=results.agent_names, key=ns.key(f"agents_{run_id}"),
            help="Leaderboard and matrix are recomputed among the selected agents only.",
        )  # fmt: skip
        metric = st.selectbox(
            "Matrix metric", [m.key for m in MATRIX_METRICS], format_func=lambda k: METRIC_BY_KEY[k].label,
            key=ns.key("metric"),
        )  # fmt: skip
        st.caption(f"{len(runs)} run(s) in `{tournaments_root()}`")
    return Selection(paths[run_id], float(level), tuple(agents), metric)


def render_championship_page() -> None:
    configure_page("AI Championship", "🏆")
    ns = SessionNamespace(NAMESPACE)
    st.title("AI Championship")
    st.caption(
        "Tournament statistics from saved runs: win rates with confidence intervals, head-to-head verdicts, profit "
        "curves, playing style and replays."
    )
    root = tournaments_root()
    try:
        runs = list_runs(root)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not read the tournament runs in `{root}`: {exc}")
        return
    if runs.empty:
        _render_empty(ns, root)
        return

    selection = _sidebar(ns, runs)
    results = _open_or_stop(selection.run_dir)
    _render_run_header(results)
    _render_launcher(ns, float(results.meta.get("hands_per_second") or 0) or None, expanded=False)
    signature = run_signature(selection.run_dir)
    view = cached_interval_view(selection.run_dir, signature, selection.agents, selection.level)
    page = Page(results, view, selection.level, selection.metric, ns)
    renderers = (
        _tab_leaderboard, _tab_head_to_head, _tab_curves, _tab_style, _tab_opponent_street, _tab_models,
        _tab_latency, _tab_hands,
    )  # fmt: skip
    for tab, render in zip(st.tabs(list(TAB_NAMES)), renderers, strict=True):
        with tab:
            render(page)
