"""AI Championship dashboard: pure builders on a small synthetic tournament + an AppTest of the page.

The tournament is really simulated (three baseline agents, a few hundred hands, no trained models
needed) into a temporary folder, so ``artifacts/tournaments/latest.json`` is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from poker_arena.agents.registry import AgentSpec
from poker_arena.config import TOURNAMENTS_DIR, GameConfig
from poker_arena.tournament.config import KeepHandsPolicy, MatchConfig
from poker_arena.tournament.results import TournamentResults, list_runs, load_results
from poker_arena.tournament.tournament import run_tournament
from poker_arena.ui.replay import REPLAY_BADGE, replay_frames
from poker_arena.ui.table import render_table_html
from poker_arena.ui.tournament_data import (
    ENV_MODELS_DIR,
    ENV_TOURNAMENTS_DIR,
    LaunchSpec,
    estimate_duration_seconds,
    format_ci,
    format_duration,
    interval_view,
    launch_command,
    leaderboard_display,
    load_model_card,
    load_model_cards,
    matrix_from_pairwise,
    models_root,
    reproduce_command,
    run_facts,
    tournaments_root,
    validate_launch,
)
from poker_arena.ui.tournament_figures import (
    agent_colors,
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

APP_PAGE = Path(__file__).resolve().parents[1] / "app" / "pages" / "2_AI_Championship.py"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_tournament.py"
AGENTS = ["maniac", "call_station", "random"]
C = "pa.champ."
TIMEOUT_SECONDS = 120


@pytest.fixture(scope="module")
def runs_root(tmp_path_factory) -> Path:
    """A real (tiny) tournament written to a temporary tournaments folder."""
    root = tmp_path_factory.mktemp("tournaments")
    cfg = MatchConfig(
        game=GameConfig(),
        num_hands=400,
        seed=7,
        session_hands=40,  # 10 sessions: enough for session-level intervals (the default unit)
        keep_hands=KeepHandsPolicy(first=3, biggest_pots=2, biggest_wins_each=2),
    )
    run_tournament([AgentSpec(k) for k in AGENTS], cfg, run_name="unit", out_dir=root, mark_latest=True)
    return root


@pytest.fixture(scope="module")
def legacy_protocol_root(tmp_path_factory) -> Path:
    """A run made with the legacy single-table protocol: shown, but with a loud warning."""
    root = tmp_path_factory.mktemp("legacy_protocol")
    cfg = MatchConfig(game=GameConfig(), num_hands=80, seed=7, session_hands=10, protocol="single_table",
                      keep_hands=KeepHandsPolicy(0, 0, 0))  # fmt: skip
    run_tournament([AgentSpec(k) for k in AGENTS], cfg, run_name="legacy", out_dir=root, mark_latest=True)
    return root


@pytest.fixture(scope="module")
def few_sessions_root(tmp_path_factory) -> Path:
    """A run with only 3 sessions per pairing: its intervals fall back to pairs and must say so."""
    root = tmp_path_factory.mktemp("few_sessions")
    cfg = MatchConfig(game=GameConfig(), num_hands=120, seed=7, session_hands=40, keep_hands=KeepHandsPolicy(0, 0, 0))
    run_tournament([AgentSpec(k) for k in AGENTS], cfg, run_name="few", out_dir=root, mark_latest=True)
    return root


@pytest.fixture(scope="module")
def results(runs_root: Path) -> TournamentResults:
    return load_results("latest", runs_root)


@pytest.fixture
def point_at(monkeypatch, runs_root: Path):
    monkeypatch.setenv(ENV_TOURNAMENTS_DIR, str(runs_root))


# --------------------------------------------------------------------------------- roots
def test_roots_follow_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_TOURNAMENTS_DIR, raising=False)
    monkeypatch.delenv(ENV_MODELS_DIR, raising=False)
    assert tournaments_root() == TOURNAMENTS_DIR
    monkeypatch.setenv(ENV_TOURNAMENTS_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path / "m"))
    assert tournaments_root() == tmp_path and models_root() == tmp_path / "m"


# ------------------------------------------------------------------------------ run facts
def test_run_facts_and_reproduce_command(results):
    facts = dict(run_facts(results))
    assert facts["Hands per pairing"] == "400" and facts["Total hands"] == "1,200" and facts["Seed"] == "7"
    assert facts["Stack"].startswith("1,000") and facts["Blinds"] == "5/10" and facts["Agents"] == "3"
    assert "duplicate" in facts["Dealing"] and facts["Interval"].startswith("95%") and facts["Date"][:2] == "20"
    command = reproduce_command(results)
    assert command.startswith(".venv/Scripts/python scripts/run_tournament.py --agents ")
    assert "--set RANDOM_SEED=7" in command and "--set DUPLICATE_DEALS=true" in command
    assert "--set CI_UNIT=session" in command and "--set PROTOCOL=two_table" in command
    assert facts["Protocol"] == "two-table duplicate" and results.protocol_note is None
    assert facts["Statistical unit"] == "sessions of 40 hands, 20 duplicate pairs each"


# ------------------------------------------------------------------------------- launcher
def test_launch_command_is_exact_and_accepted_by_the_script(monkeypatch):
    spec = LaunchSpec(("ebm", "random", "maniac"), hands=5000, seed=3, starting_stack=2000, small_blind=10,
                      big_blind=20, workers=2, duplicate=False, ci_method="bootstrap", confidence=0.9, run_name="try1")
    command = launch_command(spec)
    assert command.startswith(".venv/Scripts/python scripts/run_tournament.py --agents ebm,random,maniac ")
    for part in ("NUM_TOURNAMENT_HANDS=5000", "RANDOM_SEED=3", "STARTING_STACK=2000", "SMALL_BLIND=10",
                 "BIG_BLIND=20", "DUPLICATE_DEALS=false", "CI_METHOD=bootstrap", "CONFIDENCE_LEVEL=0.9",
                 "N_WORKERS=2", "MARK_LATEST=false", "RUN_NAME=try1"):
        assert f"--set {part}" in command

    # the real script must parse it and see exactly these values
    script_spec = importlib.util.spec_from_file_location("run_tournament_script", SCRIPT)
    module = importlib.util.module_from_spec(script_spec)
    sys.modules["run_tournament_script"] = module
    try:
        script_spec.loader.exec_module(module)
        argv = shlex.split(command)[2:]
        effective = module.apply_cli_overrides(module.__dict__, module.SMOKE_OVERRIDES, argv)
        monkeypatch.setattr(sys, "argv", ["run_tournament.py", *argv])
        parsed = module.parse_args()
    finally:
        sys.modules.pop("run_tournament_script", None)
    assert parsed.agents == "ebm,random,maniac"
    assert (effective["NUM_TOURNAMENT_HANDS"], effective["RANDOM_SEED"], effective["STARTING_STACK"]) == (5000, 3, 2000)
    assert (effective["SMALL_BLIND"], effective["BIG_BLIND"], effective["N_WORKERS"]) == (10, 20, 2)
    assert effective["DUPLICATE_DEALS"] is False and effective["CI_METHOD"] == "bootstrap"
    assert effective["CONFIDENCE_LEVEL"] == 0.9 and effective["MARK_LATEST"] is False and effective["RUN_NAME"] == "try1"


def test_launch_validation():
    base = LaunchSpec(("random", "maniac"))
    assert validate_launch(base) is None
    assert "two agents" in validate_launch(LaunchSpec(("random",)))
    assert validate_launch(LaunchSpec(("random", "maniac"), small_blind=20, big_blind=10))
    assert validate_launch(LaunchSpec(("random", "maniac"), starting_stack=10))
    assert validate_launch(LaunchSpec(("random", "maniac"), run_name="2026"))  # would be parsed as a number
    assert validate_launch(LaunchSpec(("random", "maniac"), run_name="bad name; rm"))
    with pytest.raises(ValueError):
        launch_command(LaunchSpec(("random",)))


def test_duration_estimate():
    spec = LaunchSpec(("a", "b", "c"), hands=1000)
    assert estimate_duration_seconds(spec, 300.0) == pytest.approx(10.0)
    assert estimate_duration_seconds(spec, None) is None and estimate_duration_seconds(spec, 0) is None
    assert format_duration(30) == "30 s" and format_duration(600) == "10 min" and format_duration(7200) == "2.0 h"


# --------------------------------------------------------------------------- interval view
def test_stored_view_is_the_saved_leaderboard(results):
    view = interval_view(results, results.agent_names, results.confidence)
    assert not view.recomputed
    assert view.leaderboard.equals(results.leaderboard())
    assert view.order == list(results.leaderboard()["agent"])


def test_other_level_widens_the_intervals(results):
    narrow = interval_view(results, results.agent_names, 0.80)
    wide = interval_view(results, results.agent_names, 0.99)
    assert narrow.recomputed and wide.recomputed and wide.level == 0.99
    stored = results.leaderboard().set_index("agent")
    for a in results.agent_names:
        n, w = narrow.leaderboard.set_index("agent").loc[a], wide.leaderboard.set_index("agent").loc[a]
        assert n["bb_per_100"] == pytest.approx(stored.loc[a, "bb_per_100"])  # same point estimate
        assert (w["ci_high"] - w["ci_low"]) > (n["ci_high"] - n["ci_low"])
    half = lambda row: (row["ci_high"] - row["ci_low"]) / 2  # noqa: E731
    widths = {lvl: half(interval_view(results, results.agent_names, lvl).pairwise.iloc[0]) for lvl in (0.8, 0.9, 0.99)}
    assert widths[0.8] < widths[0.9] < widths[0.99]


def test_other_levels_are_re_estimated_over_sessions(results):
    """The dashboard re-views the same independent unit as the stored intervals: the 10 session values."""
    from scipy import stats

    view = interval_view(results, results.agent_names, 0.90)
    a, b = results.pairings()[0]
    row = view.pairwise.set_index(["agent", "opponent"]).loc[(a, b)]
    sessions = results.session_profits(a, b)
    se = sessions.std(ddof=1) / len(sessions) ** 0.5
    assert row["se"] == pytest.approx(se) and (row["ci_high"] - row["ci_low"]) / 2 == pytest.approx(stats.t.ppf(0.95, 9) * se)
    pairs = results.ci_samples(a, b, "pair")
    pair_se = pairs.std(ddof=1) / len(pairs) ** 0.5
    assert row["pair_se"] == pytest.approx(pair_se)  # the pair-level extras follow the displayed level, too
    assert (row["pair_ci_high"] - row["pair_ci_low"]) / 2 == pytest.approx(stats.t.ppf(0.95, len(pairs) - 1) * pair_se)


def test_recomputation_at_the_stored_level_reproduces_the_stored_interval(results):
    """Two agents at the stored level: the leaderboard of the pair equals the stored pairwise row."""
    a, b = results.pairings()[0]
    view = interval_view(results, [a, b], results.confidence)
    assert view.recomputed and len(view.leaderboard) == 2
    stored = results.pairwise_table().set_index(["agent", "opponent"]).loc[(a, b)]
    mine = view.leaderboard.set_index("agent").loc[a]
    for column in ("bb_per_100", "se", "ci_low", "ci_high"):
        assert mine[column] == pytest.approx(stored[column], rel=1e-9)
    assert view.verdict_codes.loc[a, b] == stored["verdict"]
    assert view.verdict_texts.loc[a, b] == stored["verdict_text"]
    assert set(view.pairwise["agent"]) == {a, b}


def test_subset_view_is_consistent(results):
    subset = [results.agent_names[0], results.agent_names[2]]
    view = interval_view(results, subset, 0.90)
    assert sorted(view.order) == sorted(subset) and list(view.leaderboard["rank"]) == [1, 2]
    assert view.leaderboard["bb_per_100"].iloc[0] >= view.leaderboard["bb_per_100"].iloc[1]
    assert math.isclose(view.leaderboard["bb_per_100"].sum(), 0.0, abs_tol=1e-6)  # zero-sum between two agents
    assert list(view.verdict_codes.index) == view.order


# ----------------------------------------------------------------------------- tables
def test_leaderboard_display_and_matrix(results):
    board = results.leaderboard()
    table = leaderboard_display(board, 0.95)
    assert list(table.columns)[:4] == ["Rank", "Agent", "BB/100", "95% CI"]
    assert table["95% CI"].iloc[0].startswith("[") and "W / D / L" in table.columns
    assert set(table["Significant"]) <= {"yes, positive", "yes, negative", "no (CI includes 0)"}
    matrix = matrix_from_pairwise(results.pairwise_table(), "bb_per_100", results.leaderboard()["agent"])
    assert matrix.shape == (3, 3) and np.isnan(np.diag(matrix.to_numpy())).all()
    assert np.allclose(matrix.to_numpy()[~np.eye(3, dtype=bool)], -matrix.to_numpy().T[~np.eye(3, dtype=bool)])
    assert format_ci(-12.4, 30.6) == "[-12, +31]" and format_ci(math.nan, 1) == "n/a"


# ---------------------------------------------------------------------------- figures
def test_forest_figure(results):
    board = results.leaderboard()
    fig = forest_figure(board, 0.95)
    plotted = sum(len(trace.x) for trace in fig.data)
    assert plotted == len(board)
    for trace in fig.data:
        assert len(trace.error_x.array) == len(trace.x) and (np.asarray(trace.error_x.array) >= 0).all()
    assert any(shape.type == "line" for shape in fig.layout.shapes)  # the zero line


def test_heatmap_marks_significance_and_hides_the_diagonal(results):
    view = interval_view(results, results.agent_names, results.confidence)
    order = view.order
    inputs = {
        "values": matrix_from_pairwise(view.pairwise, "bb_per_100", order),
        "ci_low": matrix_from_pairwise(view.pairwise, "ci_low", order),
        "ci_high": matrix_from_pairwise(view.pairwise, "ci_high", order),
        "significant": matrix_from_pairwise(view.pairwise, "significant", order).fillna(False),
    }
    values = inputs.pop("values")
    fig = matchup_heatmap(values, "bb_per_100", verdicts=view.verdict_texts, level=0.95, **inputs)
    heat = fig.data[0]
    z = np.asarray(heat.z, dtype=float)
    assert z.shape == (3, 3) and np.isnan(np.diag(z)).all()
    assert heat.zmin == -heat.zmax and heat.zmid == 0
    cells = [t for row in heat.text for t in row if t]
    assert len(cells) == 6 and all("±" in t for t in cells)
    significant = int(inputs["significant"].to_numpy(dtype=bool).sum())
    assert sum("★" in t for t in cells) == significant
    other = matchup_heatmap(results.matchup_matrix("win_rate").reindex(index=order, columns=order), "win_rate")
    assert other.data[0].zmin == 0 and other.data[0].zmax == 1


def test_verdict_table_html(results):
    codes, texts = results.verdict_matrix(), results.verdict_matrix(text=True)
    html = verdict_table_html(codes, texts)
    assert html.count("<tr>") == 1 + len(codes) and "pa-v-self" in html
    assert "significant" in html and ("pa-v-win" in html or "pa-v-draw" in html)
    evil = codes.copy()
    evil.iloc[0, 1] = "win"
    hostile = texts.copy()
    hostile.iloc[0, 1] = "<img src=x onerror=alert(1)>"
    assert "<img" not in verdict_table_html(evil, hostile)


def test_curves(results):
    a, b = results.pairings()[0]
    frame = results.cumulative_profit(a, b, 200)
    single = cumulative_figure({f"{a} vs {b}": frame}, luck_band=True, level=0.95)
    assert len(single.data) == 3 and single.data[1].fill == "tonexty"  # band (2 traces) + curve
    assert single.data[-1].name == f"{a} vs {b}"
    many = cumulative_figure({"x": frame, "y": frame}, luck_band=True, level=0.95)
    assert len(many.data) == 2  # no band when several curves are compared
    assert len(frame) <= 200 and frame["hand"].iloc[-1] == 400


def test_style_and_street_figures(results):
    style = results.style_table().fillna(0.0)
    radar = style_radar(style)
    assert len(radar.data) == 3 and all(t.r[0] == t.r[-1] for t in radar.data)  # closed polygons
    assert len(style_bars(style).data) == 3
    streets = street_actions_figure(results.per_street_table())
    assert len({t.xaxis for t in streets.data}) == 3  # fold / call / raise panels
    ends = results.profit_by_end_table()
    figure = profit_by_end_figure(ends)
    for trace, agent in zip(figure.data, ends["agent"].drop_duplicates(), strict=True):
        assert sum(trace.y) == pytest.approx(ends[ends["agent"] == agent]["bb_per_100_contribution"].sum())


def test_opponent_and_latency_figures(results):
    pairwise = results.pairwise_table()
    rows = pairwise[pairwise["agent"] == results.agent_names[0]]
    fig = opponent_figure(rows, results.agent_names[0], 0.95)
    assert len(fig.data[0].x) == 2
    latency = latency_figure(results.latency_table())
    assert latency.layout.yaxis.type == "log" and len(latency.data) == 3


def test_agent_colors_are_stable():
    assert agent_colors(["a", "b"]) == agent_colors(["a", "b"]) and len(set(agent_colors(list("abcde")).values())) == 5


# ---------------------------------------------------------------------------------- replay
def test_replay_frames_of_a_stored_hand(results):
    record = results.sample_hands()[0]
    frames = replay_frames(record)
    assert len(frames) == len(record.events) + 1
    assert frames[0].caption == "Cards dealt, blinds posted" and len(frames[0].log) >= 2
    assert all(len(b.log) >= len(a.log) for a, b in zip(frames, frames[1:]))
    for frame in frames:
        html = render_table_html(frame.table)
        assert REPLAY_BADGE in html and 'aria-label="Face-down card"' not in html  # post-hoc view shows both hands
    last = frames[-1]
    assert last.table.result_text and last.log[-1].kind == "result"
    assert any(e.kind == "action" for e in last.log)


def test_every_stored_hand_replays_to_its_recorded_result(results):
    for record in results.sample_hands()[:12]:
        assert record.verify()
        assert len(replay_frames(record)) == len(record.events) + 1


# ------------------------------------------------------------------------------ model cards
def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_missing_or_broken_models_are_reported_not_trained(tmp_path):
    cards = load_model_cards(tmp_path)
    assert [c.kind for c in cards] == ["ebm", "xgboost", "ga"] and not any(c.trained for c in cards)
    assert "train_ebm" in cards[0].note and "train_xgboost" in cards[1].note
    (tmp_path / "ebm").mkdir()
    (tmp_path / "ebm" / "metadata.json").write_text("{not json", encoding="utf-8")
    assert not load_model_card("ebm", tmp_path).trained
    _write_json(tmp_path / "xgb" / "metadata.json", {})  # an empty but valid file is still "trained"
    assert load_model_card("xgboost", tmp_path).trained


def test_model_card_reads_imitation_metrics_and_importances(tmp_path):
    _write_json(
        tmp_path / "ebm" / "metadata.json",
        {
            "schema_version": 1, "created_at": "2026-01-02T03:04:05+00:00", "feature_names": ["a", "b", "c"],
            "seeds": {"train": 1}, "training": {"n_train_decisions": 12345, "interactions": 5},
            "validation_metrics": {"accuracy": 0.9, "log_loss": 0.3, "note": "x"},
            "global_importance": {"equity": 0.7, "pot_odds": -0.9, "spr": 0.1},
        },
    )
    card = load_model_card("ebm", tmp_path)
    assert card.trained and card.created_at == "2026-01-02 03:04"
    facts = dict(card.facts)
    assert facts["Training examples"] == "12,345" and facts["Input features"] == "3"
    assert dict(card.imitation) == {"accuracy": "0.9", "log loss": "0.3"}  # accuracy-like numbers only
    assert ("interactions", "5") in card.training
    assert [n for n, _ in card.top_features] == ["pot_odds", "equity", "spr"]  # by absolute value
    assert importance_figure(card.top_features).data[0].orientation == "h"


def test_model_card_reads_the_ga_strategy(tmp_path):
    _write_json(
        tmp_path / "ga" / "metadata.json",
        {"created_at": "2026-01-02T03:04:05+00:00",
         "gene_specs": [{"name": "bluff", "low": 0.0, "high": 1.0, "group": "post", "description": "bluff rate"}],
         "training": {"generations": 15, "training_opponents": ["random"]}},
    )
    _write_json(tmp_path / "ga" / "best_genome.json",
                {"genes": {"bluff": 0.25}, "strategy": ["Bluffs 25% of weak hands."], "final_selection_fitness_bb100": 12.5})
    card = load_model_card("ga", tmp_path)
    assert card.rules == ("Bluffs 25% of weak hands.",)
    assert card.genes[0]["gene"] == "bluff" and card.genes[0]["high"] == 1.0
    assert ("generations", "15") in card.training and any("fitness" in k for k, _ in card.training)
    assert not card.imitation  # evolved through play, not imitation
    fig = gene_figure(card.genes)
    assert fig.data[0].x[0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------------- AppTest
def open_page() -> AppTest:
    return AppTest.from_file(str(APP_PAGE), default_timeout=TIMEOUT_SECONDS).run()


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_page_without_runs_shows_the_command(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_TOURNAMENTS_DIR, str(tmp_path))
    at = open_page()
    assert_clean(at)
    assert any("No tournament results" in i.value for i in at.info)
    assert any("scripts/run_tournament.py" in c.value for c in at.code)
    assert any("--agents" in c.value for c in at.code)  # the command builder is open and filled in


def test_page_renders_every_section(point_at, runs_root):
    at = open_page()
    assert_clean(at)
    assert [t.label for t in at.tabs] == [
        "Leaderboard", "Head-to-head", "Profit curves", "Playing style", "By opponent & street", "Models",
        "Latency", "Interesting hands",
    ]
    run_id = list_runs(runs_root)["run_id"][0]
    assert at.selectbox(key=C + "run").value == run_id  # the run marked latest is preselected
    html = "".join(str(e.value) for e in at.get("html"))
    assert "Hands per pairing" in html and "pa-vtable" in html and "pa-card" in html
    code = " ".join(c.value for c in at.code)
    assert "--agents" in code and "MARK_LATEST=false" in code  # command builder + reproduce command
    assert any("Imitation accuracy is not poker skill" in i.value for i in at.info)


def test_page_interval_level_and_agent_subset(point_at):
    at = open_page()
    at.selectbox(key=C + "level").select(0.99).run()
    assert_clean(at)
    assert any("re-estimated at 99%" in c.value for c in at.caption)
    at.selectbox(key=C + "level").select(0.95).run()
    assert_clean(at)
    assert not any("re-estimated" in c.value for c in at.caption)

    run_id = at.selectbox(key=C + "run").value
    agents = at.multiselect(key=f"{C}agents_{run_id}")
    agents.unselect(agents.options[0]).run()
    assert_clean(at)
    assert any("re-estimated" in c.value for c in at.caption)  # a subset forces the re-view
    agents = at.multiselect(key=f"{C}agents_{run_id}")
    agents.unselect(agents.options[1]).run()  # only one agent left
    assert_clean(at)
    assert any("Select at least two agents" in w.value for w in at.warning)


def test_page_matrix_metric_and_pair_details(point_at):
    at = open_page()
    at.selectbox(key=C + "metric").select("win_rate").run()
    assert_clean(at)
    at.selectbox(key=C + "pair_a").select_index(1).run()
    assert_clean(at)
    assert at.selectbox(key=C + "pair_a").value != at.selectbox(key=C + "pair_b").value
    assert at.metric  # BB/100, half-width, win rate, showdown win rate


def test_page_curves_style_street_and_latency_controls(point_at):
    at = open_page()
    at.checkbox(key=C + "curve_all").check().run()
    assert_clean(at)
    at.radio(key=C + "style_view").set_value("Grouped bars").run()
    assert_clean(at)
    at.selectbox(key=C + "style_vs").select_index(1).run()
    assert_clean(at)
    at.selectbox(key=C + "street_vs").select_index(1).run()
    assert_clean(at)
    at.selectbox(key=C + "opp_agent").select_index(2).run()
    assert_clean(at)


def test_page_hand_replay_steps(point_at):
    at = open_page()
    hand_key = next(s.key for s in at.selectbox if s.key and s.key.startswith(C + "hand_None_"))
    assert at.selectbox(key=C + "hand_tag").options[0] == "All stored hands"
    step_key = next(s.key for s in at.slider if s.key and s.key.startswith(C + "step_"))
    assert at.slider(key=step_key).value == 0
    at.slider(key=step_key).set_value(2).run()
    assert_clean(at)
    assert any("Step 2 of" in m.value for m in at.markdown)
    at.selectbox(key=C + "hand_tag").select("biggest_pot").run()
    assert_clean(at)
    assert hand_key not in {s.key for s in at.selectbox}  # the hand picker follows the filter


def test_page_shows_trained_models(point_at, monkeypatch, tmp_path):
    _write_json(tmp_path / "ebm" / "metadata.json",
                {"training": {"n_train_decisions": 999}, "validation_metrics": {"accuracy": 0.8}})
    monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path))
    at = open_page()
    assert_clean(at)
    html = "".join(str(e.value) for e in at.get("html"))
    assert "999" in html and "accuracy" in html
    assert any("Not trained yet" in m.value for m in at.markdown)  # xgboost / ga still missing


def test_page_never_modifies_the_tournament_folder(point_at, runs_root):
    before = sorted((p.name, p.stat().st_mtime_ns) for p in runs_root.rglob("*") if p.is_file())
    at = open_page()
    at.selectbox(key=C + "level").select(0.9).run()
    after = sorted((p.name, p.stat().st_mtime_ns) for p in runs_root.rglob("*") if p.is_file())
    assert before == after


def test_fallback_to_pairs_is_flagged_in_the_data_and_on_the_page(monkeypatch, few_sessions_root):
    results = load_results("latest", few_sessions_root)
    assert results.ci_unit == "pair" and results.ci_unit_note
    assert dict(run_facts(results))["Statistical unit"] == "duplicate pairs"
    view = interval_view(results, results.agent_names, 0.9)  # re-estimation uses the unit the run used
    assert view.recomputed and len(view.leaderboard) == 3
    monkeypatch.setenv(ENV_TOURNAMENTS_DIR, str(few_sessions_root))
    at = open_page()
    assert not at.exception and not at.error
    assert any("Only 3 sessions" in w.value for w in at.warning)


def test_legacy_protocol_is_flagged_in_the_data_and_on_the_page(monkeypatch, legacy_protocol_root):
    with pytest.warns(UserWarning, match="second-mover artifact"):
        results = load_results("latest", legacy_protocol_root)
    assert results.protocol == "single_table"
    facts = dict(run_facts(results))
    assert facts["Protocol"].startswith("single-table duplicate (legacy") and "--set PROTOCOL=single_table" in reproduce_command(results)
    monkeypatch.setenv(ENV_TOURNAMENTS_DIR, str(legacy_protocol_root))
    at = open_page()
    assert not at.exception and not at.error
    assert any("second-mover artifact" in w.value for w in at.warning)
    html = "".join(str(e.value) for e in at.get("html"))
    assert "Protocol" in html and "single-table duplicate" in html
