"""Tests for ``run_tournament`` (files, determinism, composition independence) and the results loader."""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poker_arena.agents import registry
from poker_arena.agents.base import Agent, Decision, ModelArtifactError
from poker_arena.agents.baselines import CallStationAgent
from poker_arena.agents.registry import AgentSpec
from poker_arena.engine.history import HandRecord
from poker_arena.tournament import storage
from poker_arena.tournament.config import KeepHandsPolicy, MatchConfig
from poker_arena.tournament.results import TournamentResults, list_runs, load_results
from poker_arena.tournament.stats import scale_to_bb_per_100, t_estimate
from poker_arena.tournament.tournament import run_tournament

KINDS = ["random", "call_station", "maniac"]
LABELS = ["Call Station", "Maniac", "Random"]  # registry names, alphabetical
POLICY = KeepHandsPolicy(first=3, biggest_pots=2, biggest_wins_each=2, all_hands=True)
HANDS, SESSION_HANDS = 160, 20  # 8 sessions of 10 duplicate pairs: the smallest session-level run
CFG = MatchConfig(num_hands=HANDS, session_hands=SESSION_HANDS, seed=5, keep_hands=POLICY)
FEW_SESSIONS_CFG = MatchConfig(num_hands=120, session_hands=40, seed=5)  # 3 sessions: falls back to pairs
MANIAC_HALF = AgentSpec("maniac", {"raise_probability": 0.5}, "Maniac50")
PAIRS = 3  # three agents -> three pairings


def without_latency(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[c for c in frame.columns if "latency" not in c]]


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tournaments")


@pytest.fixture(scope="module")
def run_dir(root: Path) -> Path:
    return run_tournament(KINDS, CFG, run_name="unit test", out_dir=root, extra_config={"origin": "pytest"})


@pytest.fixture(scope="module")
def res(run_dir: Path) -> TournamentResults:
    return load_results(run_dir)


@pytest.fixture(scope="module")
def parallel_res(root: Path, run_dir: Path) -> TournamentResults:
    """Same tournament on two workers, written next to the first (not marked latest)."""
    path = run_tournament(KINDS, replace(CFG, n_workers=2), run_name="parallel", out_dir=root, mark_latest=False)
    return load_results(path)


# ----------------------------------------------------------------------------- files
def test_run_directory_layout(run_dir: Path, root: Path) -> None:
    assert re.fullmatch(r"\d{8}-\d{6}_unit_test", run_dir.name)
    assert {p.name for p in run_dir.iterdir()} == {
        "config.json", "summary.json", "sample_hands.jsonl", "matches", "hands", "full_history",
    }  # fmt: skip
    stems = {"Call_Station__vs__Maniac", "Call_Station__vs__Random", "Maniac__vs__Random"}
    assert {p.stem for p in (run_dir / "matches").glob("*.json")} == stems
    assert {p.stem for p in (run_dir / "hands").glob("*.npz")} == stems
    assert {p.name for p in (run_dir / "full_history").iterdir()} == {f"hands_full_{s}.jsonl.gz" for s in stems}
    assert json.loads((root / "latest.json").read_text())["run_id"] == run_dir.name
    assert not list(root.glob("*.partial")) and not list(root.glob("*.tmp"))


def test_all_json_is_strict_json(run_dir: Path) -> None:
    def refuse(constant: str) -> None:
        raise AssertionError(f"non-standard JSON constant {constant}")

    for path in [*run_dir.glob("*.json"), *run_dir.glob("matches/*.json")]:
        json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)
    for line in (run_dir / "sample_hands.jsonl").read_text().splitlines():
        json.loads(line, parse_constant=refuse)


def test_config_json_records_everything_needed_to_reproduce(res: TournamentResults) -> None:
    config = res.config
    assert MatchConfig.from_dict(config["match_config"]) == CFG
    assert [a["label"] for a in config["agents"]] == ["Random", "Call Station", "Maniac"]  # as given
    assert all({"label", "slug", "spec", "describe"} <= set(a) for a in config["agents"])
    assert config["seeds"]["namespace"] == "eval" and config["seeds"]["master_seed"] == 5
    assert {"python", "platform", "libraries"} <= set(config["environment"])
    assert {"pokerkit", "numpy"} <= set(config["environment"]["libraries"])
    assert config["script_config"] == {"origin": "pytest"} and config["run_name"] == "unit test"


def test_meta(res: TournamentResults) -> None:
    meta = res.meta
    assert meta["run_id"] == res.run_id and meta["total_hands"] == 3 * HANDS
    assert meta["ci_unit"]["used"] == "session" and not meta["recomputed_on_load"]
    assert meta["wall_seconds"] > 0 and meta["hands_per_second"] > 0 and meta["n_workers_used"] == 1
    assert "python" in meta["environment"] and "noisy" in meta["ranking_note"]


# ----------------------------------------------------------------------------- locating runs
def test_load_results_by_latest_id_and_path(res: TournamentResults, run_dir: Path, root: Path) -> None:
    for handle in ("latest", run_dir.name, run_dir, str(run_dir)):
        assert load_results(handle, root=root).run_id == res.run_id
    assert repr(res).startswith("TournamentResults(run_id=")


def test_load_results_errors_are_helpful(root: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Available"):
        load_results("no-such-run", root=root)
    with pytest.raises(FileNotFoundError, match="run_tournament.py"):
        load_results("latest", root=tmp_path)


def test_list_runs_marks_the_latest(res: TournamentResults, parallel_res: TournamentResults, root: Path) -> None:
    runs = list_runs(root)
    assert set(runs["run_id"]) == {res.run_id, parallel_res.run_id}
    latest = runs[runs["is_latest"]]
    assert list(latest["run_id"]) == [res.run_id]  # the parallel run was not marked
    row = runs.set_index("run_id").loc[res.run_id]
    assert (row["num_agents"], row["hands_per_match"], row["total_hands"], row["seed"]) == (3, HANDS, 3 * HANDS, 5)
    assert row["agents"] == "Random, Call Station, Maniac"  # in the order they were given
    assert (runs["ci_unit"] == "session").all()
    assert list_runs(root / "missing").empty


# ----------------------------------------------------------------------------- tables
def test_agents_table(res: TournamentResults) -> None:
    agents = res.agents()
    assert sorted(agents["agent"]) == LABELS
    assert set(agents["kind"]) == set(KINDS)
    assert {"architecture", "description", "supports_explanations", "params"} <= set(agents.columns)


def test_leaderboard_schema_and_invariants(res: TournamentResults) -> None:
    board = res.leaderboard()
    assert list(board["rank"]) == [1, 2, 3] and sorted(board["agent"]) == LABELS
    assert board["bb_per_100"].is_monotonic_decreasing
    assert {"se", "ci_low", "ci_high", "significant", "pooled_bb_per_100", "pooled_se", "pooled_ci_low", "pooled_ci_high",
            "pooled_significant", "wins", "losses", "draws", "insufficient", "hands", "opponents"} <= set(board.columns)  # fmt: skip
    assert (board["ci_low"] <= board["bb_per_100"]).all() and (board["bb_per_100"] <= board["ci_high"]).all()
    assert (board["hands"] == 2 * HANDS).all() and (board["opponents"] == 2).all()
    assert (board["ci_unit"] == "session").all()
    assert {"pair_se", "pair_ci_low", "pair_ci_high", "pair_significant"} <= set(board.columns)
    assert (board["pair_ci_low"] <= board["bb_per_100"]).all() and (board["bb_per_100"] <= board["pair_ci_high"]).all()
    assert (board[["wins", "losses", "draws", "insufficient"]].sum(axis=1) == 2).all()
    assert board["bb_per_100"].sum() == pytest.approx(0.0, abs=1e-9)  # zero-sum game, equal match sizes
    assert board["bb_per_100"].to_numpy() == pytest.approx(board["pooled_bb_per_100"].to_numpy())  # equal sizes
    assert (board["significant"] == ((board["ci_low"] > 0) | (board["ci_high"] < 0))).all()


def test_pairwise_table_matches_a_recomputation_from_the_raw_samples(res: TournamentResults) -> None:
    """Headline columns = t-interval over the 8 session values; pair_* = over the 80 duplicate pairs."""
    table = res.pairwise_table()
    assert len(table) == 2 * PAIRS and (table["unit"] == "pair").all() and (table["units"] == HANDS // 2).all()
    assert (table["ci_unit"] == "session").all() and (table["ci_units"] == 8).all() and (table["n_sessions"] == 8).all()
    for _, row in table.iterrows():
        sessions = res.session_profits(row["agent"], row["opponent"])
        pairs = res.pair_profits(row["agent"], row["opponent"])
        assert len(sessions) == 8 and len(pairs) == HANDS // 2
        est = t_estimate(sessions, res.confidence, unit="session")
        assert row["bb_per_100"] == pytest.approx(est.mean) and row["se"] == pytest.approx(est.se)
        assert (row["ci_low"], row["ci_high"]) == pytest.approx((est.ci_low, est.ci_high))
        pair_est = t_estimate(scale_to_bb_per_100(pairs, 2), res.confidence)
        assert row["bb_per_100"] == pytest.approx(pair_est.mean)  # same point estimate at either level
        assert (row["pair_se"], row["pair_ci_low"], row["pair_ci_high"]) == pytest.approx(
            (pair_est.se, pair_est.ci_low, pair_est.ci_high)
        )
        assert row["session_over_pair_se"] == pytest.approx(est.se / pair_est.se, rel=1e-9)
        assert row["net_bb"] == pytest.approx(pairs.sum())
        mirror = table[(table["agent"] == row["opponent"]) & (table["opponent"] == row["agent"])].iloc[0]
        assert mirror["bb_per_100"] == pytest.approx(-row["bb_per_100"]) and mirror["se"] == pytest.approx(row["se"])
        assert mirror["ci_low"] == pytest.approx(-row["ci_high"])
        assert mirror["pair_ci_low"] == pytest.approx(-row["pair_ci_high"])
        assert 0 < row["se_ratio"] < 2
        assert 0 <= row["win_rate_ci_low"] <= row["win_rate"] <= row["win_rate_ci_high"] <= 1


def test_session_profits_and_ci_samples(res: TournamentResults) -> None:
    sessions = res.session_profits("Maniac", "Random")
    pairs = res.pair_profits("Maniac", "Random")
    assert sessions.shape == (8,) and pairs.shape == (HANDS // 2,)
    assert sessions.mean() == pytest.approx(pairs.sum() * 100 / HANDS)  # BB/100 of the match
    assert np.array_equal(res.session_profits("Random", "Maniac"), -sessions)
    assert np.array_equal(res.ci_samples("Maniac", "Random"), sessions)  # the run's unit is the session
    assert np.allclose(res.ci_samples("Maniac", "Random", "pair"), scale_to_bb_per_100(pairs, 2))
    per_session_hands = SESSION_HANDS // 2  # 10 pairs per session
    assert sessions[0] == pytest.approx(pairs[:per_session_hands].sum() * 100 / SESSION_HANDS)
    assert res.ci_unit == "session" and res.ci_unit_note is None
    with pytest.raises(KeyError):
        res.session_profits("Maniac", "Nobody")


def test_matrix_exposes_session_values_and_pair_extras(res: TournamentResults) -> None:
    order = list(res.leaderboard()["agent"])
    low, pair_low = res.matchup_matrix(part="ci_low"), res.matchup_matrix(part="pair_ci_low")
    table = res.pairwise_table().set_index(["agent", "opponent"])
    for a in order:
        for b in order:
            if a != b:
                assert low.loc[a, b] == pytest.approx(table.loc[(a, b), "ci_low"])
                assert pair_low.loc[a, b] == pytest.approx(table.loc[(a, b), "pair_ci_low"])
                assert res.matchup_matrix(part="pair_significant").loc[a, b] == table.loc[(a, b), "pair_significant"]
    with pytest.raises(ValueError):
        res.matchup_matrix("win_rate", part="pair_significant")


def test_matchup_matrix_and_variants(res: TournamentResults) -> None:
    order = list(res.leaderboard()["agent"])
    matrix = res.matchup_matrix()
    assert list(matrix.index) == order and list(matrix.columns) == order
    assert matrix.isna().to_numpy().diagonal().all() and matrix.isna().sum().sum() == 3
    assert np.allclose(matrix.to_numpy(), -matrix.to_numpy().T, equal_nan=True)  # zero-sum
    low, high = res.matchup_matrix(part="ci_low"), res.matchup_matrix(part="ci_high")
    assert ((low < matrix) | matrix.isna()).all().all() and ((matrix < high) | matrix.isna()).all().all()
    assert (res.matchup_matrix(part="se").fillna(1) > 0).all().all()
    significant = res.matchup_matrix(part="significant")
    assert significant.loc[order[0], order[1]] in (True, False)
    win_low = res.matchup_matrix("win_rate", part="ci_low")
    assert ((win_low <= res.matchup_matrix("win_rate")) | win_low.isna()).all().all()
    assert res.matchup_matrix("net_bb").notna().sum().sum() == 6
    with pytest.raises(KeyError):
        res.matchup_matrix("no_such_metric")
    with pytest.raises(ValueError):
        res.matchup_matrix("win_rate", part="significant")


def test_verdict_matrix_is_consistent_with_the_intervals(res: TournamentResults) -> None:
    verdicts, texts = res.verdict_matrix(), res.verdict_matrix(text=True)
    table = res.pairwise_table().set_index(["agent", "opponent"])
    flip = {"win": "lose", "lose": "win", "draw": "draw", "insufficient": "insufficient"}
    for (agent, opponent), row in table.iterrows():
        code = verdicts.loc[agent, opponent]
        assert code == row["verdict"] and flip[code] == verdicts.loc[opponent, agent]
        assert (code == "win") == (row["ci_low"] > 0) and (code == "lose") == (row["ci_high"] < 0)
        assert texts.loc[agent, opponent] == row["verdict_text"]
        if code == "win":
            assert texts.loc[agent, opponent] == f"{agent} > {opponent} (significant)"
        elif code == "draw":
            assert texts.loc[agent, opponent] == "no significant difference"
    assert all(verdicts.loc[a, a] == "" for a in verdicts.index)


def test_cumulative_profit(res: TournamentResults) -> None:
    curve = res.cumulative_profit("Maniac", "Random")
    assert list(curve.columns) == ["hand", "cum_bb", "luck_low", "luck_high"]
    assert curve["hand"].tolist() == list(range(2, HANDS + 1, 2))
    net = res.pairwise_table().query("agent == 'Maniac' and opponent == 'Random'")["net_bb"].iloc[0]
    assert curve["cum_bb"].iloc[-1] == pytest.approx(net)
    assert np.allclose(curve["luck_low"], -curve["luck_high"], equal_nan=True) and (curve["luck_high"].iloc[1:] > 0).all()
    row = res.pairwise_table().set_index(["agent", "opponent"]).loc[("Maniac", "Random")]
    half_width_in_bb = (row["ci_high"] - row["ci_low"]) / 2 * HANDS / 100  # BB/100 interval -> BB over the match
    assert curve["luck_high"].iloc[-1] == pytest.approx(half_width_in_bb, rel=1e-9)  # band == session-level interval
    reverse = res.cumulative_profit("Random", "Maniac")
    assert np.allclose(reverse["cum_bb"], -curve["cum_bb"])
    thin = res.cumulative_profit("Maniac", "Random", max_points=10)
    assert len(thin) <= 10 and thin["hand"].iloc[-1] == HANDS and thin["cum_bb"].iloc[-1] == pytest.approx(net)
    with pytest.raises(KeyError):
        res.cumulative_profit("Maniac", "Nobody")


def test_style_table_reflects_the_baselines(res: TournamentResults) -> None:
    style = res.style_table().set_index("agent")
    station, maniac = style.loc["Call Station"], style.loc["Maniac"]
    assert station["pfr"] == 0 and station["raise_freq"] == 0 and station["fold_freq"] == 0 and station["call_freq"] == 1
    assert station["aggression_factor"] == 0
    assert maniac["raise_freq"] > 0.5 and maniac["pfr"] > 0.5
    assert (style["hands"] == 2 * HANDS).all()
    for column in ("vpip", "pfr", "win_rate", "showdown_rate", "fold_freq", "call_freq", "raise_freq"):
        assert style[column].between(0, 1).all(), column
    assert np.allclose(style[["fold_freq", "call_freq", "raise_freq"]].sum(axis=1), 1.0)
    versus = res.style_table(vs="Random").set_index("agent")
    assert set(versus.index) == {"Call Station", "Maniac"} and (versus["hands"] == HANDS).all()  # Random never plays itself
    with pytest.raises(KeyError):
        res.style_table(vs="Nobody")


def test_per_street_and_profit_by_end_tables(res: TournamentResults) -> None:
    streets = res.per_street_table()
    assert len(streets) == 12 and set(streets["street"]) == {"preflop", "flop", "turn", "river"}
    assert (streets[["fold", "check", "call", "bet", "raise"]].sum(axis=1) == streets["decisions"]).all()
    ends = res.profit_by_end_table()
    assert len(ends) == 15 and set(ends["ended"]) == {"preflop", "flop", "turn", "river", "showdown"}
    pooled = ends.groupby("agent")["bb_per_100_contribution"].sum()
    board = res.leaderboard().set_index("agent")
    assert pooled.to_numpy() == pytest.approx(board.loc[pooled.index, "pooled_bb_per_100"].to_numpy())
    assert (ends.groupby("agent")["hands"].sum() == 2 * HANDS).all()


def test_latency_table(res: TournamentResults) -> None:
    latency = res.latency_table()
    assert sorted(latency["agent"]) == LABELS and (latency["n"] > 0).all()
    assert (latency["median_ms"] > 0).all() and (latency["p95_ms"] >= latency["median_ms"]).all()
    assert (latency["max_ms"] >= latency["p95_ms"]).all()


# ----------------------------------------------------------------------------- hands
def test_sample_hands_are_replayable_and_filterable(res: TournamentResults) -> None:
    hands = res.sample_hands()
    info = res.sample_hands_info()
    assert len(hands) == len(info) > 0
    assert all(isinstance(h, HandRecord) and h.verify() for h in hands)
    assert set(",".join(info["tags"]).split(",")) <= {"first", "biggest_pot", "biggest_win_a", "biggest_win_b"}
    first = res.sample_hands_info(tag="first")
    assert len(first) == 3 * PAIRS and (first["hand_index"] < 3).all()
    assert len(res.sample_hands("first")) == len(first)
    involving = res.sample_hands_info(a="Maniac")
    assert set(involving["agent_a"]) | set(involving["agent_b"]) >= {"Maniac"}
    assert not ((involving["agent_a"] != "Maniac") & (involving["agent_b"] != "Maniac")).any()
    only_pair = res.sample_hands_info(a="Random", b="Maniac")
    assert set(zip(only_pair["agent_a"], only_pair["agent_b"])) == {("Maniac", "Random")}
    assert res.sample_hands("no_such_tag") == []
    for record, (_, row) in zip(hands, info.iterrows()):
        assert record.result.payoffs[0] / 10 == pytest.approx(row["profit_a_bb"])
        assert record.agent_names == (row["agent_a"], row["agent_b"])
        assert row["pair_index"] == row["hand_index"] // 2


def test_full_history_files(res: TournamentResults) -> None:
    path = res.full_history_path("Random", "Maniac")
    assert path is not None and path.name == "hands_full_Maniac__vs__Random.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert [line["hand_index"] for line in lines] == list(range(HANDS))
    assert all(HandRecord.from_dict(line["record"]).verify() for line in lines[:10])
    assert lines[0]["record"]["deal"] == lines[1]["record"]["deal"]  # a duplicate pair


def test_full_history_is_off_by_default(tmp_path: Path) -> None:
    assert not KeepHandsPolicy().all_hands
    run = run_tournament(KINDS[:2], replace(CFG, num_hands=40, session_hands=20, keep_hands=KeepHandsPolicy()), out_dir=tmp_path)
    assert not (run / storage.FULL_HISTORY_DIR).exists()
    assert load_results(run).full_history_path("Random", "Call Station") is None
    assert len(load_results(run).sample_hands()) > 0  # the default policy keeps a replay sample...
    nothing = run_tournament(KINDS[:2], replace(CFG, num_hands=40, session_hands=20, keep_hands=KeepHandsPolicy(0, 0, 0)), out_dir=tmp_path)
    assert load_results(nothing).sample_hands() == []  # ...and a zero policy keeps none


# ----------------------------------------------------------------------------- reproducibility
def test_identical_results_with_one_or_many_workers(res: TournamentResults, parallel_res: TournamentResults) -> None:
    assert parallel_res.meta["n_workers_used"] == 2 and res.meta["n_workers_used"] == 1
    pd.testing.assert_frame_equal(without_latency(res.pairwise_table()), without_latency(parallel_res.pairwise_table()))
    pd.testing.assert_frame_equal(res.leaderboard(), parallel_res.leaderboard())
    pd.testing.assert_frame_equal(res.verdict_matrix(), parallel_res.verdict_matrix())
    pd.testing.assert_frame_equal(res.sample_hands_info(), parallel_res.sample_hands_info())
    for a, b in res.pairings():
        assert np.array_equal(res.pair_profits(a, b), parallel_res.pair_profits(a, b))


def test_a_pairing_does_not_depend_on_tournament_composition_or_agent_order(res: TournamentResults, tmp_path: Path) -> None:
    bigger = run_tournament([MANIAC_HALF, *reversed(KINDS)], CFG, out_dir=tmp_path)
    other = load_results(bigger)
    assert len(other.pairings()) == 6
    for a, b in res.pairings():
        assert np.array_equal(res.pair_profits(a, b), other.pair_profits(a, b)), (a, b)
    assert other.agent_names[0] == "Maniac50"


def test_rerunning_the_same_tournament_reproduces_the_numbers(res: TournamentResults, tmp_path: Path) -> None:
    again = load_results(run_tournament(KINDS, CFG, out_dir=tmp_path))
    pd.testing.assert_frame_equal(without_latency(res.pairwise_table()), without_latency(again.pairwise_table()))
    changed = load_results(run_tournament(KINDS, replace(CFG, seed=6), out_dir=tmp_path))
    assert not np.array_equal(res.pair_profits("Maniac", "Random"), changed.pair_profits("Maniac", "Random"))


# ----------------------------------------------------------------------------- options
def test_bootstrap_confidence_intervals(tmp_path: Path) -> None:
    cfg = MatchConfig(num_hands=200, session_hands=20, seed=3, ci_method="bootstrap", bootstrap_resamples=300)
    runs = [load_results(run_tournament(["call_station", "maniac"], cfg, out_dir=tmp_path)) for _ in range(2)]
    table = runs[0].pairwise_table()
    assert len(table) == 2 and runs[0].config["match_config"]["ci_method"] == "bootstrap"
    assert (table["ci_low"] < table["bb_per_100"]).all() and (table["bb_per_100"] < table["ci_high"]).all()
    pd.testing.assert_frame_equal(without_latency(table), without_latency(runs[1].pairwise_table()))  # seeded
    assert runs[0].match_summary("Maniac", "Call Station")["players"]["a"]["bb_per_100"]["method"] == "bootstrap"
    assert not runs[0].leaderboard()["ci_low"].isna().any()
    assert (table["ci_unit"] == "session").all() and (table["ci_units"] == 10).all()  # bootstrap over the 10 sessions
    stored = runs[0].match_summary("Maniac", "Call Station")["players"]["a"]
    assert stored["bb_per_100"]["unit"] == "session" and stored["bb_per_100_pair"]["unit"] == "pair"


def test_without_duplicate_deals_the_unit_is_the_hand(tmp_path: Path) -> None:
    cfg = MatchConfig(num_hands=100, session_hands=50, duplicate=False)
    results = load_results(run_tournament(["call_station", "maniac"], cfg, out_dir=tmp_path))
    row = results.pairwise_table().iloc[0]
    assert row["unit"] == "hand" and row["units"] == 100 and math.isclose(row["se_ratio"], row["se_ratio"])
    assert row["ci_unit"] == "hand" and results.ci_unit == "hand" and results.ci_unit_note  # 2 sessions: flagged fallback
    assert len(results.pair_profits("Maniac", "Call Station")) == 100
    assert results.cumulative_profit("Maniac", "Call Station")["hand"].tolist() == list(range(1, 101))


def test_progress_callback(tmp_path: Path) -> None:
    seen = []
    run_tournament(KINDS, replace(CFG, keep_hands=KeepHandsPolicy()), out_dir=tmp_path, progress=seen.append)
    done = [p.hands_done for p in seen]
    assert done == sorted(done) and len(seen) == 3 * 8  # 3 pairings x 8 sessions
    last = seen[-1]
    assert last.fraction == 1.0 and last.matches_done == last.matches_total == 3 and last.hands_total == 3 * HANDS


# ----------------------------------------------------------------------------- failure handling
def test_invalid_agent_lists_are_rejected_before_anything_is_written(tmp_path: Path) -> None:
    for specs in (["random"], ["random", "random"], [AgentSpec("random", {}, "X"), AgentSpec("maniac", {}, "x")]):
        with pytest.raises(ValueError):
            run_tournament(specs, CFG, out_dir=tmp_path)
    with pytest.raises(KeyError):
        run_tournament(["random", "no_such_kind"], CFG, out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_model_artifacts_abort_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing(**_: object) -> Agent:
        raise ModelArtifactError("model file missing: train it first")

    monkeypatch.setitem(registry._FACTORIES, "missing_model_test", missing)
    with pytest.raises(ModelArtifactError, match="train it first"):
        run_tournament(["random", "missing_model_test"], CFG, out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_crash_mid_run_leaves_no_partial_run_and_keeps_latest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Exploding(CallStationAgent):
        def act(self, obs) -> Decision:
            raise RuntimeError("agent crashed")

    monkeypatch.setitem(registry._FACTORIES, "exploding_test", Exploding)
    good = run_tournament(KINDS[:2], replace(CFG, num_hands=20, session_hands=20), out_dir=tmp_path)
    pointer = (tmp_path / "latest.json").read_text()
    with pytest.raises(RuntimeError, match="agent crashed"):
        run_tournament(["random", "exploding_test"], CFG, out_dir=tmp_path)
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [good.name]
    assert (tmp_path / "latest.json").read_text() == pointer


# ----------------------------------------------------------------------------- unit selection and legacy runs
def test_too_few_sessions_fall_back_to_pairs_and_say_so(tmp_path: Path) -> None:
    results = load_results(run_tournament(KINDS, FEW_SESSIONS_CFG, out_dir=tmp_path))
    assert results.ci_unit == "pair" and "3 sessions" in results.ci_unit_note and "NOT independent" in results.ci_unit_note
    assert results.meta["ci_unit"]["requested"] == "session" and results.meta["ci_unit"]["used"] == "pair"
    board, table = results.leaderboard(), results.pairwise_table()
    assert (board["ci_unit"] == "pair").all() and (table["ci_unit"] == "pair").all() and (table["n_sessions"] == 3).all()
    assert table["se"].to_numpy() == pytest.approx(table["pair_se"].to_numpy())  # headline IS the pair level
    assert json.loads((results.run_dir / "summary.json").read_text())["ci_unit"]["note"] == results.ci_unit_note
    assert list_runs(tmp_path)["ci_unit"].tolist() == ["pair"]
    wide = load_results(run_tournament(KINDS, replace(FEW_SESSIONS_CFG, ci_unit="pair", num_hands=160, session_hands=20), out_dir=tmp_path))
    assert wide.ci_unit == "pair" and "requested" in wide.ci_unit_note  # asking for pairs is flagged too


def test_pair_level_runs_can_still_be_requested(tmp_path: Path) -> None:
    cfg = replace(CFG, ci_unit="pair", keep_hands=KeepHandsPolicy())
    results = load_results(run_tournament(KINDS, cfg, out_dir=tmp_path))
    table = results.pairwise_table()
    assert (table["ci_unit"] == "pair").all() and (table["ci_units"] == HANDS // 2).all()
    assert table["se"].to_numpy() == pytest.approx(table["pair_se"].to_numpy())
    assert table["session_over_pair_se"].notna().all()  # the session-level comparison is still reported


def make_legacy_copy(source: Path, target: Path) -> Path:
    """Rewrite a finished run into the schema-1 format (intervals over pairs, no session fields)."""
    shutil.copytree(source, target)
    config = json.loads((target / "config.json").read_text())
    del config["match_config"]["ci_unit"]
    del config["match_config"]["protocol"]
    (target / "config.json").write_text(json.dumps(config))
    summary = json.loads((target / "summary.json").read_text())
    summary["schema_version"] = 1
    del summary["ci_unit"]
    del summary["protocol"]
    for row in summary["leaderboard"]:
        row["bb_per_100"] = row.pop("bb_per_100_pair")  # schema 1 stored the pair-level estimates
        del row["ci_unit"]
    for matrix_row in summary["matrix"].values():
        for cell in matrix_row.values():
            cell.update(se=0.123, ci_low=-0.5, ci_high=0.5)  # stale, must be recomputed on load
    (target / "summary.json").write_text(json.dumps(summary))
    for path in (target / "matches").glob("*.json"):
        match = json.loads(path.read_text())
        for key in ("ci_unit_used", "ci_units", "num_sessions", "protocol"):
            del match[key]
        del match["variance"]["session_over_pair_se"]
        for player in match["players"].values():
            player["bb_per_100"] = player.pop("bb_per_100_pair")
            player["verdict"] = {"code": "draw", "text": "stale"}
        path.write_text(json.dumps(match))
    return target


def test_old_runs_are_re_estimated_on_load_like_a_fresh_run(res: TournamentResults, run_dir: Path, tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="second-mover artifact"):  # no protocol field = the legacy protocol
        legacy = load_results(make_legacy_copy(run_dir, tmp_path / "legacy_run"))
    assert legacy.recomputed_on_load and legacy.meta["recomputed_on_load"] and legacy.ci_unit == "session"
    assert legacy.meta["schema_version"] == 1
    for name in ("leaderboard", "pairwise_table", "verdict_matrix", "matchup_matrix"):
        fresh, old = (without_latency(getattr(r, name)()).drop(columns="protocol", errors="ignore") for r in (res, legacy))
        pd.testing.assert_frame_equal(fresh, old, check_exact=False)
    pd.testing.assert_frame_equal(res.matchup_matrix(part="ci_high"), legacy.matchup_matrix(part="ci_high"))
    assert np.array_equal(res.session_profits("Maniac", "Random"), legacy.session_profits("Maniac", "Random"))
    pd.testing.assert_frame_equal(res.cumulative_profit("Maniac", "Random"), legacy.cumulative_profit("Maniac", "Random"))
    assert legacy.match_config["ci_unit"] == "session" and list_runs(tmp_path)["ci_unit"].tolist() == ["session"]


def test_unknown_schema_versions_are_refused(run_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "future"
    shutil.copytree(run_dir, copy)
    summary = json.loads((copy / "summary.json").read_text())
    summary["schema_version"] = 99
    (copy / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="schema 99"):
        load_results(copy)


# ----------------------------------------------------------------------------- duplicate protocol
def strip_protocol_fields(run: Path) -> None:
    """Rewrite a finished run as if it had been written before the ``protocol`` field existed."""
    config = json.loads((run / "config.json").read_text())
    del config["match_config"]["protocol"]
    (run / "config.json").write_text(json.dumps(config))
    summary = json.loads((run / "summary.json").read_text())
    del summary["protocol"]
    (run / "summary.json").write_text(json.dumps(summary))
    for path in (run / "matches").glob("*.json"):
        match = json.loads(path.read_text())
        del match["protocol"]
        path.write_text(json.dumps(match))


def test_current_runs_record_the_two_table_protocol_and_load_silently(res: TournamentResults, run_dir: Path) -> None:
    assert res.protocol == "two_table" and res.protocol_note is None
    assert res.config["match_config"]["protocol"] == "two_table"
    assert json.loads((run_dir / "summary.json").read_text())["protocol"] == {"used": "two_table", "note": None}
    assert all(res.match_summary(a, b)["protocol"] == "two_table" for a, b in res.pairings())
    assert (res.leaderboard()["protocol"] == "two_table").all() and (res.pairwise_table()["protocol"] == "two_table").all()
    assert res.meta["protocol"] == {"used": "two_table", "note": None}
    assert (list_runs(run_dir.parent)["protocol"] == "two_table").all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a current run must not warn
        load_results(run_dir)


def test_a_two_table_run_has_the_complementary_button_pattern(res: TournamentResults) -> None:
    """Hands 2p / 2p+1 are the two plays of deal p, but A is on the button in hand 2p only at even p."""
    for stem in ("Call_Station__vs__Maniac", "Call_Station__vs__Random", "Maniac__vs__Random"):
        with np.load(res.run_dir / "hands" / f"{stem}.npz") as data:
            assert data["a_on_button"].tolist() == [True, False, False, True] * (HANDS // 4), stem


def test_runs_without_a_protocol_field_are_read_as_single_table_and_flagged(tmp_path: Path) -> None:
    cfg = replace(CFG, protocol="single_table", keep_hands=KeepHandsPolicy())
    run = run_tournament(KINDS, cfg, out_dir=tmp_path)
    with pytest.warns(UserWarning, match="second-mover artifact"):
        explicit = load_results(run)  # still carries the field: single_table is announced, too
    assert explicit.protocol == "single_table"
    strip_protocol_fields(run)
    with pytest.warns(UserWarning, match="second-mover artifact"):
        legacy = load_results(run)
    assert legacy.protocol == "single_table" and "second-mover artifact" in legacy.protocol_note
    assert legacy.meta["protocol"] == {"used": "single_table", "note": legacy.protocol_note}
    assert (legacy.leaderboard()["protocol"] == "single_table").all()
    assert (legacy.pairwise_table()["protocol"] == "single_table").all()
    assert list_runs(tmp_path)["protocol"].tolist() == ["single_table"]
    assert legacy.match_config["protocol"] == "single_table"
    # the numbers are read exactly as before the field was removed
    pd.testing.assert_frame_equal(without_latency(explicit.pairwise_table()), without_latency(legacy.pairwise_table()))
    with np.load(legacy.run_dir / "hands" / "Call_Station__vs__Maniac.npz") as data:
        assert data["a_on_button"].tolist() == [True, False] * (HANDS // 2)  # one table: A always first


def test_non_duplicate_runs_have_no_pairs_and_therefore_no_protocol_warning(tmp_path: Path) -> None:
    cfg = MatchConfig(num_hands=40, session_hands=20, duplicate=False)
    assert cfg.protocol == "single_table" and cfg.protocol_note is None
    results = load_results(run_tournament(KINDS[:2], cfg, out_dir=tmp_path))  # must not warn
    assert results.protocol == "single_table" and results.protocol_note is None
