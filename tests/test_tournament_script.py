"""Tests for ``scripts/run_tournament.py`` (CLI smoke run, config conventions, agent selection)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from poker_arena.agents.base import ModelArtifactError
from poker_arena.config import REPO_ROOT
from poker_arena.script_config import config_variables
from poker_arena.tournament.config import MIN_SESSIONS_FOR_CI
from poker_arena.tournament.results import load_results

SCRIPT = REPO_ROOT / "scripts" / "run_tournament.py"
BASELINES = "random,call_station,maniac"
SMOKE_TIME_LIMIT_S = 90


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=SMOKE_TIME_LIMIT_S
    )


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_tournament_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # only defines the config block and functions; main() is guarded
    return module


# ----------------------------------------------------------------------------- the CLI
def test_smoke_run_with_baselines_writes_artifacts_to_the_given_folder(tmp_path: Path) -> None:
    started = time.perf_counter()
    done = run_script("--smoke", "--agents", BASELINES, "--out", str(tmp_path))
    assert time.perf_counter() - started < SMOKE_TIME_LIMIT_S
    assert done.returncode == 0, done.stderr
    for heading in ("LEADERBOARD", "MATCHUP MATRIX", "VERDICTS"):
        assert heading in done.stdout
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert run_dir.name.endswith("_smoke")
    assert not (tmp_path / "latest.json").exists()  # a smoke run is never the official run
    results = load_results(run_dir)
    assert sorted(results.agent_names) == ["CallStation", "Maniac", "Random"]
    assert results.match_config["num_hands"] == 400 and results.meta["total_hands"] == 3 * 400
    script_config = results.config["script_config"]
    assert script_config["script"] == "scripts/run_tournament.py" and script_config["config"]["RUN_NAME"] == "smoke"
    assert len(results.leaderboard()) == 3 and results.verdict_matrix().shape == (3, 3)
    assert results.ci_unit == "session" and results.ci_unit_note is None and "WARNING" not in done.stdout
    assert script_config["config"]["CI_UNIT"] == "session" and results.match_config["session_hands"] == 40
    assert results.protocol == "two_table" and script_config["config"]["PROTOCOL"] == "two_table"


def test_set_overrides_and_mark_latest(tmp_path: Path) -> None:
    done = run_script(
        "--smoke", "--agents", "random,call_station", "--out", str(tmp_path),
        "--set", "NUM_TOURNAMENT_HANDS=100", "--set", "SESSION_HANDS=50", "--set", "RUN_NAME=custom",
        "--set", "MARK_LATEST=true", "--set", "N_WORKERS=1", "--set", "CI_METHOD=bootstrap",
        "--set", "BOOTSTRAP_RESAMPLES=200", "--set", "KEEP_FULL_HISTORY=true", "--set", "PROTOCOL=single_table",
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    pointer = json.loads((tmp_path / "latest.json").read_text())
    with pytest.warns(UserWarning, match="second-mover artifact"):  # PROTOCOL=single_table is the legacy protocol
        results = load_results("latest", root=tmp_path)
    assert pointer["run_id"] == results.run_id and results.run_id.endswith("_custom")
    assert results.match_config["num_hands"] == 100 and results.match_config["ci_method"] == "bootstrap"
    assert results.full_history_path("Random", "CallStation") is not None
    assert results.ci_unit == "pair" and "Only 2 sessions" in results.ci_unit_note  # 100 hands / 50 = 2 sessions
    assert "WARNING: Only 2 sessions" in done.stdout  # the fallback is announced loudly in the report
    assert results.protocol == "single_table" and "WARNING: Legacy single-table" in done.stdout  # so is the legacy protocol


def test_unknown_config_key_is_a_usage_error() -> None:
    done = run_script("--set", "NO_SUCH_SETTING=1")
    assert done.returncode == 2 and "NO_SUCH_SETTING" in done.stderr


def test_help_documents_the_shared_switches() -> None:
    done = run_script("--help")
    assert done.returncode == 0
    for option in ("--smoke", "--set", "--agents", "--out"):
        assert option in done.stdout


# ----------------------------------------------------------------------------- conventions
def test_config_block_follows_the_script_convention(script: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    header = "# ==========================================\n# TOURNAMENT CONFIGURATION\n# ==========================================\n"
    assert header in source
    before_block = source.split(header)[0]
    assert "import" in before_block and "RANDOM_SEED" not in before_block  # imports first, block right after
    variables = config_variables(vars(script))
    for required in ("RANDOM_SEED", "STARTING_STACK", "SMALL_BLIND", "BIG_BLIND", "NUM_TOURNAMENT_HANDS", "SESSION_HANDS",
                     "N_WORKERS", "AGENTS", "REQUIRE_ALL_AGENTS", "RUN_NAME", "KEEP_FULL_HISTORY", "CI_UNIT", "PROTOCOL"):  # fmt: skip
        assert required in variables, required
    assert set(script.SMOKE_OVERRIDES) <= set(variables)
    assert script.SMOKE_OVERRIDES["NUM_TOURNAMENT_HANDS"] < script.NUM_TOURNAMENT_HANDS
    assert script.SMOKE_OVERRIDES["MARK_LATEST"] is False
    assert script.CI_UNIT == "session" and "Why sessions are the unit" in script.__doc__
    assert script.PROTOCOL == "two_table" and "Why every session is played at two tables" in script.__doc__
    smoke_sessions = script.SMOKE_OVERRIDES["NUM_TOURNAMENT_HANDS"] // script.SMOKE_OVERRIDES["SESSION_HANDS"]
    assert smoke_sessions >= MIN_SESSIONS_FOR_CI  # the smoke run exercises the default (session) path
    kinds = [kind for kind, _, _ in script.AGENTS]
    assert {"ebm", "xgboost", "ga", "teacher", "random", "call_station", "maniac"} <= set(kinds)


# ----------------------------------------------------------------------------- agent selection
def test_unavailable_agents_are_skipped_with_a_loud_warning(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_create = script.create_agent

    def fake_create(spec):  # the "trained" kind has no artifacts
        if spec.kind == "ebm":
            raise ModelArtifactError("artifacts/models/ebm/model.joblib is missing; run scripts/train_ebm.py")
        return real_create(spec)

    monkeypatch.setattr(script, "create_agent", fake_create)
    entries = [("ebm", {}, "EBM"), ("random", {}, "Random"), ("maniac", {}, "Maniac")]
    specs = script.select_agents(entries, only=None, require_all=False)
    assert [s.label for s in specs] == ["Random", "Maniac"]
    err = capsys.readouterr().err
    assert "SKIPPING AGENT 'EBM'" in err and "train_ebm.py" in err and "!!!!" in err
    with pytest.raises(SystemExit, match="REQUIRE_ALL_AGENTS"):
        script.select_agents(entries, only=None, require_all=True)


def test_agent_filter_accepts_labels_or_kinds_case_insensitively(script: ModuleType) -> None:
    entries = [("random", {}, "Random"), ("call_station", {}, "CallStation"), ("maniac", {}, "TheManiac")]
    assert [s.label for s in script.select_agents(entries, "RANDOM, themaniac", False)] == ["Random", "TheManiac"]
    assert [s.kind for s in script.select_agents(entries, "call_station,maniac", False)] == ["call_station", "maniac"]
    with pytest.raises(SystemExit, match="at least two"):
        script.select_agents(entries, "random", False)
    with pytest.raises(SystemExit, match="at least two"):
        script.select_agents([("ebm", {}, "EBM")], None, False)
