"""Sanity checks of the artifacts that are committed to the repository (models, tournament runs, notebooks)."""

from __future__ import annotations

import json
import re

import pytest

from poker_arena.agents.registry import create_agent
from poker_arena.config import ARTIFACTS_DIR, MODELS_DIR, REPO_ROOT, TOURNAMENTS_DIR

TEXT_SUFFIXES = {".json", ".md", ".csv", ".jsonl", ".ipynb"}
# an absolute local path: a Windows drive path, or a Unix home directory
LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]+Users|/home/\w+|/Users/\w+")


def _committed_text_files():
    for root in (MODELS_DIR, TOURNAMENTS_DIR, ARTIFACTS_DIR / "teacher", REPO_ROOT / "notebooks"):
        if root.exists():
            yield from (p for p in root.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES)


def test_saved_artifacts_contain_no_absolute_local_paths():
    offenders = []
    for path in _committed_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if str(REPO_ROOT) in text or str(REPO_ROOT).replace("\\", "\\\\") in text or LOCAL_PATH.search(text):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, f"absolute local paths found in saved artifacts: {offenders}"


@pytest.mark.parametrize("kind", ["ebm", "xgboost", "ga"])
def test_committed_models_load_and_describe_themselves(kind):
    agent = create_agent(kind)
    info = agent.describe()
    assert info["name"] and info["architecture"]


def test_latest_tournament_pointer_resolves_to_an_existing_run():
    pointer = TOURNAMENTS_DIR / "latest.json"
    if not pointer.exists():
        pytest.skip("no official tournament run is committed")
    run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    run_dir = TOURNAMENTS_DIR / run_id
    assert (run_dir / "summary.json").exists() and (run_dir / "config.json").exists()
