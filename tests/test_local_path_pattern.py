"""The local-path detector used by ``test_artifacts_committed`` must itself be trustworthy."""

from __future__ import annotations

import pytest
from test_artifacts_committed import LOCAL_PATH

BS = chr(92)  # a single backslash, built explicitly so that no escaping level can change it


@pytest.mark.parametrize(
    "leaked",
    [
        "C:" + BS + "Users" + BS + "bob" + BS + "repo",  # raw Windows path
        "C:" + BS * 2 + "Users" + BS * 2 + "bob",  # the same path JSON-escaped
        "C:/Users/bob/repo",
        "/home/bob/repo",
        "/Users/bob/repo",
    ],
)
def test_detects_absolute_local_paths(leaked):
    assert LOCAL_PATH.search(leaked)


@pytest.mark.parametrize(
    "harmless",
    [
        "notebook:" + BS + "n",  # a JSON-escaped newline after a colon
        "def f(x: float) -> str:" + BS + "n",
        "artifacts/models/ebm",
        "ratio 10:30",
    ],
)
def test_ignores_ordinary_text(harmless):
    assert not LOCAL_PATH.search(harmless)
