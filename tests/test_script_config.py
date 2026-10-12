"""Tests for ``poker_arena.script_config`` (the --smoke / --set convention of every script)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytest

from poker_arena.script_config import (
    ConfigOverrideError,
    add_config_arguments,
    apply_cli_overrides,
    coerce_value,
    config_variables,
    is_config_name,
)


def make_globals() -> dict:
    """A stand-in for a script's ``globals()``."""
    return {
        "__name__": "script",
        "math": math,  # imported module: not config
        "Path": Path,  # class: not config
        "RANDOM_SEED": 42,
        "LEARNING_RATE": 0.1,
        "VERBOSE": False,
        "RUN_NAME": None,
        "OUT_DIR": Path("artifacts"),
        "TAG": "base",
        "AGENTS": [("random", {}, "Random")],
        "SIZES": (1, 2),
        "PARAMS": {"depth": 3},
        "_PRIVATE": 1,
        "lowercase": 5,
        "SMOKE_OVERRIDES": {"RANDOM_SEED": 1, "VERBOSE": True},
    }


def apply(g: dict, *argv: str) -> dict:
    return apply_cli_overrides(g, g["SMOKE_OVERRIDES"], list(argv))


def test_defaults_without_arguments_return_the_config_variables_only() -> None:
    g = make_globals()
    cfg = apply(g)
    assert set(cfg) == {"RANDOM_SEED", "LEARNING_RATE", "VERBOSE", "RUN_NAME", "OUT_DIR", "TAG", "AGENTS", "SIZES", "PARAMS"}
    assert cfg["RANDOM_SEED"] == 42 and g["RANDOM_SEED"] == 42


def test_is_config_name() -> None:
    assert is_config_name("RANDOM_SEED") and is_config_name("N_WORKERS2")
    assert not is_config_name("_PRIVATE") and not is_config_name("lowercase") and not is_config_name("Mixed_Case")


def test_config_variables_skip_modules_callables_private_and_smoke_overrides() -> None:
    variables = config_variables(make_globals())
    assert "SMOKE_OVERRIDES" not in variables and "_PRIVATE" not in variables and "Path" not in variables


def test_smoke_applies_overrides_in_place() -> None:
    g = make_globals()
    cfg = apply(g, "--smoke")
    assert cfg["RANDOM_SEED"] == 1 and cfg["VERBOSE"] is True
    assert g["RANDOM_SEED"] == 1  # module globals are updated, the script keeps reading plain variables


def test_set_overrides_win_over_smoke_and_coerce_to_the_existing_type() -> None:
    cfg = apply(make_globals(), "--smoke", "--set", "RANDOM_SEED=7", "--set", "LEARNING_RATE=3", "--set", "VERBOSE=false")
    assert cfg["RANDOM_SEED"] == 7 and isinstance(cfg["RANDOM_SEED"], int)
    assert cfg["LEARNING_RATE"] == 3.0 and isinstance(cfg["LEARNING_RATE"], float)
    assert cfg["VERBOSE"] is False


@pytest.mark.parametrize(
    ("current", "raw", "expected"),
    [
        (1, "20000", 20000),
        (1, "2e4", 20000),
        (1, "1_000", 1000),
        (1.5, "0.25", 0.25),
        (False, "yes", True),
        (True, "0", False),
        (True, "Off", False),
        ("x", "a=b c", "a=b c"),
        (Path("a"), "b/c", Path("b/c")),
        (None, "none", None),
        (None, "5", 5),
        (None, "[1, 2]", [1, 2]),
        (None, "hello", "hello"),
        ([0], "[(1, 2), (3, 4)]", [(1, 2), (3, 4)]),
        ((0,), "[1, 2]", (1, 2)),
        ({}, "{'a': 1}", {"a": 1}),
    ],
)
def test_coerce_value(current, raw, expected) -> None:
    result = coerce_value("KEY", raw, current)
    assert result == expected and type(result) is type(expected)


@pytest.mark.parametrize(
    ("current", "raw"),
    [(1, "abc"), (1, "2.5"), (1.0, "x"), (True, "maybe"), ([], "5"), ([], "{'a': 1}"), ({}, "[1]"), ([], "not python")],
)
def test_coerce_value_rejects_bad_input(current, raw) -> None:
    with pytest.raises(ConfigOverrideError):
        coerce_value("KEY", raw, current)


def test_list_override_replaces_agents() -> None:
    cfg = apply(make_globals(), "--set", "AGENTS=[('maniac', {}, 'M')]")
    assert cfg["AGENTS"] == [("maniac", {}, "M")]


def test_unknown_key_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        apply(make_globals(), "--set", "NOPE=1")
    assert excinfo.value.code == 2
    assert "NOPE" in capsys.readouterr().err


def test_lowercase_or_private_keys_are_not_overridable() -> None:
    for key in ("lowercase", "_PRIVATE", "SMOKE_OVERRIDES"):
        with pytest.raises(SystemExit):
            apply(make_globals(), "--set", f"{key}=1")


def test_bad_value_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        apply(make_globals(), "--set", "RANDOM_SEED=abc")
    assert "RANDOM_SEED" in capsys.readouterr().err


def test_set_without_equals_is_an_error() -> None:
    with pytest.raises(SystemExit):
        apply(make_globals(), "--set", "RANDOM_SEED")


def test_unknown_smoke_key_is_a_programming_error() -> None:
    g = make_globals()
    g["SMOKE_OVERRIDES"] = {"TYPO_KEY": 1}
    with pytest.raises(ConfigOverrideError):
        apply(g, "--smoke")


def test_smoke_overrides_are_ignored_without_the_flag_and_none_is_accepted() -> None:
    g = make_globals()
    assert apply_cli_overrides(g, None, ["--smoke"])["RANDOM_SEED"] == 42
    assert apply(g)["RANDOM_SEED"] == 42


def test_options_of_the_script_itself_are_ignored() -> None:
    cfg = apply(make_globals(), "--agents", "a,b", "--out", "x", "--set", "TAG=new", "--verbose")
    assert cfg["TAG"] == "new"


def test_argv_defaults_to_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["script.py", "--set", "TAG=from_argv"])
    assert apply_cli_overrides(make_globals(), {})["TAG"] == "from_argv"


def test_add_config_arguments_registers_the_shared_switches() -> None:
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args(["--smoke", "--set", "A=1", "--set", "B=2"])
    assert args.smoke is True and args.overrides == ["A=1", "B=2"]
    assert parser.parse_args([]).overrides == []
