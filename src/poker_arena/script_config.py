"""Command-line layer for the "configuration block at the top of the script" convention.

ARCHITECTURE.md section 6 says every script in ``scripts/`` starts with a block of UPPER_CASE
configuration variables plus a ``SMOKE_OVERRIDES`` dict. This module gives all of them the same
two switches without each script re-implementing argument parsing:

* ``--smoke``            apply ``SMOKE_OVERRIDES`` (a tiny, fast configuration for a quick end-to-end check)
* ``--set KEY=VALUE``    override any config variable (repeatable); the value is converted to the type
  of the variable's current value, so ``--set NUM_TOURNAMENT_HANDS=5000`` stays an ``int`` and
  ``--set REQUIRE_ALL_AGENTS=false`` becomes ``False``.

Typical use (inside ``main()``, never at import time, so worker processes that re-import the script
under ``spawn`` do not re-parse the command line)::

    def main() -> None:
        cfg = apply_cli_overrides(globals(), SMOKE_OVERRIDES)
        ...  # read cfg["NUM_TOURNAMENT_HANDS"] (the module globals have been updated as well)

Scripts with additional options build their own ``argparse`` parser, call
:func:`add_config_arguments` on it (so ``--help`` documents the shared switches and typos are
rejected) and still call :func:`apply_cli_overrides` - extra options are ignored by it.

Precedence: value in the file < ``--smoke`` < ``--set``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

SMOKE_OVERRIDES_NAME = "SMOKE_OVERRIDES"
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})
_NONE_WORDS = frozenset({"none", "null"})


class ConfigOverrideError(ValueError):
    """An override names an unknown variable or has a value of the wrong type."""


def is_config_name(name: str) -> bool:
    """True for public UPPER_CASE identifiers (``RANDOM_SEED``, ``N_WORKERS``); False for ``_PRIVATE``."""
    return name.isidentifier() and name.isupper() and not name.startswith("_")


def config_variables(config_globals: MutableMapping[str, Any]) -> dict[str, Any]:
    """The configuration variables of a script: UPPER_CASE globals that are plain values.

    ``SMOKE_OVERRIDES`` itself, imported modules, classes and functions (e.g. an imported
    ``SomeEnum`` or ``MAX_FN``) are not configuration and are skipped.
    """
    return {
        name: value
        for name, value in config_globals.items()
        if is_config_name(name)
        and name != SMOKE_OVERRIDES_NAME
        and not inspect.ismodule(value)
        and not callable(value)
    }


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--smoke`` and ``--set KEY=VALUE`` options to a script's own parser."""
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="apply SMOKE_OVERRIDES: a tiny configuration that finishes quickly (end-to-end check)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a configuration variable of the script's config block (repeatable)",
    )


def coerce_value(key: str, raw: str, current: Any) -> Any:
    """Convert the CLI string ``raw`` to the type of ``current`` (the variable's present value).

    * ``bool``: true/false/yes/no/on/off/1/0     * ``int``: ``20000``, ``2e4``, ``1_000``
    * ``float``, ``str``, ``Path``              * ``list``/``tuple``/``dict``: a Python literal
    * ``None`` (an optional setting): a Python literal if it parses (``5``, ``[1, 2]``), ``none``/``null``
      for ``None``, otherwise the raw string.
    """
    text = raw.strip()
    try:
        if isinstance(current, bool):  # before int: bool is an int subclass
            lowered = text.lower()
            if lowered in _TRUE_WORDS:
                return True
            if lowered in _FALSE_WORDS:
                return False
            raise ValueError("expected true/false")
        if isinstance(current, int):
            try:
                return int(text)
            except ValueError:
                as_float = float(text)
                if not as_float.is_integer():
                    raise
                return int(as_float)
        if isinstance(current, float):
            return float(text)
        if isinstance(current, Path):
            return Path(raw)
        if isinstance(current, str):
            return raw
        if isinstance(current, (list, tuple, dict, set)):
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, (list, tuple, dict, set)) or isinstance(parsed, dict) != isinstance(current, dict):
                raise ValueError(f"expected a {type(current).__name__} literal")
            return type(current)(parsed)
        if current is None:
            if text.lower() in _NONE_WORDS:
                return None
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return raw
    except (ValueError, SyntaxError) as exc:
        raise ConfigOverrideError(
            f"cannot convert {raw!r} for {key} (current value {current!r}, type {type(current).__name__}): {exc}"
        ) from exc
    raise ConfigOverrideError(f"{key} has type {type(current).__name__}, which cannot be overridden from the CLI")


def _parse_set_option(option: str) -> tuple[str, str]:
    key, separator, raw = option.partition("=")
    if not separator or not key.strip():
        raise ConfigOverrideError(f"--set expects KEY=VALUE, got {option!r}")
    return key.strip(), raw


def apply_cli_overrides(
    config_globals: MutableMapping[str, Any],
    smoke_overrides: dict[str, Any] | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply ``--smoke`` / ``--set KEY=VALUE`` to the script's config variables.

    ``config_globals`` is the script's ``globals()``; it is updated *in place* so the rest of the
    script can keep reading plain module variables. ``argv`` defaults to ``sys.argv[1:]``; options this
    function does not know are ignored (the script may parse its own). Returns the effective
    configuration ``{NAME: value}`` of all UPPER_CASE globals, ready to be saved in a result file.

    An unknown key or an unconvertible ``--set`` value prints a clear error and exits with status 2
    (argparse convention); a ``smoke_overrides`` key that is not a config variable is a programming
    error and raises :class:`ConfigOverrideError`.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_config_arguments(parser)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else list(argv))

    known = config_variables(config_globals)
    if args.smoke and smoke_overrides:
        unknown = sorted(set(smoke_overrides) - set(known))
        if unknown:
            raise ConfigOverrideError(f"{SMOKE_OVERRIDES_NAME} names unknown config variables: {unknown}")
        config_globals.update(smoke_overrides)
        known = config_variables(config_globals)

    try:
        for option in args.overrides:
            key, raw = _parse_set_option(option)
            if key not in known:
                raise ConfigOverrideError(f"unknown config variable {key!r}; known: {sorted(known)}")
            config_globals[key] = known[key] = coerce_value(key, raw, known[key])
    except ConfigOverrideError as exc:
        parser.error(str(exc))
    return config_variables(config_globals)
