"""Headless, reproducible head-to-head matches and round-robin tournaments.

Produce:  :func:`run_match` (one pairing, in memory) and :func:`run_tournament` (round-robin, written to
          ``artifacts/tournaments/<run_id>/``).
Consume:  :func:`load_results` -> :class:`TournamentResults` (DataFrame accessors) - the ONLY interface
          the Streamlit app and the notebooks should use - and :func:`list_runs`.
Judge:    :mod:`poker_arena.tournament.stats` (CIs on duplicate-pair profits, ``compare_verdict``).

Design notes (where this package refines the original brief)
-------------------------------------------------------------
* In every pairing the label that sorts first is agent A, so a pairing's deals, seeds and results do not
  depend on the order or composition of the agent list.
* **Two-table duplicate protocol** (``MatchConfig.protocol``, default ``"two_table"``): every session is
  played at two independent tables (own agent copies and opponent trackers, the same deals, seats
  complemented), so nothing about a deal crosses between its two plays. Hands ``2p`` / ``2p + 1`` are
  always the two plays of deal ``p`` (pair sums unchanged), but A is on the button in hand ``2p`` only for
  even ``p``: read ``arrays.a_on_button``, never the index parity. Identical *stateless* agents cancel
  exactly in every pair under any protocol; identical adaptive agents cancel in expectation (exactly, if
  also deterministic, like the teacher). The legacy ``"single_table"`` protocol plays both hands of a deal
  back-to-back against one opponent model - the second mover then profits from having seen the first play
  of the same cards (roughly -10 to -13 BB/100 for the first-named of two identical teachers). It is kept for
  diagnostics and for reading old runs, which are flagged (``protocol_note``, ``UserWarning``, dashboard).
* The independent unit of every interval is the **session** (``MatchConfig.ci_unit``), not the duplicate
  pair: agents that adapt to opponent statistics make the pairs of one session dependent, sessions start
  from freshly built agents. ``ci_method`` ("t" over per-session BB/100, or a seeded bootstrap over
  sessions) selects the one interval that is stored, shown and used for every verdict; the pair-level
  interval is kept as ``pair_*`` extras. With fewer than ``config.MIN_SESSIONS_FOR_CI`` sessions per match
  the pair level is used and flagged (``ci_unit_note``). Verdicts need ``stats.MIN_SESSIONS_FOR_VERDICT``
  sessions (``MIN_UNITS_FOR_VERDICT`` pairs), otherwise they read "insufficient".
* Per-hand arrays live in ``hands/<A>__vs__<B>.npz`` (a sub-folder, not a ``hands_`` file prefix);
  raw per-decision latencies are summarised in the JSON files instead of being stored.
* Results are identical for every ``n_workers`` (sessions are independent); only latencies vary.

Names are imported lazily so that spawned worker processes, which only need the match code, do not
pay for pandas / scipy at start-up.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "MatchConfig": "config",
    "KeepHandsPolicy": "config",
    "run_match": "match",
    "MatchResult": "match",
    "Progress": "match",
    "run_tournament": "tournament",
    "load_results": "results",
    "list_runs": "results",
    "TournamentResults": "results",
    "summarize_match": "summary",
}
__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:  # pragma: no cover
    from poker_arena.tournament.config import KeepHandsPolicy, MatchConfig
    from poker_arena.tournament.match import MatchResult, Progress, run_match
    from poker_arena.tournament.results import TournamentResults, list_runs, load_results
    from poker_arena.tournament.summary import summarize_match
    from poker_arena.tournament.tournament import run_tournament


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
