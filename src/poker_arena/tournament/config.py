"""Configuration of a match / tournament.

One frozen :class:`MatchConfig` fully determines *what is simulated* (game rules, number of hands,
seed, session length, duplicate dealing) and how the result is summarised (confidence level, CI
method, CI unit). Together with the agent specs and the model artifacts it makes a run reproducible. The one
field that does **not** influence results is ``n_workers`` (see :func:`resolve_n_workers`).
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from poker_arena.config import GameConfig

#: Hands per independent "session" (agents are re-created and opponent statistics reset at the start of
#: each). Sessions are the unit of parallel work, so this also bounds how finely a match can be split.
DEFAULT_SESSION_HANDS = 200
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
CI_METHODS = ("t", "bootstrap")
#: Independent units confidence intervals can be computed over: whole sessions (default) or the finest
#: unit, the duplicate pair (the single hand when ``duplicate=False``).
CI_UNITS = ("session", "pair")
#: How the two plays of a duplicate deal are organised (see :mod:`poker_arena.tournament.duplicate`):
#: ``two_table`` - two independent tables per session, no information about a deal crosses between its two
#: plays (the real duplicate-poker protocol; default). ``single_table`` - both plays back-to-back against
#: the same agents and opponent statistics: the legacy protocol, kept only to reproduce / diagnose old runs.
TWO_TABLE = "two_table"
SINGLE_TABLE = "single_table"
PROTOCOLS = (TWO_TABLE, SINGLE_TABLE)
#: Fewer sessions than this per match and the session-level interval is not trustworthy (a t-interval
#: with a handful of clusters); the interval then falls back to the pair level and says so.
MIN_SESSIONS_FOR_CI = 8
#: Windows' ``WaitForMultipleObjects`` limit for ``ProcessPoolExecutor`` worker handles.
MAX_POOL_WORKERS = 61
#: Cores left free for the OS / the parent process when ``n_workers <= 0`` ("auto").
RESERVED_CORES = 1


def _even_up(n: int) -> int:
    """Round up to the next even number (duplicate pairs and button alternation need even counts)."""
    return n + (n % 2)


@dataclass(frozen=True)
class KeepHandsPolicy:
    """Which hands of a match are stored in full (replayable ``HandRecord.to_dict()`` JSON).

    Per-hand *numeric* data is always stored for every hand; full records are large, so by default
    only a sample is kept: the first hands (to read through a typical session) and the extremes
    (to study how big pots were won and lost).
    """

    first: int = 10  # the first N hands of the match
    biggest_pots: int = 5  # the N largest pots
    biggest_wins_each: int = 5  # for each player, the N hands in which they won the most chips
    all_hands: bool = False  # additionally write EVERY hand to a gzip-jsonl file (small runs only!)

    def __post_init__(self) -> None:
        if min(self.first, self.biggest_pots, self.biggest_wins_each) < 0:
            raise ValueError("KeepHandsPolicy counts must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KeepHandsPolicy":
        return cls(**d)


@dataclass(frozen=True)
class MatchConfig:
    """Everything that defines a head-to-head match (and the settings of a round-robin built from it).

    num_hands        hands per pairing; rounded UP to an even number (a duplicate pair is two hands).
    seed             master seed; all deals and agent seeds are ``derive_seed("eval", seed, ...)``.
    duplicate        True: every deal is played twice with the seats swapped (matched pairs, the
                     statistical unit is the pair). False: independent deal per hand, button alternates.
    session_hands    hands per session (rounded up to even). Agents are rebuilt + reset and opponent
                     trackers cleared at each session start, which makes results independent of
                     ``n_workers`` and lets sessions run in parallel.
    n_workers        1 = run in this process; N > 1 = process pool; <= 0 = all cores but one.
    confidence       confidence level of every interval (0.95 = 95% CI).
    ci_method        "t" (t-interval on unit profits) or "bootstrap" (seeded percentile bootstrap).
    bootstrap_resamples  number of bootstrap resamples (only used for ``ci_method="bootstrap"``).
    protocol         "two_table" (default) or "single_table" (legacy, diagnostics only): how the two plays of a
                     duplicate deal are organised. In a single-table session the second play of a deal is played
                     against an opponent model that already contains the first play of the SAME deal, so
                     identical *adaptive* agents do not cancel (the first-named agent loses roughly 10-13 BB/100 in A/A
                     matches of the teacher). Two tables per session - each with its own agent copies and
                     opponent trackers, playing the same deals with the seats complemented - remove that leak.
                     Without duplicate deals there are no pairs and the protocol is always "single_table".
    ci_unit          independent unit of every interval: "session" (default; the t-interval / bootstrap is
                     taken over per-session BB/100, df = sessions - 1) or "pair" (duplicate pairs). Agents
                     that adapt to opponent statistics make the pairs of one session dependent, so pair-level
                     intervals can be too narrow; sessions are independent by construction. With fewer than
                     ``MIN_SESSIONS_FOR_CI`` sessions per match the pair level is used instead and flagged
                     (:attr:`ci_unit_used`, :attr:`ci_unit_note`).

    Chips reset to ``game.starting_stack`` at the start of every hand (fixed-depth cash-game evaluation).
    """

    game: GameConfig = field(default_factory=GameConfig)
    num_hands: int = 2000
    seed: int = 42
    duplicate: bool = True
    session_hands: int = DEFAULT_SESSION_HANDS
    keep_hands: KeepHandsPolicy = field(default_factory=KeepHandsPolicy)
    n_workers: int = 1
    confidence: float = DEFAULT_CONFIDENCE
    ci_method: str = "t"
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    ci_unit: str = "session"
    protocol: str = TWO_TABLE

    def __post_init__(self) -> None:
        if self.num_hands < 2:
            raise ValueError("num_hands must be at least 2")
        if self.session_hands < 2:
            raise ValueError("session_hands must be at least 2")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.ci_method not in CI_METHODS:
            raise ValueError(f"ci_method must be one of {CI_METHODS}, got {self.ci_method!r}")
        if self.ci_unit not in CI_UNITS:
            raise ValueError(f"ci_unit must be one of {CI_UNITS}, got {self.ci_unit!r}")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        object.__setattr__(self, "num_hands", _even_up(int(self.num_hands)))
        object.__setattr__(self, "session_hands", _even_up(int(self.session_hands)))
        if not self.duplicate:
            object.__setattr__(self, "protocol", SINGLE_TABLE)  # no pairs, nothing to keep apart

    # ------------------------------------------------------------------ derived quantities
    @property
    def hands_per_unit(self) -> int:
        """Hands per statistical unit: 2 (a duplicate pair) or 1 (an independent hand)."""
        return 2 if self.duplicate else 1

    @property
    def num_units(self) -> int:
        """Number of independent statistical units per match."""
        return self.num_hands // self.hands_per_unit

    @property
    def num_sessions(self) -> int:
        return math.ceil(self.num_hands / self.session_hands)

    @property
    def protocol_note(self) -> str | None:
        """Warning for the legacy protocol (shown wherever its results are shown), else ``None``."""
        if not (self.duplicate and self.protocol == SINGLE_TABLE):
            return None
        return (
            "Legacy single-table duplicate protocol: the two hands of a deal were played back-to-back against "
            "the same agents and opponent statistics, so the second hand of every deal is played against an "
            "opponent model that already contains the first hand of the same deal (second-mover artifact "
            "present for adaptive agents). Identical adaptive agents do NOT cancel - their first-named side "
            "loses roughly 10-13 BB/100 in A/A matches of the teacher. Compare with protocol='two_table' runs only "
            "after re-running; stateless agents are not affected."
        )

    @property
    def fine_unit(self) -> str:
        """Name of the finest statistical unit: "pair" for duplicate matches, else "hand"."""
        return "pair" if self.duplicate else "hand"

    @property
    def ci_unit_used(self) -> str:
        """Unit the intervals are really computed over: "session", or :attr:`fine_unit` when pairs were
        requested or there are too few sessions (see ``MIN_SESSIONS_FOR_CI``)."""
        if self.ci_unit == "session" and self.num_sessions >= MIN_SESSIONS_FOR_CI:
            return "session"
        return self.fine_unit

    @property
    def ci_unit_note(self) -> str | None:
        """Warning to show next to every interval when it is not session-level, else ``None``."""
        if self.ci_unit_used == "session":
            return None
        caveat = (
            f"Intervals are computed over {self.fine_unit}s, which are NOT independent when agents adapt to "
            "opponent statistics within a session (they can be too narrow)."
        )
        if self.ci_unit == "session":
            return (
                f"Only {self.num_sessions} sessions per match (fewer than {MIN_SESSIONS_FOR_CI}): session-level "
                f"intervals are not reliable, so the {self.fine_unit} level is used instead. {caveat} "
                "Use more hands or a smaller SESSION_HANDS for session-level intervals."
            )
        return f"{self.fine_unit}-level intervals were requested (ci_unit='pair'). {caveat}"

    def session_bounds(self) -> list[tuple[int, int]]:
        """``[(first_hand_index, number_of_hands), ...]`` covering the match; every start is even."""
        return [
            (first, min(self.session_hands, self.num_hands - first))
            for first in range(0, self.num_hands, self.session_hands)
        ]

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["game"] = self.game.to_dict()
        d["keep_hands"] = self.keep_hands.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MatchConfig":
        fields = dict(d)
        fields.setdefault("protocol", SINGLE_TABLE)  # files written before the field existed used one table
        fields["game"] = GameConfig.from_dict(d["game"])
        fields["keep_hands"] = KeepHandsPolicy.from_dict(d["keep_hands"])
        return cls(**fields)


def resolve_n_workers(requested: int, n_tasks: int) -> int:
    """Number of worker processes to use: ``requested`` (``<= 0`` means all cores but one), capped by
    the number of independent tasks and the Windows pool limit; never below 1."""
    available = max(1, (os.cpu_count() or 1) - RESERVED_CORES)
    wanted = available if requested <= 0 else requested
    return max(1, min(wanted, n_tasks, MAX_POOL_WORKERS))
