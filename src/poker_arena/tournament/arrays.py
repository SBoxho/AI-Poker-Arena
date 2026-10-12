"""Columnar per-hand data of a match.

:class:`HandArrays` holds one entry per played hand (row ``h``; hands ``2i`` and ``2i + 1`` are the
two mirrored hands of duplicate pair ``i``). It is deliberately plain numpy so that a million hands
cost a few tens of MB, can be sliced / summed without Python loops and round-trips through
``.npz``. Player index ``0`` is agent **A** (poker seat 0) and ``1`` is agent **B** (seat 1);
A is on the button in even hands and B in odd hands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Action kinds as produced by the engine (``ActionEvent.kind``); the last axis of ``action_counts``.
ACTION_KINDS = ("fold", "check", "call", "bet", "raise")
KIND_INDEX = {kind: i for i, kind in enumerate(ACTION_KINDS)}
STREET_NAMES = ("preflop", "flop", "turn", "river")
NUM_STREETS = len(STREET_NAMES)
#: Where a hand ended: by a fold on one of the four streets, or at a showdown.
END_CATEGORIES = (*STREET_NAMES, "showdown")
SHOWDOWN_CATEGORY = len(STREET_NAMES)
PLAYER_A, PLAYER_B = 0, 1

# Column name -> (dtype, trailing shape); every column has the number of hands as first axis.
_PER_HAND_COLUMNS: dict[str, tuple[type, tuple[int, ...]]] = {
    "profit_chips": (np.int32, ()),  # net chips won by A (B's profit is the negative)
    "a_on_button": (np.bool_, ()),
    "showdown": (np.bool_, ()),
    "ended_street": (np.int8, ()),  # board street (0-3) when the hand ended
    "pot_chips": (np.int32, ()),  # total chips contested
    "action_counts": (np.uint16, (2, NUM_STREETS, len(ACTION_KINDS))),  # [player, street, kind]
    "three_bet": (np.uint8, (2,)),  # player re-raised an open-raise preflop (second voluntary raise)
    "three_bet_opportunity": (np.uint8, (2,)),  # player acted facing exactly one preflop raise
}
# Per-decision latencies in play order: flat, one array per player (not aligned with the hand axis).
_LATENCY_COLUMNS = ("latency_a_ms", "latency_b_ms")


@dataclass
class HandArrays:
    """All per-hand columns of one match (see module docstring for the layout)."""

    profit_chips: np.ndarray
    a_on_button: np.ndarray
    showdown: np.ndarray
    ended_street: np.ndarray
    pot_chips: np.ndarray
    action_counts: np.ndarray
    three_bet: np.ndarray
    three_bet_opportunity: np.ndarray
    latency_a_ms: np.ndarray
    latency_b_ms: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.profit_chips)
        for name, (dtype, trailing) in _PER_HAND_COLUMNS.items():
            column = np.asarray(getattr(self, name), dtype=dtype)
            if column.shape != (n, *trailing):
                raise ValueError(f"column {name} has shape {column.shape}, expected {(n, *trailing)}")
            setattr(self, name, column)
        for name in _LATENCY_COLUMNS:
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float32))

    def __len__(self) -> int:
        return len(self.profit_chips)

    # ------------------------------------------------------------------ derived views
    @property
    def end_category(self) -> np.ndarray:
        """Per hand: 0-3 = ended by a fold on that street, 4 = went to showdown (``END_CATEGORIES``)."""
        return np.where(self.showdown, SHOWDOWN_CATEGORY, self.ended_street).astype(np.int8)

    def latency_ms(self, player: int) -> np.ndarray:
        return self.latency_a_ms if player == PLAYER_A else self.latency_b_ms

    # ------------------------------------------------------------------ construction / io
    @classmethod
    def concatenate(cls, parts: Sequence["HandArrays"]) -> "HandArrays":
        """Join session results in the order given (hand axis and latency streams)."""
        columns = {
            name: np.concatenate([getattr(p, name) for p in parts])
            if parts
            else np.zeros((0, *trailing), dtype=dtype)
            for name, (dtype, trailing) in _PER_HAND_COLUMNS.items()
        }
        latencies = {
            name: np.concatenate([getattr(p, name) for p in parts]) if parts else np.zeros(0, np.float32)
            for name in _LATENCY_COLUMNS
        }
        return cls(**columns, **latencies)

    def save(self, path: Path, *, with_latency: bool = False) -> None:
        """Write a compressed ``.npz`` (``np.load`` reads columns lazily, so plots stay cheap).

        Raw per-decision latencies are incompressible floats that dominate the file size and are
        summarised (mean / median / p95) in the run's JSON instead, so they are only written on request.
        """
        names = (*_PER_HAND_COLUMNS, *(_LATENCY_COLUMNS if with_latency else ()))
        np.savez_compressed(path, **{name: getattr(self, name) for name in names})

    @classmethod
    def load(cls, path: Path) -> "HandArrays":
        """Read an ``.npz`` written by :meth:`save` (latency streams are empty if they were not saved)."""
        with np.load(path) as data:
            latencies = {name: data[name] if name in data else np.zeros(0, np.float32) for name in _LATENCY_COLUMNS}
            return cls(**{name: data[name] for name in _PER_HAND_COLUMNS}, **latencies)


class HandArraysBuilder:
    """Accumulates one row per hand with cheap Python appends, then freezes into :class:`HandArrays`."""

    def __init__(self) -> None:
        self._rows: dict[str, list] = {name: [] for name in _PER_HAND_COLUMNS}
        self._latency: dict[str, list[float]] = {name: [] for name in _LATENCY_COLUMNS}

    def add_hand(
        self,
        *,
        profit_chips: int,
        a_on_button: bool,
        showdown: bool,
        ended_street: int,
        pot_chips: int,
        action_counts: np.ndarray,
        three_bet: tuple[int, int],
        three_bet_opportunity: tuple[int, int],
        latency_a_ms: Sequence[float],
        latency_b_ms: Sequence[float],
    ) -> None:
        rows = self._rows
        rows["profit_chips"].append(profit_chips)
        rows["a_on_button"].append(a_on_button)
        rows["showdown"].append(showdown)
        rows["ended_street"].append(ended_street)
        rows["pot_chips"].append(pot_chips)
        rows["action_counts"].append(action_counts)
        rows["three_bet"].append(three_bet)
        rows["three_bet_opportunity"].append(three_bet_opportunity)
        self._latency["latency_a_ms"].extend(latency_a_ms)
        self._latency["latency_b_ms"].extend(latency_b_ms)

    def build(self) -> HandArrays:
        n = len(self._rows["profit_chips"])
        columns = {
            name: np.array(values, dtype=dtype).reshape(n, *trailing)
            for (name, values), (dtype, trailing) in zip(self._rows.items(), _PER_HAND_COLUMNS.values())
        }
        return HandArrays(**columns, **{name: np.array(v, np.float32) for name, v in self._latency.items()})
