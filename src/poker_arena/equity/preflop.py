"""Precomputed preflop equity vs a random hand for the 169 starting-hand classes.

A *hand class* ignores which suits are held: 13 pairs (``"77"``), 78 suited hands
(``"T9s"``) and 78 offsuit hands (``"T9o"``). Equities live in
``data/preflop_equity.json`` and are looked up in O(1) - no sampling at decision time.

Regenerate the table with::

    python -m poker_arena.equity.preflop --samples 200000 --seed 0
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from poker_arena.cards import RANKS
from poker_arena.equity._inputs import parse_hole
from poker_arena.equity.monte_carlo import equity_vs_random_with_se

TABLE_PATH = Path(__file__).parent / "data" / "preflop_equity.json"
NUM_STARTING_HANDS = 52 * 51 // 2  # 1326 two-card combinations
_SUITS_PER_RANK = 4
_PAIR_COMBOS, _SUITED_COMBOS, _OFFSUIT_COMBOS = 6, 4, 12
_DEFAULT_SAMPLES = 200_000


def _class_name(high: int, low: int, suited: bool) -> str:
    if high == low:
        return RANKS[high] * 2
    return f"{RANKS[high]}{RANKS[low]}{'s' if suited else 'o'}"


def _enumerate_classes() -> tuple[tuple[tuple[int, int, bool], ...], tuple[int, ...]]:
    """(high rank, low rank, suited) of all 169 classes, strongest-rank first, + combo counts."""
    keys: list[tuple[int, int, bool]] = []
    combos: list[int] = []
    for high in range(len(RANKS) - 1, -1, -1):
        for low in range(high, -1, -1):
            if high == low:
                keys.append((high, low, False))
                combos.append(_PAIR_COMBOS)
            else:
                keys.append((high, low, True))
                combos.append(_SUITED_COMBOS)
                keys.append((high, low, False))
                combos.append(_OFFSUIT_COMBOS)
    return tuple(keys), tuple(combos)


_CLASS_KEYS, _CLASS_COMBOS = _enumerate_classes()
PREFLOP_CLASSES: tuple[str, ...] = tuple(_class_name(*key) for key in _CLASS_KEYS)
"""The 169 class names, by high then low rank: ``"AA"``, ``"AKs"``, ``"AKo"``, ``"AQs"``, ..."""
_CLASS_INDEX = {key: i for i, key in enumerate(_CLASS_KEYS)}


def _class_index(hole: Sequence[int | str]) -> int:
    first, second = sorted(parse_hole(hole), reverse=True)
    high, low = first // _SUITS_PER_RANK, second // _SUITS_PER_RANK
    suited = high != low and first % _SUITS_PER_RANK == second % _SUITS_PER_RANK
    return _CLASS_INDEX[high, low, suited]


def preflop_class(hole: Sequence[int | str]) -> str:
    """Starting-hand class of two hole cards: ``"AA"``, ``"AKs"`` (suited), ``"T9o"`` (offsuit)."""
    return PREFLOP_CLASSES[_class_index(hole)]


@lru_cache(maxsize=1)
def _load_table() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(equity per class, strictly-weaker-combo fraction per class), both in class order."""
    if not TABLE_PATH.exists():
        raise FileNotFoundError(
            f"{TABLE_PATH} is missing; regenerate it with "
            "`python -m poker_arena.equity.preflop --samples 200000 --seed 0`"
        )
    stored = json.loads(TABLE_PATH.read_text(encoding="utf-8"))["equity"]
    equity = np.array([stored[name] for name in PREFLOP_CLASSES], dtype=np.float64)
    combos = np.array(_CLASS_COMBOS, dtype=np.float64)
    weaker = np.array([combos[equity < value].sum() for value in equity]) / NUM_STARTING_HANDS
    return equity, weaker


def preflop_equity(hole: Sequence[int | str]) -> float:
    """Equity of two hole cards against one random hand (table lookup, no sampling)."""
    return float(_load_table()[0][_class_index(hole)])


def preflop_rank_percentile(hole: Sequence[int | str]) -> float:
    """Fraction (0..1) of all 1326 starting hands that are strictly weaker than ``hole``.

    Hands are ranked by :func:`preflop_equity`; every class is weighted by its number of
    card combinations (pair 6, suited 4, offsuit 12), so 0.0 is the worst hand (``32o``)
    and values near 1.0 are premium hands (``AA``).
    """
    return float(_load_table()[1][_class_index(hole)])


def _representative_hole(high: int, low: int, suited: bool) -> tuple[int, int]:
    """Concrete hole cards of a class (clubs; the offsuit/pair second card is a diamond)."""
    first = high * _SUITS_PER_RANK
    second = low * _SUITS_PER_RANK + (0 if suited else 1)
    return first, second


def build_preflop_table(
    n_samples_per_class: int = _DEFAULT_SAMPLES,
    seed: int = 0,
    output_path: Path | None = TABLE_PATH,
) -> dict[str, Any]:
    """Estimate every class's equity vs a random hand by Monte Carlo and save it as JSON.

    Class ``i`` is simulated with its own stream ``default_rng([seed, i])``, so the table is
    reproducible and independent of evaluation order.

    Args:
        n_samples_per_class: random deals per class (>= 2).
        seed: base seed.
        output_path: JSON destination; ``None`` skips writing.

    Returns:
        The JSON payload: ``metadata``, ``equity`` and ``std_error`` (class name -> float).
    """
    equity: dict[str, float] = {}
    std_error: dict[str, float] = {}
    for i, (key, name) in enumerate(zip(_CLASS_KEYS, PREFLOP_CLASSES, strict=True)):
        rng = np.random.default_rng([seed, i])
        value, error = equity_vs_random_with_se(
            _representative_hole(*key), (), n_samples_per_class, rng
        )
        equity[name] = round(value, 6)
        std_error[name] = round(error, 6)
    payload: dict[str, Any] = {
        "metadata": {
            "method": "Monte Carlo vs one uniformly random opponent hand, empty board; "
            "ties count 0.5",
            "n_samples_per_class": n_samples_per_class,
            "seed": seed,
            "date": date.today().isoformat(),
            "n_classes": len(PREFLOP_CLASSES),
        },
        "equity": equity,
        "std_error": std_error,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        if output_path == TABLE_PATH:
            _load_table.cache_clear()
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: rebuild the shipped preflop table."""
    parser = argparse.ArgumentParser(description="Rebuild the preflop equity table.")
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES, help="deals per class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=TABLE_PATH, help="JSON destination")
    args = parser.parse_args(argv)
    payload = build_preflop_table(args.samples, args.seed, args.output)
    worst = max(payload["std_error"].values())
    print(f"Wrote {args.output} ({len(payload['equity'])} classes, max std error {worst:.5f})")


if __name__ == "__main__":
    main()
