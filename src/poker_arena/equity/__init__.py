"""Hand evaluation and equity estimation (fast, numpy-vectorised, validated vs PokerKit).

* :mod:`~poker_arena.equity.evaluator` - score 5-7 card hands, batched.
* :mod:`~poker_arena.equity.monte_carlo` - equity vs a random hand (sampled or exact).
* :mod:`~poker_arena.equity.preflop` - precomputed preflop equity for the 169 hand classes.
* :mod:`~poker_arena.equity.features` - observable-only hand/board descriptors.

Every public function only accepts the hero's own hole cards and the public board.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from poker_arena.equity.evaluator import CATEGORY_NAMES, evaluate, evaluate_batch, hand_category
from poker_arena.equity.features import (
    board_texture,
    draw_features,
    hole_card_features,
    made_hand_category,
)
from poker_arena.equity.monte_carlo import (
    equity_vs_random,
    equity_vs_random_with_se,
    exact_equity_vs_random,
)

if TYPE_CHECKING:
    from poker_arena.equity.preflop import (
        build_preflop_table,
        preflop_class,
        preflop_equity,
        preflop_rank_percentile,
    )

# The preflop names are imported lazily (PEP 562) so that `python -m
# poker_arena.equity.preflop` does not find its own module already imported by this package.
_PREFLOP_EXPORTS = frozenset(
    {"build_preflop_table", "preflop_class", "preflop_equity", "preflop_rank_percentile"}
)


def __getattr__(name: str) -> Any:
    if name in _PREFLOP_EXPORTS:
        return getattr(import_module("poker_arena.equity.preflop"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CATEGORY_NAMES",
    "board_texture",
    "build_preflop_table",
    "draw_features",
    "equity_vs_random",
    "equity_vs_random_with_se",
    "evaluate",
    "evaluate_batch",
    "exact_equity_vs_random",
    "hand_category",
    "hole_card_features",
    "made_hand_category",
    "preflop_class",
    "preflop_equity",
    "preflop_rank_percentile",
]
