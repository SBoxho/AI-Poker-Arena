"""``Observation`` -> feature vector, the one function used by dataset generation AND inference.

INFORMATION RULE (anti-cheating boundary): features are computed from the restricted
:class:`~poker_arena.engine.observation.Observation` only - own hole cards, revealed board,
public chip counts, public action history and the opponent statistics from *previous* hands.
Nothing here imports the engine's hidden-state modules; ``tests/test_features_leakage.py`` checks the
behaviour differentially (same public history + different hidden cards => identical vector).

Determinism: the only stochastic ingredient is the Monte-Carlo equity on the flop and turn. Its
random generator is seeded from the (sorted) hole cards and board, so the same observation always
yields the same vector in any process and in training as well as inference. The estimate has a
standard error of at most ``0.5 / sqrt(EQUITY_SAMPLES_FLOP_TURN)`` = 0.016 (about 0.014 on
average); preflop uses an exact-ish lookup table and the river is enumerated exactly (990 opponent
hands). Because the noise is a fixed function of the state, a model trained on these features sees
exactly the same values at play time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from poker_arena.cards import card_to_int
from poker_arena.engine.observation import Observation
from poker_arena.equity import (
    board_texture,
    draw_features,
    equity_vs_random,
    exact_equity_vs_random,
    hole_card_features,
    made_hand_category,
    preflop_equity,
    preflop_rank_percentile,
)
from poker_arena.features.schema import (
    BB_CAP,
    BET_FRACTION_CAP,
    EQUITY_SAMPLES_FLOP_TURN,
    EQUITY_SEED_SALT,
    FEATURE_NAMES,
    LOG_HANDS_CAP,
    NUM_FEATURES,
    NUM_RAISES_HAND_CAP,
    NUM_RAISES_STREET_CAP,
    POT_BB_CAP,
    SPR_CAP,
)

if TYPE_CHECKING:
    import pandas as pd

EQUITY_CACHE_SIZE = 8192
"""Distinct (hole, board) states whose equity is memoised per process (a hand re-asks the same state)."""

_BOARD_SIZE_RIVER = 5
_BET_KINDS = ("bet", "raise")


@lru_cache(maxsize=EQUITY_CACHE_SIZE)
def _equity_vs_random(hole: tuple[int, ...], board: tuple[int, ...]) -> float:
    """Equity vs a random hand for canonical (sorted) cards; deterministic for given cards."""
    if not board:
        return preflop_equity(hole)
    if len(board) == _BOARD_SIZE_RIVER:
        return exact_equity_vs_random(hole, board)
    seed = np.random.SeedSequence([EQUITY_SEED_SALT, len(board), *hole, *board])
    return equity_vs_random(hole, board, EQUITY_SAMPLES_FLOP_TURN, np.random.default_rng(seed))


def hand_equity(obs: Observation) -> float:
    """Equity of ``obs.my_hole`` on ``obs.board`` against a random hand (the ``equity`` feature)."""
    hole = tuple(sorted(card_to_int(c) for c in obs.my_hole))
    board = tuple(sorted(card_to_int(c) for c in obs.board))
    return _equity_vs_random(hole, board)


def _clip(value: float, high: float, low: float = 0.0) -> float:
    return min(max(value, low), high)


def _aggression_flags(obs: Observation) -> tuple[int, int, int, int]:
    """(I raised this street, opp raised this street, I have initiative, opp has initiative)."""
    i_raised = opp_raised = 0
    last_prior_by_me: bool | None = None
    for action in obs.history:
        if action.kind not in _BET_KINDS:
            continue
        if action.street < obs.street:
            last_prior_by_me = action.by_me
        elif action.street == obs.street:
            if action.by_me:
                i_raised = 1
            else:
                opp_raised = 1
    return (
        i_raised,
        opp_raised,
        int(last_prior_by_me is True),
        int(last_prior_by_me is False),
    )


def build_features(obs: Observation) -> NDArray[np.float32]:
    """Feature vector (1-D ``float32``, order = :data:`FEATURE_NAMES`) of one decision state."""
    bb = obs.big_blind
    hole = obs.my_hole
    board = obs.board
    to_call = obs.to_call
    pot = obs.pot
    equity = hand_equity(obs)
    holef = hole_card_features(hole)
    boardf = board_texture(board)
    drawf = draw_features(hole, board)
    street = int(obs.street)

    facing = to_call > 0
    pot_after_call = pot + to_call
    pot_odds = to_call / pot_after_call if facing else 0.0
    effective = obs.effective_stack
    i_raised, opp_raised, i_init, opp_init = _aggression_flags(obs)
    my_total = obs.my_committed + obs.my_stack
    opp_total = obs.opp_committed + obs.opp_stack
    stats = obs.opponent_stats

    values = {
        "equity": equity,
        "preflop_percentile": preflop_rank_percentile(hole),
        "made_hand_category": made_hand_category(hole, board),
        "hole_high_rank": holef["high_rank"],
        "hole_low_rank": holef["low_rank"],
        "is_pair": holef["is_pair"],
        "is_suited": holef["is_suited"],
        "hole_gap": holef["gap"],
        "overcards": drawf["overcards"],
        "pair_with_board": drawf["has_pair_with_board"],
        "flush_draw": drawf["flush_draw"],
        "straight_outs": drawf["straight_outs"],
        "street": street,
        "board_paired": boardf["board_paired"],
        "board_max_suit_count": boardf["board_max_suit_count"],
        "board_straightness": boardf["board_straightness"],
        "board_high_rank": boardf["board_high_rank"],
        "is_button": int(obs.is_button),
        "pot_bb": _clip(pot / bb, POT_BB_CAP),
        "my_stack_bb": _clip(obs.my_stack / bb, BB_CAP),
        "opp_stack_bb": _clip(obs.opp_stack / bb, BB_CAP),
        "effective_stack_bb": _clip(effective / bb, BB_CAP),
        "spr": _clip(effective / max(pot_after_call, 1), SPR_CAP),
        "my_commit_frac": obs.my_committed / my_total if my_total else 0.0,
        "opp_commit_frac": obs.opp_committed / opp_total if opp_total else 0.0,
        "facing_bet": int(facing),
        "facing_all_in": int(facing and obs.opp_stack == 0),
        "can_raise": int(obs.legal.can_raise),
        "to_call_bb": _clip(to_call / bb, BB_CAP),
        "pot_odds": pot_odds,
        "equity_minus_pot_odds": equity - pot_odds,
        "bet_fraction_of_pot": _clip(to_call / max(pot - to_call, 1), BET_FRACTION_CAP) if facing else 0.0,
        "call_stack_fraction": to_call / obs.my_stack if facing and obs.my_stack else 0.0,
        "num_raises_street": min(obs.num_raises_this_street, NUM_RAISES_STREET_CAP),
        "num_raises_hand": min(obs.num_raises, NUM_RAISES_HAND_CAP),
        "i_raised_this_street": i_raised,
        "opp_raised_this_street": opp_raised,
        "i_have_initiative": i_init,
        "opp_has_initiative": opp_init,
        "opp_vpip": stats.vpip,
        "opp_pfr": stats.pfr,
        "opp_aggression": stats.aggression,
        "opp_fold_to_bet": stats.fold_to_bet,
        "opp_showdown_rate": stats.showdown_rate,
        "opp_log_hands": _clip(math.log1p(stats.hands), LOG_HANDS_CAP),
    }
    # Indexing by name (instead of positional tuples) makes a typo or a schema/builder mismatch a
    # loud KeyError / length error rather than a silently shifted column.
    if len(values) != NUM_FEATURES:  # pragma: no cover - guards edits of the table above
        raise RuntimeError("build_features and the feature schema disagree")
    return np.fromiter((values[name] for name in FEATURE_NAMES), dtype=np.float32, count=NUM_FEATURES)


def vector_to_dict(vector: Sequence[float] | NDArray[np.floating]) -> dict[str, float]:
    """Name -> value mapping of a feature vector (plain Python floats)."""
    if len(vector) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, got {len(vector)}")
    return dict(zip(FEATURE_NAMES, np.asarray(vector, dtype=np.float32).tolist(), strict=True))


def build_feature_dict(obs: Observation) -> dict[str, float]:
    """Like :func:`build_features` but as ``{feature name: value}`` (convenient for explanations)."""
    return vector_to_dict(build_features(obs))


def build_features_matrix(observations: Iterable[Observation]) -> NDArray[np.float32]:
    """Feature matrix with one row per observation, shape ``(n, NUM_FEATURES)``."""
    rows = [build_features(obs) for obs in observations]
    if not rows:
        return np.empty((0, NUM_FEATURES), dtype=np.float32)
    return np.vstack(rows)


def features_frame(observations: Iterable[Observation]) -> "pd.DataFrame":
    """Feature table (``float32`` columns named by :data:`FEATURE_NAMES`), one row per observation."""
    import pandas as pd

    return pd.DataFrame(build_features_matrix(observations), columns=list(FEATURE_NAMES))


__all__ = [
    "build_feature_dict",
    "build_features",
    "build_features_matrix",
    "features_frame",
    "hand_equity",
    "vector_to_dict",
]
