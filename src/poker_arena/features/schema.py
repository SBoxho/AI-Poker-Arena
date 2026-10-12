"""The feature schema: names, groups, one-sentence descriptions and documented value ranges.

This module is the single source of truth shared by training and inference. A model is only valid
for the schema it was trained on, so trained agents must store :func:`schema_hash` (and
:data:`FEATURE_SCHEMA_VERSION`) in their metadata and refuse to load against a different one.

Every feature is a plain number (binary flags are 0/1) so a feature vector is one flat ``float32``
array, and every feature has a *documented range* that the test-suite enforces. Quantities that are
naturally unbounded (stacks, pot) are expressed in big blinds and clipped at a named cap, so a
shape plot of an EBM never has to extrapolate over a handful of extreme rows.

Reading guide for the explainability UI: the ``description`` of each feature is written as a
sentence a poker player can read next to a shape plot or a SHAP bar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

#: Bump when a feature is added/removed/redefined. Stored in every model's metadata.
FEATURE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------------------------
# Named constants that are part of the feature *semantics* (and therefore of the schema hash).
# ---------------------------------------------------------------------------------------------
EQUITY_SAMPLES_FLOP_TURN = 1000
"""Monte-Carlo deals for the equity feature on the flop and turn (standard error <= ~0.016)."""

EQUITY_SEED_SALT = 20_240_611
"""Mixed into the per-state equity RNG seed; changing it changes every sampled equity value."""

BB_CAP = 400.0
"""Stacks / call sizes are clipped at this many big blinds (4x the default 100 bb stack)."""
POT_BB_CAP = 2 * BB_CAP
SPR_CAP = 100.0
BET_FRACTION_CAP = 5.0
"""A bet bigger than 5x the pot is treated like a 5x pot overbet."""
NUM_RAISES_STREET_CAP = 6
NUM_RAISES_HAND_CAP = 12
LOG_HANDS_CAP = 12.0
"""``log(1 + hands observed)`` is clipped here (about 160,000 hands)."""

# Feature groups (used by the UI to colour / fold the shape plots).
G_HAND = "hand_strength"
G_BOARD = "board"
G_POSITION = "position"
G_MONEY = "money"
G_BETTING = "betting"
G_HISTORY = "history"
G_OPPONENT = "opponent"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature: its name, group, human description and documented inclusive value range."""

    name: str
    group: str
    description: str
    lo: float
    hi: float
    binary: bool = False


def _binary(name: str, group: str, description: str) -> FeatureSpec:
    return FeatureSpec(name, group, description, 0.0, 1.0, binary=True)


_SPECS: tuple[FeatureSpec, ...] = (
    # ---- hand strength -------------------------------------------------------------------------
    FeatureSpec(
        "equity", G_HAND,
        "Chance (0-1) that my hand beats a uniformly random opponent hand at showdown, counting "
        "ties as half and all cards still to come.", 0.0, 1.0,
    ),
    FeatureSpec(
        "preflop_percentile", G_HAND,
        "Share (0-1) of all starting hands that are strictly weaker than my two hole cards "
        "(1 = pocket aces).", 0.0, 1.0,
    ),
    FeatureSpec(
        "made_hand_category", G_HAND,
        "Best five-card hand made so far: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, "
        "5 flush, 6 full house, 7 quads, 8 straight flush (preflop: 1 only for a pocket pair).",
        0.0, 8.0,
    ),
    FeatureSpec("hole_high_rank", G_HAND, "Rank of my higher hole card (0 = deuce ... 12 = ace).", 0.0, 12.0),
    FeatureSpec("hole_low_rank", G_HAND, "Rank of my lower hole card (0 = deuce ... 12 = ace).", 0.0, 12.0),
    _binary("is_pair", G_HAND, "1 if my hole cards are a pocket pair."),
    _binary("is_suited", G_HAND, "1 if my two hole cards share a suit."),
    FeatureSpec(
        "hole_gap", G_HAND,
        "Rank distance between my hole cards (0 pair, 1 connectors, 2 one-gapper, ...).", 0.0, 12.0,
    ),
    FeatureSpec(
        "overcards", G_HAND,
        "How many of my hole cards (0-2) are higher than every board card; 0 before the flop.",
        0.0, 2.0,
    ),
    _binary("pair_with_board", G_HAND, "1 if one of my hole cards pairs a card on the board."),
    _binary("flush_draw", G_HAND, "1 if I hold four cards of one suit (using a hole card) on the flop or turn."),
    FeatureSpec(
        "straight_outs", G_HAND,
        "Cards that would complete a straight using a hole card, as 0 / 4 (gutshot) / 8 (open-ended) "
        "on the flop or turn.", 0.0, 8.0,
    ),
    # ---- board ---------------------------------------------------------------------------------
    FeatureSpec("street", G_BOARD, "Betting round: 0 preflop, 1 flop, 2 turn, 3 river.", 0.0, 3.0),
    _binary("board_paired", G_BOARD, "1 if the board shows a pair or better (any rank twice)."),
    FeatureSpec(
        "board_max_suit_count", G_BOARD,
        "Most board cards of a single suit (3 = flush is possible, 0 before the flop).", 0.0, 5.0,
    ),
    FeatureSpec(
        "board_straightness", G_BOARD,
        "Most board ranks inside one five-rank window (3 = a straight is two cards away).", 0.0, 5.0,
    ),
    FeatureSpec(
        "board_high_rank", G_BOARD,
        "Rank of the highest board card (0 = deuce ... 12 = ace); -1 before the flop.", -1.0, 12.0,
    ),
    # ---- position ------------------------------------------------------------------------------
    _binary(
        "is_button", G_POSITION,
        "1 if I am on the button (small blind): first to act preflop but last to act on every later street.",
    ),
    # ---- money (all chip amounts in big blinds) ------------------------------------------------
    FeatureSpec("pot_bb", G_MONEY, "Chips in the middle including bets this street, in big blinds.", 0.0, POT_BB_CAP),
    FeatureSpec("my_stack_bb", G_MONEY, "My chips behind, in big blinds.", 0.0, BB_CAP),
    FeatureSpec("opp_stack_bb", G_MONEY, "The opponent's chips behind, in big blinds.", 0.0, BB_CAP),
    FeatureSpec(
        "effective_stack_bb", G_MONEY,
        "Chips that can still go in after the bet I face is called (the shorter stack), in big blinds.",
        0.0, BB_CAP,
    ),
    FeatureSpec(
        "spr", G_MONEY,
        "Stack-to-pot ratio: effective stack divided by the pot after I call (low = pot-committed).",
        0.0, SPR_CAP,
    ),
    FeatureSpec(
        "my_commit_frac", G_MONEY,
        "Share (0-1) of my chips for this hand that are already in the pot.", 0.0, 1.0,
    ),
    FeatureSpec(
        "opp_commit_frac", G_MONEY,
        "Share (0-1) of the opponent's chips for this hand that are already in the pot.", 0.0, 1.0,
    ),
    # ---- betting (the decision I face) ---------------------------------------------------------
    _binary("facing_bet", G_BETTING, "1 if I must put chips in to continue (a bet, raise or the small blind to complete)."),
    _binary("facing_all_in", G_BETTING, "1 if I face a bet and the opponent has no chips left behind."),
    _binary("can_raise", G_BETTING, "1 if raising is possible (stacks and betting rules allow it)."),
    FeatureSpec("to_call_bb", G_BETTING, "Chips I must add to call, in big blinds (0 when I can check).", 0.0, BB_CAP),
    FeatureSpec(
        "pot_odds", G_BETTING,
        "Share of the final pot I would pay to call: call / (pot + call) = the equity needed to break even.",
        0.0, 1.0,
    ),
    FeatureSpec(
        "equity_minus_pot_odds", G_BETTING,
        "My equity minus the pot odds: positive means calling beats a random hand's break-even price.",
        -1.0, 1.0,
    ),
    FeatureSpec(
        "bet_fraction_of_pot", G_BETTING,
        "Size of the bet or raise I face as a fraction of the pot before it (0.5 = half pot, 1 = pot).",
        0.0, BET_FRACTION_CAP,
    ),
    FeatureSpec(
        "call_stack_fraction", G_BETTING,
        "Share (0-1) of my remaining chips that calling would cost (1 = a call puts me all-in).", 0.0, 1.0,
    ),
    # ---- history -------------------------------------------------------------------------------
    FeatureSpec(
        "num_raises_street", G_HISTORY,
        "Bets and raises made so far on this street by either player.", 0.0, float(NUM_RAISES_STREET_CAP),
    ),
    FeatureSpec(
        "num_raises_hand", G_HISTORY,
        "Bets and raises made so far in the whole hand by either player.", 0.0, float(NUM_RAISES_HAND_CAP),
    ),
    _binary("i_raised_this_street", G_HISTORY, "1 if I have already bet or raised on this street."),
    _binary("opp_raised_this_street", G_HISTORY, "1 if the opponent has already bet or raised on this street."),
    _binary(
        "i_have_initiative", G_HISTORY,
        "1 if I made the last bet or raise on an earlier street (e.g. I was the preflop aggressor on the flop).",
    ),
    _binary("opp_has_initiative", G_HISTORY, "1 if the opponent made the last bet or raise on an earlier street."),
    # ---- opponent model (public information from previous hands only) ---------------------------
    FeatureSpec("opp_vpip", G_OPPONENT, "Share of hands the opponent voluntarily puts chips in preflop (smoothed).", 0.0, 1.0),
    FeatureSpec("opp_pfr", G_OPPONENT, "Share of hands the opponent raises preflop (smoothed).", 0.0, 1.0),
    FeatureSpec(
        "opp_aggression", G_OPPONENT,
        "Opponent's postflop bets+raises divided by bets+raises+calls (smoothed).", 0.0, 1.0,
    ),
    FeatureSpec(
        "opp_fold_to_bet", G_OPPONENT,
        "Share of the opponent's decisions facing a bet or raise that were folds (smoothed).", 0.0, 1.0,
    ),
    FeatureSpec("opp_showdown_rate", G_OPPONENT, "Share of hands that reached a showdown (smoothed).", 0.0, 1.0),
    FeatureSpec(
        "opp_log_hands", G_OPPONENT,
        "log(1 + number of hands observed against this opponent): how much to trust the opponent statistics.",
        0.0, LOG_HANDS_CAP,
    ),
)

FEATURE_SPECS: tuple[FeatureSpec, ...] = _SPECS
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in _SPECS)
"""Ordered feature names: column ``i`` of every feature matrix is ``FEATURE_NAMES[i]``."""
NUM_FEATURES = len(FEATURE_NAMES)
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
FEATURE_DESCRIPTIONS: dict[str, str] = {spec.name: spec.description for spec in _SPECS}
FEATURE_RANGES: dict[str, tuple[float, float]] = {spec.name: (spec.lo, spec.hi) for spec in _SPECS}
BINARY_FEATURES: tuple[str, ...] = tuple(spec.name for spec in _SPECS if spec.binary)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    group: tuple(spec.name for spec in _SPECS if spec.group == group)
    for group in dict.fromkeys(spec.group for spec in _SPECS)
}
FEATURE_GROUP_OF: dict[str, str] = {spec.name: spec.group for spec in _SPECS}

if len(set(FEATURE_NAMES)) != NUM_FEATURES:  # pragma: no cover - guards edits of the table above
    raise RuntimeError("Duplicate feature names in the schema")


def schema_dict() -> dict:
    """The complete, JSON-serialisable definition of the schema (what :func:`schema_hash` hashes)."""
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "equity_samples_flop_turn": EQUITY_SAMPLES_FLOP_TURN,
        "equity_seed_salt": EQUITY_SEED_SALT,
        "features": [[s.name, s.group, s.lo, s.hi, s.binary] for s in _SPECS],
    }


def schema_hash() -> str:
    """Short fingerprint of names, order, ranges and equity settings (16 hex characters).

    Descriptions are deliberately excluded: rewording a sentence must not invalidate a trained model.
    """
    payload = json.dumps(schema_dict(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
