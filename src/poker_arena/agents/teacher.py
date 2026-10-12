"""The transparent rule-based "teacher" that the supervised learners (EBM, XGBoost) imitate.

Design decisions
----------------
* **A pure function of the feature vector.** :meth:`TeacherPolicy.decide_features` maps
  ``(feature vector, legal mask)`` to an abstract action and nothing else - no hidden state, no
  randomness, no access to the observation. The agent (:class:`TeacherAgent`), the dataset labeller
  and the explanation code all run the very same function, so a learner that sees the same features
  can in principle reproduce the teacher exactly.
* **Equity vs a random hand is the backbone, corrected for what the opponent's action tells us.**
  A player who bets or raises holds a stronger-than-random range, so the equity used to *call* is
  reduced by ``penalty`` equity points that grow with the bet size (``b / (b + half)``) and with every
  extra raise, and is given back in proportion to how aggressive the opponent has been so far
  (a maniac bets with everything). Everything is additive in the features, which keeps the rules
  readable and is friendly to additive models such as an EBM.
* **Exploitation only through public statistics.** Value-bet thresholds slide with the opponent's
  fold-to-bet rate (a calling station is value-bet thinner), calls slide with its aggression and
  preflop raise rate. The statistics are smoothed from hand one, so the early behaviour is neutral.
* **Legal by construction.** Every rule names a preferred action and a fallback; the result is
  resolved against ``obs.abstract_mask`` (an unavailable raise collapses into the all-in, then into
  the fallback; folding is replaced by a check when a free check exists).

All thresholds live in :class:`TeacherParams`; the canonical teacher uses the defaults and the style
presets (:data:`TEACHER_STYLES`) shift them to create diverse sparring partners.

Imitation accuracy is not poker skill, and this teacher is a heuristic, not a solver: its strength is
measured (duplicate-deal BB/100 with confidence intervals, see ``scripts/evaluate_teacher.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from poker_arena.agents.base import Agent, Decision, Explanation
from poker_arena.engine.actions import NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.engine.observation import Observation, OpponentStats
from poker_arena.features import FEATURE_NAMES, NUM_FEATURES, build_features

_NEUTRAL = OpponentStats()  # smoothed prior = what the model assumes about an unknown opponent
AGGRESSION_NEUTRAL = _NEUTRAL.aggression
FOLD_TO_BET_NEUTRAL = _NEUTRAL.fold_to_bet
PFR_NEUTRAL = _NEUTRAL.pfr

CONFIDENCE_SCALE = 0.15
"""A decision whose deciding quantity is this far (in equity) from its threshold has confidence 1."""
WET_SUIT_COUNT = 3
WET_STRAIGHTNESS = 3
DRY_MAX_SUIT_COUNT = 2
DRY_MAX_STRAIGHTNESS = 2
MIN_VALUE_THRESHOLD = 0.52
MAX_VALUE_THRESHOLD = 0.92
_FLAG = 0.5  # binary features are stored as 0.0 / 1.0

_FOLD = int(AbstractAction.FOLD)
_CALL = int(AbstractAction.CHECK_CALL)
_SMALL = int(AbstractAction.RAISE_SMALL)
_POT = int(AbstractAction.RAISE_POT)
_ALL_IN = int(AbstractAction.ALL_IN)
_RAISES = (AbstractAction.RAISE_SMALL, AbstractAction.RAISE_POT, AbstractAction.ALL_IN)


@dataclass(frozen=True, slots=True)
class TeacherParams:
    """Every threshold of the teacher. Defaults = the canonical teacher.

    "pct" = ``preflop_percentile`` (share of starting hands that are weaker; 0.8 = top 20%).
    "equity" = ``equity`` feature (chance to beat a random hand). Chip sizes are in big blinds.

    ===========================  =========  =======================================================
    parameter                    default    meaning
    ===========================  =========  =======================================================
    push_fold_bb                 12.0       effective stack (bb) at or below which preflop is shove/fold
    push_pct                     0.30       shove first-to-act short stack with pct >= this
    open_pct                     0.18       button raises first in with pct >= this (top 82%)
    open_pot_pct                 0.88       ... to pot size (3 bb) instead of 2 bb with pct >= this
    limp_pct                     0.06       button limps with pct >= this, folds below
    iso_pct                      0.60       big blind raises over a limp with pct >= this
    iso_pot_pct                  0.90       ... to pot size with pct >= this
    threebet_pct                 0.82       re-raise an open with pct >= this
    fourbet_pct                  0.96       re-raise a 3-bet or later with pct >= this
    shove_pct                    0.985      shove over a 3-bet or later with pct >= this (aces)
    pre_penalty_max              0.12       preflop: equity points a large raise takes from my equity
    pre_penalty_half             1.0        preflop: bet size (pot fractions) at half the max penalty
    pre_raise_penalty            0.10       preflop: extra points per raise beyond the first
    pre_call_margin              0.03       preflop: equity above the pot-odds price required to call
    pre_pfr_slope                0.50       preflop: points given back per +1.0 of (opp PFR - neutral)
    deep_stack_bb                40.0       raises narrow the range fully only when stacks are this deep
    value_equity                 0.57       bet for value (half pot) with equity >= this (neutral opp)
    value_fold_slope             0.40       threshold moves by this per +1.0 of (opp fold-to-bet - neutral)
    pot_bet_equity               0.84       bet pot size with equity >= this
    wet_pot_equity               0.76       ... on a wet board (flush/straight possible) with equity >= this
    monster_equity               0.93       "nuts": pot bet, or all-in when spr <= shove_spr
    shove_spr                    2.5        stack-to-pot ratio at or below which monsters shove
    semi_bluff_outs              8          straight outs counting as a strong draw (flush draws always do)
    semi_bluff_min_fold          0.15       semi-bluff bets/raises only if opp fold-to-bet >= this
    cbet_min_equity              0.38       dry-flop continuation bet with initiative needs equity >=
    cbet_min_fold                0.35       ... and opp fold-to-bet >= this
    penalty_max                  0.60       postflop: equity points a huge bet takes from my equity
    penalty_half                 0.25       postflop: bet size (pot fractions) at half the max penalty
    raise_penalty                0.06       postflop: extra points per raise beyond the first
    penalty_relief_equity        0.70       the penalty fades linearly to 0 between this equity and 1.0 (nuts beat any range)
    draw_credit                  0.12       postflop: points given back when I hold a strong draw (draws keep their equity)
    aggression_slope             0.80       postflop: points given back per +1.0 of (opp aggression - neutral)
    call_margin                  0.03       postflop: equity above the price required to call
    oop_margin                   0.02       extra margin when out of position (big blind postflop)
    raise_equity                 0.86       raise a bet for value with equity >= this
    raise_extra                  0.04       ... plus this per raise already made this street
    ===========================  =========  =======================================================
    """

    push_fold_bb: float = 12.0
    push_pct: float = 0.30
    open_pct: float = 0.18
    open_pot_pct: float = 0.88
    limp_pct: float = 0.06
    iso_pct: float = 0.60
    iso_pot_pct: float = 0.90
    threebet_pct: float = 0.82
    fourbet_pct: float = 0.96
    shove_pct: float = 0.985
    pre_penalty_max: float = 0.12
    pre_penalty_half: float = 1.0
    pre_raise_penalty: float = 0.10
    pre_call_margin: float = 0.03
    pre_pfr_slope: float = 0.50
    deep_stack_bb: float = 40.0
    value_equity: float = 0.57
    value_fold_slope: float = 0.40
    pot_bet_equity: float = 0.84
    wet_pot_equity: float = 0.76
    monster_equity: float = 0.93
    shove_spr: float = 2.5
    semi_bluff_outs: int = 8
    semi_bluff_min_fold: float = 0.15
    cbet_min_equity: float = 0.38
    cbet_min_fold: float = 0.35
    penalty_max: float = 0.60
    penalty_half: float = 0.25
    raise_penalty: float = 0.06
    penalty_relief_equity: float = 0.70
    draw_credit: float = 0.12
    aggression_slope: float = 0.80
    call_margin: float = 0.03
    oop_margin: float = 0.02
    raise_equity: float = 0.86
    raise_extra: float = 0.04

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


CANONICAL_PARAMS = TeacherParams()

#: Named variations of the canonical teacher: diverse sparring partners and behaviour policies.
TEACHER_STYLES: dict[str, TeacherParams] = {
    "canonical": CANONICAL_PARAMS,
    "tight": replace(
        CANONICAL_PARAMS,
        open_pct=0.35, limp_pct=0.15, iso_pct=0.72, threebet_pct=0.93, push_pct=0.45,
        pre_call_margin=0.06, pre_penalty_max=0.16,
        value_equity=0.68, pot_bet_equity=0.88, raise_equity=0.90, cbet_min_equity=0.50,
        call_margin=0.06, penalty_max=0.75, aggression_slope=0.60,
    ),
    "loose": replace(
        CANONICAL_PARAMS,
        open_pct=0.0, limp_pct=0.0, iso_pct=0.40, threebet_pct=0.75, push_pct=0.15,
        pre_call_margin=-0.02, pre_penalty_max=0.08,
        value_equity=0.54, pot_bet_equity=0.78, raise_equity=0.80, cbet_min_equity=0.28,
        call_margin=-0.02, penalty_max=0.35,
    ),
    "aggressive": replace(
        CANONICAL_PARAMS,
        open_pct=0.05, open_pot_pct=0.55, iso_pct=0.40, iso_pot_pct=0.70, threebet_pct=0.70,
        fourbet_pct=0.90, value_equity=0.54, value_fold_slope=0.15, pot_bet_equity=0.72,
        wet_pot_equity=0.66, monster_equity=0.88, semi_bluff_min_fold=0.05,
        cbet_min_equity=0.25, cbet_min_fold=0.20, raise_equity=0.74,
        call_margin=0.0, penalty_max=0.40, aggression_slope=0.60,
    ),
    "passive": replace(
        CANONICAL_PARAMS,
        open_pct=0.30, open_pot_pct=0.97, limp_pct=0.05, iso_pct=0.85, iso_pot_pct=0.98,
        threebet_pct=0.96, fourbet_pct=0.99, value_equity=0.78, pot_bet_equity=0.92,
        wet_pot_equity=0.86, monster_equity=0.96, semi_bluff_min_fold=2.0, cbet_min_fold=2.0,
        raise_equity=0.95, call_margin=0.0, penalty_max=0.35, pre_call_margin=0.0,
    ),
}


def teacher_params(style: str = "canonical", overrides: Mapping[str, float] | None = None) -> TeacherParams:
    """Parameters of a named style, optionally with individual thresholds replaced."""
    try:
        params = TEACHER_STYLES[style]
    except KeyError:
        raise KeyError(f"Unknown teacher style {style!r}; known: {sorted(TEACHER_STYLES)}") from None
    return replace(params, **dict(overrides)) if overrides else params


# --------------------------------------------------------------------------------------------
# Decisions and their human-readable explanations
# --------------------------------------------------------------------------------------------
RULE_TEXT: dict[str, str] = {
    "pre_push": "Short stack ({stack_bb:.0f} bb): shove, hand strength {pct:.0%} is above the {threshold:.0%} push line.",
    "pre_fold_short": "Short stack ({stack_bb:.0f} bb): hand strength {pct:.0%} is below the {threshold:.0%} push line, so fold.",
    "pre_open_pot": "Button with a strong hand ({pct:.0%} >= {threshold:.0%}): open to pot size.",
    "pre_open": "Button: hand strength {pct:.0%} clears the {threshold:.0%} opening line, so raise.",
    "pre_limp": "Button: too weak to raise ({pct:.0%} < {threshold:.0%}) but cheap to complete, so limp.",
    "pre_fold_trash": "Button: hand strength {pct:.0%} is among the worst starting hands (< {threshold:.0%}), so fold.",
    "pre_iso_pot": "Big blind over a limp with a strong hand ({pct:.0%} >= {threshold:.0%}): raise to pot size.",
    "pre_iso": "Big blind over a limp: hand strength {pct:.0%} >= {threshold:.0%}, so raise for value.",
    "pre_check_option": "Big blind over a limp: hand strength {pct:.0%} < {threshold:.0%}, so take the free check.",
    "pre_shove_premium": "Premium hand ({pct:.0%}): put all the chips in over the raise.",
    "pre_3bet": "Strong hand ({pct:.0%} >= {threshold:.0%}): re-raise (3-bet) the open.",
    "pre_4bet": "Very strong hand ({pct:.0%} >= {threshold:.0%}): re-raise again.",
    "pre_call": "Call: equity {equity:.0%} less {penalty:.0%} for the raiser's range is {adj_equity:.0%}, above the {required_equity:.0%} price.",
    "pre_fold": "Fold: equity {equity:.0%} less {penalty:.0%} for the raiser's range is {adj_equity:.0%}, below the {required_equity:.0%} price.",
    "monster_shove": "Monster ({equity:.0%} equity) with stack-to-pot ratio {spr:.1f}: all-in.",
    "monster_bet": "Monster ({equity:.0%} equity): bet pot size to build the pot.",
    "value_bet_big": "Strong hand ({equity:.0%} equity) on a {texture} board: bet pot size for value and protection.",
    "value_bet": "Value bet: equity {equity:.0%} clears the {threshold:.0%} value line (opponent folds {fold_to_bet:.0%} to bets).",
    "semi_bluff_bet": "Semi-bluff: strong draw with {fold_to_bet:.0%} opponent fold rate, bet half pot.",
    "cbet_bluff": "Continuation bet: I raised before, the board is dry and the opponent folds {fold_to_bet:.0%} of the time.",
    "check_weak": "Check: equity {equity:.0%} is below the {threshold:.0%} value line and no bluff applies.",
    "raise_value": "Raise for value: equity {equity:.0%} >= {threshold:.0%} even against a betting range.",
    "raise_semi_bluff": "Semi-bluff raise: strong draw ({equity:.0%} equity) and the opponent folds {fold_to_bet:.0%} of the time.",
    "call_odds": "Call: equity {equity:.0%} less {penalty:.0%} for the bettor's range is {adj_equity:.0%}, above the {required_equity:.0%} price (pot odds {pot_odds:.0%}).",
    "fold_odds": "Fold: equity {equity:.0%} less {penalty:.0%} for the bettor's range is {adj_equity:.0%}, below the {required_equity:.0%} price (pot odds {pot_odds:.0%}).",
}
"""Sentence template of every deciding rule; filled with :attr:`TeacherDecision.numbers`."""


@dataclass(frozen=True, slots=True)
class TeacherDecision:
    """What the teacher chose and why.

    ``rules``   situation tags followed by the deciding rule (last element), e.g.
                ``("postflop", "facing_bet", "call_odds")``.
    ``margin``  how far (in equity or percentile points, >= 0) the deciding quantity is from the
                threshold that would have changed the decision: a natural measure of label noise
                near decision boundaries (use it as a sample weight or to find hard examples).
    ``numbers`` the key quantities behind the rule (equity, pot odds, thresholds, ...).
    """

    action: AbstractAction
    rules: tuple[str, ...]
    margin: float
    numbers: dict[str, float] = field(default_factory=dict)

    @property
    def rule(self) -> str:
        return self.rules[-1]

    @property
    def confidence(self) -> float:
        """Margin mapped to 0..1 (1 = clearly inside the rule's region)."""
        return min(1.0, self.margin / CONFIDENCE_SCALE)

    def summary(self) -> str:
        """One readable sentence explaining the decision."""
        template = RULE_TEXT[self.rule]
        return template.format_map({"texture": "wet" if self.numbers.get("wet") else "dry", **self.numbers})

    def to_explanation(self) -> Explanation:
        details: dict[str, Any] = {
            "action": self.action.name.lower(),
            "rule": self.rule,
            "rules": list(self.rules),
            "margin": self.margin,
            "confidence": self.confidence,
            "numbers": dict(self.numbers),
        }
        return Explanation(self.summary(), details)


def _resolve(preferred: AbstractAction, fallback: AbstractAction, legal: tuple[bool, ...]) -> AbstractAction:
    """Turn a wish into a legal abstract action.

    A raise size that is not in the mask has collapsed into the all-in (same chips), which is the
    intended action; with no raise possible at all the ``fallback`` applies. Folding is replaced by a
    check when a free check exists.
    """
    if preferred in _RAISES:
        if legal[preferred]:
            return preferred
        if legal[_ALL_IN]:
            return AbstractAction.ALL_IN
        return _resolve(fallback, AbstractAction.CHECK_CALL, legal)
    if preferred is AbstractAction.FOLD and not legal[_FOLD]:
        return AbstractAction.CHECK_CALL
    return preferred


def _range_penalty(equity: float, penalty: float, relief_equity: float) -> float:
    """Equity points a stronger-than-random opponent range takes from ``equity``.

    The shrink fades out linearly above ``relief_equity``: a hand that beats 97% of random hands also
    beats most strong ranges, and the absolute nuts (equity 1) must never be talked out of a call.
    """
    return penalty * min(1.0, (1.0 - equity) / (1.0 - relief_equity))


class TeacherPolicy:
    """The teacher as a pure function ``(features, legal mask) -> TeacherDecision``."""

    def __init__(self, params: TeacherParams = CANONICAL_PARAMS, style: str = "canonical") -> None:
        self.params = params
        self.style = style

    # ----------------------------------------------------------------------------- entry points
    def decide(
        self,
        source: Observation | NDArray[np.floating] | Mapping[str, float],
        legal_mask: NDArray[np.bool_] | None = None,
    ) -> TeacherDecision:
        """Decide from an :class:`Observation`, or from a feature vector/mapping plus a legal mask."""
        if isinstance(source, Observation):
            return self.decide_features(build_features(source), source.abstract_mask)
        if legal_mask is None:
            raise ValueError("A legal-action mask is required when deciding from features")
        return self.decide_features(source, legal_mask)

    def decide_features(
        self, features: NDArray[np.floating] | Mapping[str, float], legal_mask: NDArray[np.bool_]
    ) -> TeacherDecision:
        """The decision for a feature vector (order = ``FEATURE_NAMES``) and a length-5 legal mask."""
        if isinstance(features, Mapping):
            f = dict(features)
        else:
            if len(features) != NUM_FEATURES:
                raise ValueError(f"Expected {NUM_FEATURES} features, got {len(features)}")
            f = dict(zip(FEATURE_NAMES, np.asarray(features, dtype=np.float32).tolist(), strict=True))
        if len(legal_mask) != NUM_ABSTRACT_ACTIONS:
            raise ValueError(f"Legal mask must have {NUM_ABSTRACT_ACTIONS} entries")
        legal = tuple(bool(v) for v in legal_mask)
        if f["street"] < _FLAG:
            return self._preflop(f, legal)
        return self._postflop(f, legal)

    # -------------------------------------------------------------------------------- preflop
    def _preflop(self, f: dict[str, float], legal: tuple[bool, ...]) -> TeacherDecision:
        p = self.params
        pct = f["preflop_percentile"]
        raises = int(f["num_raises_street"])
        stack_bb = f["effective_stack_bb"] + f["to_call_bb"]
        short = stack_bb <= p.push_fold_bb
        base = {"pct": pct, "equity": f["equity"], "stack_bb": stack_bb}

        def pick(
            preferred: AbstractAction, rule: str, margin: float, threshold: float,
            fallback: AbstractAction = AbstractAction.CHECK_CALL,
        ) -> TeacherDecision:
            action = _resolve(preferred, fallback, legal)
            return TeacherDecision(
                action, ("preflop", rule), abs(margin), {**base, "threshold": threshold}
            )

        if raises == 0 and f["is_button"] > _FLAG:  # button / small blind acts first
            if short:
                if pct >= p.push_pct:
                    return pick(AbstractAction.ALL_IN, "pre_push", pct - p.push_pct, p.push_pct)
                return pick(AbstractAction.FOLD, "pre_fold_short", pct - p.push_pct, p.push_pct)
            if pct >= p.open_pot_pct:
                return pick(AbstractAction.RAISE_POT, "pre_open_pot", pct - p.open_pot_pct, p.open_pot_pct)
            if pct >= p.open_pct:
                return pick(AbstractAction.RAISE_SMALL, "pre_open", pct - p.open_pct, p.open_pct)
            if pct >= p.limp_pct:
                return pick(AbstractAction.CHECK_CALL, "pre_limp", p.open_pct - pct, p.open_pct)
            return pick(AbstractAction.FOLD, "pre_fold_trash", p.limp_pct - pct, p.limp_pct)

        if raises == 0:  # big blind with the option after a limp
            if short and pct >= p.push_pct:
                return pick(AbstractAction.ALL_IN, "pre_push", pct - p.push_pct, p.push_pct)
            if pct >= p.iso_pot_pct:
                return pick(AbstractAction.RAISE_POT, "pre_iso_pot", pct - p.iso_pot_pct, p.iso_pot_pct)
            if pct >= p.iso_pct:
                return pick(AbstractAction.RAISE_SMALL, "pre_iso", pct - p.iso_pct, p.iso_pct)
            return pick(AbstractAction.CHECK_CALL, "pre_check_option", p.iso_pct - pct, p.iso_pct)

        # Facing a raise: price the call first, because it is the fallback of every re-raise.
        call = self._preflop_call(f, raises, short)
        calling = call.action is AbstractAction.CHECK_CALL
        fallback = AbstractAction.CHECK_CALL if calling else AbstractAction.FOLD
        if raises >= 2 and pct >= p.shove_pct:
            return pick(AbstractAction.ALL_IN, "pre_shove_premium", pct - p.shove_pct, p.shove_pct, AbstractAction.CHECK_CALL)
        if raises == 1 and pct >= p.threebet_pct:
            return pick(AbstractAction.RAISE_POT, "pre_3bet", pct - p.threebet_pct, p.threebet_pct, fallback)
        if raises >= 2 and pct >= p.fourbet_pct:
            return pick(AbstractAction.RAISE_POT, "pre_4bet", pct - p.fourbet_pct, p.fourbet_pct, fallback)
        return TeacherDecision(_resolve(call.action, AbstractAction.CHECK_CALL, legal), call.rules, call.margin, call.numbers)

    def _preflop_call(self, f: dict[str, float], raises: int, short: bool) -> TeacherDecision:
        """Call or fold facing a raise: equity (less the raiser's range strength) against the price."""
        p = self.params
        bet = f["bet_fraction_of_pot"]
        stack_bb = f["effective_stack_bb"] + f["to_call_bb"]
        depth = min(1.0, stack_bb / p.deep_stack_bb)
        penalty = p.pre_penalty_max * bet / (bet + p.pre_penalty_half) * depth
        penalty += p.pre_raise_penalty * (raises - 1)
        penalty = max(0.0, penalty - p.pre_pfr_slope * (f["opp_pfr"] - PFR_NEUTRAL))
        penalty = _range_penalty(f["equity"], penalty, p.penalty_relief_equity)
        adjusted = f["equity"] - penalty
        required = f["pot_odds"] + p.pre_call_margin
        calls = adjusted >= required
        numbers = {
            "pct": f["preflop_percentile"], "equity": f["equity"], "penalty": penalty,
            "adj_equity": adjusted, "required_equity": required, "pot_odds": f["pot_odds"],
        }
        rule = "pre_call" if calls else "pre_fold"
        action = AbstractAction.CHECK_CALL if calls else AbstractAction.FOLD
        return TeacherDecision(action, ("preflop", "facing_raise", rule), abs(adjusted - required), numbers)

    # ------------------------------------------------------------------------------- postflop
    def _postflop(self, f: dict[str, float], legal: tuple[bool, ...]) -> TeacherDecision:
        if f["facing_bet"] > _FLAG:
            return self._facing_bet(f, legal)
        return self._first_or_checked_to(f, legal)

    @staticmethod
    def _is_wet(f: dict[str, float]) -> bool:
        return f["board_max_suit_count"] >= WET_SUIT_COUNT or f["board_straightness"] >= WET_STRAIGHTNESS

    def _strong_draw(self, f: dict[str, float]) -> bool:
        """A flush draw or an open-ended straight draw on the flop/turn (rivers have no draws)."""
        return f["flush_draw"] > _FLAG or f["straight_outs"] >= self.params.semi_bluff_outs

    @staticmethod
    def _combo_draw(f: dict[str, float]) -> bool:
        """A flush draw together with a straight draw: enough outs to raise for fold equity."""
        return f["flush_draw"] > _FLAG and f["straight_outs"] > 0

    def _first_or_checked_to(self, f: dict[str, float], legal: tuple[bool, ...]) -> TeacherDecision:
        """No bet to face: bet for value, semi-bluff / continuation-bet, or check."""
        p = self.params
        equity = f["equity"]
        fold_rate = f["opp_fold_to_bet"]
        wet = self._is_wet(f)
        value_line = min(
            max(p.value_equity + p.value_fold_slope * (fold_rate - FOLD_TO_BET_NEUTRAL), MIN_VALUE_THRESHOLD),
            MAX_VALUE_THRESHOLD,
        )
        numbers = {"equity": equity, "fold_to_bet": fold_rate, "spr": f["spr"], "wet": float(wet)}

        def decide(preferred: AbstractAction, rule: str, value: float, threshold: float) -> TeacherDecision:
            action = _resolve(preferred, AbstractAction.CHECK_CALL, legal)
            return TeacherDecision(
                action, ("postflop", "no_bet_to_face", rule), abs(value - threshold),
                {**numbers, "threshold": threshold},
            )

        if equity >= p.monster_equity:
            action = AbstractAction.ALL_IN if f["spr"] <= p.shove_spr else AbstractAction.RAISE_POT
            return decide(action, "monster_shove" if f["spr"] <= p.shove_spr else "monster_bet", equity, p.monster_equity)
        if equity >= p.pot_bet_equity:
            return decide(AbstractAction.RAISE_POT, "value_bet_big", equity, p.pot_bet_equity)
        if wet and equity >= p.wet_pot_equity:
            return decide(AbstractAction.RAISE_POT, "value_bet_big", equity, p.wet_pot_equity)
        if equity >= value_line:
            return decide(AbstractAction.RAISE_SMALL, "value_bet", equity, value_line)
        has_position_or_initiative = f["is_button"] > _FLAG or f["i_have_initiative"] > _FLAG
        if self._strong_draw(f) and fold_rate >= p.semi_bluff_min_fold and has_position_or_initiative:
            return decide(AbstractAction.RAISE_SMALL, "semi_bluff_bet", fold_rate, p.semi_bluff_min_fold)
        dry = f["board_max_suit_count"] <= DRY_MAX_SUIT_COUNT and f["board_straightness"] <= DRY_MAX_STRAIGHTNESS
        is_flop = f["street"] < 1.5
        if (
            is_flop and f["i_have_initiative"] > _FLAG and dry
            and equity >= p.cbet_min_equity and fold_rate >= p.cbet_min_fold
        ):
            return decide(AbstractAction.RAISE_SMALL, "cbet_bluff", fold_rate, p.cbet_min_fold)
        return decide(AbstractAction.CHECK_CALL, "check_weak", equity, value_line)

    def _facing_bet(self, f: dict[str, float], legal: tuple[bool, ...]) -> TeacherDecision:
        """Facing a bet or raise: re-raise for value / as a semi-bluff, else call or fold by price."""
        p = self.params
        equity = f["equity"]
        bet = f["bet_fraction_of_pot"]
        raises = int(f["num_raises_street"])
        wet = self._is_wet(f)
        penalty = p.penalty_max * bet / (bet + p.penalty_half) + p.raise_penalty * max(0, raises - 1)
        penalty -= p.aggression_slope * (f["opp_aggression"] - AGGRESSION_NEUTRAL)
        if self._strong_draw(f):
            penalty -= p.draw_credit
        penalty = _range_penalty(equity, max(0.0, penalty), p.penalty_relief_equity)
        adjusted = equity - penalty
        required = f["pot_odds"] + p.call_margin + (0.0 if f["is_button"] > _FLAG else p.oop_margin)
        calls = adjusted >= required
        numbers = {
            "equity": equity, "penalty": penalty, "adj_equity": adjusted, "required_equity": required,
            "pot_odds": f["pot_odds"], "bet_fraction": bet, "fold_to_bet": f["opp_fold_to_bet"],
            "spr": f["spr"], "wet": float(wet),
        }
        call_action = AbstractAction.CHECK_CALL if calls else AbstractAction.FOLD
        call_result = TeacherDecision(
            _resolve(call_action, AbstractAction.CHECK_CALL, legal),
            ("postflop", "facing_bet", "call_odds" if calls else "fold_odds"),
            abs(adjusted - required),
            numbers,
        )

        raise_line = p.raise_equity + p.raise_extra * max(0, raises - 1)
        wish: tuple[AbstractAction, str, float, float] | None = None
        if equity >= raise_line:
            size = AbstractAction.RAISE_POT if (equity >= p.monster_equity or wet) else AbstractAction.RAISE_SMALL
            if f["spr"] <= p.shove_spr:
                size = AbstractAction.ALL_IN
            wish = (size, "raise_value", equity, raise_line)
        elif (
            self._combo_draw(f)
            and f["street"] < 2.5
            and raises <= 1
            and f["opp_fold_to_bet"] >= p.semi_bluff_min_fold
        ):
            wish = (AbstractAction.RAISE_SMALL, "raise_semi_bluff", f["opp_fold_to_bet"], p.semi_bluff_min_fold)
        if wish is None:
            return call_result
        preferred, rule, value, threshold = wish
        action = _resolve(preferred, call_action, legal)
        if action not in _RAISES:  # raising is not possible here: the price decides
            return call_result
        return TeacherDecision(
            action, ("postflop", "facing_bet", rule), abs(value - threshold), {**numbers, "threshold": threshold}
        )


# --------------------------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------------------------
class TeacherAgent(Agent):
    """Registry kind ``"teacher"``: plays :class:`TeacherPolicy` and explains each decision."""

    architecture = "Rule-based (equity + pot odds)"
    description = (
        "Transparent reference policy: equity against a random hand, corrected for the opponent's "
        "betting range, compared with pot odds; position-aware preflop ranges."
    )
    supports_explanations = True

    def __init__(
        self,
        name: str | None = None,
        style: str = "canonical",
        overrides: Mapping[str, float] | None = None,
    ) -> None:
        self.style = style
        self.policy = TeacherPolicy(teacher_params(style, overrides), style)
        self.name = name or ("Teacher" if style == "canonical" else f"Teacher-{style}")

    def act(self, obs: Observation) -> Decision:
        return Decision.from_abstract(obs, self.policy.decide(obs).action)

    def explain(self, obs: Observation, decision: Decision) -> Explanation | None:
        return self.policy.decide(obs).to_explanation()
