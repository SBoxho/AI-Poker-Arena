"""A small, reproducible set of REAL decision situations that the research notebooks explain.

The EBM notebook (01) and the XGBoost/SHAP notebook (02) call :func:`select_example_cases`, so both
explain exactly the same states and their explanations can be compared line by line. A case is a real
row of the *validation* split of the teacher-labelled dataset (``artifacts/datasets/teacher_bc_v2``,
hands the models never trained on), chosen by a seeded random draw inside a category such as "strong
value bet" or "fold facing a big bet":

    from poker_arena.explain.notebook_cases import select_example_cases

    for case in select_example_cases():          # 8 cases, deterministic
        print(case.title, "->", case.teacher_action_name, case.teacher_rules)
        print(case.description)                  # plain-words situation, built from the features only
        x, mask = case.features, case.legal_mask # exactly what a model sees

Design decisions
----------------
* **The dataset stores features, not cards.** No hidden information is reconstructed: a situation is
  described only through quantities the agent itself observes (hole-card ranks and suitedness, equity,
  made-hand category, draws, board texture, pot, stacks, price of the call, opponent statistics).
* **The teacher is the ground truth.** Every case carries the teacher's decision and the *rule trace*
  from :meth:`~poker_arena.agents.teacher.TeacherPolicy.decide_features` (which rule fired, the numbers
  behind it). Explanations of a learned model can therefore be checked against the rules that really
  produced the label. The stored label is asserted to equal the teacher's decision on the same vector.
* **Deterministic, independent of row order.** Candidates of a category are sorted by
  ``(hand_id, decision_idx)`` before the seeded draw, and one hand is used at most once, so the same
  ``(dataset, seed, n)`` always yields the same cases, and ``n`` smaller cases are a prefix of larger ones.
* **No side effects on import**; the dataset is read only when :func:`select_example_cases` or
  :func:`load_validation_frame` is called.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from poker_arena.agents.teacher import CANONICAL_PARAMS, TeacherDecision, TeacherPolicy
from poker_arena.cards import RANKS
from poker_arena.data.columns import LEGAL_COLUMNS
from poker_arena.data.io import load_dataset
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES
from poker_arena.features import FEATURE_INDEX, FEATURE_NAMES

DEFAULT_DATASET = "teacher_bc_v2"
DEFAULT_SEED = 2026
DEFAULT_N_CASES = 8
VALIDATION_SPLIT = "val"

STREET_NAMES = ("preflop", "flop", "turn", "river")
MADE_HAND_NAMES = (
    "high card", "one pair", "two pair", "three of a kind", "straight", "flush", "full house",
    "four of a kind", "straight flush",
)
ACTION_PHRASES = {
    "fold": "fold", "check_call": "check / call", "raise_small": "raise about half the pot",
    "raise_pot": "raise about the pot", "all_in": "go all-in",
}

CLEAR_MARGIN = 0.05
"""Teacher margin (equity points) above which a decision is clear-cut and not a coin flip."""
MARGINAL_MARGIN = 0.03
"""A decision whose deciding quantity is within this of the threshold counts as marginal."""
BIG_BET_FRACTION = 0.9
"""A bet of at least this fraction of the pot is a 'big bet'."""
RIVER_MAX_MARGIN = 0.15
"""River examples are borderline decisions (margin below this), not trivial folds with hopeless hands."""
STRONG_EQUITY = 0.88
KNOWN_OPPONENT_HANDS = 50.0
"""Opponent statistics are quoted in the description only after this many observed hands."""
FOLDS_RARELY = 0.25
"""Opponent fold-to-bet rate at or below which an opponent 'rarely folds'."""
DEEP_STACK_BB = 40.0
PLAYABLE_STACK_BB = 20.0
"""Post-flop example cases need at least this much effective stack, so the call/raise choice is not a pot-committed all-in."""

# --------------------------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------------------------
Selector = Callable[[pd.DataFrame], "pd.Series[bool]"]


@dataclass(frozen=True, slots=True)
class CaseCategory:
    """One kind of interesting situation: a key, a title, a one-line reason, and a row filter."""

    key: str
    title: str
    why: str
    select: Selector


def _rule_in(frame: pd.DataFrame, *rules: str) -> pd.Series:
    return frame["teacher_rule"].astype(str).isin(rules)


def _playable(frame: pd.DataFrame) -> pd.Series:
    return frame["effective_stack_bb"] >= PLAYABLE_STACK_BB


CATEGORIES: tuple[CaseCategory, ...] = (
    CaseCategory(
        "strong_value_bet", "Strong value bet",
        "A very strong hand on the flop/turn/river bets pot size; the teacher reads equity only.",
        lambda f: _rule_in(f, "value_bet_big") & (f["equity"] >= STRONG_EQUITY) & _playable(f),
    ),
    CaseCategory(
        "thin_value_bet", "Thin value bet",
        "Equity only just clears the (opponent-adjusted) value line: the decision sits on a boundary.",
        lambda f: _rule_in(f, "value_bet") & (f["teacher_margin"] <= MARGINAL_MARGIN) & _playable(f),
    ),
    CaseCategory(
        "semi_bluff", "Semi-bluff with a draw",
        "A flush / open-ended draw bets or raises for fold equity against an opponent who folds enough.",
        lambda f: _rule_in(f, "semi_bluff_bet", "raise_semi_bluff") & (f["teacher_margin"] >= MARGINAL_MARGIN) & _playable(f),
    ),
    CaseCategory(
        "fold_big_bet", "Fold facing a big bet",
        "A pot-sized bet implies a strong range; equity against a random hand overstates the hand.",
        lambda f: _rule_in(f, "fold_odds") & (f["bet_fraction_of_pot"] >= BIG_BET_FRACTION)
        & (f["teacher_margin"] >= CLEAR_MARGIN) & _playable(f),
    ),
    CaseCategory(
        "marginal_call", "Call with marginal equity",
        "Adjusted equity is just above the price of the call: the smallest change flips it.",
        lambda f: _rule_in(f, "call_odds") & (f["teacher_margin"] <= MARGINAL_MARGIN) & _playable(f),
    ),
    CaseCategory(
        "preflop_open", "Preflop open from the button",
        "A routine position-aware preflop decision, read from the starting-hand percentile.",
        lambda f: _rule_in(f, "pre_open") & (f["effective_stack_bb"] >= DEEP_STACK_BB)
        & (f["teacher_margin"] >= CLEAR_MARGIN),
    ),
    CaseCategory(
        "push_fold", "Short-stack push / fold",
        "With 12 big blinds or less the teacher plays shove-or-fold on the starting-hand percentile.",
        lambda f: _rule_in(f, "pre_push", "pre_fold_short"),
    ),
    CaseCategory(
        "river_decision", "River decision facing a bet",
        "No cards to come: equity is exact-ish, the price and the bettor's range decide.",
        lambda f: (f["street"] == 3) & _rule_in(f, "call_odds", "fold_odds")
        & f["teacher_margin"].between(MARGINAL_MARGIN, RIVER_MAX_MARGIN) & _playable(f),
    ),
    CaseCategory(
        "opponent_adjusted_bet", "Value bet that only exists because of the opponent",
        "Equity is below the neutral value line (0.57) but the opponent rarely folds, so the teacher bets thinner.",
        lambda f: _rule_in(f, "value_bet") & (f["equity"] < CANONICAL_PARAMS.value_equity)
        & (f["opp_fold_to_bet"] <= FOLDS_RARELY) & (np.expm1(f["opp_log_hands"]) >= KNOWN_OPPONENT_HANDS) & _playable(f),
    ),
    CaseCategory(
        "cbet_bluff", "Continuation-bet bluff",
        "A pure bluff on a dry flop: weak hand, initiative, opponent folds often. Hard for an additive model.",
        lambda f: _rule_in(f, "cbet_bluff"),
    ),
    CaseCategory(
        "preflop_3bet", "Preflop re-raise (3-bet)",
        "Facing a raise with a top hand: re-raise instead of calling.",
        lambda f: _rule_in(f, "pre_3bet") & (f["teacher_margin"] >= MARGINAL_MARGIN),
    ),
    CaseCategory(
        "monster_shove", "Monster at a low stack-to-pot ratio",
        "Near-nut equity with little behind: the teacher moves all-in.",
        lambda f: _rule_in(f, "monster_shove"),
    ),
)
CATEGORY_KEYS: tuple[str, ...] = tuple(c.key for c in CATEGORIES)


# --------------------------------------------------------------------------------------------
# Situation text (features only)
# --------------------------------------------------------------------------------------------
def _hole_text(f: dict[str, float]) -> str:
    high, low = RANKS[int(f["hole_high_rank"])], RANKS[int(f["hole_low_rank"])]
    if f["is_pair"] > 0.5:
        return f"a pocket pair of {high}s"
    return f"{high}{low} {'suited' if f['is_suited'] > 0.5 else 'offsuit'}"


def _draw_text(f: dict[str, float]) -> str:
    parts = []
    if f["flush_draw"] > 0.5:
        parts.append("a flush draw")
    outs = int(f["straight_outs"])
    if outs >= 8:
        parts.append("an open-ended straight draw (8 outs)")
    elif outs >= 4:
        parts.append("a gutshot straight draw (4 outs)")
    return " and ".join(parts)


def _overcards_text(f: dict[str, float]) -> str:
    n = int(f["overcards"])
    return f"{n} overcard{'s' if n != 1 else ''} to the board, " if n else ""


def _board_text(f: dict[str, float]) -> str:
    texture = [f"highest card {RANKS[int(f['board_high_rank'])]}"]
    if f["board_paired"] > 0.5:
        texture.append("paired")
    suits = int(f["board_max_suit_count"])
    if suits >= 3:
        texture.append(f"{suits} cards of one suit")
    if int(f["board_straightness"]) >= 3:
        texture.append("connected (straight possible)")
    if len(texture) == 1:
        texture.append("dry (no flush or straight pressure)")
    return ", ".join(texture)


def _opponent_text(f: dict[str, float], opponent: str | None) -> str:
    hands = math.expm1(f["opp_log_hands"])
    label = f"the opponent (a '{opponent}' policy)" if opponent else "the opponent"
    if hands < 1.0:
        return f"No history against {label} yet, so the opponent statistics are the neutral prior."
    return (
        f"Observed over {hands:.0f} earlier hands, {label} plays VPIP {f['opp_vpip']:.0%}, PFR {f['opp_pfr']:.0%}, "
        f"postflop aggression {f['opp_aggression']:.0%}, folds to a bet {f['opp_fold_to_bet']:.0%} of the time."
    )


def describe_situation(
    features: NDArray[np.floating] | Sequence[float], legal_mask: NDArray[np.bool_], opponent: str | None = None
) -> str:
    """A few plain sentences about a decision state, derived from the feature vector alone.

    Cards are described the way the features know them: hole-card ranks and suitedness, board texture
    and made-hand category (no hidden or reconstructed cards).
    """
    f = dict(zip(FEATURE_NAMES, np.asarray(features, dtype=np.float64).tolist(), strict=True))
    street = int(f["street"])
    button = f["is_button"] > 0.5
    if street == 0:
        position = "on the button (small blind, first to act)" if button else "in the big blind (last to act)"
    else:
        position = "on the button, in position" if button else "in the big blind, out of position"
    if street == 0:
        hand = (
            f"I hold {_hole_text(f)}, better than {f['preflop_percentile']:.0%} of all starting hands "
            f"(equity against a random hand {f['equity']:.0%})."
        )
        board = ""
    else:
        made = MADE_HAND_NAMES[int(f["made_hand_category"])]
        extra = _draw_text(f)
        hand = (
            f"I hold {_hole_text(f)}{' (pairing the board)' if f['pair_with_board'] > 0.5 else ''}; "
            f"my best five-card hand is {made}{', plus ' + extra if extra else ''}, "
            f"{_overcards_text(f)}equity against a random hand {f['equity']:.0%}."
        )
        board = f" Board: {_board_text(f)}."
    pot, call = f["pot_bb"], f["to_call_bb"]
    raises = int(f["num_raises_street"])
    if f["facing_bet"] > 0.5:
        if street == 0 and raises == 0:
            facing = f"I am first to act: fold, complete the small blind ({call:.1f} bb more) or raise."
        else:
            verb = "raise" if street == 0 else "bet"
            facing = (
                f"I face a {verb} of {call:.1f} bb ({f['bet_fraction_of_pot']:.2f} x pot, "
                f"pot odds {f['pot_odds']:.0%}"
                f"{'; the opponent is all-in' if f['facing_all_in'] > 0.5 else ''})."
            )
    elif street == 0:
        facing = "The pot was limped: I can check my big-blind option or raise."
    else:
        facing = "Nobody has bet: I can check or bet."
    legal = ", ".join(ABSTRACT_ACTION_NAMES[i] for i, ok in enumerate(np.asarray(legal_mask, dtype=bool)) if ok)
    return (
        f"{STREET_NAMES[street].capitalize()}, {position}. {hand}{board} "
        f"Pot {pot:.1f} bb, effective stack {f['effective_stack_bb']:.0f} bb (SPR {f['spr']:.1f}). {facing} "
        f"{_opponent_text(f, opponent)} Legal actions: {legal}."
    )


# --------------------------------------------------------------------------------------------
# The case record
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class NotebookCase:
    """One real validation-set decision with the teacher's ground truth.

    ``row`` is the position in the validation frame that produced it (``load_validation_frame``);
    ``hand_id`` + ``decision_idx`` identify the row independently of any frame ordering.
    ``teacher_rules`` ends with the deciding rule (situation tags first); ``teacher_numbers`` are the
    quantities behind it (equity, adjusted equity, required equity, thresholds, ...).
    """

    key: str
    title: str
    why: str
    row: int
    hand_id: int
    decision_idx: int
    opponent: str
    behavior: str
    description: str
    features: NDArray[np.float32]
    legal_mask: NDArray[np.bool_]
    teacher_action: int
    teacher_action_name: str
    teacher_rules: tuple[str, ...]
    teacher_summary: str
    teacher_margin: float
    teacher_numbers: dict[str, float]

    @property
    def teacher_rule(self) -> str:
        """The deciding rule (last element of the trace)."""
        return self.teacher_rules[-1]

    @property
    def feature_dict(self) -> dict[str, float]:
        """The feature vector as ``{name: value}`` (schema order)."""
        return dict(zip(FEATURE_NAMES, self.features.tolist(), strict=True))

    def feature(self, name: str) -> float:
        return float(self.features[FEATURE_INDEX[name]])

    def as_row_frame(self) -> pd.DataFrame:
        """A one-row DataFrame with the feature columns (what ``EBMAgent`` / sklearn-style models take)."""
        return pd.DataFrame(self.features.reshape(1, -1), columns=list(FEATURE_NAMES))

    def to_dict(self) -> dict:
        """JSON-friendly summary (no arrays), e.g. for tables."""
        return {
            "key": self.key, "title": self.title, "row": self.row, "hand_id": self.hand_id,
            "decision_idx": self.decision_idx, "opponent": self.opponent, "teacher_action": self.teacher_action_name,
            "teacher_rule": self.teacher_rule, "teacher_margin": self.teacher_margin, "description": self.description,
        }


# --------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------
def load_validation_frame(dataset_name: str = DEFAULT_DATASET) -> pd.DataFrame:
    """The validation split of the dataset, index reset (``NotebookCase.row`` indexes into it)."""
    return load_dataset(dataset_name, split=VALIDATION_SPLIT)


def _legal_matrix(frame: pd.DataFrame) -> NDArray[np.bool_]:
    return frame[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)


def build_case(category: CaseCategory, frame: pd.DataFrame, row: int, policy: TeacherPolicy | None = None) -> NotebookCase:
    """The :class:`NotebookCase` of ``frame.iloc[row]`` (also usable for any row you want to explain)."""
    policy = policy or TeacherPolicy()
    record = frame.iloc[row]
    features = record[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    mask = record[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)
    decision: TeacherDecision = policy.decide_features(features, mask)
    if int(decision.action) != int(record["label"]):
        raise ValueError(f"Row {row}: stored label {int(record['label'])} differs from the teacher's {int(decision.action)}")
    name = ABSTRACT_ACTION_NAMES[int(decision.action)]
    opponent = str(record["opponent"])
    return NotebookCase(
        key=category.key,
        title=category.title,
        why=category.why,
        row=int(row),
        hand_id=int(record["hand_id"]),
        decision_idx=int(record["decision_idx"]),
        opponent=opponent,
        behavior=str(record["behavior"]),
        description=describe_situation(features, mask, opponent),
        features=features,
        legal_mask=mask,
        teacher_action=int(decision.action),
        teacher_action_name=name,
        teacher_rules=tuple(decision.rules),
        teacher_summary=decision.summary(),
        teacher_margin=float(decision.margin),
        teacher_numbers={k: float(v) for k, v in decision.numbers.items()},
    )


def category_candidates(frame: pd.DataFrame, category: CaseCategory) -> NDArray[np.int64]:
    """Row positions of ``frame`` that belong to ``category``, sorted by ``(hand_id, decision_idx)``."""
    positions = np.flatnonzero(category.select(frame).to_numpy())
    order = np.lexsort((frame["decision_idx"].to_numpy()[positions], frame["hand_id"].to_numpy()[positions]))
    return positions[order]


def select_example_cases(
    frame: pd.DataFrame | None = None,
    n: int = DEFAULT_N_CASES,
    seed: int = DEFAULT_SEED,
    categories: Sequence[str] | None = None,
    dataset_name: str = DEFAULT_DATASET,
) -> list[NotebookCase]:
    """``n`` diverse, reproducible validation-set decision situations (one per category, round-robin).

    Args:
        frame: a dataset table (``poker_arena.data.load_dataset``); if it has a ``split`` column only its
            ``"val"`` rows are used. ``None`` loads the validation split of ``dataset_name``.
        n: number of cases. The first ``len(categories)`` are one per category in the order of
            :data:`CATEGORIES` (the default categories start with the eight of the research notebooks);
            larger ``n`` draws further cases from the same categories, always from a different hand.
        seed: seeds the draw inside each category; the same arguments always return the same cases.
        categories: keys from :data:`CATEGORY_KEYS` (default: all, in order).
        dataset_name: dataset to read when ``frame`` is ``None``.

    A category with no candidate in ``frame`` is skipped (a tiny frame may not contain every situation);
    ``ValueError`` if no case at all can be built.
    """
    if frame is None:
        frame = load_validation_frame(dataset_name)
    elif "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == VALIDATION_SPLIT]
    frame = frame.reset_index(drop=True)
    keys = tuple(categories) if categories is not None else CATEGORY_KEYS
    unknown = sorted(set(keys) - set(CATEGORY_KEYS))
    if unknown:
        raise KeyError(f"Unknown case categories {unknown}; known: {list(CATEGORY_KEYS)}")
    by_key = {c.key: c for c in CATEGORIES}
    policy = TeacherPolicy()

    # one seeded, shuffled queue of candidate rows per category; popped round-robin
    queues: dict[str, list[int]] = {}
    for key in keys:
        candidates = category_candidates(frame, by_key[key])
        order = np.random.default_rng([seed, CATEGORY_KEYS.index(key)]).permutation(len(candidates))
        queues[key] = [int(candidates[i]) for i in order]

    used_hands: set[int] = set()
    cases: list[NotebookCase] = []
    hand_ids = frame["hand_id"].to_numpy()
    while len(cases) < n and any(queues.values()):
        for key in keys:
            queue = queues[key]
            while queue and int(hand_ids[queue[0]]) in used_hands:
                queue.pop(0)  # a hand already used by another case is skipped
            if not queue or len(cases) >= n:
                continue
            row = queue.pop(0)
            used_hands.add(int(hand_ids[row]))
            cases.append(build_case(by_key[key], frame, row, policy))
    if not cases:
        raise ValueError("No example case could be built from this frame")
    return cases
