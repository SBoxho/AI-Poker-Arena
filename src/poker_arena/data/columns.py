"""Column layout of the dataset table (shared by generation, I/O, tests and the trainers)."""

from __future__ import annotations

from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES
from poker_arena.features import FEATURE_NAMES

HAND_ID_COLUMN = "hand_id"
IDENTIFIER_COLUMNS = ("hand_id", "hand_index", "decision_idx", "split", "source", "behavior", "opponent")
LEGAL_COLUMNS = tuple(f"legal_{name}" for name in ABSTRACT_ACTION_NAMES)
"""``legal_fold`` ... ``legal_all_in``: the abstract actions that were legal in the state."""
TARGET_COLUMNS = ("label", *LEGAL_COLUMNS, "behavior_action", "teacher_margin", "teacher_rule")
COLUMNS = (*IDENTIFIER_COLUMNS, *FEATURE_NAMES, *TARGET_COLUMNS)
CATEGORICAL_COLUMNS = ("split", "source", "behavior", "opponent", "teacher_rule")
