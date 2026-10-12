"""The shared example cases of the research notebooks: deterministic, validation-only, teacher ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poker_arena.agents.teacher import TeacherPolicy
from poker_arena.config import DATASETS_DIR
from poker_arena.data import DatasetConfig, generate_dataset
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES
from poker_arena.explain.notebook_cases import (
    CATEGORY_KEYS,
    DEFAULT_DATASET,
    DEFAULT_N_CASES,
    NotebookCase,
    describe_situation,
    load_validation_frame,
    select_example_cases,
)
from poker_arena.features import NUM_FEATURES

TINY_TRAIN_HANDS = 60
TINY_VAL_HANDS = 400
TINY_SEED = 11
REAL_DATASET_AVAILABLE = (DATASETS_DIR / f"{DEFAULT_DATASET}.parquet").exists()


@pytest.fixture(scope="module")
def tiny_frame() -> pd.DataFrame:
    """A small freshly generated dataset (train + validation split) built by the real pipeline."""
    return generate_dataset(DatasetConfig(seed=TINY_SEED), TINY_TRAIN_HANDS, TINY_VAL_HANDS, n_workers=1).frame


def test_cases_are_deterministic_and_prefix_stable(tiny_frame: pd.DataFrame) -> None:
    first = select_example_cases(tiny_frame, n=6, seed=3)
    again = select_example_cases(tiny_frame.sample(frac=1.0, random_state=0), n=6, seed=3)  # row order must not matter
    assert [(c.hand_id, c.decision_idx) for c in first] == [(c.hand_id, c.decision_idx) for c in again]
    longer = select_example_cases(tiny_frame, n=9, seed=3)
    assert [c.hand_id for c in longer[: len(first)]] == [c.hand_id for c in first]
    other_seed = select_example_cases(tiny_frame, n=6, seed=4)
    assert [c.hand_id for c in other_seed] != [c.hand_id for c in first]


def test_cases_come_from_validation_hands_only_and_use_each_hand_once(tiny_frame: pd.DataFrame) -> None:
    cases = select_example_cases(tiny_frame, n=10)
    val_hands = set(tiny_frame.loc[tiny_frame["split"] == "val", "hand_id"])
    train_hands = set(tiny_frame.loc[tiny_frame["split"] == "train", "hand_id"])
    assert cases and {c.hand_id for c in cases} <= val_hands
    assert not {c.hand_id for c in cases} & train_hands
    assert len({c.hand_id for c in cases}) == len(cases)


def test_case_fields_match_the_teacher_ground_truth(tiny_frame: pd.DataFrame) -> None:
    policy = TeacherPolicy()
    for case in select_example_cases(tiny_frame, n=8):
        assert isinstance(case, NotebookCase)
        assert case.features.shape == (NUM_FEATURES,) and case.features.dtype == np.float32
        assert case.legal_mask.shape == (len(ABSTRACT_ACTION_NAMES),) and case.legal_mask[case.teacher_action]
        decision = policy.decide_features(case.features, case.legal_mask)
        assert int(decision.action) == case.teacher_action
        assert ABSTRACT_ACTION_NAMES[case.teacher_action] == case.teacher_action_name
        assert case.teacher_rules == tuple(decision.rules) and case.teacher_rule == decision.rules[-1]
        assert case.teacher_summary == decision.summary()
        assert case.title and case.why and case.key in CATEGORY_KEYS
        assert case.description.startswith(("Preflop", "Flop", "Turn", "River")) and "Legal actions" in case.description
        assert case.as_row_frame().shape == (1, NUM_FEATURES)
        assert tiny_frame.loc[tiny_frame["hand_id"] == case.hand_id, "label"].notna().all()


def test_category_selection_and_errors(tiny_frame: pd.DataFrame) -> None:
    cases = select_example_cases(tiny_frame, n=3, categories=["preflop_open"])
    assert {c.key for c in cases} == {"preflop_open"}
    with pytest.raises(KeyError, match="Unknown case categories"):
        select_example_cases(tiny_frame, categories=["nope"])
    with pytest.raises(ValueError, match="No example case"):
        select_example_cases(tiny_frame.iloc[:0])


def test_description_uses_only_feature_information(tiny_frame: pd.DataFrame) -> None:
    case = select_example_cases(tiny_frame, n=1)[0]
    assert describe_situation(case.features, case.legal_mask, case.opponent) == case.description
    assert case.opponent in case.description or "neutral prior" in case.description


@pytest.mark.skipif(not REAL_DATASET_AVAILABLE, reason="artifacts/datasets/teacher_bc_v2 has not been generated")
def test_default_cases_on_the_committed_dataset() -> None:
    frame = load_validation_frame()
    assert set(frame["split"].astype(str)) == {"val"}
    cases = select_example_cases(frame)
    assert len(cases) == DEFAULT_N_CASES
    assert [c.key for c in cases] == list(CATEGORY_KEYS[:DEFAULT_N_CASES])
    assert [c.hand_id for c in cases] == [c.hand_id for c in select_example_cases()]
    for case in cases:
        assert int(frame.loc[case.row, "hand_id"]) == case.hand_id and int(frame.loc[case.row, "label"]) == case.teacher_action
