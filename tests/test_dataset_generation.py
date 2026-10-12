"""Dataset generation: reproducibility, worker independence, splits, legality, metadata, DAgger."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from poker_arena.agents.registry import AgentSpec
from poker_arena.agents.teacher import CANONICAL_PARAMS
from poker_arena.data import (
    COLUMNS,
    LEGAL_COLUMNS,
    DatasetConfig,
    collect_on_policy,
    content_hash,
    feature_matrix,
    generate_dataset,
    label_vector,
    legal_mask_matrix,
    load_dataset,
    load_meta,
    validate_frame,
    write_dataset,
)
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, AbstractAction
from poker_arena.engine.deals import Deal
from poker_arena.features import FEATURE_NAMES, schema_hash

CONFIG = DatasetConfig(seed=7, match_lengths=(15, 40))
TRAIN_HANDS, VAL_HANDS = 90, 45


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(CONFIG, TRAIN_HANDS, VAL_HANDS, n_workers=1)


# ------------------------------------------------------------------------------ determinism
def test_same_seed_gives_the_identical_dataset(dataset) -> None:
    again = generate_dataset(CONFIG, TRAIN_HANDS, VAL_HANDS, n_workers=1)
    assert again.meta["content_sha256"] == dataset.meta["content_sha256"]
    pd.testing.assert_frame_equal(again.frame, dataset.frame)


def test_output_does_not_depend_on_the_number_of_workers(dataset) -> None:
    parallel = generate_dataset(CONFIG, TRAIN_HANDS, VAL_HANDS, n_workers=2)
    assert parallel.meta["content_sha256"] == dataset.meta["content_sha256"]
    pd.testing.assert_frame_equal(parallel.frame, dataset.frame)


def test_a_different_seed_gives_different_data(dataset) -> None:
    other = generate_dataset(replace(CONFIG, seed=8), TRAIN_HANDS, VAL_HANDS, n_workers=1)
    assert other.meta["content_sha256"] != dataset.meta["content_sha256"]
    assert set(other.frame["hand_id"]).isdisjoint(set(dataset.frame["hand_id"]))


# ------------------------------------------------------------------------------------ splits
def test_splits_are_disjoint_by_hand_id_and_by_deal(dataset) -> None:
    frame = dataset.frame
    train, val = frame[frame["split"] == "train"], frame[frame["split"] == "val"]
    assert len(train) and len(val)
    assert set(train["hand_id"]).isdisjoint(set(val["hand_id"]))
    deals = {split: {Deal.from_rng(random.Random(h)).cards for h in set(part["hand_id"])} for split, part in (("train", train), ("val", val))}
    assert deals["train"].isdisjoint(deals["val"])


def test_hands_are_never_split_across_splits_and_decisions_are_numbered(dataset) -> None:
    frame = dataset.frame
    assert (frame.groupby("hand_id", observed=True)["split"].nunique() == 1).all()
    numbering = frame.groupby("hand_id", observed=True)["decision_idx"].apply(lambda s: list(s) == list(range(len(s))))
    assert numbering.all()


def test_dagger_hands_never_coincide_with_the_behaviour_cloning_hands(dataset) -> None:
    rows = collect_on_policy(AgentSpec("teacher"), 40, (CONFIG.seed, 1), config=CONFIG)
    assert set(rows["hand_id"]).isdisjoint(set(dataset.frame["hand_id"]))


# ------------------------------------------------------------------------------- legality
def test_labels_are_always_legal_and_layout_is_valid(dataset) -> None:
    validate_frame(dataset.frame)
    frame = dataset.frame
    labels, masks = label_vector(frame), legal_mask_matrix(frame)
    assert masks[np.arange(len(frame)), labels].all()
    assert masks[:, AbstractAction.CHECK_CALL].all()
    assert set(np.unique(labels)) <= set(range(len(ABSTRACT_ACTION_NAMES)))


def test_legal_mask_columns_are_consistent_with_the_features(dataset) -> None:
    frame = dataset.frame
    assert list(LEGAL_COLUMNS) == [f"legal_{name}" for name in ABSTRACT_ACTION_NAMES]
    assert (frame["legal_fold"] == (frame["facing_bet"] == 1)).all()  # folding is legal exactly when there is a price
    assert (frame["legal_all_in"] == (frame["can_raise"] == 1)).all()  # an all-in exists whenever raising does
    no_raise = frame["can_raise"] == 0
    assert not frame.loc[no_raise, ["legal_raise_small", "legal_raise_pot"]].any().any()


def test_behavior_action_is_a_legal_action(dataset) -> None:
    frame = dataset.frame
    taken = frame["behavior_action"].to_numpy(dtype=np.int64)
    assert (taken >= 0).all()
    assert legal_mask_matrix(frame)[np.arange(len(frame)), taken].all()


# ---------------------------------------------------------------------------------- metadata
def test_meta_describes_the_table(dataset) -> None:
    meta, frame = dataset.meta, dataset.frame
    assert meta["schema_hash"] == schema_hash() and meta["feature_names"] == list(FEATURE_NAMES)
    assert meta["columns"] == list(COLUMNS) == list(frame.columns)
    assert meta["rows"] == len(frame) == sum(info["rows"] for info in meta["splits"].values())
    assert meta["content_sha256"] == content_hash(frame)
    assert meta["config"] == CONFIG.to_dict()
    assert meta["teacher"]["params"] == CANONICAL_PARAMS.to_dict()
    assert meta["splits"]["train"]["requested_hands"] == TRAIN_HANDS
    assert meta["splits"]["val"]["requested_hands"] == VAL_HANDS
    counts = np.bincount(frame["label"].to_numpy(dtype=np.int64), minlength=len(ABSTRACT_ACTION_NAMES))
    for i, name in enumerate(ABSTRACT_ACTION_NAMES):
        assert meta["label_distribution"][name]["count"] == counts[i]
    assert sum(entry["fraction"] for entry in meta["label_distribution"].values()) == pytest.approx(1.0)
    assert meta["legal_statistics"]["check_call_legal_fraction"] == 1.0
    assert {"python", "numpy", "pandas", "pyarrow", "pokerkit"} <= set(meta["libraries"])
    assert meta["generation"]["rows_per_second"] > 0
    json.dumps(meta)  # JSON-serialisable


def test_parquet_roundtrip_and_schema_guard(dataset, tmp_path) -> None:
    parquet_path, meta_path = write_dataset(dataset.frame, dataset.meta, "unit", tmp_path)
    assert parquet_path.exists() and meta_path.exists()
    stored = load_meta("unit", tmp_path)
    assert stored["files"]["parquet"] == "unit.parquet" and stored["name"] == "unit"
    loaded = load_dataset("unit", tmp_path)
    assert content_hash(loaded) == dataset.meta["content_sha256"]
    assert feature_matrix(loaded).dtype == np.float32 and feature_matrix(loaded).shape[1] == len(FEATURE_NAMES)
    assert set(load_dataset("unit", tmp_path, split="val")["split"]) == {"val"}
    stored["schema_hash"] = "0" * 16
    meta_path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_dataset("unit", tmp_path)
    assert len(load_dataset("unit", tmp_path, check_schema=False)) == len(dataset.frame)
    with pytest.raises(FileNotFoundError):
        load_dataset("missing", tmp_path)


# ------------------------------------------------------------------------ coverage mixtures
def test_behavior_and_opponent_mixtures_are_respected() -> None:
    cfg = replace(CONFIG, behavior_mix={"random": 1.0}, opponent_mix={"maniac": 1.0})
    frame = generate_dataset(cfg, 40, 10).frame
    assert set(frame["behavior"]) == {"random"} and set(frame["opponent"]) == {"maniac"}
    assert set(frame["source"]) == {"bc"}


def test_unknown_policies_and_bad_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        DatasetConfig(behavior_mix={"nope": 1.0})
    with pytest.raises(ValueError):
        DatasetConfig(opponent_mix={"random": 0.0})
    with pytest.raises(ValueError):
        DatasetConfig(match_lengths=())
    with pytest.raises(ValueError):
        DatasetConfig(min_short_stack_bb=10_000)


def test_short_stacks_are_sampled_when_requested() -> None:
    deep = generate_dataset(replace(CONFIG, short_stack_probability=0.0), 60, 6).frame
    short = generate_dataset(replace(CONFIG, short_stack_probability=1.0, min_short_stack_bb=10), 60, 6).frame
    assert deep["effective_stack_bb"].mean() > short["effective_stack_bb"].mean() + 10
    assert short["effective_stack_bb"].min() >= 0


def test_epsilon_exploration_deviates_from_the_teacher() -> None:
    plain = replace(CONFIG, behavior_mix={"teacher": 1.0})
    explore = replace(CONFIG, behavior_mix={"teacher_eps": 1.0}, exploration_epsilon=0.5)
    on_policy = generate_dataset(plain, 60, 6).frame
    exploring = generate_dataset(explore, 60, 6).frame
    # follows the teacher exactly ... unless it explores
    assert (on_policy["behavior_action"] == on_policy["label"]).all()
    assert (exploring["behavior_action"] != exploring["label"]).mean() > 0.1


# ------------------------------------------------------------------------------------ DAgger
def test_dagger_rows_are_tagged_and_labelled_by_the_teacher() -> None:
    rows = collect_on_policy(AgentSpec("call_station"), 45, (CONFIG.seed, 2), config=CONFIG, round_index=2)
    validate_frame(rows)
    assert set(rows["source"]) == {"dagger_round_2"} and set(rows["behavior"]) == {"learner"}
    assert (rows["behavior_action"] == int(AbstractAction.CHECK_CALL)).all()  # what the learner did
    assert rows["label"].nunique() > 1  # what the teacher would do: not always call
    assert list(rows.columns) == list(COLUMNS)


def test_dagger_with_the_teacher_as_learner_agrees_with_itself_and_beta_mixes_it_in() -> None:
    same = collect_on_policy(AgentSpec("teacher"), 30, (CONFIG.seed, 3), config=CONFIG)
    assert (same["behavior_action"] == same["label"]).all()
    mixed = collect_on_policy(AgentSpec("random"), 30, (CONFIG.seed, 3), config=CONFIG, teacher_beta=1.0)
    assert (mixed["behavior_action"] == mixed["label"]).all()  # beta = 1: always follows the teacher


def test_dagger_is_reproducible_and_worker_independent() -> None:
    kwargs = {"learner_spec": AgentSpec("random"), "n_hands": 60, "seed_parts": (CONFIG.seed, 4), "config": CONFIG}
    one = collect_on_policy(n_workers=1, **kwargs)
    two = collect_on_policy(n_workers=2, **kwargs)
    assert content_hash(one) == content_hash(two)
    with pytest.raises(ValueError):
        collect_on_policy(split="test", **kwargs)
