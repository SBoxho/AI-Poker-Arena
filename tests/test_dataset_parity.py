"""Train/inference parity: dataset rows equal what ``build_features`` gives for a replay of the same hands.

The replay is independent of the generator: hands are rebuilt from their public ``HandRecord`` with a
fresh ``HandEngine`` and fresh opponent trackers, exactly as a live agent would experience them.
"""

from __future__ import annotations

import numpy as np

from poker_arena.agents.teacher import TeacherPolicy
from poker_arena.data import DatasetConfig, collect_on_policy, feature_matrix, label_vector, legal_mask_matrix
from poker_arena.data.generation import CollectionSpec, assemble_frame, plan_matches, run_match_plan
from poker_arena.engine.game import HandEngine
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.agents.registry import AgentSpec
from poker_arena.features import build_features

CONFIG = DatasetConfig(seed=11, match_lengths=(30,), short_stack_probability=0.3)


def make_spec() -> CollectionSpec:
    return CollectionSpec(
        config=CONFIG,
        namespace="dataset_train",
        seed_parts=(CONFIG.seed,),
        split="train",
        source="bc",
        opponent_mix=CONFIG.opponent_mix,
    )


def replay_rows(result) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """(features, legal mask, teacher label) of every decision of seat 0, rebuilt from the hand records."""
    teacher = TeacherPolicy()
    trackers = (OpponentTracker(), OpponentTracker())
    rows = []
    for record in result.records:
        engine = HandEngine(record.config, record.deal, record.button_seat, record.start_stacks, record.hand_number)
        for event in record.events:
            if engine.actor_seat == 0:
                obs = engine.observation_for(0, trackers[0].stats())
                features = build_features(obs)
                rows.append((features, obs.abstract_mask, int(teacher.decide_features(features, obs.abstract_mask).action)))
            engine.act(event.action)
        for seat in (0, 1):
            trackers[seat].update(record.public_summary(seat))
    return rows


def test_stored_rows_equal_features_rebuilt_from_a_replay() -> None:
    spec = make_spec()
    plan = plan_matches(spec, 60)[0]
    result = run_match_plan(spec, plan, keep_records=True)
    frame = assemble_frame([result], "train", "bc")
    replayed = replay_rows(result)
    assert len(replayed) == len(frame) > 20
    assert np.array_equal(feature_matrix(frame), np.vstack([features for features, _, _ in replayed]))
    assert np.array_equal(legal_mask_matrix(frame), np.vstack([mask for _, mask, _ in replayed]))
    assert label_vector(frame).tolist() == [label for _, _, label in replayed]


def test_labels_are_the_canonical_teacher_applied_to_the_stored_features() -> None:
    spec = make_spec()
    frame = assemble_frame([run_match_plan(spec, plan_matches(spec, 40)[0])], "train", "bc")
    teacher = TeacherPolicy()
    features, masks = feature_matrix(frame), legal_mask_matrix(frame)
    again = [int(teacher.decide_features(x, m).action) for x, m in zip(features, masks, strict=True)]
    assert again == label_vector(frame).tolist()


def test_dagger_rows_also_replay_exactly() -> None:
    spec = CollectionSpec(
        config=CONFIG, namespace="dagger", seed_parts=(CONFIG.seed, 1, "train"), split="train",
        source="dagger_round_1", opponent_mix=CONFIG.opponent_mix, learner=AgentSpec("random"),
    )
    result = run_match_plan(spec, plan_matches(spec, 30)[0], keep_records=True)
    frame = collect_on_policy(AgentSpec("random"), 30, (CONFIG.seed, 1), config=CONFIG)
    assert np.array_equal(feature_matrix(frame), np.vstack([features for features, _, _ in replay_rows(result)]))
