"""Feature schema contract: names, order, groups, descriptions, ranges, hash, vector construction."""

from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
import pytest

from core_helpers import random_midhand_engines
from features_helpers import acting_observation, scripted_engine

from poker_arena.features import (
    BINARY_FEATURES,
    FEATURE_DESCRIPTIONS,
    FEATURE_GROUP_OF,
    FEATURE_GROUPS,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_RANGES,
    FEATURE_SCHEMA_VERSION,
    NUM_FEATURES,
    build_feature_dict,
    build_features,
    build_features_matrix,
    features_frame,
    schema_dict,
    schema_hash,
    vector_to_dict,
)


def test_names_are_unique_ordered_and_indexed() -> None:
    assert len(FEATURE_NAMES) == NUM_FEATURES == len(set(FEATURE_NAMES))
    assert 35 <= NUM_FEATURES <= 50
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in FEATURE_NAMES)
    assert [FEATURE_INDEX[name] for name in FEATURE_NAMES] == list(range(NUM_FEATURES))


def test_every_feature_has_a_description_group_and_range() -> None:
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_NAMES) == set(FEATURE_RANGES) == set(FEATURE_GROUP_OF)
    for name in FEATURE_NAMES:
        description = FEATURE_DESCRIPTIONS[name]
        assert len(description.split()) >= 5 and description.endswith((".", ")")), name
        low, high = FEATURE_RANGES[name]
        assert low < high, name


def test_groups_partition_the_features() -> None:
    flat = [name for names in FEATURE_GROUPS.values() for name in names]
    assert sorted(flat) == sorted(FEATURE_NAMES)
    assert {"hand_strength", "board", "position", "money", "history", "opponent"} <= set(FEATURE_GROUPS)
    assert all(FEATURE_GROUP_OF[name] == group for group, names in FEATURE_GROUPS.items() for name in names)


def test_binary_features_have_unit_range() -> None:
    assert BINARY_FEATURES
    assert all(FEATURE_RANGES[name] == (0.0, 1.0) for name in BINARY_FEATURES)


def test_schema_hash_is_short_stable_hex_and_tracks_the_definition() -> None:
    assert re.fullmatch(r"[0-9a-f]{16}", schema_hash())
    assert schema_hash() == schema_hash()
    assert schema_dict()["version"] == FEATURE_SCHEMA_VERSION
    assert [row[0] for row in schema_dict()["features"]] == list(FEATURE_NAMES)


def test_schema_hash_is_identical_in_another_process() -> None:
    code = "from poker_arena.features import schema_hash; print(schema_hash())"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == schema_hash()


def test_build_features_shape_dtype_and_name_order() -> None:
    obs = acting_observation(scripted_engine("AhAd", "7c2d", "Ks9h4c3d2s"))
    vector = build_features(obs)
    assert vector.dtype == np.float32 and vector.shape == (NUM_FEATURES,)
    as_dict = build_feature_dict(obs)
    assert list(as_dict) == list(FEATURE_NAMES)
    assert np.array_equal(np.array(list(as_dict.values()), dtype=np.float32), vector)
    assert vector_to_dict(vector) == as_dict
    with pytest.raises(ValueError):
        vector_to_dict(vector[:-1])


def test_batch_helpers_match_single_calls() -> None:
    observations = [e.observation_for(e.actor_seat) for e in random_midhand_engines(11, 25)]
    matrix = build_features_matrix(observations)
    assert matrix.shape == (25, NUM_FEATURES) and matrix.dtype == np.float32
    assert np.array_equal(matrix, np.vstack([build_features(o) for o in observations]))
    frame = features_frame(observations)
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert (frame.dtypes == np.float32).all()
    assert np.array_equal(frame.to_numpy(), matrix)
    assert build_features_matrix([]).shape == (0, NUM_FEATURES)


@pytest.mark.parametrize("seed", [3, 4])
def test_features_are_finite_and_inside_their_documented_ranges(seed: int) -> None:
    for engine in random_midhand_engines(seed, 400):
        vector = build_features(engine.observation_for(engine.actor_seat))
        assert np.isfinite(vector).all()
        for name, value in zip(FEATURE_NAMES, vector.tolist(), strict=True):
            low, high = FEATURE_RANGES[name]
            assert low - 1e-6 <= value <= high + 1e-6, (name, value)
            if name in BINARY_FEATURES:
                assert value in (0.0, 1.0), name
