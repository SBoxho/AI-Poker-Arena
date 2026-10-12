"""Feature schema and ``Observation -> feature vector`` shared by training and inference.

Typical use::

    from poker_arena.features import FEATURE_NAMES, build_features, schema_hash

    x = build_features(obs)              # float32 vector, order = FEATURE_NAMES
    frame = features_frame(observations)  # DataFrame with one column per feature

See :mod:`poker_arena.features.schema` for the meaning and range of every feature and
:mod:`poker_arena.features.builder` for how they are computed (and why they are deterministic).
"""

from __future__ import annotations

from poker_arena.features.builder import (
    build_feature_dict,
    build_features,
    build_features_matrix,
    features_frame,
    hand_equity,
    vector_to_dict,
)
from poker_arena.features.schema import (
    BINARY_FEATURES,
    EQUITY_SAMPLES_FLOP_TURN,
    FEATURE_DESCRIPTIONS,
    FEATURE_GROUP_OF,
    FEATURE_GROUPS,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_RANGES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    NUM_FEATURES,
    FeatureSpec,
    schema_dict,
    schema_hash,
)

__all__ = [
    "BINARY_FEATURES",
    "EQUITY_SAMPLES_FLOP_TURN",
    "FEATURE_DESCRIPTIONS",
    "FEATURE_GROUPS",
    "FEATURE_GROUP_OF",
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "FEATURE_RANGES",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SPECS",
    "NUM_FEATURES",
    "FeatureSpec",
    "build_feature_dict",
    "build_features",
    "build_features_matrix",
    "features_frame",
    "hand_equity",
    "schema_dict",
    "schema_hash",
    "vector_to_dict",
]
