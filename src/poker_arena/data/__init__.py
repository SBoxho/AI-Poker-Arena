"""Dataset generation (teacher-labelled decision states, DAgger aggregation) and evaluation helpers.

Orchestration code: unlike ``agents/`` and ``features/`` it is allowed to drive hands through
``HandSession`` / ``play_hand``.
"""

from __future__ import annotations

from poker_arena.data.columns import (
    CATEGORICAL_COLUMNS,
    COLUMNS,
    HAND_ID_COLUMN,
    IDENTIFIER_COLUMNS,
    LEGAL_COLUMNS,
    TARGET_COLUMNS,
)
from poker_arena.data.generation import (
    DatasetConfig,
    GeneratedDataset,
    collect_on_policy,
    generate_dataset,
)
from poker_arena.data.io import (
    content_hash,
    dataset_paths,
    feature_matrix,
    label_vector,
    legal_mask_matrix,
    load_dataset,
    load_meta,
    validate_frame,
    write_dataset,
)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "COLUMNS",
    "HAND_ID_COLUMN",
    "IDENTIFIER_COLUMNS",
    "LEGAL_COLUMNS",
    "TARGET_COLUMNS",
    "DatasetConfig",
    "GeneratedDataset",
    "collect_on_policy",
    "content_hash",
    "dataset_paths",
    "feature_matrix",
    "generate_dataset",
    "label_vector",
    "legal_mask_matrix",
    "load_dataset",
    "load_meta",
    "validate_frame",
    "write_dataset",
]
