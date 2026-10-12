"""A tiny, really trained XGBoost artifact shared by the ``test_xgb_*`` suites.

Building it takes a few seconds (a few hundred hands of teacher-labelled data, a handful of trees, one
DAgger round, a handful of duplicate pairs), is done once per test session and lives in a temp folder, so
the tests exercise the *real* pipeline - dataset -> training -> artifact -> agent - without touching the
committed model in ``artifacts/models/xgb``.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd

from poker_arena.config import GameConfig
from poker_arena.data import DatasetConfig, generate_dataset, write_dataset
from poker_arena.data.evaluation import pair_results
from poker_arena.training import xgb as training_xgb
from poker_arena.training.common import DaggerSettings
from poker_arena.training.xgb import TrainingOutcome, ValidationSettings, XGBParams, XGBTrainSettings, train_xgboost

DATASET_NAME = "xgb_test"
TRAIN_HANDS = 450
VAL_HANDS = 120
DATASET_SEED = 5
TRAIN_SEED = 3
DAGGER_ROUNDS = 1
TEST_OPPONENTS = ("teacher", "call_station", "maniac", "random")
VALIDATION_PAIRS = 6


@dataclass(frozen=True)
class TinyModel:
    """Paths and results of the one-off training run."""

    root: Path
    datasets_dir: Path
    model_dir: Path
    work_dir: Path
    settings: XGBTrainSettings
    outcome: TrainingOutcome
    pair_namespaces: tuple[str, ...]  # namespaces every duplicate-deal validation call used
    validation_frame: pd.DataFrame


def settings_for(root: Path, **overrides: Any) -> XGBTrainSettings:
    """Small training settings below ``root`` (``overrides`` replace dataclass fields)."""
    fields: dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "game": GameConfig(),
        "seed": TRAIN_SEED,
        "params": XGBParams(
            n_estimators=40, max_depth=5, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0,
            min_child_weight=1.0, reg_lambda=1.0, early_stopping_rounds=8, max_bin=64,
        ),  # fmt: skip
        "max_train_rows": None,
        "class_weight_power": 0.5,
        "n_jobs": 2,
        "dagger": DaggerSettings(rounds=DAGGER_ROUNDS, hands_per_round=40, teacher_beta=0.5, n_workers=1, seed=TRAIN_SEED),
        "validation": ValidationSettings(
            opponents=TEST_OPPONENTS, pairs=VALIDATION_PAIRS, on_policy_hands=30, shap_sample_rows=60, n_workers=1
        ),
        "output_dir": root / "model",
        "work_dir": root / "rounds",
        "datasets_dir": root / "datasets",
    }
    fields.update(overrides)
    return XGBTrainSettings(**fields)


@lru_cache(maxsize=1)
def tiny_model() -> TinyModel:
    """Generate a tiny dataset and train on it (cached for the whole pytest session)."""
    root = Path(tempfile.mkdtemp(prefix="poker_arena_xgb_test_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    config = DatasetConfig(seed=DATASET_SEED, match_lengths=(30, 60), short_stack_probability=0.2)
    dataset = generate_dataset(config, TRAIN_HANDS, VAL_HANDS, n_workers=1)
    write_dataset(dataset.frame, dataset.meta, DATASET_NAME, root / "datasets")
    settings = settings_for(root)
    namespaces: list[str] = []

    def recording_pair_results(*args: Any, **kwargs: Any):
        namespaces.append(kwargs.get("namespace", "eval"))
        return pair_results(*args, **kwargs)

    with mock.patch.object(training_xgb, "pair_results", recording_pair_results):
        outcome = train_xgboost(settings, {"RANDOM_SEED": TRAIN_SEED, "OUTPUT_DIR": settings.output_dir})
    return TinyModel(
        root=root,
        datasets_dir=root / "datasets",
        model_dir=settings.output_dir,
        work_dir=settings.work_dir,
        settings=settings,
        outcome=outcome,
        pair_namespaces=tuple(namespaces),
        validation_frame=dataset.frame[dataset.frame["split"] == "val"].reset_index(drop=True),
    )


def copy_model_dir(destination: Path) -> Path:
    """A private copy of the tiny model's artifact folder (tests may corrupt it freely)."""
    shutil.copytree(tiny_model().model_dir, destination)
    return destination
