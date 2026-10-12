"""Helpers for the EBM tests: a tiny end-to-end trained model, built once per pytest process.

Plain functions (like ``core_helpers``): the model is produced by the REAL pipeline
(:func:`poker_arena.training.ebm.train_ebm` - behaviour cloning, one DAgger round, validation) on a tiny
freshly generated dataset, so the tests exercise exactly what ``scripts/train_ebm.py`` runs, in well under
a minute and without needing the large dataset in ``artifacts/``.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from poker_arena.config import GameConfig
from poker_arena.data import DatasetConfig, generate_dataset, write_dataset
from poker_arena.training.common import DaggerSettings
from poker_arena.training.ebm import EBMParams, EBMTrainSettings, TrainingOutcome, ValidationSettings, train_ebm

SEED = 7
TRAIN_HANDS = 260
VAL_HANDS = 70
DATASET_NAME = "tiny"
TINY_PARAMS = EBMParams(
    max_rounds=25,
    learning_rate=0.1,
    interactions=4,
    outer_bags=3,
    inner_bags=0,
    max_bins=32,
    max_interaction_bins=8,
    min_samples_leaf=4,
    max_leaves=2,
    early_stopping_rounds=10,
    validation_fraction=0.2,
)


@dataclass(frozen=True)
class TinyRun:
    """Where the tiny dataset and the trained model live, and what the pipeline reported."""

    root: Path
    dataset_dir: Path
    model_dir: Path
    outcome: TrainingOutcome
    settings: EBMTrainSettings


def tiny_settings(root: Path, dataset_dir: Path, **overrides: object) -> EBMTrainSettings:
    values: dict[str, object] = {
        "dataset_name": DATASET_NAME,
        "game": GameConfig(),
        "seed": SEED,
        "params": TINY_PARAMS,
        "max_train_rows": None,
        "class_weight_power": 0.5,
        "n_jobs": 2,
        "dagger": DaggerSettings(rounds=1, hands_per_round=30, teacher_beta=0.5, n_workers=1, seed=SEED),
        "validation": ValidationSettings(
            opponents=("teacher", "call_station", "maniac", "random"),
            pairs=4,
            on_policy_hands=30,
            importance_sample_rows=500,
            n_workers=1,
        ),
        "output_dir": root / "model",
        "work_dir": root / "rounds",
        "datasets_dir": dataset_dir,
    }
    values.update(overrides)
    return EBMTrainSettings(**values)  # type: ignore[arg-type]


def make_tiny_dataset(dataset_dir: Path) -> None:
    config = DatasetConfig(seed=SEED, match_lengths=(20, 60))
    dataset = generate_dataset(config, TRAIN_HANDS, VAL_HANDS, n_workers=1)
    write_dataset(dataset.frame, dataset.meta, DATASET_NAME, dataset_dir)


@lru_cache(maxsize=1)
def tiny_run() -> TinyRun:
    """Generate the tiny dataset and train on it (cached; the temp folder is removed at interpreter exit)."""
    root = Path(tempfile.mkdtemp(prefix="ebm_tests_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    dataset_dir = root / "datasets"
    make_tiny_dataset(dataset_dir)
    settings = tiny_settings(root, dataset_dir)
    outcome = train_ebm(settings, {"TEST": True}, log=lambda _: None)
    return TinyRun(root, dataset_dir, outcome.output_dir, outcome, settings)
