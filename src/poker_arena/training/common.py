"""Shared building blocks of the supervised training pipelines (EBM and XGBoost).

Both models are trained on the *same* dataset and the *same* feature schema and are judged with the
*same* metrics, which is what makes their comparison meaningful:

* :func:`load_training_data` - train/validation arrays (split by hand id, never by decision),
* :func:`class_weights` / :func:`sample_weights` - tame the class imbalance (all-in and pot raises are rare),
* :func:`evaluate_classifier` - **imitation** metrics, always reported both raw and *legal-masked*
  (the masked arg-max is what an agent actually plays),
* :func:`run_dagger` - Dataset Aggregation: the learner plays, the teacher relabels the visited states.

Reminder (also stated in the notebooks and the UI): these metrics measure how well a model reproduces
the *teacher*. They say nothing about poker skill - that is measured separately in duplicate-deal
tournaments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from poker_arena.agents.registry import AgentSpec
from poker_arena.data.io import (
    feature_matrix,
    label_vector,
    legal_mask_matrix,
    load_dataset,
    load_meta,
)
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS
from poker_arena.features import FEATURE_NAMES

if TYPE_CHECKING:  # the module itself is imported lazily inside run_dagger (orchestration code)
    from poker_arena.data.generation import DatasetConfig

CLASS_LABELS = list(range(NUM_ABSTRACT_ACTIONS))
STREET_NAMES = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True)
class SplitArrays:
    """One split of the dataset as plain arrays (row order preserved)."""

    X: NDArray[np.float32]
    y: NDArray[np.int64]
    legal: NDArray[np.bool_]
    hand_id: NDArray[np.int64]
    street: NDArray[np.int64]
    frame: pd.DataFrame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "SplitArrays":
        return cls(
            X=feature_matrix(frame),
            y=label_vector(frame),
            legal=legal_mask_matrix(frame),
            hand_id=frame["hand_id"].to_numpy(dtype=np.int64),
            street=frame["street"].to_numpy(dtype=np.int64),
            frame=frame,
        )

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class TrainingData:
    train: SplitArrays
    val: SplitArrays
    feature_names: tuple[str, ...]
    dataset_meta: dict[str, Any]


def subsample_by_hand(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Keep whole hands (random order) until ``max_rows`` rows are reached - decisions of one hand
    are never split, so the subsample stays a valid sample of game situations."""
    if max_rows is None or len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    hand_ids = frame["hand_id"].unique()
    rng.shuffle(hand_ids)
    sizes = frame.groupby("hand_id").size().reindex(hand_ids).to_numpy()
    keep = hand_ids[np.cumsum(sizes) <= max_rows]
    return frame[frame["hand_id"].isin(keep)].reset_index(drop=True)


def load_training_data(
    dataset_name: str,
    max_train_rows: int | None = None,
    seed: int = 0,
    extra_train_frames: Sequence[pd.DataFrame] = (),
    directory: Path | None = None,
) -> TrainingData:
    """Load the train/validation splits of a generated dataset (validating the feature schema).

    ``extra_train_frames`` appends DAgger rows to the training split only.
    """
    train = load_dataset(dataset_name, directory, split="train")
    if extra_train_frames:
        train = pd.concat([train, *extra_train_frames], ignore_index=True)
    train = subsample_by_hand(train, max_train_rows, seed)
    val = load_dataset(dataset_name, directory, split="val")
    return TrainingData(
        train=SplitArrays.from_frame(train),
        val=SplitArrays.from_frame(val),
        feature_names=tuple(FEATURE_NAMES),
        dataset_meta=load_meta(dataset_name, directory),
    )


def class_weights(y: NDArray[np.int64], power: float = 0.5) -> NDArray[np.float64]:
    """Per-class weights ``(n / (K * count)) ** power`` (power 0 = none, 1 = fully balanced).

    A fractional power keeps the rare classes audible without letting a handful of all-in rows dominate
    the loss. Classes absent from ``y`` get weight 1 (they cannot be predicted anyway).
    """
    counts = np.bincount(y, minlength=NUM_ABSTRACT_ACTIONS).astype(np.float64)
    weights = np.ones(NUM_ABSTRACT_ACTIONS)
    present = counts > 0
    weights[present] = (len(y) / (present.sum() * counts[present])) ** power
    return weights


def sample_weights(y: NDArray[np.int64], power: float = 0.5) -> NDArray[np.float64]:
    return class_weights(y, power)[y]


def masked_argmax(proba: NDArray[np.floating], legal: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Most probable *legal* action per row - what the agent would actually play."""
    return np.argmax(np.where(legal, proba, -np.inf), axis=1).astype(np.int64)


def evaluate_classifier(
    proba: NDArray[np.floating],
    y: NDArray[np.int64],
    legal: NDArray[np.bool_],
    street: NDArray[np.int64] | None = None,
) -> dict[str, Any]:
    """Imitation metrics of class probabilities ``proba`` (rows x 5) against the teacher labels ``y``.

    * ``accuracy`` - arg-max of the raw probabilities;
    * ``masked_accuracy`` / ``masked_macro_f1`` / ``masked_balanced_accuracy`` - arg-max over *legal* actions;
    * ``log_loss`` - on the raw probabilities (clipped), a calibration-sensitive measure;
    * ``illegal_mass`` - mean probability the raw model puts on illegal actions (what masking removes);
    * per-class precision / recall / F1 / support and the masked confusion matrix;
    * accuracy per street.
    """
    proba = np.asarray(proba, dtype=np.float64)
    pred_raw = proba.argmax(axis=1)
    pred = masked_argmax(proba, legal)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=CLASS_LABELS, zero_division=0
    )
    result: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float((pred_raw == y).mean()),
        "masked_accuracy": float((pred == y).mean()),
        "masked_macro_f1": float(f1_score(y, pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "masked_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-12, 1.0), labels=CLASS_LABELS)),
        "illegal_mass": float(np.where(legal, 0.0, proba).sum(axis=1).mean()),
        "per_class": {
            ABSTRACT_ACTION_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in CLASS_LABELS
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASS_LABELS).tolist(),
        "confusion_matrix_labels": list(ABSTRACT_ACTION_NAMES),
    }
    if street is not None:
        result["masked_accuracy_by_street"] = {
            STREET_NAMES[s]: float((pred[street == s] == y[street == s]).mean())
            for s in range(len(STREET_NAMES))
            if (street == s).any()
        }
    return result


# --------------------------------------------------------------------------------------------
# DAgger
# --------------------------------------------------------------------------------------------
FitFunction = Callable[[pd.DataFrame, pd.DataFrame, Path, int], dict[str, Any]]
"""``fit(train_frame, val_frame, out_dir, round_index) -> metrics``: train a model and save its
artifacts (including ``metadata.json``) into ``out_dir``; round 0 is plain behaviour cloning."""


@dataclass(frozen=True)
class DaggerSettings:
    rounds: int  # number of on-policy aggregation rounds after the behaviour-cloning round (0 = plain BC)
    hands_per_round: int
    teacher_beta: float  # chance that the teacher (not the learner) acts in round 1; halves every round
    n_workers: int
    seed: int


def run_dagger(
    kind: str,
    fit: FitFunction,
    base_train: pd.DataFrame,
    val: pd.DataFrame,
    work_dir: Path,
    settings: DaggerSettings,
    on_round: Callable[[int, dict[str, Any]], None] | None = None,
    dataset_config: "DatasetConfig | None" = None,
) -> list[dict[str, Any]]:
    """Behaviour cloning followed by ``settings.rounds`` rounds of Dataset Aggregation (DAgger).

    Round 0 fits on ``base_train``. In every later round the *current* model (registry ``kind``,
    loaded from the previous round's directory) plays hands against the dataset's opponent mixture;
    the canonical teacher labels every state the learner visited; the new rows are appended to the
    training set and the model is refit from scratch. This addresses the covariate shift of plain
    behaviour cloning (a learner that makes small mistakes drifts into states the teacher-driven
    dataset never covered).

    ``dataset_config`` (game rules, opponent mixture, match lengths) should be the one the base dataset
    was generated with, so the on-policy rows come from the same game; the default is ``DatasetConfig()``.

    Returns one record per round (``{"round", "model_dir", "n_train_rows", "metrics"}``); the caller
    decides which round to keep (use held-out *validation* seeds, never the tournament seeds).
    ``on_round`` may add further keys to the record (e.g. policy-validation results).
    """
    from poker_arena.data.generation import collect_on_policy  # orchestration code lives in poker_arena.data

    history: list[dict[str, Any]] = []
    train = base_train
    for k in range(settings.rounds + 1):
        out_dir = work_dir / f"round_{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = fit(train, val, out_dir, k)
        record = {"round": k, "model_dir": str(out_dir), "n_train_rows": int(len(train)), "metrics": metrics}
        history.append(record)
        if on_round is not None:
            on_round(k, record)
        if k == settings.rounds:
            break
        beta = settings.teacher_beta * 0.5**k  # classic DAgger: trust the teacher less every round
        new_rows = collect_on_policy(
            AgentSpec(kind, {"model_dir": str(out_dir)}),
            settings.hands_per_round,
            (settings.seed, k + 1),
            round_index=k + 1,
            split="train",
            teacher_beta=beta,
            n_workers=settings.n_workers,
            config=dataset_config,
        )
        train = pd.concat([train, new_rows], ignore_index=True)
    return history
