"""The XGBoost poker agent: gradient-boosted trees over the shared feature schema, explained with SHAP.

Why trees + SHAP
----------------
The model is trained to imitate the transparent equity teacher (see ``training/xgb.py``) on exactly the
same features and labels as the EBM, so the two *explainability styles* can be compared fairly:
the EBM is additive and its shape functions are the model; XGBoost is a black box of a few thousand
trees whose decisions are explained **after the fact** with exact TreeSHAP values.

Inference
---------
* ``Booster.inplace_predict`` on one float32 row: no ``DMatrix`` is built per decision, one thread,
  and (per the XGBoost docs) lock-free and safe to call from several threads on a shared booster.
* The probabilities are masked to ``obs.abstract_mask`` by :class:`LearnedPolicyAgent`, never here.

Explanations
------------
``top_features`` returns native TreeSHAP contributions (``Booster.predict(..., pred_contribs=True)``)
for the *chosen class margin*. They are additive: ``sum(contributions) + bias == raw margin`` of that
class (the test-suite checks it to 1e-4). The margin is the pre-softmax score, so a contribution is in
log-odds-like units of that class; it is **not** a probability and it explains the *model*, not poker.

Copy semantics (ARCHITECTURE.md section 5): the loaded :class:`xgboost.Booster` is read-only shared
state, ``reset`` rebinds the mutable per-match fields in the base class, so ``copy.copy(agent)`` +
``reset()`` gives an independent player that shares only the heavy model.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from poker_arena.agents.base import ModelArtifactError
from poker_arena.agents.learned import LearnedPolicyAgent, file_fingerprint, read_metadata, validate_metadata
from poker_arena.engine.actions import NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.features import FEATURE_NAMES, NUM_FEATURES

MODEL_FILE = "model.ubj"
"""XGBoost's native binary JSON: stable across XGBoost versions, ~35% smaller than text JSON."""
TRAIN_COMMAND = ".venv/Scripts/python scripts/train_xgboost.py"
EXPECTED_OBJECTIVE = "multi:softprob"
INFERENCE_THREADS = 1
"""One thread: a single 45-value row is far cheaper than spawning a thread team (and tournaments
already parallelise across matches)."""

# ``Booster.predict(DMatrix)`` keeps per-DMatrix caches and is documented as not thread safe, so the
# (rarely used, UI-only) SHAP path is serialised. ``inplace_predict`` needs no lock.
_DMATRIX_PREDICT_LOCK = threading.Lock()


def shap_contributions(booster: xgb.Booster, X: NDArray[np.floating]) -> NDArray[np.float32]:
    """Exact TreeSHAP values of every class margin, shape ``(rows, 5, NUM_FEATURES + 1)``.

    ``out[r, c, :-1]`` are the per-feature contributions to class ``c``'s raw margin for row ``r`` and
    ``out[r, c, -1]`` is the bias (the model's expected margin), so ``out[r, c].sum()`` equals the
    margin ``booster.predict(DMatrix(X), output_margin=True)[r, c]``.
    """
    matrix = np.asarray(X, dtype=np.float32).reshape(-1, NUM_FEATURES)
    dmatrix = xgb.DMatrix(matrix, feature_names=list(FEATURE_NAMES))
    with _DMATRIX_PREDICT_LOCK:
        contributions = booster.predict(dmatrix, pred_contribs=True, validate_features=False)
    return np.asarray(contributions, dtype=np.float32).reshape(len(matrix), NUM_ABSTRACT_ACTIONS, NUM_FEATURES + 1)


def class_probabilities(booster: xgb.Booster, X: NDArray[np.floating]) -> NDArray[np.float64]:
    """Class probabilities for a batch, shape ``(rows, 5)`` and exactly row-normalised.

    This is the agent's own inference path (``inplace_predict``) so that offline metrics and in-game
    decisions use the same numbers; the renormalisation removes the float32 rounding of the booster.
    """
    matrix = np.ascontiguousarray(X, dtype=np.float32).reshape(-1, NUM_FEATURES)
    proba = np.asarray(booster.inplace_predict(matrix, validate_features=False), dtype=np.float64)
    return proba / proba.sum(axis=1, keepdims=True)


def _major_version(version: str) -> int:
    try:
        return int(version.split(".")[0])
    except ValueError:
        return -1


def load_booster(model_path: Path, trained_with: str | None = None) -> xgb.Booster:
    """Read a saved booster and check that it really is the 5-class, 45-feature policy model.

    Raises :class:`ModelArtifactError` (never returns a half-valid model) when the file is missing,
    unreadable (corrupt / truncated / written by a newer XGBoost), or structurally not what this code
    expects. ``trained_with`` (the XGBoost version recorded in ``metadata.json``) only sharpens the message.
    """
    if not model_path.exists():
        raise ModelArtifactError(f"Missing model file {model_path}. Train it with `{TRAIN_COMMAND}` - no fallback is used.")
    installed = xgb.__version__
    hint = f"; trained with xgboost {trained_with}, installed {installed}" if trained_with else f"; installed xgboost {installed}"
    if trained_with and _major_version(trained_with) > _major_version(installed):
        raise ModelArtifactError(
            f"{model_path} was written by xgboost {trained_with}, which is newer than the installed {installed}. "
            f"Upgrade xgboost or retrain with `{TRAIN_COMMAND}`."
        )
    booster = xgb.Booster()
    try:
        booster.load_model(str(model_path))
        learner = json.loads(booster.save_config())["learner"]
    except (xgb.core.XGBoostError, OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelArtifactError(
            f"{model_path} could not be read as an XGBoost model ({exc}){hint}. Retrain with `{TRAIN_COMMAND}`."
        ) from exc
    params = learner.get("learner_model_param", {})
    objective = learner.get("objective", {}).get("name")
    if (
        int(params.get("num_feature", -1)) != NUM_FEATURES
        or int(params.get("num_class", -1)) != NUM_ABSTRACT_ACTIONS
        or objective != EXPECTED_OBJECTIVE
    ):
        raise ModelArtifactError(
            f"{model_path} has {params.get('num_feature')} features / {params.get('num_class')} classes / objective "
            f"{objective!r}; this code needs {NUM_FEATURES} / {NUM_ABSTRACT_ACTIONS} / {EXPECTED_OBJECTIVE!r}. "
            f"Retrain with `{TRAIN_COMMAND}`."
        )
    if booster.feature_names is not None and list(booster.feature_names) != list(FEATURE_NAMES):
        raise ModelArtifactError(f"{model_path} stores different feature names than the current schema. Retrain.")
    if booster.num_boosted_rounds() < 1:
        raise ModelArtifactError(f"{model_path} contains no trees. Retrain with `{TRAIN_COMMAND}`.")
    booster.set_param({"nthread": INFERENCE_THREADS})
    return booster


class XGBoostAgent(LearnedPolicyAgent):
    """Plays the legal abstract action preferred by an XGBoost classifier and explains it with SHAP."""

    MODEL_SUBDIR = "xgb"
    architecture = "XGBoost"
    description = (
        "Gradient-boosted decision trees trained to imitate the transparent equity teacher; every decision "
        "can be explained after the fact with exact TreeSHAP feature contributions."
    )
    contribution_unit = "SHAP value (log-odds margin)"

    def __init__(
        self,
        booster: xgb.Booster,
        metadata: dict[str, Any],
        name: str = "XGBoost",
        decision_mode: str = "greedy",
        seed: int = 0,
        fingerprint: str = "",
    ) -> None:
        super().__init__(metadata, name, decision_mode, seed, fingerprint)
        self._booster = booster  # read-only and shared between copies of this agent

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        model_dir: Path | str | None = None,
        name: str = "XGBoost",
        decision_mode: str = "greedy",
        seed: int = 0,
    ) -> "XGBoostAgent":
        """Load ``artifacts/models/xgb/`` (or ``model_dir``); raises :class:`ModelArtifactError` on any problem."""
        directory = cls.resolve_dir(model_dir)
        metadata = read_metadata(directory)
        validate_metadata(metadata, directory)
        model_path = directory / MODEL_FILE
        trained_with = (metadata.get("library_versions") or {}).get("xgboost")
        booster = load_booster(model_path, trained_with)
        fingerprint = file_fingerprint(model_path)
        stored = metadata.get("model_fingerprint")
        if stored != fingerprint:  # also catches truncation, which XGBoost's parser does not always notice
            raise ModelArtifactError(
                f"{model_path} does not belong to the metadata.json next to it (fingerprint {fingerprint} vs "
                f"{stored}). The file is damaged or from a different run; retrain with `{TRAIN_COMMAND}`."
            )
        return cls(booster, metadata, name, decision_mode, seed, fingerprint)

    @property
    def booster(self) -> xgb.Booster:
        """The underlying model (read-only by convention: it is shared between copies of the agent)."""
        return self._booster

    # --------------------------------------------------------------- model API
    def predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        return class_probabilities(self._booster, x)[0]

    def top_features(self, x: NDArray[np.float32], action: AbstractAction) -> list[dict[str, Any]]:
        contributions = shap_contributions(self._booster, x)[0, int(action), :NUM_FEATURES]
        order = np.argsort(-np.abs(contributions), kind="stable")
        return [
            {"feature": FEATURE_NAMES[i], "value": float(x[i]), "contribution": float(contributions[i])}
            for i in order
        ]
