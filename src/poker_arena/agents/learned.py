"""Base class for agents that wrap a probabilistic classifier over the shared feature schema.

EBM and XGBoost agents differ only in *how a model turns a feature vector into class probabilities*
and in *how that decision is explained*. Everything that matters for fairness and legality lives
here once:

* the observation is converted with :func:`poker_arena.features.build_features` - exactly what the
  training pipeline used, so training and inference cannot drift apart;
* probabilities are **masked to the legal abstract actions** (``obs.abstract_mask``) and
  renormalised, then either the arg-max (``greedy``, default) or a sample (``sample``) is chosen -
  the model can never produce an illegal action and never falls back to random play;
* artifacts are validated on load (feature schema hash, feature names, class layout) and a clear
  :class:`~poker_arena.agents.base.ModelArtifactError` is raised when anything is missing or stale.

Artifact layout (``artifacts/models/<subdir>/``): the model file(s) written by the trainer plus
``metadata.json`` containing at least ``artifact_schema_version``, ``feature_schema_hash``,
``feature_names`` and ``class_names`` (see :func:`validate_metadata`).

Explanation payload (``Explanation.details``) shared by every learned agent so that the UI renders
them uniformly::

    {
      "probabilities": {"fold": 0.02, "check_call": 0.61, ...},   # masked + renormalised, legal actions only
      "model_probabilities": {...},                                # raw model output for all five actions
      "chosen": "check_call", "legal": ["check_call", "raise_small", ...],
      "top_features": [{"feature": "equity", "value": 0.63, "contribution": 1.42}, ...],  # |contribution| desc.
      "contribution_unit": "log-odds"                              # or "shap (log-odds)", ...
    }
"""

from __future__ import annotations

import hashlib
import json
import random
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from poker_arena.agents.base import Agent, Decision, Explanation, ModelArtifactError
from poker_arena.config import MODELS_DIR
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.engine.observation import Observation
from poker_arena.features import FEATURE_NAMES, build_features, schema_hash

ARTIFACT_SCHEMA_VERSION = 1
METADATA_FILE = "metadata.json"
DECISION_MODES = ("greedy", "sample")
TOP_FEATURES_SHOWN = 8


def validate_metadata(metadata: dict[str, Any], model_dir: Path) -> None:
    """Check a loaded ``metadata.json`` against the code that is about to use it."""
    where = f"model directory {model_dir}"
    version = metadata.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError(
            f"Incompatible artifact schema in {where}: found {version!r}, this code expects "
            f"{ARTIFACT_SCHEMA_VERSION}. Retrain the model."
        )
    if metadata.get("feature_schema_hash") != schema_hash() or list(metadata.get("feature_names", [])) != list(
        FEATURE_NAMES
    ):
        raise ModelArtifactError(
            f"The model in {where} was trained with a different feature schema "
            f"({metadata.get('feature_schema_hash')!r} vs current {schema_hash()!r}). Retrain the model."
        )
    if list(metadata.get("class_names", [])) != list(ABSTRACT_ACTION_NAMES):
        raise ModelArtifactError(f"The model in {where} uses a different action space. Retrain the model.")


def file_fingerprint(*paths: Path) -> str:
    """Short SHA-256 over the model files - stored in ``describe()`` so results name the exact model."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def read_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / METADATA_FILE
    if not path.exists():
        raise ModelArtifactError(f"Missing {path}. Train the model first (see README) - no fallback is used.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelArtifactError(f"{path} is not valid JSON ({exc}). Retrain the model.") from exc


def format_feature_value(value: Any) -> str:
    """``'0.63'`` for a feature value, ``'(0.63, 1.00)'`` for the value pair of an interaction term."""
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(f"{v:.2f}" for v in value) + ")"
    return f"{value:.2f}"


def mask_and_normalise(proba: NDArray[np.floating], legal: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Zero out illegal actions and renormalise; raises if no legal action has any probability mass."""
    masked = np.where(legal, np.asarray(proba, dtype=np.float64), 0.0)
    total = masked.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("The model assigned no probability to any legal action")
    return masked / total


class LearnedPolicyAgent(Agent):
    """An agent that picks the legal abstract action preferred by a probabilistic classifier."""

    #: subdirectory of ``artifacts/models`` holding the artifacts (set by subclasses)
    MODEL_SUBDIR: ClassVar[str] = ""
    supports_explanations = True

    def __init__(
        self,
        metadata: dict[str, Any],
        name: str,
        decision_mode: str = "greedy",
        seed: int = 0,
        fingerprint: str = "",
    ) -> None:
        if decision_mode not in DECISION_MODES:
            raise ValueError(f"decision_mode must be one of {DECISION_MODES}")
        self.name = name
        self.metadata = metadata
        self.decision_mode = decision_mode
        self.fingerprint = fingerprint
        self._seed = seed
        self._rng = random.Random(seed)
        self._last: tuple[Observation, NDArray[np.float32], NDArray[np.float64]] | None = None

    # --------------------------------------------------------------- model API
    @abstractmethod
    def predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        """Class probabilities for ONE feature vector ``x`` (shape ``(NUM_FEATURES,)``).

        Returns shape ``(5,)`` aligned with :class:`AbstractAction` (classes the model never saw get 0).
        """

    @abstractmethod
    def top_features(self, x: NDArray[np.float32], action: AbstractAction) -> list[dict[str, Any]]:
        """Per-feature contributions towards ``action`` for this input, |contribution| descending.

        Each entry: ``{"feature": name, "value": feature value, "contribution": float}``.
        """

    #: unit of the contributions returned by :meth:`top_features` (shown in the UI)
    contribution_unit: str = "log-odds"

    # ------------------------------------------------------------------ Agent
    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(self._seed if seed is None else seed)
        self._last = None

    def act(self, obs: Observation) -> Decision:
        x = build_features(obs)
        proba = self.predict_proba(x)
        masked = mask_and_normalise(proba, obs.abstract_mask)
        if self.decision_mode == "greedy":
            choice = int(np.argmax(masked))
        else:
            choice = self._rng.choices(range(NUM_ABSTRACT_ACTIONS), weights=masked.tolist())[0]
        self._last = (obs, x, proba)
        return Decision.from_abstract(obs, AbstractAction(choice))

    def explain(self, obs: Observation, decision: Decision) -> Explanation | None:
        if decision.abstract is None:
            return None
        if self._last is not None and self._last[0] is obs:
            _, x, proba = self._last
        else:
            x = build_features(obs)
            proba = self.predict_proba(x)
        legal = obs.abstract_mask
        masked = mask_and_normalise(proba, legal)
        chosen = decision.abstract
        features = self.top_features(x, chosen)[:TOP_FEATURES_SHOWN]
        legal_names = [ABSTRACT_ACTION_NAMES[i] for i in range(NUM_ABSTRACT_ACTIONS) if legal[i]]
        drivers = ", ".join(f"{f['feature']}={format_feature_value(f['value'])}" for f in features[:3])
        summary = (
            f"{self.architecture} chose {ABSTRACT_ACTION_NAMES[chosen]} "
            f"(p={masked[chosen]:.2f} among legal actions)" + (f"; main drivers: {drivers}." if drivers else ".")
        )
        return Explanation(
            summary=summary,
            details={
                "probabilities": {ABSTRACT_ACTION_NAMES[i]: float(masked[i]) for i in range(NUM_ABSTRACT_ACTIONS) if legal[i]},
                "model_probabilities": {ABSTRACT_ACTION_NAMES[i]: float(proba[i]) for i in range(NUM_ABSTRACT_ACTIONS)},
                "chosen": ABSTRACT_ACTION_NAMES[chosen],
                "legal": legal_names,
                "top_features": features,
                "contribution_unit": self.contribution_unit,
            },
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            decision_mode=self.decision_mode,
            model_fingerprint=self.fingerprint,
            trained_at=self.metadata.get("created_at"),
            feature_schema_hash=self.metadata.get("feature_schema_hash"),
        )
        return info

    # ------------------------------------------------------------------ loading
    @classmethod
    def resolve_dir(cls, model_dir: Path | str | None) -> Path:
        return Path(model_dir) if model_dir is not None else MODELS_DIR / cls.MODEL_SUBDIR
