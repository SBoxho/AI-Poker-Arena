"""The Explainable Boosting Machine (EBM) poker agent.

Why an EBM
----------
An EBM is a *glass-box* additive model: the logit of every action is ``intercept + sum of terms``,
where a term is a lookup table over one feature (a **shape function**, e.g. "how does equity change
the log-odds of raising?") or over a pair of features (an **interaction surface**, e.g. "equity x
street"). A decision is therefore explained *exactly* by its own terms - the contributions shown to
the user add up to the model's logit, unlike post-hoc approximations (SHAP, LIME).

What this module adds on top of :class:`~poker_arena.agents.learned.LearnedPolicyAgent`
* loading with loud :class:`~poker_arena.agents.base.ModelArtifactError` failures (missing or corrupt
  model file, stale feature schema, wrong action space, incompatible ``interpret`` version);
* :meth:`EBMAgent.top_features`: per-term contributions to the chosen action's log-odds, single
  features and pairwise interactions side by side, largest first;
* single-row inference through InterpretML's own ``predict_proba`` / ``eval_terms`` (about one
  millisecond per decision - cheaper than the equity feature itself), so there is no second
  re-implementation of the binning rules that could drift away from the library.

The EBM *imitates* the transparent teacher (``agents/teacher.py``); imitation accuracy is not poker
skill. Training and the separate measurement of playing strength live in ``training/ebm.py``.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from numpy.typing import NDArray

from poker_arena.agents.base import ModelArtifactError
from poker_arena.agents.learned import (
    LearnedPolicyAgent,
    file_fingerprint,
    read_metadata,
    validate_metadata,
)
from poker_arena.engine.actions import NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.features import FEATURE_NAMES, NUM_FEATURES

MODEL_FILE = "ebm.joblib"
TRAIN_COMMAND = ".venv/Scripts/python scripts/train_ebm.py"
INTERACTION_SEPARATOR = " x "
"""Joins the feature names of an interaction term ("equity x street")."""
FEATURE_VALUE_DECIMALS = 6
"""Rounding of the feature values reported in explanations (float32 noise is not information)."""


def term_label(features: tuple[int, ...]) -> str:
    """Readable name of an additive term from its feature indices: ``equity`` or ``equity x street``."""
    return INTERACTION_SEPARATOR.join(FEATURE_NAMES[i] for i in features)


def _installed_interpret_version() -> str:
    try:
        return importlib_metadata.version("interpret")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - interpret is imported above
        return "unknown"


def _load_model(path: Path, trained_with: str) -> ExplainableBoostingClassifier:
    """Unpickle the EBM; every failure becomes a :class:`ModelArtifactError` that says what to do."""
    try:
        model = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - unpickling can raise almost anything (truncation, other library version)
        raise ModelArtifactError(
            f"Cannot read the EBM file {path} ({type(exc).__name__}: {exc}). The file is corrupt or was written "
            f"by an incompatible library version (trained with interpret {trained_with}, installed "
            f"{_installed_interpret_version()}). Retrain with `{TRAIN_COMMAND}`."
        ) from exc
    if not isinstance(model, ExplainableBoostingClassifier) or not hasattr(model, "term_features_"):
        raise ModelArtifactError(
            f"{path} does not contain a fitted ExplainableBoostingClassifier (found {type(model).__name__}). "
            f"Retrain with `{TRAIN_COMMAND}`."
        )
    return model


def _check_model(model: ExplainableBoostingClassifier, path: Path) -> None:
    """The pickled model must match the code's feature order and action space, and must actually run."""
    if list(model.feature_names_in_) != list(FEATURE_NAMES):
        raise ModelArtifactError(
            f"The model in {path} was fitted on different features than the current schema. "
            f"Retrain with `{TRAIN_COMMAND}`."
        )
    classes = [int(c) for c in model.classes_]
    if classes != sorted(set(classes)) or not set(classes) <= set(range(NUM_ABSTRACT_ACTIONS)):
        raise ModelArtifactError(f"The model in {path} predicts classes {classes}, not abstract actions 0..4.")
    probe = np.zeros((1, NUM_FEATURES), dtype=np.float32)
    try:
        probability = model.predict_proba(probe)
        model.eval_terms(probe)
    except Exception as exc:  # noqa: BLE001 - a library mismatch can surface anywhere inside interpret
        raise ModelArtifactError(
            f"The model in {path} loads but cannot predict with the installed interpret "
            f"{_installed_interpret_version()} ({type(exc).__name__}: {exc}). Retrain with `{TRAIN_COMMAND}`."
        ) from exc
    if not np.isfinite(probability).all():
        raise ModelArtifactError(f"The model in {path} produces non-finite probabilities. Retrain with `{TRAIN_COMMAND}`.")


class EBMAgent(LearnedPolicyAgent):
    """Chooses the legal abstract action preferred by a fitted ``ExplainableBoostingClassifier``.

    The fitted model is read-only shared state: ``copy.copy(agent)`` followed by ``reset()`` gives an
    independent player (ARCHITECTURE.md section 5) that shares only the heavy model.
    """

    MODEL_SUBDIR = "ebm"
    architecture = "Explainable Boosting Machine"
    contribution_unit = "log-odds (EBM term contribution)"

    def __init__(
        self,
        model: ExplainableBoostingClassifier,
        metadata: dict[str, Any],
        name: str = "EBM",
        decision_mode: str = "greedy",
        seed: int = 0,
        fingerprint: str = "",
    ) -> None:
        super().__init__(metadata, name, decision_mode, seed, fingerprint)
        self.model = model
        self.description = (
            "Additive model of readable shape functions (plus a few pairwise interactions) trained to "
            "imitate the equity / pot-odds teacher; every decision decomposes into per-term contributions."
        )
        self._class_columns = np.array([int(c) for c in model.classes_], dtype=np.int64)
        self._term_features: tuple[tuple[int, ...], ...] = tuple(tuple(int(i) for i in f) for f in model.term_features_)
        self._term_labels = tuple(term_label(f) for f in self._term_features)

    # ----------------------------------------------------------------- loading
    @classmethod
    def load(
        cls,
        model_dir: Path | str | None = None,
        name: str = "EBM",
        decision_mode: str = "greedy",
        seed: int = 0,
    ) -> "EBMAgent":
        """Load ``artifacts/models/ebm/`` (or ``model_dir``); raises :class:`ModelArtifactError` on any problem."""
        directory = cls.resolve_dir(model_dir)
        metadata = read_metadata(directory)
        validate_metadata(metadata, directory)
        path = directory / MODEL_FILE
        if not path.is_file():
            raise ModelArtifactError(f"Missing {path}. Train the model with `{TRAIN_COMMAND}` - no fallback is used.")
        model = _load_model(path, str(metadata.get("library_versions", {}).get("interpret", "unknown")))
        _check_model(model, path)
        return cls(model, metadata, name=name, decision_mode=decision_mode, seed=seed, fingerprint=file_fingerprint(path))

    # ------------------------------------------------------------- model access
    def predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        probability = np.zeros(NUM_ABSTRACT_ACTIONS, dtype=np.float64)
        probability[self._class_columns] = self.model.predict_proba(self._row(x))[0]
        return probability

    def term_contributions(self, x: NDArray[np.float32], action: AbstractAction) -> NDArray[np.float64]:
        """Contribution of every additive term to the log-odds (logit) of ``action``, shape ``(n_terms,)``.

        ``intercept[action] + contributions.sum()`` is the action's logit; the probabilities are the softmax
        of the five logits. A class the model never saw in training has no terms, so its contributions are 0.
        """
        scores = self.model.eval_terms(self._row(x))[0]  # (n_terms, n_classes_seen)
        column = np.flatnonzero(self._class_columns == int(action))
        return scores[:, column[0]] if column.size else np.zeros(len(self._term_features))

    def top_features(self, x: NDArray[np.float32], action: AbstractAction) -> list[dict[str, Any]]:
        """All additive terms by |contribution| to ``action``; interaction terms list both feature values.

        Entry: ``{"feature": "equity" | "equity x street", "value": float | [float, float],
        "contribution": float, "kind": "main" | "interaction"}``.
        """
        contributions = self.term_contributions(x, action)
        entries: list[dict[str, Any]] = []
        for term in np.argsort(-np.abs(contributions), kind="stable"):
            features = self._term_features[term]
            values = [round(float(x[i]), FEATURE_VALUE_DECIMALS) for i in features]
            entries.append(
                {
                    "feature": self._term_labels[term],
                    "value": values[0] if len(values) == 1 else values,
                    "contribution": float(contributions[term]),
                    "kind": "main" if len(features) == 1 else "interaction",
                }
            )
        return entries

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            n_terms=len(self._term_features),
            n_interaction_terms=sum(len(f) > 1 for f in self._term_features),
            dagger_round=self.metadata.get("selected_round"),
        )
        return info

    @staticmethod
    def _row(x: NDArray[np.float32]) -> NDArray[np.float32]:
        row = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if row.shape[1] != NUM_FEATURES or not np.isfinite(row).all():
            raise ValueError(f"Expected {NUM_FEATURES} finite features, got shape {row.shape}")
        return row
