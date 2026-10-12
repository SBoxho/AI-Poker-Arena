"""The common agent interface.

Every player - rule-based baseline, EBM, XGBoost, genetic algorithm, and any *future* agent
(reinforcement-learning network, local LLM via Ollama, specialised "System One" models, ...) -
implements :class:`Agent`. The poker engine, tournament system and Streamlit app only depend on
this interface, so new architectures plug in without touching them.

Contract
--------
* ``act(obs)`` receives **only** an :class:`~poker_arena.engine.observation.Observation` and returns a
  :class:`Decision` holding a *concrete* :class:`~poker_arena.engine.actions.Action`. The engine rejects
  illegal actions, so an agent must choose from ``obs.legal`` (or ``obs.abstract_mask``).
* ``reset(seed)`` is called at the start of every match: clear per-match memory and re-seed the
  agent's private RNG so that simulations are reproducible.
* ``end_hand(summary)`` is called after each hand with the public outcome (opponent hole cards only
  if they were shown at showdown). Stateless agents can ignore it.
* ``explain(obs, decision)`` optionally returns a human readable :class:`Explanation`; only called
  by the UI / analysis code, never inside tournaments unless explicitly requested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from poker_arena.engine.actions import (
    AbstractAction,
    Action,
    abstract_to_action,
)
from poker_arena.engine.observation import Observation
from poker_arena.engine.opponent import PublicHandSummary


class ModelArtifactError(RuntimeError):
    """A trained agent's saved artifacts are missing, corrupt or incompatible.

    Raised loudly - the system never silently substitutes random play for a model that failed to load.
    """


@dataclass(frozen=True, slots=True)
class Explanation:
    """Compact, JSON-serialisable account of why an action was chosen.

    ``summary``  one or two readable sentences.
    ``details``  architecture specific payload, e.g. ``{"probabilities": {...}, "top_features": [...]}``
                 for EBM/XGBoost or ``{"rules": [...], "genes": {...}}`` for the genetic agent.
    """

    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Decision:
    """An agent's answer to an observation."""

    action: Action
    abstract: AbstractAction | None = None  # set by discretised agents
    explanation: Explanation | None = None  # optional, may be filled lazily by ``Agent.explain``

    @classmethod
    def from_abstract(
        cls, obs: Observation, abstract: AbstractAction, explanation: Explanation | None = None
    ) -> "Decision":
        action = abstract_to_action(abstract, obs.pot, obs.my_bet, obs.opp_bet, obs.legal)
        return cls(action=action, abstract=abstract, explanation=explanation)


class Agent(ABC):
    """Base class of every poker player."""

    #: Display name, unique within a tournament (e.g. ``"EBM-v1"``).
    name: str = "agent"
    #: Short architecture label shown in the UI (e.g. ``"Explainable Boosting Machine"``).
    architecture: str = "unknown"
    #: One-line description of how the agent decides.
    description: str = ""
    #: True when :meth:`explain` returns something useful.
    supports_explanations: bool = False

    def reset(self, seed: int | None = None) -> None:  # noqa: B027 - intentionally optional
        """Called once at the start of a match. Reset memory and re-seed private RNGs."""

    @abstractmethod
    def act(self, obs: Observation) -> Decision:
        """Choose an action for the given restricted observation."""

    def end_hand(self, summary: PublicHandSummary) -> None:  # noqa: B027 - intentionally optional
        """Called after each hand with the publicly visible outcome."""

    def explain(self, obs: Observation, decision: Decision) -> Explanation | None:
        """Explain ``decision`` taken for ``obs``. Default: no explanation available."""
        return decision.explanation

    def describe(self) -> dict[str, Any]:
        """Metadata for the UI / result files."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "description": self.description,
            "supports_explanations": self.supports_explanations,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
