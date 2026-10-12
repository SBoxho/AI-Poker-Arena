"""Trivial reference opponents (no learning, no equity). Useful for sanity checks and as sparring
partners; the stronger rule-based teacher lives in :mod:`poker_arena.agents.teacher`."""

from __future__ import annotations

import random

from poker_arena.agents.base import Agent, Decision, Explanation
from poker_arena.engine.actions import AbstractAction, NUM_ABSTRACT_ACTIONS
from poker_arena.engine.observation import Observation


class _SeededAgent(Agent):
    def __init__(self, name: str, seed: int = 0) -> None:
        self.name = name
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        # ``reset()`` without a seed returns to the construction seed (not a hidden constant), so an agent
        # built from ``AgentSpec(kind, {"seed": 7})`` stays reproducible when the caller does not re-seed.
        self._rng = random.Random(self._seed if seed is None else seed)


class RandomAgent(_SeededAgent):
    """Picks uniformly among the currently legal abstract actions."""

    architecture = "Random"
    description = "Uniformly random over legal fold / call / half-pot / pot / all-in."
    supports_explanations = True

    def __init__(self, name: str = "Random", seed: int = 0) -> None:
        super().__init__(name, seed)

    def act(self, obs: Observation) -> Decision:
        mask = obs.abstract_mask
        choices = [AbstractAction(i) for i in range(NUM_ABSTRACT_ACTIONS) if mask[i]]
        choice = self._rng.choice(choices)
        return Decision.from_abstract(obs, choice, Explanation(f"Random pick among {len(choices)} legal actions."))


class CallStationAgent(Agent):
    """Never folds, never raises: checks or calls every time."""

    architecture = "Rule-based"
    description = "Calling station - always check/call."
    supports_explanations = True

    def __init__(self, name: str = "CallStation") -> None:
        self.name = name

    def act(self, obs: Observation) -> Decision:
        return Decision.from_abstract(obs, AbstractAction.CHECK_CALL, Explanation("Always check/call."))


class ManiacAgent(_SeededAgent):
    """Raises relentlessly: pot-sized raises, shoving when re-raised or when pot raises are unavailable."""

    architecture = "Rule-based"
    description = "Maniac - pot-sized bet/raise whenever possible, otherwise call."
    supports_explanations = True

    def __init__(self, name: str = "Maniac", seed: int = 0, raise_probability: float = 0.85) -> None:
        super().__init__(name, seed)
        self.raise_probability = raise_probability

    def act(self, obs: Observation) -> Decision:
        mask = obs.abstract_mask
        if self._rng.random() < self.raise_probability:
            for pick in (AbstractAction.RAISE_POT, AbstractAction.ALL_IN, AbstractAction.RAISE_SMALL):
                if mask[pick]:
                    return Decision.from_abstract(obs, pick, Explanation("Maniac: bet big whenever possible."))
        return Decision.from_abstract(obs, AbstractAction.CHECK_CALL, Explanation("Maniac: just calling this time."))
