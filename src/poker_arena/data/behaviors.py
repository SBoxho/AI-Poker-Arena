"""Behaviour and opponent policies for data collection, and the recorder that logs decision states.

Why a *behaviour* policy at all? A learner that only sees the states a perfect teacher visits never
sees its own mistakes. The acting seat therefore follows a mixture of policies (the teacher, the
teacher with epsilon-random exploration, uniformly random play, teacher style presets, ...) while
every visited state is still labelled by the canonical teacher.

Policy names used in the configuration mixtures:

``teacher``        the canonical teacher                   ``random``         uniformly random legal action
``teacher_eps``    teacher, but with probability epsilon   ``call_station``   always check/call
                   a uniformly random legal action         ``maniac``         bets and raises relentlessly
``tight`` ``loose`` ``aggressive`` ``passive``  teacher style presets
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from poker_arena.agents.base import Agent, Decision
from poker_arena.agents.registry import AgentSpec, create_agent
from poker_arena.agents.teacher import TEACHER_STYLES, TeacherDecision, TeacherPolicy
from poker_arena.engine.observation import Observation
from poker_arena.engine.opponent import PublicHandSummary
from poker_arena.features import build_features

TEACHER_EPS = "teacher_eps"
LEARNER = "learner"
BASELINE_POLICIES = ("random", "call_station", "maniac")
POLICY_NAMES: tuple[str, ...] = (
    "teacher", TEACHER_EPS, *BASELINE_POLICIES, *(s for s in TEACHER_STYLES if s != "canonical"),
)
NO_ACTION = -1
"""``behavior_action`` value when the executed decision carries no abstract action."""


class MixedPolicyAgent(Agent):
    """Per decision: act like ``alternative`` with probability ``p``, otherwise like ``primary``.

    Exploration is ``MixedPolicyAgent(teacher, random, epsilon)``; the DAgger mixture
    (probability ``beta`` of following the teacher) is ``MixedPolicyAgent(learner, teacher, beta)``.
    Its coin flips come from a private RNG re-seeded by :meth:`reset`.
    """

    def __init__(self, primary: Agent, alternative: Agent, probability: float, name: str) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        self.primary = primary
        self.alternative = alternative
        self.probability = probability
        self.name = name
        self._rng = random.Random(0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(0 if seed is None else seed)
        self.primary.reset(self._rng.getrandbits(62))
        self.alternative.reset(self._rng.getrandbits(62))

    def act(self, obs: Observation) -> Decision:
        chosen = self.alternative if self._rng.random() < self.probability else self.primary
        return chosen.act(obs)

    def end_hand(self, summary: PublicHandSummary) -> None:
        self.primary.end_hand(summary)
        self.alternative.end_hand(summary)


def policy_spec(name: str) -> AgentSpec:
    """Registry spec of a plain (non-mixed) policy name."""
    if name == "teacher":
        return AgentSpec("teacher", {}, "Teacher")
    if name in TEACHER_STYLES:
        return AgentSpec("teacher", {"style": name}, f"Teacher-{name}")
    if name in BASELINE_POLICIES:
        return AgentSpec(name)
    raise KeyError(f"Unknown policy {name!r}; known: {POLICY_NAMES}")


def build_policy(name: str, epsilon: float = 0.0) -> Agent:
    """Build a behaviour/opponent agent from its configuration name."""
    if name == TEACHER_EPS:
        return MixedPolicyAgent(
            create_agent(policy_spec("teacher")), create_agent(policy_spec("random")), epsilon, "Teacher-eps"
        )
    return create_agent(policy_spec(name))


def validate_mix(mix: dict[str, float], allowed: tuple[str, ...], what: str) -> None:
    """A mixture must name known policies and have positive total weight."""
    unknown = sorted(set(mix) - set(allowed))
    if unknown:
        raise ValueError(f"{what} names unknown policies {unknown}; allowed: {list(allowed)}")
    if any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError(f"{what} needs non-negative weights with a positive sum")


# ---------------------------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------------------------
@dataclass
class RowBuffer:
    """Column-wise accumulator of labelled decision states (one list per column)."""

    features: list[NDArray[np.float32]] = field(default_factory=list)
    mask: list[NDArray[np.bool_]] = field(default_factory=list)
    label: list[int] = field(default_factory=list)
    margin: list[float] = field(default_factory=list)
    rule: list[str] = field(default_factory=list)
    hand_id: list[int] = field(default_factory=list)
    hand_index: list[int] = field(default_factory=list)
    decision_idx: list[int] = field(default_factory=list)
    behavior: list[str] = field(default_factory=list)
    opponent: list[str] = field(default_factory=list)
    behavior_action: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.label)


class RecordingAgent(Agent):
    """Wraps the acting seat's agent and logs ``(features, canonical-teacher label)`` per decision.

    The wrapper sees exactly what any agent sees (the :class:`Observation`); the wrapped agent decides
    what is *played*, the canonical teacher decides what is *labelled*. One instance covers one hand.
    """

    def __init__(
        self,
        inner: Agent,
        teacher: TeacherPolicy,
        buffer: RowBuffer,
        hand_id: int,
        hand_index: int,
        behavior: str,
        opponent: str,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.teacher = teacher
        self.buffer = buffer
        self.hand_id = hand_id
        self.hand_index = hand_index
        self.behavior = behavior
        self.opponent = opponent
        self._decision_idx = 0

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)

    def act(self, obs: Observation) -> Decision:
        features = build_features(obs)
        mask = obs.abstract_mask
        label: TeacherDecision = self.teacher.decide_features(features, mask)
        decision = self.inner.act(obs)
        buf = self.buffer
        buf.features.append(features)
        buf.mask.append(mask)
        buf.label.append(int(label.action))
        buf.margin.append(label.margin)
        buf.rule.append(label.rule)
        buf.hand_id.append(self.hand_id)
        buf.hand_index.append(self.hand_index)
        buf.decision_idx.append(self._decision_idx)
        buf.behavior.append(self.behavior)
        buf.opponent.append(self.opponent)
        buf.behavior_action.append(NO_ACTION if decision.abstract is None else int(decision.abstract))
        self._decision_idx += 1
        return decision

    def end_hand(self, summary: PublicHandSummary) -> None:
        self.inner.end_hand(summary)
