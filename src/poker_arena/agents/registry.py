"""Agent registry: create agents by name, discover which ones are available.

Agents are described by a small, picklable :class:`AgentSpec` (``kind`` + keyword ``params``) so
that tournament worker processes can rebuild them. Trained agents (EBM / XGBoost / GA) are
imported lazily and load their saved artifacts from ``artifacts/models/<kind>/``. If artifacts are
missing or incompatible a :class:`~poker_arena.agents.base.ModelArtifactError` is raised - there is
no silent fallback to random play.

Adding a future architecture = write an :class:`Agent` subclass and call :func:`register_agent`
(or add it to ``_LAZY``). Nothing else in the engine / tournament / UI needs to change.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from poker_arena.agents.base import Agent, ModelArtifactError
from poker_arena.agents.baselines import CallStationAgent, ManiacAgent, RandomAgent

AgentFactory = Callable[..., Agent]


@dataclass(frozen=True)
class AgentSpec:
    """Picklable recipe for building an agent."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None  # optional display-name override

    def __hash__(self) -> int:
        # ``params`` is a dict (unhashable), which would make the generated dataclass hash raise; hash its
        # canonical text so specs can key dicts / live in sets (e.g. tournament result tables).
        return hash((self.kind, repr(sorted(self.params.items())), self.label))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params), "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "AgentSpec":
        return cls(d["kind"], dict(d.get("params", {})), d.get("label"))


# kind -> callable building the agent from keyword params
_FACTORIES: dict[str, AgentFactory] = {
    "random": RandomAgent,
    "call_station": CallStationAgent,
    "maniac": ManiacAgent,
}

# kind -> "module:attr.path" imported on demand (trained / heavier agents)
_LAZY: dict[str, str] = {
    "teacher": "poker_arena.agents.teacher:TeacherAgent",
    "ebm": "poker_arena.agents.ebm_agent:EBMAgent.load",
    "xgboost": "poker_arena.agents.xgb_agent:XGBoostAgent.load",
    "ga": "poker_arena.agents.ga_agent:GeneticAgent.load",
}

#: Human friendly names for known kinds.
KIND_LABELS = {
    "random": "Random",
    "call_station": "Call Station",
    "maniac": "Maniac",
    "teacher": "Equity Teacher",
    "ebm": "EBM",
    "xgboost": "XGBoost",
    "ga": "Genetic Algorithm",
}


def register_agent(kind: str, factory: AgentFactory) -> None:
    """Register a new agent kind (e.g. a future RL or LLM agent)."""
    _FACTORIES[kind] = factory


def known_kinds() -> list[str]:
    return list(dict.fromkeys([*_FACTORIES, *_LAZY]))


def _resolve(kind: str) -> AgentFactory:
    if kind in _FACTORIES:
        return _FACTORIES[kind]
    if kind not in _LAZY:
        raise KeyError(f"Unknown agent kind {kind!r}. Known: {known_kinds()}")
    module_name, _, attr_path = _LAZY[kind].partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            # the agent module itself (or one of its parent packages) is absent
            raise ModelArtifactError(f"Agent kind {kind!r} is not implemented yet ({exc}).") from exc
        # the module exists but one of ITS dependencies is missing - say so instead of "not implemented"
        raise ModelArtifactError(f"Agent kind {kind!r} cannot be imported: missing dependency ({exc}).") from exc
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def create_agent(spec: AgentSpec | str) -> Agent:
    """Build an agent from a spec (or just a kind string). Raises ``ModelArtifactError`` loudly."""
    if isinstance(spec, str):
        spec = AgentSpec(spec)
    agent = _resolve(spec.kind)(**spec.params)
    if spec.label:
        agent.name = spec.label
    return agent


@dataclass(frozen=True)
class AgentAvailability:
    kind: str
    label: str
    available: bool
    detail: str = ""


def check_availability(kind: str) -> AgentAvailability:
    """Try to build ``kind`` and report whether it is usable (used by the UI to grey out agents)."""
    label = KIND_LABELS.get(kind, kind)
    try:
        create_agent(kind)
    except ModelArtifactError as exc:
        return AgentAvailability(kind, label, False, str(exc))
    return AgentAvailability(kind, label, True)


def available_agents() -> list[AgentAvailability]:
    return [check_availability(k) for k in known_kinds()]
