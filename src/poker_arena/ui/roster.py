"""Agent discovery for the UI.

The roster is built from the registry, so a trained agent (EBM / XGBoost / GA ...) shows up as soon
as its artifacts exist - nothing here names a specific architecture. Anything that fails to load is
listed as *unavailable* together with the reason, never silently replaced.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

from poker_arena.agents.base import Agent
from poker_arena.agents.registry import KIND_LABELS, create_agent, known_kinds


@dataclass(frozen=True)
class RosterEntry:
    """One agent kind: either a ready-to-copy template or the reason it cannot be used."""

    kind: str
    label: str
    available: bool
    detail: str = ""  # why it is unavailable
    template: Agent | None = None  # read-only prototype; sessions work on copies
    architecture: str = ""
    description: str = ""
    supports_explanations: bool = False


def discover_roster(
    kinds: list[str] | None = None, factory: Callable[[str], Agent] = create_agent
) -> dict[str, RosterEntry]:
    """Try to build every known agent kind (ordered as in the registry)."""
    roster: dict[str, RosterEntry] = {}
    for kind in kinds if kinds is not None else known_kinds():
        label = KIND_LABELS.get(kind, kind)
        try:
            agent = factory(kind)
        except Exception as exc:  # noqa: BLE001 - any load problem must be shown, never crash the page
            roster[kind] = RosterEntry(kind, label, False, detail=str(exc) or type(exc).__name__)
            continue
        roster[kind] = RosterEntry(
            kind,
            label,
            True,
            template=agent,
            architecture=agent.architecture,
            description=agent.description,
            supports_explanations=agent.supports_explanations,
        )
    return roster


def spawn_agent(entry: RosterEntry, seed: int) -> Agent:
    """A private copy of the template with fresh per-match state (see ARCHITECTURE.md section 5)."""
    if entry.template is None:
        raise ValueError(f"Agent {entry.kind!r} is not available: {entry.detail}")
    agent = copy.copy(entry.template)
    agent.reset(seed)
    return agent
