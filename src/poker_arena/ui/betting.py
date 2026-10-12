"""Legal bet-sizing helpers for the human's controls (pure functions of an ``Observation``)."""

from __future__ import annotations

from dataclasses import dataclass

from poker_arena.engine.actions import pot_fraction_raise_to
from poker_arena.engine.observation import Observation

HALF_POT, FULL_POT = 0.5, 1.0


@dataclass(frozen=True, slots=True)
class QuickRaise:
    """A preset raise size; ``enabled`` is False when it is illegal or duplicates an earlier preset."""

    key: str
    label: str
    amount: int
    enabled: bool


def call_label(obs: Observation) -> str:
    if obs.legal.can_check:
        return "Check"
    amount = obs.legal.call_amount
    return f"Call {amount:,}" + (" (all-in)" if amount >= obs.my_stack else "")


def raise_label(obs: Observation, amount: int) -> str:
    verb = "Bet" if obs.opp_bet == 0 else "Raise to"
    return f"{verb} {amount:,}" + (" (all-in)" if amount >= obs.legal.max_raise_to else "")


def _clamped(obs: Observation, fraction: float) -> int:
    return pot_fraction_raise_to(fraction, obs.pot, obs.my_bet, obs.opp_bet, obs.legal)


def default_raise(obs: Observation) -> int:
    """Initial slider position: a half-pot raise, clamped to the legal range."""
    return _clamped(obs, HALF_POT)


def quick_raises(obs: Observation) -> list[QuickRaise]:
    """Min / half pot / pot / all-in, mapped to legal amounts (disabled when raising is impossible)."""
    legal = obs.legal
    presets = (
        ("min", "Min", legal.min_raise_to),
        ("half", "½ pot", _clamped(obs, HALF_POT)),
        ("pot", "Pot", _clamped(obs, FULL_POT)),
        ("allin", "All-in", legal.max_raise_to),
    )
    seen: set[int] = set()
    out = []
    for key, name, amount in presets:
        enabled = legal.can_raise and amount not in seen
        seen.add(amount)
        out.append(QuickRaise(key, f"{name} {amount:,}" if legal.can_raise else name, amount, enabled))
    return out
