"""Opponent modelling from *public* information only.

An :class:`OpponentTracker` is fed one :class:`PublicHandSummary` per completed hand - the same
object both players are given - and exposes smoothed tendencies as :class:`OpponentStats`.
Because the summary only contains actions and (at showdown) cards that were legitimately
revealed, these statistics can never leak private information.
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_arena.engine.observation import OpponentStats, PublicAction, Street

# Pseudo-count that pulls early estimates towards the neutral prior in OpponentStats.
SMOOTHING_WEIGHT = 5.0
_PRIOR = OpponentStats()


@dataclass(frozen=True, slots=True)
class PublicHandSummary:
    """What both players know after a hand: public actions, board, and the showdown if any.

    Everything is expressed from the *recipient's* point of view (``by_me`` in the history).
    """

    hand_number: int
    was_button: bool
    history: tuple[PublicAction, ...]
    board: tuple[str, ...]
    my_payoff: int  # net chips won (+) / lost (-) by the recipient
    showdown: bool
    opp_hole: tuple[str, str] | None  # only set if the opponent's cards were shown at showdown


def _smooth(successes: float, trials: float, prior: float) -> float:
    return (successes + prior * SMOOTHING_WEIGHT) / (trials + SMOOTHING_WEIGHT)


@dataclass
class OpponentTracker:
    """Accumulates counts about the opponent across hands of one match."""

    hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    postflop_aggressive: int = 0
    postflop_calls: int = 0
    faced_aggression: int = 0
    folded_to_aggression: int = 0
    showdowns: int = 0

    def update(self, summary: PublicHandSummary) -> None:
        self.hands += 1
        opp_actions = [a for a in summary.history if not a.by_me]
        preflop = [a for a in opp_actions if a.street == Street.PREFLOP]
        self.vpip_hands += any(a.kind in ("call", "bet", "raise") for a in preflop)
        self.pfr_hands += any(a.kind in ("bet", "raise") for a in preflop)
        for a in opp_actions:
            if a.street == Street.PREFLOP:
                continue
            if a.kind in ("bet", "raise"):
                self.postflop_aggressive += 1
            elif a.kind == "call":
                self.postflop_calls += 1
        previous: PublicAction | None = None
        for a in summary.history:
            if not a.by_me and previous is not None and previous.by_me and previous.kind in ("bet", "raise"):
                self.faced_aggression += 1
                self.folded_to_aggression += a.kind == "fold"
            previous = a
        self.showdowns += summary.showdown

    def stats(self) -> OpponentStats:
        aggr_trials = self.postflop_aggressive + self.postflop_calls
        return OpponentStats(
            hands=self.hands,
            vpip=_smooth(self.vpip_hands, self.hands, _PRIOR.vpip),
            pfr=_smooth(self.pfr_hands, self.hands, _PRIOR.pfr),
            aggression=_smooth(self.postflop_aggressive, aggr_trials, _PRIOR.aggression),
            fold_to_bet=_smooth(self.folded_to_aggression, self.faced_aggression, _PRIOR.fold_to_bet),
            showdown_rate=_smooth(self.showdowns, self.hands, _PRIOR.showdown_rate),
        )

    def reset(self) -> None:
        self.hands = self.vpip_hands = self.pfr_hands = 0
        self.postflop_aggressive = self.postflop_calls = 0
        self.faced_aggression = self.folded_to_aggression = self.showdowns = 0
