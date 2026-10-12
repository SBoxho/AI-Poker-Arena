"""``HandSession`` semantics (agents / humans meet the engine) and ``OpponentTracker`` statistics."""

from __future__ import annotations

import dataclasses
import random

import pytest

from poker_arena.agents.base import Agent, Decision, Explanation
from poker_arena.agents.baselines import CallStationAgent, ManiacAgent, RandomAgent
from poker_arena.config import derive_seed
from poker_arena.engine.actions import Action, IllegalActionError
from poker_arena.engine.deals import Deal
from poker_arena.engine.history import HandRecord
from poker_arena.engine.observation import Observation, OpponentStats, PublicAction, Street
from poker_arena.engine.opponent import SMOOTHING_WEIGHT, OpponentTracker, PublicHandSummary
from poker_arena.engine.session import HandSession, StepOutcome, play_hand

from core_helpers import DEFAULT_CFG, random_deal, random_stacks

DEAL = Deal.from_seed(2024)


# ================================================================================= test agents
class CountingStation(CallStationAgent):
    """Call station that counts the engine's callbacks."""

    def __init__(self, name: str = "Counter") -> None:
        super().__init__(name)
        self.end_hand_calls = 0
        self.summaries: list[PublicHandSummary] = []
        self.observations: list[Observation] = []

    def act(self, obs: Observation) -> Decision:
        self.observations.append(obs)
        return super().act(obs)

    def end_hand(self, summary: PublicHandSummary) -> None:
        self.end_hand_calls += 1
        self.summaries.append(summary)


class Cheater(Agent):
    """Always submits an illegal raise-to-1."""

    def __init__(self, name: str = "Cheater") -> None:
        self.name = name

    def act(self, obs: Observation) -> Decision:
        return Decision(Action.raise_to(1))


class SilentAgent(CallStationAgent):
    supports_explanations = False

    def explain(self, obs: Observation, decision: Decision) -> Explanation | None:
        return None


class ExplainingAgent(CallStationAgent):
    def explain(self, obs: Observation, decision: Decision) -> Explanation | None:
        return Explanation("because", {"street": int(obs.street), "legal_raise": obs.legal.can_raise})


# ================================================================================ HandSession
def test_human_seat_step_raises_and_submit_works() -> None:
    station = CountingStation()
    session = HandSession(DEFAULT_CFG, DEAL, 0, [None, station])  # seat 0 = human = button = first to act
    assert session.actor_seat == 0
    with pytest.raises(RuntimeError, match="externally controlled"):
        session.step()
    assert not session.engine.events, "a refused step must not change the hand"
    outcome = session.submit(Action.check_call())
    assert isinstance(outcome, StepOutcome)
    assert outcome.event.seat == 0 and outcome.event.kind == "call"
    assert outcome.decision.latency_ms == 0.0 and outcome.decision.abstract is None
    assert outcome.observation.my_hole == DEAL.button_hole and outcome.observation.legal.can_call
    # the station (seat 1) is an agent: step() works for it
    out2 = session.step()
    assert out2.event.seat == 1 and out2.event.kind == "check" and out2.decision.abstract == "CHECK_CALL"
    assert out2.decision.latency_ms >= 0 and station.observations[0].my_hole == DEAL.bb_hole


def test_all_human_hand_must_be_driven_with_submit() -> None:
    session = HandSession(DEFAULT_CFG, DEAL, 1, [None, None])
    with pytest.raises(RuntimeError):
        session.play_out()
    session.submit(Action.fold())
    assert session.is_complete and session.actor_seat is None
    record = session.finish()
    assert record.agent_names == ("human", "human") and record.result.winner == 0


def test_step_and_submit_after_the_hand_is_complete_raise() -> None:
    session = HandSession(DEFAULT_CFG, DEAL, 0, [CallStationAgent(), CallStationAgent()])
    session.play_out()
    with pytest.raises(RuntimeError, match="complete"):
        session.step()
    with pytest.raises(RuntimeError, match="complete"):
        session.submit(Action.check_call())


def test_illegal_agent_action_names_the_agent_and_leaves_the_hand_untouched() -> None:
    session = HandSession(DEFAULT_CFG, DEAL, 0, [Cheater("Mr. Cheat"), CallStationAgent()])
    with pytest.raises(IllegalActionError, match="Mr. Cheat") as info:
        session.step()
    assert "illegal action" in str(info.value) and "raise to 1" in str(info.value)
    assert isinstance(info.value.__cause__, IllegalActionError), "original engine error is chained"
    assert session.engine.events == [] and session.actor_seat == 0 and not session.is_complete
    assert session._decisions == [], "no decision record for a rejected action"
    session.submit(Action.check_call())  # still playable afterwards
    assert len(session.engine.events) == 1


def test_illegal_human_action_names_the_seat() -> None:
    session = HandSession(DEFAULT_CFG, DEAL, 0, [None, CallStationAgent()])
    with pytest.raises(IllegalActionError, match="seat 0"):
        session.submit(Action.raise_to(5))
    with pytest.raises(IllegalActionError, match="seat 0"):
        session.submit(Action.raise_to(10**9))
    session.submit(Action.check_call())
    session.step()  # BB checks the option
    # flop: seat 1 (BB, station) first; a human fold with a free check is illegal too
    session.step()
    with pytest.raises(IllegalActionError, match="free check"):
        session.submit(Action.fold())


def test_requires_exactly_two_seats() -> None:
    for agents in ([], [CallStationAgent()], [CallStationAgent()] * 3):
        with pytest.raises(ValueError, match="two seats"):
            HandSession(DEFAULT_CFG, DEAL, 0, agents)


def test_finish_is_idempotent_and_notifies_agents_exactly_once() -> None:
    a, b = CountingStation("a"), CountingStation("b")
    trackers = (OpponentTracker(), OpponentTracker())
    session = HandSession(DEFAULT_CFG, DEAL, 0, [a, b], trackers=trackers)
    with pytest.raises(RuntimeError, match="unfinished"):
        session.finish()
    while not session.is_complete:
        session.step()
    first = session.finish()
    assert session.finish() is first and session.finish() is first
    assert a.end_hand_calls == b.end_hand_calls == 1
    assert trackers[0].hands == trackers[1].hands == 1
    assert session.play_out() is first, "play_out on a finished hand just returns the record"


def test_end_hand_is_called_once_per_hand_with_the_right_perspective() -> None:
    a, b = CountingStation("a"), CountingStation("b")
    rng = random.Random(3)
    records = []
    for h in range(25):
        records.append(play_hand(DEFAULT_CFG, random_deal(rng), h % 2, [a, b], hand_number=h))
    assert a.end_hand_calls == b.end_hand_calls == 25
    for h, record in enumerate(records):
        sa, sb = a.summaries[h], b.summaries[h]
        assert sa.hand_number == sb.hand_number == h
        assert sa.my_payoff == record.result.payoffs[0] and sb.my_payoff == record.result.payoffs[1]
        assert sa.was_button == (record.button_seat == 0) and sb.was_button == (record.button_seat == 1)
        assert [x.by_me for x in sa.history] == [not y.by_me for y in sb.history], "mirrored perspectives"


def test_human_seat_gets_no_end_hand_callback_but_trackers_still_update() -> None:
    station = CountingStation()
    session = HandSession(DEFAULT_CFG, DEAL, 0, [None, station])
    session.submit(Action.fold())
    session.finish()
    assert station.end_hand_calls == 1
    assert session.trackers[0].hands == 1 and session.trackers[1].hands == 1


def test_collect_explanations_flag() -> None:
    def run(flag: bool, agents) -> HandRecord:
        session = HandSession(DEFAULT_CFG, DEAL, 0, agents, collect_explanations=flag)
        while not session.is_complete:
            if session.agents[session.actor_seat] is None:  # type: ignore[index]
                session.submit(Action.check_call())
            else:
                session.step()
        return session.finish()

    off = run(False, [ExplainingAgent("e"), ExplainingAgent("e")])
    assert all(d.explanation is None for d in off.decisions) and len(off.decisions) == len(off.events)
    on = run(True, [ExplainingAgent("e"), ExplainingAgent("e")])
    for d in on.decisions:
        assert d.explanation is not None and d.explanation["summary"] == "because"
        assert set(d.explanation["details"]) == {"street", "legal_raise"}
    # an agent that declines to explain, and a human seat, produce no explanation even when requested
    quiet = run(True, [SilentAgent("s"), SilentAgent("s")])
    assert all(d.explanation is None for d in quiet.decisions)
    mixed = run(True, [None, ExplainingAgent("e")])
    by_seat = {e.seat: d for e, d in zip(mixed.events, mixed.decisions)}
    assert by_seat[0].explanation is None and by_seat[1].explanation is not None


def test_explanation_defaults_to_the_decisions_own_explanation() -> None:
    """Baselines attach an Explanation to the Decision; with the flag on it ends up in the record (JSON-able)."""
    record = play_hand(DEFAULT_CFG, DEAL, 0, [ManiacAgent("m"), RandomAgent("r")], collect_explanations=True)
    assert all(isinstance(d.explanation, dict) and d.explanation["summary"] for d in record.decisions)
    plain = play_hand(DEFAULT_CFG, DEAL, 0, [ManiacAgent("m"), RandomAgent("r")])
    assert all(d.explanation is None for d in plain.decisions)


def test_observations_carry_the_trackers_current_statistics() -> None:
    trackers = (OpponentTracker(), OpponentTracker())
    a, b = CountingStation("a"), CountingStation("b")
    play_hand(DEFAULT_CFG, DEAL, 0, [a, b], trackers=trackers, hand_number=0)
    assert a.observations[0].opponent_stats == OpponentStats()  # nothing known before the first hand
    after_first_hand = trackers[0].stats()
    assert after_first_hand.hands == 1
    play_hand(DEFAULT_CFG, Deal.from_seed(5), 1, [a, b], trackers=trackers, hand_number=1)
    first_obs_hand2 = next(o for o in a.observations if o.hand_number == 1)
    assert first_obs_hand2.opponent_stats == after_first_hand, "stats come from PREVIOUS hands only"
    session = HandSession(DEFAULT_CFG, DEAL, 0, [a, b], trackers=trackers, hand_number=2)
    assert session.observation(0).opponent_stats == trackers[0].stats()
    assert session.observation(1).opponent_stats == trackers[1].stats()
    assert session.observation(0).opponent_stats.hands == 2


def test_play_hand_without_trackers_starts_blank_each_time() -> None:
    a, b = CountingStation("a"), CountingStation("b")
    play_hand(DEFAULT_CFG, DEAL, 0, [a, b])
    play_hand(DEFAULT_CFG, DEAL, 0, [a, b])
    assert all(o.opponent_stats.hands == 0 for o in a.observations + b.observations)


def test_hand_record_from_session_matches_engine() -> None:
    session = HandSession(DEFAULT_CFG, DEAL, 1, [ManiacAgent("m"), CallStationAgent("c")], start_stacks=(300, 500), hand_number=7)
    record = session.play_out()
    assert record.hand_number == 7 and record.start_stacks == (300, 500) and record.button_seat == 1
    assert record.agent_names == ("m", "c") and record.deal == DEAL
    assert record.events == tuple(session.engine.events) and record.result == session.engine.result()
    assert len(record.decisions) == len(record.events) and record.verify()
    assert sum(record.result.final_stacks) == 800


# ============================================================================ OpponentTracker
def pa(street: Street, by_me: bool, kind: str, chips: int = 10) -> PublicAction:
    return PublicAction(street, by_me, kind, chips, chips, False, 100)


P, F, T, R = Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER


def summary(history: list[PublicAction], showdown: bool = False, hand: int = 0) -> PublicHandSummary:
    return PublicHandSummary(hand, True, tuple(history), (), 0, showdown, None)


# the four hand-crafted hands whose statistics are worked out in the comments of the test below
HAND_1 = [  # opp raises preflop, bets flop, folds to my turn bet
    pa(P, False, "raise"), pa(P, True, "call"),
    pa(F, False, "bet"), pa(F, True, "call"),
    pa(T, False, "check"), pa(T, True, "bet"), pa(T, False, "fold"),
]
HAND_2 = [  # opp limps, calls my flop bet, bets river; showdown
    pa(P, False, "call"), pa(P, True, "check"),
    pa(F, False, "check"), pa(F, True, "bet"), pa(F, False, "call"),
    pa(T, False, "check"), pa(T, True, "check"),
    pa(R, False, "bet"), pa(R, True, "call"),
]
HAND_3 = [pa(P, True, "raise"), pa(P, False, "fold")]  # opp folds preflop to my raise
HAND_4 = [  # raising war preflop, opp bets flop and folds to my raise
    pa(P, False, "raise"), pa(P, True, "raise"), pa(P, False, "raise"), pa(P, True, "call"),
    pa(F, False, "bet"), pa(F, True, "raise"), pa(F, False, "fold"),
]


def test_tracker_starts_at_the_prior() -> None:
    stats = OpponentTracker().stats()
    prior = OpponentStats()
    assert stats.hands == 0
    for name in ("vpip", "pfr", "aggression", "fold_to_bet", "showdown_rate"):
        assert getattr(stats, name) == pytest.approx(getattr(prior, name))


def test_tracker_exact_statistics_on_handcrafted_histories() -> None:
    t = OpponentTracker()
    t.update(summary(HAND_1))
    t.update(summary(HAND_2, showdown=True))
    t.update(summary(HAND_3))
    t.update(summary(HAND_4))
    # raw counters: vpip hands = 1,2,4 ; pfr hands = 1,4 ; opp postflop bets/raises = flop(h1), river(h2), flop(h4) ;
    # opp postflop calls = flop(h2) ; times opp faced my bet/raise = h1 turn, h2 flop, h3 preflop, h4 preflop+flop ;
    # folded = h1, h3, h4(flop)
    assert (t.hands, t.vpip_hands, t.pfr_hands) == (4, 3, 2)
    assert (t.postflop_aggressive, t.postflop_calls) == (3, 1)
    assert (t.faced_aggression, t.folded_to_aggression, t.showdowns) == (5, 3, 1)
    s = t.stats()
    w = SMOOTHING_WEIGHT
    assert w == 5.0
    assert s.hands == 4
    assert s.vpip == pytest.approx((3 + 0.5 * w) / (4 + w))  # 5.5 / 9
    assert s.pfr == pytest.approx((2 + 0.3 * w) / (4 + w))  # 3.5 / 9
    assert s.aggression == pytest.approx((3 + 0.4 * w) / (4 + w))  # 5 / 9
    assert s.fold_to_bet == pytest.approx((3 + 0.4 * w) / (5 + w))  # 5 / 10
    assert s.showdown_rate == pytest.approx((1 + 0.3 * w) / (4 + w))  # 2.5 / 9
    assert s.vpip == pytest.approx(0.61111, abs=1e-5) and s.fold_to_bet == pytest.approx(0.5)


def test_tracker_ignores_my_own_actions_and_counts_voluntary_money_only() -> None:
    t = OpponentTracker()
    t.update(summary([pa(P, True, "raise"), pa(P, False, "call"), pa(F, False, "check"), pa(F, True, "check")]))
    assert (t.vpip_hands, t.pfr_hands) == (1, 0), "a call is VPIP but not PFR"
    t.update(summary([pa(P, False, "call"), pa(P, True, "check")]))  # SB limp
    assert t.vpip_hands == 2
    t.update(summary([pa(P, True, "call"), pa(P, False, "check")]))  # BB checks its option: not voluntary
    assert (t.vpip_hands, t.pfr_hands) == (2, 0)
    t.update(summary([pa(P, False, "fold")]))  # SB folds to the blind: no bet of mine preceded it
    assert t.faced_aggression == 1 and t.folded_to_aggression == 0 and t.vpip_hands == 2
    t.update(summary([pa(P, True, "raise"), pa(P, False, "raise"), pa(P, True, "call")]))  # opp 3-bets
    assert (t.faced_aggression, t.folded_to_aggression) == (2, 0)


def test_fold_to_bet_logic_edge_cases() -> None:
    t = OpponentTracker()
    # opponent folding with nothing of mine before it (first action) is not "folding to a bet"
    t.update(summary([pa(P, False, "fold")]))
    assert (t.faced_aggression, t.folded_to_aggression) == (0, 0)
    # I call (not aggression), opponent folds later: not counted
    t.update(summary([pa(P, True, "call"), pa(P, False, "fold")]))
    assert (t.faced_aggression, t.folded_to_aggression) == (0, 0)
    # I bet, opponent re-raises: faced, not folded
    t.update(summary([pa(F, True, "bet"), pa(F, False, "raise"), pa(F, True, "call")]))
    assert (t.faced_aggression, t.folded_to_aggression) == (1, 0)
    # I raise, opponent folds: faced and folded
    t.update(summary([pa(F, True, "raise"), pa(F, False, "fold")]))
    assert (t.faced_aggression, t.folded_to_aggression) == (2, 1)
    # two separate bets of mine on different streets, folded only to the second
    t.update(summary([pa(F, True, "bet"), pa(F, False, "call"), pa(T, False, "check"), pa(T, True, "bet"), pa(T, False, "fold")]))
    assert (t.faced_aggression, t.folded_to_aggression) == (4, 2)
    assert t.stats().fold_to_bet == pytest.approx((2 + 0.4 * 5) / (4 + 5))


def test_aggression_is_postflop_only() -> None:
    t = OpponentTracker()
    t.update(summary([pa(P, False, "raise"), pa(P, True, "call"), pa(F, False, "check"), pa(F, True, "check")]))
    assert (t.postflop_aggressive, t.postflop_calls) == (0, 0), "preflop raises and postflop checks do not count"
    assert t.stats().aggression == pytest.approx(OpponentStats().aggression)
    t.update(summary([pa(F, True, "bet"), pa(F, False, "raise"), pa(F, True, "raise"), pa(F, False, "call")]))
    assert (t.postflop_aggressive, t.postflop_calls) == (1, 1)


def test_reset_returns_to_a_blank_tracker() -> None:
    t = OpponentTracker()
    for h in (HAND_1, HAND_2, HAND_4):
        t.update(summary(h, showdown=True))
    t.reset()
    assert t == OpponentTracker()
    assert t.stats() == OpponentTracker().stats()


def test_smoothing_dominates_early_and_vanishes_late() -> None:
    always_raise = [pa(P, False, "raise"), pa(P, True, "fold")]
    t = OpponentTracker()
    t.update(summary(always_raise))
    one = t.stats().pfr
    assert one == pytest.approx((1 + 0.3 * 5) / 6) and 0.3 < one < 0.5
    for _ in range(999):
        t.update(summary(always_raise))
    assert t.stats().pfr == pytest.approx(1.0, abs=0.005)
    assert 0.0 <= t.stats().vpip <= 1.0


def test_tracker_counters_are_plain_integers() -> None:
    """The tracker stores counts only - structurally unable to hold card information."""
    t = OpponentTracker()
    t.update(summary(HAND_1))
    assert all(type(getattr(t, f.name)) is int for f in dataclasses.fields(t))
    assert {f.name for f in dataclasses.fields(PublicHandSummary)} == {
        "hand_number", "was_button", "history", "board", "my_payoff", "showdown", "opp_hole",
    }


def test_trackers_never_depend_on_hidden_cards() -> None:
    """Same public actions, different hidden cards, no showdown => identical tracker statistics."""
    rng = random.Random(derive_seed("test", "tracker-hidden"))
    scripts = [
        [30, "f"],
        ["c", "c", 20, "f"],
        ["c", "c", "c", "c", 15, 45, "f"],
        [25, 80, "f"],
    ]
    for script in scripts:
        results = []
        for _ in range(2):
            trackers = (OpponentTracker(), OpponentTracker())
            session = HandSession(DEFAULT_CFG, random_deal(rng), 0, [None, None], trackers=trackers)
            for token in script:
                session.submit(Action.fold() if token == "f" else Action.check_call() if token == "c" else Action.raise_to(token))
            session.finish()
            results.append((trackers[0], trackers[1]))
        assert results[0] == results[1]


def test_session_trackers_match_an_independent_recount_from_raw_events() -> None:
    """300 random hands: the session's trackers agree with counts recomputed straight from the event log."""
    rng = random.Random(derive_seed("test", "tracker-recount"))
    agents = [RandomAgent("a"), RandomAgent("b")]
    agents[0].reset(1), agents[1].reset(2)
    trackers = (OpponentTracker(), OpponentTracker())
    records = []
    for h in range(300):
        records.append(
            play_hand(DEFAULT_CFG, random_deal(rng), h % 2, agents, start_stacks=random_stacks(rng), hand_number=h, trackers=trackers)
        )
    for observer in (0, 1):
        opp = 1 - observer
        exp = dict(vpip=0, pfr=0, agg=0, calls=0, faced=0, folded=0, showdowns=0)
        for r in records:
            pre = [e for e in r.events if e.seat == opp and e.street == Street.PREFLOP]
            exp["vpip"] += any(e.kind in ("call", "bet", "raise") for e in pre)
            exp["pfr"] += any(e.kind in ("bet", "raise") for e in pre)
            for e in r.events:
                if e.seat == opp and e.street != Street.PREFLOP:
                    exp["agg"] += e.kind in ("bet", "raise")
                    exp["calls"] += e.kind == "call"
            for prev, cur in zip(r.events, r.events[1:]):
                if prev.seat == observer and prev.kind in ("bet", "raise") and cur.seat == opp:
                    exp["faced"] += 1
                    exp["folded"] += cur.kind == "fold"
                    assert cur.seat != prev.seat
            exp["showdowns"] += r.result.showdown
        t = trackers[observer]
        assert (t.hands, t.vpip_hands, t.pfr_hands) == (300, exp["vpip"], exp["pfr"])
        assert (t.postflop_aggressive, t.postflop_calls) == (exp["agg"], exp["calls"])
        assert (t.faced_aggression, t.folded_to_aggression, t.showdowns) == (exp["faced"], exp["folded"], exp["showdowns"])
        assert exp["faced"] > 50 and exp["folded"] > 5
