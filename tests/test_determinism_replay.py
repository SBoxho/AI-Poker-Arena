"""Reproducibility: seeds -> identical hands, JSON round trips, exact replay, derive_seed namespaces."""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from poker_arena.agents.baselines import CallStationAgent, ManiacAgent, RandomAgent
from poker_arena.config import SEED_NAMESPACES, GameConfig, derive_seed
from poker_arena.engine.actions import Action
from poker_arena.engine.deals import Deal
from poker_arena.engine.game import HandResult
from poker_arena.engine.history import DecisionRecord, HandRecord
from poker_arena.engine.observation import Street
from poker_arena.engine.session import play_hand

from core_helpers import DEFAULT_CFG, HoleCardAgent, random_deal, random_stacks, run_match

N_HANDS = 200


def make_agents():
    return [RandomAgent("rand"), ManiacAgent("maniac", raise_probability=0.5)]


def strip_latency(record: HandRecord) -> dict:
    d = record.to_dict()
    for dec in d["decisions"]:
        dec["latency_ms"] = 0.0
    return d


@pytest.fixture(scope="module")
def match_a() -> list[HandRecord]:
    return run_match(make_agents(), N_HANDS, seed=123)


# ============================================================================= determinism
def test_same_seed_gives_bit_identical_records(match_a: list[HandRecord]) -> None:
    again = run_match(make_agents(), N_HANDS, seed=123)
    assert len(again) == len(match_a) == N_HANDS
    for a, b in zip(match_a, again):
        assert a == b, "dataclass equality (wall-clock latency is excluded)"
        assert json.dumps(strip_latency(a), sort_keys=True) == json.dumps(strip_latency(b), sort_keys=True)
    # the very same agent objects, reset again, reproduce the match too
    agents = make_agents()
    first = run_match(agents, 40, seed=123)
    second = run_match(agents, 40, seed=123)
    assert first == second == match_a[:40]


def test_different_seed_gives_different_hands(match_a: list[HandRecord]) -> None:
    other = run_match(make_agents(), N_HANDS, seed=124)
    differing_deals = sum(a.deal != b.deal for a, b in zip(match_a, other))
    differing_events = sum(a.events != b.events for a, b in zip(match_a, other))
    assert differing_deals == N_HANDS
    assert differing_events > N_HANDS * 0.8


def test_agent_seed_changes_play_but_not_the_deals() -> None:
    first = run_match([RandomAgent("a"), RandomAgent("b")], 60, seed=5)
    # same deals (same match seed) but the agents' private RNGs are re-seeded differently
    agents_b = [RandomAgent("a"), RandomAgent("b")]
    for i, ag in enumerate(agents_b):
        ag.reset(derive_seed("test", "other", i))
    second = []
    for h in range(60):
        deal = Deal.from_seed(derive_seed("test", 5, "deal", h))
        second.append(play_hand(DEFAULT_CFG, deal, h % 2, agents_b, hand_number=h))
    assert [r.deal for r in first] == [r.deal for r in second]
    assert [r.events for r in first] != [r.events for r in second]


def test_latency_is_recorded_but_not_part_of_record_identity(match_a: list[HandRecord]) -> None:
    record = match_a[0]
    assert all(d.latency_ms >= 0 for d in record.decisions) and len(record.decisions) == len(record.events)
    shifted = dataclasses.replace(
        record, decisions=tuple(dataclasses.replace(d, latency_ms=d.latency_ms + 123.0) for d in record.decisions)
    )
    assert shifted == record
    assert shifted.to_dict() != record.to_dict()  # ...but the number is still stored


def test_match_exercises_all_outcome_types(match_a: list[HandRecord]) -> None:
    showdowns = sum(r.result.showdown for r in match_a)
    all_ins = sum(any(e.all_in for e in r.events) for r in match_a)
    assert showdowns > 10 and N_HANDS - showdowns > 10 and all_ins > 10
    assert {r.button_seat for r in match_a} == {0, 1}
    assert {r.result.ended_street for r in match_a} >= {Street.PREFLOP, Street.RIVER}
    assert all(sum(r.result.payoffs) == 0 for r in match_a)


# ============================================================================== serialisation
def test_json_round_trip_of_every_hand(match_a: list[HandRecord]) -> None:
    for record in match_a:
        text = json.dumps(record.to_dict())
        restored = HandRecord.from_dict(json.loads(text))
        assert restored == record
        assert restored.to_dict() == record.to_dict()
        assert isinstance(restored.start_stacks, tuple) and isinstance(restored.result, HandResult)
        assert restored.result.ended_street is record.result.ended_street
        assert all(isinstance(k, int) for k in restored.result.shown_hands)


def test_json_round_trip_with_explanations() -> None:
    agents = [RandomAgent("a"), ManiacAgent("b")]
    deal = Deal.from_seed(9)
    record = play_hand(DEFAULT_CFG, deal, 1, agents, collect_explanations=True)
    assert all(d.explanation and d.explanation["summary"] for d in record.decisions)
    restored = HandRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert restored == record and restored.decisions[0].explanation == record.decisions[0].explanation
    assert restored.agent_names == ("a", "b")


def test_from_dict_tolerates_minimal_legacy_payload(match_a: list[HandRecord]) -> None:
    d = match_a[1].to_dict()
    del d["agent_names"], d["decisions"]
    restored = HandRecord.from_dict(d)
    assert restored.agent_names == ("seat0", "seat1") and restored.decisions == ()
    assert restored.verify()


def test_asymmetric_and_short_stack_records_round_trip_and_verify() -> None:
    rng = random.Random(31)
    agents = [RandomAgent("a"), ManiacAgent("b", raise_probability=0.5)]
    for i, ag in enumerate(agents):
        ag.reset(50 + i)
    all_in_pre = 0
    for h in range(150):
        stacks = random_stacks(rng)
        record = play_hand(DEFAULT_CFG, random_deal(rng), h % 2, agents, start_stacks=stacks, hand_number=h)
        assert record.start_stacks == stacks and record.verify()
        restored = HandRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored == record and restored.verify()
        assert sum(record.result.final_stacks) == sum(stacks)
        all_in_pre += any(e.all_in for e in record.events)
    assert all_in_pre > 30


def test_game_config_and_action_dict_round_trip() -> None:
    cfg = GameConfig(500, 2, 5)
    assert GameConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    for a in (Action.fold(), Action.check_call(), Action.raise_to(37)):
        assert Action.from_dict(json.loads(json.dumps(a.to_dict()))) == a


# ================================================================================== replay
def test_verify_reproduces_every_result(match_a: list[HandRecord]) -> None:
    assert all(r.verify() for r in match_a)


def test_verify_detects_tampering(match_a: list[HandRecord]) -> None:
    record = next(r for r in match_a if r.result.showdown and len(r.events) >= 2)
    bad_payoffs = dataclasses.replace(
        record, result=dataclasses.replace(record.result, payoffs=(record.result.payoffs[0] + 1, record.result.payoffs[1] - 1))
    )
    assert not bad_payoffs.verify()
    swapped = Deal(tuple(reversed(record.deal.cards)))
    other_cards = dataclasses.replace(record, deal=swapped)
    assert not other_cards.verify(), "a different deal cannot reproduce the recorded showdown"
    # replaying with an impossible action fails loudly instead of silently "verifying"
    broken = dataclasses.replace(record, events=(dataclasses.replace(record.events[0], action=Action.raise_to(1)),))
    with pytest.raises(ValueError):
        broken.verify()


def test_replay_views_match_the_record(match_a: list[HandRecord]) -> None:
    for record in match_a[:60]:
        views = list(record.replay())
        assert len(views) == len(record.events) + 1
        deal = record.deal
        button = record.button_seat
        expected_holes = [None, None]
        expected_holes[button], expected_holes[1 - button] = deal.button_hole, deal.bb_hole
        assert views[0].actor_seat in (button, None)
        for i, view in enumerate(views):
            assert view.hole_cards == tuple(expected_holes), "hole cards by seat are constant"
            assert view.button_seat == button
            assert view.events == record.events[:i]
            assert view.board == deal.board[: len(view.board)] and len(view.board) in (0, 3, 4, 5)
            assert view.complete == (i == len(views) - 1)
        assert views[-1].complete and views[-1].result == record.result
        boards = [len(v.board) for v in views]
        assert boards == sorted(boards), "board only grows"
        if not views[0].complete:  # right after the blinds, chips are in the bets
            assert views[0].stacks == (
                record.start_stacks[0] - views[0].bets[0],
                record.start_stacks[1] - views[0].bets[1],
            )


def test_replay_is_a_pure_function_of_the_record(match_a: list[HandRecord]) -> None:
    record = match_a[7]
    assert list(record.replay()) == list(record.replay())


def test_public_summary_matches_record(match_a: list[HandRecord]) -> None:
    for record in match_a[:80]:
        for seat in (0, 1):
            s = record.public_summary(seat)
            assert s.my_payoff == record.result.payoffs[seat] and s.was_button == (seat == record.button_seat)
            assert s.showdown == record.result.showdown and s.board == record.result.board
            assert [a.by_me for a in s.history] == [e.seat == seat for e in record.events]
            if s.showdown:
                assert s.opp_hole == (record.deal.bb_hole if seat == record.button_seat else record.deal.button_hole)
            else:
                assert s.opp_hole is None


# ============================================================================ duplicate deals
def test_duplicate_deals_cancel_card_luck_for_identical_agents() -> None:
    """Same Deal played twice with the two players swapped between button and BB: identical agents sum to 0."""
    rng = random.Random(4)
    nonzero_hands = 0
    for i in range(120):
        deal = random_deal(rng)
        totals = [0, 0]
        for button in (0, 1):
            if i % 2:
                agents = [HoleCardAgent("x"), HoleCardAgent("y")]
            else:
                agents = [ManiacAgent("x", raise_probability=0.6), ManiacAgent("y", raise_probability=0.6)]
            for ag in agents:
                ag.reset(77)
            record = play_hand(DEFAULT_CFG, deal, button, agents)
            totals[0] += record.result.payoffs[0]
            totals[1] += record.result.payoffs[1]
            nonzero_hands += record.result.payoffs != (0, 0)
        assert totals == [0, 0], f"deal {i}: duplicate pair must net to zero for identical agents"
    assert nonzero_hands > 100, "the pairs must contain real wins and losses"


def test_duplicate_deals_do_not_cancel_for_different_agents() -> None:
    """Sanity check of the test above: with different agents the sum is generally non-zero."""
    rng = random.Random(4)
    nonzero = 0
    for _ in range(60):
        deal = random_deal(rng)
        total = 0
        for button in (0, 1):
            agents = [HoleCardAgent("x"), CallStationAgent("y")]  # pot sizes now depend on the cards
            total += play_hand(DEFAULT_CFG, deal, button, agents).result.payoffs[0]
        nonzero += total != 0
    assert nonzero > 5


# ================================================================================ derive_seed
def test_derive_seed_is_deterministic_and_in_range() -> None:
    assert derive_seed("test", 1, 2) == derive_seed("test", 1, 2)
    for ns in SEED_NAMESPACES:
        s = derive_seed(ns, 42, "x")
        assert isinstance(s, int) and 0 <= s < 2**63
        random.Random(s)  # usable as an RNG seed


def test_derive_seed_golden_values_are_stable_across_machines() -> None:
    """sha256-based: these literals must never change, or every saved experiment becomes unreproducible."""
    assert derive_seed("test", 1, 2) == 5372671034286678118
    assert derive_seed("eval", 42, 0, 7) == 9027846690909363597
    assert derive_seed("dataset_train", 0) == 1345755324102970118


def test_derive_seed_golden_deals() -> None:
    """``random.sample`` with an int seed is stable across supported Pythons; a change here means seeds no longer reproduce."""
    assert Deal.from_seed(42).cards == ("Qc", "3s", "2d", "Ks", "6d", "5s", "5h", "4c", "3h", "Th", "3d", "Jd")


def test_derive_seed_namespaces_are_disjoint_in_practice() -> None:
    seen: dict[int, tuple] = {}
    for ns in SEED_NAMESPACES:
        for a in range(40):
            for b in range(25):
                s = derive_seed(ns, a, b)
                assert s not in seen, f"collision {ns, a, b} vs {seen[s]}"
                seen[s] = (ns, a, b)
    assert len(seen) == len(SEED_NAMESPACES) * 1000


def test_derive_seed_depends_on_every_part_and_their_order() -> None:
    base = derive_seed("eval", 1, 2, 3)
    assert base != derive_seed("eval", 1, 2, 4)
    assert base != derive_seed("eval", 3, 2, 1)
    assert base != derive_seed("eval", 1, 2)
    assert base != derive_seed("eval", 1, 2, 3, 0)
    assert derive_seed("eval") != derive_seed("ui")


def test_derive_seed_rejects_unknown_namespace() -> None:
    with pytest.raises(ValueError, match="Unknown seed namespace"):
        derive_seed("training", 1)
    with pytest.raises(ValueError):
        derive_seed("", 1)
    with pytest.raises(ValueError):
        derive_seed("TEST", 1)  # case sensitive


def test_decision_record_dict_round_trip() -> None:
    d = DecisionRecord(1.5, "ALL_IN", {"summary": "x", "details": {"a": [1, 2]}})
    assert d.to_dict() == {"latency_ms": 1.5, "abstract": "ALL_IN", "explanation": {"summary": "x", "details": {"a": [1, 2]}}}
