"""Tests for ``poker_arena.tournament.match``: accounting, duplicate dealing, determinism, sampling."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from poker_arena.agents.base import ModelArtifactError
from poker_arena.agents.registry import AgentSpec, create_agent, known_kinds
from poker_arena.config import GameConfig, derive_seed
from poker_arena.engine.deals import Deal
from poker_arena.engine.history import HandRecord
from poker_arena.tournament.config import KeepHandsPolicy, MatchConfig, resolve_n_workers
from poker_arena.tournament.match import (
    TAG_BIGGEST_POT,
    TAG_BIGGEST_WIN_A,
    TAG_BIGGEST_WIN_B,
    TAG_FIRST,
    AgentCache,
    MatchResult,
    SessionTask,
    agent_seed,
    bootstrap_seed,
    deal_seed,
    labelled,
    match_key,
    play_session,
    run_match,
)

CALL_STATION = AgentSpec("call_station", {}, "CallStation")
MANIAC = AgentSpec("maniac", {}, "Maniac")
RANDOM = AgentSpec("random", {}, "Random")
# Deterministic stand-ins (a maniac that always raises needs no RNG), so identical copies must mirror exactly.
ALWAYS_RAISE_1 = AgentSpec("maniac", {"raise_probability": 1.0}, "Raiser1")
ALWAYS_RAISE_2 = AgentSpec("maniac", {"raise_probability": 1.0}, "Raiser2")
BASIC_COLUMNS = ("profit_chips", "a_on_button", "showdown", "ended_street", "pot_chips", "action_counts",
                 "three_bet", "three_bet_opportunity")  # fmt: skip


def small_cfg(**changes) -> MatchConfig:
    fields = dict(num_hands=120, session_hands=40, seed=11, keep_hands=KeepHandsPolicy(all_hands=True))
    fields.update(changes)
    return MatchConfig(**fields)


def records_of(result: MatchResult) -> list[HandRecord]:
    assert result.all_hands is not None
    return [HandRecord.from_dict(d) for d in result.all_hands]


def assert_same_outcome(a: MatchResult, b: MatchResult) -> None:
    """Everything except wall-clock latencies must match exactly."""
    for name in BASIC_COLUMNS:
        assert np.array_equal(getattr(a.arrays, name), getattr(b.arrays, name)), name
    assert len(a.arrays.latency_a_ms) == len(b.arrays.latency_a_ms)
    assert [(s.hand_index, s.tags, s.pot_chips, s.profit_chips) for s in a.samples] == [
        (s.hand_index, s.tags, s.pot_chips, s.profit_chips) for s in b.samples
    ]
    for sa, sb in zip(a.samples, b.samples):
        assert sa.record["events"] == sb.record["events"] and sa.record["deal"] == sb.record["deal"]


# ----------------------------------------------------------------------------- accounting
@pytest.fixture(scope="module")
def call_station_vs_maniac() -> MatchResult:
    return run_match(CALL_STATION, MANIAC, small_cfg())


def test_accounting_per_hand(call_station_vs_maniac: MatchResult) -> None:
    result, arrays = call_station_vs_maniac, call_station_vs_maniac.arrays
    records = records_of(result)
    assert len(records) == len(arrays) == 120
    stack = result.cfg.game.starting_stack
    for h, rec in enumerate(records):
        a_profit, b_profit = rec.result.payoffs
        assert a_profit == -b_profit == arrays.profit_chips[h]  # zero-sum, A is seat 0
        assert sum(rec.result.final_stacks) == 2 * stack
        assert rec.result.final_stacks[0] - stack == a_profit
        assert rec.result.total_pot == arrays.pot_chips[h]
        assert rec.result.total_pot >= 2 * abs(a_profit)
        # two tables: at table (h % 2) A is on the button when deal + table is even; the deal is h // 2
        assert arrays.a_on_button[h] == ((h // 2 + h % 2) % 2 == 0)
        assert rec.button_seat == (0 if arrays.a_on_button[h] else 1)
        assert rec.result.showdown == arrays.showdown[h] and int(rec.result.ended_street) == arrays.ended_street[h]
        assert rec.agent_names == ("CallStation", "Maniac")
        assert rec.verify()


def test_totals_and_unit_profits(call_station_vs_maniac: MatchResult) -> None:
    result, arrays = call_station_vs_maniac, call_station_vs_maniac.arrays
    bb = result.cfg.game.big_blind
    pairs = result.unit_profits_bb
    assert len(pairs) == 60
    assert pairs.sum() == pytest.approx(arrays.profit_chips.sum() / bb)
    assert np.allclose(pairs, (arrays.profit_chips[0::2] + arrays.profit_chips[1::2]) / bb)
    assert result.cfg.num_units == 60 and result.cfg.hands_per_unit == 2


def test_action_counts_agree_with_the_event_logs(call_station_vs_maniac: MatchResult) -> None:
    arrays = call_station_vs_maniac.arrays
    for h, rec in enumerate(records_of(call_station_vs_maniac)):
        expected = np.zeros_like(arrays.action_counts[h])
        kinds = ("fold", "check", "call", "bet", "raise")
        for ev in rec.events:
            expected[ev.seat, int(ev.street), kinds.index(ev.kind)] += 1
        assert np.array_equal(arrays.action_counts[h], expected), h
    # call station never folds or raises; the maniac never folds either (nothing to fold to a call station)
    assert arrays.action_counts[:, 0, :, 0].sum() == 0
    assert arrays.action_counts[:, 0, :, 3:].sum() == 0


def test_latencies_are_recorded_per_decision(call_station_vs_maniac: MatchResult) -> None:
    arrays = call_station_vs_maniac.arrays
    decisions = arrays.action_counts.sum(axis=(0, 2, 3))
    assert len(arrays.latency_a_ms) == decisions[0] and len(arrays.latency_b_ms) == decisions[1]
    assert np.all(arrays.latency_a_ms >= 0) and np.all(np.isfinite(arrays.latency_b_ms))


def test_hand_numbers_restart_in_every_session(call_station_vs_maniac: MatchResult) -> None:
    """Two tables of 20 hands per session: both plays of deal p are hand p of their own table."""
    numbers = [rec.hand_number for rec in records_of(call_station_vs_maniac)]
    assert numbers == [p for p in range(20) for _ in range(2)] * 3
    single = run_match(CALL_STATION, MANIAC, small_cfg(protocol="single_table"))
    assert [rec.hand_number for rec in records_of(single)] == list(range(40)) * 3  # one table of 40 hands


# ----------------------------------------------------------------------------- duplicate dealing
def test_pairs_share_the_deal_and_swap_the_button(call_station_vs_maniac: MatchResult) -> None:
    records = records_of(call_station_vs_maniac)
    for i in range(0, len(records), 2):
        first, second = records[i], records[i + 1]
        assert first.deal == second.deal
        assert {first.button_seat, second.button_seat} == {0, 1}  # the seats are swapped within the pair
        assert first.button_seat == (i // 2) % 2  # two tables: A opens on the button at even deals only
    deals = [records[i].deal for i in range(0, len(records), 2)]
    assert len({d.cards for d in deals}) == len(deals)  # a new deal for every pair


@pytest.mark.parametrize(
    ("spec_a", "spec_b"),
    [
        (CALL_STATION, CALL_STATION),  # identical labels: a true mirror
        (CALL_STATION, AgentSpec("call_station", {}, "Other")),
        (ALWAYS_RAISE_1, ALWAYS_RAISE_2),
    ],
)
@pytest.mark.parametrize("protocol", ["two_table", "single_table"])
def test_identical_deterministic_agents_cancel_exactly_in_every_pair(
    spec_a: AgentSpec, spec_b: AgentSpec, protocol: str
) -> None:
    """Stateless agents cancel exactly under either protocol (only adaptive agents tell them apart)."""
    result = run_match(spec_a, spec_b, small_cfg(keep_hands=KeepHandsPolicy(), protocol=protocol))
    profits = result.arrays.profit_chips
    assert np.any(profits != 0)  # hands are decided, but...
    assert np.all(profits[0::2] + profits[1::2] == 0)  # ...luck cancels exactly
    assert np.all(result.unit_profits_bb == 0)


def test_different_agents_do_not_cancel() -> None:
    result = run_match(CALL_STATION, MANIAC, small_cfg(keep_hands=KeepHandsPolicy()))
    assert np.any(result.unit_profits_bb != 0)


def test_swapping_a_and_b_mirrors_the_result_for_deterministic_agents() -> None:
    """Same match key => same deals; with stateless agents only the roles flip."""
    cfg = small_cfg(keep_hands=KeepHandsPolicy())
    forward = run_match(CALL_STATION, ALWAYS_RAISE_1, cfg)
    backward = run_match(ALWAYS_RAISE_1, CALL_STATION, cfg)
    assert forward.key == backward.key
    assert np.array_equal(forward.unit_profits_bb, -backward.unit_profits_bb)


def test_without_duplicate_each_hand_has_its_own_deal() -> None:
    result = run_match(CALL_STATION, MANIAC, small_cfg(duplicate=False, num_hands=40, session_hands=20))
    records = records_of(result)
    assert len({r.deal.cards for r in records}) == 40
    assert [r.button_seat for r in records[:4]] == [0, 1, 0, 1]
    assert result.cfg.hands_per_unit == 1 and np.allclose(result.unit_profits_bb, result.arrays.profit_chips / 10)


# ----------------------------------------------------------------------------- determinism
def test_same_seed_same_result_and_different_seed_different_result() -> None:
    cfg = small_cfg(keep_hands=KeepHandsPolicy(first=3, biggest_pots=2, biggest_wins_each=2))
    first = run_match(RANDOM, MANIAC, cfg)
    assert_same_outcome(first, run_match(RANDOM, MANIAC, cfg))
    other = run_match(RANDOM, MANIAC, small_cfg(seed=12, keep_hands=cfg.keep_hands))
    assert not np.array_equal(first.arrays.profit_chips, other.arrays.profit_chips)


@pytest.mark.parametrize("protocol", ["two_table", "single_table"])
def test_result_does_not_depend_on_the_number_of_workers(protocol: str) -> None:
    cfg = small_cfg(num_hands=160, session_hands=20, protocol=protocol,
                    keep_hands=KeepHandsPolicy(first=3, biggest_pots=3, biggest_wins_each=2))  # fmt: skip
    single = run_match(RANDOM, MANIAC, cfg)
    parallel = run_match(RANDOM, MANIAC, replace(cfg, n_workers=3))
    assert_same_outcome(single, parallel)


def test_progress_callback_reports_every_session() -> None:
    seen = []
    run_match(CALL_STATION, MANIAC, small_cfg(keep_hands=KeepHandsPolicy()), progress=seen.append)
    assert [p.hands_done for p in seen] == [40, 80, 120]
    assert seen[-1].fraction == 1.0 and seen[-1].matches_done == 1 and seen[-1].last_match == "CallStation vs Maniac"


def test_agents_are_reseeded_per_session_so_sessions_are_independent_blocks() -> None:
    """A session played alone equals the same session inside the full match."""
    cfg = small_cfg(keep_hands=KeepHandsPolicy())
    full = run_match(RANDOM, MANIAC, cfg)
    task = SessionTask(0, 1, labelled(RANDOM), labelled(MANIAC), cfg, first_hand=40, num_hands=40)
    alone = play_session(task, AgentCache())
    assert np.array_equal(alone.arrays.profit_chips, full.arrays.profit_chips[40:80])


KINDS = known_kinds()


@pytest.mark.parametrize("kind", KINDS)
def test_registry_agents_obey_the_copy_and_reset_contract(kind: str) -> None:
    """Every available agent kind: a copy of a once-built prototype plays exactly like a fresh agent."""
    spec = labelled(AgentSpec(kind, {}, f"Under{kind}"))
    try:
        create_agent(spec)
    except ModelArtifactError as exc:
        pytest.skip(f"{kind} is not available: {exc}")
    cfg = small_cfg(num_hands=20, session_hands=20, keep_hands=KeepHandsPolicy())
    task = SessionTask(0, 0, spec, labelled(MANIAC), cfg, 0, 20)
    shared = AgentCache()
    first, second = play_session(task, shared), play_session(task, shared)
    fresh = play_session(task, AgentCache())
    for other in (second, fresh):
        for name in BASIC_COLUMNS:
            assert np.array_equal(getattr(first.arrays, name), getattr(other.arrays, name)), (kind, name)


# ----------------------------------------------------------------------------- 3-bet bookkeeping
def test_three_bet_bookkeeping_matches_an_independent_reading_of_the_event_log() -> None:
    result = run_match(RANDOM, MANIAC, small_cfg(num_hands=200, session_hands=100))
    arrays = result.arrays
    seen_three_bets = 0
    for h, rec in enumerate(records_of(result)):
        pre = [e for e in rec.events if int(e.street) == 0]
        raises = [i for i, e in enumerate(pre) if e.kind in ("bet", "raise")]
        opp, three = [0, 0], [0, 0]
        if raises and raises[0] + 1 < len(pre):
            opp[pre[raises[0] + 1].seat] = 1  # the player facing the first raise may 3-bet
        if len(raises) >= 2:
            three[pre[raises[1]].seat] = 1
            seen_three_bets += 1
        assert arrays.three_bet_opportunity[h].tolist() == opp, h
        assert arrays.three_bet[h].tolist() == three, h
    assert seen_three_bets > 5  # the scenario really occurs


# ----------------------------------------------------------------------------- sampled hands
def expected_top(values: np.ndarray, k: int) -> list[int]:
    """Indices of the k largest values, ties broken towards the earlier hand."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return sorted(order[:k])


def test_sampled_hands_are_the_global_extremes_whatever_the_session_split() -> None:
    policy = KeepHandsPolicy(first=3, biggest_pots=4, biggest_wins_each=3)
    result = run_match(RANDOM, MANIAC, small_cfg(num_hands=200, session_hands=20, keep_hands=policy))
    arrays = result.arrays
    by_tag = {tag: sorted(s.hand_index for s in result.samples if tag in s.tags) for tag in
              (TAG_FIRST, TAG_BIGGEST_POT, TAG_BIGGEST_WIN_A, TAG_BIGGEST_WIN_B)}  # fmt: skip
    assert by_tag[TAG_FIRST] == [0, 1, 2]
    assert by_tag[TAG_BIGGEST_POT] == expected_top(arrays.pot_chips, 4)
    profit = arrays.profit_chips.astype(int)
    assert by_tag[TAG_BIGGEST_WIN_A] == expected_top(np.where(profit > 0, profit, -1), 3)
    assert by_tag[TAG_BIGGEST_WIN_B] == expected_top(np.where(profit < 0, -profit, -1), 3)
    assert [s.hand_index for s in result.samples] == sorted({i for ids in by_tag.values() for i in ids})
    for sample in result.samples:
        record = HandRecord.from_dict(sample.record)
        assert record.verify()
        assert record.result.payoffs[0] == sample.profit_chips == arrays.profit_chips[sample.hand_index]
        assert record.result.total_pot == sample.pot_chips


def test_sampling_can_be_disabled_and_full_history_is_optional() -> None:
    off = run_match(CALL_STATION, MANIAC, small_cfg(keep_hands=KeepHandsPolicy(0, 0, 0, False)))
    assert off.samples == [] and off.all_hands is None
    with pytest.raises(ValueError):
        KeepHandsPolicy(first=-1)


# ----------------------------------------------------------------------------- seeds
def test_eval_deals_never_coincide_with_training_namespaces() -> None:
    key = match_key("A", "B")
    eval_seeds = {deal_seed(42, key, i) for i in range(2000)}
    assert eval_seeds == {derive_seed("eval", 42, key, i) for i in range(2000)}
    for namespace in ("dataset_train", "dataset_val", "dagger", "ga_fitness", "ga_validation"):
        for parts in (lambda i: (42, i), lambda i: (42, key, i), lambda i: (42, 0, i), lambda i: (i,)):
            assert not eval_seeds & {derive_seed(namespace, *parts(i)) for i in range(2000)}
    eval_deals = {Deal.from_seed(s).cards for s in list(eval_seeds)[:300]}
    train_deals = {Deal.from_seed(derive_seed("dataset_train", 42, i)).cards for i in range(300)}
    assert not eval_deals & train_deals


def test_seed_helpers_are_order_independent_where_promised_and_distinct_otherwise() -> None:
    assert match_key("B", "A") == match_key("A", "B") == "A__vs__B"
    assert deal_seed(1, "k", 0) != deal_seed(2, "k", 0) != deal_seed(2, "k", 1)
    assert agent_seed(1, "k", "A", 0) != agent_seed(1, "k", "B", 0) != agent_seed(1, "k", "B", 1)
    assert len({deal_seed(1, "k", 3), agent_seed(1, "k", "A", 3), bootstrap_seed(1, "k")}) == 3


# ----------------------------------------------------------------------------- configuration
def test_config_rounds_up_to_even_and_derives_sessions() -> None:
    cfg = MatchConfig(num_hands=101, session_hands=15)
    assert (cfg.num_hands, cfg.session_hands, cfg.num_units, cfg.num_sessions) == (102, 16, 51, 7)
    bounds = cfg.session_bounds()
    assert sum(n for _, n in bounds) == 102 and all(first % 2 == 0 for first, _ in bounds)
    assert bounds[-1] == (96, 6)
    assert len(run_match(CALL_STATION, MANIAC, MatchConfig(num_hands=7, session_hands=4)).arrays) == 8


def test_config_validation_and_round_trip() -> None:
    for bad in (dict(num_hands=1), dict(session_hands=1), dict(confidence=1.0), dict(ci_method="wald"),
                dict(bootstrap_resamples=5), dict(ci_unit="week")):  # fmt: skip
        with pytest.raises(ValueError):
            MatchConfig(**bad)
    cfg = MatchConfig(game=GameConfig(500, 2, 4), num_hands=300, seed=9, duplicate=False, ci_method="bootstrap",
                      ci_unit="pair", keep_hands=KeepHandsPolicy(1, 2, 3, True), n_workers=4)  # fmt: skip
    assert MatchConfig.from_dict(cfg.to_dict()) == cfg


def test_resolve_n_workers_is_bounded() -> None:
    assert resolve_n_workers(1, 100) == 1
    assert resolve_n_workers(8, 3) == 3  # never more workers than tasks
    assert 1 <= resolve_n_workers(0, 1000) <= 61
    assert resolve_n_workers(500, 1000) == 61  # Windows pool limit


def test_unknown_agent_kind_fails_loudly() -> None:
    with pytest.raises(KeyError):
        run_match("no_such_agent", MANIAC, small_cfg(num_hands=4, session_hands=4))


# ----------------------------------------------------------------------------- throughput
@pytest.mark.slow
def test_two_thousand_hands_single_core_finish_in_reasonable_time() -> None:
    started = time.perf_counter()
    result = run_match(CALL_STATION, MANIAC, MatchConfig(num_hands=2000, keep_hands=KeepHandsPolicy()))
    assert len(result.arrays) == 2000
    assert time.perf_counter() - started < 120  # the PokerKit showdown evaluation dominates (~100-200 hands/s)
