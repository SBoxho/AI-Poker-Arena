"""Discretised action abstraction: mask, raise amounts and abstract -> concrete translation."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from poker_arena.agents.base import Decision
from poker_arena.engine.actions import (
    ABSTRACT_ACTION_NAMES,
    NUM_ABSTRACT_ACTIONS,
    AbstractAction,
    Action,
    ActionKind,
    IllegalActionError,
    LegalActions,
    abstract_raise_amounts,
    abstract_to_action,
    legal_abstract_mask,
    pot_fraction_raise_to,
)
from poker_arena.engine.game import HandEngine

from core_helpers import DEFAULT_CFG, make_deal, new_engine, play_script, random_midhand_engines

A = AbstractAction
NO_ACTION = LegalActions(False, False, False, 0, False, 0, 0)


def raise_legal(lo: int, hi: int, *, call: int = 5, check: bool = False) -> LegalActions:
    """Synthetic legal-actions record for pure-function tests of the abstraction."""
    return LegalActions(
        can_fold=not check, can_check=check, can_call=not check,
        call_amount=0 if check else call, can_raise=True, min_raise_to=lo, max_raise_to=hi,
    )


def mask_list(mask: np.ndarray) -> list[bool]:
    return [bool(x) for x in mask]


# ================================================================================ enum basics
def test_abstract_action_enum_contract() -> None:
    assert [int(a) for a in A] == [0, 1, 2, 3, 4]
    assert [a.name for a in A] == ["FOLD", "CHECK_CALL", "RAISE_SMALL", "RAISE_POT", "ALL_IN"]
    assert NUM_ABSTRACT_ACTIONS == 5
    assert ABSTRACT_ACTION_NAMES == ("fold", "check_call", "raise_small", "raise_pot", "all_in")


def test_mask_has_expected_dtype_and_shape() -> None:
    m = legal_abstract_mask(15, 5, 10, raise_legal(20, 1000))
    assert isinstance(m, np.ndarray) and m.dtype == bool and m.shape == (NUM_ABSTRACT_ACTIONS,)


# ================================================================ hand-computed formula examples
def test_preflop_small_blind_first_to_act_pot_15() -> None:
    """pot 15, SB has 5 in, BB 10 -> after calling 5 the pot is 20: half = 10 on top of 10, pot = 20 on top of 10."""
    e = new_engine(button_seat=0)
    obs = e.observation_for(0)
    assert (obs.pot, obs.my_bet, obs.opp_bet) == (15, 5, 10)
    amounts = abstract_raise_amounts(obs.pot, obs.my_bet, obs.opp_bet, obs.legal)
    assert amounts == {A.RAISE_SMALL: 20, A.RAISE_POT: 30, A.ALL_IN: 1000}
    assert mask_list(obs.abstract_mask) == [True, True, True, True, True]
    assert abstract_to_action(A.RAISE_POT, obs.pot, obs.my_bet, obs.opp_bet, obs.legal) == Action.raise_to(30)


def test_unopened_flop_with_pot_20() -> None:
    e = new_engine(button_seat=1)
    play_script(e, ["c", "c"])
    obs = e.observation_for(e.actor_seat)  # type: ignore[arg-type]
    assert (obs.pot, obs.my_bet, obs.opp_bet) == (20, 0, 0)
    amounts = abstract_raise_amounts(obs.pot, obs.my_bet, obs.opp_bet, obs.legal)
    # half pot = 10 (= the minimum bet), pot = 20, all-in = remaining 990
    assert amounts == {A.RAISE_SMALL: 10, A.RAISE_POT: 20, A.ALL_IN: 990}
    assert mask_list(obs.abstract_mask) == [False, True, True, True, True], "no fold when a check is free"


def test_facing_a_flop_bet() -> None:
    e = new_engine(button_seat=0)
    play_script(e, ["c", "c", 20])  # BB bets 20 into 20
    obs = e.observation_for(0)
    assert (obs.pot, obs.my_bet, obs.opp_bet, obs.to_call) == (40, 0, 20, 20)
    assert (obs.legal.min_raise_to, obs.legal.max_raise_to) == (40, 990)
    # after calling the pot is 60: half = 30 on top of 20 -> 50; full = 60 on top of 20 -> 80
    assert abstract_raise_amounts(obs.pot, obs.my_bet, obs.opp_bet, obs.legal) == {
        A.RAISE_SMALL: 50, A.RAISE_POT: 80, A.ALL_IN: 990,
    }
    assert mask_list(obs.abstract_mask) == [True, True, True, True, True]


def test_pot_fraction_formula_directly() -> None:
    legal = raise_legal(1, 10_000)
    assert pot_fraction_raise_to(0.5, 100, 0, 0, legal) == 50
    assert pot_fraction_raise_to(1.0, 100, 0, 0, legal) == 100
    assert pot_fraction_raise_to(1.0, 300, 0, 100, legal) == 100 + 400  # call 100, pot 400, raise by 400
    assert pot_fraction_raise_to(0.5, 300, 50, 100, legal) == 100 + 175  # to_call 50 -> pot 350 -> half 175
    assert pot_fraction_raise_to(0.0, 100, 0, 0, legal) == 1  # clamped up to the minimum


# ======================================================================= clamping and dedupe
def test_min_raise_clamping_collapses_duplicates_in_priority_order() -> None:
    # raw targets 20 (small) and 30 (pot) are both below the minimum 40 -> both clamp to 40 -> keep RAISE_SMALL
    legal = raise_legal(40, 1000)
    assert abstract_raise_amounts(15, 5, 10, legal) == {A.RAISE_SMALL: 40, A.ALL_IN: 1000}
    assert mask_list(legal_abstract_mask(15, 5, 10, legal)) == [True, True, True, False, True]


def test_max_clamping_drops_raises_that_equal_all_in() -> None:
    legal = raise_legal(20, 25)  # stack only allows 20..25: pot raise (30) would be all-in
    assert abstract_raise_amounts(15, 5, 10, legal) == {A.RAISE_SMALL: 20, A.ALL_IN: 25}
    legal = raise_legal(20, 30)  # pot raise == all-in exactly -> only ALL_IN survives for it
    assert abstract_raise_amounts(15, 5, 10, legal) == {A.RAISE_SMALL: 20, A.ALL_IN: 30}


def test_short_all_in_only_when_min_equals_max() -> None:
    legal = raise_legal(15, 15)  # all-in below the full minimum raise
    assert abstract_raise_amounts(15, 5, 10, legal) == {A.ALL_IN: 15}
    assert mask_list(legal_abstract_mask(15, 5, 10, legal)) == [True, True, False, False, True]


def test_no_raise_means_only_fold_and_call() -> None:
    legal = LegalActions(True, False, True, 90, False, 0, 0)
    assert abstract_raise_amounts(110, 10, 100, legal) == {}
    assert mask_list(legal_abstract_mask(110, 10, 100, legal)) == [True, True, False, False, False]
    legal = LegalActions(False, True, False, 0, False, 0, 0)  # check only
    assert mask_list(legal_abstract_mask(0, 0, 0, legal)) == [False, True, False, False, False]


def test_no_actor_mask_is_all_false() -> None:
    assert mask_list(legal_abstract_mask(100, 0, 0, NO_ACTION)) == [False] * 5
    e = new_engine(button_seat=0)
    assert not e.observation_for(1).abstract_mask.any()  # not the actor
    e.act(Action.fold())
    assert not e.observation_for(0).abstract_mask.any() and not e.observation_for(1).abstract_mask.any()


# ================================================================== illegal abstract requests
def test_illegal_abstract_requests_raise() -> None:
    check_legal = raise_legal(20, 1000, check=True)
    with pytest.raises(IllegalActionError, match="Fold"):
        abstract_to_action(A.FOLD, 20, 0, 0, check_legal)
    no_raise = LegalActions(True, False, True, 50, False, 0, 0)
    for raise_kind in (A.RAISE_SMALL, A.RAISE_POT, A.ALL_IN):
        with pytest.raises(IllegalActionError):
            abstract_to_action(raise_kind, 100, 0, 50, no_raise)
    deduped = raise_legal(40, 1000)
    with pytest.raises(IllegalActionError, match="RAISE_POT"):
        abstract_to_action(A.RAISE_POT, 15, 5, 10, deduped)
    with pytest.raises(IllegalActionError):
        abstract_to_action(A.CHECK_CALL, 0, 0, 0, NO_ACTION)
    with pytest.raises(IllegalActionError):
        abstract_to_action(A.FOLD, 0, 0, 0, NO_ACTION)


def test_decision_from_abstract_wraps_abstract_to_action() -> None:
    e = new_engine(button_seat=0)
    obs = e.observation_for(0)
    d = Decision.from_abstract(obs, A.RAISE_POT)
    assert d.action == Action.raise_to(30) and d.abstract is A.RAISE_POT
    assert Decision.from_abstract(obs, A.CHECK_CALL).action == Action.check_call()
    assert Decision.from_abstract(obs, A.FOLD).action == Action.fold()
    with pytest.raises(IllegalActionError):
        Decision.from_abstract(new_engine(button_seat=0).observation_for(1), A.CHECK_CALL)  # not to act


# ===================================================================== property checks on states
def check_state(engine: HandEngine) -> int:
    """All abstraction invariants for the engine's current decision point. Returns #true mask entries."""
    seat = engine.actor_seat
    assert seat is not None
    obs = engine.observation_for(seat)
    lg = obs.legal
    mask = obs.abstract_mask
    amounts = abstract_raise_amounts(obs.pot, obs.my_bet, obs.opp_bet, lg)

    assert mask.any(), "the actor always has at least one abstract action"
    assert mask[A.CHECK_CALL], "check/call is always legal for the actor"
    if lg.can_check:
        assert not mask[A.FOLD], "never fold when a free check exists"
    else:
        assert mask[A.FOLD]
    assert not engine.observation_for(1 - seat).abstract_mask.any(), "the waiting player has no actions"

    raises = [a for a in (A.RAISE_SMALL, A.RAISE_POT, A.ALL_IN) if mask[a]]
    assert set(raises) == set(amounts)
    assert mask[A.ALL_IN] == lg.can_raise
    if lg.can_raise:
        assert amounts[A.ALL_IN] == lg.max_raise_to
        for a in (A.RAISE_SMALL, A.RAISE_POT):
            if a in amounts:
                assert lg.min_raise_to <= amounts[a] < lg.max_raise_to
        assert len(set(amounts.values())) == len(amounts), "distinct abstractions -> distinct amounts"
        if A.RAISE_SMALL in amounts and A.RAISE_POT in amounts:
            assert amounts[A.RAISE_SMALL] < amounts[A.RAISE_POT]
    else:
        assert not amounts

    for a in A:
        if not mask[a]:
            with pytest.raises(IllegalActionError):
                abstract_to_action(a, obs.pot, obs.my_bet, obs.opp_bet, lg)
            continue
        action = abstract_to_action(a, obs.pot, obs.my_bet, obs.opp_bet, lg)
        clone = copy.deepcopy(engine)
        event = clone.act(action)  # the ENGINE accepts every action the mask promises
        if a is A.FOLD:
            assert action.kind is ActionKind.FOLD and event.kind == "fold"
        elif a is A.CHECK_CALL:
            assert action.kind is ActionKind.CHECK_CALL and event.kind in ("check", "call")
            assert event.chips == lg.call_amount
        else:
            assert action == Action.raise_to(amounts[a]) and event.kind in ("bet", "raise")
            assert event.street_total == amounts[a]
        d = Decision.from_abstract(obs, a)
        assert d.action == action and d.abstract is a
    return int(mask.sum())


@pytest.mark.parametrize("seed", range(6))
def test_every_true_mask_entry_is_accepted_by_the_engine(seed: int) -> None:
    n_true = 0
    for engine in random_midhand_engines(seed, 70):
        n_true += check_state(engine)
    assert n_true > 70 * 2


GRID = [1, 3, 5, 6, 9, 10, 11, 14, 15, 16, 20, 21, 25, 30]  # around the blinds / min-raise boundaries


def test_exhaustive_short_stack_grid_at_the_start_of_a_hand() -> None:
    """Stack pairs around the blinds, both button seats: first decision and the BB option after a limp."""
    for s0 in GRID:
        for s1 in GRID:
            for button in (0, 1):
                e = HandEngine(DEFAULT_CFG, make_deal("AsKs", "2c7d", "9h8d3cQsJd"), button, (s0, s1))
                if e.is_complete:
                    continue
                check_state(e)
                lg = e.legal_actions()
                if lg.can_call and lg.call_amount < e.stack(e.actor_seat):  # type: ignore[arg-type]
                    e.act(Action.check_call())
                    if not e.is_complete:
                        check_state(e)


def test_asymmetric_deep_vs_short_after_a_raise() -> None:
    """Deep stack raises, short stack must respond: RAISE_* collapse as the stack shrinks."""
    sizes_seen = set()
    for short in range(11, 140, 3):
        e = HandEngine(DEFAULT_CFG, make_deal("AsKs", "2c7d", "9h8d3cQsJd"), 0, (1000, short))
        e.act(Action.raise_to(30))
        if e.is_complete:
            continue
        sizes_seen.add(check_state(e))
    assert sizes_seen >= {3, 4, 5} or sizes_seen >= {2, 3, 4, 5}, sizes_seen


def test_all_in_amount_equals_remaining_stack_plus_bet_every_street() -> None:
    e = new_engine(button_seat=0, stacks=(500, 800))
    obs = e.observation_for(0)
    assert abstract_raise_amounts(obs.pot, obs.my_bet, obs.opp_bet, obs.legal)[A.ALL_IN] == 500
    play_script(e, ["c", "c", 40, "c"])  # to the turn, pot 100; BB (800 start) has 750 left, button 450
    obs = e.observation_for(e.actor_seat)  # type: ignore[arg-type]
    assert (obs.my_stack, obs.opp_stack, obs.my_bet) == (750, 450, 0)
    # ALL_IN is the actor's whole stack (PokerKit's maximum), even though only 450 can be called
    assert abstract_raise_amounts(obs.pot, obs.my_bet, obs.opp_bet, obs.legal)[A.ALL_IN] == 750
    assert obs.effective_stack == 450
