"""Heads-up No-Limit Hold'em behaviour of ``HandEngine`` (PokerKit is the rules authority).

Hand-computed scenarios (blinds 5/10, stacks 1000 unless stated) + a chip-conservation fuzz.
"""

from __future__ import annotations

import random

import pytest

from poker_arena.cards import ALL_CARDS
from poker_arena.config import GameConfig
from poker_arena.engine.actions import Action, IllegalActionError, LegalActions
from poker_arena.engine.deals import Deal
from poker_arena.engine.game import HandEngine
from poker_arena.engine.observation import BOARD_CARDS_BY_STREET, Street

from core_helpers import (
    DEFAULT_CFG,
    MAX_ACTIONS_PER_HAND,
    make_deal,
    new_engine,
    play_script,
    random_action,
    random_deal,
    random_stacks,
)

NO_ACTION = LegalActions(False, False, False, 0, False, 0, 0)
BUTTONS = pytest.mark.parametrize("button", [0, 1])


def legal(**kw) -> LegalActions:
    base = dict(can_fold=False, can_check=False, can_call=False, call_amount=0, can_raise=False,
                min_raise_to=0, max_raise_to=0)
    base.update(kw)
    return LegalActions(**base)


# =========================================================================== blinds / order
@BUTTONS
def test_blinds_posted_and_button_acts_first_preflop(button: int) -> None:
    e = new_engine(button_seat=button)
    other = 1 - button
    assert e.actor_seat == button, "heads-up: the button (small blind) acts first preflop"
    assert (e.bet(button), e.bet(other)) == (5, 10)
    assert (e.stack(button), e.stack(other)) == (995, 990)
    assert e.pot == 15
    assert (e.committed(button), e.committed(other)) == (5, 10)
    assert e.street == Street.PREFLOP and e.board == ()
    assert e.is_button(button) and not e.is_button(other)
    assert e.legal_actions() == legal(
        can_fold=True, can_call=True, call_amount=5, can_raise=True, min_raise_to=20, max_raise_to=1000
    )


@BUTTONS
def test_non_actor_sees_no_legal_actions(button: int) -> None:
    e = new_engine(button_seat=button)
    assert e.observation_for(button).legal == e.legal_actions()
    assert e.observation_for(1 - button).legal == NO_ACTION
    assert not e.observation_for(1 - button).abstract_mask.any()


@BUTTONS
def test_limp_then_big_blind_option_check(button: int) -> None:
    e = new_engine(button_seat=button)
    ev = e.act(Action.check_call())  # small blind completes
    assert (ev.kind, ev.chips, ev.street_total, ev.pot_after, ev.seat) == ("call", 5, 10, 20, button)
    assert e.street == Street.PREFLOP and e.board == (), "the BB still has the option"
    assert e.actor_seat == 1 - button
    assert e.legal_actions() == legal(
        can_check=True, can_raise=True, min_raise_to=20, max_raise_to=1000
    ), "BB option: check or raise, never fold"
    ev = e.act(Action.check_call())
    assert (ev.kind, ev.chips, ev.street, ev.seat) == ("check", 0, Street.PREFLOP, 1 - button)
    # street closes -> flop, big blind (non-button) acts first
    assert e.street == Street.FLOP and len(e.board) == 3
    assert e.actor_seat == 1 - button
    assert e.pot == 20 and (e.bet(0), e.bet(1)) == (0, 0)


@BUTTONS
def test_limp_then_big_blind_raises(button: int) -> None:
    e = new_engine(button_seat=button)
    e.act(Action.check_call())
    ev = e.act(Action.raise_to(30))
    assert (ev.kind, ev.chips, ev.street_total, ev.pot_after) == ("raise", 20, 30, 40)
    assert e.actor_seat == button
    assert e.legal_actions() == legal(
        can_fold=True, can_call=True, call_amount=20, can_raise=True, min_raise_to=50, max_raise_to=1000
    )


@BUTTONS
def test_preflop_raise_reraise_sequence(button: int) -> None:
    e = new_engine(button_seat=button)
    ev1 = e.act(Action.raise_to(30))  # SB raises: first preflop raise is a "raise" (blinds are a bet)
    assert (ev1.kind, ev1.chips, ev1.street_total, ev1.pot_after) == ("raise", 25, 30, 40)
    assert e.legal_actions().min_raise_to == 50 and e.legal_actions().call_amount == 20
    ev2 = e.act(Action.raise_to(90))  # BB 3-bets: increment 60
    assert (ev2.kind, ev2.chips, ev2.street_total, ev2.pot_after) == ("raise", 80, 90, 120)
    assert e.actor_seat == button
    assert e.legal_actions().min_raise_to == 150 and e.legal_actions().call_amount == 60
    ev3 = e.act(Action.raise_to(150))  # 4-bet, minimum
    assert (ev3.chips, ev3.street_total, ev3.pot_after) == (120, 150, 240)
    assert e.legal_actions().min_raise_to == 210 and e.legal_actions().call_amount == 60
    ev4 = e.act(Action.check_call())
    assert (ev4.kind, ev4.chips, ev4.street_total, ev4.pot_after) == ("call", 60, 150, 300)
    assert ev4.street == Street.PREFLOP, "the closing call belongs to the street it closes"
    assert e.street == Street.FLOP and e.pot == 300 and e.actor_seat == 1 - button
    assert [x.seat for x in e.events] == [button, 1 - button, button, 1 - button]


@BUTTONS
def test_preflop_min_raise_chain_uses_previous_increment(button: int) -> None:
    e = new_engine(button_seat=button)
    e.act(Action.raise_to(20))  # increment 10
    assert e.legal_actions().min_raise_to == 30
    e.act(Action.raise_to(30))
    assert e.legal_actions().min_raise_to == 40
    e.act(Action.raise_to(60))  # increment 30
    assert e.legal_actions().min_raise_to == 90


def test_fold_ends_hand_without_showdown() -> None:
    e = new_engine(button_seat=0)
    ev = e.act(Action.fold())
    assert (ev.kind, ev.chips, ev.all_in, ev.pot_after, ev.street_total) == ("fold", 0, False, 15, 5)
    r = e.result()
    assert e.is_complete and e.actor_seat is None
    assert r.payoffs == (-5, 5) and r.final_stacks == (995, 1005)
    assert r.winner == 1 and not r.showdown and r.shown_hands == {} and r.board == ()
    assert r.ended_street == Street.PREFLOP and r.total_pot == 10


# =========================================================================== street progression
@BUTTONS
def test_check_down_street_progression_and_board_reveal(button: int) -> None:
    deal = Deal.from_seed(99)
    e = new_engine(deal, button)
    seen: list[tuple[Street, int]] = []
    streets_first_actor: dict[Street, int] = {}
    while not e.is_complete:
        seen.append((e.street, len(e.board)))
        streets_first_actor.setdefault(e.street, e.actor_seat)  # type: ignore[arg-type]
        # nothing beyond the dealt board is ever visible
        assert e.board == deal.board[: len(e.board)]
        assert e.observation_for(0).board == e.board == e.observation_for(1).board
        assert e.spectator_view().board == e.board
        e.act(Action.check_call())
    assert [s for s, _ in dict.fromkeys(seen)] == [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
    assert [n for _, n in dict.fromkeys(seen)] == [0, 3, 4, 5]
    assert streets_first_actor == {
        Street.PREFLOP: button,
        Street.FLOP: 1 - button,
        Street.TURN: 1 - button,
        Street.RIVER: 1 - button,
    }
    assert len(e.events) == 2 + 2 + 2 + 2
    r = e.result()
    assert r.showdown and r.board == deal.board and r.ended_street == Street.RIVER
    assert r.total_pot == 20 and sorted(r.shown_hands) == [0, 1]


def test_burn_cards_are_dealt_in_order_before_each_street() -> None:
    deal = Deal.from_seed(7)
    e = new_engine(deal, 0)
    play_script(e, ["c"] * 8)
    assert [repr(c) for c in e._state.burn_cards] == list(deal.burns)
    assert [repr(c) for cards in e._state.board_cards for c in cards] == list(deal.board)


def test_no_more_board_cards_than_streets_reached_on_early_fold() -> None:
    deal = Deal.from_seed(11)
    e = new_engine(deal, 1)
    play_script(e, ["c", "c", "c", "c", 40, "f"])  # fold to a turn bet
    r = e.result()
    assert r.board == deal.board[:4] and r.ended_street == Street.TURN and not r.showdown


@BUTTONS
def test_event_street_attribution_for_actions_closing_a_street(button: int) -> None:
    e = new_engine(button_seat=button)
    evs = play_script(e, ["c", "c", "c", "c", "c", "c", 20, "c"])
    assert [ev.street for ev in evs] == [Street.PREFLOP] * 2 + [Street.FLOP] * 2 + [Street.TURN] * 2 + [
        Street.RIVER
    ] * 2
    assert [ev.seat for ev in evs[-2:]] == [1 - button, button]
    assert evs[-2].kind == "bet" and evs[-1].kind == "call"


@BUTTONS
def test_postflop_bet_raise_and_min_raise_rules(button: int) -> None:
    bb = 1 - button
    e = new_engine(button_seat=button)
    play_script(e, ["c", "c"])  # limp, check -> flop, pot 20
    assert e.actor_seat == bb
    assert e.legal_actions() == legal(can_check=True, can_raise=True, min_raise_to=10, max_raise_to=990)
    assert e.pot == 20
    ev = e.act(Action.raise_to(40))
    assert (ev.kind, ev.chips, ev.street_total, ev.pot_after) == ("bet", 40, 40, 60)
    assert e.legal_actions() == legal(
        can_fold=True, can_call=True, call_amount=40, can_raise=True, min_raise_to=80, max_raise_to=990
    )
    ev = e.act(Action.raise_to(80))
    assert ev.kind == "raise" and ev.pot_after == 140
    assert e.legal_actions().min_raise_to == 120 and e.legal_actions().call_amount == 40
    ev = e.act(Action.raise_to(200))  # increment 120
    assert ev.chips == 160 and ev.pot_after == 300
    assert e.legal_actions().min_raise_to == 320 and e.legal_actions().call_amount == 120
    e.act(Action.check_call())
    assert e.street == Street.TURN and e.pot == 420 and len(e.board) == 4
    assert e.actor_seat == bb


@BUTTONS
def test_check_raise_line(button: int) -> None:
    bb = 1 - button
    e = new_engine(button_seat=button)
    play_script(e, ["c", "c", "c"])  # flop: BB checks
    assert e.actor_seat == button
    ev = e.act(Action.raise_to(15))
    assert ev.kind == "bet" and e.actor_seat == bb
    assert e.legal_actions().min_raise_to == 30
    ev = e.act(Action.raise_to(45))
    assert ev.kind == "raise" and ev.street_total == 45


@pytest.mark.parametrize("amount", [-10, -1, 0, 1, 9, 19])
def test_raise_below_minimum_is_illegal_preflop(amount: int) -> None:
    e = new_engine(button_seat=0)
    before = (e.pot, e.actor_seat, len(e.events), e.legal_actions())
    with pytest.raises(IllegalActionError, match="outside legal range"):
        e.act(Action.raise_to(amount))
    assert (e.pot, e.actor_seat, len(e.events), e.legal_actions()) == before, "state untouched by an illegal action"


@pytest.mark.parametrize("amount", [1001, 1500, 10**6])
def test_raise_above_stack_is_illegal(amount: int) -> None:
    e = new_engine(button_seat=1)
    with pytest.raises(IllegalActionError):
        e.act(Action.raise_to(amount))


def test_min_raise_boundary_postflop() -> None:
    e = new_engine(button_seat=0)
    play_script(e, ["c", "c"])
    with pytest.raises(IllegalActionError):
        e.act(Action.raise_to(9))  # below the minimum opening bet (= big blind)
    e.act(Action.raise_to(10))  # exactly the minimum opens
    with pytest.raises(IllegalActionError):
        e.act(Action.raise_to(19))  # min raise-to is 20
    e.act(Action.raise_to(20))


def test_max_raise_is_all_in_and_all_in_flag() -> None:
    e = new_engine(button_seat=0)
    ev = e.act(Action.raise_to(1000))
    assert ev.all_in and ev.chips == 995 and ev.pot_after == 1010
    assert e.stack(0) == 0
    # opponent can only fold or call; no raise offered
    assert e.legal_actions() == legal(can_fold=True, can_call=True, call_amount=990)
    with pytest.raises(IllegalActionError, match="not legal"):
        e.act(Action.raise_to(1000))


@BUTTONS
def test_fold_with_free_check_is_illegal(button: int) -> None:
    # 1) big blind option after a limp
    e = new_engine(button_seat=button)
    e.act(Action.check_call())
    snapshot = (e.pot, e.actor_seat, len(e.events))
    with pytest.raises(IllegalActionError, match="free check"):
        e.act(Action.fold())
    assert (e.pot, e.actor_seat, len(e.events)) == snapshot
    e.act(Action.check_call())
    # 2) first to act postflop
    with pytest.raises(IllegalActionError, match="free check"):
        e.act(Action.fold())
    e.act(Action.check_call())
    # 3) second to act after a check
    with pytest.raises(IllegalActionError, match="free check"):
        e.act(Action.fold())
    assert e.legal_actions().can_fold is False and e.legal_actions().can_check


def test_acting_after_hand_end_raises() -> None:
    e = new_engine(button_seat=0)
    e.act(Action.fold())
    for action in (Action.fold(), Action.check_call(), Action.raise_to(20)):
        with pytest.raises(IllegalActionError, match="over"):
            e.act(action)
    assert e.legal_actions() == NO_ACTION
    assert e.observation_for(0).legal == NO_ACTION and e.observation_for(1).legal == NO_ACTION
    assert len(e.events) == 1


def test_result_before_end_raises() -> None:
    with pytest.raises(RuntimeError):
        new_engine().result()
    assert new_engine().spectator_view().result is None


def test_start_stacks_are_normalised_to_plain_int_tuples() -> None:
    import json

    import numpy as np

    from poker_arena.engine.session import play_hand
    from poker_arena.agents.baselines import CallStationAgent

    e = HandEngine(DEFAULT_CFG, Deal.from_seed(1), 0, [np.int64(300), 500])  # type: ignore[arg-type]
    assert e.start_stacks == (300, 500) and type(e.start_stacks) is tuple
    assert all(type(x) is int for x in e.start_stacks)
    record = play_hand(DEFAULT_CFG, Deal.from_seed(1), 0, [CallStationAgent(), CallStationAgent()], start_stacks=[300, 500])  # type: ignore[arg-type]
    assert record.start_stacks == (300, 500)
    json.dumps(record.to_dict())  # numpy ints would raise here
    with pytest.raises(ValueError, match="one stack per seat"):
        HandEngine(DEFAULT_CFG, Deal.from_seed(1), 0, (100, 100, 100))  # type: ignore[arg-type]


def test_invalid_engine_arguments() -> None:
    with pytest.raises(ValueError):
        HandEngine(DEFAULT_CFG, Deal.from_seed(1), 2)
    with pytest.raises(ValueError):
        HandEngine(DEFAULT_CFG, Deal.from_seed(1), 0, (0, 100))
    with pytest.raises(ValueError):
        HandEngine(DEFAULT_CFG, Deal.from_seed(1), 0, (100, -5))


# =========================================================================== short stacks / all-ins
DEAL_BTN_WINS = make_deal("AsAh", "2c7d", "KdQs9h4c3d")  # button has aces, BB has air
DEAL_BB_WINS = make_deal("2c7d", "AsAh", "KdQs9h4c3d")


@pytest.mark.parametrize(
    "button,deal,expected_payoff_of_button",
    [(0, DEAL_BTN_WINS, +3), (1, DEAL_BTN_WINS, +3), (0, DEAL_BB_WINS, -3), (1, DEAL_BB_WINS, -3)],
)
def test_small_blind_with_less_than_the_blind_is_all_in_and_excess_is_returned(
    button: int, deal: Deal, expected_payoff_of_button: int
) -> None:
    stacks = [1000, 1000]
    stacks[button] = 3
    e = HandEngine(DEFAULT_CFG, deal, button, (stacks[0], stacks[1]))
    # nobody can act: SB is all-in for 3, BB's extra 7 chips are uncalled
    assert e.is_complete and e.actor_seat is None and e.events == []
    r = e.result()
    bb = 1 - button
    assert r.payoffs[button] == expected_payoff_of_button and r.payoffs[bb] == -expected_payoff_of_button
    assert r.showdown and r.total_pot == 6 and r.board == deal.board and r.ended_street == Street.RIVER
    assert r.final_stacks[button] == 3 + expected_payoff_of_button
    assert r.final_stacks[bb] == 1000 - expected_payoff_of_button
    assert r.shown_hands == {s: (deal.button_hole if s == button else deal.bb_hole) for s in (0, 1)}


@BUTTONS
def test_big_blind_with_less_than_the_blind_button_may_call_or_fold(button: int) -> None:
    bb = 1 - button
    stacks = [1000, 1000]
    stacks[bb] = 7
    e = HandEngine(DEFAULT_CFG, DEAL_BTN_WINS, button, (stacks[0], stacks[1]))
    assert e.actor_seat == button and e.pot == 12
    assert (e.bet(button), e.bet(bb), e.stack(bb)) == (5, 7, 0)
    assert e.legal_actions() == legal(can_fold=True, can_call=True, call_amount=2), "opponent all-in: fold/call only"
    ev = e.act(Action.check_call())
    assert ev.kind == "call" and ev.chips == 2 and not ev.all_in
    assert e.is_complete  # automatic runout
    r = e.result()
    assert r.showdown and r.board == DEAL_BTN_WINS.board and r.total_pot == 14
    assert r.payoffs[button] == 7 and r.payoffs[bb] == -7 and r.winner == button


@BUTTONS
@pytest.mark.parametrize("bb_stack", [1, 3, 5])
def test_big_blind_posting_no_more_than_the_small_blind_ends_the_betting(button: int, bb_stack: int) -> None:
    """BB all-in for <= the SB amount: the SB's surplus is uncalled, nobody has a decision."""
    bb = 1 - button
    stacks = [1000, 1000]
    stacks[bb] = bb_stack
    e = HandEngine(DEFAULT_CFG, DEAL_BTN_WINS, button, (stacks[0], stacks[1]))
    assert e.is_complete and e.events == []
    r = e.result()
    assert r.total_pot == 2 * bb_stack and r.showdown and r.board == DEAL_BTN_WINS.board
    assert r.payoffs[button] == bb_stack and r.payoffs[bb] == -bb_stack and r.winner == button
    assert r.final_stacks[button] == 1000 + bb_stack and r.final_stacks[bb] == 0


@BUTTONS
def test_big_blind_short_button_folds_loses_only_the_small_blind(button: int) -> None:
    stacks = [1000, 1000]
    stacks[1 - button] = 7
    e = HandEngine(DEFAULT_CFG, DEAL_BTN_WINS, button, (stacks[0], stacks[1]))
    e.act(Action.fold())
    r = e.result()
    assert r.payoffs[button] == -5 and r.payoffs[1 - button] == 5 and not r.showdown and r.total_pot == 10


@BUTTONS
def test_blind_exactly_equal_to_stack_means_opponent_cannot_raise(button: int) -> None:
    stacks = [1000, 1000]
    stacks[1 - button] = 10  # BB posts its whole stack
    e = HandEngine(DEFAULT_CFG, DEAL_BTN_WINS, button, (stacks[0], stacks[1]))
    assert e.actor_seat == button
    assert e.legal_actions() == legal(can_fold=True, can_call=True, call_amount=5)


@BUTTONS
def test_both_players_below_the_blinds(button: int) -> None:
    stacks = [0, 0]
    stacks[button], stacks[1 - button] = 3, 4
    e = HandEngine(DEFAULT_CFG, DEAL_BTN_WINS, button, (stacks[0], stacks[1]))
    assert e.is_complete
    r = e.result()
    # matched amount is 3 (BB's 4th chip is returned); button holds the aces
    assert r.total_pot == 6 and r.payoffs[button] == 3 and r.payoffs[1 - button] == -3
    assert r.final_stacks[button] == 6 and r.final_stacks[1 - button] == 1


@BUTTONS
@pytest.mark.parametrize("big_stack_wins", [True, False])
def test_call_all_in_for_less_returns_uncalled_excess(button: int, big_stack_wins: bool) -> None:
    big, small = button, 1 - button  # button has 1000, BB has only 300
    stacks = [0, 0]
    stacks[big], stacks[small] = 1000, 300
    base = DEAL_BTN_WINS if big_stack_wins else DEAL_BB_WINS
    e = HandEngine(DEFAULT_CFG, base, button, (stacks[0], stacks[1]))
    ev = e.act(Action.raise_to(500))
    assert not ev.all_in and ev.chips == 495
    assert e.actor_seat == small
    assert e.legal_actions() == legal(can_fold=True, can_call=True, call_amount=290), "call is for less than the bet"
    assert e.observation_for(small).to_call == 290
    ev = e.act(Action.check_call())
    assert ev.kind == "call" and ev.chips == 290 and ev.all_in and ev.street_total == 300
    r = e.result()
    assert r.total_pot == 600, "only the matched 300 + 300 is contested; 200 is returned"
    if big_stack_wins:
        assert r.payoffs[big] == 300 and r.payoffs[small] == -300 and r.final_stacks[big] == 1300
        assert r.final_stacks[small] == 0 and r.winner == big
    else:
        assert r.payoffs[small] == 300 and r.payoffs[big] == -300 and r.final_stacks[small] == 600
        assert r.final_stacks[big] == 700 and r.winner == small
    assert sum(r.final_stacks) == 1300


@BUTTONS
def test_short_all_in_below_min_raise_is_offered_and_ends_raising(button: int) -> None:
    bb = 1 - button
    stacks = [0, 0]
    stacks[button], stacks[bb] = 1000, 150
    e = HandEngine(DEFAULT_CFG, DEAL_BTN_WINS, button, (stacks[0], stacks[1]))
    e.act(Action.raise_to(100))
    # BB (150 total) cannot make a full raise (min 190) but may shove for less
    assert e.legal_actions() == legal(
        can_fold=True, can_call=True, call_amount=90, can_raise=True, min_raise_to=150, max_raise_to=150
    )
    with pytest.raises(IllegalActionError):
        e.act(Action.raise_to(149))
    ev = e.act(Action.raise_to(150))
    assert ev.kind == "raise" and ev.all_in and ev.chips == 140 and ev.pot_after == 250
    # the shover is all-in: the original raiser can only fold or call the 50 difference
    assert e.legal_actions() == legal(can_fold=True, can_call=True, call_amount=50)
    e.act(Action.check_call())
    r = e.result()
    assert r.total_pot == 300 and r.payoffs[button] == 150 and r.payoffs[bb] == -150


def test_opponent_all_in_offers_only_fold_or_call() -> None:
    e = new_engine(button_seat=0)
    e.act(Action.raise_to(1000))
    obs = e.observation_for(1)
    assert obs.legal.can_fold and obs.legal.can_call and not obs.legal.can_raise and not obs.legal.can_check
    assert obs.legal.call_amount == 990 and list(obs.abstract_mask) == [True, True, False, False, False]
    assert obs.opp_stack == 0 and obs.effective_stack == 0


def test_call_for_less_than_stack_flagged_all_in_only_when_stack_is_gone() -> None:
    e = new_engine(button_seat=0, stacks=(1000, 400))
    e.act(Action.raise_to(300))
    ev = e.act(Action.check_call())  # BB has 400: calls 290 and keeps 100
    assert ev.kind == "call" and ev.chips == 290 and not ev.all_in
    assert e.stack(1) == 100 and e.street == Street.FLOP


@BUTTONS
def test_both_all_in_preflop_runs_out_the_correct_board(button: int) -> None:
    deal = Deal.from_seed(2024)
    e = new_engine(deal, button)
    e.act(Action.raise_to(1000))
    e.act(Action.check_call())
    assert e.is_complete
    r = e.result()
    assert r.board == deal.board and r.showdown and r.ended_street == Street.RIVER
    assert [repr(c) for c in e._state.burn_cards] == list(deal.burns)
    assert r.total_pot == 2000 and sum(r.final_stacks) == 2000 and sorted(r.payoffs) in ([-1000, 1000], [0, 0])


@BUTTONS
@pytest.mark.parametrize("all_in_street_script,expected_board_at_all_in", [
    (["c", "c", 990, "c"], 3),  # BB shoves flop, button calls
    (["c", "c", "c", "c", 990, "c"], 4),  # shove turn
    (["c", "c", "c", "c", "c", "c", 990, "c"], 5),  # shove river
])
def test_all_in_on_later_streets_runs_out_remaining_cards(
    button: int, all_in_street_script: list, expected_board_at_all_in: int
) -> None:
    deal = Deal.from_seed(31)
    e = new_engine(deal, button)
    evs = play_script(e, all_in_street_script[:-2])
    assert len(e.board) == expected_board_at_all_in and not e.is_complete
    evs += play_script(e, all_in_street_script[-2:])
    assert evs[-2].all_in and e.is_complete
    r = e.result()
    assert r.board == deal.board and r.ended_street == Street.RIVER and r.showdown
    assert r.total_pot == 2000
    assert [repr(c) for c in e._state.burn_cards] == list(deal.burns)


@BUTTONS
def test_all_in_then_fold_does_not_reveal_remaining_board(button: int) -> None:
    deal = Deal.from_seed(31)
    e = new_engine(deal, button)
    play_script(e, ["c", "c", 990, "f"])  # BB shoves the flop, button folds
    r = e.result()
    assert not r.showdown and r.board == deal.board[:3] and r.ended_street == Street.FLOP
    assert r.shown_hands == {} and r.winner == 1 - button and r.payoffs[1 - button] == 10
    assert r.total_pot == 20


@BUTTONS
def test_raise_when_opponent_is_all_in_posting_the_blind_is_not_offered(button: int) -> None:
    stacks = [0, 0]
    stacks[button], stacks[1 - button] = 1000, 10  # BB all-in just from the blind
    e = HandEngine(DEFAULT_CFG, DEAL_BTN_WINS, button, (stacks[0], stacks[1]))
    assert not e.legal_actions().can_raise and e.legal_actions().can_call


def test_stacks_just_above_call_allow_only_all_in_raise() -> None:
    # button 1000 raises to 100; BB behind has 95 -> BB's total 105: call 90 or shove to 105 (<min raise)
    e = new_engine(button_seat=0, stacks=(1000, 105))
    e.act(Action.raise_to(100))
    lg = e.legal_actions()
    assert lg.can_raise and lg.min_raise_to == lg.max_raise_to == 105 and lg.call_amount == 90
    e2 = new_engine(button_seat=0, stacks=(1000, 100))  # BB total 100 == bet: call is all-in, cannot raise
    e2.act(Action.raise_to(100))
    assert not e2.legal_actions().can_raise and e2.legal_actions().call_amount == 90


# =========================================================================== showdown & settlement
# name -> (board, winning hole, losing hole)
SHOWDOWNS = {
    "pair_beats_high_card": ("Kd8c4h2s9d", "AsAh", "QcJd"),
    "kicker_decides": ("As7d4c2h9s", "AhKd", "AcQd"),
    "flush_beats_straight": ("5h6h9hKc2d", "Ah3h", "7c8d"),
    "higher_flush_wins": ("2h7h9hKc3d", "AhJh", "Qh8h"),
    "full_house_beats_flush": ("9h9dKc4h2h", "KsKd", "Ah3h"),
    "wheel_beats_two_pair": ("2d3c4h9sKd", "Ah5c", "Kc9d"),
    "six_high_straight_beats_wheel": ("2d3c4h9sKd", "5s6s", "Ah5c"),
    "quads_beat_full_house": ("8h8d8cKdKs", "8s2c", "KhQc"),
    "trips_with_better_kicker": ("Qh6d6c2s9d", "6hAc", "6sKc"),
    "straight_flush_beats_quads": ("5h6h7h7d7c", "8h9h", "7s2c"),
}


def _deal_with_winner(name: str, winner_is_button: bool) -> Deal:
    board, win, lose = SHOWDOWNS[name]
    return make_deal(win, lose, board) if winner_is_button else make_deal(lose, win, board)


@pytest.mark.parametrize("name", list(SHOWDOWNS))
@pytest.mark.parametrize("button", [0, 1])
@pytest.mark.parametrize("winner_is_button", [True, False])
def test_showdown_winner_all_in_preflop(name: str, button: int, winner_is_button: bool) -> None:
    deal = _deal_with_winner(name, winner_is_button)
    e = HandEngine(DEFAULT_CFG, deal, button)
    play_script(e, [1000, "c"])
    r = e.result()
    winner = button if winner_is_button else 1 - button
    loser = 1 - winner
    assert r.winner == winner, f"{name}: PokerKit hole-card assignment must match the Deal layout"
    assert r.payoffs[winner] == 1000 and r.payoffs[loser] == -1000
    assert r.final_stacks[winner] == 2000 and r.final_stacks[loser] == 0
    assert r.showdown and r.total_pot == 2000 and r.board == deal.board and r.ended_street == Street.RIVER
    assert r.shown_hands[button] == deal.button_hole and r.shown_hands[1 - button] == deal.bb_hole


@pytest.mark.parametrize("name", list(SHOWDOWNS))
@pytest.mark.parametrize("winner_is_button", [True, False])
def test_showdown_winner_after_river_bet_and_call(name: str, winner_is_button: bool) -> None:
    deal = _deal_with_winner(name, winner_is_button)
    button = 0
    e = HandEngine(DEFAULT_CFG, deal, button, (700, 1000))
    play_script(e, ["c", "c", "c", "c", "c", "c", 50, "c"])  # BB bets 50 on the river, button calls
    r = e.result()
    winner = button if winner_is_button else 1
    assert r.winner == winner
    assert r.payoffs[winner] == 60 and r.payoffs[1 - winner] == -60  # 10 preflop + 50 river
    assert r.final_stacks == ((760, 940) if winner_is_button else (640, 1060))
    assert r.total_pot == 120


@pytest.mark.parametrize("button", [0, 1])
@pytest.mark.parametrize(
    "button_hole,bb_hole,board",
    [
        ("2c3d", "4h5c", "TsJsQdKhAd"),  # board plays: broadway on the board
        ("QcJd", "QhJs", "Ad9c5h2sKd"),  # identical ranks, different suits
        ("AcKc", "AhKh", "2d7s9d4cJh"),  # same hand, no flush for either
    ],
)
def test_split_pot_returns_each_player_their_chips(button: int, button_hole: str, bb_hole: str, board: str) -> None:
    deal = make_deal(button_hole, bb_hole, board)
    e = HandEngine(DEFAULT_CFG, deal, button, (1000, 640))
    play_script(e, [640, "c"])
    r = e.result()
    assert r.payoffs == (0, 0) and r.winner is None and r.showdown
    assert r.final_stacks == ((1000, 640))
    assert r.total_pot == 1280


@pytest.mark.parametrize("cfg", [GameConfig(101, 1, 3), GameConfig(99, 3, 7), GameConfig(55, 2, 5)])
def test_split_pot_with_odd_blinds_never_loses_a_chip(cfg: GameConfig) -> None:
    """Heads-up pots are always even (equal matched contributions), so ties split exactly.

    PokerKit gives the odd chip of an odd pot to the lowest-index winner, but with only two players
    and uncalled chips returned that situation cannot arise; this guards that assumption across many
    odd bets (the Broadway board makes every showdown a tie).
    """
    rng = random.Random(77)
    board = ("Ts", "Js", "Qd", "Kh", "Ad")
    rest = [c for c in ALL_CARDS if c not in board]
    ties = 0
    for i in range(150):
        h = rng.sample(rest, 7)
        deal = Deal((h[0], h[1], h[2], h[3], h[4], *board[:3], h[5], board[3], h[6], board[4]))
        stacks = (rng.randint(3, 120), rng.randint(3, 120))
        e = HandEngine(cfg, deal, i % 2, stacks)
        while not e.is_complete:
            e.act(random_action(rng, e))
        r = e.result()
        if r.showdown:
            ties += 1
            assert r.payoffs == (0, 0) and r.winner is None
            assert r.final_stacks == stacks
        else:
            assert sum(r.payoffs) == 0 and r.winner is not None
    assert ties > 20, "the test must actually reach showdowns"


# =========================================================================== results / views
def test_result_to_dict_and_spectator_view() -> None:
    deal = Deal.from_seed(3)
    e = new_engine(deal, 0)
    play_script(e, [30, "c", "c", "c", "c", "c", "c", "c"])
    view = e.spectator_view()
    assert view.complete and view.result == e.result() and view.actor_seat is None
    assert view.hole_cards == (deal.button_hole, deal.bb_hole)
    assert view.button_seat == 0 and len(view.events) == len(e.events)
    d = e.result().to_dict()
    assert d["payoffs"] == list(e.result().payoffs) and d["ended_street"] == int(Street.RIVER)
    assert set(d["shown_hands"]) == {"0", "1"}
    assert e.events[0].to_dict()["action"] == {"kind": "raise_to", "amount": 30}


def test_observation_values_each_street() -> None:
    """``my_committed`` / ``effective_stack`` / ``num_raises`` hand-checked along a hand."""
    e = new_engine(button_seat=0, stacks=(1000, 600))
    o = e.observation_for(0)
    assert (o.my_committed, o.opp_committed, o.my_bet, o.opp_bet, o.pot) == (5, 10, 5, 10, 15)
    assert (o.my_stack, o.opp_stack, o.effective_stack, o.num_raises, o.to_call) == (995, 590, 590, 0, 5)
    assert o.is_button and not e.observation_for(1).is_button
    e.act(Action.raise_to(30))
    o = e.observation_for(1)
    assert (o.my_committed, o.opp_committed, o.num_raises, o.effective_stack) == (10, 30, 1, 570)
    assert o.history[0].by_me is False and o.history[0].kind == "raise"
    e.act(Action.check_call())  # BB calls 20
    o = e.observation_for(1)  # flop, BB first to act
    assert o.street == Street.FLOP and (o.my_bet, o.opp_bet, o.pot) == (0, 0, 60)
    assert (o.my_committed, o.opp_committed) == (30, 30), "committed counts the whole hand"
    assert o.effective_stack == 570 and o.num_raises == 1 and o.num_raises_this_street == 0
    e.act(Action.raise_to(100))
    o = e.observation_for(0)
    assert (o.my_bet, o.opp_bet, o.my_committed, o.opp_committed, o.pot) == (0, 100, 30, 130, 160)
    assert o.num_raises == 2 and o.num_raises_this_street == 1 and o.effective_stack == 470
    e.act(Action.check_call())
    o = e.observation_for(1)  # turn
    assert o.street == Street.TURN and (o.my_committed, o.opp_committed, o.pot) == (130, 130, 260)
    assert o.num_raises == 2 and o.num_raises_this_street == 0 and o.effective_stack == 470


def test_effective_stack_is_never_negative_when_facing_an_overbet() -> None:
    e = new_engine(button_seat=0, stacks=(1000, 300))
    e.act(Action.raise_to(500))  # more than the BB can ever call
    assert e.observation_for(1).effective_stack == 0
    assert e.observation_for(0).effective_stack == 0


# =========================================================================== chip-conservation fuzz
CONFIGS = [
    GameConfig(1000, 5, 10),
    GameConfig(100, 1, 2),
    GameConfig(100, 1, 3),
    GameConfig(50, 2, 5),
]


def test_game_config_validation() -> None:
    assert GameConfig().stack_in_bb == 100
    assert GameConfig.from_dict(GameConfig(300, 2, 4).to_dict()) == GameConfig(300, 2, 4)
    for bad in [(1000, 0, 10), (1000, 10, 5), (1000, -1, 10), (19, 5, 10)]:
        with pytest.raises(ValueError):
            GameConfig(*bad)
    # equal blinds are rejected: PokerKit would let the big blind act first preflop heads-up
    with pytest.raises(ValueError):
        GameConfig(1000, 10, 10)


def expected_min_raise_to(e: HandEngine, start: tuple[int, int], cfg: GameConfig) -> int:
    """No-limit rule, derived from the event log alone: min raise = current bet + size of the last full bet/raise."""
    btn, street = e.button_seat, e.street
    bets = [0, 0]
    if street == Street.PREFLOP:
        bets[btn], bets[1 - btn] = min(cfg.small_blind, start[btn]), min(cfg.big_blind, start[1 - btn])
    max_bet, increment = max(bets), cfg.big_blind  # the big blind (or a street's minimum bet) is the first increment
    for ev in e.events:
        if ev.street != street or ev.action.kind.value != "raise_to":
            continue
        if ev.street_total - max_bet >= increment:  # a full raise re-sets the increment; short all-ins do not
            increment = ev.street_total - max_bet
        max_bet = max(max_bet, ev.street_total)
    actor = e.actor_seat
    assert actor is not None
    return min(e.stack(actor) + e.bet(actor), max_bet + increment)


def _check_live_state(
    e: HandEngine, start: tuple[int, int], deal: Deal, deep: bool = True, cfg: GameConfig | None = None
) -> None:
    assert e.pot == e.committed(0) + e.committed(1), "pot == chips committed"
    for s in (0, 1):
        assert e.stack(s) >= 0 and e.stack(s) + e.committed(s) == start[s]
    assert len(e.board) in (0, 3, 4, 5) and e.board == deal.board[: len(e.board)]
    assert BOARD_CARDS_BY_STREET[e.street] == len(e.board)
    seat = e.actor_seat
    assert seat is not None
    lg = e.legal_actions()
    assert lg.can_check == (lg.call_amount == 0) and lg.can_fold == (not lg.can_check)
    assert lg.can_call == (lg.call_amount > 0)
    opp = 1 - seat
    assert lg.call_amount == min(e.stack(seat), max(0, e.bet(opp) - e.bet(seat)))
    if lg.can_raise:
        if cfg is not None:
            assert lg.min_raise_to == expected_min_raise_to(e, start, cfg), "min-raise rule"
        assert 0 < lg.min_raise_to <= lg.max_raise_to == e.stack(seat) + e.bet(seat)
        assert lg.min_raise_to > e.bet(opp), "a raise must exceed the opponent's bet"
        assert e.stack(opp) > 0, "never offered a raise against an all-in opponent"
        assert lg.call_amount < e.stack(seat), "a call-all-in cannot be a raise"
    else:
        assert (lg.min_raise_to, lg.max_raise_to) == (0, 0)
    if deep:  # observation construction dominates the cost, so it is sampled in the big fuzz runs
        obs = e.observation_for(seat)
        assert obs.legal == lg and obs.effective_stack >= 0
        mask = obs.abstract_mask
        assert mask.any() and mask[1]  # CHECK_CALL always available to the actor


def _check_finished_hand(e: HandEngine, start: tuple[int, int], cfg: GameConfig, deal: Deal) -> None:
    r = e.result()
    assert len(e.events) <= MAX_ACTIONS_PER_HAND
    assert sum(r.payoffs) == 0
    assert r.final_stacks == (start[0] + r.payoffs[0], start[1] + r.payoffs[1])
    assert sum(r.final_stacks) == sum(start) and min(r.final_stacks) >= 0
    assert all(r.payoffs[s] >= -start[s] for s in (0, 1))
    assert r.board == deal.board[: len(r.board)] and BOARD_CARDS_BY_STREET[r.ended_street] == len(r.board)
    last_fold = bool(e.events) and e.events[-1].kind == "fold"
    assert r.showdown == (not last_fold)
    assert (sorted(r.shown_hands) == [0, 1]) == r.showdown
    if r.showdown:
        assert len(r.board) == 5, "a showdown always has the full board (runouts are completed)"
    # contested pot: posted blinds + all chips put in, uncalled excess returned
    btn = e.button_seat
    posted = [0, 0]
    posted[btn], posted[1 - btn] = min(cfg.small_blind, start[btn]), min(cfg.big_blind, start[1 - btn])
    contributed = [posted[s] + sum(ev.chips for ev in e.events if ev.seat == s) for s in (0, 1)]
    assert r.total_pot == 2 * min(contributed)
    positive = [s for s in (0, 1) if r.payoffs[s] > 0]
    assert r.winner == (positive[0] if positive else None)
    if r.winner is not None:
        assert r.payoffs[r.winner] == r.total_pot // 2 == -r.payoffs[1 - r.winner]
    else:
        assert r.payoffs == (0, 0)
    if last_fold:
        folder = e.events[-1].seat
        assert r.payoffs[folder] < 0 and r.winner == 1 - folder


def _check_event_log(e: HandEngine, start: tuple[int, int], cfg: GameConfig) -> None:
    """Independent re-derivation of kind / chips / all_in / pot_after / turn order from first principles."""
    btn = e.button_seat
    stack = list(start)
    street_bet = [0, 0]
    street_bet[btn] = min(cfg.small_blind, start[btn])
    street_bet[1 - btn] = min(cfg.big_blind, start[1 - btn])
    stack[btn] -= street_bet[btn]
    stack[1 - btn] -= street_bet[1 - btn]
    pot = sum(street_bet)
    prev_street, prev_seat = None, None
    for ev in e.events:
        if ev.street != prev_street:
            if prev_street is not None:
                assert ev.street > prev_street
                street_bet = [0, 0]
            expected_first = btn if ev.street == Street.PREFLOP else 1 - btn
            assert ev.seat == expected_first, f"first actor on {ev.street.name}"
        else:
            assert ev.seat != prev_seat, "turns strictly alternate within a street"
        prev_street, prev_seat = ev.street, ev.seat
        s, o = ev.seat, 1 - ev.seat
        if ev.action.kind.value == "fold":
            assert (ev.kind, ev.chips, ev.all_in) == ("fold", 0, False)
            assert street_bet[o] > street_bet[s], "folds only when facing a bet"
        elif ev.action.kind.value == "check_call":
            chips = min(stack[s], street_bet[o] - street_bet[s])
            assert ev.chips == chips and ev.kind == ("call" if chips else "check")
            assert ev.all_in == (chips > 0 and stack[s] == chips)
        else:
            chips = ev.action.amount - street_bet[s]
            assert ev.chips == chips > 0 and ev.street_total == ev.action.amount
            assert ev.kind == ("raise" if street_bet[o] > 0 else "bet")
            assert ev.all_in == (stack[s] == chips)
            assert ev.action.amount > street_bet[o]
        stack[s] -= ev.chips
        street_bet[s] += ev.chips
        pot += ev.chips
        assert ev.street_total == street_bet[s] and ev.pot_after == pot


def play_fuzz_hand(
    rng: random.Random, i: int, deep: bool = True
) -> tuple[HandEngine, tuple[int, int], GameConfig, Deal]:
    cfg = rng.choice(CONFIGS)
    stacks = random_stacks(rng) if rng.random() < 0.85 else (cfg.starting_stack, cfg.starting_stack)
    deal = random_deal(rng)
    e = HandEngine(cfg, deal, rng.randint(0, 1), stacks, i)
    n = 0
    while not e.is_complete:
        _check_live_state(e, stacks, deal, deep, cfg)
        e.act(random_action(rng, e))
        n += 1
        assert n <= MAX_ACTIONS_PER_HAND, "hand did not terminate"
    return e, stacks, cfg, deal


def run_fuzz(n_hands: int, seed: int, deep_every: int = 1) -> None:
    """Random legal play (half via the abstraction, half uniform concrete amounts) with full invariants."""
    rng = random.Random(seed)
    showdowns = folds = all_ins = 0
    for i in range(n_hands):
        e, stacks, cfg, deal = play_fuzz_hand(rng, i, deep=i % deep_every == 0)
        _check_finished_hand(e, stacks, cfg, deal)
        _check_event_log(e, stacks, cfg)
        r = e.result()
        showdowns += r.showdown
        folds += not r.showdown
        all_ins += any(ev.all_in for ev in e.events)
        if i % 25 == 0:  # replay determinism: same actions on a fresh engine => same result
            clone = HandEngine(cfg, deal, e.button_seat, stacks, i)
            for ev in e.events:
                clone.act(ev.action)
            assert clone.result() == r and clone.events == e.events
    assert showdowns > n_hands * 0.2 and folds > n_hands * 0.2 and all_ins > n_hands * 0.1, "fuzz coverage"


def test_chip_conservation_fuzz_quick() -> None:
    run_fuzz(400, seed=1)


@pytest.mark.slow
def test_chip_conservation_fuzz_5000_hands() -> None:
    run_fuzz(5000, seed=20240607, deep_every=5)


@pytest.mark.slow
def test_fuzz_tiny_stacks_only() -> None:
    """Both stacks within a few chips of the blinds: the all-in-on-the-blinds corner cases."""
    rng = random.Random(5)
    for i in range(600):
        cfg = rng.choice(CONFIGS)
        stacks = (rng.randint(1, 14), rng.randint(1, 14))
        deal = random_deal(rng)
        e = HandEngine(cfg, deal, i % 2, stacks, i)
        while not e.is_complete:
            _check_live_state(e, stacks, deal, deep=i % 3 == 0, cfg=cfg)
            e.act(random_action(rng, e))
        _check_finished_hand(e, stacks, cfg, deal)
        _check_event_log(e, stacks, cfg)


def test_throughput_is_reasonable() -> None:
    """Not a benchmark - just a guard against accidentally quadratic engine code."""
    import time

    rng = random.Random(0)
    start = time.perf_counter()
    n = 300
    for i in range(n):
        e = HandEngine(DEFAULT_CFG, random_deal(rng), i % 2)
        while not e.is_complete:
            e.act(random_action(rng, e))
    assert n / (time.perf_counter() - start) > 50, "engine slower than 50 hands/s"
