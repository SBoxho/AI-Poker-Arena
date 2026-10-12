"""Values and determinism of ``build_features`` on hand-picked states."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core_helpers import random_midhand_engines
from features_helpers import acting_observation, scripted_engine

from poker_arena.cards import parse_cards
from poker_arena.engine.observation import OpponentStats
from poker_arena.equity import equity_vs_random_with_se, exact_equity_vs_random, preflop_equity
from poker_arena.features import (
    EQUITY_SAMPLES_FLOP_TURN,
    FEATURE_NAMES,
    build_feature_dict,
    build_features,
    builder,
    hand_equity,
)

TESTS_DIR = Path(__file__).parent
MC_TOLERANCE = 4 * 0.5 / np.sqrt(EQUITY_SAMPLES_FLOP_TURN)  # four worst-case standard errors


def features_of(*args, **kwargs) -> dict[str, float]:
    return build_feature_dict(acting_observation(scripted_engine(*args, **kwargs)))


# ------------------------------------------------------------------------------- exact values
def test_preflop_small_blind_with_aces() -> None:
    f = features_of("AhAd", "7c2d", "Ks9h4c3d2s")
    assert f["equity"] == pytest.approx(preflop_equity(["Ah", "Ad"]), abs=1e-6)
    assert f["preflop_percentile"] > 0.99
    assert (f["is_pair"], f["is_suited"], f["hole_gap"], f["hole_high_rank"], f["hole_low_rank"]) == (1, 0, 0, 12, 12)
    assert (f["street"], f["is_button"], f["board_high_rank"]) == (0, 1, -1)
    assert f["pot_bb"] == pytest.approx(1.5) and f["to_call_bb"] == pytest.approx(0.5)
    assert f["my_stack_bb"] == pytest.approx(99.5) and f["opp_stack_bb"] == pytest.approx(99.0)
    assert f["effective_stack_bb"] == pytest.approx(99.0)
    assert f["spr"] == pytest.approx(99.0 * 10 / (15 + 5))
    assert f["pot_odds"] == pytest.approx(0.25)
    assert f["equity_minus_pot_odds"] == pytest.approx(f["equity"] - 0.25, abs=1e-6)
    assert f["bet_fraction_of_pot"] == pytest.approx(0.5)  # 5 to call into a pot of 10 before the call
    assert f["my_commit_frac"] == pytest.approx(0.005) and f["opp_commit_frac"] == pytest.approx(0.01)
    assert (f["facing_bet"], f["facing_all_in"], f["can_raise"]) == (1, 0, 1)
    assert (f["num_raises_street"], f["num_raises_hand"]) == (0, 0)


def test_big_blind_option_has_no_price_to_pay() -> None:
    f = features_of("AhAd", "7c2d", "Ks9h4c3d2s", ["c"], button_seat=0)
    assert f["is_button"] == 0 and f["facing_bet"] == 0
    assert f["to_call_bb"] == 0 and f["pot_odds"] == 0 and f["bet_fraction_of_pot"] == 0
    assert f["equity_minus_pot_odds"] == pytest.approx(f["equity"], abs=1e-6)


def test_flop_draw_board_texture_and_initiative() -> None:
    # Button raises to 20, big blind calls, big blind checks the flop: the button holds the nut flush draw.
    f = features_of("AhKh", "7c2d", "2h7h9sTc3d", [20, "c", "c"])
    assert (f["street"], f["is_button"], f["facing_bet"]) == (1, 1, 0)
    assert (f["flush_draw"], f["straight_outs"], f["overcards"], f["pair_with_board"]) == (1, 0, 2, 0)
    assert (f["board_max_suit_count"], f["board_paired"], f["board_high_rank"]) == (2, 0, 7)
    assert f["pot_bb"] == pytest.approx(4.0) and f["spr"] == pytest.approx(98.0 * 10 / 40)
    assert (f["i_have_initiative"], f["opp_has_initiative"]) == (1, 0)
    assert (f["i_raised_this_street"], f["opp_raised_this_street"]) == (0, 0)
    assert (f["num_raises_street"], f["num_raises_hand"]) == (0, 1)


def test_facing_a_bet_describes_its_size_and_who_is_aggressive() -> None:
    f = features_of("AhKh", "7c2d", "2h7h9sTc3d", [20, "c", "c", 30])  # big blind faces a 30-chip bet into 40
    assert f["is_button"] == 0 and f["facing_bet"] == 1
    assert f["pot_odds"] == pytest.approx(0.3) and f["bet_fraction_of_pot"] == pytest.approx(0.75)
    assert (f["opp_raised_this_street"], f["i_raised_this_street"]) == (1, 0)
    assert (f["opp_has_initiative"], f["i_have_initiative"]) == (1, 0)
    assert (f["num_raises_street"], f["num_raises_hand"]) == (1, 2)


def test_river_equity_is_exact_and_made_hand_is_two_pair() -> None:
    engine = scripted_engine("AhKh", "7c2d", "2h7h9sTc3d", [20, "c", "c", "c", "c", "c"])
    obs = acting_observation(engine)  # big blind at the river holds two pair
    f = build_feature_dict(obs)
    assert f["street"] == 3 and f["made_hand_category"] == 2
    assert f["equity"] == pytest.approx(exact_equity_vs_random(obs.my_hole, obs.board), abs=1e-6)
    assert (f["flush_draw"], f["straight_outs"]) == (0, 0)  # no cards to come


def test_all_in_call_is_flagged() -> None:
    obs = acting_observation(scripted_engine("AhKh", "7c2d", "2h7h9sTc3d", [400], stacks=(400, 1000)))  # button shoves
    f = build_feature_dict(obs)
    assert f["facing_all_in"] == 1 and f["facing_bet"] == 1
    assert f["call_stack_fraction"] == pytest.approx(f["to_call_bb"] * 10 / obs.my_stack)
    assert f["effective_stack_bb"] == 0


def test_opponent_statistics_are_passed_through() -> None:
    stats = OpponentStats(hands=99, vpip=0.8, pfr=0.6, aggression=0.7, fold_to_bet=0.2, showdown_rate=0.5)
    obs = acting_observation(scripted_engine("AhAd", "7c2d", "Ks9h4c3d2s"), stats)
    f = build_feature_dict(obs)
    assert (f["opp_vpip"], f["opp_pfr"], f["opp_aggression"]) == pytest.approx((0.8, 0.6, 0.7))
    assert (f["opp_fold_to_bet"], f["opp_showdown_rate"]) == pytest.approx((0.2, 0.5))
    assert f["opp_log_hands"] == pytest.approx(np.log(100.0), abs=1e-5)


def test_features_respond_to_the_own_cards() -> None:
    aces = build_features(acting_observation(scripted_engine("AhAd", "7c2d", "Ks9h4c3d2s")))
    trash = build_features(acting_observation(scripted_engine("7h2d", "AcAs", "Ks9h4c3d2s")))
    assert not np.array_equal(aces, trash)


# ---------------------------------------------------------------------------- equity feature
def test_flop_and_turn_equity_are_within_the_documented_standard_error() -> None:
    for script in ([20, "c"], [20, "c", "c", "c"]):  # flop, turn
        obs = acting_observation(scripted_engine("AhKh", "7c2d", "2h7h9sTc3d", script))
        exact = exact_equity_vs_random(obs.my_hole, obs.board)
        assert abs(hand_equity(obs) - exact) < MC_TOLERANCE


def test_documented_standard_error_bound() -> None:
    _, se = equity_vs_random_with_se(
        parse_cards("AhKh"), parse_cards("2h7h9s"), EQUITY_SAMPLES_FLOP_TURN, np.random.default_rng(0)
    )
    assert se <= 0.5 / np.sqrt(EQUITY_SAMPLES_FLOP_TURN)


# ------------------------------------------------------------------------------- determinism
def test_same_observation_always_gives_the_same_vector() -> None:
    observations = [e.observation_for(e.actor_seat) for e in random_midhand_engines(21, 60)]
    first = [build_features(o) for o in observations]
    builder._equity_vs_random.cache_clear()  # recompute everything, not just read the memo
    second = [build_features(o) for o in observations]
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_card_order_does_not_change_the_equity() -> None:
    first_obs = acting_observation(scripted_engine("AhKh", "7c2d", "2h7h9sTc3d", [20, "c"]))
    second_obs = acting_observation(scripted_engine("KhAh", "7c2d", "9s2h7hTc3d", [20, "c"]))  # same cards reordered
    assert set(first_obs.board) == set(second_obs.board) and first_obs.board != second_obs.board
    builder._equity_vs_random.cache_clear()
    first = hand_equity(first_obs)
    builder._equity_vs_random.cache_clear()
    assert hand_equity(second_obs) == first


def test_vectors_are_identical_in_another_process() -> None:
    code = (
        "import sys, hashlib; sys.path.insert(0, sys.argv[1])\n"
        "from core_helpers import random_midhand_engines\n"
        "from poker_arena.features import build_features\n"
        "h = hashlib.sha256()\n"
        "for e in random_midhand_engines(33, 80):\n"
        "    h.update(build_features(e.observation_for(e.actor_seat)).tobytes())\n"
        "print(h.hexdigest())\n"
    )
    here = hashlib.sha256()
    for engine in random_midhand_engines(33, 80):
        here.update(build_features(engine.observation_for(engine.actor_seat)).tobytes())
    result = subprocess.run([sys.executable, "-c", code, str(TESTS_DIR)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == here.hexdigest()


def test_feature_dict_covers_exactly_the_schema() -> None:
    f = features_of("AhAd", "7c2d", "Ks9h4c3d2s")
    assert tuple(f) == FEATURE_NAMES
