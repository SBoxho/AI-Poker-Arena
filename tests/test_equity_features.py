"""Observable-only hand/board descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from poker_arena.cards import ALL_CARDS, parse_cards
from poker_arena.equity.evaluator import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    evaluate,
    hand_category,
)
from poker_arena.equity.features import (
    board_texture,
    draw_features,
    hole_card_features,
    made_hand_category,
)

ACE, KING, QUEEN, JACK, TEN = 12, 11, 10, 9, 8


def cards(text: str) -> list[str]:
    return list(parse_cards(text))


# --------------------------------------------------------------------------- hole cards


@pytest.mark.parametrize(
    ("hole", "expected"),
    [
        ("AsKs", dict(high_rank=ACE, low_rank=KING, is_pair=0, is_suited=1, gap=1)),
        ("KdAc", dict(high_rank=ACE, low_rank=KING, is_pair=0, is_suited=0, gap=1)),
        ("7c7d", dict(high_rank=5, low_rank=5, is_pair=1, is_suited=0, gap=0)),
        ("2cAd", dict(high_rank=ACE, low_rank=0, is_pair=0, is_suited=0, gap=12)),
        ("Jh9h", dict(high_rank=JACK, low_rank=7, is_pair=0, is_suited=1, gap=2)),
    ],
)
def test_hole_card_features(hole: str, expected: dict[str, int]) -> None:
    assert hole_card_features(cards(hole)) == expected


def test_hole_card_features_accept_ints_and_reject_bad_hands() -> None:
    assert hole_card_features([ALL_CARDS.index("As"), ALL_CARDS.index("Ks")]) == hole_card_features(
        ["As", "Ks"]
    )
    for bad in (["As"], ["As", "As"], ["As", "Kd", "Qc"]):
        with pytest.raises(ValueError):
            hole_card_features(bad)


# --------------------------------------------------------------------------- made hand


@pytest.mark.parametrize(
    ("hole", "board", "category"),
    [
        ("AsKd", "2c7h9d", HIGH_CARD),
        ("AsKd", "Ac7h9d", ONE_PAIR),
        ("AsKd", "Ac7hKc", TWO_PAIR),
        ("AsAd", "Ac7h9d", THREE_OF_A_KIND),
        ("9s8d", "7c6hTd", STRAIGHT),
        ("AsKs", "2s7s9s", FLUSH),
        ("AsAd", "AcKhKd", FULL_HOUSE),
        ("AsAd", "AcAh9d", FOUR_OF_A_KIND),
        ("AsKs", "QsJsTs", STRAIGHT_FLUSH),
        ("AsKd", "2c7h9d4s", HIGH_CARD),  # turn
        ("AsKd", "2c7h9dKs3c", ONE_PAIR),  # river
        ("2c3d", "AsKsQsJsTs", STRAIGHT_FLUSH),  # the board plays
    ],
)
def test_made_hand_category(hole: str, board: str, category: int) -> None:
    assert made_hand_category(cards(hole), cards(board)) == category


def test_made_hand_category_preflop_is_a_pair_indicator() -> None:
    assert made_hand_category(cards("7c7d"), []) == ONE_PAIR
    assert made_hand_category(cards("AsKs"), []) == HIGH_CARD


def test_made_hand_category_agrees_with_evaluator_on_random_deals() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        deal = [ALL_CARDS[c] for c in rng.permutation(52)[: 2 + int(rng.choice([3, 4, 5]))]]
        expected = hand_category(evaluate(deal))
        assert made_hand_category(deal[:2], deal[2:]) == expected


def test_made_hand_category_rejects_partial_boards_and_duplicates() -> None:
    for board in (["2c"], ["2c", "3d"]):
        with pytest.raises(ValueError):
            made_hand_category(cards("AsKd"), board)
    with pytest.raises(ValueError):
        made_hand_category(cards("AsKd"), cards("As7h9d"))


# --------------------------------------------------------------------------- board texture


def test_board_texture_empty_board() -> None:
    assert board_texture([]) == {
        "n_board_cards": 0,
        "board_paired": 0,
        "board_max_suit_count": 0,
        "board_straightness": 0,
        "board_high_rank": -1,
    }


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        ("Ks7d2c", dict(n_board_cards=3, board_paired=0, board_max_suit_count=1,
                        board_straightness=1, board_high_rank=KING)),
        ("KsKd7s", dict(n_board_cards=3, board_paired=1, board_max_suit_count=2,
                        board_straightness=1, board_high_rank=KING)),
        ("Ah8h3h", dict(n_board_cards=3, board_paired=0, board_max_suit_count=3,
                        board_straightness=2, board_high_rank=ACE)),
        ("5h6d7c", dict(n_board_cards=3, board_paired=0, board_max_suit_count=1,
                        board_straightness=3, board_high_rank=5)),
        ("Ah2d3c", dict(n_board_cards=3, board_paired=0, board_max_suit_count=1,
                        board_straightness=3, board_high_rank=ACE)),  # wheel window
        ("2c3d4h5s", dict(n_board_cards=4, board_paired=0, board_max_suit_count=1,
                          board_straightness=4, board_high_rank=3)),
        ("AsKsQsJsTs", dict(n_board_cards=5, board_paired=0, board_max_suit_count=5,
                            board_straightness=5, board_high_rank=ACE)),
        ("9s9d9h9c2d", dict(n_board_cards=5, board_paired=1, board_max_suit_count=2,
                            board_straightness=1, board_high_rank=7)),
    ],
)
def test_board_texture(board: str, expected: dict[str, int]) -> None:
    assert board_texture(cards(board)) == expected


def test_board_texture_validation() -> None:
    for bad in (["2c"], ["2c", "3d"], ["2c", "2c", "3d"], ["2c", "3d", "Zz"]):
        with pytest.raises(ValueError):
            board_texture(bad)


# --------------------------------------------------------------------------- draws


def draws(hole: str, board: str) -> dict[str, int]:
    return draw_features(cards(hole), cards(board))


def test_draw_features_empty_board_is_all_zero() -> None:
    assert draws("AsKd", "") == dict(
        flush_draw=0, straight_outs=0, overcards=0, has_pair_with_board=0
    )


@pytest.mark.parametrize(
    ("hole", "board", "flush_draw"),
    [
        ("AhKh", "2h7c9h", 1),  # two hole hearts + two on board
        ("Ah2d", "7h9hKh", 1),  # one hole heart + three on board
        ("AhKh", "2h7h9h", 0),  # already a flush
        ("AsKd", "2h7h9hJh", 0),  # four on board, none in hand: not hero's draw
        ("AhKh", "2h7c9d", 0),  # three to a flush only
        ("AhKh", "2h7h9dJc", 1),  # turn flush draw
        ("AhKh", "2h7h9dJcTs", 0),  # river: nothing to draw to
        ("AhKh", "2h7h9hJh", 0),  # made flush on the turn
    ],
)
def test_flush_draw(hole: str, board: str, flush_draw: int) -> None:
    assert draws(hole, board)["flush_draw"] == flush_draw


@pytest.mark.parametrize(
    ("hole", "board", "outs"),
    [
        ("8h9d", "6c7sKh", 8),  # open-ended: 5 or T
        ("8h9d", "6cTsKh", 4),  # gutshot: 7
        ("Ah2d", "3c4sKh", 4),  # wheel draw: only a five completes it
        ("AhKd", "QcJs2h", 4),  # broadway gutshot: only a ten completes it
        ("5h6d", "7c8sKh", 8),  # open-ended: 4 or 9
        ("8h9d", "6c7sTh", 0),  # already a straight
        ("KcKd", "5h6s7d8c", 0),  # straight draw is entirely on the board
        ("Tc2d", "5h6s7d8c", 4),  # a 9 makes 6-T with the hole ten (the board draw alone is 0)
        ("AsKd", "2c7h9d", 0),
        ("8h9d", "6c7sKh2d3c", 0),  # river
    ],
)
def test_straight_outs_examples(hole: str, board: str, outs: int) -> None:
    assert draws(hole, board)["straight_outs"] == outs


def reference_straight_outs(hole: list[int], board: list[int]) -> int:
    """Independent set-based reimplementation of the documented ``straight_outs``."""
    held = {c // 4 for c in hole + board}
    hole_ranks = {c // 4 for c in hole}
    windows = [{12, 0, 1, 2, 3}] + [set(range(low, low + 5)) for low in range(9)]
    if any(window <= held for window in windows):
        return 0
    completing = {
        rank
        for rank in range(13)
        if rank not in held
        for window in windows
        if rank in window and window - {rank} <= held and window & hole_ranks
    }
    return 4 * len(completing)


def test_straight_outs_match_independent_reference_on_random_flops_and_turns() -> None:
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(4_000):
        deal = [int(c) for c in rng.permutation(52)[: 2 + int(rng.choice([3, 4]))]]
        outs = draw_features(deal[:2], deal[2:])["straight_outs"]
        assert outs == reference_straight_outs(deal[:2], deal[2:])
        seen.add(outs)
    assert seen == {0, 4, 8}  # gutshots, open-enders and nothing - never anything else


def test_straight_outs_really_complete_a_straight() -> None:
    """Every out counted makes a straight-or-better using a hole card; the others do not."""
    hole, board = cards("8h9d"), cards("6c7sKh")
    outs = [c for c in ALL_CARDS if c not in hole + board]
    completing = [c for c in outs if hand_category(evaluate(hole + board + [c])) >= STRAIGHT]
    assert len(completing) == draw_features(hole, board)["straight_outs"] == 8


@pytest.mark.parametrize(
    ("hole", "board", "overcards", "pair"),
    [
        ("AsKd", "2c7h9d", 2, 0),
        ("AsQd", "Kc7h9d", 1, 0),
        ("QsJd", "Kc7h9d", 0, 0),
        ("7sJd", "Kc7h9d", 0, 1),
        ("KsKd", "Kc7h9d", 0, 1),
        ("AsKd", "2c7h9d4s", 2, 0),
        ("AsKd", "2c7h9dKs3c", 1, 1),
    ],
)
def test_overcards_and_pair_with_board(hole: str, board: str, overcards: int, pair: int) -> None:
    features = draws(hole, board)
    assert features["overcards"] == overcards
    assert features["has_pair_with_board"] == pair


def test_river_has_no_draws_but_keeps_relative_features() -> None:
    features = draws("AhKh", "2h7h9dJc3h")
    assert features == dict(flush_draw=0, straight_outs=0, overcards=2, has_pair_with_board=0)


def test_features_are_plain_ints_and_order_independent() -> None:
    hole, board = cards("8h9d"), cards("6c7sKh")
    for features in (
        hole_card_features(hole),
        board_texture(board),
        draw_features(hole, board),
    ):
        assert all(type(value) is int for value in features.values())
    assert draw_features(hole[::-1], board[::-1]) == draw_features(hole, board)
    assert board_texture(board[::-1]) == board_texture(board)
    assert hole_card_features(hole[::-1]) == hole_card_features(hole)


def test_every_feature_dict_has_stable_keys() -> None:
    boards = ("", "2c7h9d", "2c7h9d4s", "2c7h9d4s5s")
    keys = {tuple(draw_features(cards("AsKd"), cards(board))) for board in boards}
    assert len(keys) == 1
    assert len({tuple(board_texture(cards(b))) for b in ("", "2c7h9d")}) == 1


def test_draw_features_validation() -> None:
    with pytest.raises(ValueError):
        draw_features(cards("AsKd"), cards("As7h9d"))  # board repeats a hole card
    with pytest.raises(ValueError):
        draw_features(cards("As"), cards("2c7h9d"))
    with pytest.raises(ValueError):
        draw_features(cards("AsKd"), cards("2c7h"))
