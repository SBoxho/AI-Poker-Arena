"""Card encoding and 12-card ``Deal`` contract."""

from __future__ import annotations

import random

import pytest

from poker_arena.cards import (
    ALL_CARDS,
    NUM_CARDS,
    RANKS,
    SUITS,
    assert_no_duplicates,
    card_to_int,
    card_to_str,
    cards_to_ints,
    cards_to_strs,
    parse_cards,
    rank_of,
    shuffled_deck,
    suit_of,
)
from poker_arena.engine.deals import DEAL_SIZE, Deal

from core_helpers import make_deal


# ---------------------------------------------------------------------------- cards
def test_universe_is_52_unique_cards_in_documented_order() -> None:
    assert NUM_CARDS == 52 == len(ALL_CARDS) == len(set(ALL_CARDS))
    assert ALL_CARDS[0] == "2c" and ALL_CARDS[1] == "2d" and ALL_CARDS[3] == "2s" and ALL_CARDS[51] == "As"
    assert RANKS == "23456789TJQKA" and SUITS == "cdhs"


def test_int_str_round_trip_for_every_card() -> None:
    for i in range(NUM_CARDS):
        s = card_to_str(i)
        assert card_to_int(s) == i
        assert ALL_CARDS[i] == s
        assert i == rank_of(i) * 4 + suit_of(i)
        assert RANKS[rank_of(i)] == s[0] and SUITS[suit_of(i)] == s[1]
        # idempotent on the already-converted representation
        assert card_to_int(i) == i and card_to_str(s) == s


def test_rank_and_suit_examples() -> None:
    assert rank_of("2c") == 0 and rank_of("Ah") == 12 and rank_of("Td") == 8
    assert suit_of("Ac") == 0 and suit_of("Ad") == 1 and suit_of("Ah") == 2 and suit_of("As") == 3


@pytest.mark.parametrize("bad", ["", "A", "Ahh", "1h", "ah", "AH", "Ax", "10h", " Ah", "Jokr"])
def test_invalid_card_strings_raise(bad: str) -> None:
    with pytest.raises(ValueError):
        card_to_int(bad)
    with pytest.raises(ValueError):
        card_to_str(bad)


@pytest.mark.parametrize("bad", [-1, 52, 1000])
def test_out_of_range_ints_raise(bad: int) -> None:
    with pytest.raises(ValueError):
        card_to_int(bad)
    with pytest.raises(ValueError):
        card_to_str(bad)
    with pytest.raises(ValueError):
        rank_of(bad)


def test_bulk_conversion_and_parse() -> None:
    assert cards_to_ints(["2c", "As"]) == (0, 51)
    assert cards_to_strs([0, 51, "Kd"]) == ("2c", "As", "Kd")
    assert parse_cards("AsKd") == ("As", "Kd")
    assert parse_cards("As Kd") == ("As", "Kd")
    assert parse_cards("As, Kd") == ("As", "Kd")
    assert parse_cards("") == ()
    with pytest.raises(ValueError):
        parse_cards("AsK")  # odd length
    with pytest.raises(ValueError):
        parse_cards("AsZz")


def test_assert_no_duplicates() -> None:
    assert_no_duplicates(["As", "Kd", 3])
    with pytest.raises(ValueError, match="Duplicate"):
        assert_no_duplicates(["As", "Kd", "As"])
    with pytest.raises(ValueError, match="Duplicate"):
        assert_no_duplicates(["2c", 0])  # same card in both representations


def test_shuffled_deck_is_a_seeded_permutation() -> None:
    a = shuffled_deck(random.Random(7))
    b = shuffled_deck(random.Random(7))
    c = shuffled_deck(random.Random(8))
    assert a == b and a != c
    assert sorted(a) == sorted(ALL_CARDS)


# ---------------------------------------------------------------------------- deals
def test_deal_layout_positions() -> None:
    """Dealing order: button hole, BB hole, burn, flop x3, burn, turn, burn, river."""
    cards = ("As", "Ks", "Qd", "Jd", "2c", "3c", "4c", "5c", "6c", "7c", "8c", "9c")
    deal = Deal(cards)
    assert deal.button_hole == ("As", "Ks")
    assert deal.bb_hole == ("Qd", "Jd")
    assert deal.burns == ("2c", "6c", "8c")
    assert deal.board == ("3c", "4c", "5c", "7c", "9c")
    assert deal.to_list() == list(cards)
    assert Deal.from_list(deal.to_list()) == deal


def test_deal_positions_are_a_partition_of_the_12_cards() -> None:
    deal = Deal.from_seed(123)
    parts = [*deal.button_hole, *deal.bb_hole, *deal.burns, *deal.board]
    assert len(parts) == DEAL_SIZE == 12
    assert sorted(parts) == sorted(deal.cards)


def test_deal_normalises_int_cards_to_strings() -> None:
    deal = Deal(tuple(range(12)))
    assert deal.cards == ALL_CARDS[:12]


@pytest.mark.parametrize("n", [0, 1, 11, 13, 52])
def test_deal_requires_exactly_12_cards(n: int) -> None:
    with pytest.raises(ValueError, match="exactly"):
        Deal(tuple(ALL_CARDS[:n]))


def test_deal_rejects_duplicates_and_invalid_cards() -> None:
    cards = list(ALL_CARDS[:12])
    cards[11] = cards[0]
    with pytest.raises(ValueError, match="Duplicate"):
        Deal(tuple(cards))
    cards[11] = "Zz"
    with pytest.raises(ValueError):
        Deal(tuple(cards))


def test_deal_is_frozen() -> None:
    deal = Deal.from_seed(1)
    with pytest.raises(AttributeError):
        deal.cards = ()  # type: ignore[misc]


def test_from_seed_is_deterministic_and_seed_sensitive() -> None:
    assert Deal.from_seed(42) == Deal.from_seed(42)
    assert Deal.from_seed(42) != Deal.from_seed(43)
    assert Deal.from_rng(random.Random(5)) == Deal.from_seed(5)


def test_from_seed_always_yields_valid_unique_deals() -> None:
    for seed in range(2000):
        cards = Deal.from_seed(seed).cards
        assert len(set(cards)) == 12
        assert set(cards) <= set(ALL_CARDS)


def test_from_seed_card_frequencies_are_roughly_uniform() -> None:
    """Sanity check on the dealing: every card appears in about 12/52 of the deals."""
    counts = dict.fromkeys(ALL_CARDS, 0)
    n = 4000
    for seed in range(n):
        for c in Deal.from_seed(seed).cards:
            counts[c] += 1
    expected = n * 12 / 52  # ~923
    assert all(abs(v - expected) < 0.2 * expected for v in counts.values())


def test_make_deal_helper_matches_layout() -> None:
    deal = make_deal("AhKh", "2c3c", "QsJsTd5c2d")
    assert deal.button_hole == ("Ah", "Kh") and deal.bb_hole == ("2c", "3c")
    assert deal.board == ("Qs", "Js", "Td", "5c", "2d")
    assert len(set(deal.burns) & set(deal.board + deal.button_hole + deal.bb_hole)) == 0
