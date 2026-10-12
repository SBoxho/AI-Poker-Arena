"""No-leak property test for the feature vector (differential, using the real ``HandEngine``).

Two engines play the *same public history*; their decks differ only in cards the observing player
cannot know at that moment (the opponent's hole cards, the burn cards, the undealt board). The
feature vectors of the observing player must be bit-identical throughout.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from core_helpers import DEFAULT_CFG, random_action, random_deal, random_stacks

from poker_arena.cards import ALL_CARDS
from poker_arena.config import derive_seed
from poker_arena.engine.deals import Deal
from poker_arena.engine.game import HandEngine
from poker_arena.features import build_features

# Positions inside Deal.cards = button hole (0,1), bb hole (2,3), burn, flop (5,6,7), burn, turn (9), burn, river (11)
BUTTON_HOLE, BB_HOLE = (0, 1), (2, 3)
REVEALED_BY_BOARD_LENGTH = {0: (), 3: (5, 6, 7), 4: (5, 6, 7, 9), 5: (5, 6, 7, 9, 11)}
DEALS_PER_CASE = 40


def hidden_variant(rng: random.Random, deal: Deal, hero_is_button: bool, board_length: int) -> Deal:
    """Same hero cards and same board up to ``board_length``; everything else re-drawn."""
    known = set(BUTTON_HOLE if hero_is_button else BB_HOLE) | set(REVEALED_BY_BOARD_LENGTH[board_length])
    kept = {deal.cards[i] for i in known}
    pool = [card for card in ALL_CARDS if card not in kept]
    rng.shuffle(pool)
    cards = [deal.cards[i] if i in known else pool.pop() for i in range(len(deal.cards))]
    return Deal(tuple(cards))


@pytest.mark.parametrize("hero", [0, 1])
@pytest.mark.parametrize("board_length", [0, 3, 4, 5])
def test_hidden_cards_never_change_the_features(hero: int, board_length: int) -> None:
    rng = random.Random(derive_seed("test", "feature-leak", hero, board_length))
    compared = 0
    for hand in range(DEALS_PER_CASE):
        button = rng.randint(0, 1)
        deal = random_deal(rng)
        stacks = random_stacks(rng)
        variant = hidden_variant(rng, deal, hero_is_button=(hero == button), board_length=board_length)
        a = HandEngine(DEFAULT_CFG, deal, button, stacks, hand)
        b = HandEngine(DEFAULT_CFG, variant, button, stacks, hand)
        while not a.is_complete:
            if a.board != b.board:  # a street beyond what the variant shares has been revealed
                break
            va, vb = build_features(a.observation_for(hero)), build_features(b.observation_for(hero))
            assert np.array_equal(va, vb), f"hand {hand}: hidden cards changed the features"
            compared += 1
            action = random_action(rng, a)
            a.act(action)
            b.act(action)
    assert compared >= DEALS_PER_CASE  # the property was actually exercised


def test_the_property_test_would_notice_a_leak() -> None:
    """Guard against a vacuous test: features of different *visible* information do differ."""
    rng = random.Random(7)
    deal = random_deal(rng)
    other = Deal((*deal.cards[2:4], *deal.cards[0:2], *deal.cards[4:]))  # hero (button) gets the other player's cards
    a, b = HandEngine(DEFAULT_CFG, deal, 0), HandEngine(DEFAULT_CFG, other, 0)
    assert not np.array_equal(build_features(a.observation_for(0)), build_features(b.observation_for(0)))
