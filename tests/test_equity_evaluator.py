"""The vectorised evaluator must order hands exactly like PokerKit's StandardHighHand."""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np
import pytest
from pokerkit import Card, StandardHighHand
from pokerkit.lookups import Entry

from poker_arena.cards import ALL_CARDS, parse_cards
from poker_arena.equity.evaluator import (
    CATEGORY_NAMES,
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
    evaluate_batch,
    hand_category,
)

NUM_RANKS = 13
LOOKUP = StandardHighHand.lookup


def ints(text: str) -> list[int]:
    return [ALL_CARDS.index(card) for card in parse_cards(text)]


def pokerkit_entry(cards: list[int]) -> Entry:
    """Best 5-card PokerKit lookup entry among ``cards`` (any of 5, 6, 7 cards)."""
    parsed = list(Card.parse("".join(ALL_CARDS[c] for c in cards)))
    return max(LOOKUP.get_entry(combo) for combo in combinations(parsed, 5))


def score(text: str) -> int:
    return evaluate(ints(text))


# --------------------------------------------------------------------------- reference checks


def test_reference_helper_matches_pokerkit_from_game() -> None:
    """The fast lookup-based reference used below equals the documented from_game API."""
    rng = np.random.default_rng(0)
    for _ in range(150):
        cards = [int(c) for c in rng.permutation(52)[:7]]
        text = [ALL_CARDS[c] for c in cards]
        hole, board = "".join(text[:2]), "".join(text[2:])
        assert StandardHighHand.from_game(hole, board).entry == pokerkit_entry(cards)


def test_all_7462_five_card_classes_rank_exactly_like_pokerkit() -> None:
    """One representative per distinct 5-card hand class: same strict order, no collisions."""
    hands: list[list[int]] = []
    for ranks in combinations_with_replacement(range(NUM_RANKS), 5):
        if any(ranks.count(r) > 4 for r in set(ranks)) or ranks[0] == ranks[4]:
            continue
        hands.append([r * 4 + i % 4 for i, r in enumerate(ranks)])  # distinct suits per rank
        if len(set(ranks)) == 5:
            hands.append([r * 4 for r in ranks])  # all clubs: flush / straight flush
    reference = np.array([pokerkit_entry(h).index for h in hands])
    scores = evaluate_batch(np.array(hands))
    assert len(hands) == len(set(reference.tolist())) == 7462
    assert np.all(np.diff(scores[np.argsort(reference)]) > 0)


def clustered_hands(rng: np.random.Generator, n: int, size: int) -> np.ndarray:
    """Hands drawn from a few ranks/suits so straights, flushes, full houses are frequent."""
    hands = np.empty((n, size), dtype=np.int64)
    for i in range(n):
        while True:
            ranks = rng.choice(NUM_RANKS, size=rng.integers(5, 9), replace=False)
            suits = rng.choice(4, size=rng.integers(1, 5), replace=False)
            pool = (ranks[:, None] * 4 + suits[None, :]).ravel()
            if pool.size >= size:
                break
        hands[i] = rng.choice(pool, size=size, replace=False)
    return hands


def uniform_hands(rng: np.random.Generator, n: int, size: int) -> np.ndarray:
    return np.argsort(rng.random((n, 52)), axis=1)[:, :size].astype(np.int64)


@pytest.mark.parametrize(
    ("size", "n_uniform", "n_clustered"),
    [(7, 10_000, 10_000), (6, 3_000, 3_000), (5, 3_000, 3_000)],
)
def test_random_hands_compare_like_pokerkit(size: int, n_uniform: int, n_clustered: int) -> None:
    rng = np.random.default_rng(size)
    hands = np.vstack(
        [uniform_hands(rng, n_uniform, size), clustered_hands(rng, n_clustered, size)]
    )
    entries = [pokerkit_entry(h.tolist()) for h in hands]
    reference = np.array([e.index for e in entries])
    scores = evaluate_batch(hands)

    # Same category label as PokerKit for every hand ...
    assert [CATEGORY_NAMES[c] for c in hand_category(scores)] == [e.label.value for e in entries]
    # ... and identical (>, ==, <) outcome for many random pairs of hands.
    first = rng.integers(0, len(hands), size=200_000)
    second = rng.integers(0, len(hands), size=200_000)
    mine = np.sign(scores[first].astype(np.int64) - scores[second])
    assert np.array_equal(mine, np.sign(reference[first] - reference[second]))
    if size == 7:  # the stress mix must actually reach every category
        assert set(hand_category(scores).tolist()) == set(range(9))


# --------------------------------------------------------------------------- specific cases


@pytest.mark.parametrize(
    ("weaker", "stronger"),
    [
        # the wheel is the lowest straight; a six-high straight beats it
        ("Ah 2d 3c 4s 5h", "2d 3c 4s 5h 6c"),
        ("Ah 2d 3c 4s 5h", "Ts Jd Qc Ks Ah"),
        # ... but it is still a straight, so it beats trips
        ("Ah Ad Ac 4s 5h", "Ah 2d 3c 4s 5c"),
        # steel wheel is the lowest straight flush; six-high beats it; royal is the best
        ("Ah 2h 3h 4h 5h", "2h 3h 4h 5h 6h"),
        ("2h 3h 4h 5h 6h", "Th Jh Qh Kh Ah"),
        # a straight flush beats quads
        ("Ac Ad Ah As Kd", "Ah 2h 3h 4h 5h"),
        # flush kickers decide
        ("As Ks 9s 5s 2s", "As Ks 9s 5s 3s"),
        ("As Ks 9s 5s 3s", "As Ks Ts 5s 3s"),
        # straight < flush < full house < quads
        ("5h 6d 7c 8s 9h", "2h 5h 9h Jh Kh"),
        ("As Ks 9s 5s 2s", "2h 2d 2c 3s 3h"),
        ("Ac Ad Ah Ks Kd", "2c 2d 2h 2s 3h"),
        # full house tie-breaks: trips first, then the pair
        ("Qc Qd Qh As Ad", "Kc Kd Kh 2s 2d"),
        ("Ac Ad Ah 2s 2d", "Ac Ad Ah 3s 3d"),
        # quads kicker
        ("Ac Ad Ah As Qd", "Ac Ad Ah As Kd"),
        # two pair: high pair, then low pair, then kicker
        ("Kc Kd Qh Qs Ad", "Ac Ad 2h 2s Kd"),
        ("Ac Ad Kh Ks 2d", "Ac Ad Kh Ks 3d"),
        # pair / trips kickers and plain high cards
        ("Ac Ad Kh Qs 2d", "Ac Ad Kh Qs 3d"),
        ("7c 7d 7h Ks 2d", "7c 7d 7h Ks 3d"),
        ("Ac Kd Qh Js 8d", "Ac Kd Qh Js 9d"),
        # the bottom of the category ladder
        ("Ac Kd Qh Js 9d", "2c 2d 3h 4s 5d"),
        ("Ac Ad Kh Ks Qd", "2c 2d 2h 3s 4d"),
    ],
)
def test_hand_ordering_five_cards(weaker: str, stronger: str) -> None:
    assert score(weaker) < score(stronger)


@pytest.mark.parametrize(
    ("weaker", "stronger"),
    [
        # 7 cards: the best five are chosen, not the first five
        ("Ac Kd 2h 5s 9d Jc 3d", "Ac Kd 2h 5s 9d Jc Jd"),
        ("Ah 2d 3c 4s 5h Kd Qc", "2d 3c 4s 5h 6c Kd Qc"),
        # quads + trips: the kicker is the best remaining card
        ("4c 4d 4h 4s 3c 3d 3h", "4c 4d 4h 4s 3c 3d Kh"),
        ("4c 4d 4h 4s 3c 3d 3h", "4c 4d 4h 4s 5c 3d 3h"),
        # three pairs: the third pair's rank can be the kicker, but an ace kicker beats it
        ("7c 7d 5h 5s 3c 3d 2h", "7c 7d 5h 5s 3c 3d Ah"),
        # two trips make a full house; the better trips + best pair-or-trips win
        ("2c 2d 2h 3s 3c 3d 9h", "4c 4d 4h 3s 3c 9d 9h"),
        # a straight takes the highest run of a longer sequence
        ("5c 6d 7h 8s 9d 2c 2d", "5c 6d 7h 8s 9d Tc 2d"),
        ("5c 6d 7h 8s 9d Tc Jd", "Tc Jd Qh Ks Ad 2c 3d"),
        # a flush uses the best five suited cards
        ("2s 3s 4s 9s Ks Ad Ah", "2s 3s 4s 9s Ks Qs Ad"),
        # a straight flush hidden inside a flush beats a higher plain flush
        ("As Ks Qs 9s 2s 5s Jd", "2s 3s 4s 5s 6s As Kd"),
    ],
)
def test_hand_ordering_seven_cards(weaker: str, stronger: str) -> None:
    assert score(weaker) < score(stronger)


@pytest.mark.parametrize(
    ("hand_a", "hand_b"),
    [
        # three pairs: the best kicker among the remaining cards is used either way
        ("7c 7d 5h 5s 2c 3d 4h", "7c 7d 5h 5s 3c 3d 4h"),
        # two sets of trips: KKK + QQ whichever Q is dropped
        ("Kc Kd Kh Qs Qc 2d 3h", "Kc Kd Kh Qs Qc Qd 3h"),
        # only the best five cards of a six-card flush count
        ("As Ks Qs 9s 5s 2s Jd", "As Ks Qs 9s 5s 3s Jd"),
    ],
)
def test_equal_hands_tie(hand_a: str, hand_b: str) -> None:
    assert score(hand_a) == score(hand_b)


def test_wheel_is_lowest_straight_and_steel_wheel_lowest_straight_flush() -> None:
    wheel, six_high = score("Ah 2d 3c 4s 5h"), score("2d 3c 4s 5h 6c")
    assert hand_category(wheel) == hand_category(six_high) == STRAIGHT
    assert wheel < six_high
    # six cards that contain the wheel and a six: the six-high straight wins
    assert score("Ah 2d 3c 4s 5h 6c") == six_high
    steel = score("Ah 2h 3h 4h 5h")
    assert hand_category(steel) == STRAIGHT_FLUSH
    assert steel < score("2h 3h 4h 5h 6h")


def test_no_wraparound_straights() -> None:
    assert hand_category(score("Qh Kd Ac 2s 3h")) == HIGH_CARD
    assert hand_category(score("Kh Ad 2c 3s 4h")) == HIGH_CARD


def test_board_plays_and_exact_ties() -> None:
    royal_board = "Ts Js Qs Ks As"
    assert score(f"2c 3d {royal_board}") == score(f"4h 5h {royal_board}") == score(royal_board)
    # Ties ignore suits and card order.
    assert score("Ac Kd Qh Js 9d") == score("As Kc Qd Jh 9s") == score("9d Js Qh Kd Ac")
    # Different hole cards, same best five.
    assert score("2c 3d Ah Ad Ac Kc Qd") == score("4s 5s Ah Ad Ac Kc Qd")


def test_category_labels() -> None:
    expected = {
        "Ac Kd Qh Js 9d": HIGH_CARD,
        "Ac Ad Qh Js 9d": ONE_PAIR,
        "Ac Ad Qh Qs 9d": TWO_PAIR,
        "Ac Ad Ah Qs 9d": THREE_OF_A_KIND,
        "Ac Kd Qh Js Td": STRAIGHT,
        "Ac Kc Qc Jc 9c": FLUSH,
        "Ac Ad Ah Qs Qd": FULL_HOUSE,
        "Ac Ad Ah As Qd": FOUR_OF_A_KIND,
        "Ac Kc Qc Jc Tc": STRAIGHT_FLUSH,
    }
    for text, category in expected.items():
        assert hand_category(score(text)) == category
    assert CATEGORY_NAMES[STRAIGHT_FLUSH] == "Straight flush"
    assert len(CATEGORY_NAMES) == 9


# --------------------------------------------------------------------------- API behaviour


def test_scores_are_invariant_to_card_order_and_suit_relabelling() -> None:
    rng = np.random.default_rng(3)
    hands = clustered_hands(rng, 2_000, 7)
    base = evaluate_batch(hands)
    permuted = np.take_along_axis(hands, np.argsort(rng.random(hands.shape), axis=1), axis=1)
    assert np.array_equal(evaluate_batch(permuted), base)
    suit_map = np.array([2, 0, 3, 1])
    relabelled = (hands // 4) * 4 + suit_map[hands % 4]
    assert np.array_equal(evaluate_batch(relabelled), base)


def test_evaluate_batch_dtype_shape_and_input_types() -> None:
    hands = uniform_hands(np.random.default_rng(4), 500, 7)
    scores = evaluate_batch(hands)
    assert scores.shape == (500,)
    assert scores.dtype == np.int32
    for dtype in (np.int8, np.uint8, np.int32):
        assert np.array_equal(evaluate_batch(hands.astype(dtype)), scores)
    assert evaluate_batch(np.empty((0, 7), dtype=np.int64)).shape == (0,)
    assert evaluate(hands[0].tolist()) == scores[0]
    assert evaluate([ALL_CARDS[c] for c in hands[0]]) == scores[0]


def test_hand_category_scalar_and_array() -> None:
    scores = evaluate_batch(np.array([ints("Ac Kc Qc Jc Tc"), ints("2c 3d 4h 5s 7d")]))
    assert list(hand_category(scores)) == [STRAIGHT_FLUSH, HIGH_CARD]
    assert hand_category(int(scores[0])) == STRAIGHT_FLUSH


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 4), dtype=np.int64),
        np.zeros((3, 8), dtype=np.int64),
        np.zeros(7, dtype=np.int64),
        np.zeros((3, 7), dtype=np.float64),
        np.full((3, 7), 52, dtype=np.int64),
        np.full((3, 7), -1, dtype=np.int64),
    ],
)
def test_evaluate_batch_rejects_bad_input(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        evaluate_batch(bad)


def test_evaluate_rejects_duplicates_wrong_sizes_and_bad_cards() -> None:
    with pytest.raises(ValueError):
        evaluate(["Ah", "Ah", "Kd", "Qc", "Js"])
    with pytest.raises(ValueError):
        evaluate(["Ah", "Kd", "Qc", "Js"])
    with pytest.raises(ValueError):
        evaluate(["Ah", "Kd", "Qc", "Js", "Xx"])
