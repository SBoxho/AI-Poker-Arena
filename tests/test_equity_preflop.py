"""Precomputed preflop equity table: shape, sanity, lookups and regeneration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from poker_arena.cards import ALL_CARDS, RANKS
from poker_arena.equity import preflop
from poker_arena.equity.monte_carlo import equity_vs_random_with_se
from poker_arena.equity.preflop import (
    NUM_STARTING_HANDS,
    PREFLOP_CLASSES,
    TABLE_PATH,
    build_preflop_table,
    preflop_class,
    preflop_equity,
    preflop_rank_percentile,
)

MAX_STD_ERROR = 0.002
COMBOS = {n: 6 if len(n) == 2 else 4 if n[2] == "s" else 12 for n in PREFLOP_CLASSES}


@pytest.fixture(scope="module")
def stored() -> dict[str, Any]:
    return json.loads(TABLE_PATH.read_text(encoding="utf-8"))


def representative(name: str) -> list[str]:
    """Concrete hole cards for a class name such as ``"AKs"`` / ``"T9o"`` / ``"77"``."""
    first, second = name[0], name[1]
    if len(name) == 2:
        return [first + "c", second + "d"]
    return [first + "h", second + ("h" if name[2] == "s" else "c")]


def equity_of(name: str) -> float:
    return preflop_equity(representative(name))


# --------------------------------------------------------------------------- the stored table


def test_table_has_169_unique_well_formed_classes(stored: dict[str, Any]) -> None:
    assert len(PREFLOP_CLASSES) == len(set(PREFLOP_CLASSES)) == 169
    assert set(stored["equity"]) == set(PREFLOP_CLASSES) == set(stored["std_error"])
    pair = re.compile(rf"^([{RANKS}])\1$")
    other = re.compile(rf"^[{RANKS}][{RANKS}][so]$")
    for name in PREFLOP_CLASSES:
        assert pair.match(name) or other.match(name)
        if len(name) == 3:  # the higher card is written first
            assert RANKS.index(name[0]) > RANKS.index(name[1])
    assert sum(COMBOS.values()) == NUM_STARTING_HANDS == 1326


def test_table_metadata_and_precision(stored: dict[str, Any]) -> None:
    metadata = stored["metadata"]
    assert metadata["n_samples_per_class"] >= 100_000
    assert metadata["n_classes"] == 169
    assert {"seed", "date", "method"} <= set(metadata)
    assert max(stored["std_error"].values()) < MAX_STD_ERROR
    assert all(0.0 < value < 1.0 for value in stored["equity"].values())


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        ("AA", 0.845, 0.858),
        ("KK", 0.815, 0.830),
        ("22", 0.495, 0.510),
        ("AKs", 0.662, 0.682),
        ("AKo", 0.645, 0.662),
        ("72o", 0.338, 0.352),
        ("32o", 0.315, 0.330),
    ],
)
def test_known_equities_vs_random_hand(name: str, low: float, high: float) -> None:
    assert low < equity_of(name) < high


def test_extremes_of_the_table() -> None:
    values = {name: equity_of(name) for name in PREFLOP_CLASSES}
    assert max(values, key=lambda name: values[name]) == "AA"
    assert min(values, key=lambda name: values[name]) == "32o"


def test_equity_is_monotonic_in_pair_rank_and_kicker() -> None:
    slack = 3 * MAX_STD_ERROR  # neighbouring classes differ by more than the noise
    pairs = [equity_of(r + r) for r in reversed(RANKS)]  # AA, KK, ..., 22
    assert all(a > b + 0.005 for a, b in zip(pairs, pairs[1:], strict=False))
    for high in range(len(RANKS) - 1, 0, -1):
        for suffix in "so":
            names = [RANKS[high] + RANKS[low] + suffix for low in range(high - 1, -1, -1)]
            kickers = [equity_of(name) for name in names]
            assert all(a > b - slack for a, b in zip(kickers, kickers[1:], strict=False)), names


def test_suited_beats_offsuit_for_every_rank_combination() -> None:
    for high in range(1, 13):
        for low in range(high):
            name = RANKS[high] + RANKS[low]
            assert equity_of(name + "s") > equity_of(name + "o")


def test_table_agrees_with_fresh_simulation_using_other_suits() -> None:
    rng = np.random.default_rng(99)
    for hole, name in ((["Ks", "Qs"], "KQs"), (["9d", "9s"], "99"), (["Jc", "4h"], "J4o")):
        estimate, error = equity_vs_random_with_se(hole, [], 60_000, rng)
        assert abs(estimate - equity_of(name)) < 4 * np.hypot(error, MAX_STD_ERROR)


# --------------------------------------------------------------------------- lookups


@pytest.mark.parametrize(
    ("hole", "name"),
    [
        (["Ah", "Kh"], "AKs"),
        (["Kh", "Ah"], "AKs"),  # order does not matter
        (["Kd", "Ac"], "AKo"),
        (["7s", "7h"], "77"),
        (["Td", "9d"], "T9s"),
        (["2c", "Ad"], "A2o"),
        (["Ah", "As"], "AA"),
    ],
)
def test_preflop_class(hole: list[str], name: str) -> None:
    assert preflop_class(hole) == name
    assert preflop_class([ALL_CARDS.index(c) for c in hole]) == name  # ints work too


def test_every_two_card_combination_maps_to_the_right_class() -> None:
    counts: dict[str, int] = {}
    for a in range(52):
        for b in range(a + 1, 52):
            name = preflop_class([a, b])
            counts[name] = counts.get(name, 0) + 1
    assert counts == COMBOS


def test_equity_depends_only_on_the_class() -> None:
    assert preflop_equity(["Ah", "Kh"]) == preflop_equity(["As", "Ks"]) == equity_of("AKs")
    assert preflop_equity(["Ah", "Kd"]) == preflop_equity(["Ac", "Ks"]) == equity_of("AKo")


@pytest.mark.parametrize("hole", [["As"], ["As", "Kd", "Qh"], ["As", "As"], ["As", "Kx"], [52, 3]])
def test_invalid_hole_cards_raise(hole: list[str | int]) -> None:
    for function in (preflop_class, preflop_equity, preflop_rank_percentile):
        with pytest.raises(ValueError):
            function(hole)


# --------------------------------------------------------------------------- percentile


def test_percentile_is_in_unit_interval_and_follows_equity() -> None:
    percentiles = np.array([preflop_rank_percentile(representative(n)) for n in PREFLOP_CLASSES])
    equities = np.array([equity_of(n) for n in PREFLOP_CLASSES])
    assert np.all((percentiles >= 0.0) & (percentiles <= 1.0))
    assert np.all(np.diff(percentiles[np.argsort(equities)]) >= 0)
    assert percentiles.min() == 0.0  # nothing is weaker than 32o
    assert percentiles.max() == pytest.approx(1 - 6 / 1326)  # AA beats all but its own combos


def test_percentile_is_combo_weighted_fraction_of_weaker_hands() -> None:
    equities = {name: equity_of(name) for name in PREFLOP_CLASSES}
    for name in ("AA", "AKs", "T9o", "72o", "55"):
        weaker = sum(COMBOS[other] for other in PREFLOP_CLASSES if equities[other] < equities[name])
        assert preflop_rank_percentile(representative(name)) == pytest.approx(weaker / 1326)


# --------------------------------------------------------------------------- regeneration


def test_build_preflop_table_is_reproducible_and_leaves_shipped_file_alone(tmp_path: Path) -> None:
    shipped = TABLE_PATH.read_bytes()
    out = tmp_path / "table.json"
    first = build_preflop_table(3_000, seed=5, output_path=out)
    second = build_preflop_table(3_000, seed=5, output_path=None)
    assert first["equity"] == second["equity"]
    assert json.loads(out.read_text(encoding="utf-8")) == first
    assert first["metadata"]["n_samples_per_class"] == 3_000
    assert first["metadata"]["seed"] == 5
    assert len(first["equity"]) == 169
    assert first["equity"]["AA"] > first["equity"]["72o"]
    assert build_preflop_table(3_000, seed=6, output_path=None)["equity"] != first["equity"]
    assert TABLE_PATH.read_bytes() == shipped


def test_command_line_entry_point(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli.json"
    preflop.main(["--samples", "500", "--seed", "1", "--output", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["metadata"]["seed"] == 1
    assert "169 classes" in capsys.readouterr().out
