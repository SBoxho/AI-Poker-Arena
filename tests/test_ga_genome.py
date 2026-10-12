"""Genome: bounds, validation, conversions, presets and the readable strategy card."""

from __future__ import annotations

import json

import numpy as np
import pytest

from poker_arena.agents.ga_agent import (
    GENE_HIGHS,
    GENE_LOWS,
    GENE_NAMES,
    GENE_SPECS,
    HELD_OUT_PRESETS,
    PRESETS,
    TRAINING_PRESETS,
    Genome,
)


def test_gene_specs_are_documented_and_consistent() -> None:
    assert 12 <= len(GENE_SPECS) <= 18
    assert len(set(GENE_NAMES)) == len(GENE_NAMES)
    for spec in GENE_SPECS:
        assert spec.low < spec.high
        assert spec.low <= spec.default <= spec.high
        assert len(spec.description) > 20
        assert spec.group
    assert np.all(GENE_LOWS < GENE_HIGHS)
    assert list(GENE_NAMES) == [spec.name for spec in GENE_SPECS]


def test_default_genome_is_valid_and_vector_round_trip() -> None:
    genome = Genome()
    vector = genome.to_vector()
    assert vector.shape == (len(GENE_NAMES),)
    assert Genome.from_vector(vector) == genome


def test_out_of_bounds_and_non_finite_genes_are_rejected() -> None:
    spec = GENE_SPECS[0]
    with pytest.raises(ValueError, match=spec.name):
        Genome(**{spec.name: spec.high + 0.01})
    with pytest.raises(ValueError, match=spec.name):
        Genome(**{spec.name: spec.low - 0.01})
    with pytest.raises(ValueError):
        Genome(**{spec.name: float("nan")})
    with pytest.raises(ValueError):
        Genome(**{spec.name: "high"})


def test_from_vector_clip_projects_into_bounds_and_checks_shape() -> None:
    wild = np.where(np.arange(len(GENE_NAMES)) % 2 == 0, 1e6, -1e6)
    with pytest.raises(ValueError):
        Genome.from_vector(wild)
    clipped = Genome.from_vector(wild, clip=True).to_vector()
    assert np.all(clipped >= GENE_LOWS) and np.all(clipped <= GENE_HIGHS)
    with pytest.raises(ValueError, match="gene values"):
        Genome.from_vector(np.zeros(3))
    with pytest.raises(ValueError):
        Genome.from_vector(np.full(len(GENE_NAMES), np.nan), clip=True)


def test_json_round_trip_is_exact() -> None:
    genome = Genome.random(np.random.default_rng(3))
    restored = Genome.from_dict(json.loads(json.dumps(genome.to_dict())))
    assert restored == genome
    assert list(genome.to_dict()) == list(GENE_NAMES)


def test_from_dict_rejects_missing_and_unknown_genes() -> None:
    genes = Genome().to_dict()
    missing = {k: v for k, v in genes.items() if k != "value_bet"}
    with pytest.raises(ValueError, match="value_bet"):
        Genome.from_dict(missing)
    with pytest.raises(ValueError, match="nonsense"):
        Genome.from_dict({**genes, "nonsense": 1.0})


def test_random_genomes_are_in_bounds_and_seeded() -> None:
    first = [Genome.random(np.random.default_rng(7)) for _ in range(2)]
    assert first[0] == first[1]
    other = Genome.random(np.random.default_rng(8))
    assert other != first[0]
    for seed in range(50):
        vector = Genome.random(np.random.default_rng(seed)).to_vector()
        assert np.all(vector >= GENE_LOWS) and np.all(vector <= GENE_HIGHS)


def test_presets_are_valid_distinct_styles() -> None:
    assert set(TRAINING_PRESETS) | set(HELD_OUT_PRESETS) <= set(PRESETS)
    assert not set(TRAINING_PRESETS) & set(HELD_OUT_PRESETS)
    assert len({tuple(g.to_vector()) for g in PRESETS.values()}) == len(PRESETS)
    # the styles really differ in the direction their names promise
    assert PRESETS["nit"].pf_play_btn > PRESETS["lag"].pf_play_btn
    assert PRESETS["maniac-ish"].bluff_freq > PRESETS["tag"].bluff_freq > PRESETS["nit"].bluff_freq


def test_strategy_card_mentions_the_gene_values() -> None:
    genome = Genome(value_bet=0.71, bluff_freq=0.25)
    card = "\n".join(genome.strategy_card())
    assert "0.71" in card and "25%" in card
    assert all(isinstance(line, str) and line for line in genome.strategy_card())
