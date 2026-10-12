"""GeneticAgent: legality, explanations, determinism, copy semantics, artifact loading."""

from __future__ import annotations

import ast
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

from poker_arena.agents import ga_agent as ga_module
from poker_arena.agents.base import Decision, ModelArtifactError
from poker_arena.agents.baselines import CallStationAgent, ManiacAgent, RandomAgent
from poker_arena.agents.ga_agent import (
    BEST_GENOME_FILE,
    GA_SCHEMA_VERSION,
    GENE_NAMES,
    METADATA_FILE,
    PRESETS,
    GeneticAgent,
    Genome,
)
from poker_arena.agents.registry import AgentSpec, create_agent
from poker_arena.config import MODELS_DIR, derive_seed
from poker_arena.engine.actions import AbstractAction
from poker_arena.engine.deals import Deal
from poker_arena.engine.observation import Street
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.engine.session import play_hand

from core_helpers import DEFAULT_CFG, random_midhand_engines

LEGALITY_STATES = 3000


def _agents_for_fuzzing() -> list[GeneticAgent]:
    rng = np.random.default_rng(11)
    agents = [GeneticAgent.from_preset(name, seed=i) for i, name in enumerate(PRESETS)]
    agents += [GeneticAgent(Genome.random(rng), name=f"rand{i}", seed=100 + i) for i in range(4)]
    return agents


def test_agent_only_chooses_legal_actions_over_thousands_of_random_states() -> None:
    agents = _agents_for_fuzzing()
    streets_seen: set[Street] = set()
    for index, engine in enumerate(random_midhand_engines(seed=2024, count=LEGALITY_STATES)):
        agent = agents[index % len(agents)]
        seat = engine.actor_seat
        obs = engine.observation_for(seat)
        mask = obs.abstract_mask
        decision = agent.act(obs)
        assert decision.abstract is not None and mask[decision.abstract], (decision.abstract, mask)
        engine.act(decision.action)  # the engine is the final authority: must not raise
        streets_seen.add(obs.street)
    assert streets_seen == set(Street)  # the fuzzing really covered every street


def test_agent_plays_complete_hands_against_baselines_with_explanations() -> None:
    ga = GeneticAgent.from_preset("tag")
    ga.reset(1)
    opponents = [RandomAgent(), CallStationAgent(), ManiacAgent(), GeneticAgent.from_preset("lag")]
    for opponent_index, opponent in enumerate(opponents):
        opponent.reset(opponent_index)
        trackers = (OpponentTracker(), OpponentTracker())
        for hand in range(30):
            deal = Deal.from_seed(derive_seed("test", "ga_agent", opponent_index, hand // 2))
            record = play_hand(
                DEFAULT_CFG, deal, hand % 2, (ga, opponent), hand_number=hand, trackers=trackers,
                collect_explanations=True,
            )  # fmt: skip
            assert record.verify()
            ga_decisions = [d for e, d in zip(record.events, record.decisions) if e.seat == 0]
            assert all(d.explanation and d.explanation["summary"] for d in ga_decisions)


def test_explanations_list_rules_genes_and_numbers() -> None:
    agent = GeneticAgent.from_preset("tag")
    agent.reset(5)
    engines = list(random_midhand_engines(seed=7, count=60))
    texts = []
    for engine in engines:
        obs = engine.observation_for(engine.actor_seat)
        decision = agent.act(obs)
        explanation = agent.explain(obs, decision)
        assert explanation is not None and explanation.summary.endswith(".")
        assert set(explanation.details) == {"action", "rules", "genes", "facts"}
        assert explanation.details["action"] == decision.abstract.name.lower()
        assert explanation.details["rules"] and explanation.details["genes"]
        for name, value in explanation.details["genes"].items():
            assert name in GENE_NAMES
            assert value == pytest.approx(getattr(agent.genome, name), abs=1e-3)
        assert any(ch.isdigit() for ch in explanation.summary)  # numbers, not just prose
        texts.append(explanation.summary)
    assert len(set(texts)) > 10  # different situations -> different explanations


def test_explanation_quotes_the_threshold_that_fired() -> None:
    agent = GeneticAgent(Genome(pf_open_raise=0.38, pf_play_btn=0.33), seed=3)
    for engine in random_midhand_engines(seed=9, count=200):
        obs = engine.observation_for(engine.actor_seat)
        if obs.street is Street.PREFLOP and obs.is_button and obs.num_raises == 0:
            decision = agent.act(obs)
            if decision.abstract in (AbstractAction.RAISE_SMALL, AbstractAction.RAISE_POT):
                assert "pf_open_raise 0.38" in decision.explanation.summary
                return
    pytest.fail("no open-raise state found")


def test_same_seed_same_decisions_different_seed_differs() -> None:
    bluffy = Genome(bluff_freq=0.6, semi_bluff_freq=0.6, big_bet_prob=0.5, value_bet=0.9, value_raise=0.98)

    def trace(seed: int) -> list[str]:
        agent = GeneticAgent(bluffy, seed=seed)
        agent.reset(seed)
        return [
            agent.act(engine.observation_for(engine.actor_seat)).abstract.name
            for engine in random_midhand_engines(seed=31, count=300)
        ]

    assert trace(4) == trace(4)
    assert trace(4) != trace(5)


def test_reset_restores_the_initial_stream() -> None:
    agent = GeneticAgent(Genome(bluff_freq=0.6), seed=1)
    states = list(random_midhand_engines(seed=12, count=40))

    def run() -> list[str]:
        return [agent.act(e.observation_for(e.actor_seat)).abstract.name for e in states]

    agent.reset(8)
    first = run()
    agent.reset(8)
    assert run() == first


def test_copy_plus_reset_gives_independent_agents_sharing_only_the_genome() -> None:
    original = GeneticAgent(Genome(bluff_freq=0.6), seed=1)
    clone = copy.copy(original)
    original.reset(1)
    clone.reset(2)
    assert clone.genome is original.genome  # immutable, shared on purpose
    assert clone._rng is not original._rng and clone._equity_rng is not original._equity_rng
    assert clone._equity_cache is not original._equity_cache
    before = original._rng.bit_generator.state
    states = list(random_midhand_engines(seed=14, count=50))
    for engine in states:
        clone.act(engine.observation_for(engine.actor_seat))
    assert original._rng.bit_generator.state == before  # using the clone never advances the original's stream
    assert not original._equity_cache and original._equity_cache == {}


def test_equity_cache_is_cleared_after_each_hand() -> None:
    agent = GeneticAgent.from_preset("tag", seed=2)
    deal = Deal.from_seed(5)
    play_hand(DEFAULT_CFG, deal, 0, (agent, CallStationAgent()))
    assert agent._equity_cache == {}


def test_describe_contains_genome_and_strategy() -> None:
    info = GeneticAgent.from_preset("lag").describe()
    assert info["architecture"] == "Genetic Algorithm" and info["supports_explanations"] is True
    assert info["genome"] == PRESETS["lag"].to_dict()
    assert info["strategy"] and info["equity_samples"] > 1


def test_unknown_preset_and_bad_equity_samples() -> None:
    with pytest.raises(KeyError, match="no-such-style"):
        GeneticAgent.from_preset("no-such-style")
    with pytest.raises(ValueError):
        GeneticAgent(equity_samples=1)


# --------------------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------------------
def write_artifacts(directory: Path, genome: Genome | None = None, **metadata_overrides) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    genome = genome or Genome(value_bet=0.7)
    (directory / BEST_GENOME_FILE).write_text(
        json.dumps({"schema_version": GA_SCHEMA_VERSION, "genes": genome.to_dict()}), encoding="utf-8"
    )
    metadata = {"schema_version": GA_SCHEMA_VERSION, "gene_names": list(GENE_NAMES), "equity_samples": 77,
                "training": {"generations": 3}}  # fmt: skip
    metadata.update(metadata_overrides)
    (directory / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_load_round_trip_uses_stored_genome_and_equity_samples(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    agent = GeneticAgent.load(tmp_path)
    assert agent.genome == Genome(value_bet=0.7) and agent.equity_samples == 77
    assert agent.describe()["training"] == {"generations": 3}
    assert GeneticAgent.load(str(tmp_path), name="custom", seed=9).name == "custom"
    via_registry = create_agent(AgentSpec("ga", {"model_dir": tmp_path}, label="Evolved"))
    assert via_registry.name == "Evolved" and via_registry.genome == agent.genome


def test_load_missing_directory_or_file_is_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="train_ga.py"):
        GeneticAgent.load(tmp_path / "nowhere")
    write_artifacts(tmp_path / "half")
    (tmp_path / "half" / METADATA_FILE).unlink()
    with pytest.raises(ModelArtifactError, match=METADATA_FILE):
        GeneticAgent.load(tmp_path / "half")


@pytest.mark.parametrize("filename", [BEST_GENOME_FILE, METADATA_FILE])
def test_load_rejects_wrong_schema_version(tmp_path: Path, filename: str) -> None:
    write_artifacts(tmp_path)
    path = tmp_path / filename
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = GA_SCHEMA_VERSION + 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="schema_version"):
        GeneticAgent.load(tmp_path)


def test_load_rejects_incompatible_gene_names_and_values(tmp_path: Path) -> None:
    write_artifacts(tmp_path / "names", gene_names=list(GENE_NAMES[:-1]))
    with pytest.raises(ModelArtifactError, match="genes"):
        GeneticAgent.load(tmp_path / "names")

    write_artifacts(tmp_path / "values")
    path = tmp_path / "values" / BEST_GENOME_FILE
    document = json.loads(path.read_text(encoding="utf-8"))
    document["genes"]["value_bet"] = 5.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="valid genome"):
        GeneticAgent.load(tmp_path / "values")

    write_artifacts(tmp_path / "samples", equity_samples="many")
    with pytest.raises(ModelArtifactError, match="equity_samples"):
        GeneticAgent.load(tmp_path / "samples")


def test_load_rejects_corrupt_json(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    (tmp_path / BEST_GENOME_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="Cannot read"):
        GeneticAgent.load(tmp_path)


def test_shipped_model_loads_and_plays_legal_hands() -> None:
    """The committed ``artifacts/models/ga`` must be loadable through the registry (``create_agent("ga")``)."""
    if not (MODELS_DIR / "ga" / BEST_GENOME_FILE).exists():
        pytest.skip("no trained GA model in artifacts/models/ga (run scripts/train_ga.py)")
    agent = create_agent("ga")
    assert isinstance(agent, GeneticAgent)
    rng = random.Random(0)
    for engine in random_midhand_engines(seed=rng.randint(0, 10**6), count=200):
        obs = engine.observation_for(engine.actor_seat)
        decision = agent.act(obs)
        assert isinstance(decision, Decision) and obs.abstract_mask[decision.abstract]


# --------------------------------------------------------------------------------------
# Anti-cheating boundary (static)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("module", [ga_module, __import__("poker_arena.training.ga", fromlist=["x"])])
def test_ga_modules_never_import_hidden_state_modules(module) -> None:
    forbidden = ("poker_arena.engine.game", "poker_arena.engine.history", "poker_arena.engine.deals",
                 "poker_arena.engine.session", "pokerkit")  # fmt: skip
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        )
        for name in names:
            assert not any(name == f or name.startswith(f + ".") for f in forbidden), name
