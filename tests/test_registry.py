"""Agent registry, specs and the copy semantics promised in ARCHITECTURE.md section 5."""

from __future__ import annotations

import copy
import pickle
import random
import textwrap

import pytest

from poker_arena.agents import registry
from poker_arena.agents.base import Agent, Decision, Explanation, ModelArtifactError
from poker_arena.agents.baselines import CallStationAgent, ManiacAgent, RandomAgent
from poker_arena.agents.registry import (
    KIND_LABELS,
    AgentAvailability,
    AgentSpec,
    available_agents,
    check_availability,
    create_agent,
    known_kinds,
    register_agent,
)
from poker_arena.engine.actions import AbstractAction
from poker_arena.engine.observation import Observation
from poker_arena.engine.session import play_hand

from core_helpers import DEFAULT_CFG, random_deal, random_midhand_engines

BASELINES = ("random", "call_station", "maniac")


class DummyAgent(Agent):
    name = "Dummy"
    architecture = "test double"

    def __init__(self, label: str = "dummy", flavour: int = 0) -> None:
        self.label = label
        self.flavour = flavour

    def act(self, obs: Observation) -> Decision:
        return Decision.from_abstract(obs, AbstractAction.CHECK_CALL)


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch):
    """Private copies of the registry tables so tests can register / break kinds freely."""
    monkeypatch.setattr(registry, "_FACTORIES", dict(registry._FACTORIES))
    monkeypatch.setattr(registry, "_LAZY", dict(registry._LAZY))
    return registry


# =============================================================================== creation
@pytest.mark.parametrize("kind,cls", [("random", RandomAgent), ("call_station", CallStationAgent), ("maniac", ManiacAgent)])
def test_create_agent_for_baselines_by_string_and_by_spec(kind: str, cls: type) -> None:
    for spec in (kind, AgentSpec(kind)):
        agent = create_agent(spec)
        assert type(agent) is cls and isinstance(agent, Agent)
        assert agent.name and agent.architecture and agent.describe()["name"] == agent.name
    assert create_agent(kind) is not create_agent(kind), "every call builds a fresh agent"


def test_spec_params_and_label_are_applied() -> None:
    agent = create_agent(AgentSpec("maniac", {"seed": 3, "raise_probability": 0.1}, label="Calm Maniac"))
    assert isinstance(agent, ManiacAgent) and agent.raise_probability == 0.1 and agent.name == "Calm Maniac"
    assert create_agent(AgentSpec("random", {"name": "Dice"})).name == "Dice"
    assert create_agent(AgentSpec("random", {"name": "Dice"}, label="Override")).name == "Override"
    with pytest.raises(TypeError):
        create_agent(AgentSpec("call_station", {"bogus": 1}))  # bad params fail loudly


def test_unknown_kind_raises_keyerror_listing_known_kinds() -> None:
    with pytest.raises(KeyError, match="Unknown agent kind 'warp_drive'") as info:
        create_agent("warp_drive")
    assert "random" in str(info.value) and "maniac" in str(info.value)
    with pytest.raises(KeyError):
        create_agent(AgentSpec(""))


def test_lazy_kind_with_missing_module_raises_model_artifact_error(isolated_registry) -> None:
    isolated_registry._LAZY["ghost"] = "poker_arena.agents.no_such_module_xyz:GhostAgent"
    with pytest.raises(ModelArtifactError, match="not implemented yet") as info:
        create_agent("ghost")
    assert isinstance(info.value.__cause__, ModuleNotFoundError)
    assert not isinstance(info.value, KeyError)
    # ``ModelArtifactError`` is the loud failure type - it must NOT be a silent fallback to random play
    assert issubclass(ModelArtifactError, RuntimeError)


def test_lazy_kind_with_missing_dependency_is_reported_as_such(isolated_registry, tmp_path, monkeypatch) -> None:
    (tmp_path / "fake_agent_needing_dep.py").write_text("import definitely_not_installed_package_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    isolated_registry._LAZY["needs_dep"] = "fake_agent_needing_dep:Agent"
    with pytest.raises(ModelArtifactError, match="missing dependency") as info:
        create_agent("needs_dep")
    assert "not implemented" not in str(info.value)
    assert "definitely_not_installed_package_xyz" in str(info.value)


def test_lazy_kind_resolves_attribute_paths(isolated_registry, tmp_path, monkeypatch) -> None:
    (tmp_path / "fake_lazy_agent_mod.py").write_text(
        textwrap.dedent(
            """
            from poker_arena.agents.baselines import CallStationAgent

            class Holder:
                @classmethod
                def load(cls, name="lazy-loaded"):
                    return CallStationAgent(name)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    isolated_registry._LAZY["lazy_ok"] = "fake_lazy_agent_mod:Holder.load"
    agent = create_agent(AgentSpec("lazy_ok", {"name": "from-load"}))
    assert isinstance(agent, CallStationAgent) and agent.name == "from-load"
    assert "lazy_ok" in known_kinds()


def test_factory_raising_model_artifact_error_propagates(isolated_registry) -> None:
    def broken_factory(**_kw):
        raise ModelArtifactError("model.joblib missing: run scripts/train_x.py")

    register_agent("broken_model", broken_factory)
    with pytest.raises(ModelArtifactError, match="model.joblib missing"):
        create_agent("broken_model")


# ============================================================================== availability
def test_check_availability_reports_without_raising(isolated_registry) -> None:
    ok = check_availability("random")
    assert ok == AgentAvailability("random", "Random", True, "")
    isolated_registry._LAZY["ghost"] = "poker_arena.agents.no_such_module_xyz:GhostAgent"
    ghost = check_availability("ghost")
    assert not ghost.available and "not implemented" in ghost.detail and ghost.label == "ghost"

    def broken(**_kw):
        raise ModelArtifactError("artifact version 2 != 3")

    register_agent("stale_model", broken)
    stale = check_availability("stale_model")
    assert not stale.available and "version 2 != 3" in stale.detail


def test_check_availability_does_not_swallow_programming_errors(isolated_registry) -> None:
    def buggy(**_kw):
        raise ZeroDivisionError("bug in agent factory")

    register_agent("buggy", buggy)
    with pytest.raises(ZeroDivisionError):
        check_availability("buggy")


def test_available_agents_lists_every_known_kind() -> None:
    listing = available_agents()
    assert [a.kind for a in listing] == known_kinds()
    by_kind = {a.kind: a for a in listing}
    assert all(by_kind[k].available for k in BASELINES)
    for item in listing:
        assert item.label and (item.available or item.detail), "unavailable agents must say why"


def test_known_kinds_and_labels() -> None:
    kinds = known_kinds()
    assert len(kinds) == len(set(kinds))
    assert list(kinds[:3]) == list(BASELINES), "baselines first"
    assert {"teacher", "ebm", "xgboost", "ga"} <= set(kinds)
    assert set(kinds) <= set(KIND_LABELS)


# ================================================================================ registering
def test_register_agent_adds_a_creatable_kind(isolated_registry) -> None:
    register_agent("dummy", DummyAgent)
    assert "dummy" in known_kinds()
    agent = create_agent(AgentSpec("dummy", {"label": "x", "flavour": 7}, label="Shown Name"))
    assert isinstance(agent, DummyAgent) and (agent.label, agent.flavour, agent.name) == ("x", 7, "Shown Name")
    assert check_availability("dummy").available
    # and it can actually play a hand
    record = play_hand(DEFAULT_CFG, random_deal(random.Random(1)), 0, [create_agent("dummy"), create_agent("maniac")])
    assert record.verify()
    assert any(a.kind == "dummy" and a.available for a in available_agents())


def test_register_agent_can_override_and_lambda_factories_work(isolated_registry) -> None:
    register_agent("random", lambda **kw: CallStationAgent("pretend-random"))
    assert create_agent("random").name == "pretend-random"


def test_registering_does_not_leak_between_tests() -> None:
    assert "dummy" not in known_kinds()
    assert registry._FACTORIES["random"] is RandomAgent


# ===================================================================================== specs
def test_agent_spec_pickle_and_dict_round_trip() -> None:
    spec = AgentSpec("maniac", {"seed": 9, "raise_probability": 0.3}, "Hyper")
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert AgentSpec.from_dict(spec.to_dict()) == spec
    assert AgentSpec.from_dict({"kind": "random"}) == AgentSpec("random", {}, None)
    # to_dict must hand out a copy: mutating it must not corrupt the (frozen) spec
    d = spec.to_dict()
    d["params"]["seed"] = 0
    assert spec.params["seed"] == 9
    for restored in (pickle.loads(pickle.dumps(spec)), AgentSpec.from_dict(spec.to_dict())):
        a, b = create_agent(spec), create_agent(restored)
        assert (a.name, a.raise_probability) == (b.name, b.raise_probability)  # type: ignore[attr-defined]


def test_agent_spec_is_hashable_so_it_can_key_result_tables() -> None:
    a = AgentSpec("maniac", {"seed": 1, "raise_probability": 0.5}, "M")
    b = AgentSpec("maniac", {"raise_probability": 0.5, "seed": 1}, "M")  # same params, different insertion order
    assert a == b and hash(a) == hash(b)
    table = {a: "result"}
    assert table[b] == "result"
    assert len({a, b, AgentSpec("maniac"), AgentSpec("random")}) == 3


def test_agent_spec_is_frozen() -> None:
    spec = AgentSpec("random")
    with pytest.raises(Exception):
        spec.kind = "maniac"  # type: ignore[misc]


def test_specs_can_be_rebuilt_in_a_worker_process() -> None:
    """Tournament workers receive pickled specs; the agents they build must behave identically."""
    spec = AgentSpec("maniac", {"seed": 4})
    clone = pickle.loads(pickle.dumps(spec))
    results = []
    for s in (spec, clone):
        agents = [create_agent(s), create_agent("call_station")]
        agents[0].reset(11)
        results.append(play_hand(DEFAULT_CFG, random_deal(random.Random(2)), 0, agents).events)
    assert results[0] == results[1]


# ========================================================================= copy semantics
def action_stream(agent: Agent, observations: list[Observation]):
    return [agent.act(o).action for o in observations]


@pytest.fixture(scope="module")
def observations() -> list[Observation]:
    obs = []
    for engine in random_midhand_engines(4, 60):
        obs.append(engine.observation_for(engine.actor_seat))  # type: ignore[arg-type]
    return obs


@pytest.mark.parametrize("make", [lambda: RandomAgent("r"), lambda: ManiacAgent("m", raise_probability=0.5)], ids=["random", "maniac"])
def test_copy_then_reset_yields_an_independent_agent(make, observations) -> None:
    original = make()
    original.reset(1)
    clone = copy.copy(original)
    assert clone is not original and clone.name == original.name
    clone.reset(2)
    # the clone's RNG is a new object after reset ...
    assert clone._rng is not original._rng
    # ... so interleaved use of one agent does not disturb the other
    expected_original = action_stream(_reset(make(), 1), observations)
    expected_clone = action_stream(_reset(make(), 2), observations)
    got_original, got_clone = [], []
    for o in observations:
        got_original.append(original.act(o).action)
        got_clone.append(clone.act(o).action)
        original.act(o)  # extra draws on the original must not shift the clone's stream
    assert got_clone == expected_clone
    assert got_original[0] == expected_original[0]  # (the original also drew extra values, so only its first act is comparable)
    assert expected_original != expected_clone, "different seeds => different streams (test is meaningful)"


def _reset(agent: Agent, seed: int) -> Agent:
    agent.reset(seed)
    return agent


def test_copy_before_reset_shares_the_rng_which_is_why_reset_is_required() -> None:
    """Documents the shallow-copy contract: only ``reset`` makes the copy independent."""
    a = RandomAgent("a")
    b = copy.copy(a)
    assert b._rng is a._rng
    b.reset(5)
    assert b._rng is not a._rng


def test_reset_is_deterministic_and_reseeds_fully(observations) -> None:
    agent = RandomAgent("r")
    agent.reset(42)
    first = action_stream(agent, observations)
    agent.reset(42)
    assert action_stream(agent, observations) == first
    agent.reset(43)
    assert action_stream(agent, observations) != first


def test_reset_without_seed_returns_to_the_construction_seed(observations) -> None:
    agent = RandomAgent("r", seed=77)
    first = action_stream(agent, observations)
    agent.reset()
    assert action_stream(agent, observations) == first, "reset() restores the constructor seed"
    other = RandomAgent("r", seed=78)
    assert action_stream(other, observations) != first
    assert action_stream(create_agent(AgentSpec("random", {"seed": 77})), observations) == first


def test_stateless_call_station_copy_is_trivially_independent(observations) -> None:
    a = CallStationAgent("a")
    b = copy.copy(a)
    b.reset(3)
    assert action_stream(a, observations) == action_stream(b, observations)
    b.name = "renamed"
    assert a.name == "a"


def test_baseline_agents_pickle_round_trip(observations) -> None:
    for agent in (RandomAgent("r", seed=3), ManiacAgent("m", seed=3), CallStationAgent("c")):
        clone = pickle.loads(pickle.dumps(agent))
        assert action_stream(clone, observations) == action_stream(agent, observations)


def test_every_baseline_only_returns_actions_the_mask_allows(observations) -> None:
    for kind in BASELINES:
        agent = create_agent(kind)
        agent.reset(5)
        for obs in observations:
            decision = agent.act(obs)
            assert isinstance(decision, Decision) and decision.abstract is not None
            assert obs.abstract_mask[decision.abstract], f"{kind} chose an illegal abstract action"


def test_agent_base_class_defaults() -> None:
    agent = DummyAgent()
    assert agent.reset(1) is None and agent.end_hand(None) is None  # type: ignore[arg-type]
    engine = next(random_midhand_engines(1, 1))
    obs = engine.observation_for(engine.actor_seat)  # type: ignore[arg-type]
    decision = Decision.from_abstract(obs, AbstractAction.CHECK_CALL, Explanation("because", {"k": 1}))
    assert agent.explain(obs, decision) is decision.explanation
    assert agent.describe() == {
        "name": "Dummy", "architecture": "test double", "description": "", "supports_explanations": False,
    }
    assert repr(agent) == "DummyAgent(name='Dummy')"
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]
