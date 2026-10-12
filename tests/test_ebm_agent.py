"""EBM agent: loud loading failures, legal decisions, train/inference parity, explanations, copy semantics.

The model under test is a tiny EBM produced by the real training pipeline (see ``ebm_helpers``).
"""

from __future__ import annotations

import copy
import json
import random
import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest
from core_helpers import DEFAULT_CFG, random_deal, random_stacks
from ebm_helpers import tiny_run
from test_information_boundary import hidden_variant

from poker_arena.agents.base import Agent, Decision, ModelArtifactError
from poker_arena.agents.baselines import CallStationAgent, ManiacAgent, RandomAgent
from poker_arena.agents.ebm_agent import MODEL_FILE, EBMAgent
from poker_arena.agents.learned import METADATA_FILE
from poker_arena.agents.registry import AgentSpec, check_availability, create_agent
from poker_arena.config import derive_seed
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, AbstractAction
from poker_arena.engine.observation import BOARD_CARDS_BY_STREET, Observation
from poker_arena.engine.session import play_hand
from poker_arena.features import FEATURE_NAMES, NUM_FEATURES, build_features
from poker_arena.training.ebm import class_probabilities


@pytest.fixture(scope="module")
def model_dir() -> Path:
    return tiny_run().model_dir


@pytest.fixture(scope="module")
def agent(model_dir: Path) -> EBMAgent:
    return EBMAgent.load(model_dir)


def broken_copy(model_dir: Path, tmp_path: Path) -> Path:
    target = tmp_path / "broken"
    shutil.copytree(model_dir, target)
    return target


def edit_metadata(directory: Path, **changes: object) -> None:
    path = directory / METADATA_FILE
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(changes)
    path.write_text(json.dumps(document), encoding="utf-8")


# ------------------------------------------------------------------------------------ loading
def test_load_and_registry_roundtrip(model_dir: Path) -> None:
    loaded = EBMAgent.load(model_dir, name="Mine", decision_mode="sample", seed=3)
    assert (loaded.name, loaded.decision_mode, loaded.architecture) == ("Mine", "sample", "Explainable Boosting Machine")
    assert loaded.supports_explanations and loaded.fingerprint
    via_registry = create_agent(AgentSpec("ebm", {"model_dir": str(model_dir)}, "Registered"))
    assert isinstance(via_registry, EBMAgent) and via_registry.name == "Registered"
    assert via_registry.describe()["model_fingerprint"] == loaded.fingerprint
    assert via_registry.describe()["n_interaction_terms"] >= 1


def test_missing_directory_or_metadata_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="metadata.json"):
        EBMAgent.load(tmp_path / "does_not_exist")
    with pytest.raises(ModelArtifactError):
        create_agent(AgentSpec("ebm", {"model_dir": str(tmp_path)}))


def test_missing_model_file_raises(model_dir: Path, tmp_path: Path) -> None:
    directory = broken_copy(model_dir, tmp_path)
    (directory / MODEL_FILE).unlink()
    with pytest.raises(ModelArtifactError, match=MODEL_FILE):
        EBMAgent.load(directory)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"feature_schema_hash": "0123456789abcdef"}, "feature schema"),
        ({"feature_names": ["equity"]}, "feature schema"),
        ({"artifact_schema_version": 999}, "artifact schema"),
        ({"class_names": ["a", "b"]}, "action space"),
    ],
)
def test_stale_metadata_raises(model_dir: Path, tmp_path: Path, change: dict[str, object], message: str) -> None:
    directory = broken_copy(model_dir, tmp_path)
    edit_metadata(directory, **change)
    with pytest.raises(ModelArtifactError, match=message):
        EBMAgent.load(directory)


def test_invalid_metadata_json_raises(model_dir: Path, tmp_path: Path) -> None:
    directory = broken_copy(model_dir, tmp_path)
    (directory / METADATA_FILE).write_text("{ not json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="not valid JSON"):
        EBMAgent.load(directory)


@pytest.mark.parametrize("damage", ["garbage", "truncated", "wrong_object"])
def test_corrupt_model_file_raises(model_dir: Path, tmp_path: Path, damage: str) -> None:
    directory = broken_copy(model_dir, tmp_path)
    path = directory / MODEL_FILE
    if damage == "garbage":
        path.write_bytes(b"this is not a pickle")
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    else:
        joblib.dump({"not": "an EBM"}, path)
    with pytest.raises(ModelArtifactError, match="Retrain"):
        EBMAgent.load(directory)


def test_model_fitted_on_other_features_raises(model_dir: Path, tmp_path: Path) -> None:
    directory = broken_copy(model_dir, tmp_path)
    model = joblib.load(directory / MODEL_FILE)
    model.feature_names_in_ = ["renamed"] + list(model.feature_names_in_[1:])
    joblib.dump(model, directory / MODEL_FILE)
    with pytest.raises(ModelArtifactError, match="different features"):
        EBMAgent.load(directory)


def test_registry_reports_default_directory_problems_as_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EBMAgent, "resolve_dir", classmethod(lambda cls, model_dir: tmp_path / "nothing"))
    availability = check_availability("ebm")
    assert not availability.available and "metadata.json" in availability.detail


# ------------------------------------------------------------------------------ legality in play
class CheckingAgent(Agent):
    """Wraps an agent and asserts, on every decision, that it chose a legal abstract action."""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.name = inner.name
        self.decisions = 0
        self.streets: set[int] = set()
        self.raises = 0

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)

    def act(self, obs: Observation) -> Decision:
        decision = self.inner.act(obs)
        assert decision.abstract is not None and obs.abstract_mask[decision.abstract], (decision, obs.abstract_mask)
        self.decisions += 1
        self.streets.add(int(obs.street))
        self.raises += int(decision.abstract >= AbstractAction.RAISE_SMALL)
        return decision

    def end_hand(self, summary) -> None:  # type: ignore[no-untyped-def]
        self.inner.end_hand(summary)


@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_decisions_are_always_legal_over_thousands_of_states(model_dir: Path, mode: str) -> None:
    """The engine itself rejects illegal actions; the checker additionally asserts the mask on every state."""
    rng = random.Random(derive_seed("test", "ebm-legality", mode))
    hero = CheckingAgent(EBMAgent.load(model_dir, decision_mode=mode))
    hero.reset(1)
    opponents = [CallStationAgent("cs"), ManiacAgent("m", seed=2, raise_probability=0.5), RandomAgent("r", seed=3)]
    for opponent in opponents:
        opponent.reset(4)
    for hand in range(1_300):
        stacks = random_stacks(rng) if hand % 2 else None  # tiny / asymmetric stacks exercise the mask edge cases
        agents = [hero, opponents[hand % 3]] if hand % 4 < 2 else [opponents[hand % 3], hero]
        play_hand(DEFAULT_CFG, random_deal(rng), hand % 2, agents, start_stacks=stacks, hand_number=hand)
    assert hero.decisions > 2_500
    assert hero.streets == {0, 1, 2, 3} and hero.raises > 50


# -------------------------------------------------------------------- parity and the fast path
def test_predict_proba_agrees_with_the_library_model(agent: EBMAgent) -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.uniform(lo, hi, size=300) for lo, hi in agent.model.feature_bounds_]).T.astype(np.float32)
    batch = class_probabilities(agent.model, X)
    single = np.vstack([agent.predict_proba(row) for row in X])
    assert np.abs(single - batch).max() < 1e-6
    assert np.allclose(single.sum(axis=1), 1.0)


def test_contributions_add_up_to_the_logit(agent: EBMAgent) -> None:
    """intercept + sum of term contributions = the action's logit, and the softmax of the logits = probabilities."""
    rng = np.random.default_rng(1)
    intercept = np.zeros(5)
    intercept[[int(c) for c in agent.model.classes_]] = agent.model.intercept_
    for _ in range(40):
        x = np.array([rng.uniform(lo, hi) for lo, hi in agent.model.feature_bounds_], dtype=np.float32)
        logits = np.array(
            [intercept[a] + agent.term_contributions(x, AbstractAction(a)).sum() for a in range(5)]
        )
        seen = [int(c) for c in agent.model.classes_]
        shifted = np.exp(logits[seen] - logits[seen].max())
        assert np.abs(shifted / shifted.sum() - agent.predict_proba(x)[seen]).max() < 1e-6


class RecordingEBM(Agent):
    """Plays like the EBM and keeps (observation, features, decision) so the tests can re-derive everything."""

    def __init__(self, inner: EBMAgent) -> None:
        self.inner = inner
        self.name = inner.name
        self.log: list[tuple[Observation, np.ndarray, Decision]] = []

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)

    def act(self, obs: Observation) -> Decision:
        decision = self.inner.act(obs)
        self.log.append((obs, build_features(obs), decision))
        return decision


def test_inference_features_and_decisions_match_the_batch_pipeline(model_dir: Path) -> None:
    """Train/inference parity spot check: what the agent sees equals ``build_features`` row by row, and its
    decision is the legal-masked arg-max of the model's batch prediction for exactly that row."""
    recorder = RecordingEBM(EBMAgent.load(model_dir))
    rng = random.Random(derive_seed("test", "ebm-parity"))
    for hand in range(60):
        play_hand(DEFAULT_CFG, random_deal(rng), hand % 2, [recorder, ManiacAgent("m", seed=hand, raise_probability=0.3)])
    rows = np.vstack([features for _, features, _ in recorder.log])
    assert rows.shape[1] == NUM_FEATURES and len(rows) > 100
    batch = class_probabilities(recorder.inner.model, rows)
    for (obs, _, decision), probabilities in zip(recorder.log, batch, strict=True):
        masked = np.where(obs.abstract_mask, probabilities, -np.inf)
        assert decision.abstract == int(np.argmax(masked))


# ------------------------------------------------------------------------------- explanations
def explained_decisions(model_dir: Path, hands: int = 12) -> list[tuple[Observation, Decision, object]]:
    agent = EBMAgent.load(model_dir)
    rng = random.Random(derive_seed("test", "ebm-explain"))
    out = []

    class Explaining(Agent):
        name = "explaining"

        def act(self, obs: Observation) -> Decision:
            decision = agent.act(obs)
            out.append((obs, decision, agent.explain(obs, decision)))
            return decision

    for hand in range(hands):
        play_hand(DEFAULT_CFG, random_deal(rng), hand % 2, [Explaining(), CallStationAgent("cs")])
    return out


def test_explanation_payload_has_the_documented_keys(model_dir: Path) -> None:
    results = explained_decisions(model_dir)
    assert len(results) > 15
    for obs, decision, explanation in results:
        assert explanation is not None and isinstance(explanation.summary, str) and "Explainable Boosting Machine" in explanation.summary
        details = explanation.details
        assert set(details) == {
            "probabilities", "model_probabilities", "chosen", "legal", "top_features", "contribution_unit",
        }  # fmt: skip
        assert details["chosen"] == ABSTRACT_ACTION_NAMES[decision.abstract]
        assert details["contribution_unit"] == "log-odds (EBM term contribution)"
        assert details["legal"] == [n for n, ok in zip(ABSTRACT_ACTION_NAMES, obs.abstract_mask, strict=True) if ok]
        assert set(details["probabilities"]) == set(details["legal"])
        assert sum(details["probabilities"].values()) == pytest.approx(1.0)
        assert set(details["model_probabilities"]) == set(ABSTRACT_ACTION_NAMES)
        features = details["top_features"]
        assert 1 <= len(features) <= 8
        magnitudes = [abs(f["contribution"]) for f in features]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for f in features:
            assert set(f) == {"feature", "value", "contribution", "kind"}
            parts = f["feature"].split(" x ")
            assert set(parts) <= set(FEATURE_NAMES) and (f["kind"] == "interaction") == (len(parts) == 2)
            assert (isinstance(f["value"], list) and len(f["value"]) == 2) == (f["kind"] == "interaction")
        json.dumps({"summary": explanation.summary, "details": details})  # the UI stores it as JSON


def test_top_features_name_interactions_and_report_values(agent: EBMAgent) -> None:
    x = np.zeros(NUM_FEATURES, dtype=np.float32)
    x[FEATURE_NAMES.index("equity")] = 0.9
    entries = agent.top_features(x, AbstractAction.RAISE_POT)
    assert len(entries) == len(agent.model.term_features_)
    assert any(e["kind"] == "interaction" and " x " in e["feature"] for e in entries)
    equity = next(e for e in entries if e["feature"] == "equity")
    assert equity["value"] == pytest.approx(0.9)
    total = sum(e["contribution"] for e in entries)
    assert total == pytest.approx(agent.term_contributions(x, AbstractAction.RAISE_POT).sum())


def test_explain_without_a_cached_decision_recomputes(model_dir: Path) -> None:
    results = explained_decisions(model_dir, hands=3)
    obs, decision, first = results[0]
    fresh = EBMAgent.load(model_dir)  # never acted: has no cache
    second = fresh.explain(obs, decision)
    assert second is not None and first is not None
    assert second.details["top_features"] == first.details["top_features"]


# ---------------------------------------------------------------------------------- copy semantics
def test_copy_shares_only_the_read_only_model_and_resets_independently(model_dir: Path) -> None:
    prototype = EBMAgent.load(model_dir, decision_mode="sample")
    first, second = copy.copy(prototype), copy.copy(prototype)
    first.reset(11)
    second.reset(12)
    assert first.model is second.model is prototype.model
    assert first._rng is not second._rng and first._rng is not prototype._rng

    def play(agent: EBMAgent) -> list[int]:
        rng = random.Random(5)
        recorder = RecordingEBM(agent)
        for hand in range(15):
            play_hand(DEFAULT_CFG, random_deal(rng), hand % 2, [recorder, CallStationAgent("cs")])
        return [int(d.abstract) for _, _, d in recorder.log]

    fresh = EBMAgent.load(model_dir, decision_mode="sample")
    fresh.reset(11)
    first_run = play(first)
    assert first_run == play(fresh)  # a copy + reset behaves exactly like a fresh agent
    play(second)  # heavy use of a second copy must not disturb the first
    first.reset(11)
    assert play(first) == first_run  # reset restores the stream
    # a different seed gives a different random stream (checked on the stream itself: whether sampled *play* differs
    # depends on how confident the tiny test model happens to be, which is not what this test is about)
    stream_a, stream_b = copy.copy(prototype), copy.copy(prototype)
    stream_a.reset(11)
    stream_b.reset(12)
    assert stream_a._rng.random() != stream_b._rng.random()


# ------------------------------------------------------------------------------ anti-cheating
@pytest.mark.parametrize("hero", [0, 1])
def test_decisions_do_not_depend_on_hidden_cards(model_dir: Path, hero: int) -> None:
    """Same public information + different hidden cards (opponent hole cards, burns, future board) => the
    same decisions. Mirrors ``test_registered_agents_decide_only_from_visible_information`` for the tiny model."""
    template = EBMAgent.load(model_dir)
    rng = random.Random(derive_seed("test", "ebm-real-agent", hero))

    def run(deal, button: int):  # type: ignore[no-untyped-def]
        hero_agent = copy.copy(template)
        hero_agent.reset(7)
        opponent = ManiacAgent("m", seed=3, raise_probability=0.3)  # ignores cards, so the betting path is fixed
        opponent.reset(3)
        agents: list[Agent] = [opponent, opponent]
        agents[hero], agents[1 - hero] = hero_agent, opponent
        return play_hand(DEFAULT_CFG, deal, button, agents)

    checked = 0
    for h in range(15):
        button = h % 2
        deal = random_deal(rng)
        base = run(deal, button)
        known = max([BOARD_CARDS_BY_STREET[e.street] for e in base.events if e.seat == hero] or [0])
        for mode in ("opp", "future", "both"):
            other = run(hidden_variant(rng, deal, hero, button, known, mode), button)
            assert [(e.seat, e.action) for e in other.events] == [(e.seat, e.action) for e in base.events], mode
            checked += 1
    assert checked == 45
