"""XGBoostAgent: loud loading, legality, parity with the training data, SHAP additivity, copy semantics."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from poker_arena.agents.base import ModelArtifactError
from poker_arena.agents.baselines import CallStationAgent, ManiacAgent, RandomAgent
from poker_arena.agents.learned import TOP_FEATURES_SHOWN, mask_and_normalise
from poker_arena.agents.registry import AgentSpec, create_agent
from poker_arena.agents.xgb_agent import MODEL_FILE, XGBoostAgent, class_probabilities, shap_contributions
from poker_arena.config import derive_seed
from poker_arena.data import collect_on_policy, feature_matrix, label_vector, legal_mask_matrix
from poker_arena.data.generation import DatasetConfig, assemble_frame, plan_matches, run_match_plan
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.engine.deals import Deal
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.engine.session import HandSession, play_hand
from poker_arena.features import FEATURE_NAMES, NUM_FEATURES, build_features
from poker_arena.training.common import masked_argmax

from core_helpers import DEFAULT_CFG, random_midhand_engines
from test_dataset_parity import make_spec, replay_rows  # replays hand records with fresh engines and trackers
from xgb_helpers import TinyModel, copy_model_dir, tiny_model

LEGALITY_STATES = 2000
SHAP_ROWS = 200
ADDITIVITY_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def tiny() -> TinyModel:
    return tiny_model()


@pytest.fixture(scope="module")
def agent(tiny: TinyModel) -> XGBoostAgent:
    return XGBoostAgent.load(tiny.model_dir)


def softmax(margins: np.ndarray) -> np.ndarray:
    exp = np.exp(margins - margins.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------ loading
def test_loads_through_the_registry_with_the_documented_identity(tiny: TinyModel) -> None:
    built = create_agent(AgentSpec("xgboost", {"model_dir": str(tiny.model_dir)}, "MyXGB"))
    assert isinstance(built, XGBoostAgent) and built.name == "MyXGB"
    info = built.describe()
    assert info["architecture"] == "XGBoost" and info["supports_explanations"] is True
    assert info["model_fingerprint"] and info["feature_schema_hash"] == tiny.outcome.metadata["feature_schema_hash"]
    assert XGBoostAgent.MODEL_SUBDIR == "xgb"
    assert XGBoostAgent.load(tiny.model_dir, name="X", decision_mode="sample", seed=3).decision_mode == "sample"


def test_a_missing_directory_or_file_is_a_loud_error_not_a_fallback(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="metadata.json"):
        XGBoostAgent.load(tmp_path / "does_not_exist")
    with pytest.raises(ModelArtifactError):
        create_agent(AgentSpec("xgboost", {"model_dir": str(tmp_path / "does_not_exist")}))

    folder = copy_model_dir(tmp_path / "no_metadata")
    (folder / "metadata.json").unlink()
    with pytest.raises(ModelArtifactError, match="metadata.json"):
        XGBoostAgent.load(folder)

    folder = copy_model_dir(tmp_path / "no_model")
    (folder / MODEL_FILE).unlink()
    with pytest.raises(ModelArtifactError, match=MODEL_FILE):
        XGBoostAgent.load(folder)


def edit_metadata(folder: Path, **changes: object) -> None:
    path = folder / "metadata.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(changes)
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"feature_schema_hash": "0000000000000000"}, "feature schema"),
        ({"feature_names": ["a", "b"]}, "feature schema"),
        ({"artifact_schema_version": 999}, "artifact schema"),
        ({"class_names": ["x", "y"]}, "action space"),
        ({"library_versions": {"xgboost": "99.0.0"}}, "newer"),
    ],
)
def test_stale_or_incompatible_metadata_is_rejected(tmp_path: Path, changes: dict, message: str) -> None:
    folder = copy_model_dir(tmp_path / "stale")
    edit_metadata(folder, **changes)
    with pytest.raises(ModelArtifactError, match=message):
        XGBoostAgent.load(folder)


def test_invalid_metadata_json_is_rejected(tmp_path: Path) -> None:
    folder = copy_model_dir(tmp_path / "bad_json")
    (folder / "metadata.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="not valid JSON"):
        XGBoostAgent.load(folder)


@pytest.mark.parametrize("damage", ["garbage", "truncated", "empty"])
def test_a_corrupted_model_file_is_rejected(tmp_path: Path, damage: str) -> None:
    folder = copy_model_dir(tmp_path / damage)
    path = folder / MODEL_FILE
    original = path.read_bytes()
    path.write_bytes({"garbage": random.Random(0).randbytes(4096), "truncated": original[: len(original) // 2], "empty": b""}[damage])
    # unparsable files fail in XGBoost; a truncated file may still parse, and is caught by the fingerprint
    with pytest.raises(ModelArtifactError, match="could not be read|does not belong"):
        XGBoostAgent.load(folder)


def test_metadata_without_a_fingerprint_is_rejected(tmp_path: Path) -> None:
    folder = copy_model_dir(tmp_path / "no_fingerprint")
    edit_metadata(folder, model_fingerprint=None)
    with pytest.raises(ModelArtifactError, match="does not belong"):
        XGBoostAgent.load(folder)


def test_a_model_file_that_does_not_belong_to_the_metadata_is_rejected(tmp_path: Path, tiny: TinyModel) -> None:
    folder = copy_model_dir(tmp_path / "swapped")
    other = tiny.work_dir / "round_0" / MODEL_FILE  # a valid model of the same run, but not the selected one
    if tiny.outcome.selected_round == 0:
        other = tiny.work_dir / "round_1" / MODEL_FILE
    (folder / MODEL_FILE).write_bytes(other.read_bytes())
    with pytest.raises(ModelArtifactError, match="does not belong"):
        XGBoostAgent.load(folder)


def test_a_booster_with_the_wrong_shape_is_rejected(tmp_path: Path) -> None:
    folder = copy_model_dir(tmp_path / "wrong_shape")
    rng = np.random.default_rng(0)
    other = xgb.train(
        {"objective": "multi:softprob", "num_class": 3, "tree_method": "hist"},
        xgb.DMatrix(rng.normal(size=(60, 4)), rng.integers(0, 3, 60)),
        3,
    )
    other.save_model(str(folder / MODEL_FILE))
    with pytest.raises(ModelArtifactError, match="features"):
        XGBoostAgent.load(folder)


# ----------------------------------------------------------------------------------- legality
@pytest.mark.slow
def test_decisions_are_always_legal_over_thousands_of_random_states(tiny: TinyModel) -> None:
    agents = [
        XGBoostAgent.load(tiny.model_dir),
        XGBoostAgent.load(tiny.model_dir, decision_mode="sample", seed=1),
    ]
    chosen_counts = np.zeros(NUM_ABSTRACT_ACTIONS, dtype=int)
    for index, engine in enumerate(random_midhand_engines(seed=2024, count=LEGALITY_STATES)):
        player = agents[index % len(agents)]
        obs = engine.observation_for(engine.actor_seat)
        decision = player.act(obs)
        assert decision.abstract is not None and obs.abstract_mask[decision.abstract], (decision.abstract, obs.abstract_mask)
        if player.decision_mode == "greedy":
            expected = int(np.argmax(mask_and_normalise(player.predict_proba(build_features(obs)), obs.abstract_mask)))
            assert int(decision.abstract) == expected
        chosen_counts[decision.abstract] += 1
        engine.act(decision.action)  # the engine is the final authority: an illegal action raises
    assert (chosen_counts > 0).sum() >= 3  # the fuzzing exercised several different actions


def test_a_model_that_prefers_an_illegal_action_is_still_masked(agent: XGBoostAgent) -> None:
    masked_agent = copy.copy(agent)
    checked = 0
    for engine in random_midhand_engines(seed=7, count=600):
        obs = engine.observation_for(engine.actor_seat)
        illegal = np.flatnonzero(~obs.abstract_mask)
        if not len(illegal):
            continue  # every action is legal here: nothing to check
        proba = np.full(NUM_ABSTRACT_ACTIONS, 0.01)
        proba[illegal[0]] = 0.96  # the model's favourite is an action the rules do not allow
        masked_agent.predict_proba = lambda x, proba=proba: proba  # type: ignore[method-assign]
        decision = masked_agent.act(obs)
        assert decision.abstract is not None and obs.abstract_mask[decision.abstract]
        checked += 1
    assert checked > 50


@pytest.mark.slow
def test_plays_complete_hands_against_baselines_through_hand_sessions(agent: XGBoostAgent) -> None:
    baselines = [RandomAgent(), CallStationAgent(), ManiacAgent(), create_agent("teacher")]
    hands = 0
    for opponent_index, opponent in enumerate(baselines):
        player = copy.copy(agent)
        player.reset(opponent_index)
        opponent.reset(100 + opponent_index)
        trackers = (OpponentTracker(), OpponentTracker())
        for hand in range(15):
            deal = Deal.from_seed(derive_seed("test", "xgb_agent", opponent_index, hand // 2))
            session = HandSession(
                DEFAULT_CFG, deal, hand % 2, (player, opponent), hand_number=hand, trackers=trackers,
                collect_explanations=True,
            )  # fmt: skip
            while not session.is_complete:
                outcome = session.step()  # raises IllegalActionError for any illegal action
                if session.agents[outcome.event.seat] is player:
                    assert outcome.decision.abstract is not None
                    assert outcome.observation.abstract_mask[AbstractAction[outcome.decision.abstract]]
                    assert outcome.decision.explanation and outcome.decision.explanation["summary"]
            assert session.finish().verify()
            hands += 1
    assert hands == 60


# ---------------------------------------------------------- the training data is what the agent sees
def test_offline_masked_predictions_reproduce_the_played_actions_exactly(tiny: TinyModel, agent: XGBoostAgent) -> None:
    """Dataset features + legal mask + masked arg-max == what the agent really played in those hands."""
    config = DatasetConfig(seed=9, match_lengths=(25,), short_stack_probability=0.3)
    frame = collect_on_policy(
        AgentSpec("xgboost", {"model_dir": str(tiny.model_dir)}), 40, ("test", "xgb_parity"), config=config, split="val"
    )
    assert len(frame) > 60
    X, legal = feature_matrix(frame), legal_mask_matrix(frame)
    offline = masked_argmax(class_probabilities(agent.booster, X), legal)
    assert np.array_equal(offline, frame["behavior_action"].to_numpy())
    assert legal[np.arange(len(frame)), offline].all() and label_vector(frame).shape == offline.shape


def test_dataset_rows_equal_build_features_of_a_replay_and_give_identical_probabilities(agent: XGBoostAgent) -> None:
    """Spot check of train/inference parity: stored rows == what a live agent would build from the same hands."""
    spec = make_spec()
    result = run_match_plan(spec, plan_matches(spec, 40)[0], keep_records=True)
    frame = assemble_frame([result], "train", "bc")
    replayed = replay_rows(result)
    stored = feature_matrix(frame)
    assert len(replayed) == len(stored) > 40
    live = np.vstack([features for features, _, _ in replayed])
    assert np.array_equal(stored, live)
    batch = class_probabilities(agent.booster, stored)
    single = np.vstack([agent.predict_proba(features) for features in live])
    assert np.allclose(batch, single, atol=1e-12)


# ---------------------------------------------------------------------------------------- SHAP
def validation_rows(tiny: TinyModel, count: int) -> np.ndarray:
    X = feature_matrix(tiny.validation_frame)
    return X[np.random.default_rng(0).choice(len(X), size=min(count, len(X)), replace=False)]


def test_shap_contributions_plus_bias_equal_the_raw_margin_of_every_class(tiny: TinyModel, agent: XGBoostAgent) -> None:
    X = validation_rows(tiny, SHAP_ROWS)
    contributions = shap_contributions(agent.booster, X)
    assert contributions.shape == (len(X), NUM_ABSTRACT_ACTIONS, NUM_FEATURES + 1)
    margins = agent.booster.predict(xgb.DMatrix(X, feature_names=list(FEATURE_NAMES)), output_margin=True)
    assert np.abs(contributions.sum(axis=2) - margins).max() < ADDITIVITY_TOLERANCE
    # the margins are what the softmax turns into the probabilities the agent plays with
    assert np.abs(softmax(margins) - class_probabilities(agent.booster, X)).max() < 1e-5


def test_top_features_are_exact_shap_values_of_the_chosen_action(tiny: TinyModel, agent: XGBoostAgent) -> None:
    X = validation_rows(tiny, 40)
    margins = agent.booster.predict(xgb.DMatrix(X, feature_names=list(FEATURE_NAMES)), output_margin=True)
    for row, margin in zip(X, margins, strict=True):
        action = AbstractAction(int(np.argmax(margin)))
        features = agent.top_features(row, action)
        assert len(features) == NUM_FEATURES and {f["feature"] for f in features} == set(FEATURE_NAMES)
        bias = shap_contributions(agent.booster, row)[0, int(action), NUM_FEATURES]
        total = sum(f["contribution"] for f in features) + bias
        assert abs(total - margin[int(action)]) < ADDITIVITY_TOLERANCE
        magnitudes = [abs(f["contribution"]) for f in features]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for f in features:
            assert f["value"] == pytest.approx(float(row[FEATURE_NAMES.index(f["feature"])]))


def test_explanation_payload_has_the_shared_shape(tiny: TinyModel, agent: XGBoostAgent) -> None:
    player = copy.copy(agent)
    player.reset(1)
    engine = next(iter(random_midhand_engines(seed=5, count=1)))
    obs = engine.observation_for(engine.actor_seat)
    decision = player.act(obs)
    explanation = player.explain(obs, decision)
    assert explanation is not None and "XGBoost" in explanation.summary
    details = explanation.details
    assert set(details) == {"probabilities", "model_probabilities", "chosen", "legal", "top_features", "contribution_unit"}
    assert details["contribution_unit"] == "SHAP value (log-odds margin)" == XGBoostAgent.contribution_unit
    legal_names = [ABSTRACT_ACTION_NAMES[i] for i in range(NUM_ABSTRACT_ACTIONS) if obs.abstract_mask[i]]
    assert details["legal"] == legal_names and set(details["probabilities"]) == set(legal_names)
    assert sum(details["probabilities"].values()) == pytest.approx(1.0)
    assert set(details["model_probabilities"]) == set(ABSTRACT_ACTION_NAMES)
    assert details["chosen"] == ABSTRACT_ACTION_NAMES[int(decision.abstract)]
    top = details["top_features"]
    assert 0 < len(top) <= TOP_FEATURES_SHOWN
    assert all(set(item) == {"feature", "value", "contribution"} for item in top)
    json.dumps(details)  # JSON-serialisable: stored in hand records and shown in the UI
    # explaining an observation the agent did not just act on recomputes the same thing
    again = copy.copy(agent).explain(obs, decision)
    assert again is not None and again.details["top_features"] == top


# ----------------------------------------------------------------------------- copy semantics
def play_fingerprint(player: XGBoostAgent, seed: int) -> list[tuple[int, str]]:
    player.reset(seed)
    opponent = ManiacAgent("m", seed=3)
    opponent.reset(3)
    trackers = (OpponentTracker(), OpponentTracker())
    actions: list[tuple[int, str]] = []
    for hand in range(6):
        record = play_hand(
            DEFAULT_CFG, Deal.from_seed(derive_seed("test", "xgb_copy", hand)), hand % 2, (player, opponent),
            hand_number=hand, trackers=trackers,
        )  # fmt: skip
        actions += [(e.seat, str(e.action)) for e in record.events]
    return actions


def test_copies_share_only_the_read_only_model_and_reset_rebinds_the_rest(tiny: TinyModel) -> None:
    template = XGBoostAgent.load(tiny.model_dir, decision_mode="sample", seed=5)
    raw_before = bytes(template.booster.save_raw())
    first, second = copy.copy(template), copy.copy(template)
    first.reset(11)
    second.reset(12)
    assert first.booster is template.booster is second.booster
    assert len({id(template._rng), id(first._rng), id(second._rng)}) == 3
    state = second._rng.getstate()
    first_run = play_fingerprint(first, 11)  # heavy use of one copy ...
    assert second._rng.getstate() == state and second._last is None  # ... never leaks into the other
    assert play_fingerprint(copy.copy(template), 11) == first_run  # a fresh copy + same seed = same play
    # different seeds really give different random streams (checked on the stream itself: whether sampled *play*
    # differs depends on how confident the tiny test model happens to be, which is not what this test is about)
    stream_a, stream_b = copy.copy(template), copy.copy(template)
    stream_a.reset(11)
    stream_b.reset(12)
    assert stream_a._rng.random() != stream_b._rng.random()
    assert bytes(template.booster.save_raw()) == raw_before  # playing never mutates the shared booster


def test_greedy_play_does_not_depend_on_the_reset_seed(agent: XGBoostAgent) -> None:
    assert play_fingerprint(copy.copy(agent), 1) == play_fingerprint(copy.copy(agent), 99)


def test_single_row_inference_is_fast(tiny: TinyModel, agent: XGBoostAgent) -> None:
    rows = validation_rows(tiny, 200)
    agent.predict_proba(rows[0])  # warm-up
    timings = []
    for row in rows:
        started = time.perf_counter()
        agent.predict_proba(row)
        timings.append(time.perf_counter() - started)
    assert float(np.median(timings)) < 0.005, "median single-row inference should be well below the 5 ms hot-path budget"
