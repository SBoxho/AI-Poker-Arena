"""EBM training pipeline: hand-aware early stopping, DAgger bookkeeping, honest validation, artifact contents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ebm_helpers import DATASET_NAME, SEED, make_tiny_dataset, tiny_run, tiny_settings

from poker_arena.agents.ebm_agent import MODEL_FILE, EBMAgent
from poker_arena.agents.learned import ARTIFACT_SCHEMA_VERSION, METADATA_FILE, read_metadata, validate_metadata
from poker_arena.config import SEED_NAMESPACES, GameConfig
from poker_arena.data import load_dataset, load_meta
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS
from poker_arena.features import FEATURE_NAMES, schema_hash
from poker_arena.training import ebm as training
from poker_arena.training.common import SplitArrays, class_weights, evaluate_classifier, load_training_data

REQUIRED_METADATA_KEYS = {
    "artifact_schema_version", "feature_schema_hash", "feature_names", "class_names", "created_at", "library_versions",
    "dataset", "config", "seeds", "training_time_s", "n_train_rows", "n_val_rows", "class_weights", "imitation",
    "dagger", "policy_validation", "global_importance", "top_interactions", "selected_round", "selection_rule",
}  # fmt: skip


@pytest.fixture(scope="module")
def metadata() -> dict[str, Any]:
    return tiny_run().outcome.metadata


# ----------------------------------------------------------------------------- hand-aware bags
def test_hand_aware_bags_hold_out_whole_hands() -> None:
    hands = np.repeat(np.arange(200, dtype=np.int64) * 7919, 3)  # 200 hands x 3 decisions
    bags = training.hand_aware_bags(hands, n_bags=4, validation_fraction=0.25, seed=1)
    assert bags.shape == (600, 4) and set(np.unique(bags)) == {-1, 1}
    for bag in range(4):
        per_hand = bags[:, bag].reshape(200, 3)
        assert (per_hand == per_hand[:, :1]).all()  # all decisions of one hand sit on the same side
        assert (per_hand[:, 0] == -1).sum() == 50
    assert not np.array_equal(bags[:, 0], bags[:, 1])  # every bag holds out different hands
    assert np.array_equal(bags, training.hand_aware_bags(hands, 4, 0.25, 1))  # reproducible


def test_select_round_prefers_the_best_mean_and_the_earlier_round_on_ties() -> None:
    history = [{"round": 0, "mean_bb_per_100": 10.0}, {"round": 1, "mean_bb_per_100": 30.0}, {"round": 2, "mean_bb_per_100": 30.0}]
    assert training.select_round(history) == 1
    assert training.select_round(history[:1]) == 0
    assert training.select_round([{"round": 0, "mean_bb_per_100": 5.0}, {"round": 1, "mean_bb_per_100": 4.9}]) == 0


def test_dataset_must_match_the_configured_game(tmp_path: Path) -> None:
    make_tiny_dataset(tmp_path)
    meta = load_meta(DATASET_NAME, tmp_path)
    assert training.dataset_config_from_meta(meta, GameConfig()).game == GameConfig()
    with pytest.raises(ValueError, match="regenerate the dataset"):
        training.dataset_config_from_meta(meta, GameConfig(starting_stack=2000))


# --------------------------------------------------------------------------- the whole pipeline
def test_artifacts_exist_and_the_agent_validates_them() -> None:
    run = tiny_run()
    assert {p.name for p in run.model_dir.iterdir()} >= {MODEL_FILE, METADATA_FILE, "config.json"}
    metadata = read_metadata(run.model_dir)
    validate_metadata(metadata, run.model_dir)
    assert metadata["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert (run.model_dir / MODEL_FILE).stat().st_size == metadata["model_file_bytes"]
    assert not run.settings.work_dir.exists()  # candidate models of the rounds are cleaned up by default


def test_metadata_contains_everything_the_notebooks_and_the_ui_read(metadata: dict[str, Any]) -> None:
    assert REQUIRED_METADATA_KEYS <= set(metadata)
    assert metadata["feature_schema_hash"] == schema_hash() and metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["class_names"] == list(ABSTRACT_ACTION_NAMES)
    assert {"python", "interpret", "numpy", "scikit-learn"} <= set(metadata["library_versions"])
    dataset = load_meta(DATASET_NAME, tiny_run().dataset_dir)
    assert metadata["dataset"]["name"] == DATASET_NAME
    assert metadata["dataset"]["content_sha256"] == dataset["content_sha256"]
    assert metadata["dataset"]["rows"] == dataset["rows"] and metadata["dataset"]["schema_hash"] == schema_hash()
    assert set(metadata["class_weights"]) == set(ABSTRACT_ACTION_NAMES)
    assert metadata["seeds"]["master_seed"] == SEED and metadata["seeds"]["eval_namespace_used"] is False
    assert metadata["config"]["script_configuration"] == {"TEST": True}
    assert metadata["config"]["ebm_params"]["interactions"] == 4
    json.dumps(metadata)  # plain JSON


def test_importances_are_descending_and_rank_interactions_with_main_effects(metadata: dict[str, Any]) -> None:
    importance = list(metadata["global_importance"].values())
    assert importance == sorted(importance, reverse=True) and importance[0] > 0
    assert len(importance) == metadata["n_terms"] == len(FEATURE_NAMES) + metadata["n_interaction_terms"]
    assert metadata["n_interaction_terms"] == 4
    assert all(" x " in entry["term"] and len(entry["features"]) == 2 for entry in metadata["top_interactions"])
    assert set(metadata["global_importance_by_action"]) == set(ABSTRACT_ACTION_NAMES)
    assert "global_importance_training_weighted" in metadata


def test_dagger_table_covers_every_round_and_marks_the_selected_one(metadata: dict[str, Any]) -> None:
    table = metadata["dagger"]
    assert [row["round"] for row in table] == [0, 1]
    assert sum(row["selected"] for row in table) == 1
    assert table[metadata["selected_round"]]["selected"]
    assert table[1]["n_train_rows"] > table[0]["n_train_rows"] == metadata["dagger"][0]["n_train_rows"]
    assert table[0]["n_dagger_rows_total"] == 0 and table[1]["n_dagger_rows_added"] == table[1]["n_dagger_rows_total"] > 0
    assert table[0]["teacher_beta_of_added_rows"] is None and table[1]["teacher_beta_of_added_rows"] == 0.5
    for row in table:
        assert 0.0 <= row["bc_validation"]["masked_accuracy"] <= 1.0
        assert row["on_policy_validation"]["n"] > 0
        assert set(row["policy_validation"]) == {"teacher", "call_station", "maniac", "random"}
        assert row["mean_bb_per_100"] == pytest.approx(np.mean([r["bb_per_100"] for r in row["policy_validation"].values()]))
    assert "paired_vs_bc" in table[1] and "paired_vs_bc" not in table[0]
    best = max(table, key=lambda r: (r["mean_bb_per_100"], -r["round"]))
    assert best["round"] == metadata["selected_round"]


def test_policy_validation_reports_confidence_intervals(metadata: dict[str, Any]) -> None:
    validation = metadata["policy_validation"]
    assert validation["seed_namespace"] == "dataset_val" and validation["pairs_per_opponent"] == 4
    for opponent, result in validation["results"].items():
        low, high = result["ci95_pairs"]
        assert low <= result["bb_per_100"] <= high and result["pairs"] == 4 and result["hands"] == 8, opponent
    chosen = metadata["dagger"][metadata["selected_round"]]
    assert validation["mean_bb_per_100"] == chosen["mean_bb_per_100"]


def test_validation_never_uses_the_tournament_seed_namespace() -> None:
    assert training.VALIDATION_NAMESPACE in SEED_NAMESPACES and training.VALIDATION_NAMESPACE != "eval"


def test_validation_games_pass_the_validation_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain behaviour-cloning run (no DAgger) whose duplicate-deal matches are all played in ``dataset_val``."""
    seen: list[str] = []
    real = training.pair_results

    def spy(*args: Any, **kwargs: Any) -> np.ndarray:
        seen.append(kwargs["namespace"])
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "pair_results", spy)
    run = tiny_run()
    settings = tiny_settings(
        tmp_path,
        run.dataset_dir,
        dagger=training.DaggerSettings(rounds=0, hands_per_round=1, teacher_beta=0.5, n_workers=1, seed=SEED),
        validation=training.ValidationSettings(("call_station", "random"), 3, 10, 200, 1),
    )
    outcome = training.train_ebm(settings, log=lambda _: None)
    assert seen == ["dataset_val", "dataset_val"] and outcome.selected_round == 0
    assert [row["round"] for row in outcome.metadata["dagger"]] == [0]
    assert (tmp_path / "model" / MODEL_FILE).exists()


# --------------------------------------------------------------------------- imitation metrics
def test_masked_imitation_is_well_above_the_majority_baseline(metadata: dict[str, Any]) -> None:
    val = load_dataset(DATASET_NAME, tiny_run().dataset_dir, split="val")
    majority = val["label"].value_counts(normalize=True).max()
    assert metadata["imitation"]["n"] == len(val) == metadata["n_val_rows"]
    assert metadata["imitation"]["masked_accuracy"] > majority + 0.10
    # recompute independently of the pipeline: load the saved agent and score the validation split
    agent = EBMAgent.load(tiny_run().model_dir)
    data = SplitArrays.from_frame(val)
    proba = training.class_probabilities(agent.model, data.X)
    again = evaluate_classifier(proba, data.y, data.legal, data.street)
    assert again["masked_accuracy"] == pytest.approx(metadata["imitation"]["masked_accuracy"])
    assert again["illegal_mass"] >= 0.0 and metadata["imitation"]["confusion_matrix_labels"] == list(ABSTRACT_ACTION_NAMES)


def test_class_weights_in_metadata_match_the_training_labels(metadata: dict[str, Any]) -> None:
    data = load_training_data(DATASET_NAME, directory=tiny_run().dataset_dir)
    base = class_weights(data.train.y, 0.5)  # the behaviour-cloning rows; DAgger rows shift the counts a little
    saved = np.array([metadata["class_weights"][name] for name in ABSTRACT_ACTION_NAMES])
    assert saved.shape == base.shape == (NUM_ABSTRACT_ACTIONS,)
    assert np.argmax(saved) == np.argmax(base) and (saved >= 1.0).sum() >= 1  # the rarest class is the most up-weighted


def test_fit_handles_a_training_set_without_some_actions() -> None:
    """A subsample may lack a rare class (e.g. all-in): probabilities of unseen actions are 0, never an error."""
    run = tiny_run()
    data = load_training_data(DATASET_NAME, directory=run.dataset_dir)
    keep = np.isin(data.train.y, [1, 2])
    subset = SplitArrays(
        X=data.train.X[keep], y=data.train.y[keep], legal=data.train.legal[keep], hand_id=data.train.hand_id[keep],
        street=data.train.street[keep], frame=data.train.frame[keep],
    )  # fmt: skip
    fitted = training.fit_ebm(subset, run.settings.params, 0.5, n_jobs=1, seed=SEED)
    proba = training.class_probabilities(fitted.model, data.val.X[:50])
    assert proba.shape == (50, NUM_ABSTRACT_ACTIONS) and (proba[:, [0, 3, 4]] == 0).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    agent = EBMAgent(fitted.model, {}, "partial")
    x = data.val.X[0]
    assert agent.predict_proba(x)[[0, 3, 4]].sum() == 0
    assert (agent.term_contributions(x, training.AbstractAction.ALL_IN) == 0).all()
