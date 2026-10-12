"""XGBoost training pipeline: artifact contract, DAgger bookkeeping, honest validation, determinism."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import xgboost as xgb

from poker_arena.agents.learned import validate_metadata
from poker_arena.agents.registry import AgentSpec
from poker_arena.agents.xgb_agent import MODEL_FILE, XGBoostAgent
from poker_arena.config import REPO_ROOT, GameConfig
from poker_arena.data import load_meta
from poker_arena.data.evaluation import pair_results
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES
from poker_arena.features import FEATURE_NAMES
from poker_arena.training.common import class_weights, load_training_data
from poker_arena.training.xgb import dataset_config_from_meta, fit_booster, portable_config, select_round

from xgb_helpers import DAGGER_ROUNDS, DATASET_NAME, TEST_OPPONENTS, VALIDATION_PAIRS, TinyModel, tiny_model

REQUIRED_KEYS = (
    "artifact_schema_version feature_schema_hash feature_names class_names created_at library_versions dataset config "
    "seeds training_time_s n_train_rows n_val_rows class_weights imitation dagger policy_validation global_importance "
    "global_importance_gain best_iteration"
).split()
MAJORITY_MARGIN = 0.25  # the tiny model must beat "always predict the commonest action" by this much


@pytest.fixture(scope="module")
def tiny() -> TinyModel:
    return tiny_model()


@pytest.fixture(scope="module")
def metadata(tiny: TinyModel) -> dict:
    return json.loads((tiny.model_dir / "metadata.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ the artifact
def test_artifact_files_and_metadata_contract(tiny: TinyModel, metadata: dict) -> None:
    for name in (MODEL_FILE, "metadata.json", "config.json"):
        assert (tiny.model_dir / name).is_file()
    assert (tiny.model_dir / MODEL_FILE).stat().st_size < 15_000_000
    missing = [key for key in REQUIRED_KEYS if key not in metadata]
    assert not missing, missing
    validate_metadata(metadata, tiny.model_dir)  # exactly what the agent checks on load
    assert metadata["feature_names"] == list(FEATURE_NAMES) and metadata["class_names"] == list(ABSTRACT_ACTION_NAMES)
    assert metadata["library_versions"]["xgboost"] == xgb.__version__
    assert set(metadata["class_weights"]) == set(ABSTRACT_ACTION_NAMES)
    assert metadata["training_time_s"] > 0 and metadata["n_train_rows"] > 0
    config = json.loads((tiny.model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["booster_params"]["objective"] == "multi:softprob" and config["booster_params"]["tree_method"] == "hist"
    json.dumps(metadata)  # everything is plain JSON


def test_dataset_provenance_matches_the_dataset_that_was_used(tiny: TinyModel, metadata: dict) -> None:
    meta = load_meta(DATASET_NAME, tiny.datasets_dir)
    assert metadata["dataset"]["name"] == DATASET_NAME
    assert metadata["dataset"]["content_sha256"] == meta["content_sha256"]
    assert metadata["dataset"]["rows"] == meta["rows"]
    assert metadata["dataset"]["schema_hash"] == meta["schema_hash"] == metadata["feature_schema_hash"]
    assert metadata["n_val_rows"] == meta["splits"]["val"]["rows"]
    assert metadata["n_bc_train_rows"] == meta["splits"]["train"]["rows"]


def test_the_saved_model_holds_exactly_the_best_iteration_and_no_later_trees(tiny: TinyModel, metadata: dict) -> None:
    agent = XGBoostAgent.load(tiny.model_dir)
    assert agent.booster.num_boosted_rounds() == metadata["best_iteration"] + 1 == metadata["n_trees_per_class"]


def test_imitation_metrics_are_masked_and_beat_the_majority_baseline(tiny: TinyModel, metadata: dict) -> None:
    imitation = metadata["imitation"]
    labels = tiny.validation_frame["label"].to_numpy()
    majority = float(np.bincount(labels).max() / len(labels))
    assert imitation["n"] == len(labels)
    assert imitation["masked_accuracy"] > majority + MAJORITY_MARGIN
    assert imitation["masked_accuracy"] >= imitation["accuracy"] - 1e-12  # masking can only remove illegal predictions
    assert 0.0 <= imitation["illegal_mass"] <= 1.0 and len(imitation["confusion_matrix"]) == len(ABSTRACT_ACTION_NAMES)


def test_global_importance_is_ordered_complete_and_normalised(metadata: dict) -> None:
    shap_values, gain = metadata["global_importance"], metadata["global_importance_gain"]
    assert set(shap_values) == set(gain) == set(FEATURE_NAMES)
    assert list(shap_values.values()) == sorted(shap_values.values(), reverse=True) and min(shap_values.values()) >= 0
    assert list(gain.values()) == sorted(gain.values(), reverse=True)
    assert sum(gain.values()) == pytest.approx(1.0)
    by_class = metadata["global_importance_by_class"]
    assert set(by_class) == set(ABSTRACT_ACTION_NAMES)
    for name in FEATURE_NAMES:  # the overall number is the sum of the per-class numbers
        assert shap_values[name] == pytest.approx(sum(by_class[c][name] for c in ABSTRACT_ACTION_NAMES), rel=1e-4)
    assert metadata["shap_library_check"]["max_abs_difference"] < 1e-4


# -------------------------------------------------------------------------------------- DAgger
def test_dagger_table_has_one_row_per_round_and_consistent_bookkeeping(tiny: TinyModel, metadata: dict) -> None:
    table = metadata["dagger"]
    assert [row["round"] for row in table] == list(range(DAGGER_ROUNDS + 1))
    selected = [row["round"] for row in table if row["selected"]]
    assert selected == [metadata["selected_round"]]
    base = metadata["n_bc_train_rows"]
    assert table[0]["n_train_rows"] == base and table[0]["n_dagger_rows_total"] == 0 and "paired_vs_bc" not in table[0]
    for row in table[1:]:
        assert row["n_dagger_rows_total"] == row["n_train_rows"] - base > 0  # on-policy rows really were aggregated
        assert set(row["paired_vs_bc"]) == set(TEST_OPPONENTS)
    for row in table:
        results = row["policy_validation"]
        assert set(results) == set(TEST_OPPONENTS)
        assert row["mean_bb_per_100"] == pytest.approx(np.mean([r["bb_per_100"] for r in results.values()]))
        assert all(r["pairs"] == VALIDATION_PAIRS and r["hands"] == 2 * VALIDATION_PAIRS for r in results.values())
        for part in ("bc_validation", "on_policy_validation"):  # both imitation measurements exist for EVERY round
            assert 0.0 <= row[part]["masked_accuracy"] <= 1.0 and row[part]["n"] > 0
    assert metadata["n_train_rows"] == table[metadata["selected_round"]]["n_train_rows"]
    final = metadata["policy_validation"]
    assert final["mean_bb_per_100"] == pytest.approx(table[metadata["selected_round"]]["mean_bb_per_100"])
    assert set(final["results"]) == set(TEST_OPPONENTS) and final["seed_namespace"] == "dataset_val"
    assert metadata["imitation_on_policy"]["n"] == table[metadata["selected_round"]]["on_policy_validation"]["n"]


def test_round_selection_prefers_the_best_mean_and_breaks_ties_towards_the_earlier_round() -> None:
    def history(*means: float) -> list[dict]:
        return [{"round": k, "mean_bb_per_100": mean} for k, mean in enumerate(means)]

    assert select_round(history(10.0, 30.0, 20.0)) == 1
    assert select_round(history(5.0, 5.0, 5.0)) == 0
    assert select_round(history(1.0, 7.0, 7.0, 3.0)) == 1
    assert select_round(history(-50.0)) == 0
    assert select_round(history(-9.0, -3.0, -3.0)) == 1


def test_every_round_has_a_loadable_artifact_and_the_selected_one_is_copied(tiny: TinyModel, metadata: dict) -> None:
    for k in range(DAGGER_ROUNDS + 1):
        XGBoostAgent.load(tiny.work_dir / f"round_{k}")
    selected = tiny.work_dir / f"round_{metadata['selected_round']}"
    assert (selected / MODEL_FILE).read_bytes() == (tiny.model_dir / MODEL_FILE).read_bytes()


# ---------------------------------------------------------------------------------- honesty
def test_validation_never_uses_the_tournament_eval_seed_namespace(tiny: TinyModel) -> None:
    assert len(tiny.pair_namespaces) == (DAGGER_ROUNDS + 1) * len(TEST_OPPONENTS)
    assert set(tiny.pair_namespaces) == {"dataset_val"} and "eval" not in tiny.pair_namespaces


def test_the_namespace_of_duplicate_evaluation_defaults_to_eval_and_changes_the_deals() -> None:
    kwargs = {"spec_a": AgentSpec("random"), "spec_b": AgentSpec("call_station"), "n_pairs": 8, "seed_parts": ("test", "ns")}
    default = pair_results(**kwargs)
    assert np.array_equal(default, pair_results(**kwargs, namespace="eval"))
    assert not np.array_equal(default, pair_results(**kwargs, namespace="dataset_val"))
    assert np.array_equal(pair_results(**kwargs, namespace="dataset_val"), pair_results(**kwargs, namespace="dataset_val"))


def test_a_dataset_for_other_game_rules_is_refused(tiny: TinyModel) -> None:
    meta = load_meta(DATASET_NAME, tiny.datasets_dir)
    assert dataset_config_from_meta(meta, GameConfig()).game == GameConfig()
    with pytest.raises(ValueError, match="regenerate the dataset"):
        dataset_config_from_meta(meta, GameConfig(starting_stack=500, small_blind=5, big_blind=10))


def test_paths_in_the_saved_configuration_are_relative_to_the_repository() -> None:
    saved = portable_config({"OUTPUT_DIR": REPO_ROOT / "artifacts" / "models" / "xgb", "SEED": 3, "NAME": "x"})
    assert saved == {"OUTPUT_DIR": "artifacts/models/xgb", "SEED": 3, "NAME": "x"}


# ------------------------------------------------------------------------------- determinism
def test_fitting_is_reproducible_for_a_fixed_seed(tiny: TinyModel) -> None:
    settings = replace(tiny.settings, n_jobs=1, max_train_rows=None)
    data = load_training_data(DATASET_NAME, None, settings.seed, directory=tiny.datasets_dir)
    args = (data.train, data.val, settings.params, settings.class_weight_power, 1, settings.seed)
    first, second = fit_booster(*args), fit_booster(*args)
    assert bytes(first.booster.save_raw()) == bytes(second.booster.save_raw())
    assert first.best_iteration == second.best_iteration
    assert np.allclose(first.class_weights, class_weights(data.train.y, settings.class_weight_power))


def test_early_stopping_keeps_the_best_iteration_of_the_validation_curve(tiny: TinyModel) -> None:
    data = load_training_data(DATASET_NAME, None, 0, directory=tiny.datasets_dir)
    fitted = fit_booster(data.train, data.val, tiny.settings.params, 0.5, 1, 0)
    curve = fitted.learning_curve
    assert curve["round"] == sorted(curve["round"]) and len(curve["round"]) == len(curve["val_mlogloss"])
    assert fitted.booster.num_boosted_rounds() == fitted.best_iteration + 1 <= tiny.settings.params.n_estimators

