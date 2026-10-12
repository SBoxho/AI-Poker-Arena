"""XGBoost explanation helpers: SHAP samples, importance tables / figures, local waterfalls, static figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import xgboost as xgb

matplotlib.use("Agg")

from poker_arena.agents.xgb_agent import XGBoostAgent, class_probabilities, shap_contributions
from poker_arena.data import feature_matrix
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, AbstractAction
from poker_arena.explain.xgb import (
    GROUP_COLORS,
    ShapSample,
    gain_importance,
    importance_figure,
    importance_frame,
    load_validation_sample,
    local_contributions,
    local_explanation_figure,
    mean_abs_shap,
    save_training_figures,
    shap_library_check,
    shap_library_values,
)
from poker_arena.features import FEATURE_GROUP_OF, FEATURE_GROUPS, FEATURE_NAMES, NUM_FEATURES

from xgb_helpers import DATASET_NAME, TinyModel, tiny_model

ROWS = 60
TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def tiny() -> TinyModel:
    return tiny_model()


@pytest.fixture(scope="module")
def agent(tiny: TinyModel) -> XGBoostAgent:
    return XGBoostAgent.load(tiny.model_dir)


@pytest.fixture(scope="module")
def X(tiny: TinyModel) -> np.ndarray:
    matrix = feature_matrix(tiny.validation_frame)
    return matrix[np.random.default_rng(1).choice(len(matrix), size=ROWS, replace=False)]


@pytest.fixture(scope="module")
def metadata(tiny: TinyModel) -> dict:
    return json.loads((tiny.model_dir / "metadata.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ SHAP sample
def test_shap_sample_is_additive_and_has_the_layout_the_shap_library_expects(agent: XGBoostAgent, X: np.ndarray) -> None:
    sample = ShapSample.compute(agent.booster, X)
    assert sample.values.shape == (ROWS, len(ABSTRACT_ACTION_NAMES), NUM_FEATURES) and sample.base_values.shape == (ROWS, 5)
    margins = agent.booster.predict(xgb.DMatrix(X, feature_names=list(FEATURE_NAMES)), output_margin=True)
    assert np.abs(sample.margins - margins).max() < TOLERANCE
    assert list(sample.X.columns) == list(FEATURE_NAMES) and np.array_equal(sample.X.to_numpy(), X)
    assert sample.for_action("raise_small").shape == (ROWS, NUM_FEATURES)
    assert np.array_equal(sample.for_action(AbstractAction.RAISE_SMALL), sample.for_action("raise_small"))
    frame = sample.frame(AbstractAction.FOLD)
    assert list(frame.columns) == list(FEATURE_NAMES) and len(frame) == ROWS
    assert sample.mean_abs().shape == (5, NUM_FEATURES) and list(sample.mean_abs().index) == list(ABSTRACT_ACTION_NAMES)


def test_the_shap_library_agrees_with_the_native_contributions(agent: XGBoostAgent, X: np.ndarray) -> None:
    library = shap_library_values(agent.booster, X[:20])
    native = shap_contributions(agent.booster, X[:20])[:, :, :NUM_FEATURES]
    assert library.shape == native.shape and np.abs(library - native).max() < TOLERANCE
    check = shap_library_check(agent.booster, X[:20])
    assert check["rows"] == 20 and check["max_abs_difference"] < TOLERANCE


def test_mean_abs_shap_is_sorted_and_consistent_with_the_sample(agent: XGBoostAgent, X: np.ndarray) -> None:
    sample = ShapSample.compute(agent.booster, X)
    overall, by_action = mean_abs_shap(sample)
    assert list(overall.values()) == sorted(overall.values(), reverse=True) and set(overall) == set(FEATURE_NAMES)
    assert set(by_action) == set(ABSTRACT_ACTION_NAMES)
    assert overall[next(iter(overall))] == pytest.approx(sum(by_action[a][next(iter(overall))] for a in by_action), rel=1e-5)
    gain = gain_importance(agent.booster)
    assert set(gain) == set(FEATURE_NAMES) and sum(gain.values()) == pytest.approx(1.0)


def test_validation_sample_loader_returns_features_and_labels(tiny: TinyModel) -> None:
    sample = load_validation_sample(DATASET_NAME, rows=40, seed=3, directory=tiny.datasets_dir)
    assert len(sample) == 40 and set(FEATURE_NAMES) <= set(sample.columns) and "label" in sample.columns
    assert set(sample["split"]) == {"val"}
    again = load_validation_sample(DATASET_NAME, rows=40, seed=3, directory=tiny.datasets_dir)
    assert sample["hand_id"].tolist() == again["hand_id"].tolist()


# ------------------------------------------------------------------------- global importance
def test_importance_frame_orders_by_shap_and_names_groups(metadata: dict) -> None:
    frame = importance_frame(metadata)
    assert list(frame["feature"][:3]) == list(metadata["global_importance"])[:3]
    assert frame["mean_abs_shap"].is_monotonic_decreasing and len(frame) == NUM_FEATURES
    assert dict(zip(frame["feature"], frame["group"], strict=True)) == dict(FEATURE_GROUP_OF)
    assert set(GROUP_COLORS) == set(FEATURE_GROUPS) and frame["description"].str.len().min() > 10


@pytest.mark.parametrize("metric", ["mean_abs_shap", "gain_share"])
def test_importance_figure_colours_bars_by_group(metadata: dict, metric: str) -> None:
    frame = importance_frame(metadata)
    figure = importance_figure(frame, metric=metric, top=12)
    shown = [feature for trace in figure.data for feature in trace.y]
    assert len(shown) == len(set(shown)) == 12
    for trace in figure.data:
        assert trace.marker.color == GROUP_COLORS[trace.name.replace(" ", "_")]
        assert all(FEATURE_GROUP_OF[feature] == trace.name.replace(" ", "_") for feature in trace.y)
    assert set(shown) == set(frame.sort_values(metric, ascending=False, kind="stable")["feature"].head(12))
    figure.to_dict()  # a valid Plotly figure
    with pytest.raises(ValueError, match="metric"):
        importance_figure(frame, metric="nonsense")


# ---------------------------------------------------------------------------- local explanation
def test_local_contributions_frame_is_sorted_with_groups(agent: XGBoostAgent, X: np.ndarray) -> None:
    frame = local_contributions(agent, X[0], AbstractAction.CHECK_CALL)
    assert list(frame.columns) == ["feature", "value", "shap", "group"] and len(frame) == NUM_FEATURES
    assert frame["shap"].abs().is_monotonic_decreasing
    assert (frame["group"] == frame["feature"].map(FEATURE_GROUP_OF)).all()


@pytest.mark.parametrize("top_n", [3, 10, NUM_FEATURES])
def test_local_waterfall_adds_up_to_the_model_margin(agent: XGBoostAgent, X: np.ndarray, top_n: int) -> None:
    row = X[5]
    margins = agent.booster.predict(xgb.DMatrix(row.reshape(1, -1), feature_names=list(FEATURE_NAMES)), output_margin=True)[0]
    action = AbstractAction(int(np.argmax(margins)))
    figure = local_explanation_figure(agent, row, action, top_n=top_n)
    waterfall = figure.data[0]
    steps, measures = list(waterfall.x), list(waterfall.measure)
    assert measures[0] == "absolute" and measures[-1] == "total"
    assert sum(steps[:-1]) == pytest.approx(steps[-1], abs=TOLERANCE) and steps[-1] == pytest.approx(margins[int(action)], abs=TOLERANCE)
    top_contribution = local_contributions(agent, row, action).iloc[0]
    assert str(top_contribution["feature"]) in waterfall.y[1]
    probability = class_probabilities(agent.booster, row)[0][int(action)]
    assert ABSTRACT_ACTION_NAMES[int(action)] in figure.layout.title.text and f"{probability:.0%}" in figure.layout.title.text
    expected_rows = 2 + min(top_n, NUM_FEATURES) + (1 if top_n < NUM_FEATURES else 0)
    assert len(steps) == expected_rows


# ----------------------------------------------------------------------------- static figures
def test_training_figures_are_written_as_pngs(metadata: dict, tmp_path: Path) -> None:
    paths = save_training_figures(metadata, tmp_path)
    assert sorted(path.name for path in paths) == ["xgb_dagger_rounds.png", "xgb_global_importance.png"]
    assert all(path.stat().st_size > 5_000 for path in paths)
