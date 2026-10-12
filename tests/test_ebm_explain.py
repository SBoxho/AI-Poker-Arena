"""EBM explanation helpers: every plotted number must equal what the model itself computes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from ebm_helpers import DATASET_NAME, tiny_run

matplotlib.use("Agg")

from poker_arena.agents.ebm_agent import EBMAgent  # noqa: E402
from poker_arena.data import load_dataset  # noqa: E402
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, AbstractAction  # noqa: E402
from poker_arena.explain import ebm as explain  # noqa: E402
from poker_arena.features import FEATURE_NAMES, NUM_FEATURES  # noqa: E402
from poker_arena.training.common import SplitArrays  # noqa: E402

ROWS = 120


@pytest.fixture(scope="module")
def agent() -> EBMAgent:
    return EBMAgent.load(tiny_run().model_dir)


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    val = load_dataset(DATASET_NAME, tiny_run().dataset_dir, split="val")
    return SplitArrays.from_frame(val).X


def action_column(agent: EBMAgent, action: str) -> int:
    return [int(c) for c in agent.model.classes_].index(ABSTRACT_ACTION_NAMES.index(action))


# --------------------------------------------------------------------------------- importance
def test_term_importance_from_data_matches_a_direct_computation(agent: EBMAgent, X: np.ndarray) -> None:
    frame = explain.term_importance(agent, X, max_rows=len(X))
    assert list(frame["importance"]) == sorted(frame["importance"], reverse=True)
    assert len(frame) == len(agent.model.term_features_) and set(frame["kind"]) == {"main", "interaction"}
    assert set(frame["group"]) >= {"interaction"}
    direct = np.abs(agent.model.eval_terms(X)).mean(axis=0)  # (terms, classes)
    row = frame.set_index("term").loc["equity"]
    t = explain.term_names(agent).index("equity")
    assert row["importance"] == pytest.approx(direct[t].mean())
    assert row["raise_small"] == pytest.approx(direct[t, action_column(agent, "raise_small")])


def test_term_importance_without_data_uses_the_training_weights(agent: EBMAgent) -> None:
    frame = explain.term_importance(agent)
    native = agent.model.term_importances()
    by_term = dict(zip(explain.term_names(agent), native, strict=True))
    assert frame.set_index("term")["importance"].to_dict() == pytest.approx(by_term)


def test_importance_frame_from_metadata_is_descending_with_descriptions() -> None:
    frame = explain.importance_frame(tiny_run().outcome.metadata)
    assert list(frame["importance"]) == sorted(frame["importance"], reverse=True)
    assert frame["description"].str.len().min() > 10 and {"main", "interaction"} <= set(frame["kind"])


# ------------------------------------------------------------------------------ shape functions
@pytest.mark.parametrize("feature", ["equity", "pot_odds", "street", "is_button", "to_call_bb"])
def test_shape_function_reproduces_the_model_scores(agent: EBMAgent, X: np.ndarray, feature: str) -> None:
    """The score of the bin that holds a value equals the model's own contribution of that term."""
    frame = explain.shape_frame(agent, feature, X)
    t = explain.term_names(agent).index(feature)
    contributions = agent.model.eval_terms(X[:ROWS])[:, t, :]
    column = FEATURE_NAMES.index(feature)
    cuts = frame["high"].to_numpy()[:-1]
    bins = np.searchsorted(cuts, X[:ROWS, column], side="right")
    for name in ABSTRACT_ACTION_NAMES:
        expected = contributions[:, action_column(agent, name)]
        assert np.allclose(frame[name].to_numpy()[bins], expected), (feature, name)
    assert frame["share"].sum() == pytest.approx(1.0) and (frame["low"] <= frame["high"]).all()
    assert (explain.shape_frame(agent, feature)["share"].sum()) == pytest.approx(1.0)  # training-weight version
    assert (frame[[f"{n}_std" for n in ABSTRACT_ACTION_NAMES]] >= 0).all().all()


# -------------------------------------------------------------------------------- interactions
def test_interaction_surface_reproduces_the_model_scores(agent: EBMAgent, X: np.ndarray) -> None:
    pair = explain.interaction_terms(agent)[0]
    surface = explain.interaction_surface(agent, pair, "check_call")
    first, second = surface.attrs["features"]
    t = explain.term_names(agent).index(pair)
    contribution = agent.model.eval_terms(X[:ROWS])[:, t, action_column(agent, "check_call")]
    columns = np.searchsorted(surface.attrs["x_edges"][1:-1], X[:ROWS, FEATURE_NAMES.index(first)], side="right")
    rows = np.searchsorted(surface.attrs["y_edges"][1:-1], X[:ROWS, FEATURE_NAMES.index(second)], side="right")
    assert np.allclose(surface.to_numpy()[rows, columns], contribution)
    same = explain.interaction_surface(agent, tuple(reversed(pair.split(" x "))), AbstractAction.CHECK_CALL)
    assert same.shape == surface.shape  # the order of the two names does not matter
    with pytest.raises(KeyError):
        explain.interaction_surface(agent, ("equity", "equity"), "fold")


# -------------------------------------------------------------------------------- local reasons
def test_local_contributions_and_logits_are_consistent(agent: EBMAgent, X: np.ndarray) -> None:
    x = X[3]
    logits = explain.local_logits(agent, x)
    assert list(logits["action"]) == list(ABSTRACT_ACTION_NAMES)
    assert logits["probability"].to_numpy() == pytest.approx(agent.predict_proba(x), abs=1e-6)
    for name in ABSTRACT_ACTION_NAMES:
        contributions = explain.local_contributions(agent, x, name)
        assert list(contributions["contribution"].abs()) == sorted(contributions["contribution"].abs(), reverse=True)
        total = contributions["contribution"].sum() + logits.set_index("action").loc[name, "intercept"]
        assert total == pytest.approx(logits.set_index("action").loc[name, "logit"])
    equity = explain.local_contributions(agent, x, "fold").set_index("term").loc["equity"]
    assert equity["value"] == pytest.approx(float(x[FEATURE_NAMES.index("equity")]))


def test_agent_top_features_equal_the_helper(agent: EBMAgent, X: np.ndarray) -> None:
    x = X[7]
    helper = explain.local_contributions(agent, x, AbstractAction.RAISE_SMALL)
    entries = agent.top_features(x, AbstractAction.RAISE_SMALL)
    assert [e["feature"] for e in entries] == list(helper["term"])
    assert [e["contribution"] for e in entries] == pytest.approx(list(helper["contribution"]))


def test_explanation_data_bundles_plain_frames(agent: EBMAgent, X: np.ndarray) -> None:
    bundle = explain.explanation_data(agent, X[0], AbstractAction.CHECK_CALL)
    assert set(bundle) == {"action", "contributions", "logits", "shape_functions"}
    assert bundle["action"] == "check_call" and isinstance(bundle["contributions"], pd.DataFrame)
    assert 1 <= len(bundle["shape_functions"]) <= 3
    assert all(isinstance(frame, pd.DataFrame) for frame in bundle["shape_functions"].values())


# -------------------------------------------------------------------------------------- figures
def test_figures_are_built_from_the_same_numbers(agent: EBMAgent, X: np.ndarray) -> None:
    metadata = tiny_run().outcome.metadata
    importance = explain.importance_figure(explain.importance_frame(metadata), top=8)
    assert isinstance(importance, go.Figure) and sum(len(t.x) for t in importance.data) == 8

    shape = explain.shape_figure(agent, "equity", actions=["fold", AbstractAction.RAISE_POT], X=X)
    names = [t.name for t in shape.data if t.name]
    assert names == ["where the data is", "fold", "raise_pot"]
    assert len(explain.shape_figure(agent, "street").data) > 5  # all five actions by default

    pair = explain.interaction_terms(agent)[0]
    heat = explain.interaction_figure(agent, pair, "raise_small")
    assert np.asarray(heat.data[0].z).shape == explain.interaction_surface(agent, pair, "raise_small").shape

    waterfall = explain.local_explanation_figure(agent, X[5], AbstractAction.CHECK_CALL, top_n=5)
    steps = np.asarray(waterfall.data[0].x)
    logit = explain.local_logits(agent, X[5]).set_index("action").loc["check_call", "logit"]
    assert steps[-1] == pytest.approx(logit) and steps[:-1].sum() == pytest.approx(logit)  # additive, no residual


def test_static_figures_are_written(agent: EBMAgent, X: np.ndarray, tmp_path: Path) -> None:
    paths = explain.save_training_figures(agent, tiny_run().outcome.metadata, tmp_path, X)
    assert [p.name for p in paths] == [
        "ebm_global_importance.png", "ebm_shape_functions.png", "ebm_interactions.png", "ebm_dagger_rounds.png",
    ]  # fmt: skip
    assert all(p.exists() and p.stat().st_size > 5_000 for p in paths)


def test_every_feature_has_a_shape_function(agent: EBMAgent) -> None:
    frames: dict[str, Any] = {name: explain.shape_frame(agent, name) for name in FEATURE_NAMES}
    assert len(frames) == NUM_FEATURES and all(len(f) >= 2 for f in frames.values())
