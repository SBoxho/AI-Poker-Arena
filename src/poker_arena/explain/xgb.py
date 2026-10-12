"""Explanation helpers for the XGBoost agent: SHAP samples, global importance and local explanations.

The agent explains *a decision* (``XGBoostAgent.top_features``); these helpers answer the broader
questions of the notebooks and the UI: which features drive the model overall, how does a feature's SHAP
value change with its value, and why did the model choose this action in this state.

Everything returns plain data (``DataFrame`` / dict / dataclass) or a Plotly figure. Figures use
transparent-friendly dark styling and colour features by their :data:`FEATURE_GROUPS` group (the group
is also in every hover text, so colour is never the only channel).

Reminder repeated wherever these numbers are shown: SHAP values explain **the model** (a student of the
teacher), measured in log-odds-like margin units of one action; they say nothing about poker skill.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from numpy.typing import NDArray

from poker_arena.agents.xgb_agent import XGBoostAgent, shap_contributions
from poker_arena.data.io import load_dataset
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.explain import notebook_plots as nbp
from poker_arena.features import FEATURE_DESCRIPTIONS, FEATURE_GROUP_OF, FEATURE_NAMES, NUM_FEATURES

#: One colour per feature group (the palette of the research notebooks, plus a cyan for the 7th group).
GROUP_COLORS: dict[str, str] = {
    "hand_strength": nbp.SERIES[0],
    "board": nbp.SERIES[1],
    "position": nbp.SERIES[2],
    "betting": nbp.SERIES[3],
    "money": nbp.SERIES[4],
    "history": nbp.SERIES[5],
    "opponent": "#4fc3d9",
}
POSITIVE_COLOR = nbp.COOL
NEGATIVE_COLOR = nbp.WARM
DEFAULT_TOP_FEATURES = 15
LOCAL_TOP_FEATURES = 10
FIGURE_HEIGHT_PER_BAR = 26
SAMPLE_ROWS = 2_000
UNIT = "SHAP value (log-odds margin)"


# ---------------------------------------------------------------------------------------------
# SHAP on a sample of states
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapSample:
    """Exact TreeSHAP values of a sample of states, for every action class.

    ``values[r, c, f]`` is feature ``f``'s contribution to the raw margin of class ``c`` in row ``r`` and
    ``base_values[r, c]`` the model's expected margin, so ``values.sum(axis=2) + base_values`` are the
    margins the softmax turns into probabilities.
    """

    X: pd.DataFrame
    values: NDArray[np.float32]
    base_values: NDArray[np.float32]

    @classmethod
    def compute(cls, booster: xgb.Booster, X: NDArray[np.floating] | pd.DataFrame) -> "ShapSample":
        matrix = X[list(FEATURE_NAMES)].to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)
        contributions = shap_contributions(booster, matrix)
        return cls(
            X=pd.DataFrame(matrix, columns=list(FEATURE_NAMES)),
            values=contributions[:, :, :NUM_FEATURES],
            base_values=contributions[:, :, NUM_FEATURES],
        )

    @property
    def margins(self) -> NDArray[np.float32]:
        """Raw class margins, shape ``(rows, 5)``."""
        return self.values.sum(axis=2) + self.base_values

    def for_action(self, action: AbstractAction | str | int) -> NDArray[np.float32]:
        """``(rows, features)`` SHAP matrix of one action - the layout ``shap.summary_plot`` expects."""
        return self.values[:, _class_index(action), :]

    def frame(self, action: AbstractAction | str | int) -> pd.DataFrame:
        """SHAP values of one action as a DataFrame (columns = feature names)."""
        return pd.DataFrame(self.for_action(action), columns=list(FEATURE_NAMES))

    def mean_abs(self) -> pd.DataFrame:
        """Mean |SHAP| per action (rows) and feature (columns)."""
        return pd.DataFrame(
            np.abs(self.values).mean(axis=0), index=list(ABSTRACT_ACTION_NAMES), columns=list(FEATURE_NAMES)
        )


def _class_index(action: AbstractAction | str | int) -> int:
    if isinstance(action, str):
        return ABSTRACT_ACTION_NAMES.index(action)
    return int(action)


def gain_importance(booster: xgb.Booster) -> dict[str, float]:
    """Share of the total split gain per feature (sums to 1, descending); unused features get 0."""
    raw = booster.get_score(importance_type="total_gain")
    total = float(sum(raw.values())) or 1.0
    shares = {name: float(raw.get(name, 0.0)) / total for name in FEATURE_NAMES}
    return dict(sorted(shares.items(), key=lambda item: -item[1]))


def mean_abs_shap(sample: ShapSample) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Mean |SHAP| per feature: summed over the five actions, and per action (both descending).

    The sum over actions is what ``shap.summary_plot`` stacks for a multi-class model; the per-action
    view answers "what drives *raising* rather than folding".
    """
    per_action = sample.mean_abs()
    total = per_action.sum(axis=0).sort_values(ascending=False, kind="stable")
    overall = {name: float(value) for name, value in total.items()}
    by_action = {
        action: {name: float(value) for name, value in row.sort_values(ascending=False, kind="stable").items()}
        for action, row in per_action.iterrows()
    }
    return overall, by_action


def load_validation_sample(
    dataset_name: str, rows: int = SAMPLE_ROWS, seed: int = 0, directory: Path | None = None
) -> pd.DataFrame:
    """A random sample of the validation split (feature columns + ``label``, ``legal_*``...) for SHAP plots."""
    frame = load_dataset(dataset_name, directory, split="val")
    return frame.sample(n=min(rows, len(frame)), random_state=seed).reset_index(drop=True)


def shap_library_values(booster: xgb.Booster, X: NDArray[np.floating]) -> NDArray[np.float32]:
    """``shap.TreeExplainer`` values in the layout of :class:`ShapSample` ``(rows, classes, features)``.

    The notebooks use the ``shap`` library for its plots; this is the bridge that lets a test (and
    :func:`shap_library_check`) assert that the library and XGBoost's native ``pred_contribs`` agree.
    """
    import shap

    values = np.asarray(shap.TreeExplainer(booster).shap_values(np.asarray(X, dtype=np.float32)))
    if values.shape[-1] == NUM_ABSTRACT_ACTIONS:  # recent shap: (rows, features, classes)
        values = np.moveaxis(values, -1, 1)
    return values.astype(np.float32)


def shap_library_check(booster: xgb.Booster, X: NDArray[np.floating]) -> dict[str, Any]:
    """Largest absolute difference between ``shap.TreeExplainer`` and the native TreeSHAP on ``X``."""
    native = shap_contributions(booster, X)[:, :, :NUM_FEATURES]
    return {
        "rows": int(len(X)),
        "shap_version": importlib_metadata.version("shap"),
        "max_abs_difference": float(np.abs(native - shap_library_values(booster, X)).max()),
        "note": "shap.TreeExplainer delegates to XGBoost's own pred_contribs for XGBoost models, so equality is expected",
    }


# ---------------------------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------------------------
def importance_frame(metadata: Mapping[str, Any]) -> pd.DataFrame:
    """Per-feature importances saved in ``metadata.json`` (descending mean |SHAP|), with group and meaning."""
    shap_values, gain = metadata["global_importance"], metadata["global_importance_gain"]
    frame = pd.DataFrame(
        {
            "feature": list(FEATURE_NAMES),
            "group": [FEATURE_GROUP_OF[name] for name in FEATURE_NAMES],
            "mean_abs_shap": [shap_values[name] for name in FEATURE_NAMES],
            "gain_share": [gain[name] for name in FEATURE_NAMES],
            "description": [FEATURE_DESCRIPTIONS[name] for name in FEATURE_NAMES],
        }
    )
    return frame.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)


_IMPORTANCE_LABELS = {
    "mean_abs_shap": "mean |SHAP| (summed over the five actions)",
    "gain_share": "share of total split gain",
}


def style_figure(fig: go.Figure, title: str, height: int) -> go.Figure:
    """The shared dark look (see :func:`poker_arena.explain.notebook_plots.style_plotly`) plus a title."""
    nbp.style_plotly(fig, height=height)
    fig.update_layout(title={"text": title, "x": 0.0, "xanchor": "left"}, margin={"l": 20, "r": 30, "t": 60, "b": 50})
    fig.update_yaxes(automargin=True)  # long feature names (and their values) get the room they need
    return fig


def importance_figure(frame: pd.DataFrame, metric: str = "mean_abs_shap", top: int = DEFAULT_TOP_FEATURES) -> go.Figure:
    """Horizontal bars of the ``top`` most important features, coloured by feature group.

    ``metric`` is ``"mean_abs_shap"`` (what the model uses, measured on predictions) or ``"gain_share"``
    (what the trees split on). The two rankings usually agree on the leaders and differ in the tail.
    """
    if metric not in _IMPORTANCE_LABELS:
        raise ValueError(f"metric must be one of {sorted(_IMPORTANCE_LABELS)}")
    shown = frame.sort_values(metric, ascending=False, kind="stable").head(top)
    fig = go.Figure()
    for group, color in GROUP_COLORS.items():
        part = shown[shown["group"] == group]
        if part.empty:
            continue
        fig.add_bar(
            x=part[metric],
            y=part["feature"],
            orientation="h",
            name=group.replace("_", " "),
            marker_color=color,
            customdata=np.stack([part["group"], part["description"]], axis=1),
            hovertemplate="%{y} (%{customdata[0]})<br>%{x:.4f}<br>%{customdata[1]}<extra></extra>",
        )
    fig.update_layout(barmode="overlay", legend={"title": "feature group"})
    fig.update_yaxes(categoryorder="array", categoryarray=list(shown["feature"])[::-1])
    fig.update_xaxes(title=_IMPORTANCE_LABELS[metric])
    return style_figure(fig, f"XGBoost: top {len(shown)} features by {_IMPORTANCE_LABELS[metric]}", FIGURE_HEIGHT_PER_BAR * len(shown) + 150)


# ---------------------------------------------------------------------------------------------
# Local explanation
# ---------------------------------------------------------------------------------------------
def local_contributions(agent: XGBoostAgent, x: NDArray[np.floating], action: AbstractAction) -> pd.DataFrame:
    """SHAP contributions of one state towards one action, |contribution| descending, with feature values."""
    row = agent.top_features(np.asarray(x, dtype=np.float32), action)
    frame = pd.DataFrame(row).rename(columns={"contribution": "shap"})
    frame["group"] = frame["feature"].map(FEATURE_GROUP_OF)
    return frame


def local_explanation_figure(
    agent: XGBoostAgent, x: NDArray[np.floating], action: AbstractAction, top_n: int = LOCAL_TOP_FEATURES
) -> go.Figure:
    """Waterfall of one decision: the expected margin, the ``top_n`` biggest SHAP contributions, the margin.

    The bars add up exactly to the raw margin of ``action`` (the additivity property of SHAP); the
    margin is what the softmax turns into the probability shown in the title. Blue pushes towards the
    action, red away from it.
    """
    vector = np.asarray(x, dtype=np.float32)
    contributions = shap_contributions(agent.booster, vector)[0, int(action)]
    bias, shap_row = float(contributions[NUM_FEATURES]), contributions[:NUM_FEATURES]
    order = np.argsort(-np.abs(shap_row), kind="stable")
    shown, rest = order[:top_n], order[top_n:]
    labels = ["expected margin (bias)"] + [f"{FEATURE_NAMES[i]} = {vector[i]:.3g}" for i in shown]
    steps = [bias] + [float(shap_row[i]) for i in shown]
    measures = ["absolute"] + ["relative"] * len(shown)
    if len(rest):
        labels.append(f"{len(rest)} other features")
        steps.append(float(shap_row[rest].sum()))
        measures.append("relative")
    labels.append("model margin")
    steps.append(float(bias + shap_row.sum()))
    measures.append("total")
    probability = float(agent.predict_proba(vector)[int(action)])
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=labels,
            x=steps,
            measure=measures,
            increasing={"marker": {"color": POSITIVE_COLOR}},
            decreasing={"marker": {"color": NEGATIVE_COLOR}},
            totals={"marker": {"color": nbp.ACCENT}},
            connector={"line": {"color": nbp.MUTED}},
            hovertemplate="%{y}<br>%{x:+.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=UNIT)
    name = ABSTRACT_ACTION_NAMES[int(action)]
    return style_figure(
        fig, f"Why {name}? model probability {probability:.0%} (margin {steps[-1]:+.2f})", FIGURE_HEIGHT_PER_BAR * len(labels) + 130
    )


# ---------------------------------------------------------------------------------------------
# Static figures written by the training script
# ---------------------------------------------------------------------------------------------
def save_training_figures(metadata: Mapping[str, Any], directory: Path) -> list[Path]:
    """Write ``xgb_global_importance.png`` and ``xgb_dagger_rounds.png`` (static, for README / notebooks)."""
    import matplotlib.pyplot as plt

    nbp.apply_style()
    paths: list[Path] = []

    frame = importance_frame(metadata).head(DEFAULT_TOP_FEATURES)
    fig, ax = plt.subplots(figsize=(9.0, 0.38 * len(frame) + 1.6))
    colors = [GROUP_COLORS[g] for g in frame["group"]]
    ax.barh(frame["feature"], frame["mean_abs_shap"], color=colors, height=nbp.BAR_THICKNESS)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |SHAP| over validation states, summed over the five actions")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in dict.fromkeys(frame["group"])]
    ax.legend(handles, [g.replace("_", " ") for g in dict.fromkeys(frame["group"])], loc="lower right", title="feature group")
    nbp.title(ax, "What the XGBoost agent looks at", "global SHAP importance; explains the model, not poker skill")
    paths.append(nbp.savefig("xgb_global_importance", fig, directory=directory))
    plt.close(fig)

    rounds = metadata["dagger"]
    opponents = list(metadata["policy_validation"]["opponents"])
    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(opponents) + 2.0))
    labels = opponents
    series = {}
    for row in rounds:
        estimates = [row["policy_validation"][o]["bb_per_100"] for o in opponents]
        lows = [row["policy_validation"][o]["ci95_pairs"][0] for o in opponents]
        highs = [row["policy_validation"][o]["ci95_pairs"][1] for o in opponents]
        series[f"round {row['round']}" + (" (selected)" if row["selected"] else "")] = (estimates, lows, highs)
    nbp.forest_plot(ax, labels, series, x_label="validation BB/100 of XGBoost against the opponent (duplicate deals, 95% CI)")
    nbp.title(ax, "Does DAgger change how XGBoost plays?", "round 0 = behaviour cloning; hollow marker = CI contains 0")
    paths.append(nbp.savefig("xgb_dagger_rounds", fig, directory=directory))
    plt.close(fig)
    return paths
