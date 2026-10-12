"""Explanation helpers for the EBM agent: global importance, shape functions, interactions, local reasons.

An EBM is additive, so *everything* it does can be read off its tables: a one-feature term is a
**shape function** (a step function of the feature, one curve per action) and a two-feature term is an
**interaction surface**. These helpers return plain pandas data first (so notebooks can combine them with
InterpretML's own ``show(ebm.explain_global())`` widgets) and Plotly figures second.

Why this module exists next to InterpretML's own explanations: for a *multi-class* model
``explain_global()`` silently drops the pairwise terms ("we cannot currently visualize pairs"), yet
the interactions carry a large part of what the EBM learned from the teacher. :func:`interaction_surface`
and :func:`interaction_figure` fill that gap, and :func:`term_importance` ranks main effects and
interactions on one scale.

Units: every score is a contribution to the **log-odds (logit) of one action** in the model's softmax.
Each term is centred on the training distribution (its weighted mean is 0 for every action), so a bar
means "this value pushes the action up/down relative to the average training state". They explain the
*model* - a student of the teacher - and say nothing about poker skill.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from interpret.glassbox import ExplainableBoostingClassifier
from numpy.typing import NDArray

from poker_arena.agents.ebm_agent import EBMAgent, term_label
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.explain import notebook_plots as nbp
from poker_arena.features import FEATURE_DESCRIPTIONS, FEATURE_GROUP_OF, FEATURE_INDEX, FEATURE_NAMES

INTERACTION_GROUP = "interaction"
#: One colour per feature group (same as the XGBoost figures) plus a neutral light grey for interaction terms.
GROUP_COLORS: dict[str, str] = {
    "hand_strength": nbp.SERIES[0],
    "board": nbp.SERIES[1],
    "position": nbp.SERIES[2],
    "betting": nbp.SERIES[3],
    "money": nbp.SERIES[4],
    "history": nbp.SERIES[5],
    "opponent": "#4fc3d9",
    INTERACTION_GROUP: "#c9d4ce",
}
ACTION_COLORS = dict(zip(ABSTRACT_ACTION_NAMES, nbp.SERIES, strict=False))  # five actions, six palette colours
UNIT = "contribution to the action's log-odds"
DEFAULT_TOP_TERMS = 15
LOCAL_TOP_TERMS = 10
IMPORTANCE_SAMPLE_ROWS = 20_000
FIGURE_HEIGHT_PER_BAR = 26
STATIC_SHAPE_PANELS = 6
STATIC_INTERACTION_PANELS = 4
_SHAPE_CLIP_QUANTILE = 0.98  # static shape plots clip the y-range: a few extreme tail bins would flatten the rest
_MISSING_BIN, _UNSEEN_BIN = 0, -1  # first / last row of every term tensor; the real bins sit in between

ModelLike = ExplainableBoostingClassifier | EBMAgent


# ---------------------------------------------------------------------------------------------
# Low-level access to the term tables
# ---------------------------------------------------------------------------------------------
def as_model(model: ModelLike) -> ExplainableBoostingClassifier:
    """The fitted EBM of an agent (or the EBM itself)."""
    return model.model if isinstance(model, EBMAgent) else model


def class_index(action: AbstractAction | str | int) -> int:
    if isinstance(action, str):
        return ABSTRACT_ACTION_NAMES.index(action)
    return int(action)


def _class_columns(model: ExplainableBoostingClassifier) -> NDArray[np.int64]:
    """Action index of every score column (a class absent from training has no column)."""
    return np.array([int(c) for c in model.classes_], dtype=np.int64)


def _expand_to_actions(model: ExplainableBoostingClassifier, scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(..., classes seen)`` -> ``(..., 5)`` with zeros for actions the model never saw."""
    full = np.zeros((*scores.shape[:-1], NUM_ABSTRACT_ACTIONS), dtype=np.float64)
    full[..., _class_columns(model)] = scores
    return full


def term_names(model: ModelLike) -> list[str]:
    """``"equity"`` / ``"equity x street"`` for every additive term, in model order."""
    return [term_label(tuple(f)) for f in as_model(model).term_features_]


def _find_term(model: ExplainableBoostingClassifier, features: tuple[str, ...]) -> int:
    wanted = tuple(sorted(FEATURE_INDEX[f] for f in features))
    for t, stored in enumerate(model.term_features_):
        if tuple(sorted(stored)) == wanted:
            return t
    raise KeyError(f"The model has no term for {' x '.join(features)}; available: {term_names(model)}")


def _cuts(model: ExplainableBoostingClassifier, feature: int, n_dims: int) -> NDArray[np.float64]:
    """Bin boundaries of ``feature`` as used inside a term with ``n_dims`` features (pairs use coarser bins)."""
    levels = model.bins_[feature]
    return np.asarray(levels[min(len(levels), n_dims) - 1], dtype=np.float64)


def _edges(model: ExplainableBoostingClassifier, feature: int, n_dims: int) -> NDArray[np.float64]:
    """Lower/upper edges of the real bins: feature minimum, the cuts, feature maximum."""
    low, high = model.feature_bounds_[feature]
    return np.concatenate([[low], _cuts(model, feature, n_dims), [high]])


def _bin_shares(model: ExplainableBoostingClassifier, term: int, X: NDArray[np.floating] | None) -> NDArray[np.float64]:
    """Share of the data per real bin of a one-feature term: from ``X`` if given, else the training weights."""
    feature = model.term_features_[term][0]
    n_bins = len(_cuts(model, feature, 1)) + 1
    if X is None:
        weights = np.asarray(model.bin_weights_[term], dtype=np.float64)[1:-1]
    else:
        index = np.searchsorted(_cuts(model, feature, 1), np.asarray(X)[:, feature], side="right")
        weights = np.bincount(index, minlength=n_bins).astype(np.float64)
    return weights / max(weights.sum(), 1.0)


# ---------------------------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------------------------
def term_importance(
    model: ModelLike, X: NDArray[np.floating] | None = None, max_rows: int = IMPORTANCE_SAMPLE_ROWS, seed: int = 0
) -> pd.DataFrame:
    """Importance of every term: mean |term score| averaged over the five actions, descending.

    With ``X`` (feature rows, e.g. validation states) the average is over those states - "how much does
    this term move the logits where the game actually goes"; without it, InterpretML's own bin-weighted
    average over the training data is used. Columns: ``term``, ``kind`` (main/interaction), ``features``,
    ``group``, ``importance`` and one column per action with that action's mean |score|.
    """
    ebm = as_model(model)
    if X is None:
        per_action = np.empty((len(ebm.term_features_), len(ebm.classes_)))
        for t, scores in enumerate(ebm.term_scores_):
            weights = np.asarray(ebm.bin_weights_[t], dtype=np.float64)
            axes = list(range(weights.ndim))
            per_action[t] = np.tensordot(weights, np.abs(scores), axes=(axes, axes)) / weights.sum()
    else:
        rows = np.asarray(X, dtype=np.float32)
        if len(rows) > max_rows:
            rows = rows[np.random.default_rng(seed).choice(len(rows), size=max_rows, replace=False)]
        per_action = np.abs(ebm.eval_terms(rows)).mean(axis=0)
    full = _expand_to_actions(ebm, per_action)
    features = [tuple(FEATURE_NAMES[i] for i in f) for f in ebm.term_features_]
    frame = pd.DataFrame(
        {
            "term": term_names(ebm),
            "kind": ["main" if len(f) == 1 else INTERACTION_GROUP for f in features],
            "features": features,
            "group": [FEATURE_GROUP_OF[f[0]] if len(f) == 1 else INTERACTION_GROUP for f in features],
            "importance": full[:, _class_columns(ebm)].mean(axis=1),
            **{name: full[:, i] for i, name in enumerate(ABSTRACT_ACTION_NAMES)},
        }
    )
    return frame.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)


def importance_frame(metadata: Mapping[str, Any]) -> pd.DataFrame:
    """The importances saved in ``metadata.json`` (descending), with kind, group and meaning of each term."""
    rows = []
    for term, value in metadata["global_importance"].items():
        features = tuple(term.split(" x "))
        rows.append(
            {
                "term": term,
                "kind": "main" if len(features) == 1 else INTERACTION_GROUP,
                "group": FEATURE_GROUP_OF[features[0]] if len(features) == 1 else INTERACTION_GROUP,
                "importance": float(value),
                "description": FEATURE_DESCRIPTIONS[features[0]] if len(features) == 1 else " | ".join(FEATURE_DESCRIPTIONS[f] for f in features),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)


def style_figure(fig: go.Figure, title: str, height: int) -> go.Figure:
    """The shared dark look (:func:`poker_arena.explain.notebook_plots.style_plotly`) plus a left-aligned title."""
    nbp.style_plotly(fig, height=height)
    fig.update_layout(title={"text": title, "x": 0.0, "xanchor": "left"}, margin={"l": 60, "r": 30, "t": 60, "b": 50})
    fig.update_yaxes(automargin=True)  # interaction labels ("preflop_percentile x equity_minus_pot_odds") are long
    return fig


def importance_figure(frame: pd.DataFrame, top: int = DEFAULT_TOP_TERMS) -> go.Figure:
    """Horizontal bars of the ``top`` most important terms, coloured by feature group (interactions in grey)."""
    shown = frame.sort_values("importance", ascending=False, kind="stable").head(top)
    fig = go.Figure()
    for group, color in GROUP_COLORS.items():
        part = shown[shown["group"] == group]
        if part.empty:
            continue
        fig.add_bar(
            x=part["importance"],
            y=part["term"],
            orientation="h",
            name=group.replace("_", " "),
            marker_color=color,
            hovertemplate="%{y} (" + group + ")<br>mean |score| %{x:.3f}<extra></extra>",
        )
    fig.update_layout(barmode="overlay", legend={"title": "feature group", "orientation": "h", "y": -0.12})
    fig.update_yaxes(categoryorder="array", categoryarray=list(shown["term"])[::-1])
    fig.update_xaxes(title="mean |term score| (log-odds), averaged over the five actions")
    return style_figure(fig, f"EBM: top {len(shown)} terms by importance", FIGURE_HEIGHT_PER_BAR * len(shown) + 150)


# ---------------------------------------------------------------------------------------------
# Shape functions (one feature)
# ---------------------------------------------------------------------------------------------
def shape_frame(model: ModelLike, feature: str, X: NDArray[np.floating] | None = None) -> pd.DataFrame:
    """The shape function of a feature: one row per bin with its value range and every action's score.

    Columns: ``low``, ``high`` (bin edges), ``share`` (fraction of the data in the bin - from ``X`` when
    given, else the training weights), then per action ``<action>`` (term score) and ``<action>_std``
    (standard deviation of that score over the EBM's bagged fits: wide = unreliable).
    """
    ebm = as_model(model)
    term = _find_term(ebm, (feature,))
    index = ebm.term_features_[term][0]
    edges = _edges(ebm, index, 1)
    scores = _expand_to_actions(ebm, np.asarray(ebm.term_scores_[term])[1:-1])
    spread = _expand_to_actions(ebm, np.asarray(ebm.standard_deviations_[term])[1:-1])
    data: dict[str, Any] = {"low": edges[:-1], "high": edges[1:], "share": _bin_shares(ebm, term, X)}
    for i, name in enumerate(ABSTRACT_ACTION_NAMES):
        data[name], data[f"{name}_std"] = scores[:, i], spread[:, i]
    return pd.DataFrame(data)


def shape_figure(
    model: ModelLike,
    feature: str,
    actions: Sequence[AbstractAction | str | int] | None = None,
    X: NDArray[np.floating] | None = None,
) -> go.Figure:
    """Step-function plot of a feature's effect on the chosen actions (default: all five) with a ±1 std band.

    The grey bars underneath show where the data lives (``X`` or the training weights): a wiggle in a
    region with almost no data is noise, not a finding. Binary features have just two bins.
    """
    frame = shape_frame(model, feature, X)
    chosen = [ABSTRACT_ACTION_NAMES[class_index(a)] for a in actions] if actions is not None else list(ABSTRACT_ACTION_NAMES)
    x = np.repeat(np.append(frame["low"].to_numpy(), frame["high"].to_numpy()[-1]), 2)[1:-1]
    fig = go.Figure()
    density = frame["share"] / (frame["high"] - frame["low"]).clip(lower=1e-9)
    fig.add_bar(
        x=(frame["low"] + frame["high"]) / 2, y=density / density.max(), width=frame["high"] - frame["low"],
        name="where the data is", marker_color=nbp.GRID, yaxis="y2", hoverinfo="skip",
    )  # fmt: skip
    for name in chosen:
        y = np.repeat(frame[name].to_numpy(), 2)
        spread = np.repeat(frame[f"{name}_std"].to_numpy(), 2)
        color = ACTION_COLORS[name]
        fig.add_scatter(x=x, y=y + spread, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(
            x=x, y=y - spread, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=_with_alpha(color, 0.18),
            showlegend=False, hoverinfo="skip",
        )  # fmt: skip
        fig.add_scatter(
            x=x, y=y, mode="lines", name=name, line={"width": 2.2, "color": color},
            hovertemplate=name + ": %{y:+.2f}<extra></extra>",
        )  # fmt: skip
    fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 4], "showgrid": False, "visible": False})
    fig.update_xaxes(title=f"{feature} (bin edges from the fitted model)")
    fig.update_yaxes(title=UNIT)
    group = FEATURE_GROUP_OF[feature].replace("_", " ")
    return style_figure(fig, f"EBM shape function: {feature} ({group})", 420)


def _with_alpha(color: str, alpha: float) -> str:
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


# ---------------------------------------------------------------------------------------------
# Interactions (two features)
# ---------------------------------------------------------------------------------------------
def interaction_terms(model: ModelLike) -> list[str]:
    """Names of the pairwise terms (``"equity x street"``) in model order."""
    ebm = as_model(model)
    return [name for name, f in zip(term_names(ebm), ebm.term_features_, strict=True) if len(f) == 2]


def interaction_surface(model: ModelLike, term: str | tuple[str, str], action: AbstractAction | str | int) -> pd.DataFrame:
    """Score surface of an interaction term for one action: rows = bins of the 2nd feature, columns = 1st.

    ``term`` is ``"equity x street"`` or ``("equity", "street")`` (either order). The index/columns are
    bin mid-points; ``DataFrame.attrs`` holds ``"features"`` (column feature, row feature) and the bin edges.
    """
    ebm = as_model(model)
    names = tuple(term.split(" x ")) if isinstance(term, str) else tuple(term)
    t = _find_term(ebm, names)
    first, second = ebm.term_features_[t]
    column = list(_class_columns(ebm)).index(class_index(action))
    surface = np.asarray(ebm.term_scores_[t])[1:-1, 1:-1, column]  # drop missing/unseen bins of both axes
    x_edges, y_edges = _edges(ebm, first, 2), _edges(ebm, second, 2)
    frame = pd.DataFrame(
        surface.T, index=(y_edges[:-1] + y_edges[1:]) / 2, columns=(x_edges[:-1] + x_edges[1:]) / 2
    )
    frame.attrs.update(features=(FEATURE_NAMES[first], FEATURE_NAMES[second]), x_edges=x_edges, y_edges=y_edges)
    return frame


def interaction_figure(model: ModelLike, term: str | tuple[str, str], action: AbstractAction | str | int) -> go.Figure:
    """Heat map of one interaction surface (blue = pushes the action up, red = down, grey = no effect)."""
    surface = interaction_surface(model, term, action)
    first, second = surface.attrs["features"]
    limit = float(np.abs(surface.to_numpy()).max()) or 1.0
    fig = go.Figure(
        go.Heatmap(
            x=surface.columns, y=surface.index, z=surface.to_numpy(), zmin=-limit, zmax=limit, zmid=0.0,
            colorscale=[[0.0, nbp.WARM], [0.5, nbp.NEUTRAL], [1.0, nbp.COOL]],
            colorbar={"title": "log-odds"},
            hovertemplate=f"{first}=%{{x:.3g}}<br>{second}=%{{y:.3g}}<br>%{{z:+.2f}}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(title=first)
    fig.update_yaxes(title=second)
    name = ABSTRACT_ACTION_NAMES[class_index(action)]
    return style_figure(fig, f"EBM interaction {first} x {second}: effect on {name}", 460)


# ---------------------------------------------------------------------------------------------
# Local explanation (one decision)
# ---------------------------------------------------------------------------------------------
def local_contributions(model: ModelLike, x: NDArray[np.floating], action: AbstractAction | str | int) -> pd.DataFrame:
    """Every term's contribution to the log-odds of ``action`` for one state, |contribution| descending.

    Columns: ``term``, ``kind``, ``group``, ``value`` (feature value or tuple of values), ``contribution``.
    """
    ebm = as_model(model)
    vector = np.asarray(x, dtype=np.float32)
    scores = _expand_to_actions(ebm, ebm.eval_terms(vector.reshape(1, -1))[0])[:, class_index(action)]
    features = [tuple(f) for f in ebm.term_features_]
    frame = pd.DataFrame(
        {
            "term": term_names(ebm),
            "kind": ["main" if len(f) == 1 else INTERACTION_GROUP for f in features],
            "group": [FEATURE_GROUP_OF[FEATURE_NAMES[f[0]]] if len(f) == 1 else INTERACTION_GROUP for f in features],
            "value": [float(vector[f[0]]) if len(f) == 1 else tuple(float(vector[i]) for i in f) for f in features],
            "contribution": scores,
        }
    )
    order = np.argsort(-np.abs(scores), kind="stable")
    return frame.iloc[order].reset_index(drop=True)


def local_logits(model: ModelLike, x: NDArray[np.floating]) -> pd.DataFrame:
    """Intercept, logit and probability of the five actions for one state (``logit = intercept + sum of terms``)."""
    ebm = as_model(model)
    vector = np.asarray(x, dtype=np.float32).reshape(1, -1)
    intercept = _expand_to_actions(ebm, np.asarray(ebm.intercept_, dtype=np.float64))
    logit = intercept + _expand_to_actions(ebm, ebm.eval_terms(vector)[0]).sum(axis=0)
    probability = np.zeros(NUM_ABSTRACT_ACTIONS)
    probability[_class_columns(ebm)] = ebm.predict_proba(vector)[0]
    return pd.DataFrame({"action": ABSTRACT_ACTION_NAMES, "intercept": intercept, "logit": logit, "probability": probability})


def explanation_data(model: ModelLike, x: NDArray[np.floating], action: AbstractAction | str | int) -> dict[str, Any]:
    """Everything a local explanation shows, as plain data: contributions, per-action logits, shape curves.

    ``{"action", "contributions" (DataFrame), "logits" (DataFrame), "shape_functions" (feature -> DataFrame
    of the shape function of the three most influential main effects)}``.
    """
    contributions = local_contributions(model, x, action)
    main = contributions[contributions["kind"] == "main"].head(3)
    return {
        "action": ABSTRACT_ACTION_NAMES[class_index(action)],
        "contributions": contributions,
        "logits": local_logits(model, x),
        "shape_functions": {term: shape_frame(model, term) for term in main["term"]},
    }


def local_explanation_figure(
    model: ModelLike, x: NDArray[np.floating], action: AbstractAction | str | int, top_n: int = LOCAL_TOP_TERMS
) -> go.Figure:
    """Waterfall of one decision: intercept, the ``top_n`` largest term contributions, the rest, the logit.

    The bars add up exactly to the action's logit (the EBM is additive - no approximation). Blue pushes
    towards the action, red away from it; interaction terms show both feature values.
    """
    contributions = local_contributions(model, x, action)
    summary = local_logits(model, x).iloc[class_index(action)]
    shown, rest = contributions.head(top_n), contributions.iloc[top_n:]
    labels = ["intercept (average state)"] + [f"{r.term} = {_format_value(r.value)}" for r in shown.itertuples()]
    steps = [float(summary["intercept"])] + shown["contribution"].tolist()
    measures = ["absolute"] + ["relative"] * len(shown)
    if len(rest):
        labels.append(f"{len(rest)} smaller terms")
        steps.append(float(rest["contribution"].sum()))
        measures.append("relative")
    labels.append("logit")
    steps.append(float(summary["logit"]))
    measures.append("total")
    fig = go.Figure(
        go.Waterfall(
            orientation="h", y=labels, x=steps, measure=measures,
            increasing={"marker": {"color": nbp.COOL}}, decreasing={"marker": {"color": nbp.WARM}},
            totals={"marker": {"color": nbp.ACCENT}}, connector={"line": {"color": nbp.MUTED}},
            hovertemplate="%{y}<br>%{x:+.3f}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=UNIT)
    name = ABSTRACT_ACTION_NAMES[class_index(action)]
    title = f"Why {name}? model probability {summary['probability']:.0%} (logit {summary['logit']:+.2f})"
    return style_figure(fig, title, FIGURE_HEIGHT_PER_BAR * len(labels) + 130)


def _format_value(value: float | tuple[float, ...]) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(f"{v:.3g}" for v in value) + ")"
    return f"{value:.3g}"


# ---------------------------------------------------------------------------------------------
# Static figures written by the training script (matplotlib PNGs: GitHub renders them)
# ---------------------------------------------------------------------------------------------
def save_training_figures(
    model: ModelLike, metadata: Mapping[str, Any], directory: Path, X: NDArray[np.floating] | None = None
) -> list[Path]:
    """Write ``ebm_global_importance.png``, ``ebm_shape_functions.png``, ``ebm_interactions.png`` and
    ``ebm_dagger_rounds.png``; ``X`` (validation rows) supplies the data-density context."""
    import matplotlib.pyplot as plt

    nbp.apply_style()
    ebm = as_model(model)
    paths: list[Path] = []

    frame = importance_frame(metadata).head(DEFAULT_TOP_TERMS + 5)
    fig, ax = plt.subplots(figsize=(9.0, 0.38 * len(frame) + 1.6))
    ax.barh(frame["term"], frame["importance"], color=[GROUP_COLORS[g] for g in frame["group"]], height=nbp.BAR_THICKNESS)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |term score| over validation states, averaged over the five actions (log-odds)")
    groups = list(dict.fromkeys(frame["group"]))
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in groups], [g.replace("_", " ") for g in groups], loc="lower right")
    nbp.title(ax, "What the EBM looks at", "global term importance; explains the model, not poker skill")
    paths.append(nbp.savefig("ebm_global_importance", fig, directory=directory))
    plt.close(fig)

    main_terms = [t for t in importance_frame(metadata)["term"] if " x " not in t][:STATIC_SHAPE_PANELS]
    fig, axes = plt.subplots(
        2, (len(main_terms) + 1) // 2, figsize=(4.4 * ((len(main_terms) + 1) // 2), 6.8), squeeze=False, layout="constrained"
    )
    for ax, feature in zip(axes.ravel(), main_terms, strict=False):
        shape = shape_frame(ebm, feature, X)
        edges = np.append(shape["low"].to_numpy(), shape["high"].to_numpy()[-1])
        scores = shape[list(ABSTRACT_ACTION_NAMES)].to_numpy()
        for i, name in enumerate(ABSTRACT_ACTION_NAMES):
            ax.stairs(scores[:, i], edges, color=ACTION_COLORS[name], label=name, linewidth=1.8)
        limit = float(np.quantile(np.abs(scores), _SHAPE_CLIP_QUANTILE)) * 1.2 or 1.0
        ax.set_ylim(-limit, limit)
        ax.axhline(0.0, color=nbp.MUTED, linewidth=0.8)
        ax.set_xlabel(feature)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels), fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("log-odds contribution")
    fig.suptitle("EBM shape functions of the most important features (one curve per action)", color=nbp.TEXT, x=0.01, ha="left", fontweight="bold")
    paths.append(nbp.savefig("ebm_shape_functions", fig, directory=directory))
    plt.close(fig)

    pairs = interaction_terms(ebm)
    ranked = [t for t in importance_frame(metadata)["term"] if t in pairs][:STATIC_INTERACTION_PANELS]
    if ranked:
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), squeeze=False, layout="constrained")
        for ax, term in zip(axes.ravel(), ranked, strict=False):
            strongest = term_importance(ebm, X).set_index("term").loc[term, list(ABSTRACT_ACTION_NAMES)].astype(float).idxmax()
            surface = interaction_surface(ebm, term, strongest)
            limit = float(np.abs(surface.to_numpy()).max()) or 1.0
            mesh = ax.pcolormesh(surface.attrs["x_edges"], surface.attrs["y_edges"], surface.to_numpy(), cmap=nbp.DIVERGING, vmin=-limit, vmax=limit)
            ax.grid(False)
            ax.set_xlabel(surface.attrs["features"][0])
            ax.set_ylabel(surface.attrs["features"][1])
            ax.set_title(f"{term}: effect on {strongest}", loc="left", fontsize=9)
            fig.colorbar(mesh, ax=ax, label="log-odds")
        for ax in axes.ravel()[len(ranked) :]:
            ax.axis("off")
        fig.suptitle("EBM interaction surfaces (strongest action shown per pair)", color=nbp.TEXT, x=0.01, ha="left", fontweight="bold")
        paths.append(nbp.savefig("ebm_interactions", fig, directory=directory))
        plt.close(fig)

    paths.append(_save_dagger_figure(metadata, directory))
    return paths


def _save_dagger_figure(metadata: Mapping[str, Any], directory: Path) -> Path:
    """Per DAgger round: imitation on the dataset vs on the model's own states, and validation BB/100."""
    import matplotlib.pyplot as plt

    rounds = metadata["dagger"]
    opponents = list(metadata["policy_validation"]["opponents"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 0.55 * len(opponents) + 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
    index = [row["round"] for row in rounds]
    left.plot(index, [row["bc_validation"]["masked_accuracy"] for row in rounds], marker="o", label="dataset validation states")
    left.plot(index, [row["on_policy_validation"]["masked_accuracy"] for row in rounds], marker="s", label="states the EBM itself visits")
    left.set_xticks(index)
    left.set_xlabel("DAgger round (0 = behaviour cloning)")
    left.set_ylabel("legal-masked agreement with the teacher")
    left.legend(loc="lower right")
    nbp.title(left, "Imitation (not poker skill)", "covariate shift = gap between the two lines")
    series = {}
    for row in rounds:
        results = row["policy_validation"]
        series[f"round {row['round']}" + (" (selected)" if row["selected"] else "")] = (
            [results[o]["bb_per_100"] for o in opponents],
            [results[o]["ci95_pairs"][0] for o in opponents],
            [results[o]["ci95_pairs"][1] for o in opponents],
        )
    nbp.forest_plot(right, opponents, series, x_label="validation BB/100 of the EBM against the opponent (duplicate deals, 95% CI)")
    nbp.title(right, "Does DAgger change how the EBM plays?", "hollow marker = CI contains 0")
    path = nbp.savefig("ebm_dagger_rounds", fig, directory=directory)
    plt.close(fig)
    return path
