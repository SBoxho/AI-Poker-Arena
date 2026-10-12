"""Show agent explanations and the observation an agent saw.

Agents return an :class:`~poker_arena.agents.base.Explanation` (``summary`` + free-form
``details``). The detail payload differs per architecture, so rendering is *tolerant*: it
recognises the conventional keys below and shows anything else as JSON instead of failing.

* ``probabilities``  mapping action name -> probability (EBM / XGBoost) -> bar chart
* ``top_features``   feature contributions (EBM terms / SHAP values) -> diverging bar chart + table
* ``rules``          list of readable rules (teacher / GA) -> bullet list
* ``genes``          mapping gene name -> value (GA) -> table

Any other mapping of scalars (``numbers``, ``facts`` ...) becomes a table, other scalars a caption.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from poker_arena.agents.base import Explanation
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES
from poker_arena.engine.observation import Observation, Street
from poker_arena.engine.session import StepOutcome
from poker_arena.ui.cards import cards_text
from poker_arena.ui.history import STREET_NAMES, describe_action
from poker_arena.ui.theme import PALETTE, style_figure

EQUITY_SAMPLES = 3000
EQUITY_SEED = 7
MAX_FEATURES_SHOWN = 10
PCT = 100.0

_NAME_KEYS = ("feature", "name", "term", "label")
_VALUE_KEYS = ("value", "feature_value", "x")
_CONTRIBUTION_KEYS = ("contribution", "shap", "shap_value", "score", "effect", "impact", "importance")
_KNOWN_DETAIL_KEYS = ("probabilities", "top_features", "rules")
ACTION_LABELS = {
    "fold": "Fold",
    "check_call": "Check / call",
    "raise_small": "Raise ½ pot",
    "raise_pot": "Raise pot",
    "all_in": "All-in",
}
TABLE_TITLES = {"genes": "Genes", "numbers": "Key numbers", "facts": "Facts"}
SCALAR_TYPES = (int, float, str, bool, np.integer, np.floating)
FLOAT_DIGITS = 3


# ------------------------------------------------------------------ explanation parsing
@dataclass(frozen=True, slots=True)
class Contribution:
    feature: str
    contribution: float
    value: str = ""


@dataclass(frozen=True, slots=True)
class ExplanationView:
    summary: str
    probabilities: dict[str, float]
    contributions: tuple[Contribution, ...]
    rules: tuple[str, ...]
    tables: dict[str, dict[str, Any]]  # title -> {name: scalar}
    scalars: dict[str, Any]
    extra: dict[str, Any]  # anything else, shown as JSON


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    return next((mapping[k] for k in keys if k in mapping), None)


def _parse_probabilities(raw: Any) -> dict[str, float]:
    if isinstance(raw, Mapping):
        return {str(k): float(v) for k, v in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, str) and len(raw) == len(ABSTRACT_ACTION_NAMES):
        return {name: float(p) for name, p in zip(ABSTRACT_ACTION_NAMES, raw)}
    return {}


def _parse_contribution(item: Any) -> Contribution | None:
    if isinstance(item, Mapping):
        name, amount = _first(item, _NAME_KEYS), _first(item, _CONTRIBUTION_KEYS)
        value = _first(item, _VALUE_KEYS)
    elif isinstance(item, Sequence) and not isinstance(item, str) and len(item) in (2, 3):
        name, amount = item[0], item[-1]
        value = item[1] if len(item) == 3 else None
    else:
        return None
    if name is None or not isinstance(amount, (int, float, np.integer, np.floating)):
        return None
    return Contribution(str(name), float(amount), "" if value is None else str(value))


def _parse_contributions(raw: Any) -> tuple[Contribution, ...]:
    if isinstance(raw, Mapping):
        items: Iterable[Any] = [(k, v) for k, v in raw.items()]
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        items = raw
    else:
        return ()
    parsed = (_parse_contribution(i) for i in items)
    return tuple(c for c in parsed if c is not None)


def normalise_explanation(explanation: Explanation | Mapping[str, Any] | None) -> ExplanationView | None:
    """Accept an ``Explanation`` or the ``{"summary", "details"}`` dict stored in a ``DecisionRecord``."""
    if explanation is None:
        return None
    if isinstance(explanation, Explanation):
        summary, details = explanation.summary, dict(explanation.details)
    else:
        summary, details = str(explanation.get("summary", "")), dict(explanation.get("details") or {})
    rules = details.get("rules") or ()
    if isinstance(rules, str):
        rules = (rules,)
    tables: dict[str, dict[str, Any]] = {}
    scalars: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in details.items():
        if key in _KNOWN_DETAIL_KEYS:
            continue
        if isinstance(value, Mapping) and value and all(isinstance(v, SCALAR_TYPES) for v in value.values()):
            tables[TABLE_TITLES.get(key, key.replace("_", " ").capitalize())] = dict(value)
        elif isinstance(value, SCALAR_TYPES):
            scalars[key] = value
        else:
            extra[key] = value
    return ExplanationView(
        summary=summary,
        probabilities=_parse_probabilities(details.get("probabilities")),
        contributions=_parse_contributions(details.get("top_features")),
        rules=tuple(str(r) if not isinstance(r, Mapping) else _rule_text(r) for r in rules),
        tables=tables,
        scalars=scalars,
        extra=extra,
    )


def _rule_text(rule: Mapping[str, Any]) -> str:
    return str(_first(rule, ("text", "rule", "description")) or dict(rule))


# --------------------------------------------------------------------------- figures
def probability_figure(probabilities: Mapping[str, float], chosen: str | None = None) -> go.Figure:
    """Horizontal bars of the model's action probabilities; the chosen action is highlighted."""
    names = list(probabilities)
    values = [probabilities[n] for n in names]
    colours = [PALETTE["accent"] if chosen and n.lower() == chosen.lower() else "#4f6b5d" for n in names]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[ACTION_LABELS.get(n.lower(), n.replace("_", " ").title()) for n in names],
            orientation="h",
            marker_color=colours,
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title={"text": "Action probabilities", "font": {"size": 14}}, showlegend=False)
    return style_figure(fig, height=60 + 34 * len(names))


def contribution_figure(contributions: Sequence[Contribution], limit: int = MAX_FEATURES_SHOWN) -> go.Figure:
    """Signed horizontal bars of the largest feature contributions."""
    top = sorted(contributions, key=lambda c: abs(c.contribution), reverse=True)[:limit]
    fig = go.Figure(
        go.Bar(
            x=[c.contribution for c in top],
            y=[c.feature for c in top],
            orientation="h",
            marker_color=[PALETTE["accent"] if c.contribution >= 0 else PALETTE["red"] for c in top],
            customdata=[c.value for c in top],
            hovertemplate="%{y} (value %{customdata}): %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title={"text": "Top feature contributions", "font": {"size": 14}}, showlegend=False)
    return style_figure(fig, height=70 + 30 * len(top))


# ------------------------------------------------------------------------- rendering
def render_explanation(
    explanation: Explanation | Mapping[str, Any] | None,
    *,
    key: str,
    chosen: str | None = None,
    expanded: bool = False,
) -> None:
    """Summary line plus an expander with charts / rules. ``key`` must be unique per call site."""
    view = normalise_explanation(explanation)
    if view is None:
        st.caption("This agent does not provide an explanation for this decision.")
        return
    st.html(f'<div class="pa-explain">{escape(view.summary)}</div>')
    if not (view.probabilities or view.contributions or view.rules or view.tables or view.scalars or view.extra):
        return
    with st.expander("Why? (details)", expanded=expanded):
        if view.probabilities:
            st.plotly_chart(
                probability_figure(view.probabilities, chosen), key=f"{key}-prob", theme=None, width="stretch"
            )
        if view.contributions:
            st.plotly_chart(
                contribution_figure(view.contributions), key=f"{key}-contrib", theme=None, width="stretch"
            )
            st.dataframe(
                [
                    {"Feature": c.feature, "Value": c.value, "Contribution": round(c.contribution, 4)}
                    for c in sorted(view.contributions, key=lambda c: -abs(c.contribution))
                ],
                hide_index=True,
                width="stretch",
            )
        if view.rules:
            st.markdown("**Rules that fired**\n" + "\n".join(f"- {r}" for r in view.rules))
        for title, table in view.tables.items():
            st.markdown(f"**{title}**")
            st.dataframe(
                [{"Name": k, "Value": _format_scalar(v)} for k, v in table.items()],
                hide_index=True,
                width="stretch",
            )
        if view.scalars:
            st.caption(" · ".join(f"{k.replace('_', ' ')}: {_format_scalar(v)}" for k, v in view.scalars.items()))
        if view.extra:
            st.json(view.extra, expanded=False)


def _format_scalar(value: Any) -> str:
    return f"{value:.{FLOAT_DIGITS}f}" if isinstance(value, (float, np.floating)) else str(value)


# ------------------------------------------------------------------ observation features
@dataclass(frozen=True, slots=True)
class FeatureRow:
    group: str
    name: str
    value: str


def estimate_equity(obs: Observation) -> tuple[float, float] | None:
    """Equity of the observed hand vs one random hand, if the equity package is available."""
    try:
        from poker_arena.equity.monte_carlo import equity_vs_random_with_se
    except ImportError:
        return None
    try:
        rng = np.random.default_rng(EQUITY_SEED)
        return equity_vs_random_with_se(obs.my_hole, obs.board, EQUITY_SAMPLES, rng)
    except ValueError:
        return None


def _pct(x: float) -> str:
    return f"{x * PCT:.0f}%"


def observation_features(
    obs: Observation, *, include_hole: bool = True, include_equity: bool = True
) -> list[FeatureRow]:
    """The values an agent saw, as readable name/value pairs (derived quantities included)."""
    bb = obs.big_blind
    pot_after_call = obs.pot + obs.to_call
    rows = [
        FeatureRow("Situation", "Street", STREET_NAMES[Street(obs.street)]),
        FeatureRow(
            "Situation",
            "Position",
            "Button / small blind (acts last after the flop)" if obs.is_button else "Big blind (acts first after the flop)",
        ),
        FeatureRow("Situation", "Board", cards_text(obs.board) if obs.board else "-"),
    ]
    if include_hole:
        rows.insert(0, FeatureRow("Situation", "Hole cards", cards_text(obs.my_hole)))
    if include_equity and include_hole:
        equity = estimate_equity(obs)
        if equity is not None:
            rows.append(
                FeatureRow("Situation", "Equity vs random hand", f"{equity[0]:.1%} +/- {equity[1]:.1%}")
            )
    rows += [
        FeatureRow("Money", "Pot", f"{obs.pot:,} ({obs.pot / bb:.1f} BB)"),
        FeatureRow("Money", "To call", f"{obs.to_call:,}"),
        FeatureRow(
            "Money",
            "Pot odds (equity needed to call)",
            f"{obs.to_call / pot_after_call:.1%}" if obs.to_call > 0 else "-",
        ),
        FeatureRow("Money", "Own stack / opponent stack", f"{obs.my_stack:,} / {obs.opp_stack:,}"),
        FeatureRow("Money", "Effective stack", f"{obs.effective_stack:,} ({obs.effective_stack / bb:.1f} BB)"),
        FeatureRow(
            "Money",
            "Stack-to-pot ratio",
            f"{obs.effective_stack / obs.pot:.1f}" if obs.pot > 0 else "-",
        ),
        FeatureRow("Money", "Bets this street (own / opp)", f"{obs.my_bet:,} / {obs.opp_bet:,}"),
        FeatureRow("Betting", "Raises this hand / this street", f"{obs.num_raises} / {obs.num_raises_this_street}"),
        FeatureRow("Betting", "Legal actions", _legal_text(obs)),
    ]
    stats = obs.opponent_stats
    rows += [
        FeatureRow("Opponent model", "Hands observed", str(stats.hands)),
        FeatureRow("Opponent model", "VPIP / PFR", f"{_pct(stats.vpip)} / {_pct(stats.pfr)}"),
        FeatureRow("Opponent model", "Postflop aggression", _pct(stats.aggression)),
        FeatureRow("Opponent model", "Fold to bet", _pct(stats.fold_to_bet)),
        FeatureRow("Opponent model", "Showdown rate", _pct(stats.showdown_rate)),
    ]
    return rows


def _legal_text(obs: Observation) -> str:
    legal = obs.legal
    parts = []
    if legal.can_fold:
        parts.append("fold")
    parts.append("check" if legal.can_check else f"call {legal.call_amount:,}")
    if legal.can_raise:
        parts.append(f"raise {legal.min_raise_to:,}-{legal.max_raise_to:,}")
    return ", ".join(parts)


def render_features_html(rows: Sequence[FeatureRow]) -> str:
    out: list[str] = []
    group = None
    for row in rows:
        if row.group != group:
            group = row.group
            out.append(f'<div class="pa-feature-group">{escape(group)}</div>')
        out.append(
            f'<div class="pa-feature"><span>{escape(row.name)}</span><span>{escape(row.value)}</span></div>'
        )
    return f'<div class="pa-features">{"".join(out)}</div>'


def render_observation_features(obs: Observation, *, include_hole: bool = True) -> None:
    """Expander listing what the agent saw when it decided."""
    with st.expander("Observation features (what the agent saw)"):
        st.html(render_features_html(observation_features(obs, include_hole=include_hole)))
        st.caption(
            "Equity is estimated by Monte-Carlo against one random hand - it is not profitability. "
            "Opponent statistics come from previously observed public actions only."
        )


def render_step_detail(step: StepOutcome, mention: str, *, key: str, include_hole: bool = True) -> None:
    """One decision: who did what, the agent's explanation and the observation it saw."""
    event = step.event
    meta = [STREET_NAMES[event.street], f"{step.decision.latency_ms:.2f} ms"]
    if step.decision.abstract:
        meta.insert(1, f"abstract action {step.decision.abstract}")
    st.html(
        f"<div><b>{escape(mention)}</b> {escape(describe_action(event))}<br>"
        f'<span class="pa-muted pa-small">{escape(" · ".join(meta))}</span></div>'
    )
    render_explanation(step.decision.explanation, key=key, chosen=step.decision.abstract)
    render_observation_features(step.observation, include_hole=include_hole)
