"""Explanation parsing / charts and the observation-feature view."""

from __future__ import annotations

from ui_helpers import agent_match, human_match, human_obs

from poker_arena.agents.base import Explanation
from poker_arena.ui.explain import (
    Contribution,
    contribution_figure,
    normalise_explanation,
    observation_features,
    probability_figure,
    render_features_html,
)


def test_none_and_plain_summary():
    assert normalise_explanation(None) is None
    view = normalise_explanation(Explanation("Always check/call."))
    assert view is not None and view.summary == "Always check/call."
    assert not (view.probabilities or view.contributions or view.rules or view.tables or view.extra)


def test_dict_form_from_a_decision_record_is_accepted():
    view = normalise_explanation({"summary": "s", "details": {"rules": ["a", "b"]}})
    assert view is not None and view.rules == ("a", "b")
    assert normalise_explanation({"summary": "only", "details": None}).summary == "only"  # type: ignore[union-attr]


def test_probabilities_as_mapping_or_aligned_list():
    mapping = normalise_explanation(Explanation("s", {"probabilities": {"fold": 0.1, "check_call": 0.9}}))
    assert mapping is not None and mapping.probabilities == {"fold": 0.1, "check_call": 0.9}
    aligned = normalise_explanation(Explanation("s", {"probabilities": [0.1, 0.2, 0.3, 0.2, 0.2]}))
    assert aligned is not None and list(aligned.probabilities) == [
        "fold", "check_call", "raise_small", "raise_pot", "all_in",
    ]


def test_feature_contributions_in_the_common_shapes():
    as_dicts = [{"feature": "equity", "value": 0.7, "contribution": 0.4}, {"name": "pot", "shap": -0.2}]
    as_pairs = [("equity", 0.4), ("pot", -0.2)]
    as_triples = [("equity", 0.7, 0.4)]
    as_mapping = {"equity": 0.4, "pot": -0.2}
    for raw in (as_dicts, as_pairs, as_mapping):
        view = normalise_explanation(Explanation("s", {"top_features": raw}))
        assert view is not None
        assert {c.feature: c.contribution for c in view.contributions} == {"equity": 0.4, "pot": -0.2}
    triple = normalise_explanation(Explanation("s", {"top_features": as_triples}))
    assert triple is not None and triple.contributions == (Contribution("equity", 0.4, "0.7"),)
    junk = normalise_explanation(Explanation("s", {"top_features": [("x", "not a number"), 3]}))
    assert junk is not None and junk.contributions == ()


def test_teacher_and_ga_style_payloads_become_tables_and_scalars():
    details = {
        "action": "raise_small",
        "rule": "pre_open",
        "rules": ["preflop", "pre_open"],
        "margin": 0.1234,
        "numbers": {"equity": 0.39, "threshold": 0.18},
        "genes": {"aggression": 1.5, "bluff": 0.2},
        "nested": {"a": {"b": 1}},
    }
    view = normalise_explanation(Explanation("s", details))
    assert view is not None
    assert view.rules == ("preflop", "pre_open")
    assert set(view.tables) == {"Key numbers", "Genes"} and view.tables["Genes"]["bluff"] == 0.2
    assert view.scalars == {"action": "raise_small", "rule": "pre_open", "margin": 0.1234}
    assert view.extra == {"nested": {"a": {"b": 1}}}


def test_figures_have_one_bar_per_item_and_highlight_the_choice():
    fig = probability_figure({"fold": 0.1, "check_call": 0.6, "all_in": 0.3}, chosen="CHECK_CALL")
    bar = fig.data[0]
    assert list(bar.x) == [0.1, 0.6, 0.3] and bar.marker.color[1] != bar.marker.color[0]
    items = [Contribution(f"f{i}", (-1) ** i * i / 10) for i in range(1, 15)]
    top = contribution_figure(items, limit=5).data[0]
    assert len(top.x) == 5 and abs(top.x[0]) >= abs(top.x[-1])
    assert top.marker.color[0] != "" and set(top.marker.color) <= {"#2ecc71", "#ff6b6b"}


def test_observation_features_cover_the_inputs_an_agent_sees():
    match = agent_match(3)
    obs = match.session.observation(match.actor_seat)
    rows = {r.name: r.value for r in observation_features(obs)}
    for name in ("Hole cards", "Street", "Position", "Pot", "To call", "Effective stack", "Legal actions",
                 "VPIP / PFR", "Equity vs random hand"):
        assert name in rows, name
    assert rows["Street"] == "Preflop" and "BB" in rows["Pot"]
    html = render_features_html(observation_features(obs))
    assert html.count("pa-feature-group") == 4 and "Hole cards" in html


def test_private_cards_can_be_left_out():
    obs = human_obs(human_match(2))
    rows = {r.name for r in observation_features(obs, include_hole=False)}
    assert "Hole cards" not in rows and "Equity vs random hand" not in rows


def test_pot_odds_are_only_shown_when_facing_a_bet():
    obs = human_obs(human_match(1, "maniac"))
    rows = {r.name: r.value for r in observation_features(obs, include_equity=False)}
    assert ("%" in rows["Pot odds (equity needed to call)"]) == (obs.to_call > 0)
