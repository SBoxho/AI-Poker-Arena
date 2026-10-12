"""The teacher policy: legality, determinism, agent/policy equivalence, explanations, sanity, strength."""

from __future__ import annotations

import numpy as np
import pytest

from core_helpers import random_midhand_engines
from features_helpers import acting_observation, scripted_engine

from poker_arena.agents.base import Decision
from poker_arena.agents.registry import AgentSpec, create_agent
from poker_arena.agents.teacher import (
    CANONICAL_PARAMS,
    RULE_TEXT,
    TEACHER_STYLES,
    TeacherAgent,
    TeacherDecision,
    TeacherParams,
    TeacherPolicy,
    teacher_params,
)
from poker_arena.data.evaluation import evaluate_duplicate
from poker_arena.engine.actions import AbstractAction
from poker_arena.engine.observation import OpponentStats
from poker_arena.features import FEATURE_INDEX, build_features, builder

RAISES = (AbstractAction.RAISE_SMALL, AbstractAction.RAISE_POT, AbstractAction.ALL_IN)
STATES_PER_STYLE = 700


def decide(*args, stats: OpponentStats | None = None, **kwargs) -> TeacherDecision:
    return TeacherPolicy().decide(acting_observation(scripted_engine(*args, **kwargs), stats))


# ------------------------------------------------------------------------------ legality
@pytest.mark.parametrize("style", sorted(TEACHER_STYLES))
def test_every_decision_is_legal_and_accepted_by_the_engine(style: str) -> None:
    policy = TeacherPolicy(teacher_params(style), style)
    for engine in random_midhand_engines(5 + sorted(TEACHER_STYLES).index(style), STATES_PER_STYLE):
        obs = engine.observation_for(engine.actor_seat)
        decision = policy.decide(obs)
        assert obs.abstract_mask[decision.action], (style, decision)
        engine.act(Decision.from_abstract(obs, decision.action).action)  # raises IllegalActionError otherwise


def test_never_folds_when_a_free_check_exists_and_never_raises_when_it_cannot() -> None:
    policy = TeacherPolicy()
    for engine in random_midhand_engines(91, 600):
        obs = engine.observation_for(engine.actor_seat)
        action = policy.decide(obs).action
        if not obs.legal.can_fold:
            assert action is not AbstractAction.FOLD
        if not obs.legal.can_raise:
            assert action not in RAISES


# ----------------------------------------------------------------------------- determinism
def test_decisions_are_deterministic_and_independent_of_instances_and_caches() -> None:
    observations = [e.observation_for(e.actor_seat) for e in random_midhand_engines(17, 200)]
    first = [TeacherPolicy().decide(o) for o in observations]
    builder._equity_vs_random.cache_clear()
    second = [TeacherPolicy().decide(o) for o in observations]
    assert first == second  # action, rules, margin and numbers


def test_policy_agent_and_feature_path_make_identical_decisions() -> None:
    agent = TeacherAgent()
    policy = agent.policy
    for engine in random_midhand_engines(23, 300):
        obs = engine.observation_for(engine.actor_seat)
        from_obs = policy.decide(obs)
        from_features = policy.decide(build_features(obs), obs.abstract_mask)
        from_agent = agent.act(obs)
        assert from_obs == from_features
        assert from_agent.abstract is from_obs.action
        assert from_agent == Decision.from_abstract(obs, from_obs.action)


def test_policy_requires_a_mask_for_raw_features_and_checks_shapes() -> None:
    obs = acting_observation(scripted_engine("AhAd", "7c2d", "Ks9h4c3d2s"))
    policy = TeacherPolicy()
    with pytest.raises(ValueError, match="mask"):
        policy.decide(build_features(obs))
    with pytest.raises(ValueError):
        policy.decide(build_features(obs)[:-1], obs.abstract_mask)
    with pytest.raises(ValueError):
        policy.decide(build_features(obs), obs.abstract_mask[:-1])


# ---------------------------------------------------------------------------- explanations
def test_explanations_are_readable_and_every_rule_has_a_template() -> None:
    agent = TeacherAgent()
    seen: set[str] = set()
    for engine in random_midhand_engines(29, 900):
        obs = engine.observation_for(engine.actor_seat)
        decision = agent.act(obs)
        explanation = agent.explain(obs, decision)
        assert explanation is not None and explanation.summary.strip()
        assert "{" not in explanation.summary
        details = explanation.details
        assert details["rule"] in RULE_TEXT and details["rules"][-1] == details["rule"]
        assert details["action"] == decision.abstract.name.lower()
        assert 0.0 <= details["confidence"] <= 1.0 and details["margin"] >= 0.0
        assert details["numbers"]["equity"] == pytest.approx(build_features(obs)[FEATURE_INDEX["equity"]], abs=1e-6)
        seen.add(details["rule"])
    assert len(seen) >= 12  # the sample exercises a wide range of rules


def test_explanation_mentions_the_numbers_behind_a_call() -> None:
    # Big blind facing a pot-sized river bet with a strong hand: the price, not a hunch, decides.
    decision = decide("AhKh", "7c2d", "2h7h9sTc3d", [20, "c", "c", "c", "c", "c", "c", 40])
    assert decision.rule in ("call_odds", "fold_odds", "raise_value")
    summary = decision.summary()
    assert "%" in summary and decision.numbers["pot_odds"] == pytest.approx(40 / 120, abs=1e-6)


def test_agent_supports_explanations_and_describes_itself() -> None:
    agent = create_agent(AgentSpec("teacher", {"style": "tight"}, "Tight teacher"))
    assert isinstance(agent, TeacherAgent) and agent.supports_explanations
    assert agent.name == "Tight teacher" and agent.describe()["architecture"]
    assert TeacherAgent().name == "Teacher" and TeacherAgent(style="loose").name == "Teacher-loose"


# ------------------------------------------------------------------------------- parameters
def test_params_are_frozen_documented_and_validated() -> None:
    with pytest.raises(AttributeError):
        CANONICAL_PARAMS.value_equity = 0.5  # type: ignore[misc]
    assert set(CANONICAL_PARAMS.to_dict()) == set(TeacherParams.__dataclass_fields__)
    doc = TeacherParams.__doc__ or ""
    assert all(name in doc for name in TeacherParams.__dataclass_fields__)  # the docstring table lists every threshold
    assert {"canonical", "tight", "loose", "aggressive", "passive"} <= set(TEACHER_STYLES)
    assert teacher_params("tight", {"call_margin": 0.5}).call_margin == 0.5
    with pytest.raises(KeyError):
        teacher_params("nonexistent")
    with pytest.raises(TypeError):
        teacher_params("canonical", {"not_a_parameter": 1.0})


def test_docstring_table_lists_the_actual_defaults() -> None:
    rows = {}
    for line in (TeacherParams.__doc__ or "").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in TeacherParams.__dataclass_fields__:
            rows[parts[0]] = float(parts[1])
    assert set(rows) == set(TeacherParams.__dataclass_fields__)
    assert rows == pytest.approx(CANONICAL_PARAMS.to_dict())


def test_styles_differ_in_behaviour() -> None:
    observations = [e.observation_for(e.actor_seat) for e in random_midhand_engines(31, 400)]
    actions = {
        style: [TeacherPolicy(params, style).decide(o).action for o in observations]
        for style, params in TEACHER_STYLES.items()
    }
    for style in ("tight", "loose", "aggressive", "passive"):
        assert actions[style] != actions["canonical"], style


# ---------------------------------------------------------------------------------- sanity
def test_preflop_button_opens_aces_and_gives_up_the_worst_hand() -> None:
    assert decide("AhAd", "7c2d", "Ks9h4c3d2s").action in RAISES
    assert decide("7h2d", "AcAs", "Ks9h4c3d2s").action in (AbstractAction.FOLD, AbstractAction.CHECK_CALL)


def test_big_blind_checks_trash_over_a_limp_and_raises_aces() -> None:
    assert decide("AhAd", "7c2d", "Ks9h4c3d2s", ["c"]).action is AbstractAction.CHECK_CALL
    assert decide("2h3c", "AcAs", "Ks9h4c3d2s", ["c"]).action in RAISES


def test_short_stack_is_push_or_fold() -> None:
    assert decide("AhAd", "7c2d", "Ks9h4c3d2s", stacks=(80, 80)).action is AbstractAction.ALL_IN
    assert decide("7h2d", "AcAs", "Ks9h4c3d2s", stacks=(80, 80)).action is AbstractAction.FOLD


def test_value_bets_a_made_monster_and_folds_air_to_a_big_bet() -> None:
    # River, big blind first to act with the nut flush (As Ks) on a four-spade board.
    monster = decide("2h2d", "AsKs", "7s9sTs3d4c", [20, "c", "c", "c", "c", "c"])
    assert monster.action in RAISES
    # River, button facing a big bet from the big blind with nothing.
    air = decide("2h3d", "AsKs", "7c9dTs5s4c", [20, "c", "c", "c", "c", "c", 60])
    assert air.action is AbstractAction.FOLD


def test_calls_down_a_maniac_lighter_than_a_tight_opponent() -> None:
    maniac = OpponentStats(hands=300, vpip=0.97, pfr=0.87, aggression=0.95, fold_to_bet=0.0, showdown_rate=0.5)
    nit = OpponentStats(hands=300, vpip=0.2, pfr=0.15, aggression=0.2, fold_to_bet=0.5, showdown_rate=0.3)
    script = [20, "c", "c", "c", "c", "c", "c", 60]  # river, big blind (holding a pair of sevens) faces 60 into 40
    against_maniac = decide("AhKh", "7c2d", "9d7h3sTc4s", script, stats=maniac)
    against_nit = decide("AhKh", "7c2d", "9d7h3sTc4s", script, stats=nit)
    assert against_maniac.numbers["penalty"] < against_nit.numbers["penalty"]
    assert against_maniac.numbers["adj_equity"] >= against_nit.numbers["adj_equity"]


def test_registry_builds_the_teacher_by_kind() -> None:
    assert isinstance(create_agent("teacher"), TeacherAgent)


# ---------------------------------------------------------------------------------- strength
@pytest.mark.slow
@pytest.mark.parametrize(
    "opponent, check",
    [("random", "wins"), ("call_station", "wins"), ("maniac", "does_not_lose")],
)
def test_teacher_beats_or_matches_the_baselines_in_duplicate_play(opponent: str, check: str) -> None:
    result = evaluate_duplicate(AgentSpec("teacher"), AgentSpec(opponent), 500, ("test", "teacher-strength", opponent))
    if check == "wins":
        assert result.ci_low > 0 and result.match_ci_low > 0, str(result)
    else:
        assert result.ci_high > 0 and result.bb_per_100 > -50, str(result)


def test_duplicate_evaluation_of_identical_deterministic_agents_is_exactly_zero() -> None:
    result = evaluate_duplicate(AgentSpec("call_station"), AgentSpec("call_station"), 30, ("test", "mirror"))
    assert result.bb_per_100 == 0.0 and result.ci_low == 0.0 == result.ci_high
    assert np.isfinite(result.std_error)
