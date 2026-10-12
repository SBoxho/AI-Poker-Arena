"""Teacher-labelled decision datasets: behaviour cloning data and DAgger aggregation.

Design decisions
----------------
* **Rows = decision states of the learner-controlled seat** (seat 0). Every hand is played through
  :class:`~poker_arena.engine.session.HandSession` / :func:`play_hand`; the acting seat sees only
  its :class:`Observation` (through :class:`RecordingAgent`), features come from
  :func:`poker_arena.features.build_features`, and the label is always the *canonical teacher's*
  abstract action for that very feature vector.
* **State coverage.** Imitating a policy from its own trajectories leaves the learner blind to the
  states its own mistakes lead to (covariate shift). So the acting seat's *behaviour* policy is
  sampled per hand from ``behavior_mix`` (teacher, teacher with epsilon exploration, uniformly random,
  style presets, baselines) and plays against an opponent drawn per match from ``opponent_mix``; a
  fraction of hands starts with shortened stacks. The label stays the canonical teacher's.
* **DAgger** (Dataset Aggregation, Ross et al. 2011) is the principled fix for the same problem:
  :func:`collect_on_policy` lets a *trained learner* (any registered agent spec, e.g. ``ebm`` or
  ``xgboost``) drive the acting seat while the teacher labels every state it visits; the new rows
  (``source = "dagger_round_k"``) are concatenated with the earlier data and the model is refit.
  ``teacher_beta`` optionally mixes the teacher back in per decision (the classic beta schedule).
* **Matches, not isolated hands.** The opponent-statistics features need history, so hands are
  grouped into *matches* (a fixed opponent, persistent trackers, lengths from ``match_lengths`` so
  the model sees both unknown and well-known opponents). A match is the unit of parallel work and
  every random choice derives from ``derive_seed(namespace, ..., hand_index)``, therefore the output
  is byte-identical for any number of worker processes.
* **Splits by hand.** Train and validation use disjoint seed namespaces (``dataset_train`` /
  ``dataset_val``; DAgger uses ``dagger``), so no hand - and no deal - ever appears in two splits.
  ``hand_id`` is the 63-bit hand seed, globally unique across namespaces.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from poker_arena.agents.base import Agent
from poker_arena.agents.registry import AgentSpec, create_agent
from poker_arena.agents.teacher import CANONICAL_PARAMS, TeacherPolicy
from poker_arena.config import GameConfig, derive_seed
from poker_arena.data.behaviors import (
    LEARNER,
    POLICY_NAMES,
    MixedPolicyAgent,
    RecordingAgent,
    RowBuffer,
    build_policy,
    policy_spec,
    validate_mix,
)
from poker_arena.data.columns import CATEGORICAL_COLUMNS, COLUMNS, HAND_ID_COLUMN, LEGAL_COLUMNS
from poker_arena.data.io import content_hash, library_versions
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS
from poker_arena.engine.deals import Deal
from poker_arena.engine.history import HandRecord
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.engine.session import play_hand
from poker_arena.features import FEATURE_NAMES, NUM_FEATURES

SPLIT_NAMESPACES = {"train": "dataset_train", "val": "dataset_val"}
DAGGER_NAMESPACE = "dagger"
SOURCE_BC = "bc"

DEFAULT_BEHAVIOR_MIX = {
    "teacher": 0.30, "teacher_eps": 0.20, "random": 0.10,
    "tight": 0.10, "loose": 0.10, "aggressive": 0.10, "passive": 0.10,
}
DEFAULT_OPPONENT_MIX = {
    "random": 0.14, "call_station": 0.14, "maniac": 0.14, "teacher": 0.18,
    "tight": 0.10, "loose": 0.10, "aggressive": 0.10, "passive": 0.10,
}


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that defines a dataset besides the number of hands (all of it is saved in the meta file)."""

    game: GameConfig = field(default_factory=GameConfig)
    seed: int = 42
    behavior_mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BEHAVIOR_MIX))
    opponent_mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OPPONENT_MIX))
    exploration_epsilon: float = 0.15
    match_lengths: tuple[int, ...] = (30, 100, 300, 1000)
    short_stack_probability: float = 0.10
    min_short_stack_bb: int = 10

    def __post_init__(self) -> None:
        validate_mix(self.behavior_mix, POLICY_NAMES, "behavior_mix")
        validate_mix(self.opponent_mix, POLICY_NAMES, "opponent_mix")
        if not 0.0 <= self.exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be in [0, 1]")
        if not self.match_lengths or min(self.match_lengths) < 1:
            raise ValueError("match_lengths needs positive hand counts")
        if not 0.0 <= self.short_stack_probability <= 1.0:
            raise ValueError("short_stack_probability must be in [0, 1]")
        if self.min_short_stack_bb * self.game.big_blind > self.game.starting_stack:
            raise ValueError("min_short_stack_bb exceeds the starting stack")

    def to_dict(self) -> dict:
        return {
            "game": self.game.to_dict(),
            "seed": self.seed,
            "behavior_mix": dict(self.behavior_mix),
            "opponent_mix": dict(self.opponent_mix),
            "exploration_epsilon": self.exploration_epsilon,
            "match_lengths": list(self.match_lengths),
            "short_stack_probability": self.short_stack_probability,
            "min_short_stack_bb": self.min_short_stack_bb,
        }


@dataclass(frozen=True)
class CollectionSpec:
    """One collection job: which seeds, which behaviour, which opponents (picklable)."""

    config: DatasetConfig
    namespace: str
    seed_parts: tuple[int | str, ...]
    split: str
    source: str
    opponent_mix: dict[str, float]
    learner: AgentSpec | None = None
    teacher_beta: float = 0.0


@dataclass(frozen=True, slots=True)
class MatchPlan:
    """A run of consecutive hands against one opponent with persistent opponent statistics."""

    match_index: int
    first_hand: int
    n_hands: int
    opponent: str


@dataclass
class MatchResult:
    """Output of one match: its labelled rows and (optionally) the full hand records."""

    plan: MatchPlan
    buffer: RowBuffer
    records: tuple[HandRecord, ...] = ()
    hand_ids: tuple[int, ...] = ()


def plan_matches(spec: CollectionSpec, n_hands: int) -> list[MatchPlan]:
    """Split ``n_hands`` into matches (lengths cycle through ``match_lengths``) with sampled opponents."""
    lengths = spec.config.match_lengths
    names = sorted(spec.opponent_mix)
    weights = [spec.opponent_mix[n] for n in names]
    plans: list[MatchPlan] = []
    first = 0
    while first < n_hands:
        index = len(plans)
        rng = random.Random(derive_seed(spec.namespace, *spec.seed_parts, "match", index))
        length = min(lengths[index % len(lengths)], n_hands - first)
        plans.append(MatchPlan(index, first, length, rng.choices(names, weights)[0]))
        first += length
    return plans


def _learner_agent(spec: CollectionSpec) -> Agent:
    assert spec.learner is not None
    agent = create_agent(spec.learner)
    if spec.teacher_beta > 0.0:
        teacher = create_agent(policy_spec("teacher"))
        return MixedPolicyAgent(agent, teacher, spec.teacher_beta, agent.name)
    return agent


def run_match_plan(spec: CollectionSpec, plan: MatchPlan, keep_records: bool = False) -> MatchResult:
    """Play one match and record the learner seat's labelled decisions (top-level for pickling)."""
    config = spec.config
    game = config.game
    ns, parts = spec.namespace, spec.seed_parts
    teacher = TeacherPolicy()
    opponent = build_policy(plan.opponent, config.exploration_epsilon)
    opponent.reset(derive_seed(ns, *parts, "opponent", plan.match_index))
    trackers = (OpponentTracker(), OpponentTracker())
    behaviors = sorted(config.behavior_mix)
    weights = [config.behavior_mix[b] for b in behaviors]
    agents: dict[str, Agent] = {}
    buffer = RowBuffer()
    records: list[HandRecord] = []
    hand_ids: list[int] = []
    short_low = config.min_short_stack_bb * game.big_blind

    for hand in range(plan.first_hand, plan.first_hand + plan.n_hands):
        hand_id = derive_seed(ns, *parts, "hand", hand)
        rng = random.Random(hand_id)
        deal = Deal.from_rng(rng)
        behavior = LEARNER if spec.learner is not None else rng.choices(behaviors, weights)[0]
        stacks = None
        if rng.random() < config.short_stack_probability:
            stacks = (rng.randint(short_low, game.starting_stack), rng.randint(short_low, game.starting_stack))
        agent_seed = rng.getrandbits(62)
        if behavior not in agents:
            agents[behavior] = (
                _learner_agent(spec) if behavior == LEARNER else build_policy(behavior, config.exploration_epsilon)
            )
        agents[behavior].reset(agent_seed)
        recorder = RecordingAgent(agents[behavior], teacher, buffer, hand_id, hand, behavior, plan.opponent)
        seats: list[Agent] = [recorder, opponent]
        record = play_hand(
            game, deal, hand % 2, seats, start_stacks=stacks, hand_number=hand, trackers=trackers
        )
        hand_ids.append(hand_id)
        if keep_records:
            records.append(record)
    return MatchResult(plan, buffer, tuple(records), tuple(hand_ids))


def _run_match_task(task: tuple[CollectionSpec, MatchPlan]) -> MatchResult:
    return run_match_plan(*task)


def assemble_frame(results: Sequence[MatchResult], split: str, source: str) -> pd.DataFrame:
    """Concatenate match results (ordered by match index) into the dataset table."""
    ordered = sorted(results, key=lambda r: r.plan.match_index)
    buffers = [r.buffer for r in ordered if len(r.buffer)]
    n_rows = sum(len(b) for b in buffers)
    columns: dict[str, np.ndarray | list] = {}
    features = np.vstack([np.vstack(b.features) for b in buffers]) if buffers else np.empty((0, NUM_FEATURES), np.float32)
    masks = np.vstack([np.vstack(b.mask) for b in buffers]) if buffers else np.empty((0, NUM_ABSTRACT_ACTIONS), bool)

    def cat(name: str) -> list:
        return [value for b in buffers for value in getattr(b, name)]

    columns["hand_id"] = np.asarray(cat("hand_id"), dtype=np.int64)
    columns["hand_index"] = np.asarray(cat("hand_index"), dtype=np.int32)
    columns["decision_idx"] = np.asarray(cat("decision_idx"), dtype=np.int16)
    columns["split"] = [split] * n_rows
    columns["source"] = [source] * n_rows
    columns["behavior"] = cat("behavior")
    columns["opponent"] = cat("opponent")
    for j, name in enumerate(FEATURE_NAMES):
        columns[name] = features[:, j]
    columns["label"] = np.asarray(cat("label"), dtype=np.int8)
    for j, name in enumerate(LEGAL_COLUMNS):
        columns[name] = masks[:, j]
    columns["behavior_action"] = np.asarray(cat("behavior_action"), dtype=np.int8)
    columns["teacher_margin"] = np.asarray(cat("margin"), dtype=np.float32)
    columns["teacher_rule"] = cat("rule")
    frame = pd.DataFrame(columns, columns=list(COLUMNS))
    for name in CATEGORICAL_COLUMNS:
        frame[name] = frame[name].astype("category")
    return frame


def collect_rows(
    spec: CollectionSpec, n_hands: int, n_workers: int = 1
) -> pd.DataFrame:
    """Play ``n_hands`` hands for ``spec`` and return the labelled decision table."""
    plans = plan_matches(spec, n_hands)
    tasks = [(spec, plan) for plan in sorted(plans, key=lambda p: -p.n_hands)]  # long matches first
    if n_workers <= 1 or len(tasks) <= 1:
        results = [_run_match_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
            results = list(pool.map(_run_match_task, tasks, chunksize=1))
    return assemble_frame(results, spec.split, spec.source)


@dataclass(frozen=True)
class GeneratedDataset:
    """The generated table plus its provenance (see :func:`dataset_meta`)."""

    frame: pd.DataFrame
    meta: dict


def class_distribution(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Counts and fractions of the teacher labels (by abstract action name)."""
    counts = Counter(frame["label"].tolist())
    total = max(len(frame), 1)
    return {
        name: {"count": counts.get(i, 0), "fraction": counts.get(i, 0) / total}
        for i, name in enumerate(ABSTRACT_ACTION_NAMES)
    }


def legal_statistics(frame: pd.DataFrame) -> dict[str, float]:
    """How often each abstract action is legal and how many choices a decision has on average."""
    legal = frame[list(LEGAL_COLUMNS)].to_numpy()
    stats = {f"{name}_legal_fraction": float(legal[:, i].mean()) for i, name in enumerate(ABSTRACT_ACTION_NAMES)}
    stats["mean_legal_actions"] = float(legal.sum(axis=1).mean())
    return stats


def generate_dataset(
    config: DatasetConfig,
    n_train_hands: int,
    n_val_hands: int,
    n_workers: int = 1,
    behavior_agent_spec: AgentSpec | None = None,
    teacher_beta: float = 0.0,
) -> GeneratedDataset:
    """Generate the train and validation tables (disjoint seed namespaces) as one frame.

    With ``behavior_agent_spec`` the acting seat is driven by that agent instead of the behaviour
    mixture (labels remain the canonical teacher's) - see :func:`collect_on_policy` for the DAgger
    wrapper that also tags the rows as aggregation rounds.
    """
    started = time.perf_counter()
    frames = []
    for split, n_hands in (("train", n_train_hands), ("val", n_val_hands)):
        spec = CollectionSpec(
            config=config,
            namespace=SPLIT_NAMESPACES[split],
            seed_parts=(config.seed,),
            split=split,
            source=SOURCE_BC,
            opponent_mix=config.opponent_mix,
            learner=behavior_agent_spec,
            teacher_beta=teacher_beta,
        )
        frames.append(collect_rows(spec, n_hands, n_workers))
    frame = pd.concat(frames, ignore_index=True)
    for name in CATEGORICAL_COLUMNS:
        frame[name] = frame[name].astype("category")
    seconds = time.perf_counter() - started
    meta = dataset_meta(frame, config, {"train": n_train_hands, "val": n_val_hands}, seconds, n_workers)
    return GeneratedDataset(frame, meta)


def collect_on_policy(
    learner_spec: AgentSpec,
    n_hands: int,
    seed_parts: Sequence[int | str],
    opponents: dict[str, float] | None = None,
    config: DatasetConfig | None = None,
    round_index: int = 1,
    split: str = "train",
    teacher_beta: float = 0.0,
    n_workers: int = 1,
) -> pd.DataFrame:
    """DAgger data collection: the learner plays, the canonical teacher labels the visited states.

    Args:
        learner_spec: registry spec of the trained agent that drives the acting seat
            (e.g. ``AgentSpec("ebm")``); loaded once per worker process.
        n_hands: hands to play.
        seed_parts: identifies the collection (e.g. ``(42, round_index)``); the seed namespace is
            ``"dagger"`` so these hands can never coincide with the train/validation hands of the
            behaviour-cloning dataset.
        opponents: opponent mixture (policy name -> weight); default ``config.opponent_mix``.
        config: game and sampling settings (default :class:`DatasetConfig`).
        round_index: DAgger iteration ``k``; rows get ``source = "dagger_round_k"``.
        split: ``"train"`` or ``"val"`` tag stored in the ``split`` column (hands are always new).
        teacher_beta: per-decision probability of following the teacher instead of the learner.
        n_workers: worker processes (results do not depend on this number).

    Returns:
        The same table layout as the behaviour-cloning dataset; concatenate it with the earlier rows
        (``pd.concat``) and refit.
    """
    if split not in SPLIT_NAMESPACES:
        raise ValueError(f"split must be one of {sorted(SPLIT_NAMESPACES)}")
    cfg = config or DatasetConfig()
    mix = dict(opponents) if opponents is not None else dict(cfg.opponent_mix)
    validate_mix(mix, POLICY_NAMES, "opponents")
    cfg = replace(cfg, opponent_mix=mix)
    spec = CollectionSpec(
        config=cfg,
        namespace=DAGGER_NAMESPACE,
        seed_parts=(*seed_parts, split),
        split=split,
        source=f"dagger_round_{round_index}",
        opponent_mix=mix,
        learner=learner_spec,
        teacher_beta=teacher_beta,
    )
    return collect_rows(spec, n_hands, n_workers)


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------
def dataset_meta(
    frame: pd.DataFrame,
    config: DatasetConfig,
    hands: dict[str, int],
    seconds: float,
    n_workers: int,
) -> dict:
    """JSON-serialisable description of a generated table (counts, distributions, schema, seeds)."""
    from poker_arena.features import FEATURE_SCHEMA_VERSION, schema_hash

    per_split = {
        split: {
            "rows": int((frame["split"] == split).sum()),
            "hands": int(frame.loc[frame["split"] == split, HAND_ID_COLUMN].nunique()),
            "requested_hands": hands.get(split, 0),
            "label_distribution": class_distribution(frame[frame["split"] == split]),
            "content_sha256": content_hash(frame[frame["split"] == split]),
        }
        for split in sorted(frame["split"].cat.categories)
    }
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "schema_hash": schema_hash(),
        "feature_names": list(FEATURE_NAMES),
        "columns": list(frame.columns),
        "config": config.to_dict(),
        "seeds": {
            "master_seed": config.seed,
            "namespaces": dict(SPLIT_NAMESPACES),
            "hand_id": "derive_seed(namespace, master_seed, 'hand', hand_index)",
        },
        "teacher": {"style": "canonical", "params": CANONICAL_PARAMS.to_dict()},
        "rows": len(frame),
        "splits": per_split,
        "label_distribution": class_distribution(frame),
        "legal_statistics": legal_statistics(frame),
        "behavior_counts": {k: int(v) for k, v in frame["behavior"].value_counts().items()},
        "opponent_counts": {k: int(v) for k, v in frame["opponent"].value_counts().items()},
        "content_sha256": content_hash(frame),
        "generation": {
            "seconds": seconds,
            "rows_per_second": len(frame) / seconds if seconds > 0 else 0.0,
            "workers": n_workers,
        },
        "libraries": library_versions(),
    }
