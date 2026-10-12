"""Training pipeline of the EBM agent: behaviour cloning, DAgger, selection by poker performance.

What is trained, and what the numbers mean
------------------------------------------
An ``ExplainableBoostingClassifier`` (additive shape functions + a few pairwise interaction surfaces)
imitates the transparent equity teacher on the shared decision dataset (the XGBoost agent learns from the
very same rows). **Imitation accuracy is not poker skill.** The pipeline therefore reports two clearly
separated families of numbers for every candidate model:

* *imitation* - agreement with the teacher's labels, always legal-masked (what the agent plays): on the
  hand-disjoint validation split of the behaviour-cloning dataset and on **fresh on-policy validation
  states** (states the model itself visits; the gap between the two is the covariate shift);
* *performance* - duplicate-deal BB/100 with confidence intervals against the teacher and the baselines.

Choices that matter (and why)
-----------------------------
* **Class imbalance**: rows are weighted by ``common.sample_weights`` (``(n / (K * count)) ** power``),
  so the rare all-in and pot-raise decisions are not drowned out by check/call.
* **Early stopping without leakage**: the EBM stops boosting on a validation share of the *training*
  hands (``bags``: each outer bag holds out whole hands, because the decisions of one hand are strongly
  dependent). The dataset's validation split is therefore never used to stop training.
* **Interactions**: the teacher is a threshold rule on derived quantities (equity vs pot odds, per street,
  per bet size), which an additive model of single features only approximates; a few pairwise terms close
  most of the gap while staying readable (``artifacts/models/ebm/README_training.md`` has the measured trade-off).
* **DAgger** (:func:`poker_arena.training.common.run_dagger`): the fitted EBM plays, the teacher relabels
  the states it visits, the model is refit on everything. Whether that *helps* is an empirical question, so
  every round is evaluated on held-out validation data (never the tournament's ``eval`` seeds) and the final
  model is the round with the best mean validation BB/100 over the opponents (ties: the earlier round).
  Selecting among a few candidates on noisy BB/100 can only be as reliable as the confidence intervals stored
  next to it - the complete per-round table is saved in ``metadata.json`` under ``dagger``.

Honest caveats stored with the artifact: hyper-parameters were chosen on the validation split, so the
*behaviour-cloning* validation metrics are slightly optimistic; the on-policy validation states and the
duplicate-deal results use seeds that no training step ever saw.
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from numpy.typing import NDArray

from poker_arena.agents.ebm_agent import MODEL_FILE, EBMAgent
from poker_arena.agents.learned import ARTIFACT_SCHEMA_VERSION, METADATA_FILE, file_fingerprint
from poker_arena.agents.registry import AgentSpec
from poker_arena.config import GameConfig, portable_json_default
from poker_arena.data.behaviors import policy_spec
from poker_arena.data.evaluation import pair_results, summarize
from poker_arena.data.generation import DatasetConfig, collect_on_policy
from poker_arena.data.io import library_versions
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.explain.ebm import save_training_figures, term_importance
from poker_arena.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, NUM_FEATURES, schema_hash
from poker_arena.training.common import (
    DaggerSettings,
    SplitArrays,
    TrainingData,
    class_weights,
    evaluate_classifier,
    load_training_data,
    run_dagger,
    sample_weights,
)

CONFIG_FILE = "config.json"
ARCHITECTURE = "Explainable Boosting Machine"
KIND = "ebm"
POLICY_LABEL = "EBM"

#: Seed namespace of the duplicate-deal validation (never ``eval``, which belongs to the tournaments).
VALIDATION_NAMESPACE = "dataset_val"
POLICY_VALIDATION_TAG = "ebm_policy_validation"
ON_POLICY_VALIDATION_TAG = "ebm_on_policy_validation"
IMPORTANCE_TAG = 2_001
JOBLIB_COMPRESSION = 3
"""zlib level: the saved score tables compress ~2-3x at this level for little extra load time."""
LATENCY_ROWS = 300
MS_PER_S = 1000.0
TOP_INTERACTIONS_SAVED = 12
IMITATION_SUMMARY_KEYS = ("masked_accuracy", "masked_macro_f1", "masked_balanced_accuracy", "log_loss")
OPPONENT_SUMMARY_KEYS = ("bb_per_100", "ci95_pairs", "ci95_matches", "pairs", "hands", "matches")
_PAIR_WARNING = "For multiclass we cannot currently visualize pairs"  # interpret's note; explain/ebm.py draws them


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EBMParams:
    """Hyper-parameters of one ``ExplainableBoostingClassifier`` (names follow the InterpretML documentation)."""

    max_rounds: int  # boosting rounds cap (one round = one step for every term)
    learning_rate: float
    interactions: int  # number of pairwise terms picked by InterpretML's FAST ranking (0 = main effects only)
    outer_bags: int  # bagged sub-models averaged into the final scores (smooths the shape functions)
    inner_bags: int
    max_bins: int  # bins per feature for the main effects
    max_interaction_bins: int  # bins per feature inside the pairwise tables (drives the file size)
    min_samples_leaf: int
    max_leaves: int  # 2 = a plain cut per step (the classic EBM)
    early_stopping_rounds: int
    validation_fraction: float  # share of the training HANDS held out by each outer bag for early stopping


@dataclass(frozen=True)
class ValidationSettings:
    """Held-out validation of every DAgger round (all of it uses validation seeds, never ``eval``)."""

    opponents: tuple[str, ...]  # policy names understood by ``poker_arena.data.behaviors.policy_spec``
    pairs: int  # duplicate pairs (two hands each) per opponent
    on_policy_hands: int  # hands played by the model to collect fresh validation states
    importance_sample_rows: int  # validation rows the global importance is computed on
    n_workers: int


@dataclass(frozen=True)
class EBMTrainSettings:
    dataset_name: str
    game: GameConfig
    seed: int
    params: EBMParams
    max_train_rows: int | None
    class_weight_power: float
    n_jobs: int
    dagger: DaggerSettings
    validation: ValidationSettings
    output_dir: Path
    work_dir: Path
    figures_dir: Path | None = None
    keep_round_models: bool = False
    datasets_dir: Path | None = None


@dataclass(frozen=True)
class TrainingOutcome:
    """What :func:`train_ebm` returns: where the model is and the final metadata."""

    output_dir: Path
    selected_round: int
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, default=portable_json_default) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dataset_config_from_meta(meta: Mapping[str, Any], game: GameConfig) -> DatasetConfig:
    """The :class:`DatasetConfig` a dataset was generated with, so DAgger rows come from the same game.

    Raises ``ValueError`` if the requested ``game`` differs from the dataset's: a model must be trained,
    relabelled and evaluated under one set of rules.
    """
    stored = meta["config"]
    stored_game = GameConfig.from_dict(stored["game"])
    if stored_game != game:
        raise ValueError(
            f"The dataset was generated for {stored_game} but this run is configured for {game}; "
            "regenerate the dataset or fix the game settings of the config block."
        )
    return DatasetConfig(
        game=game,
        seed=int(stored["seed"]),
        behavior_mix=dict(stored["behavior_mix"]),
        opponent_mix=dict(stored["opponent_mix"]),
        exploration_epsilon=float(stored["exploration_epsilon"]),
        match_lengths=tuple(stored["match_lengths"]),
        short_stack_probability=float(stored["short_stack_probability"]),
        min_short_stack_bb=int(stored["min_short_stack_bb"]),
    )


def summarise_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """The headline imitation numbers of an :func:`evaluate_classifier` result (for the per-round table)."""
    summary = {key: metrics[key] for key in IMITATION_SUMMARY_KEYS}
    summary["n"] = metrics["n"]
    summary["masked_accuracy_by_street"] = metrics.get("masked_accuracy_by_street", {})
    return summary


# ---------------------------------------------------------------------------------------------
# Fitting one EBM
# ---------------------------------------------------------------------------------------------
def hand_aware_bags(hand_ids: NDArray[np.int64], n_bags: int, validation_fraction: float, seed: int) -> NDArray[np.int8]:
    """InterpretML ``bags`` matrix ``(rows, n_bags)``: +1 = train, -1 = validation, whole hands at a time.

    Every outer bag holds out its own random set of hands for early stopping. Splitting by hand matters:
    the decisions of one hand (same hole cards, same board) are near-duplicates, and a row-wise split would
    let the validation loss reward memorising them.
    """
    unique = np.unique(hand_ids)
    rng = np.random.default_rng(seed)
    held_out = max(1, int(round(len(unique) * validation_fraction)))
    bags = np.ones((len(hand_ids), n_bags), dtype=np.int8)
    for bag in range(n_bags):
        bags[np.isin(hand_ids, rng.choice(unique, size=held_out, replace=False)), bag] = -1
    return bags


@dataclass(frozen=True)
class FittedEBM:
    model: ExplainableBoostingClassifier
    class_weights: NDArray[np.float64]
    seconds: float
    rounds: dict[str, Any]


def fit_ebm(train: SplitArrays, params: EBMParams, class_weight_power: float, n_jobs: int, seed: int) -> FittedEBM:
    """Fit the EBM on class-weighted rows with hand-aware early stopping (see the module docstring)."""
    started = time.perf_counter()
    model = ExplainableBoostingClassifier(
        feature_names=list(FEATURE_NAMES),
        feature_types=["continuous"] * NUM_FEATURES,  # binary flags stay continuous: one cut at 0.5
        max_bins=params.max_bins,
        max_interaction_bins=params.max_interaction_bins,
        interactions=params.interactions,
        outer_bags=params.outer_bags,
        inner_bags=params.inner_bags,
        learning_rate=params.learning_rate,
        max_rounds=params.max_rounds,
        early_stopping_rounds=params.early_stopping_rounds,
        min_samples_leaf=params.min_samples_leaf,
        max_leaves=params.max_leaves,
        n_jobs=n_jobs,
        random_state=seed,
    )
    bags = hand_aware_bags(train.hand_id, params.outer_bags, params.validation_fraction, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_PAIR_WARNING)
        model.fit(train.X, train.y, sample_weight=sample_weights(train.y, class_weight_power), bags=bags)
    return FittedEBM(
        model=model,
        class_weights=class_weights(train.y, class_weight_power),
        seconds=time.perf_counter() - started,
        rounds=boosting_rounds(model, params),
    )


def boosting_rounds(model: ExplainableBoostingClassifier, params: EBMParams) -> dict[str, Any]:
    """Boosting rounds actually used (mean over the outer bags) and whether the round cap ended training.

    InterpretML counts *steps* (one per term and round); stage 0 boosts the main effects, stage 1 the pairs.
    """
    n_main = sum(len(f) == 1 for f in model.term_features_)
    n_pairs = len(model.term_features_) - n_main
    steps = np.asarray(model.best_iteration_, dtype=np.float64)
    result: dict[str, Any] = {
        "main_effect_rounds_mean": float(steps[0].mean() / n_main),
        "main_effect_rounds_hit_cap": bool((steps[0] / n_main >= params.max_rounds).any()),
    }
    if n_pairs and len(steps) > 1:
        result["interaction_rounds_mean"] = float(steps[1].mean() / n_pairs)
        result["interaction_rounds_hit_cap"] = bool((steps[1] / n_pairs >= params.max_rounds).any())
    return result


def class_probabilities(model: ExplainableBoostingClassifier, X: NDArray[np.float32]) -> NDArray[np.float64]:
    """Probabilities of the five abstract actions for many rows (classes absent in training get 0)."""
    proba = np.zeros((len(X), NUM_ABSTRACT_ACTIONS), dtype=np.float64)
    proba[:, [int(c) for c in model.classes_]] = model.predict_proba(X)
    return proba


def measure_latency(agent: EBMAgent, X: NDArray[np.float32]) -> dict[str, float]:
    """Median single-row model latency (ms) of probabilities and of a full term decomposition, on this machine."""
    proba_ms: list[float] = []
    explain_ms: list[float] = []
    for row in X:
        started = time.perf_counter()
        proba = agent.predict_proba(row)
        proba_ms.append((time.perf_counter() - started) * MS_PER_S)
        started = time.perf_counter()
        agent.top_features(row, AbstractAction(int(np.argmax(proba))))
        explain_ms.append((time.perf_counter() - started) * MS_PER_S)
    return {
        "predict_proba_median_ms": float(np.median(proba_ms)),
        "predict_proba_p95_ms": float(np.percentile(proba_ms, 95)),
        "term_decomposition_median_ms": float(np.median(explain_ms)),
        "rows": int(len(X)),
    }


def select_round(history: list[dict[str, Any]]) -> int:
    """The round with the highest mean validation BB/100; a tie goes to the earlier (simpler) round."""
    return max(history, key=lambda record: (record["mean_bb_per_100"], -record["round"]))["round"]


# ---------------------------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------------------------
class EBMTrainer:
    """Holds the state shared by the callbacks of :func:`run_dagger` (settings, data, pair results)."""

    def __init__(
        self,
        settings: EBMTrainSettings,
        data: TrainingData,
        dataset_config: DatasetConfig,
        effective_config: Mapping[str, Any],
        log: Callable[[str], None],
    ) -> None:
        self.settings = settings
        self.data = data
        self.dataset_config = dataset_config
        self.effective_config = dict(effective_config)
        self.log = log
        self.pairs_by_round: dict[int, dict[str, NDArray[np.float64]]] = {}
        self.fit_info: dict[int, dict[str, Any]] = {}

    # ----------------------------------------------------------------- round 0..K: fit
    def fit(self, train_frame: pd.DataFrame, val_frame: pd.DataFrame, out_dir: Path, round_index: int) -> dict[str, Any]:
        """``FitFunction`` of :func:`run_dagger`: fit on the aggregated rows, save a loadable artifact."""
        s = self.settings
        train, val = SplitArrays.from_frame(train_frame), SplitArrays.from_frame(val_frame)
        fitted = fit_ebm(train, s.params, s.class_weight_power, s.n_jobs, s.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / MODEL_FILE
        joblib.dump(fitted.model, model_path, compress=JOBLIB_COMPRESSION)
        imitation = evaluate_classifier(class_probabilities(fitted.model, val.X), val.y, val.legal, val.street)
        self.fit_info[round_index] = {"fit_seconds": fitted.seconds, **fitted.rounds}
        self.log(
            f"round {round_index}: fitted on {len(train):,} rows in {fitted.seconds:.0f} s "
            f"(validation masked accuracy {imitation['masked_accuracy']:.4f})"
        )
        write_json(
            out_dir / METADATA_FILE,
            self.round_metadata(fitted, imitation, train, val, round_index, file_fingerprint(model_path)),
        )
        write_json(out_dir / CONFIG_FILE, self.config_document())
        return imitation

    def config_document(self) -> dict[str, Any]:
        s = self.settings
        return {
            "script_configuration": self.effective_config,
            "ebm_params": asdict(s.params),
            "class_weight_power": s.class_weight_power,
            "max_train_rows": s.max_train_rows,
            "n_jobs": s.n_jobs,
            "game": s.game.to_dict(),
        }

    def round_metadata(
        self,
        fitted: FittedEBM,
        imitation: Mapping[str, Any],
        train: SplitArrays,
        val: SplitArrays,
        round_index: int,
        fingerprint: str,
    ) -> dict[str, Any]:
        s = self.settings
        meta = self.data.dataset_meta
        return {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_kind": KIND,
            "architecture": ARCHITECTURE,
            "model_file": MODEL_FILE,
            "model_fingerprint": fingerprint,
            "feature_schema_hash": schema_hash(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "class_names": list(ABSTRACT_ACTION_NAMES),
            "created_at": now_utc(),
            "library_versions": library_versions(),
            "game_config": s.game.to_dict(),
            "dataset": {
                "name": s.dataset_name,
                "content_sha256": meta["content_sha256"],
                "rows": meta["rows"],
                "schema_hash": meta["schema_hash"],
                "train_split_sha256": meta["splits"]["train"]["content_sha256"],
                "val_split_sha256": meta["splits"]["val"]["content_sha256"],
            },
            "config": self.config_document(),
            "seeds": self.seeds_document(),
            "training_time_s": fitted.seconds,
            "n_train_rows": len(train),
            "n_val_rows": len(val),
            "n_bc_train_rows": len(self.data.train),
            "dagger_round": round_index,
            "class_weight_power": s.class_weight_power,
            "class_weights": {name: float(w) for name, w in zip(ABSTRACT_ACTION_NAMES, fitted.class_weights, strict=True)},
            "n_terms": len(fitted.model.term_features_),
            "n_interaction_terms": sum(len(f) > 1 for f in fitted.model.term_features_),
            "boosting": fitted.rounds,
            "imitation": dict(imitation),
        }

    def seeds_document(self) -> dict[str, Any]:
        s = self.settings
        return {
            "master_seed": s.seed,
            "ebm_random_state": s.seed,
            "train_subsample": f"subsample_by_hand(seed={s.seed}, max_rows={s.max_train_rows})",
            "early_stopping_bags": f"hand-level holdout, default_rng({s.seed})",
            "dagger_collection": f'derive_seed("dagger", {s.seed}, round, "train", ...)',
            "on_policy_validation": f'derive_seed("dagger", "{ON_POLICY_VALIDATION_TAG}", {s.seed}, "val", ...)',
            "policy_validation": f'derive_seed("{VALIDATION_NAMESPACE}", "{POLICY_VALIDATION_TAG}", {s.seed}, opponent, ...)',
            "policy_validation_namespace": VALIDATION_NAMESPACE,
            "eval_namespace_used": False,
        }

    # ------------------------------------------------------- per-round held-out evaluation
    def validate_round(self, round_index: int, record: dict[str, Any]) -> None:
        """``on_round`` of :func:`run_dagger`: covariate-shift measurement and duplicate-deal BB/100."""
        s, v = self.settings, self.settings.validation
        model_dir = Path(record["model_dir"])
        agent_spec = AgentSpec(KIND, {"model_dir": str(model_dir)}, POLICY_LABEL)
        model = EBMAgent.load(model_dir).model  # the artifact exactly as an agent will load it

        states = collect_on_policy(
            agent_spec,
            v.on_policy_hands,
            (ON_POLICY_VALIDATION_TAG, s.seed),
            config=self.dataset_config,
            round_index=round_index,
            split="val",
            n_workers=v.n_workers,
        )
        visited = SplitArrays.from_frame(states)
        record["on_policy_validation"] = evaluate_classifier(
            class_probabilities(model, visited.X), visited.y, visited.legal, visited.street
        )

        pairs: dict[str, NDArray[np.float64]] = {}
        for opponent in v.opponents:
            pairs[opponent] = pair_results(
                agent_spec,
                policy_spec(opponent),
                v.pairs,
                (POLICY_VALIDATION_TAG, s.seed, opponent),
                s.game,
                v.n_workers,
                namespace=VALIDATION_NAMESPACE,
            )
        self.pairs_by_round[round_index] = pairs
        record["policy_validation"] = {
            opponent: summarize(POLICY_LABEL, opponent, values).to_dict() for opponent, values in pairs.items()
        }
        record["mean_bb_per_100"] = float(np.mean([r["bb_per_100"] for r in record["policy_validation"].values()]))
        self.log(
            f"round {round_index}: on-policy masked accuracy {record['on_policy_validation']['masked_accuracy']:.4f}, "
            f"mean validation BB/100 {record['mean_bb_per_100']:+.1f}"
        )
        if round_index > 0:  # same deals in every round => the per-pair difference is a paired comparison
            baseline = self.pairs_by_round[0]
            record["paired_vs_bc"] = {
                opponent: summarize(f"round {round_index}", "round 0", pairs[opponent] - baseline[opponent]).to_dict()
                for opponent in pairs
            }

    # --------------------------------------------------------------------- final artifact
    def dagger_table(self, history: list[dict[str, Any]], selected: int) -> list[dict[str, Any]]:
        s = self.settings
        table = []
        previous_rows = len(self.data.train)
        for record in history:
            k = record["round"]
            entry: dict[str, Any] = {
                "round": k,
                "selected": k == selected,
                "n_train_rows": record["n_train_rows"],
                "n_dagger_rows_total": record["n_train_rows"] - len(self.data.train),
                "n_dagger_rows_added": record["n_train_rows"] - previous_rows,
                "teacher_beta_of_added_rows": None if k == 0 else s.dagger.teacher_beta * 0.5 ** (k - 1),
                **self.fit_info[k],
                "bc_validation": summarise_metrics(record["metrics"]),
                "on_policy_validation": summarise_metrics(record["on_policy_validation"]),
                "mean_bb_per_100": record["mean_bb_per_100"],
                "policy_validation": {
                    opponent: {key: result[key] for key in OPPONENT_SUMMARY_KEYS}
                    for opponent, result in record["policy_validation"].items()
                },
            }
            if "paired_vs_bc" in record:
                entry["paired_vs_bc"] = {
                    opponent: {key: result[key] for key in OPPONENT_SUMMARY_KEYS}
                    for opponent, result in record["paired_vs_bc"].items()
                }
            previous_rows = record["n_train_rows"]
            table.append(entry)
        return table

    def explanation_summary(self, model: ExplainableBoostingClassifier) -> dict[str, Any]:
        """Global importance of every term on validation states, plus the strongest interactions."""
        s, v = self.settings, self.settings.validation
        rng = np.random.default_rng([s.seed, IMPORTANCE_TAG])
        rows = rng.choice(len(self.data.val), size=min(v.importance_sample_rows, len(self.data.val)), replace=False)
        frame = term_importance(model, self.data.val.X[rows])
        native = term_importance(model)  # InterpretML's bin-weighted average over the (class-weighted) training data
        pairs = frame[frame["kind"] == "interaction"].head(TOP_INTERACTIONS_SAVED)
        return {
            "global_importance": {t: float(i) for t, i in zip(frame["term"], frame["importance"], strict=True)},
            "global_importance_by_action": {
                name: {t: float(i) for t, i in zip(frame["term"], frame[name], strict=True)} for name in ABSTRACT_ACTION_NAMES
            },
            "global_importance_training_weighted": {t: float(i) for t, i in zip(native["term"], native["importance"], strict=True)},
            "top_interactions": [
                {"term": row.term, "features": list(row.features), "importance": float(row.importance)} for row in pairs.itertuples()
            ],
            "importance_definitions": {
                "global_importance": (
                    f"mean |term score| (log-odds) averaged over the five actions, on {len(rows)} random validation states; "
                    "main effects and pairwise interactions are ranked together"
                ),
                "global_importance_by_action": "the same average for one action's logit",
                "global_importance_training_weighted": (
                    "InterpretML's term_importances(): mean |score| over the training bins, weighted by the (class-weighted) "
                    "training data - what ebm.explain_global() shows"
                ),
            },
        }

    def finalise(self, history: list[dict[str, Any]], total_seconds: float) -> TrainingOutcome:
        """Copy the best round to ``output_dir`` and complete its metadata (importance, DAgger table...)."""
        s, v = self.settings, self.settings.validation
        selected = select_round(history)
        source = Path(history[selected]["model_dir"])
        s.output_dir.mkdir(parents=True, exist_ok=True)
        for stale in (MODEL_FILE, METADATA_FILE, CONFIG_FILE):
            (s.output_dir / stale).unlink(missing_ok=True)
        for name in (MODEL_FILE, CONFIG_FILE, METADATA_FILE):
            shutil.copy2(source / name, s.output_dir / name)
        metadata = json.loads((source / METADATA_FILE).read_text(encoding="utf-8"))

        agent = EBMAgent.load(s.output_dir)  # also proves the saved artifact is loadable
        record = history[selected]
        metadata.update(
            {
                "created_at": now_utc(),
                "selected_round": selected,
                "selection_rule": "highest mean validation BB/100 over the opponents; ties -> earlier round",
                "total_pipeline_time_s": total_seconds,
                "training_time_s_all_rounds": sum(info["fit_seconds"] for info in self.fit_info.values()),
                "imitation_on_policy": record["on_policy_validation"],
                "policy_validation": {
                    "opponents": list(v.opponents),
                    "pairs_per_opponent": v.pairs,
                    "seed_namespace": VALIDATION_NAMESPACE,
                    "mean_bb_per_100": record["mean_bb_per_100"],
                    "results": record["policy_validation"],
                },
                "dagger": self.dagger_table(history, selected),
                "dagger_settings": {
                    "rounds": s.dagger.rounds,
                    "hands_per_round": s.dagger.hands_per_round,
                    "teacher_beta": s.dagger.teacher_beta,
                    "on_policy_validation_hands": v.on_policy_hands,
                },
                **self.explanation_summary(agent.model),
                "inference": measure_latency(agent, self.data.val.X[:LATENCY_ROWS]),
                "model_file_bytes": (s.output_dir / MODEL_FILE).stat().st_size,
                "notes": (
                    "Imitation metrics measure agreement with the equity teacher, not poker skill. The hyper-parameters were "
                    "chosen on the validation split, so 'imitation' is slightly optimistic; 'imitation_on_policy' and "
                    "'policy_validation' use fresh validation seeds, but the selected round was picked on those very BB/100 "
                    "numbers (winner's curse), and the mean BB/100 of different rounds differs by far less than the width of "
                    "its confidence intervals - 'dagger[*].paired_vs_bc' compares every round with round 0 on identical deals. "
                    "EBM scores are contributions to each action's log-odds; correlated "
                    "features (equity, equity_minus_pot_odds, pot_odds, preflop_percentile) share credit, so read their shape "
                    "functions together."
                ),
            }
        )
        write_json(s.output_dir / METADATA_FILE, metadata)
        if s.figures_dir is not None:
            save_training_figures(agent.model, metadata, s.figures_dir, self.data.val.X)
        if not s.keep_round_models:
            shutil.rmtree(s.work_dir, ignore_errors=True)
        return TrainingOutcome(s.output_dir, selected, metadata)


def train_ebm(
    settings: EBMTrainSettings,
    effective_config: Mapping[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> TrainingOutcome:
    """Behaviour cloning + DAgger + held-out evaluation; writes the best round to ``settings.output_dir``.

    ``log`` receives one progress line per fitted / validated round (the standard preset runs ~20 minutes).
    """
    started = time.perf_counter()
    data = load_training_data(settings.dataset_name, settings.max_train_rows, settings.seed, directory=settings.datasets_dir)
    dataset_config = dataset_config_from_meta(data.dataset_meta, settings.game)
    trainer = EBMTrainer(settings, data, dataset_config, effective_config or {}, log)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    for stale in settings.work_dir.glob("round_*"):  # earlier runs must not leak into this one
        shutil.rmtree(stale)
    history = run_dagger(
        KIND,
        trainer.fit,
        data.train.frame,
        data.val.frame,
        settings.work_dir,
        settings.dagger,
        on_round=trainer.validate_round,
        dataset_config=dataset_config,
    )
    return trainer.finalise(history, time.perf_counter() - started)
