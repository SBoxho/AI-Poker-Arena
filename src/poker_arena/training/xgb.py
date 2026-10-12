"""Training pipeline of the XGBoost agent: behaviour cloning, DAgger, selection by poker performance.

What is trained, and what the numbers mean
------------------------------------------
The classifier imitates the transparent equity teacher on the shared decision dataset (the EBM learns
from the very same rows). **Imitation accuracy is not poker skill.** The pipeline therefore reports two
clearly separated families of numbers for every candidate model:

* *imitation* - agreement with the teacher's labels, always legal-masked (what the agent plays):
  on the hand-disjoint validation split of the behaviour-cloning dataset and on **fresh on-policy
  validation states** (states the model itself visits; the gap between the two is the covariate shift);
* *performance* - duplicate-deal BB/100 with confidence intervals against the teacher and the baselines.

Why DAgger, and how the final model is chosen
---------------------------------------------
Plain behaviour cloning (round 0) never sees the states the learner's own mistakes lead to. Dataset
Aggregation (:func:`poker_arena.training.common.run_dagger`) lets the fitted model play, has the teacher
relabel the visited states, and refits on everything. Whether that *helps* is an empirical question, so
every round is evaluated on held-out validation data (never the tournament's ``eval`` seeds) and the
final model is the round with the best mean validation BB/100 over the opponents (ties: the earlier
round). Selecting among a few candidates on noisy BB/100 can only be as reliable as the confidence
intervals stored next to it - the per-round table is saved in ``metadata.json`` under ``dagger``.

Honest caveats stored with the artifact: the number of trees and the hyper-parameters were chosen on
the validation split, so the *behaviour-cloning* validation metrics are slightly optimistic; the
on-policy validation states and the duplicate-deal results are independent of that choice.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray

from poker_arena.agents.learned import ARTIFACT_SCHEMA_VERSION, METADATA_FILE, file_fingerprint
from poker_arena.agents.registry import AgentSpec
from poker_arena.agents.xgb_agent import MODEL_FILE, XGBoostAgent, class_probabilities
from poker_arena.config import REPO_ROOT, GameConfig, portable_json_default
from poker_arena.data.behaviors import policy_spec
from poker_arena.data.evaluation import pair_results, summarize
from poker_arena.data.generation import DatasetConfig, collect_on_policy
from poker_arena.data.io import library_versions
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, NUM_ABSTRACT_ACTIONS, AbstractAction
from poker_arena.explain.xgb import ShapSample, gain_importance, mean_abs_shap, shap_library_check
from poker_arena.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, schema_hash
from poker_arena.training.common import (
    DaggerSettings,
    SplitArrays,
    TrainingData,
    class_weights,
    evaluate_classifier,
    load_training_data,
    run_dagger,
)

CONFIG_FILE = "config.json"
TUNING_FILE = "tuning.json"
ARCHITECTURE = "XGBoost"
KIND = "xgboost"
POLICY_LABEL = "XGBoost"

#: Seed namespace of the duplicate-deal validation (never ``eval``, which belongs to the tournaments).
VALIDATION_NAMESPACE = "dataset_val"
POLICY_VALIDATION_TAG = "xgboost_policy_validation"
ON_POLICY_VALIDATION_TAG = "xgboost_on_policy_validation"
SHAP_SAMPLE_TAG = 1_001
CURVE_STRIDE = 10
"""Learning curves are stored every ``CURVE_STRIDE`` rounds (plus the last one) to keep metadata small."""
LATENCY_ROWS = 300
SHAP_LIBRARY_CHECK_ROWS = 100
IMITATION_SUMMARY_KEYS = ("masked_accuracy", "masked_macro_f1", "masked_balanced_accuracy", "log_loss")
OPPONENT_SUMMARY_KEYS = ("bb_per_100", "ci95_pairs", "ci95_matches", "pairs", "hands", "matches")
MS_PER_S = 1000.0


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class XGBParams:
    """Hyper-parameters of one booster (names follow the XGBoost documentation)."""

    n_estimators: int  # upper bound of boosting rounds (early stopping usually ends earlier)
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    reg_lambda: float
    early_stopping_rounds: int
    max_bin: int

    def booster_params(self, n_jobs: int, seed: int) -> dict[str, Any]:
        return {
            "objective": "multi:softprob",
            "num_class": NUM_ABSTRACT_ACTIONS,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "max_bin": self.max_bin,
            "nthread": n_jobs,
            "seed": seed,
            "verbosity": 0,
        }


@dataclass(frozen=True)
class ValidationSettings:
    """Held-out validation of every DAgger round (all of it uses validation seeds, never ``eval``)."""

    opponents: tuple[str, ...]  # policy names understood by ``poker_arena.data.behaviors.policy_spec``
    pairs: int  # duplicate pairs (two hands each) per opponent
    on_policy_hands: int  # hands played by the model to collect fresh validation states
    shap_sample_rows: int  # validation rows the global mean |SHAP| is computed on
    n_workers: int


@dataclass(frozen=True)
class XGBTrainSettings:
    dataset_name: str
    game: GameConfig
    seed: int
    params: XGBParams
    max_train_rows: int | None
    class_weight_power: float
    n_jobs: int
    dagger: DaggerSettings
    validation: ValidationSettings
    output_dir: Path
    work_dir: Path
    datasets_dir: Path | None = None


@dataclass(frozen=True)
class TrainingOutcome:
    """What :func:`train_xgboost` returns: where the model is and the final metadata."""

    output_dir: Path
    selected_round: int
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def portable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The script configuration with paths below the repository written relative to it (no local paths in artifacts)."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            try:
                return value.resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                return value.as_posix()
        return value

    return {key: convert(value) for key, value in config.items()}


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, default=portable_json_default) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def all_library_versions() -> dict[str, str]:
    versions = library_versions()
    try:
        versions["shap"] = importlib_metadata.version("shap")
    except importlib_metadata.PackageNotFoundError:
        versions["shap"] = "not installed"
    return versions


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
# Fitting one booster
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FittedBooster:
    booster: xgb.Booster
    best_iteration: int
    hit_estimator_cap: bool
    class_weights: NDArray[np.float64]
    learning_curve: dict[str, list[float]]
    seconds: float


def fit_booster(
    train: SplitArrays, val: SplitArrays, params: XGBParams, class_weight_power: float, n_jobs: int, seed: int
) -> FittedBooster:
    """Fit a ``multi:softprob`` booster with early stopping on the (hand-disjoint) validation split.

    Training rows carry the class weights ``(n / (K * count)) ** power`` of
    :func:`poker_arena.training.common.class_weights` (the convention shared with the EBM); the validation loss uses the same weights so that
    early stopping optimises the quantity that is being fitted. The booster is cut back to its best
    iteration so the saved file holds no trees past the optimum.
    """
    started = time.perf_counter()
    weights = class_weights(train.y, class_weight_power)
    names = list(FEATURE_NAMES)
    dtrain = xgb.QuantileDMatrix(
        train.X, train.y, weight=weights[train.y], feature_names=names, max_bin=params.max_bin, nthread=n_jobs
    )
    dval = xgb.QuantileDMatrix(
        val.X, val.y, weight=weights[val.y], ref=dtrain, feature_names=names, max_bin=params.max_bin, nthread=n_jobs
    )
    history: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params.booster_params(n_jobs, seed),
        dtrain,
        num_boost_round=params.n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],  # early stopping watches the last one
        early_stopping_rounds=params.early_stopping_rounds,
        evals_result=history,
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    booster = booster[: best_iteration + 1]
    booster.set_param({"nthread": n_jobs})
    kept = list(range(0, len(history["val"]["mlogloss"]), CURVE_STRIDE))
    if kept[-1] != len(history["val"]["mlogloss"]) - 1:
        kept.append(len(history["val"]["mlogloss"]) - 1)
    curve = {
        "round": [i + 1 for i in kept],
        "train_mlogloss": [history["train"]["mlogloss"][i] for i in kept],
        "val_mlogloss": [history["val"]["mlogloss"][i] for i in kept],
    }
    return FittedBooster(
        booster=booster,
        best_iteration=best_iteration,
        hit_estimator_cap=best_iteration + 1 >= params.n_estimators,
        class_weights=weights,
        learning_curve=curve,
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------------------------
# Cost of using the model
# ---------------------------------------------------------------------------------------------
def measure_latency(agent: XGBoostAgent, X: NDArray[np.float32]) -> dict[str, float]:
    """Median single-row model latency (ms) of probabilities and of a SHAP explanation, on this machine."""
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
        "shap_explanation_median_ms": float(np.median(explain_ms)),
        "rows": int(len(X)),
    }


# ---------------------------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------------------------
def select_round(history: Sequence[Mapping[str, Any]]) -> int:
    """The DAgger round with the highest mean validation BB/100; ties go to the earlier (simpler) round."""
    return int(max(history, key=lambda record: (record["mean_bb_per_100"], -record["round"]))["round"])


class XGBoostTrainer:
    """Holds the state shared by the callbacks of :func:`run_dagger` (settings, data, pair results)."""

    def __init__(
        self,
        settings: XGBTrainSettings,
        data: TrainingData,
        dataset_config: DatasetConfig,
        effective_config: Mapping[str, Any],
        progress: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.progress = progress
        self.settings = settings
        self.data = data
        self.dataset_config = dataset_config
        self.effective_config = portable_config(effective_config)
        self.pairs_by_round: dict[int, dict[str, NDArray[np.float64]]] = {}
        self.fit_info: dict[int, dict[str, Any]] = {}

    # ----------------------------------------------------------------- round 0..K: fit
    def fit(self, train_frame: pd.DataFrame, val_frame: pd.DataFrame, out_dir: Path, round_index: int) -> dict[str, Any]:
        """``FitFunction`` of :func:`run_dagger`: fit on the aggregated rows, save a loadable artifact."""
        s = self.settings
        train, val = SplitArrays.from_frame(train_frame), SplitArrays.from_frame(val_frame)
        fitted = fit_booster(train, val, s.params, s.class_weight_power, s.n_jobs, s.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / MODEL_FILE
        fitted.booster.save_model(str(model_path))
        imitation = evaluate_classifier(class_probabilities(fitted.booster, val.X), val.y, val.legal, val.street)
        self.fit_info[round_index] = {"fit_seconds": fitted.seconds, "best_iteration": fitted.best_iteration}
        write_json(
            out_dir / METADATA_FILE,
            self.round_metadata(fitted, imitation, train, val, round_index, file_fingerprint(model_path)),
        )
        write_json(out_dir / CONFIG_FILE, self.config_document())
        self.progress(
            f"round {round_index}: fitted {len(train):,} rows in {fitted.seconds:.0f} s ({fitted.best_iteration + 1} trees per class); "
            f"BC-validation masked accuracy {imitation['masked_accuracy']:.4f}"
        )
        return imitation

    def config_document(self) -> dict[str, Any]:
        s = self.settings
        return {
            "script_configuration": self.effective_config,
            "booster_params": s.params.booster_params(s.n_jobs, s.seed),
            "n_estimators_cap": s.params.n_estimators,
            "early_stopping_rounds": s.params.early_stopping_rounds,
            "game": s.game.to_dict(),
        }

    def round_metadata(
        self,
        fitted: FittedBooster,
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
            "library_versions": all_library_versions(),
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
            "best_iteration": fitted.best_iteration,
            "n_trees_per_class": fitted.best_iteration + 1,
            "hit_estimator_cap": fitted.hit_estimator_cap,
            "learning_curve": fitted.learning_curve,
            "imitation": dict(imitation),
        }

    def seeds_document(self) -> dict[str, Any]:
        s = self.settings
        return {
            "master_seed": s.seed,
            "xgboost_seed": s.seed,
            "train_subsample": f"subsample_by_hand(seed={s.seed}, max_rows={s.max_train_rows})",
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
        booster = XGBoostAgent.load(model_dir).booster  # the artifact exactly as an agent will load it

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
            class_probabilities(booster, visited.X), visited.y, visited.legal, visited.street
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
        if round_index > 0:  # same deals in every round => the per-pair difference is a paired comparison
            baseline = self.pairs_by_round[0]
            record["paired_vs_bc"] = {
                opponent: summarize(f"round {round_index}", "round 0", pairs[opponent] - baseline[opponent]).to_dict()
                for opponent in pairs
            }
        self.progress(
            f"round {round_index}: on-policy masked accuracy {record['on_policy_validation']['masked_accuracy']:.4f}; "
            f"mean validation BB/100 {record['mean_bb_per_100']:+.1f} "
            + ", ".join(f"{name} {r['bb_per_100']:+.0f}" for name, r in record["policy_validation"].items())
        )

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

        agent = XGBoostAgent.load(s.output_dir)  # also proves the saved artifact is loadable
        agent.booster.set_param({"nthread": s.n_jobs})
        rng = np.random.default_rng([s.seed, SHAP_SAMPLE_TAG])
        sample = rng.choice(len(self.data.val), size=min(v.shap_sample_rows, len(self.data.val)), replace=False)
        shap_values, shap_by_class = mean_abs_shap(ShapSample.compute(agent.booster, self.data.val.X[sample]))
        library_check = shap_library_check(agent.booster, self.data.val.X[sample[:SHAP_LIBRARY_CHECK_ROWS]])
        agent.booster.set_param({"nthread": 1})
        record = history[selected]
        metadata.update(
            {
                "created_at": now_utc(),
                "selected_round": selected,
                "selection_rule": "highest mean validation BB/100 over the opponents; ties -> earlier round",
                "total_pipeline_time_s": total_seconds,
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
                "global_importance": shap_values,
                "global_importance_gain": gain_importance(agent.booster),
                "global_importance_by_class": shap_by_class,
                "importance_definitions": {
                    "global_importance": (
                        f"mean |SHAP| (exact TreeSHAP, log-odds margin) per feature on {len(sample)} random validation rows, "
                        "summed over the five action classes"
                    ),
                    "global_importance_gain": "share of the total split gain per feature (sums to 1)",
                    "global_importance_by_class": "mean |SHAP| per class and feature on the same rows",
                },
                "shap_library_check": library_check,
                "inference": measure_latency(agent, self.data.val.X[sample[:LATENCY_ROWS]]),
                "model_file_bytes": (s.output_dir / MODEL_FILE).stat().st_size,
                "notes": (
                    "Imitation metrics measure agreement with the equity teacher, not poker skill. The tree count and "
                    "hyper-parameters were selected on the validation split, so 'imitation' is slightly optimistic; "
                    "'imitation_on_policy' and 'policy_validation' use fresh validation seeds."
                ),
            }
        )
        write_json(s.output_dir / METADATA_FILE, metadata)
        return TrainingOutcome(s.output_dir, selected, metadata)


def train_xgboost(
    settings: XGBTrainSettings,
    effective_config: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> TrainingOutcome:
    """Behaviour cloning + DAgger + held-out evaluation; writes the best round to ``settings.output_dir``.

    ``progress`` receives one human-readable line per finished step (the scripts pass ``print``).
    """
    started = time.perf_counter()
    data = load_training_data(settings.dataset_name, settings.max_train_rows, settings.seed, directory=settings.datasets_dir)
    dataset_config = dataset_config_from_meta(data.dataset_meta, settings.game)
    trainer = XGBoostTrainer(settings, data, dataset_config, effective_config or {}, progress or (lambda message: None))
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


# ---------------------------------------------------------------------------------------------
# Hyper-parameter sweep
# ---------------------------------------------------------------------------------------------
TUNABLE_PARAMS = frozenset(field.name for field in fields(XGBParams))
TUNABLE_SETTINGS = frozenset({"class_weight_power", "max_train_rows"})


def tune_one_at_a_time(
    settings: XGBTrainSettings,
    grid: Mapping[str, Sequence[Any]],
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Fit the configured model and one variant per ``grid`` entry; report validation imitation metrics.

    ``grid`` maps a name from :data:`TUNABLE_PARAMS` / :data:`TUNABLE_SETTINGS` to the values to try, one
    change at a time (a one-factor-at-a-time sweep: cheap, and enough to see which knobs matter). Only the
    behaviour-cloning train/validation split is used - no policy evaluation, no ``eval`` seeds - and the
    metrics are the unweighted legal-masked ones, so variants with different class weights are comparable.
    """
    unknown = set(grid) - TUNABLE_PARAMS - TUNABLE_SETTINGS
    if unknown:
        raise ValueError(f"cannot tune {sorted(unknown)}; choose from {sorted(TUNABLE_PARAMS | TUNABLE_SETTINGS)}")
    variants: list[tuple[str, dict[str, Any]]] = [("configured", {})]
    variants += [(f"{name}={value}", {name: value}) for name, values in grid.items() for value in values]
    datasets: dict[int | None, TrainingData] = {}
    results: list[dict[str, Any]] = []
    for label, change in variants:
        max_rows = change.get("max_train_rows", settings.max_train_rows)
        if max_rows not in datasets:
            datasets[max_rows] = load_training_data(
                settings.dataset_name, max_rows, settings.seed, directory=settings.datasets_dir
            )
        data = datasets[max_rows]
        params = replace(settings.params, **{k: v for k, v in change.items() if k in TUNABLE_PARAMS})
        power = change.get("class_weight_power", settings.class_weight_power)
        fitted = fit_booster(data.train, data.val, params, power, settings.n_jobs, settings.seed)
        metrics = evaluate_classifier(class_probabilities(fitted.booster, data.val.X), data.val.y, data.val.legal)
        results.append(
            {
                "variant": label,
                "change": change,
                "n_train_rows": len(data.train),
                "trees_per_class": fitted.best_iteration + 1,
                "fit_seconds": fitted.seconds,
                **{key: metrics[key] for key in IMITATION_SUMMARY_KEYS},
            }
        )
        if progress is not None:
            progress(
                f"{label:28s} masked accuracy {metrics['masked_accuracy']:.5f}  macro-F1 {metrics['masked_macro_f1']:.5f}  "
                f"log-loss {metrics['log_loss']:.5f}  ({fitted.best_iteration + 1} trees, {fitted.seconds:.0f} s)"
            )
    return results


def save_tuning(settings: XGBTrainSettings, results: Sequence[Mapping[str, Any]], effective_config: Mapping[str, Any]) -> Path:
    """Write the sweep next to the model so the choice of hyper-parameters is documented, not folklore."""
    path = settings.output_dir / TUNING_FILE
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "created_at": now_utc(),
            "method": "one factor at a time around the configured values; early stopping on the validation split",
            "metrics": "legal-masked imitation metrics on the hand-disjoint validation split of the behaviour-cloning data",
            "caveat": "imitation agreement with the teacher, not poker skill; the tournament eval seeds were never used",
            "dataset": settings.dataset_name,
            "library_versions": all_library_versions(),
            "script_configuration": portable_config(effective_config),
            "results": list(results),
        },
    )
    return path
