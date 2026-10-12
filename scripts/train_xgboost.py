"""Train the XGBoost poker agent on the teacher-labelled decision dataset (behaviour cloning + DAgger).

    .venv/Scripts/python scripts/train_xgboost.py                  # standard preset (~15 min on 10 cores)
    .venv/Scripts/python scripts/train_xgboost.py --smoke          # ~2 min end-to-end check (separate output folder)
    .venv/Scripts/python scripts/train_xgboost.py --set XGB_MAX_DEPTH=6 --set DAGGER_ROUNDS=0
    .venv/Scripts/python scripts/train_xgboost.py --tune           # one-factor-at-a-time sweep -> artifacts/models/xgb/tuning.json

Writes the best round to ``artifacts/models/xgb/`` (``model.ubj``, ``metadata.json``, ``config.json``; used by
``create_agent("xgboost")``) and static figures to ``artifacts/figures/xgb_*.png``. Needs the dataset made by
``scripts/generate_dataset.py``. All logic lives in :mod:`poker_arena.training.xgb`; this file holds the configuration.

Imitation accuracy is NOT poker skill: the summary prints agreement with the teacher and duplicate-deal BB/100
(validation seeds, never the tournament's ``eval`` seeds) separately.
"""

from __future__ import annotations

import argparse

from poker_arena import config as arena_config
from poker_arena.config import GameConfig
from poker_arena.script_config import add_config_arguments, apply_cli_overrides
from poker_arena.training.common import DaggerSettings
from poker_arena.training.xgb import (
    ValidationSettings,
    XGBParams,
    XGBTrainSettings,
    save_tuning,
    train_xgboost,
    tune_one_at_a_time,
)

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
RANDOM_SEED = 42  # master seed: row subsampling, XGBoost, DAgger / validation hands derive from it
# Game (must equal the settings the dataset was generated with; checked at start)
STARTING_STACK = 1000
SMALL_BLIND = 5
BIG_BLIND = 10
# Data
DATASET_NAME = "teacher_bc_v2"  # file stem under DATASETS_DIR (made by scripts/generate_dataset.py)
DATASETS_DIR = arena_config.DATASETS_DIR
MAX_TRAIN_ROWS = 150_000  # whole hands are kept until this many rows (None = all); same budget as the EBM for a fair comparison
CLASS_WEIGHT_POWER = 0.5  # sample weight = (n / (K * class count)) ** power; 0 = none, 1 = fully balanced (EBM convention)
# XGBoost (multi:softprob, tree_method=hist). Tuned on the validation split, never on the tournament seeds: max_bin mattered most
# (256 -> 1024 bins: masked accuracy 99.59% -> 99.75%); depth 6-10, learning rate, sub-sampling and lambda changed <= 0.05 pp
XGB_N_ESTIMATORS = 1500  # cap on boosting rounds; early stopping normally ends far earlier
XGB_MAX_DEPTH = 8  # depth of every tree (the teacher is a threshold rule on derived quantities: needs interactions)
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 1.0  # share of rows per tree
XGB_COLSAMPLE_BYTREE = 1.0  # share of features per tree
XGB_MIN_CHILD_WEIGHT = 1.0  # minimum summed (weighted) hessian in a leaf
XGB_REG_LAMBDA = 1.0  # L2 penalty on leaf values
XGB_MAX_BIN = 1024  # histogram bins per feature: the teacher thresholds continuous features (equity, pot odds) finely
XGB_EARLY_STOPPING_ROUNDS = 40  # stop when the validation loss has not improved for this many rounds
N_JOBS = 10  # CPU threads for training and SHAP (this script never uses more)
# DAgger: after behaviour cloning (round 0) the model plays, the teacher relabels the visited states, the model is refit
DAGGER_ROUNDS = 3  # 0 = plain behaviour cloning
DAGGER_HANDS_PER_ROUND = 10_000  # hands the current model plays per round (about 3 labelled decisions per hand; same as the EBM)
DAGGER_TEACHER_BETA = 0.5  # chance that the teacher (not the model) acts per decision in round 1; halves every round
DAGGER_N_WORKERS = 10  # worker processes for the on-policy hands
# Held-out validation of EVERY round (validation seeds; the tournament's "eval" seeds are never touched)
VALIDATION_OPPONENTS = ["teacher", "call_station", "maniac", "random"]
VALIDATION_PAIRS = 1000  # duplicate-deal pairs (two hands each) per opponent; 1000 = 20 independent matches of 50 pairs
VALIDATION_ON_POLICY_HANDS = 4_000  # hands the model plays to collect fresh states for the covariate-shift measurement
EVAL_N_WORKERS = 10  # worker processes for the duplicate-deal matches (results do not depend on this number)
SHAP_SAMPLE_ROWS = 5_000  # validation rows for the global mean |SHAP| importance stored in metadata.json
# --tune: sweep one setting at a time around the values above (validation imitation metrics only, no tournament seeds);
# keys are XGBParams field names (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight,
# reg_lambda, early_stopping_rounds, max_bin) or class_weight_power / max_train_rows (None = all rows)
TUNING_GRID = {
    "max_bin": [256, 512, 2048],
    "max_depth": [6, 10],
    "learning_rate": [0.05, 0.2],
    "min_child_weight": [5.0],
    "reg_lambda": [0.1, 10.0],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "class_weight_power": [0.0, 1.0],
    "max_train_rows": [None],
}
# Output
OUTPUT_DIR = arena_config.MODELS_DIR / "xgb"  # the committed model
WORK_DIR = arena_config.ARTIFACTS_DIR / "tmp" / "xgb_rounds"  # every round's candidate model (git-ignored)
FIGURES_DIR = arena_config.FIGURES_DIR
SAVE_FIGURES = True
SMOKE_OVERRIDES = {  # applied by --smoke: tiny run into a throw-away folder (never overwrites the committed model)
    "MAX_TRAIN_ROWS": 4_000,
    "XGB_N_ESTIMATORS": 60,
    "XGB_MAX_DEPTH": 6,
    "XGB_EARLY_STOPPING_ROUNDS": 10,
    "TUNING_GRID": {"max_depth": [4]},
    "DAGGER_ROUNDS": 1,
    "DAGGER_HANDS_PER_ROUND": 300,
    "VALIDATION_PAIRS": 20,
    "VALIDATION_ON_POLICY_HANDS": 150,
    "SHAP_SAMPLE_ROWS": 300,
    "OUTPUT_DIR": arena_config.ARTIFACTS_DIR / "tmp" / "xgb_smoke",
    "WORK_DIR": arena_config.ARTIFACTS_DIR / "tmp" / "xgb_smoke_rounds",
    "FIGURES_DIR": arena_config.ARTIFACTS_DIR / "tmp" / "xgb_smoke_figures",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="do not train: sweep the TUNING_GRID settings one at a time and write tuning.json next to the model",
    )
    return parser.parse_args()


def build_settings() -> XGBTrainSettings:
    return XGBTrainSettings(
        dataset_name=DATASET_NAME,
        game=GameConfig(STARTING_STACK, SMALL_BLIND, BIG_BLIND),
        seed=RANDOM_SEED,
        params=XGBParams(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            reg_lambda=XGB_REG_LAMBDA,
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            max_bin=XGB_MAX_BIN,
        ),
        max_train_rows=MAX_TRAIN_ROWS,
        class_weight_power=CLASS_WEIGHT_POWER,
        n_jobs=N_JOBS,
        dagger=DaggerSettings(
            rounds=DAGGER_ROUNDS,
            hands_per_round=DAGGER_HANDS_PER_ROUND,
            teacher_beta=DAGGER_TEACHER_BETA,
            n_workers=DAGGER_N_WORKERS,
            seed=RANDOM_SEED,
        ),
        validation=ValidationSettings(
            opponents=tuple(VALIDATION_OPPONENTS),
            pairs=VALIDATION_PAIRS,
            on_policy_hands=VALIDATION_ON_POLICY_HANDS,
            shap_sample_rows=SHAP_SAMPLE_ROWS,
            n_workers=EVAL_N_WORKERS,
        ),
        output_dir=OUTPUT_DIR,
        work_dir=WORK_DIR,
        datasets_dir=DATASETS_DIR,
    )


def format_summary(metadata: dict) -> list[str]:
    """Plain-text report: imitation (teacher agreement) and poker performance, kept strictly apart."""
    lines = ["", "IMITATION of the teacher (legal-masked agreement; NOT poker skill)"]
    lines.append(f"  {'round':>5} {'rows':>8} {'trees':>6} | {'BC-val acc':>10} {'macro-F1':>9} | {'on-policy acc':>13} {'macro-F1':>9}")
    for row in metadata["dagger"]:
        bc, live = row["bc_validation"], row["on_policy_validation"]
        marker = "*" if row["selected"] else " "
        lines.append(
            f"  {row['round']:>4}{marker} {row['n_train_rows']:>8,} {row['best_iteration'] + 1:>6} | "
            f"{bc['masked_accuracy']:>10.4f} {bc['masked_macro_f1']:>9.4f} | "
            f"{live['masked_accuracy']:>13.4f} {live['masked_macro_f1']:>9.4f}"
        )
    opponents = list(metadata["policy_validation"]["opponents"])
    lines += ["", "POKER PERFORMANCE: validation duplicate-deal BB/100 [95% CI over pairs] (* = selected round)"]
    lines.append("  " + " " * 6 + "".join(f"{name:>26}" for name in opponents) + f"{'mean':>10}")
    for row in metadata["dagger"]:
        cells = ""
        for name in opponents:
            result = row["policy_validation"][name]
            low, high = result["ci95_pairs"]
            cells += f"{result['bb_per_100']:>+10.1f} [{low:>+6.0f},{high:>+6.0f}]  "
        marker = "*" if row["selected"] else " "
        lines.append(f"  {row['round']:>4}{marker} {cells}{row['mean_bb_per_100']:>+9.1f}")
    final = metadata["imitation"]
    lines += [
        "",
        f"Selected round {metadata['selected_round']}: {metadata['n_trees_per_class']} trees per class, "
        f"{metadata['model_file_bytes'] / 1e6:.1f} MB, fit {metadata['training_time_s']:.0f} s "
        f"(whole pipeline {metadata['total_pipeline_time_s']:.0f} s)",
        f"  validation (legal-masked): accuracy {final['masked_accuracy']:.4f}, macro-F1 {final['masked_macro_f1']:.4f}, "
        f"balanced accuracy {final['masked_balanced_accuracy']:.4f}, log-loss {final['log_loss']:.4f}",
        f"  latency: {metadata['inference']['predict_proba_median_ms']:.2f} ms/decision (model), "
        f"{metadata['inference']['shap_explanation_median_ms']:.1f} ms per SHAP explanation",
        f"  shap library vs native pred_contribs: max |difference| {metadata['shap_library_check']['max_abs_difference']:.2e}",
        "",
        "Top features by mean |SHAP| (summed over the five actions):",
    ]
    for name, value in list(metadata["global_importance"].items())[:10]:
        lines.append(f"  {name:24s} {value:.3f}")
    return lines


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(globals(), SMOKE_OVERRIDES)
    settings = build_settings()
    if args.tune:
        results = tune_one_at_a_time(settings, TUNING_GRID, progress=lambda message: print(message, flush=True))
        print(f"\nwrote {save_tuning(settings, results, cfg)}")
        return
    print(f"Training XGBoost on '{DATASET_NAME}' (<= {MAX_TRAIN_ROWS} rows, {DAGGER_ROUNDS} DAgger rounds, {N_JOBS} threads)")
    outcome = train_xgboost(settings, cfg, progress=lambda message: print(message, flush=True))
    for line in format_summary(outcome.metadata):
        print(line)
    if SAVE_FIGURES:
        from poker_arena.explain.xgb import save_training_figures

        for path in save_training_figures(outcome.metadata, FIGURES_DIR):
            print(f"figure: {path}")
    print(f"\nwrote {outcome.output_dir}")


if __name__ == "__main__":
    main()
