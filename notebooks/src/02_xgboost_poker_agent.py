# %% [markdown]
# # Notebook 2 - XGBoost Poker Agent (gradient-boosted trees + SHAP)
#
# > **Reproduce:** `.venv/Scripts/python scripts/build_notebooks.py --execute 02_xgboost_poker_agent`
# > (about 3-4 minutes on 4 worker processes; the source is `notebooks/src/02_xgboost_poker_agent.py`, the adjustable
# > numbers are in the *NOTEBOOK CONFIGURATION* cell below). The agent itself was trained by
# > `.venv/Scripts/python scripts/train_xgboost.py` (about 8 minutes on 10 threads; `--tune` wrote the sweep table); this
# > notebook only *reads* those artifacts from `artifacts/models/xgb/` and runs fresh, seeded analyses and simulations on top of them.
#
# ## 1. Overview and research question
#
# **Research question.** *A gradient-boosted tree ensemble of thousands of trees can imitate a transparent poker rule-set almost
# perfectly. Can it be explained after the fact (SHAP) well enough to recognise the rule that actually produced a decision -
# and how does that compare with the EBM, a model that is its own explanation?*
#
# Sub-questions that structure the notebook:
#
# 1. **How well does it imitate?** Agreement with the teacher on states it never trained on, by street, class and teacher rule - and where it errs.
# 2. **Is imitation the same as playing well?** Duplicate-deal BB/100 with confidence intervals, kept strictly apart from imitation accuracy.
# 3. **What does SHAP say the model uses?** Gain vs SHAP vs permutation importance, per-action effects, dependence and interactions - and where those views disagree (collinear features).
# 4. **Are the explanations faithful?** Does the feature that SHAP ranks first for a decision belong to the features the teacher's firing rule really reads?
# 5. **How does it differ from the EBM?** Same data, same features, same labels, different model class.
# 6. **Did it learn the teacher's *structure*** (thresholds, rules) or only its outputs?
#
# ### Background in five lines (for readers who do not play poker)
#
# * **Heads-up no-limit Texas Hold'em**: two players, two private cards each, up to five shared cards revealed in three steps (flop, turn, river);
#   a betting round (a *street*) follows each reveal, the first one is *preflop*. The player on the *button* acts first preflop and last afterwards.
# * **Actions** are discretised to five: fold, check/call, raise about half the pot, raise about the pot, all-in. A **legal mask** says which of the five exist right now
#   (you cannot fold when you may check for free); the agent is only ever allowed to pick a legal action.
# * **Equity** is the chance that my hand beats a *random* opponent hand at showdown; **pot odds** = call / (pot + call) is the break-even equity of a call;
#   `equity - pot_odds` > 0 means "calling beats a random hand's price" (it ignores that a bettor does not hold a random hand - the teacher corrects for that).
# * **BB/100** = big blinds won per 100 hands. Poker is very noisy, so strengths are measured on **duplicate deals**: every deal is played twice with the seats
#   swapped, which cancels the luck of the cards.
# * **Imitation accuracy is not poker skill.** A model can copy a mediocre teacher perfectly, or a strong one badly. The two are reported separately.
#
# ### What are gradient-boosted trees and SHAP?
#
# **Gradient boosting** builds an ensemble stage by stage: each new decision tree (here depth 8) is fitted to the *errors* of the ensemble so far, and its small, shrunken
# output is added to a running score. With five actions XGBoost keeps **one tree per class per round**; the class scores (*margins*, log-odds-like) go through a softmax to give
# action probabilities. A tree can split on any feature at any depth, so the model is **non-additive**: the effect of `equity` may depend on the street, the bet size and the
# opponent - exactly what a threshold rule with exceptions needs, and exactly what makes the model unreadable by inspection (hundreds of thousands of split nodes).
#
# **SHAP** (Shapley additive explanations) turns one prediction into per-feature contributions that add up exactly: `margin(action) = base value + sum of the 45 feature contributions`.
# **TreeSHAP** computes the Shapley values of a tree ensemble exactly (not by sampling). The contributions answer *"how did each feature value move this state's score away from the
# average state?"*; they explain the **model**, they are not causal statements about poker, and for *correlated* features the split of credit is not unique (section 5).
#
# ### How it differs from the EBM (Notebook 1)
#
# | | EBM (Notebook 1) | XGBoost (this notebook) |
# |---|---|---|
# | Model class | **additive** shape functions + a few dozen pairwise interaction surfaces | **non-additive** ensemble of depth-8 trees (one tree per class per boosting round) |
# | Capacity | low-order by construction | high: any interaction depth up to 8 |
# | Explanation | the model *is* the explanation (glass box): contributions are its own terms | **post-hoc**: SHAP values computed from the fitted trees |
# | Faithfulness of the explanation | exact by construction | exact for the *additive attribution of the margin*, but the attribution is a choice (how to share credit between correlated features) |
# | Data, features, labels, split | identical - same dataset file, same 45-feature schema, same teacher labels, same hand-level split | |
#
# ### Evidence labels used throughout
#
# * **[saved]** - read from the artifacts of the training run (`artifacts/models/xgb/`, the dataset metadata), never recomputed.
# * **[recomputed]** - computed here from the saved model on the validation split of the shared dataset (and checked against the saved numbers where they exist).
# * **[fresh]** - simulated in this notebook with a visible seed (`EVAL_SEED`, seed namespace `eval`): deals that no training or validation step ever saw.
# * **[live]** - a small illustration fitted or computed here (a toy model, a counterfactual scan); it is *not* the committed model and not evidence about strength.
#
# Every number printed below is computed or read; sentences that quote numbers are generated from them. A plain separation of *empirical findings*, *assumptions*,
# *speculation* and *limitations* closes the notebook (section 9).

# %%
# ==========================================
# NOTEBOOK CONFIGURATION
# ==========================================
EVAL_SEED = 777_003  # master seed of every FRESH simulation here (namespace "eval"); training used 42, official tournament 42
EVAL_PAIRS = 400  # duplicate pairs (2 hands each) per opponent against the baselines (section 4.4)
EVAL_PAIRS_TEACHER = 500  # more pairs against the teacher and in the XGBoost-vs-EBM head-to-head: the noisiest, most informative matchups
ANALYSIS_SEED = 7  # seeds every sample of validation states, bootstrap and permutation in this notebook
BOOTSTRAP_REPS = 1000  # hand-level (cluster) bootstrap resamples for metric confidence intervals
LIVE_FIT_ROWS = 20_000  # training rows of the toy fit in section 3.5 (whole hands; the committed model used 245,525)
LIVE_FIT_DEPTHS = (1, 2, 8)  # tree depths compared in the toy fit: 1 = additive, 2 = pairwise interactions, 8 = committed
LIVE_FIT_MAX_ROUNDS = 600  # boosting-round cap of the toy fits (early stopping usually ends sooner)
SHAP_ROWS = 3000  # validation states explained with TreeSHAP for the summary / dependence plots (section 5)
INTERACTION_ROWS = 500  # states for SHAP interaction values (the cost grows with rows x 46 x 46)
INTERVENTIONAL_ROWS, INTERVENTIONAL_BACKGROUND = 200, 25  # states / background rows of the interventional-SHAP check (slow)
PERMUTATION_ROWS, PERMUTATION_REPEATS = 5000, 5  # permutation importance (section 5.2)
FIDELITY_STATES = 1500  # validation states of the explanation-fidelity study (section 6.3); 2 SHAP calls per state at ~13 ms
COUNTERFACTUAL_TEMPLATES = 300  # real states used as templates in the threshold-recovery scans (section 8.2)
N_WORKERS = 4  # worker processes for the simulations (shared machine: keep <= 4)
N_THREADS = 4  # XGBoost / SHAP threads

# %%
import io
import json
import time
import warnings
from collections import Counter
from contextlib import redirect_stderr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from IPython.display import Markdown, display
from matplotlib.colors import PowerNorm, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from poker_arena.agents.ebm_agent import EBMAgent
from poker_arena.agents.learned import mask_and_normalise
from poker_arena.agents.registry import AgentSpec
from poker_arena.agents.teacher import CANONICAL_PARAMS, FOLD_TO_BET_NEUTRAL, MAX_VALUE_THRESHOLD, MIN_VALUE_THRESHOLD, TeacherPolicy
from poker_arena.agents.xgb_agent import XGBoostAgent, class_probabilities, shap_contributions
from poker_arena.config import ARTIFACTS_DIR, MODELS_DIR, GameConfig
from poker_arena.data.behaviors import policy_spec
from poker_arena.data.evaluation import pair_results, summarize
from poker_arena.data.io import content_hash, feature_matrix, label_vector, legal_mask_matrix, load_dataset, load_meta
from poker_arena.engine.actions import ABSTRACT_ACTION_NAMES, AbstractAction
from poker_arena.explain import notebook_plots as nbp
from poker_arena.explain.ebm import term_importance
from poker_arena.explain.notebook_cases import select_example_cases
from poker_arena.explain.xgb import (
    GROUP_COLORS,
    ShapSample,
    gain_importance,
    mean_abs_shap,
    shap_library_values,
)
from poker_arena.features import BINARY_FEATURES, FEATURE_DESCRIPTIONS, FEATURE_GROUP_OF, FEATURE_GROUPS, FEATURE_INDEX, FEATURE_NAMES, NUM_FEATURES, schema_hash
from poker_arena.training.common import STREET_NAMES, class_weights, evaluate_classifier, masked_argmax, subsample_by_hand, SplitArrays

warnings.filterwarnings("ignore", category=FutureWarning)
NOTEBOOK_STARTED = time.perf_counter()
nbp.apply_style()
ACTIONS = list(ABSTRACT_ACTION_NAMES)  # fold, check_call, raise_small, raise_pot, all_in
XGB_COLOR, EBM_COLOR = nbp.SERIES[1], nbp.SERIES[2]  # one colour per model throughout

# ---- the saved artifacts (everything labelled [saved] comes from these files) ----
XGB_DIR, EBM_DIR = MODELS_DIR / "xgb", MODELS_DIR / "ebm"


def read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


meta, run_cfg, tuning = read_json(XGB_DIR / "metadata.json"), read_json(XGB_DIR / "config.json"), read_json(XGB_DIR / "tuning.json")
ebm_meta = read_json(EBM_DIR / "metadata.json")
DATASET = meta["dataset"]["name"]
GAME = GameConfig.from_dict(meta["game_config"])
AGENT = XGBoostAgent.load()  # validates schema version, feature schema hash, fingerprint
AGENT.booster.set_param({"nthread": N_THREADS})
EBM = EBMAgent.load()
TEACHER = TeacherPolicy()
PARAMS = CANONICAL_PARAMS

# ---- the shared dataset: the validation split as arrays ----
frame = load_dataset(DATASET)
ds_meta = load_meta(DATASET)
VAL = frame[frame["split"] == "val"].reset_index(drop=True)
X_VAL, Y_VAL, L_VAL = feature_matrix(VAL), label_vector(VAL), legal_mask_matrix(VAL)
STREET_VAL = VAL["street"].to_numpy(dtype=np.int64)
HANDS_VAL = VAL["hand_id"].to_numpy(dtype=np.int64)
RULE_VAL = VAL["teacher_rule"].astype(str).to_numpy()
MARGIN_VAL = VAL["teacher_margin"].to_numpy(dtype=float)
N_VAL = len(VAL)


# ---- small display helpers (formatting only) ----
def md_table(table: pd.DataFrame, floatfmt: str = "{:.3f}") -> Markdown:
    """A pandas frame as a GitHub-flavoured markdown table (deterministic, renders everywhere)."""
    escape = lambda text: str(text).replace("|", "\\|")  # noqa: E731 - a pipe inside a cell would split the column
    header = "| " + " | ".join(escape(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    cell = lambda v: escape(floatfmt.format(v) if isinstance(v, (float, np.floating)) else v)  # noqa: E731
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return Markdown("\n".join([header, rule, *rows]))


def ci_text(estimate: float, low: float, high: float) -> str:
    """``+12 [-30, +55]`` (rounded to whole BB/100, so no ``-0``)."""
    return f"{round(estimate):+d} [{round(low):+d}, {round(high):+d}]"


def verdict(low: float, high: float) -> str:
    """Words for a confidence interval of a BB/100 value or difference from zero (never more than the CI supports)."""
    return "wins" if low > 0 else "loses" if high < 0 else "no significant difference"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f} pp"


def legend_below(ax, names) -> None:
    """Legend of a ``forest_plot`` below the axis (same colours as the plot, which uses the palette in order)."""
    handles = [Line2D([0], [0], color=nbp.SERIES[k % len(nbp.SERIES)], marker="o", markersize=7, linewidth=1.8, label=name) for k, name in enumerate(names)]
    handles.append(Line2D([0], [0], color=nbp.MUTED, marker="o", markersize=7, markerfacecolor=nbp.SURFACE, linewidth=1.8, label="hollow: 95% CI contains 0"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=min(len(handles), 4), fontsize=8.5)


def retint_axes(fig) -> None:
    """The ``shap`` plots use dark-grey labels; recolour every text of the figure for the dark felt surface."""
    for axis in fig.axes:
        axis.tick_params(axis="both", labelcolor=nbp.MUTED)
        axis.xaxis.label.set_color(nbp.MUTED)
        axis.yaxis.label.set_color(nbp.MUTED)
        axis.title.set_color(nbp.TEXT)
        for text in axis.texts:
            if to_hex(text.get_color()).lower() in ("#333333", "#999999", "#666666", "#000000"):
                text.set_color(nbp.TEXT)


display(Markdown(
    f"Loaded the saved XGBoost agent (`model_fingerprint` {meta['model_fingerprint']}, trained {meta['created_at']}, "
    f"{meta['n_trees_per_class']} trees per class, {meta['model_file_bytes'] / 1e6:.1f} MB) and the EBM it is compared with "
    f"(`{ebm_meta['model_fingerprint']}`, {ebm_meta['n_terms']} terms). Validation split of `{DATASET}`: {N_VAL:,} decisions "
    f"from {VAL['hand_id'].nunique():,} hands. Kernel libraries: xgboost {xgb.__version__}, shap {shap.__version__} "
    f"(model trained with xgboost {meta['library_versions']['xgboost']}, shap {meta['library_versions']['shap']})."
))

# %% [markdown]
# ### Reproducibility box
#
# Everything needed to regenerate the agent, the seeds that keep every random stream apart, and the cost of doing so.

# %%
cfg = run_cfg["script_configuration"]
lib = meta["library_versions"]
seeds = meta["seeds"]
display(Markdown(f"""
| item | value |
|---|---|
| **Training command** | `.venv/Scripts/python scripts/train_xgboost.py` (configuration block at the top of the script; `--smoke` for a 2-minute check, `--tune` for the sweep behind `tuning.json`) |
| **Dataset command** | `.venv/Scripts/python scripts/generate_dataset.py` -> `{DATASET}` ({meta['dataset']['rows']:,} labelled decisions) |
| **Seeds** | master seed `{seeds['master_seed']}` (XGBoost `seed` `{seeds['xgboost_seed']}`, behaviour-cloning row subsample `{seeds['train_subsample']}`); DAgger hands `{seeds['dagger_collection']}`; validation duplicate deals in namespace `{seeds['policy_validation_namespace']}`; the training run never used the `eval` namespace (`eval_namespace_used: {seeds['eval_namespace_used']}`) |
| **This notebook** | `EVAL_SEED = {EVAL_SEED}` in namespace `eval` for every fresh simulation; `ANALYSIS_SEED = {ANALYSIS_SEED}` for samples, bootstraps and permutations; {N_WORKERS} worker processes (results do not depend on the worker count) |
| **Duplicate protocol (fresh games)** | the **two-table** duplicate protocol, the default of `poker_arena.data.evaluation.pair_results`: both plays of a deal are made at independent tables (own agent copies, own opponent trackers), seats complemented (section 4.4). The BB/100 stored in the models' `metadata.json` come from the *legacy single-table* protocol and are informational only (`evaluation_protocol_note`) |
| **Model** | `model.ubj` ({meta['model_file_bytes'] / 1e6:.1f} MB), `multi:softprob`, `tree_method=hist`, {meta['n_trees_per_class']} rounds x 5 classes = {5 * meta['n_trees_per_class']:,} trees; fingerprint `{meta['model_fingerprint']}`; selected DAgger round {meta['selected_round']} |
| **Training rows** | {meta['n_train_rows']:,} = {meta['n_bc_train_rows']:,} behaviour-cloning rows + {meta['n_train_rows'] - meta['n_bc_train_rows']:,} DAgger rows; early stopping on the {meta['n_val_rows']:,}-row validation split |
| **Cost** | whole pipeline {meta['total_pipeline_time_s'] / 60:.1f} min (final fit {meta['training_time_s']:.0f} s) on {cfg['N_JOBS']} threads |
| **Libraries at training time** | python {lib['python']}, numpy {lib['numpy']}, pandas {lib['pandas']}, pokerkit {lib['pokerkit']}, scikit-learn {lib['scikit-learn']}, xgboost {lib['xgboost']}, shap {lib['shap']} |
"""))

# %% [markdown]
# ## 2. Dataset and feature preparation
#
# ### 2.1 The shared dataset: same rows, same features, same labels as the EBM
#
# XGBoost and the EBM are trained on the **same dataset file** (`artifacts/datasets/teacher_bc_v2.parquet`) with the **same 45-feature schema**
# and the same hand-level train/validation split; the metadata of both models records the content hash of that file. Below we check that the two
# hashes agree with each other *and* with a hash recomputed from the file on disk.

# %%
recomputed_hash = content_hash(frame)
hashes = {
    "dataset metadata (`teacher_bc_v2.meta.json`)": ds_meta["content_sha256"],
    "XGBoost `metadata.json`": meta["dataset"]["content_sha256"],
    "EBM `metadata.json`": ebm_meta["dataset"]["content_sha256"],
    "recomputed from the parquet file now": recomputed_hash,
}
assert len(set(hashes.values())) == 1, hashes
assert meta["feature_schema_hash"] == ebm_meta["feature_schema_hash"] == schema_hash() == ds_meta["schema_hash"]
train_hands, val_hands = set(frame.loc[frame["split"] == "train", "hand_id"]), set(VAL["hand_id"])
assert not train_hands & val_hands  # the split is by hand: no hand contributes decisions to both sides
display(Markdown(
    f"* dataset content SHA-256 **`{recomputed_hash[:16]}...`** - identical in the dataset metadata, in the XGBoost metadata, in the EBM metadata and recomputed now "
    f"(feature schema hash `{schema_hash()}` also identical in all four places);\n"
    f"* **{len(frame):,} labelled decisions**: {ds_meta['splits']['train']['rows']:,} train rows from {ds_meta['splits']['train']['hands']:,} hands and "
    f"{ds_meta['splits']['val']['rows']:,} validation rows from {ds_meta['splits']['val']['hands']:,} hands; **{len(train_hands & val_hands)}** hands appear on both sides (the split is by hand id, never by decision), "
    f"generated in {ds_meta['generation']['seconds']:.0f} s with master seed {ds_meta['seeds']['master_seed']} (namespaces `{ds_meta['seeds']['namespaces']['train']}` / `{ds_meta['seeds']['namespaces']['val']}`);\n"
    f"* **label source = the teacher** (`style: {ds_meta['teacher']['style']}`): every state, however it came about, is labelled by the same transparent rule function. "
    f"The states are produced by a *mixture* of behaviour policies (teacher {pct(ds_meta['config']['behavior_mix']['teacher'], 0)}, teacher with {pct(ds_meta['config']['exploration_epsilon'], 0)} random exploration "
    f"{pct(ds_meta['config']['behavior_mix']['teacher_eps'], 0)}, uniformly random {pct(ds_meta['config']['behavior_mix']['random'], 0)}, four teacher styles 10% each) against a mixture of opponents "
    f"(random, call station, maniac, teacher and four styles), so the model also sees the strange states a bad player creates.\n\n"
    f"**Imitation caveat.** The teacher is a heuristic (equity against a random hand, corrected for the opponent's betting range), not a solver. "
    f"Whatever accuracy is reported below is *agreement with that rule set*; section 4.4 measures play separately."
))

# %% [markdown]
# ### 2.2 Class imbalance, legal masks and class weighting
#
# The five actions are very unequally frequent, and the set of legal actions changes from state to state. Both facts shape training and evaluation:
#
# * **Class weights.** Sample weight = `(n / (K * class count)) ** 0.5` (`class_weight_power` 0.5: a compromise between ignoring the imbalance, power 0, and fully
#   balancing it, power 1 - the tuning sweep below tries both). Early stopping uses the *same* weights in the validation loss, so it optimises what is being fitted.
# * **Legal masking.** The raw classifier outputs five probabilities; the agent plays the arg-max over *legal* actions only, and every accuracy below is the *masked*
#   one (what the agent would really do). The raw probability that the model puts on illegal actions is reported as `illegal_mass`.

# %%
split_rows = {name: ds_meta["splits"][name] for name in ("train", "val")}
distribution = pd.DataFrame(
    {
        "action": [f"`{a}`" for a in ACTIONS],
        "train rows": [split_rows["train"]["label_distribution"][a]["count"] for a in ACTIONS],
        "train share": [pct(split_rows["train"]["label_distribution"][a]["fraction"]) for a in ACTIONS],
        "validation rows": [split_rows["val"]["label_distribution"][a]["count"] for a in ACTIONS],
        "validation share": [pct(split_rows["val"]["label_distribution"][a]["fraction"]) for a in ACTIONS],
        "legal in (share of states)": [pct(ds_meta["legal_statistics"][f"{a}_legal_fraction"]) for a in ACTIONS],
        "class weight (saved)": [meta["class_weights"][a] for a in ACTIONS],
    }
)
shares = [split_rows["train"]["label_distribution"][a]["fraction"] for a in ACTIONS]
labels_legal = bool(L_VAL[np.arange(N_VAL), Y_VAL].all())
display(md_table(distribution, "{:.2f}"))
display(Markdown(
    f"**[saved]** The most frequent action (`{ACTIONS[int(np.argmax(shares))]}`, {pct(max(shares), 0)}) is {max(shares) / min(shares):.0f}x more frequent than the rarest "
    f"(`{ACTIONS[int(np.argmin(shares))]}`, {pct(min(shares))}); the class weights range from {min(meta['class_weights'].values()):.2f} to {max(meta['class_weights'].values()):.2f}. "
    f"On average {ds_meta['legal_statistics']['mean_legal_actions']:.2f} of the 5 actions are legal; folding is legal in only {pct(ds_meta['legal_statistics']['fold_legal_fraction'], 0)} of the states "
    f"(a free check forbids it). **[recomputed]** every validation label is a legal action: {labels_legal}."
))

# %% [markdown]
# ### 2.3 The 45 features (short recap - Notebook 1 has the long version)
#
# Every feature is a plain number computed from the *observation only* (no hidden cards, no seeds); the same `build_features` function is used for training and
# for play, so the two cannot drift apart. The groups, and the six features that matter most below:

# %%
group_table = pd.DataFrame(
    {
        "group": [f"**{g.replace('_', ' ')}**" for g in FEATURE_GROUPS],
        "n": [len(v) for v in FEATURE_GROUPS.values()],
        "features": [", ".join(f"`{f}`" for f in v) for v in FEATURE_GROUPS.values()],
    }
)
display(md_table(group_table))
KEY_FEATURES = ["equity", "equity_minus_pot_odds", "pot_odds", "preflop_percentile", "spr", "opp_aggression", "opp_fold_to_bet"]
display(md_table(pd.DataFrame({"feature": [f"`{f}`" for f in KEY_FEATURES], "group": [FEATURE_GROUP_OF[f] for f in KEY_FEATURES], "meaning": [FEATURE_DESCRIPTIONS[f] for f in KEY_FEATURES]})))

# %% [markdown]
# ### 2.4 What the model has to learn: the teacher's rules
#
# The teacher is a pure function of the feature vector and the legal mask (`TeacherPolicy.decide_features`), so *in principle* a model that sees the same features can
# reproduce it exactly - there is no hidden information and no label noise, only a function with sharp thresholds. Every dataset row also stores **which rule fired**
# (`teacher_rule`) and the **margin** - how far (in equity or percentile points) the deciding quantity lies from the threshold that would have changed the decision.
# Both are ground truth that we use for error analysis (4.3), for the fidelity study (6.2) and for the structure checks (8). Rule frequencies on the validation split:

# %%
rule_counts = pd.Series(RULE_VAL).value_counts()
dominant_action = {r: ACTIONS[int(np.bincount(Y_VAL[RULE_VAL == r], minlength=5).argmax())] for r in rule_counts.index}
rules_table = pd.DataFrame(
    {
        "rule": [f"`{r}`" for r in rule_counts.index],
        "street": ["preflop" if r.startswith("pre_") else "post-flop" for r in rule_counts.index],
        "decisions": rule_counts.to_numpy(),
        "share": [pct(c / N_VAL) for c in rule_counts],
        "action": [f"`{dominant_action[r]}`" for r in rule_counts.index],
        "median margin": [float(np.median(MARGIN_VAL[RULE_VAL == r])) for r in rule_counts.index],
    }
)
display(md_table(rules_table, "{:.3f}"))
display(Markdown(
    "In words: **preflop** the teacher ranks the 169 starting-hand classes by `preflop_percentile` and compares it with a ladder of thresholds that depends on position and on how many raises "
    "were made (open / limp / fold, isolate, 3-bet, 4-bet, shove); against a raise it prices a call with equity minus a *range penalty*. **Post-flop** it compares `equity` with thresholds "
    f"(value-bet line {PARAMS.value_equity:g} - moved by the opponent's fold-to-bet rate, pot bet {PARAMS.pot_bet_equity:g}, monster {PARAMS.monster_equity:g}, ...), semi-bluffs strong draws, and when facing a bet calls "
    f"if the equity - corrected for the bet size, the number of raises, the opponent's aggression and draws - beats the pot odds. All thresholds are the {len(PARAMS.to_dict())} numbers of `TeacherParams`."
))

# %% [markdown]
# ## 3. Training configuration
#
# ### 3.1 Hyper-parameters of the committed model `[saved]`
#
# XGBoost's `multi:softprob` objective with the fast histogram algorithm (`tree_method=hist`): every feature is first cut into at most `max_bin` quantile bins, and trees
# only consider those cut points as thresholds. The values below are read from `config.json`.

# %%
booster = run_cfg["booster_params"]
hyper = pd.DataFrame(
    [
        ("objective", booster["objective"], "one softmax over the 5 actions; one tree per class in every boosting round"),
        ("max_depth", booster["max_depth"], "depth of every tree: how many features one tree may combine (depth 1 would be additive)"),
        ("eta (learning rate)", booster["eta"], "shrinkage of every tree's contribution"),
        ("n_estimators cap / early stopping", f"{run_cfg['n_estimators_cap']} / {run_cfg['early_stopping_rounds']}", "stop when the (class-weighted) validation log-loss has not improved for that many rounds, keep the best round"),
        ("max_bin", booster["max_bin"], "histogram bins per feature: the teacher thresholds continuous features (equity, pot odds) finely"),
        ("subsample / colsample_bytree", f"{booster['subsample']} / {booster['colsample_bytree']}", "share of rows / features per tree (1 = no randomisation: the fit is deterministic)"),
        ("min_child_weight", booster["min_child_weight"], "minimum summed hessian in a leaf"),
        ("reg_lambda", booster["reg_lambda"], "L2 penalty on leaf values"),
        ("class_weight_power", meta["class_weight_power"], "sample weight (n / (K * count)) ** power"),
        ("seed", booster["seed"], "only matters with sub-sampling; recorded for completeness"),
    ],
    columns=["parameter", "value", "role"],
)
display(md_table(hyper))
display(Markdown(
    f"Early stopping ended the committed fit after {meta['best_iteration'] + 1} boosting rounds (cap {run_cfg['n_estimators_cap']}, `hit_estimator_cap: {meta['hit_estimator_cap']}`) - "
    f"{5 * meta['n_trees_per_class']:,} trees, {meta['model_file_bytes'] / 1e6:.1f} MB on disk."
))

# %% [markdown]
# ### 3.2 What mattered? A one-factor-at-a-time sweep `[saved]`
#
# `train_xgboost.py --tune` fits the configured model and then one variant per setting, changing **one setting at a time**, all on the behaviour-cloning rows and scored on the
# hand-disjoint validation split with the unweighted legal-masked metrics (so variants with different class weights are comparable). One fit per row: there is no repeat, so
# differences of a few hundredths of a percentage point are within what one cannot resolve. (The sweep never touches the tournament `eval` seeds.)

# %%
sweep = pd.DataFrame(tuning["results"])
base = sweep[sweep["variant"] == "configured"].iloc[0]
sweep["d_acc_pp"] = 100 * (sweep["masked_accuracy"] - base["masked_accuracy"])
sweep_by_variant = sweep.set_index("variant")
variants = sweep[sweep["variant"] != "configured"].copy()
variants["label"] = [v.replace("max_train_rows=None", "all training rows (no 150k cap)") for v in variants["variant"]]
order = variants.sort_values("d_acc_pp").reset_index(drop=True)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
colors = [nbp.ACCENT if d > 0.02 else nbp.SERIES[2] if d < -0.02 else nbp.MUTED for d in order["d_acc_pp"]]
ax.barh(order["label"], order["d_acc_pp"], color=colors, height=nbp.BAR_THICKNESS)
ax.axvline(0, color=nbp.TEXT, linewidth=1.0)
for y_pos, (d, trees) in enumerate(zip(order["d_acc_pp"], order["trees_per_class"])):
    ax.annotate(f"{d:+.3f} pp  ({trees} trees)", (max(d, 0), y_pos), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=nbp.MUTED)
ax.set_xlim(order["d_acc_pp"].min() * 1.25, order["d_acc_pp"].max() + 0.14)
ax.grid(axis="y", visible=False)
ax.set_xlabel("change of validation masked accuracy vs the configured model (percentage points)")
nbp.title(ax, "What moved imitation accuracy?", f"configured: {pct(base['masked_accuracy'], 3)} ({int(base['trees_per_class'])} trees); green > +0.02 pp, orange < -0.02 pp")
ax2.hlines(range(len(order)), base["log_loss"], order["log_loss"], color=colors, linewidth=2.2)
ax2.scatter(order["log_loss"], range(len(order)), color=colors, s=46, zorder=3)
ax2.axvline(base["log_loss"], color=nbp.TEXT, linewidth=1.0, linestyle="--")
ax2.set_xscale("log")
ax2.set_ylim(-0.6, len(order) - 0.4)
ax2.set_yticks([])
ax2.grid(axis="y", visible=False)
ax2.set_xlabel("validation log-loss (log scale; dashed = configured)")
nbp.title(ax2, "... and the probabilities", "log-loss is more sensitive than accuracy to the sharpness of the fit")
fig.tight_layout()
nbp.savefig("nb02_tuning", fig)
plt.show()

shown = variants.sort_values("variant")[["variant", "n_train_rows", "trees_per_class", "fit_seconds", "masked_accuracy", "masked_macro_f1", "log_loss", "d_acc_pp"]]
shown = shown.assign(variant=lambda f: "`" + f["variant"] + "`", masked_accuracy=lambda f: f["masked_accuracy"].map(lambda v: pct(v, 3)), masked_macro_f1=lambda f: f["masked_macro_f1"].map(lambda v: pct(v, 3)))
shown.columns = ["variant", "train rows", "trees / class", "fit s", "masked accuracy", "macro-F1", "log-loss", "delta acc. (pp)"]
display(md_table(shown, "{:.4f}"))
bins = sweep_by_variant
rest = variants[~variants["variant"].str.startswith(("max_bin", "max_train_rows"))]
more_rows = variants[variants["variant"].str.startswith("max_train_rows")].iloc[0]
display(Markdown(
    f"**[saved] What the sweep shows.** The histogram resolution is the knob that matters among the model settings: 256 bins {bins.loc['max_bin=256', 'd_acc_pp']:+.2f} pp, 512 bins {bins.loc['max_bin=512', 'd_acc_pp']:+.2f} pp, "
    f"2048 bins {bins.loc['max_bin=2048', 'd_acc_pp']:+.2f} pp (log-loss {bins.loc['max_bin=256', 'log_loss'] / base['log_loss']:.1f}x / {bins.loc['max_bin=512', 'log_loss'] / base['log_loss']:.1f}x / {bins.loc['max_bin=2048', 'log_loss'] / base['log_loss']:.2f}x of the configured one) - "
    f"the teacher's thresholds on `equity` and `pot_odds` fall *between* coarse bin edges, so a coarse histogram simply cannot place the split where the rule is. "
    f"Depth (6/10), learning rate, sub-sampling, `min_child_weight` and `reg_lambda` all stay within {rest['d_acc_pp'].abs().max():.2f} pp of the configured model. "
    f"The class-weight power of 0.5 beats both 0 ({bins.loc['class_weight_power=0.0', 'd_acc_pp']:+.2f} pp, log-loss {bins.loc['class_weight_power=0.0', 'log_loss']:.4f}) and 1 "
    f"({bins.loc['class_weight_power=1.0', 'd_acc_pp']:+.2f} pp, {bins.loc['class_weight_power=1.0', 'log_loss']:.4f}). The single largest effect is **data**: using all "
    f"{int(more_rows['n_train_rows']):,} behaviour-cloning rows instead of the 150,000-row budget (kept equal to the EBM's for a fair comparison) gives {more_rows['d_acc_pp']:+.3f} pp and {more_rows['log_loss']:.4f} log-loss, "
    f"with {int(more_rows['trees_per_class'])} instead of {int(base['trees_per_class'])} trees. "
    f"Two caveats: the sweep is *one-factor-at-a-time*, so interactions between settings are not explored; and `tuning.json` records a configuration block that differs from the final `config.json` in "
    f"two settings that do not touch this sweep (`DAGGER_HANDS_PER_ROUND` {tuning['script_configuration']['DAGGER_HANDS_PER_ROUND']:,} vs {cfg['DAGGER_HANDS_PER_ROUND']:,}, `SHAP_SAMPLE_ROWS` {tuning['script_configuration']['SHAP_SAMPLE_ROWS']:,} vs {cfg['SHAP_SAMPLE_ROWS']:,}) - "
    f"the sweep used behaviour-cloning rows only, so its numbers are unaffected, but it was evidently run with an earlier version of the script."
))

# %% [markdown]
# ### 3.3 Early stopping and the learning curve of the committed model `[saved]`
#
# The curve is the (class-weighted) multi-class log-loss after each boosting round, sampled every 10 rounds by the training pipeline. Training loss keeps falling; the
# validation loss falls with it, flattens, and the 40-round patience rule keeps the best round.

# %%
curve = pd.DataFrame(meta["learning_curve"])
fig, ax = plt.subplots(figsize=(9.6, 4.2))
ax.plot(curve["round"], curve["train_mlogloss"], color=nbp.SERIES[2], label="training rows")
ax.plot(curve["round"], curve["val_mlogloss"], color=nbp.ACCENT, label="validation rows (hand-disjoint)")
ax.axvline(meta["best_iteration"] + 1, color=nbp.MUTED, linestyle="--", linewidth=1.2)
ax.annotate(f"kept: {meta['best_iteration'] + 1} rounds", (meta["best_iteration"] + 1, curve["val_mlogloss"].max() * 0.5), xytext=(6, 0), textcoords="offset points", color=nbp.MUTED, fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("boosting round (one tree per class in each)")
ax.set_ylabel("class-weighted log-loss (log scale)")
ax.legend()
nbp.title(ax, "Learning curve of the committed model", f"last stored round {int(curve['round'].iloc[-1])}; early stopping patience {run_cfg['early_stopping_rounds']}")
fig.tight_layout()
nbp.savefig("nb02_learning_curve", fig)
plt.show()
gap = curve["val_mlogloss"].iloc[-1] / curve["train_mlogloss"].iloc[-1]
display(Markdown(
    f"**[saved]** At the last stored round the validation loss ({curve['val_mlogloss'].iloc[-1]:.4f}) is {gap:.1f}x the training loss ({curve['train_mlogloss'].iloc[-1]:.4f}): the trees keep sharpening their fit of the training rows after the "
    f"validation loss has flattened, which is exactly what the patience rule detects. The labels are a noise-free function of the features, so what separates the two curves is not noise but the *exact placement of thresholds* that finite data cannot pin down "
    f"(section 4.3 shows that the remaining errors sit next to the teacher's thresholds). The curves are stored every 10 rounds and include the {run_cfg['early_stopping_rounds']} patience rounds after the kept one."
))

# %% [markdown]
# ### 3.4 DAgger: training on the states the model itself visits `[saved; BB/100 from the legacy protocol]`
#
# **The problem.** Plain behaviour cloning only sees states that the *behaviour* policies visit. A learner that makes small mistakes drifts into states the dataset covers thinly
# (*covariate shift*), where it may err more. **DAgger** (Dataset Aggregation) closes this loop: let the *current model* play, let the **teacher relabel every state it visits**, add those
# labelled states to the training set and refit. In round `k` the teacher acts instead of the model with probability beta (here 0.5, halved every round) so that early rounds are not hopelessly off-distribution.
# Here: round 0 = behaviour cloning on 150,000 rows, then 3 rounds of ~10,000 hands each (about 3 labelled decisions per hand).
#
# Every round is evaluated on **three** held-out things, kept apart: imitation on the dataset's validation split, imitation on **fresh on-policy states** (4,000 validation hands played by *that round's own model*,
# so the gap between the two is the covariate shift), and **validation BB/100** (duplicate deals in the `dataset_val` namespace, 1,000 pairs per opponent). The final model is the round with the best mean validation
# BB/100 (ties: the earlier round) - a rule that selects on a noisy quantity, so we also look at the paired differences.
#
# **Protocol caveat.** These BB/100 tables were computed *at training time* with the **legacy single-table duplicate protocol**, in which the two hands of a deal shared agents and opponent trackers (section 4.4 explains why that is biased for
# agents that adapt to opponent statistics). They are shown for transparency and are **informational**; the imitation and on-policy agreement columns, which involve no games between adaptive agents, are not affected.

# %%
rounds = meta["dagger"]
opps = list(meta["policy_validation"]["opponents"])
display(Markdown("> **`evaluation_protocol_note` in `metadata.json`:** " + meta.get("evaluation_protocol_note", "(missing)")))
dag = pd.DataFrame(
    {
        "round": [("0 (BC)" if r["round"] == 0 else str(r["round"])) + (" **selected**" if r["selected"] else "") for r in rounds],
        "train rows": [f"{r['n_train_rows']:,}" for r in rounds],
        "teacher beta": ["-" if r["teacher_beta_of_added_rows"] is None else f"{r['teacher_beta_of_added_rows']:g}" for r in rounds],
        "trees / class": [r["best_iteration"] + 1 for r in rounds],
        "val. agreement": [pct(r["bc_validation"]["masked_accuracy"], 3) for r in rounds],
        "on-policy agreement": [pct(r["on_policy_validation"]["masked_accuracy"], 3) for r in rounds],
        "shift (pp)": [100 * (r["bc_validation"]["masked_accuracy"] - r["on_policy_validation"]["masked_accuracy"]) for r in rounds],
        "mean BB/100 (legacy protocol)": [r["mean_bb_per_100"] for r in rounds],
    }
)
display(md_table(dag, "{:.2f}"))
rows = []
for r in rounds:
    row = {"round": r["round"]}
    for o in opps:
        v = r["policy_validation"][o]
        row[o] = ci_text(v["bb_per_100"], *v["ci95_pairs"])
    rows.append(row)
display(Markdown("**Validation BB/100 per opponent - legacy single-table protocol, informational, see `evaluation_protocol_note`** (`[95% CI over 1,000 duplicate pairs]`; XGBoost's result against the opponent):"))
display(md_table(pd.DataFrame(rows)))
paired = [
    {"round": r["round"], **{o: ci_text(r["paired_vs_bc"][o]["bb_per_100"], *r["paired_vs_bc"][o]["ci95_pairs"]) for o in opps}}
    for r in rounds if r["round"] > 0
]
display(Markdown("**Paired difference to round 0 - legacy single-table protocol, informational** (identical deals in every round; round minus behaviour cloning, BB/100):"))
display(md_table(pd.DataFrame(paired)))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
xs = [r["round"] for r in rounds]
ax.plot(xs, [100 * r["bc_validation"]["masked_accuracy"] for r in rounds], "o-", color=nbp.ACCENT, label="dataset validation split")
ax.plot(xs, [100 * r["on_policy_validation"]["masked_accuracy"] for r in rounds], "s--", color=nbp.SERIES[2], label="fresh on-policy states")
ax.set_xticks(xs, ["0\n(BC)"] + [str(x) for x in xs[1:]])
ax.set_xlabel("DAgger round")
ax.set_ylabel("masked agreement with the teacher (%)")
ax.legend(loc="lower right")
nbp.title(ax, "Imitation per round", "the gap between the curves is the covariate shift")
series = {f"round {r['round']} - round 0": ([r["paired_vs_bc"][o]["bb_per_100"] for o in opps], [r["paired_vs_bc"][o]["ci95_pairs"][0] for o in opps], [r["paired_vs_bc"][o]["ci95_pairs"][1] for o in opps]) for r in rounds if r["round"] > 0}
nbp.forest_plot(ax2, opps, series, x_label="validation BB/100, paired difference to behaviour cloning (95% CI over pairs; legacy protocol)", legend=False)
legend_below(ax2, list(series))
nbp.title(ax2, "Does DAgger change how it plays?", "legacy single-table protocol (informational); same deals in every round; hollow = CI contains 0")
fig.tight_layout()
nbp.savefig("nb02_dagger", fig)
plt.show()

bc, last = rounds[0], rounds[-1]
shift_bc = bc["bc_validation"]["masked_accuracy"] - bc["on_policy_validation"]["masked_accuracy"]
shift_last = last["bc_validation"]["masked_accuracy"] - last["on_policy_validation"]["masked_accuracy"]
resolved = [(r["round"], o) for r in rounds if r["round"] > 0 for o in opps if r["paired_vs_bc"][o]["ci95_pairs"][0] > 0 or r["paired_vs_bc"][o]["ci95_pairs"][1] < 0]
mean_bb = ", ".join(f"{r['mean_bb_per_100']:.0f}" for r in rounds)
maniac_half_width = (bc["policy_validation"]["maniac"]["ci95_pairs"][1] - bc["policy_validation"]["maniac"]["ci95_pairs"][0]) / 2
resolved_text = (" (" + ", ".join(f"round {k} vs {o}" for k, o in resolved) + ")") if resolved else ""
display(Markdown(
    f"**Honest conclusion `[saved]`** (the BB/100 parts rest on the legacy protocol and are informational). (1) *Covariate shift is small and DAgger shrinks it a little*: behaviour cloning agrees with the teacher on {pct(bc['bc_validation']['masked_accuracy'], 2)} of the dataset's validation states but "
    f"on {pct(bc['on_policy_validation']['masked_accuracy'], 2)} of its own states (gap {100 * shift_bc:.2f} pp); after three rounds the gap is {100 * shift_last:.2f} pp "
    f"(on-policy {pct(last['on_policy_validation']['masked_accuracy'], 2)}). (2) *Part of that is simply more data*: the DAgger model trains on {last['n_train_rows']:,} rows, and the sweep above shows that "
    f"{int(more_rows['n_train_rows']):,} plain behaviour-cloning rows reach {pct(more_rows['masked_accuracy'], 3)} on the validation split against {pct(last['bc_validation']['masked_accuracy'], 3)} for the DAgger model - "
    f"so these artifacts cannot attribute the gain to *on-policy* data rather than to *more* data (that would need a plain-BC model with the same number of rows evaluated on-policy, which was not run). "
    f"(3) *No round is resolved as stronger or weaker*: of the {3 * len(opps)} paired round-vs-round-0 comparisons (legacy protocol), {len(resolved)} have a confidence interval that excludes zero{resolved_text}. "
    f"(4) *The candidates are near-tied*: mean validation BB/100 is {mean_bb} for rounds 0-3, differences far inside the noise of the opponents with the largest swings (the maniac's CI half-width is about "
    f"{maniac_half_width:.0f} BB/100), so round {meta['selected_round']} was picked by a rule that cannot separate the rounds - its selection is not evidence that it is the best one - and its stored validation numbers are mildly optimistic."
))

# %% [markdown]
# ### 3.5 A live toy fit: the XGBoost API, learning curves and why capacity matters `[live]`
#
# This is **not** the committed model. It fits three small boosters on a 20,000-row sample of the training split (whole hands, seeded) with the committed settings except for the tree depth,
# using XGBoost's own API (`QuantileDMatrix`, `xgb.train`, `evals_result`, early stopping) and scores them on the *full* validation split. Depth 1 means every tree splits on a single feature, so the ensemble is
# **additive** (the same model class as an EBM without interaction terms); depth 2 lets each tree combine two features; depth 8 is the committed setting.

# %%
toy_train_frame = subsample_by_hand(frame[frame["split"] == "train"].reset_index(drop=True), LIVE_FIT_ROWS, run_cfg["script_configuration"]["RANDOM_SEED"])
toy_train, toy_val = SplitArrays.from_frame(toy_train_frame), SplitArrays.from_frame(VAL)
toy_weights = class_weights(toy_train.y, meta["class_weight_power"])
feature_list = list(FEATURE_NAMES)
toy_params = {**booster, "nthread": N_THREADS}
toy_runs = {}
started = time.perf_counter()
for depth in LIVE_FIT_DEPTHS:
    t0 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(toy_train.X, toy_train.y, weight=toy_weights[toy_train.y], feature_names=feature_list, max_bin=booster["max_bin"])
    dval = xgb.QuantileDMatrix(toy_val.X, toy_val.y, weight=toy_weights[toy_val.y], ref=dtrain, feature_names=feature_list, max_bin=booster["max_bin"])
    history = {}
    fitted = xgb.train(
        {**toy_params, "max_depth": depth}, dtrain, num_boost_round=LIVE_FIT_MAX_ROUNDS,
        evals=[(dtrain, "train"), (dval, "val")], early_stopping_rounds=run_cfg["early_stopping_rounds"], evals_result=history, verbose_eval=False,
    )  # fmt: skip
    kept = fitted[: fitted.best_iteration + 1]
    metrics = evaluate_classifier(class_probabilities(kept, toy_val.X), toy_val.y, toy_val.legal, toy_val.street)
    toy_runs[depth] = {"history": history, "rounds": fitted.best_iteration + 1, "metrics": metrics, "seconds": time.perf_counter() - t0, "hit_cap": fitted.best_iteration + 1 >= LIVE_FIT_MAX_ROUNDS}

fig, ax = plt.subplots(figsize=(10.0, 4.6))
for depth, color in zip(LIVE_FIT_DEPTHS, (nbp.SERIES[2], nbp.SERIES[4], nbp.ACCENT)):
    history = toy_runs[depth]["history"]
    ax.plot(history["val"]["mlogloss"], color=color, label=f"depth {depth}: validation")
    ax.plot(history["train"]["mlogloss"], color=color, linestyle=":", linewidth=1.6)
ax.plot(curve["round"], curve["val_mlogloss"], color=nbp.TEXT, linewidth=1.2, alpha=0.7, label="committed model (245k rows, depth 8): validation")
ax.set_yscale("log")
ax.set_xlabel("boosting round")
ax.set_ylabel("class-weighted log-loss (log scale)")
ax.legend(fontsize=8.5, loc="upper right")
nbp.title(ax, f"Toy fits on {len(toy_train):,} rows", "solid: validation loss, dotted: training loss of the same fit")
fig.tight_layout()
nbp.savefig("nb02_live_fit", fig)
plt.show()
toy_table = pd.DataFrame(
    {
        "tree depth": [f"{d}" + (" (additive)" if d == 1 else " (pairwise)" if d == 2 else " (committed)" if d == booster["max_depth"] else "") for d in LIVE_FIT_DEPTHS],
        "rounds kept": [f"{toy_runs[d]['rounds']}" + (" (hit the cap)" if toy_runs[d]["hit_cap"] else "") for d in LIVE_FIT_DEPTHS],
        "fit s": [toy_runs[d]["seconds"] for d in LIVE_FIT_DEPTHS],
        "masked accuracy": [pct(toy_runs[d]["metrics"]["masked_accuracy"], 2) for d in LIVE_FIT_DEPTHS],
        "macro-F1": [pct(toy_runs[d]["metrics"]["masked_macro_f1"], 2) for d in LIVE_FIT_DEPTHS],
        "log-loss": [toy_runs[d]["metrics"]["log_loss"] for d in LIVE_FIT_DEPTHS],
    }
)
display(md_table(toy_table, "{:.3f}"))
acc = {d: toy_runs[d]["metrics"]["masked_accuracy"] for d in LIVE_FIT_DEPTHS}
display(Markdown(
    f"**[live]** ({time.perf_counter() - started:.0f} s for {len(LIVE_FIT_DEPTHS)} fits.) With the same {len(toy_train):,} rows an additive boosted model reaches {pct(acc[LIVE_FIT_DEPTHS[0]], 1)}, allowing two-feature interactions "
    f"{pct(acc[LIVE_FIT_DEPTHS[1]], 1)}, and depth {LIVE_FIT_DEPTHS[-1]} {pct(acc[LIVE_FIT_DEPTHS[-1]], 1)} - "
    f"the committed model, with {meta['n_train_rows'] / len(toy_train):.0f}x the data, reaches {pct(meta['imitation']['masked_accuracy'], 2)}. The step from depth 2 to depth 8 is what the teacher's *nested* structure (a threshold that moves with the street, "
    f"the bet size, the opponent's statistics) needs; the EBM gets part of the way with {ebm_meta['n_interaction_terms']} automatically selected pairwise tables (its saved agreement is {pct(ebm_meta['imitation']['masked_accuracy'], 1)}). "
    f"This is a toy fit on one sample: it shows the API and the *direction* of the capacity effect, not exact numbers."
))

# %% [markdown]
# ## 4. Model results
#
# **Imitation and policy strength are two different questions and are answered separately**: 4.1-4.3 ask *how closely does the model reproduce the teacher on held-out states?*; 4.4 asks *how well does the
# resulting player do in actual games?*
#
# ### 4.1 Imitation on the validation split `[recomputed]`
#
# We load the saved agent, predict all validation decisions with the agent's own inference path (`class_probabilities`, i.e. `Booster.inplace_predict` on float32 rows), take the
# arg-max over the **legal** actions (what the agent plays), and recompute every headline metric. The recomputation must reproduce the numbers stored by the training pipeline. The EBM is evaluated the
# same way on the same states. Confidence intervals come from a **hand-level (cluster) bootstrap**: the 3-4 decisions of one hand are strongly dependent, so hands, not decisions, are resampled.
# Caveat repeated from the artifact: the tree count and hyper-parameters were chosen on this very split, so these numbers are slightly optimistic; the on-policy numbers of section 3.4 are not.

# %%
P_XGB = class_probabilities(AGENT.booster, X_VAL)
assert [int(c) for c in EBM.model.classes_] == list(range(5))
P_EBM = EBM.model.predict_proba(X_VAL)
PRED_XGB, PRED_EBM = masked_argmax(P_XGB, L_VAL), masked_argmax(P_EBM, L_VAL)
M_XGB = evaluate_classifier(P_XGB, Y_VAL, L_VAL, STREET_VAL)
M_EBM = evaluate_classifier(P_EBM, Y_VAL, L_VAL, STREET_VAL)
saved_keys = ("masked_accuracy", "masked_macro_f1", "masked_balanced_accuracy", "log_loss", "illegal_mass")
max_diff = max(abs(M_XGB[k] - meta["imitation"][k]) for k in saved_keys)
assert max_diff < 1e-9, max_diff  # the saved model reproduces the saved metrics exactly

# ---- hand-level bootstrap (resample whole hands) ----
hand_codes, hand_ids = pd.factorize(HANDS_VAL)
n_hands = len(hand_ids)
hand_rows = np.bincount(hand_codes, minlength=n_hands).astype(float)
boot_rng = np.random.default_rng(ANALYSIS_SEED)
BOOT_INDEX = boot_rng.integers(0, n_hands, size=(BOOTSTRAP_REPS, n_hands), dtype=np.int32)  # the same resampled hands for every statistic


def hand_bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    """Mean of a per-decision quantity with a 95% cluster-bootstrap interval over hands."""
    sums = np.bincount(hand_codes, weights=values.astype(float), minlength=n_hands)
    draws = np.array([sums[chunk].sum() / hand_rows[chunk].sum() for chunk in BOOT_INDEX])
    return float(values.mean()), *np.percentile(draws, [2.5, 97.5])


row_loss = {name: -np.log(np.clip(P[np.arange(N_VAL), Y_VAL], 1e-12, 1.0)) for name, P in (("XGBoost", P_XGB), ("EBM", P_EBM))}
row_error = {"XGBoost": (PRED_XGB != Y_VAL).astype(float), "EBM": (PRED_EBM != Y_VAL).astype(float)}
err_ci = {name: hand_bootstrap(v) for name, v in row_error.items()}
loss_ci = {name: hand_bootstrap(v) for name, v in row_loss.items()}
err_diff_ci = hand_bootstrap(row_error["EBM"] - row_error["XGBoost"])
metric_rows = [
    ("masked accuracy", pct(1 - err_ci["XGBoost"][0], 3), f"[{pct(1 - err_ci['XGBoost'][2], 2)}, {pct(1 - err_ci['XGBoost'][1], 2)}]", pct(1 - err_ci["EBM"][0], 3), f"[{pct(1 - err_ci['EBM'][2], 2)}, {pct(1 - err_ci['EBM'][1], 2)}]"),
    ("masked macro-F1", pct(M_XGB["masked_macro_f1"], 3), "", pct(M_EBM["masked_macro_f1"], 3), ""),
    ("masked balanced accuracy", pct(M_XGB["masked_balanced_accuracy"], 3), "", pct(M_EBM["masked_balanced_accuracy"], 3), ""),
    ("log-loss (raw probabilities)", f"{M_XGB['log_loss']:.4f}", f"[{loss_ci['XGBoost'][1]:.4f}, {loss_ci['XGBoost'][2]:.4f}]", f"{M_EBM['log_loss']:.4f}", f"[{loss_ci['EBM'][1]:.4f}, {loss_ci['EBM'][2]:.4f}]"),
    ("raw accuracy (no legal mask)", pct(M_XGB["accuracy"], 3), "", pct(M_EBM["accuracy"], 3), ""),
    ("mean probability on illegal actions", f"{M_XGB['illegal_mass']:.1e}", "", f"{M_EBM['illegal_mass']:.1e}", ""),
    ("decisions wrong (of %s)" % f"{N_VAL:,}", f"{int(row_error['XGBoost'].sum())}", "", f"{int(row_error['EBM'].sum())}", ""),
]
display(Markdown(
    f"**[recomputed = saved]** The recomputed XGBoost metrics equal the saved ones to {max_diff:.0e}. Intervals: {BOOTSTRAP_REPS:,} cluster-bootstrap resamples of the {n_hands:,} validation hands."
))
display(md_table(pd.DataFrame(metric_rows, columns=["metric", "XGBoost", "95% CI", "EBM", "95% CI "])))
display(Markdown(
    f"The XGBoost agent makes **{int(row_error['XGBoost'].sum())} errors in {N_VAL:,} decisions** ({pct(err_ci['XGBoost'][0], 2)} error rate), the EBM {int(row_error['EBM'].sum())} ({pct(err_ci['EBM'][0], 2)}); "
    f"paired on the same hands, the EBM makes {100 * err_diff_ci[0]:.2f} percentage points more errors (95% CI [{100 * err_diff_ci[1]:.2f}, {100 * err_diff_ci[2]:.2f}]). "
    f"Log-loss is {M_EBM['log_loss'] / M_XGB['log_loss']:.1f}x higher for the EBM: it is *less sharp*, not only less accurate."
))

# %%
# ---- confusion matrix and per-class precision / recall ----
cm = np.array(M_XGB["confusion_matrix"])
row_share = cm / cm.sum(axis=1, keepdims=True)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
image = ax.imshow(row_share, cmap=nbp.SEQUENTIAL, norm=PowerNorm(0.35, vmin=0, vmax=1))
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9.5, color=nbp.TEXT if row_share[i, j] < 0.6 else "#06110c", fontweight="bold" if i == j else "normal")
ax.set_xticks(range(5), ACTIONS, rotation=25, ha="right")
ax.set_yticks(range(5), ACTIONS)
ax.set_xlabel("XGBoost (legal-masked arg-max)")
ax.set_ylabel("teacher label")
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
nbp.title(ax, "Confusion matrix (validation, counts)", "colour = share of the teacher-label row (power scale, so rare errors stay visible)")
positions = np.arange(5)
for k, (name, M, color, marker) in enumerate((("XGBoost", M_XGB, XGB_COLOR, "o"), ("EBM", M_EBM, EBM_COLOR, "s"))):
    recall = [M["per_class"][a]["recall"] for a in ACTIONS]
    precision = [M["per_class"][a]["precision"] for a in ACTIONS]
    ax2.scatter(recall, positions - 0.12 + 0.24 * k, color=color, marker=marker, s=70, label=f"{name} recall", zorder=3)
    ax2.scatter(precision, positions - 0.12 + 0.24 * k, facecolors="none", edgecolors=color, marker=marker, s=70, linewidths=1.8, label=f"{name} precision", zorder=3)
ax2.set_yticks(positions, [f"{a}  (n={M_XGB['per_class'][a]['support']:,})" for a in ACTIONS])
ax2.invert_yaxis()
ax2.set_xlim(0.92, 1.002)
ax2.set_xlabel("per-class recall (filled) and precision (hollow), validation")
ax2.grid(axis="y", visible=False)
ax2.legend(ncols=2, loc="lower left", fontsize=8.5)
nbp.title(ax2, "Per-class recall and precision", "zoomed to 92-100%; the EBM is weakest where classes sit next to each other")
fig.tight_layout()
nbp.savefig("nb02_confusion", fig)
plt.show()
class_rows = [
    {"action": f"`{a}`", "support": M_XGB["per_class"][a]["support"], "XGB recall": pct(M_XGB["per_class"][a]["recall"], 2), "XGB precision": pct(M_XGB["per_class"][a]["precision"], 2),
     "EBM recall": pct(M_EBM["per_class"][a]["recall"], 2), "EBM precision": pct(M_EBM["per_class"][a]["precision"], 2)}
    for a in ACTIONS
]
display(md_table(pd.DataFrame(class_rows)))
off_diagonal = [(ACTIONS[i], ACTIONS[j], int(cm[i, j])) for i in range(5) for j in range(5) if i != j]
off_diagonal.sort(key=lambda t: -t[2])
worst_recall = min(ACTIONS, key=lambda a: M_XGB["per_class"][a]["recall"])
display(Markdown(
    f"**[recomputed]** The most frequent confusions (teacher label -> XGBoost) are " + ", ".join(f"`{a}` -> `{b}` ({n})" for a, b, n in off_diagonal[:4]) +
    f". The class with the lowest recall is `{worst_recall}` ({pct(M_XGB['per_class'][worst_recall]['recall'], 2)}), but every class is above {pct(min(M_XGB['per_class'][a]['recall'] for a in ACTIONS), 1)}: "
    f"rare classes are not sacrificed (class weights, section 2.2)."
))

# %%
# ---- accuracy by street ----
street_names = list(STREET_NAMES)
street_stats = {}
for name, error in row_error.items():
    for s, label in enumerate(street_names):
        mask = STREET_VAL == s
        sums = np.bincount(hand_codes[mask], weights=error[mask], minlength=n_hands)
        cnt = np.bincount(hand_codes[mask], minlength=n_hands).astype(float)
        draws = np.array([sums[c].sum() / max(cnt[c].sum(), 1.0) for c in BOOT_INDEX])
        street_stats[name, label] = (error[mask].mean(), *np.percentile(draws, [2.5, 97.5]), int(mask.sum()), int(error[mask].sum()))
fig, ax = plt.subplots(figsize=(9.6, 3.9))
width = 0.36
for k, (name, color) in enumerate((("XGBoost", XGB_COLOR), ("EBM", EBM_COLOR))):
    est = np.array([100 * street_stats[name, s][0] for s in street_names])
    lo = np.array([100 * street_stats[name, s][1] for s in street_names])
    hi = np.array([100 * street_stats[name, s][2] for s in street_names])
    x_pos = np.arange(4) + (k - 0.5) * width
    ax.bar(x_pos, est, width=width * 0.92, color=color, label=name)
    ax.vlines(x_pos, lo, hi, color=nbp.TEXT, linewidth=1.3)
    for xv, e, h in zip(x_pos, est, hi):
        ax.annotate(f"{e:.2f}%", (xv, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=nbp.MUTED)
ax.set_xticks(np.arange(4), [f"{s}\n(n={street_stats['XGBoost', s][3]:,})" for s in street_names])
ax.set_ylabel("error rate against the teacher (%)")
ax.grid(axis="x", visible=False)
ax.legend()
nbp.title(ax, "Error rate by street", "bars: error rate; whiskers: 95% hand-level bootstrap CI")
fig.tight_layout()
nbp.savefig("nb02_street_errors", fig)
plt.show()
worst_street = max(street_names, key=lambda s: street_stats["XGBoost", s][0])
best_street = min(street_names, key=lambda s: street_stats["XGBoost", s][0])
display(Markdown(
    f"**[recomputed]** XGBoost's error rate is highest on the **{worst_street}** ({pct(street_stats['XGBoost', worst_street][0], 2)}, {street_stats['XGBoost', worst_street][4]} errors) and lowest on the **{best_street}** "
    f"({pct(street_stats['XGBoost', best_street][0], 2)}); the intervals are wide because only {int(row_error['XGBoost'].sum())} errors are spread over four streets. The EBM is worst post-flop too "
    f"({', '.join(f'{s} {pct(street_stats['EBM', s][0], 1)}' for s in street_names)}), where the rules combine equity with bet size, aggression and draws."
))

# %% [markdown]
# ### 4.2 Calibration `[recomputed]`
#
# When the model says "98% sure", is it right 98% of the time? The reliability check uses the probabilities the agent really acts on - **masked to the legal actions and renormalised** - and
# the confidence of the action it would play. Because the model is right 99.8% of the time, almost every decision sits in the top bin of an ordinary reliability diagram; the diagram below therefore
# plots the *error* probability `1 - confidence` against the observed error rate on **log-log axes**, with the bins chosen to resolve the tail. Points on the diagonal are calibrated; hollow markers are bins in which
# *no* error occurred (the marker shows the upper bound 1/n).

# %%
def masked_confidence(probabilities: np.ndarray, legal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(legal, probabilities, 0.0)
    masked = masked / masked.sum(axis=1, keepdims=True)
    return masked.argmax(axis=1), masked.max(axis=1)


EDGES = np.array([0.0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.2, 0.35, 0.5])
calibration = {}
for name, P in (("XGBoost", P_XGB), ("EBM", P_EBM)):
    pred, conf = masked_confidence(P, L_VAL)
    wrong = (pred != Y_VAL).astype(float)
    which = np.digitize(1.0 - conf, EDGES[1:-1], right=False)
    rows_ = []
    for b in range(len(EDGES) - 1):
        mask = which == b
        if mask.sum():
            rows_.append((float((1.0 - conf[mask]).mean()), float(wrong[mask].mean()), int(mask.sum()), int(wrong[mask].sum())))
    ece_bins = np.linspace(0.5, 1.0, 11)
    which_ece = np.clip(np.digitize(conf, ece_bins) - 1, 0, 9)
    ece = sum(abs(conf[which_ece == b].mean() - (1 - wrong[which_ece == b].mean())) * (which_ece == b).mean() for b in range(10) if (which_ece == b).any())
    calibration[name] = {"bins": rows_, "ece": float(ece), "mean_conf": float(conf.mean()), "conf_wrong": float(conf[wrong == 1].mean()), "conf_right": float(conf[wrong == 0].mean())}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]})
lims = (3e-6, 0.6)
ax.plot(lims, lims, color=nbp.MUTED, linestyle="--", linewidth=1.2, label="perfectly calibrated")
for name, color, marker in (("XGBoost", XGB_COLOR, "o"), ("EBM", EBM_COLOR, "s")):
    pts = calibration[name]["bins"]
    predicted = np.array([p[0] for p in pts])
    observed = np.array([p[1] for p in pts])
    n_bin = np.array([p[2] for p in pts])
    has_error = observed > 0
    ax.plot(predicted[has_error], observed[has_error], marker + "-", color=color, markersize=7, label=name)
    ax.plot(predicted[~has_error], 1.0 / n_bin[~has_error], marker, color=color, markerfacecolor=nbp.SURFACE, markersize=8, markeredgewidth=1.6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("mean predicted error probability, 1 - confidence (log scale)")
ax.set_ylabel("observed error rate (log scale)")
ax.legend(loc="upper left")
nbp.title(ax, "Reliability of the played action", "below the diagonal = under-confident, above = over-confident")
CONF_RANGES = [(0.999, 1.0001, ">= 99.9%"), (0.99, 0.999, "99-99.9%"), (0.9, 0.99, "90-99%"), (0.0, 0.9, "< 90%")]
range_stats = {}
for name, P in (("XGBoost", P_XGB), ("EBM", P_EBM)):
    pred, conf = masked_confidence(P, L_VAL)
    for lo, hi, label in CONF_RANGES:
        mask = (conf >= lo) & (conf < hi)
        range_stats[name, label] = (int(mask.sum()), float((1 - conf[mask]).sum()), int((pred[mask] != Y_VAL[mask]).sum()))
x_pos = np.arange(len(CONF_RANGES))
for k, (name, color) in enumerate((("XGBoost", XGB_COLOR), ("EBM", EBM_COLOR))):
    expected = [max(range_stats[name, r[2]][1], 0.3) for r in CONF_RANGES]
    observed = [max(range_stats[name, r[2]][2], 0.3) for r in CONF_RANGES]
    offset = (k - 0.5) * 0.38
    ax2.bar(x_pos + offset, observed, width=0.34, color=color, label=f"{name}: observed errors")
    ax2.bar(x_pos + offset, expected, width=0.34, facecolor="none", edgecolor=nbp.TEXT, linewidth=1.4, linestyle="--", label="expected from its own confidences" if k == 0 else None)
ax2.set_yscale("log")
ax2.set_xticks(x_pos, [f"{r[2]}\nXGB n={range_stats['XGBoost', r[2]][0]:,}\nEBM n={range_stats['EBM', r[2]][0]:,}" for r in CONF_RANGES], fontsize=8)
ax2.set_xlabel("confidence of the action played (n = decisions in the range)")
ax2.set_ylabel("errors (log scale)")
ax2.legend(fontsize=8.2, loc="upper left")
nbp.title(ax2, "Expected vs observed errors", "dashed outline = sum of (1 - confidence): the errors the model itself predicts")
fig.tight_layout()
nbp.savefig("nb02_calibration", fig)
plt.show()
table = pd.DataFrame(
    {
        "model": ["XGBoost", "EBM"],
        "mean confidence": [pct(calibration[n]["mean_conf"], 2) for n in ("XGBoost", "EBM")],
        "accuracy": [pct(1 - err_ci[n][0], 2) for n in ("XGBoost", "EBM")],
        "mean confidence when right": [pct(calibration[n]["conf_right"], 2) for n in ("XGBoost", "EBM")],
        "mean confidence when wrong": [pct(calibration[n]["conf_wrong"], 1) for n in ("XGBoost", "EBM")],
        "ECE (10 bins)": [f"{calibration[n]['ece']:.4f}" for n in ("XGBoost", "EBM")],
    }
)
display(md_table(table))
expected_total = {n: sum(range_stats[n, r[2]][1] for r in CONF_RANGES) for n in ("XGBoost", "EBM")}
display(Markdown(
    f"**[recomputed]** Both models' confidences are informative: when XGBoost is wrong its mean confidence is {pct(calibration['XGBoost']['conf_wrong'], 0)}, when it is right {pct(calibration['XGBoost']['conf_right'], 2)}; "
    f"its expected calibration error is {calibration['XGBoost']['ece']:.4f} (EBM {calibration['EBM']['ece']:.4f}). Summing the model's own error probabilities predicts {expected_total['XGBoost']:.0f} errors for XGBoost "
    f"(observed {int(row_error['XGBoost'].sum())}) and {expected_total['EBM']:.0f} for the EBM (observed {int(row_error['EBM'].sum())}): both models are, if anything, slightly *under*-confident in the bulk and well calibrated in the tail bins (points on or just below the diagonal). "
    f"Calibration here is about *agreement with the teacher*, not about whether the action is good poker."
))

# %% [markdown]
# ### 4.3 Error analysis `[recomputed]`
#
# The teacher's rules are sharp thresholds, so the errors of a smooth learner should sit **next to the thresholds**. The dataset stores, for every decision, the teacher's *margin* - how far (in equity or percentile points)
# the deciding quantity lies from the threshold that would flip the decision - and the rule that fired. Both are used below.

# %%
margin_edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.01]
margin_bin = np.digitize(MARGIN_VAL, margin_edges[1:-1], right=False)
margin_labels = [f"{lo:g}-{hi:g}" if hi < 1 else f">{lo:g}" for lo, hi in zip(margin_edges[:-1], margin_edges[1:])]
margin_stats = {}
for name, error in row_error.items():
    margin_stats[name] = [(error[margin_bin == b].mean(), int(error[margin_bin == b].sum()), int((margin_bin == b).sum())) for b in range(len(margin_labels))]
rule_table = (
    pd.DataFrame({"rule": RULE_VAL, "xgb": row_error["XGBoost"], "ebm": row_error["EBM"]}).groupby("rule").agg(n=("xgb", "size"), xgb_errors=("xgb", "sum"), ebm_errors=("ebm", "sum")).sort_values(["xgb_errors", "ebm_errors"], ascending=False)
)
top_rules = rule_table.head(8)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.7), gridspec_kw={"width_ratios": [1.0, 1.1]})
x_pos = np.arange(len(margin_labels))
for name, color, marker in (("XGBoost", XGB_COLOR, "o"), ("EBM", EBM_COLOR, "s")):
    rates = np.array([max(m[0], 0.5 / m[2]) for m in margin_stats[name]])
    ax.plot(x_pos, 100 * rates, marker + "-", color=color, label=name, markersize=7)
ax.set_yscale("log")
ax.set_xticks(x_pos, margin_labels, rotation=25, ha="right")
ax.set_xlabel("teacher margin: distance from the decision boundary (equity / percentile points)")
ax.set_ylabel("error rate (%, log scale)")
ax.legend()
nbp.title(ax, "Errors sit next to the thresholds", "error rate against the teacher by distance from the decision boundary")
labels_rules = [f"{r} (n={int(row.n):,})" for r, row in top_rules.iterrows()]
y_pos = np.arange(len(top_rules))
ax2.barh(y_pos - 0.2, top_rules["xgb_errors"], height=0.38, color=XGB_COLOR, label="XGBoost errors")
ax2.barh(y_pos + 0.2, top_rules["ebm_errors"], height=0.38, color=EBM_COLOR, label="EBM errors")
for yv, (a, b) in enumerate(zip(top_rules["xgb_errors"], top_rules["ebm_errors"])):
    ax2.annotate(f"{int(a)}", (a, yv - 0.2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=nbp.MUTED)
    ax2.annotate(f"{int(b)}", (b, yv + 0.2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=nbp.MUTED)
ax2.set_yticks(y_pos, labels_rules)
ax2.invert_yaxis()
ax2.grid(axis="y", visible=False)
ax2.set_xlabel("decisions wrong (validation)")
ax2.legend(loc="lower right")
nbp.title(ax2, "...and in the rules that weigh price against equity", "the eight teacher rules with the most XGBoost errors")
fig.tight_layout()
nbp.savefig("nb02_errors", fig)
plt.show()

err_mask = row_error["XGBoost"] == 1
near = MARGIN_VAL < 0.01
EQUITY_LINES = {"cbet_min_equity": PARAMS.cbet_min_equity, "wet_pot_equity": PARAMS.wet_pot_equity, "pot_bet_equity": PARAMS.pot_bet_equity, "raise_equity": PARAMS.raise_equity, "monster_equity": PARAMS.monster_equity, "penalty_relief_equity": PARAMS.penalty_relief_equity}
line_distance = np.min(np.abs(VAL["equity"].to_numpy()[:, None] - np.array(list(EQUITY_LINES.values()))[None, :]), axis=1)
far = err_mask & (MARGIN_VAL > 0.2)
far_near_other = far & (line_distance < 0.001)
far_text = (
    f"The margin only measures the comparison that *decided*: of the {int(far.sum())} errors whose margin exceeds 0.2, {int(far_near_other.sum())} have an equity within 0.001 of a *different* equity threshold "
    f"(e.g. 0.7595 against the 0.76 wet-board pot-bet line, 0.38 against the c-bet line; the equity is a Monte-Carlo estimate in steps of 0.0005); "
    f"the remaining {int((far & ~far_near_other).sum())} are pot-committed states (SPR at most {X_VAL[far & ~far_near_other, FEATURE_INDEX['spr']].max() if (far & ~far_near_other).any() else 0:.2f})."
    if far.sum() else ""
)
display(md_table(pd.DataFrame(
    {
        "teacher margin": margin_labels,
        "decisions": [m[2] for m in margin_stats["XGBoost"]],
        "XGBoost errors": [m[1] for m in margin_stats["XGBoost"]],
        "XGBoost error rate": [pct(m[0], 2) for m in margin_stats["XGBoost"]],
        "EBM errors": [m[1] for m in margin_stats["EBM"]],
        "EBM error rate": [pct(m[0], 2) for m in margin_stats["EBM"]],
    }
)))
display(Markdown(
    f"**[recomputed]** {pct(near[err_mask].mean(), 0)} of XGBoost's {int(err_mask.sum())} errors lie within 0.01 of a threshold, although only {pct(near.mean(), 1)} of all decisions do "
    f"(median margin of the errors {np.median(MARGIN_VAL[err_mask]):.3f}, of all decisions {np.median(MARGIN_VAL):.3f}); the error rate is {100 * margin_stats['XGBoost'][0][0]:.1f}% in the closest bin and "
    f"{100 * margin_stats['XGBoost'][-1][0]:.2f}% in the farthest. The rules with the most errors are " + ", ".join(f"`{r}` ({int(row.xgb_errors)} of {int(row.n):,})" for r, row in top_rules.head(4).iterrows()) +
    f" - the price-versus-adjusted-equity rules, whose boundary depends on four or five features at once (section 8 checks how closely that boundary is reproduced). "
    f"Errors are *not* spread over hopeless states: the model that gets them wrong is also the unsure one (section 4.2). "
    f"{far_text}"
))

# %% [markdown]
# ### 4.4 Playing strength: separate from imitation
#
# **Imitation accuracy does not tell us whether the agent wins chips.** Two views, never mixed:
#
# * **[saved] validation BB/100** from the training run: 1,000 duplicate pairs per opponent on deals of the `dataset_val` seed namespace. They were computed with the **legacy single-table protocol** (see below), so they are **informational**
#   (`evaluation_protocol_note` in `metadata.json`), and they *selected* the DAgger round, so they are mildly optimistic.
# * **[fresh] this notebook**: duplicate deals in the `eval` namespace with master seed `EVAL_SEED` - **different from every training and validation seed** (training 42, namespaces `dataset_train`, `dagger`, `dataset_val`; the official tournament 42 in `eval`).
#   Held out here: the **deals and the agents' random streams**, nothing else - the four opponents are all part of the dataset's opponent mixture, so the *opponents* are not new to the model. The EBM plays the same deals, and XGBoost plays the EBM directly.
#
# **Protocol: two-table duplicate.** Each deal is played twice with the seats complemented, so card luck cancels. XGBoost, the EBM and the teacher *adapt* to the opponent statistics they accumulate, and that created a trap in the original protocol: the two plays of a deal were made
# back to back against the same opponent model, so the second play faced a model that already contained the first play of the *same* deal - an information leak that favours the second mover and does not cancel even between two identical adaptive agents. Since then `pair_results`
# plays every match at **two independent tables** (own agent copies, own opponent trackers, the same deals, seats complemented), so the two plays of a deal cannot inform each other and identical adaptive agents cancel exactly. All fresh numbers here use it
# (`protocol="two_table"`, the default); the legacy `protocol="single_table"` is run once below as a labelled diagnostic, to show how large the artifact is on these very deals. The A/A test that exposed the leak (two identical teachers) is in Notebook 4, section 6 (6.2), and
# `docs/METHODOLOGY.md` section 5 records the history.
#
# BB/100 is the mean over pairs, the 95% interval a Student-t interval over pairs (`poker_arena.data.evaluation`); the teacher and the learned agents still adapt *within* a match, which makes consecutive pairs of a match slightly dependent, so the table also shows the more
# conservative **match-level** interval (50-pair matches).

# %%
EVAL_OPPONENTS = ["teacher", "call_station", "maniac", "random"]
PROTOCOL = "two_table"  # the default of pair_results; the legacy "single_table" is only used for the diagnostic below
XGB_SPEC, EBM_SPEC = AgentSpec("xgboost", {}, label="XGBoost"), AgentSpec("ebm", {}, label="EBM")
pairs_for = lambda opponent: EVAL_PAIRS_TEACHER if opponent == "teacher" else EVAL_PAIRS  # noqa: E731
started = time.perf_counter()
eval_pairs: dict[tuple[str, str], np.ndarray] = {}
for opponent in EVAL_OPPONENTS:
    for label, spec in (("XGBoost", XGB_SPEC), ("EBM", EBM_SPEC)):  # identical seed parts => identical deals and agent streams for both models
        eval_pairs[label, opponent] = pair_results(spec, policy_spec(opponent), pairs_for(opponent), (EVAL_SEED, opponent), GAME, N_WORKERS, namespace="eval", protocol=PROTOCOL)
eval_pairs["XGBoost", "EBM"] = pair_results(XGB_SPEC, EBM_SPEC, EVAL_PAIRS_TEACHER, (EVAL_SEED, "xgboost-vs-ebm"), GAME, N_WORKERS, namespace="eval", protocol=PROTOCOL)
eval_seconds = time.perf_counter() - started
fresh = {key: summarize(key[0], key[1], pairs) for key, pairs in eval_pairs.items()}
paired_vs_ebm = {o: summarize("XGBoost - EBM", o, eval_pairs["XGBoost", o] - eval_pairs["EBM", o]) for o in EVAL_OPPONENTS}
saved_val = meta["policy_validation"]["results"]
ebm_saved_val = ebm_meta["policy_validation"]["results"]


def result_text(r) -> str:
    return ci_text(r.bb_per_100, r.ci_low, r.ci_high)


rows = []
for o in EVAL_OPPONENTS:
    r = fresh["XGBoost", o]
    rows.append(
        {
            "opponent": f"`{o}`",
            "pairs": r.n_pairs,
            "XGBoost fresh [95% CI pairs]": result_text(r),
            "match-level CI": f"[{r.match_ci_low:+.0f}, {r.match_ci_high:+.0f}]",
            "verdict": verdict(r.ci_low, r.ci_high),
            "XGBoost saved validation (legacy protocol)": ci_text(saved_val[o]["bb_per_100"], *saved_val[o]["ci95_pairs"]),
            "EBM fresh": result_text(fresh["EBM", o]),
            "XGBoost - EBM (paired)": result_text(paired_vs_ebm[o]) + " " + ("(*)" if verdict(paired_vs_ebm[o].ci_low, paired_vs_ebm[o].ci_high) != "no significant difference" else ""),
        }
    )
head = fresh["XGBoost", "EBM"]
display(Markdown(f"**[fresh]** {eval_seconds:.0f} s on {N_WORKERS} worker processes. BB/100 of XGBoost against each opponent (`[95% CI]`); `(*)` = the paired XGBoost-minus-EBM interval excludes 0:"))
display(md_table(pd.DataFrame(rows)))
display(Markdown(
    f"**Head-to-head, XGBoost vs EBM** on {head.n_pairs} fresh duplicate pairs: {result_text(head)} BB/100 for XGBoost (pair CI), conservative match-level interval [{head.match_ci_low:+.0f}, {head.match_ci_high:+.0f}] -> **{verdict(head.ci_low, head.ci_high)}**."
))

# ---- diagnostic: the same deals under the LEGACY single-table protocol (never used for conclusions) ----
started = time.perf_counter()
legacy_specs = {("XGBoost", "teacher"): (XGB_SPEC, policy_spec("teacher"), (EVAL_SEED, "teacher")), ("EBM", "teacher"): (EBM_SPEC, policy_spec("teacher"), (EVAL_SEED, "teacher")), ("XGBoost", "EBM"): (XGB_SPEC, EBM_SPEC, (EVAL_SEED, "xgboost-vs-ebm"))}
legacy = {
    key: summarize(key[0], key[1], pair_results(a_, b_, EVAL_PAIRS_TEACHER, parts_, GAME, N_WORKERS, namespace="eval", protocol="single_table"))
    for key, (a_, b_, parts_) in legacy_specs.items()
}
legacy_seconds = time.perf_counter() - started
display(Markdown(
    f"**[fresh, diagnostic only]** ({legacy_seconds:.0f} s) The *same deals and seeds* replayed with the **legacy single-table protocol** (`protocol=\"single_table\"`), next to the two-table result used above; first-named agent's BB/100 [95% CI over pairs]:"
))
display(md_table(pd.DataFrame(
    [
        {
            "matchup": f"{a_} vs {b_}", "two-table (used)": result_text(fresh[a_, b_]), "legacy single-table (diagnostic)": result_text(legacy[a_, b_]),
            "legacy minus two-table": f"{legacy[a_, b_].bb_per_100 - fresh[a_, b_].bb_per_100:+.0f}",
        }
        for a_, b_ in legacy
    ]
)))
legacy_gap = legacy["XGBoost", "EBM"].bb_per_100 - fresh["XGBoost", "EBM"].bb_per_100

# ---- the official tournament (Notebook 4), if a run is committed: a much larger sample under the same two-table protocol ----
OFFICIAL = None
pointer = ARTIFACTS_DIR / "tournaments" / "latest.json"
if pointer.exists():
    try:
        from poker_arena.tournament import load_results

        official_run = load_results(ARTIFACTS_DIR / "tournaments" / json.loads(pointer.read_text(encoding="utf-8"))["run_id"])
        official_table = official_run.pairwise_table().set_index(["agent", "opponent"])
        OFFICIAL = {key: official_table.loc[key] for key in (("XGBoost", "EBM"), ("XGBoost", "Teacher"), ("EBM", "Teacher"))}
        OFFICIAL_PROTOCOL = str(official_table.loc[("XGBoost", "EBM")]["protocol"])
    except (KeyError, FileNotFoundError, ValueError):
        OFFICIAL = None
if OFFICIAL is not None:
    row_xe = OFFICIAL["XGBoost", "EBM"]
    display(Markdown(
        f"**[saved] The official tournament** (`artifacts/tournaments/<latest>`, protocol `{OFFICIAL_PROTOCOL}`, {int(row_xe['hands']):,} hands per pairing in {int(row_xe['n_sessions'])} sessions; Notebook 4), first-named agent's BB/100 with the session-level 95% interval, "
        f"next to this notebook's fresh result:\n\n"
        "| pairing | official [95% CI] | fresh here [95% CI] |\n|---|---|---|\n"
        + "\n".join(
            f"| {a_} vs {b_} | {ci_text(OFFICIAL[a_, b_]['bb_per_100'], OFFICIAL[a_, b_]['ci_low'], OFFICIAL[a_, b_]['ci_high'])} | {result_text(fresh[a_, 'EBM' if b_ == 'EBM' else 'teacher'])} |"
            for a_, b_ in (("XGBoost", "EBM"), ("XGBoost", "Teacher"), ("EBM", "Teacher"))
        )
    ))
    official_word = verdict(row_xe["ci_low"], row_xe["ci_high"])
    display(Markdown(
        f"The official XGBoost-vs-EBM interval is {(row_xe['ci_high'] - row_xe['ci_low']) / (head.ci_high - head.ci_low):.2f} times as wide as this notebook's ({int(row_xe['units']):,} pairs against {head.n_pairs}) and "
        f"{'contains' if row_xe['ci_low'] <= head.bb_per_100 <= row_xe['ci_high'] else 'does not contain'} the fresh point estimate ({head.bb_per_100:+.0f}); the fresh interval {'contains' if head.ci_low <= row_xe['bb_per_100'] <= head.ci_high else 'does not contain'} the official one ({row_xe['bb_per_100']:+.0f}). "
        f"Official verdict for XGBoost against the EBM: **{official_word}**."
    ))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.0, 1.25]})
series_teacher = {
    "XGBoost, saved validation (legacy protocol)": ([saved_val["teacher"]["bb_per_100"]], [saved_val["teacher"]["ci95_pairs"][0]], [saved_val["teacher"]["ci95_pairs"][1]]),
    "XGBoost, fresh (eval)": ([fresh["XGBoost", "teacher"].bb_per_100], [fresh["XGBoost", "teacher"].ci_low], [fresh["XGBoost", "teacher"].ci_high]),
    "EBM, fresh (eval)": ([fresh["EBM", "teacher"].bb_per_100], [fresh["EBM", "teacher"].ci_low], [fresh["EBM", "teacher"].ci_high]),
}
labels_a = ["vs teacher"]
nbp.forest_plot(ax, labels_a, series_teacher, x_label="BB/100 (duplicate deals, 95% CI over pairs)", legend=False, colors={"XGBoost, saved validation (legacy protocol)": nbp.MUTED, "XGBoost, fresh (eval)": XGB_COLOR, "EBM, fresh (eval)": EBM_COLOR})
ax.plot(head.bb_per_100, 1.0, "D", color=nbp.SERIES[3], markersize=8, markerfacecolor=nbp.SERIES[3] if head.ci_low > 0 or head.ci_high < 0 else nbp.SURFACE, markeredgewidth=1.8)
ax.hlines(1.0, head.ci_low, head.ci_high, color=nbp.SERIES[3], linewidth=1.8)
ax.set_yticks([0, 1], ["vs teacher", "XGBoost vs EBM"])
ax.set_ylim(1.6, -0.6)
nbp.title(ax, "The matchups that matter", "against the teacher, and against the other model (positive = XGBoost ahead)")
baselines = EVAL_OPPONENTS[1:]
series_bots = {
    "XGBoost, saved validation (legacy protocol)": ([saved_val[o]["bb_per_100"] for o in baselines], [saved_val[o]["ci95_pairs"][0] for o in baselines], [saved_val[o]["ci95_pairs"][1] for o in baselines]),
    "XGBoost, fresh (eval)": ([fresh["XGBoost", o].bb_per_100 for o in baselines], [fresh["XGBoost", o].ci_low for o in baselines], [fresh["XGBoost", o].ci_high for o in baselines]),
    "EBM, fresh (eval)": ([fresh["EBM", o].bb_per_100 for o in baselines], [fresh["EBM", o].ci_low for o in baselines], [fresh["EBM", o].ci_high for o in baselines]),
}
nbp.forest_plot(ax2, baselines, series_bots, x_label="BB/100 (duplicate deals, 95% CI over pairs)", legend=False, colors={"XGBoost, saved validation (legacy protocol)": nbp.MUTED, "XGBoost, fresh (eval)": XGB_COLOR, "EBM, fresh (eval)": EBM_COLOR})
nbp.title(ax2, "Against exploitable baselines", "huge numbers, huge intervals: a measure of how much a weakness is punished")
handles = [Line2D([0], [0], color=c, marker="o", markersize=7, linewidth=1.8, label=n) for n, c in (("XGBoost, saved validation (legacy)", nbp.MUTED), ("XGBoost, fresh", XGB_COLOR), ("EBM, fresh", EBM_COLOR))]
handles.append(Line2D([0], [0], color=nbp.MUTED, marker="o", markersize=7, markerfacecolor=nbp.SURFACE, linewidth=1.8, label="hollow: CI contains 0"))
ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=4, fontsize=8.5)
ax.legend(handles=handles[:3] + [Line2D([0], [0], color=nbp.SERIES[3], marker="D", markersize=7, linewidth=1.8, label="XGBoost vs EBM head-to-head")], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=8.5)
fig.tight_layout()
nbp.savefig("nb02_policy_eval", fig)
plt.show()
agree_saved = [
    abs(fresh["XGBoost", o].bb_per_100 - saved_val[o]["bb_per_100"]) / np.hypot(fresh["XGBoost", o].std_error, saved_val[o]["std_error"]) for o in EVAL_OPPONENTS
]
display(Markdown(
    f"**Replication check.** Fresh vs saved-validation estimate of the same quantity (|z| of the difference, using both standard errors; the saved numbers use the legacy protocol, so a difference can come from the protocol and not only from sampling): " + ", ".join(f"`{o}` {z:.1f}" for o, z in zip(EVAL_OPPONENTS, agree_saved)) +
    f" - {'all below 2' if max(agree_saved) < 2 else 'one or more above 2: a larger gap than chance alone usually produces'}. "
    f"Against the teacher XGBoost earns {result_text(fresh['XGBoost', 'teacher'])} BB/100 -> **{verdict(fresh['XGBoost', 'teacher'].ci_low, fresh['XGBoost', 'teacher'].ci_high)}**; "
    f"it wins clearly against {', '.join('`' + o + '`' for o in baselines if fresh['XGBoost', o].ci_low > 0) or 'none of the baselines'}. These baselines are exploitable by almost any sensible strategy, so those numbers measure "
    f"how heavily a weakness is punished (the maniac's variance is enormous), not the quality of the strategy. This notebook makes **no claim that either model plays better poker**; the official tournament (Notebook 4) "
    f"is the place for league standings with their intervals."
))

# %% [markdown]
# ## 5. Feature importance and SHAP analysis
#
# ### 5.1 SHAP for this model, and a check that the pieces agree `[recomputed]`
#
# For a tree ensemble the *margin* of an action (the pre-softmax score) is a sum over trees; **TreeSHAP** attributes it to the 45 features exactly:
# `margin(action) = base value(action) + sum_f phi_f(action)`, where `phi_f` is feature `f`'s Shapley value - its average marginal contribution over all orders in which the features could be "revealed". Units are the
# model's log-odds-like margin of **one action**; a positive value pushes that action up. The agent uses XGBoost's native `pred_contribs=True`; the `shap` library's `TreeExplainer` is the reference implementation behind
# all the plots below. Both are computed on the same random sample of validation states (`SHAP_ROWS`, seeded) and must agree; the additivity and the link to the probabilities are checked too.

# %%
rng = np.random.default_rng(ANALYSIS_SEED)
shap_rows = rng.choice(N_VAL, size=SHAP_ROWS, replace=False)
X_SHAP = X_VAL[shap_rows]
started = time.perf_counter()
SH = ShapSample.compute(AGENT.booster, X_SHAP)  # native TreeSHAP (what AGENT.top_features uses)
library_values = shap_library_values(AGENT.booster, X_SHAP)  # shap.TreeExplainer, same (rows, classes, features) layout
explainer = shap.TreeExplainer(AGENT.booster)
dmatrix = xgb.DMatrix(X_SHAP, feature_names=list(FEATURE_NAMES))
native_margin = AGENT.booster.predict(dmatrix, output_margin=True)
margin_softmax = np.exp(SH.margins - SH.margins.max(axis=1, keepdims=True))
margin_softmax /= margin_softmax.sum(axis=1, keepdims=True)
checks = {
    "`shap.TreeExplainer` vs native `pred_contribs` (all rows, classes and features)": float(np.abs(library_values - SH.values).max()),
    "base value + sum of the 45 SHAP values vs the model's raw margin": float(np.abs(SH.margins - native_margin).max()),
    "softmax of that margin vs `inplace_predict` probabilities (the agent's path)": float(np.abs(margin_softmax - class_probabilities(AGENT.booster, X_SHAP)).max()),
}
agent_diff = 0.0
for i in range(25):  # the agent's own per-state explanation
    chosen = AbstractAction(int(np.argmax(P_XGB[shap_rows[i]])))
    entries = {e["feature"]: e["contribution"] for e in AGENT.top_features(X_SHAP[i], chosen)}
    agent_diff = max(agent_diff, max(abs(entries[f] - SH.values[i, int(chosen), FEATURE_INDEX[f]]) for f in FEATURE_NAMES))
checks["`XGBoostAgent.top_features` vs the batch values (25 states, chosen action)"] = float(agent_diff)
display(Markdown(
    f"**[recomputed]** ({time.perf_counter() - started:.0f} s) Largest absolute differences, in margin units:\n\n" + "\n".join(f"* {k}: **{v:.1e}**" for k, v in checks.items()) +
    "\n\nThe library, the native implementation and the agent's explanation are the same numbers; additivity holds to float32 precision. Without a background data set `TreeExplainer` uses the "
    "trees' own cover statistics for the expectation (*tree-path-dependent* SHAP, as XGBoost's `pred_contribs`); section 5.7 shows what changes if the expectation is taken under independence instead."
))
assert max(checks.values()) < 1e-3

# %% [markdown]
# **One more thing to know before reading any SHAP value: the base value is *not* the average prediction.** In XGBoost the "cover" of a tree node is the sum of the *Hessians* of the training rows in it; for a
# softmax objective a row's Hessian is about `p (1 - p)`, which is close to zero for rows the model is sure about. Path-dependent TreeSHAP takes its expectation under these covers, so the base value is dominated by
# *uncertain* states, and it can sit far from the mean margin over the data:

# %%
base_table = pd.DataFrame(
    {
        "action": [f"`{a}`" for a in ACTIONS],
        "base value the SHAP values add up from (native)": SH.base_values[0],
        "`TreeExplainer.expected_value` (library attribute)": np.asarray(explainer.expected_value),
        "mean raw margin over the sampled states": native_margin.mean(axis=0),
    }
)
display(md_table(base_table, "{:+.2f}"))
display(Markdown(
    f"**[recomputed]** For `{ACTIONS[3]}` the base value is {SH.base_values[0, 3]:+.2f} but the average margin over real validation states is {native_margin[:, 3].mean():+.2f}: most states are *far below* the base value for that action, "
    f"simply because it is rarely chosen. Two consequences. (1) Read SHAP values as **differences** - between states, between feature values, between features - not as absolute pushes away from a typical state (a state with a margin "
    f"{abs(native_margin[:, 3].mean() - SH.base_values[0, 3]):.0f} units below the base value is typical, not extreme). (2) The library's `expected_value` *attribute* is computed separately and differs from the native base value by up to "
    f"{np.abs(np.asarray(explainer.expected_value) - SH.base_values[0]).max():.2f} margin units, although the SHAP *values* are identical to the native ones (first check above): additivity holds with the native base value, "
    f"which is what `pred_contribs`, `ShapSample` and every waterfall below use - never with the attribute."
))


# %% [markdown]
# ### 5.2 Three importance measures, and why they disagree `[recomputed]`
#
# * **Gain** (what the trees *split on*): each split's loss reduction, summed per feature - computed from the model's structure, no data needed (`[saved]` in `metadata.json`, recomputed here and compared).
# * **Mean |SHAP|** (what the *predictions* use): the average magnitude of a feature's contribution over validation states, summed over the five actions.
# * **Permutation importance** (what the *score* needs): shuffle one feature across states and see how much the log-loss rises (average of several shuffles, on 5,000 validation states).
#
# The three answer different questions, so they need not agree - and **collinearity makes them disagree**: `equity_minus_pot_odds` *is* `equity - pot_odds`, so the model can read the same information from any of the three.

# %%
gain = gain_importance(AGENT.booster)
assert max(abs(gain[f] - meta["global_importance_gain"][f]) for f in FEATURE_NAMES) < 1e-9
mean_abs, mean_abs_by_action = mean_abs_shap(SH)
saved_mean_abs = meta["global_importance"]
perm_rows = rng.choice(N_VAL, size=PERMUTATION_ROWS, replace=False)
XP, YP, LP = X_VAL[perm_rows], Y_VAL[perm_rows], L_VAL[perm_rows]


def score_rows(matrix: np.ndarray) -> tuple[float, float]:
    probabilities = class_probabilities(AGENT.booster, matrix)
    loss = float(-np.log(np.clip(probabilities[np.arange(len(YP)), YP], 1e-12, 1.0)).mean())
    return loss, float((masked_argmax(probabilities, LP) == YP).mean())


BASE_LOSS, BASE_ACC = score_rows(XP)


def permutation_effect(columns: list[int]) -> tuple[float, float, float]:
    """Mean rise of the log-loss, its standard deviation over the shuffles, and the mean fall of accuracy (pp) when ``columns`` are shuffled TOGETHER (same row order)."""
    d_loss, d_acc = [], []
    for _ in range(PERMUTATION_REPEATS):
        order = rng.permutation(len(XP))
        shuffled = XP.copy()
        shuffled[:, columns] = XP[order][:, columns]
        loss, acc = score_rows(shuffled)
        d_loss.append(loss - BASE_LOSS)
        d_acc.append(100 * (BASE_ACC - acc))
    return float(np.mean(d_loss)), float(np.std(d_loss, ddof=1)), float(np.mean(d_acc))


started = time.perf_counter()
perm = {f: permutation_effect([FEATURE_INDEX[f]]) for f in FEATURE_NAMES}
GROUPS = {
    "equity + equity_minus_pot_odds": ["equity", "equity_minus_pot_odds"],
    "the price of the call (pot_odds, bet_fraction_of_pot, to_call_bb, call_stack_fraction, facing_bet)": ["pot_odds", "bet_fraction_of_pot", "to_call_bb", "call_stack_fraction", "facing_bet"],
    "equity, equity_minus_pot_odds, pot_odds": ["equity", "equity_minus_pot_odds", "pot_odds"],
}
perm_groups = {name: permutation_effect([FEATURE_INDEX[f] for f in members]) for name, members in GROUPS.items()}
perm_seconds = time.perf_counter() - started

share = lambda d: {f: d[f] / sum(d.values()) for f in FEATURE_NAMES}  # noqa: E731 - normalise to a share of the total
gain_share, shap_share = share(gain), share(mean_abs)
perm_share = share({f: max(v[0], 0.0) for f, v in perm.items()})
ranking = pd.DataFrame(
    {"feature": list(FEATURE_NAMES), "group": [FEATURE_GROUP_OF[f] for f in FEATURE_NAMES], "gain": [gain_share[f] for f in FEATURE_NAMES], "shap": [shap_share[f] for f in FEATURE_NAMES], "perm": [perm_share[f] for f in FEATURE_NAMES]}
).sort_values("shap", ascending=False).reset_index(drop=True)
for column in ("gain", "shap", "perm"):
    ranking[f"rank_{column}"] = ranking[column].rank(ascending=False, method="min").astype(int)
top = ranking.head(15)
fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.6), sharey=True)
y_pos = np.arange(len(top))
for ax, column, label in zip(axes, ("gain", "shap", "perm"), ("share of total split gain", "share of mean |SHAP| (all five actions)", "share of permutation log-loss increase")):
    ax.barh(y_pos, 100 * top[column], color=[GROUP_COLORS[g] for g in top["group"]], height=nbp.BAR_THICKNESS)
    for yv, v, rk in zip(y_pos, top[column], top[f"rank_{column}"]):
        ax.annotate(f"{100 * v:.1f}%  (#{rk})", (100 * v, yv), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=nbp.MUTED)
    ax.set_xlabel(label + " (%)")
    ax.set_xlim(0, 100 * max(top[c].max() for c in ("gain", "shap", "perm")) * 1.32)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y_pos, top["feature"])
axes[0].invert_yaxis()
nbp.title(axes[0], "Gain", "what the trees split on")
nbp.title(axes[1], "Mean |SHAP|", "what the predictions use")
nbp.title(axes[2], "Permutation", "what the score needs")
handles = [Patch(color=GROUP_COLORS[g]) for g in dict.fromkeys(top["group"])]
axes[2].legend(handles, [g.replace("_", " ") for g in dict.fromkeys(top["group"])], loc="lower right", title="feature group", fontsize=8)
fig.suptitle("Three importance measures, same model (top 15 features by mean |SHAP|; bars labelled with value and rank)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=nbp.TEXT)
fig.tight_layout(rect=(0, 0, 1, 0.95))
nbp.savefig("nb02_importance", fig)
plt.show()

rank_corr = {}
for a, b in (("gain", "shap"), ("gain", "perm"), ("shap", "perm")):
    rank_corr[a, b] = (stats.spearmanr(ranking[a], ranking[b]).statistic, stats.kendalltau(ranking[a], ranking[b]).statistic)
saved_rank = stats.spearmanr([saved_mean_abs[f] for f in FEATURE_NAMES], [mean_abs[f] for f in FEATURE_NAMES]).statistic
head12 = ranking.head(12)
display(md_table(pd.DataFrame(
    {
        "feature": [f"`{f}`" for f in head12["feature"]], "group": head12["group"],
        "gain": head12["gain"].map(pct), "# gain": head12["rank_gain"],
        "mean abs SHAP": head12["shap"].map(pct), "# SHAP": head12["rank_shap"],
        "permutation": head12["perm"].map(pct), "# perm.": head12["rank_perm"],
    }
)))
display(Markdown(
    f"**[recomputed]** ({perm_seconds:.0f} s for {len(FEATURE_NAMES)} features x {PERMUTATION_REPEATS} shuffles.) Rank agreement over all 45 features (Spearman / Kendall tau): gain vs SHAP {rank_corr['gain', 'shap'][0]:.2f} / {rank_corr['gain', 'shap'][1]:.2f}, "
    f"gain vs permutation {rank_corr['gain', 'perm'][0]:.2f} / {rank_corr['gain', 'perm'][1]:.2f}, SHAP vs permutation {rank_corr['shap', 'perm'][0]:.2f} / {rank_corr['shap', 'perm'][1]:.2f}. "
    f"(The 3,000-state SHAP importances recomputed here have Spearman {saved_rank:.3f} with the 5,000-state ones saved in `metadata.json`.) "
    f"The leaders agree (the SHAP top two, `{ranking.loc[0, 'feature']}` and `{ranking.loc[1, 'feature']}`, are in the top three of {sum(set(ranking.head(2)['feature']) <= set(ranking.sort_values(c).head(3)['feature']) for c in ('rank_gain', 'rank_shap', 'rank_perm'))} of the 3 rankings) - the order within the leaders and the tail do not."
))

# %% [markdown]
# **Collinearity caveat, made visible.** Permuting `equity` *alone* leaves `equity_minus_pot_odds` intact (the model can read the same quantity from it), and vice versa - so each looks less important
# than it is. Shuffling a *set* of redundant features **together** (the same row order for all of them, which keeps their mutual relations) measures the information they carry jointly:

# %%
rows = []
single = {f: perm[f] for f in ("equity", "equity_minus_pot_odds", "pot_odds")}
for f, (d_loss, sd, d_acc) in single.items():
    rows.append({"features shuffled": f"`{f}` alone", "log-loss increase": d_loss, "+/- (sd over shuffles)": sd, "accuracy loss (pp)": d_acc})
for name, (d_loss, sd, d_acc) in perm_groups.items():
    rows.append({"features shuffled": f"**together:** {name}", "log-loss increase": d_loss, "+/- (sd over shuffles)": sd, "accuracy loss (pp)": d_acc})
display(md_table(pd.DataFrame(rows), "{:.4f}"))
pair = perm_groups["equity + equity_minus_pot_odds"]
display(Markdown(
    f"**[recomputed]** Shuffled alone, `equity` raises the log-loss by {perm['equity'][0]:.3f} and `equity_minus_pot_odds` by {perm['equity_minus_pot_odds'][0]:.3f} (sum {perm['equity'][0] + perm['equity_minus_pot_odds'][0]:.3f}); shuffled together the rise is "
    f"{pair[0]:.3f} - {pair[0] / (perm['equity'][0] + perm['equity_minus_pot_odds'][0]):.1f}x the sum of the parts. The pair carries far more than either member's permutation importance suggests; "
    f"gain and SHAP spread the same credit over the members in a different way (`equity_minus_pot_odds` is #{int(ranking.set_index('feature').loc['equity_minus_pot_odds', 'rank_gain'])} by gain, "
    f"`equity` #{int(ranking.set_index('feature').loc['equity', 'rank_gain'])}; by SHAP the order is {'reversed' if ranking.set_index('feature').loc['equity', 'rank_shap'] < ranking.set_index('feature').loc['equity_minus_pot_odds', 'rank_shap'] else 'the same'}). "
    f"*No importance number is a property of the poker situation: it is a property of this fitted model and of how credit between redundant inputs happened to be shared.*"
))

# %% [markdown]
# A **leave-one-feature-out refit** shows the same thing from the other side (`[live]`, toy models of section 3.5: 20,000 rows, depth 8). If the credit were intrinsic, removing `equity_minus_pot_odds`
# should hurt; instead its credit moves to the features it is made of.

# %%
def toy_booster(columns: list[int]) -> xgb.Booster:
    names_ = [FEATURE_NAMES[c] for c in columns]
    dtrain_ = xgb.QuantileDMatrix(toy_train.X[:, columns], toy_train.y, weight=toy_weights[toy_train.y], feature_names=names_, max_bin=booster["max_bin"])
    dval_ = xgb.QuantileDMatrix(toy_val.X[:, columns], toy_val.y, weight=toy_weights[toy_val.y], ref=dtrain_, feature_names=names_, max_bin=booster["max_bin"])
    fitted_ = xgb.train(toy_params, dtrain_, num_boost_round=LIVE_FIT_MAX_ROUNDS, evals=[(dval_, "val")], early_stopping_rounds=run_cfg["early_stopping_rounds"], verbose_eval=False)
    return fitted_[: fitted_.best_iteration + 1]


def toy_mean_abs(model: xgb.Booster, columns: list[int]) -> dict[str, float]:
    names_ = [FEATURE_NAMES[c] for c in columns]
    contributions = model.predict(xgb.DMatrix(X_SHAP[:, columns], feature_names=names_), pred_contribs=True)
    return dict(zip(names_, np.abs(contributions[:, :, :-1]).mean(axis=0).sum(axis=0)))


all_columns = list(range(NUM_FEATURES))
without = [c for c in all_columns if FEATURE_NAMES[c] != "equity_minus_pot_odds"]
started = time.perf_counter()
toy_all, toy_without = toy_booster(all_columns), toy_booster(without)


def toy_accuracy(model: xgb.Booster, columns: list[int]) -> float:
    proba = model.inplace_predict(np.ascontiguousarray(X_VAL[:, columns], dtype=np.float32), validate_features=False)
    return float((masked_argmax(proba, L_VAL) == Y_VAL).mean())


acc_all, acc_without = toy_accuracy(toy_all, all_columns), toy_accuracy(toy_without, without)
imp_all, imp_without = toy_mean_abs(toy_all, all_columns), toy_mean_abs(toy_without, without)
display(md_table(pd.DataFrame(
    {
        "feature": [f"`{f}`" for f in ("equity", "equity_minus_pot_odds", "pot_odds", "facing_bet", "street")],
        "mean abs SHAP, all 45 features": [imp_all[f] for f in ("equity", "equity_minus_pot_odds", "pot_odds", "facing_bet", "street")],
        "mean abs SHAP, without equity_minus_pot_odds": [f"{imp_without[f]:.2f}" if f in imp_without else "(removed)" for f in ("equity", "equity_minus_pot_odds", "pot_odds", "facing_bet", "street")],
    }
), "{:.2f}"))
display(Markdown(
    f"**[live]** ({time.perf_counter() - started:.0f} s) Masked validation accuracy of the toy model with all features {pct(acc_all, 2)}, without `equity_minus_pot_odds` {pct(acc_without, 2)} (a difference of {100 * (acc_without - acc_all):+.2f} pp - the information is still there, in `equity` and `pot_odds`); "
    f"`equity`'s mean |SHAP| goes from {imp_all['equity']:.1f} to {imp_without['equity']:.1f}, `pot_odds`' from {imp_all['pot_odds']:.1f} to {imp_without['pot_odds']:.1f}. One toy fit each: the *direction* is the point, not the exact numbers."
))

# %% [markdown]
# ### 5.3 What drives each action: beeswarm summary plots `[recomputed]`
#
# Each dot is one validation state (3,000 of them); the horizontal position is the SHAP value of that feature **for this action** (right = the feature value pushes the action's score up), the colour is the feature's value
# (blue low, red high). A wide, one-sided colour pattern means "high values of this feature favour this action".

# %%
EXPL = {a: shap.Explanation(values=SH.for_action(a), base_values=SH.base_values[:, i], data=X_SHAP, feature_names=list(FEATURE_NAMES)) for i, a in enumerate(ACTIONS)}
fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.4))
for ax, a in zip(axes.flat, ACTIONS):
    shap.plots.beeswarm(EXPL[a], max_display=10, ax=ax, show=False, color_bar=(a in ("raise_small", "all_in")), plot_size=None, s=9, group_remaining_features=True)
    ax.set_title(f"{a}  (mean |SHAP| of the top feature: {np.abs(EXPL[a].values).mean(axis=0).max():.1f})", loc="left", color=nbp.TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("SHAP value (margin units of this action)")
heat_ax = axes.flat[5]
heat_ax.axis("off")
heat_ax.text(0.0, 0.95, "How to read these plots", transform=heat_ax.transAxes, color=nbp.TEXT, fontsize=12, fontweight="bold", va="top")
heat_ax.text(
    0.0, 0.86,
    "One dot = one validation state.\nx: the feature's contribution to the margin of THIS action\n    (right = pushes the action up, left = pushes it down).\ncolour: the feature's value (blue low, red high).\n\n"
    "Features are ordered by mean |SHAP| for that action,\nso the order differs from panel to panel.\n\nThe last row of each panel sums the remaining features.\nSHAP explains the model - not poker.",
    transform=heat_ax.transAxes, color=nbp.MUTED, fontsize=10, va="top", linespacing=1.5,
)  # fmt: skip
fig.tight_layout()
retint_axes(fig)
nbp.savefig("nb02_shap_beeswarm", fig)
plt.show()
top_by_action = {a: [f for f, _ in list(mean_abs_by_action[a].items())[:3]] for a in ACTIONS}
display(Markdown(
    "**[recomputed]** Top three features by mean |SHAP| for each action: " + "; ".join(f"**{a}**: " + ", ".join(f"`{f}`" for f in top_by_action[a]) for a in ACTIONS) + "."
))

# %% [markdown]
# ### 5.4 Per-action effects: the same feature does different jobs `[recomputed]`
#
# Summing over actions hides the structure. The table shows the mean |SHAP| of the 14 most influential features **per action** - for instance `spr` (stack-to-pot ratio) is almost exclusively
# about the all-in decision, while `opp_aggression` is about folding.

# %%
mean_abs_table = SH.mean_abs()[list(ranking["feature"].head(14))].T  # features x actions
fig, ax = plt.subplots(figsize=(9.2, 6.2))
image = ax.imshow(mean_abs_table.to_numpy(), cmap=nbp.SEQUENTIAL, aspect="auto")
for i in range(mean_abs_table.shape[0]):
    for j in range(mean_abs_table.shape[1]):
        v = mean_abs_table.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#06110c" if v > mean_abs_table.to_numpy().max() * 0.55 else nbp.TEXT)
ax.set_xticks(range(5), ACTIONS)
ax.set_yticks(range(len(mean_abs_table)), mean_abs_table.index)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
colorbar.outline.set_visible(False)
colorbar.set_label("mean |SHAP| (margin units)", color=nbp.MUTED)
nbp.title(ax, "Which feature moves which action?", "mean |SHAP| over 3,000 validation states, per action")
fig.tight_layout()
nbp.savefig("nb02_shap_by_action", fig)
plt.show()
focus = {}
for f in ("spr", "opp_aggression", "equity", "street", "num_raises_street"):
    row = SH.mean_abs()[f]
    focus[f] = (row.idxmax(), row.max() / row.sum())
display(Markdown(
    "**[recomputed]** Where each feature's attribution goes (share of its total mean |SHAP| in its largest action): " +
    ", ".join(f"`{f}` -> **{a}** ({pct(s, 0)})" for f, (a, s) in focus.items()) +
    ". The sum over actions that `gain` and the saved global importance report is therefore an average of very different jobs."
))

# %% [markdown]
# ### 5.5 Interaction values `[recomputed]`
#
# A tree model is not additive, and SHAP can say *how much*: the **SHAP interaction values** split every prediction into main effects (diagonal) and pairwise interactions (off-diagonal), still summing exactly to the margin.
# They are expensive (a 46 x 46 matrix per state and action), so they are computed on a subsample only (`INTERACTION_ROWS` states).

# %%
X_INT = X_SHAP[:INTERACTION_ROWS]
started = time.perf_counter()
INT = np.moveaxis(np.asarray(explainer.shap_interaction_values(X_INT)), -1, 1)  # (rows, actions, features, features)
int_seconds = time.perf_counter() - started
int_total = np.abs(INT).mean(axis=0).sum(axis=0)  # features x features, summed over the five actions
off_diagonal_mass = (int_total.sum() - np.trace(int_total)) / int_total.sum()
margin_check = float(np.abs(INT.sum(axis=(2, 3)) + SH.base_values[:INTERACTION_ROWS] - SH.margins[:INTERACTION_ROWS]).max())
order_features = list(ranking["feature"].head(12))
index_top = [FEATURE_INDEX[f] for f in order_features]
sub = int_total[np.ix_(index_top, index_top)]
fig, ax = plt.subplots(figsize=(9.0, 7.0))
image = ax.imshow(np.where(np.eye(len(sub), dtype=bool), np.nan, sub), cmap=nbp.SEQUENTIAL, aspect="equal")
for i in range(len(sub)):
    for j in range(len(sub)):
        if i == j:
            ax.text(j, i, f"main\n{sub[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color=nbp.MUTED)
        else:
            ax.text(j, i, f"{sub[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#06110c" if sub[i, j] > np.nanmax(np.where(np.eye(len(sub), dtype=bool), np.nan, sub)) * 0.55 else nbp.TEXT)
ax.set_xticks(range(len(sub)), order_features, rotation=45, ha="right")
ax.set_yticks(range(len(sub)), order_features)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
nbp.title(ax, "SHAP interaction values (mean |value|, summed over actions)", f"{INTERACTION_ROWS} validation states; diagonal = main effects (grey), off-diagonal cell = half of the pair's total effect")
fig.tight_layout()
nbp.savefig("nb02_shap_interactions", fig)
plt.show()
pair_list = sorted(((FEATURE_NAMES[i], FEATURE_NAMES[j], 2 * int_total[i, j]) for i in range(NUM_FEATURES) for j in range(i + 1, NUM_FEATURES)), key=lambda t: -t[2])
display(md_table(pd.DataFrame({"pair": [f"`{a}` x `{b}`" for a, b, _ in pair_list[:8]], "mean abs interaction (both triangles)": [v for *_, v in pair_list[:8]]}), "{:.2f}"))
display(Markdown(
    f"**[recomputed]** ({int_seconds:.0f} s) Additivity check: main effects + interactions + base value reproduce the margin to {margin_check:.1e}. **{pct(off_diagonal_mass, 0)} of the total attribution mass sits off the diagonal**, i.e. in interactions: "
    f"the model is far from additive. The largest pairs are {', '.join(f'`{a}` x `{b}`' for a, b, _ in pair_list[:3])}. The first of these is partly an artefact of redundancy: `equity_minus_pot_odds` is *defined* from `equity` and `pot_odds`, "
    f"and in a state with nothing to call it equals `equity` exactly, so credit moves between the two columns (section 5.2). The pairs with `street`, `spr`, `opp_aggression`, `opp_fold_to_bet` and `num_raises_street` are the genuine structure: "
    f"the teacher's thresholds on equity move with the street, the stack depth, the opponent's statistics and the raise count."
))

# %% [markdown]
# ### 5.6 Dependence plots with interaction colouring `[recomputed]`
#
# A dependence plot shows one feature's SHAP value against the feature's value, **for the action where that feature matters most**; the colour is the feature with the strongest SHAP interaction with it
# (chosen from the interaction values above, not by eye). Vertical spread at one x position is the effect of *other* features - the visible footprint of non-additivity. In the equity panel the dotted lines are
# teacher thresholds on equity (`TeacherParams`), drawn only for orientation.

# %%
panels = []
for f in ("equity", "equity_minus_pot_odds", "spr", "opp_aggression"):
    best_action = SH.mean_abs()[f].idxmax()
    c = ACTIONS.index(best_action)
    partner_scores = np.abs(INT[:, c]).mean(axis=0)[FEATURE_INDEX[f]].copy()
    partner_scores[FEATURE_INDEX[f]] = -1
    panels.append((f, best_action, FEATURE_NAMES[int(np.argmax(partner_scores))]))
fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.2))
for ax, (f, a, partner) in zip(axes.flat, panels):
    shap.dependence_plot(f, SH.for_action(a), X_SHAP, feature_names=list(FEATURE_NAMES), interaction_index=partner, ax=ax, show=False, dot_size=9)
    ax.set_title(f"{f} -> {a}   (coloured by {partner})", loc="left", color=nbp.TEXT, fontsize=11, fontweight="bold")
    if f == "equity":
        for name, value in (("c-bet", PARAMS.cbet_min_equity), ("value", PARAMS.value_equity), ("wet pot", PARAMS.wet_pot_equity), ("pot", PARAMS.pot_bet_equity), ("monster", PARAMS.monster_equity)):
            ax.axvline(value, color=nbp.MUTED, linestyle=":", linewidth=1.1)
            ax.text(value, ax.get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=7.5, color=nbp.MUTED)
fig.tight_layout()
retint_axes(fig)
nbp.savefig("nb02_shap_dependence", fig)
plt.show()
display(Markdown(
    "**[recomputed]** Panels (feature -> action, colour partner): " + "; ".join(f"`{f}` -> {a}, coloured by `{p}`" for f, a, p in panels) +
    ". Reading guide: a step in the curve is a threshold the model has learned; a vertical band of colours at one x is another feature changing the effect there. "
    "In the equity panel the steps cluster at the teacher's lines (section 8 quantifies this), and below the value line the lowest band (blue = preflop) is separated from the post-flop bands: the same equity means something different before the flop."
))

# %% [markdown]
# ### 5.7 SHAP caveats, demonstrated `[recomputed]`
#
# 1. **Attributions depend on how the "missing" features are replaced.** The default TreeSHAP above takes the expectation under the trees' own cover (*path-dependent*): correlated features share credit as the data's
#    correlations dictate. The **interventional** variant takes it under independence over a *background sample*, which is a different question. The same model, the same states, two answers:

# %%
bg_rng = np.random.default_rng(ANALYSIS_SEED + 1)
background = X_VAL[bg_rng.choice(N_VAL, size=INTERVENTIONAL_BACKGROUND, replace=False)]
explained = X_SHAP[:INTERVENTIONAL_ROWS]
started = time.perf_counter()
with redirect_stderr(io.StringIO()):  # the library prints a progress bar
    interventional = np.asarray(shap.TreeExplainer(AGENT.booster, data=background, feature_perturbation="interventional").shap_values(explained, check_additivity=False))
interventional = np.moveaxis(interventional, -1, 1)
int_seconds2 = time.perf_counter() - started
path_dependent = np.abs(SH.values[:INTERVENTIONAL_ROWS]).mean(axis=0).sum(axis=0)
independent = np.abs(interventional).mean(axis=0).sum(axis=0)
rho_variants = stats.spearmanr(path_dependent, independent).statistic
variant_table = pd.DataFrame(
    {
        "feature": [f"`{FEATURE_NAMES[j]}`" for j in np.argsort(-path_dependent)[:8]],
        "path-dependent": [path_dependent[j] for j in np.argsort(-path_dependent)[:8]],
        "interventional": [independent[j] for j in np.argsort(-path_dependent)[:8]],
        "ratio": [independent[j] / path_dependent[j] for j in np.argsort(-path_dependent)[:8]],
    }
)
display(md_table(variant_table, "{:.2f}"))
changes = sorted((j for j in range(NUM_FEATURES) if path_dependent[j] > 0.3), key=lambda j: -abs(np.log(independent[j] / path_dependent[j])))[:3]
display(Markdown(
    f"**[recomputed]** ({int_seconds2:.0f} s; {INTERVENTIONAL_ROWS} explained states x {INTERVENTIONAL_BACKGROUND} background states, summed mean |SHAP| over the five actions.) The rankings are similar (Spearman {rho_variants:.2f}) but the magnitudes are not: "
    f"the largest changes are " + ", ".join(f"`{FEATURE_NAMES[j]}` x{independent[j] / path_dependent[j]:.2f}" for j in changes) + f" (and `equity` x{independent[FEATURE_INDEX['equity']] / path_dependent[FEATURE_INDEX['equity']]:.2f}). "
    f"Neither is 'the' importance; they answer different questions, and a small background sample adds noise of its own."
))

# %% [markdown]
# 2. **Additive attribution of a non-additive model.** SHAP forces an additive story onto interactions (an interaction's effect is split between its two features); section 5.5 shows that about half of the mass is interaction.
# 3. **Explanation is not causation, and not poker.** A SHAP value says what *this model* does with a feature relative to an average state. It does not say that changing the feature would change the hand's value, and the model is a copy of a rule set, not of reality.
# 4. **Sampling.** All SHAP summaries use a random 3,000 of the 73,777 validation states (the saved importances use 5,000); they agree closely, but they are samples.

# %% [markdown]
# ## 6. Example poker decisions and explanation fidelity
#
# ### 6.1 Which features does the teacher's rule actually use? (ground truth for checking explanations)
#
# Because the teacher is a transparent function, every decision comes with a *rule trace*, and we can ask which features that rule **needs**. Two sets, both computed from the teacher itself:
#
# * **R_read** - every feature the teacher's code *reads* on the path to this decision (a logging dictionary is passed to the policy's two entry points; asserted below to agree with the public `decide_features`).
#   A superset: the code also reads values for the explanation text (e.g. `equity` in a preflop rule that only compares the starting-hand percentile).
# * **R_cf** (the strict one) - the features that can **flip** this decision: for each feature the rule reads, set it alone to the values of its validation distribution (the deciles; both values for flags) and check whether the teacher's
#   action changes. A feature is in R_cf only if some counterfactual value changes the decision.

# %%
FEATURE_GRID = [np.array([0.0, 1.0]) if n in BINARY_FEATURES else np.unique(np.quantile(X_VAL[:, j], np.linspace(0, 1, 11))) for j, n in enumerate(FEATURE_NAMES)]


class ReadLog(dict):
    """A feature dictionary that remembers which keys the teacher looked at."""

    def __init__(self, *args):
        super().__init__(*args)
        self.read: set[str] = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def teacher_reads(x: np.ndarray, legal: np.ndarray):
    """The teacher's decision and R_read for one state (uses the policy's internal pre-/post-flop entry points)."""
    log = ReadLog(zip(FEATURE_NAMES, np.asarray(x, dtype=np.float32).tolist()))
    legal_tuple = tuple(bool(v) for v in legal)
    decision = TEACHER._preflop(log, legal_tuple) if log["street"] < 0.5 else TEACHER._postflop(log, legal_tuple)
    return decision, set(log.read)


def decisive_features(x: np.ndarray, legal: np.ndarray, reads: set[str], action: int) -> set[str]:
    """R_cf: the features among ``reads`` whose replacement by some value of their distribution changes the teacher's action."""
    probe = np.asarray(x, dtype=np.float32).copy()
    flips = set()
    for name in reads:
        j = FEATURE_INDEX[name]
        original = probe[j]
        for value in FEATURE_GRID[j]:
            if value == original:
                continue
            probe[j] = value
            if int(TEACHER.decide_features(probe, legal).action) != action:
                flips.add(name)
                break
        probe[j] = original
    return flips


for i in range(0, N_VAL, 37):  # the logged path equals the public API (every 37th validation state)
    decision, _ = teacher_reads(X_VAL[i], L_VAL[i])
    public = TEACHER.decide_features(X_VAL[i], L_VAL[i])
    assert decision.action == public.action and decision.rule == public.rule == RULE_VAL[i] and int(decision.action) == Y_VAL[i]
display(Markdown(f"**[recomputed]** Logged teacher path = public `decide_features` = stored label and rule on {len(range(0, N_VAL, 37)):,} validation states."))

# %% [markdown]
# ### 6.2 Eight real decisions, explained `[recomputed]`
#
# `select_example_cases()` (`poker_arena.explain.notebook_cases`) draws a fixed, seeded set of **real validation-set decisions** - the same eight that Notebook 1 explains with the EBM, so the two explanations can be read side by side.
# The situation is described from the features alone (the dataset stores no hidden cards). For each case: the teacher's action and the rule trace that produced it, both models' probabilities (masked to the legal actions), and the
# **SHAP waterfall of the action XGBoost chose** (`shap.plots.waterfall`, native TreeSHAP values, base value = the native base value, see 5.1: read the bars as differences, not as pushes from a typical state).

# %%
CASES = select_example_cases()
case_summary = []


def probabilities_of(agent, case) -> np.ndarray:
    return mask_and_normalise(agent.predict_proba(case.features), case.legal_mask)


for number, case in enumerate(CASES, start=1):
    p_x, p_e = probabilities_of(AGENT, case), probabilities_of(EBM, case)
    a_x, a_e = int(np.argmax(p_x)), int(np.argmax(p_e))
    reads = teacher_reads(case.features, case.legal_mask)[1]
    decisive = decisive_features(case.features, case.legal_mask, reads, case.teacher_action)
    contributions = shap_contributions(AGENT.booster, case.features)[0]  # (actions, features + bias)
    order = np.argsort(-np.abs(contributions[a_x, :NUM_FEATURES]), kind="stable")
    top3 = [FEATURE_NAMES[j] for j in order[:3]]
    numbers = {k: v for k, v in case.teacher_numbers.items() if k in ("equity", "adj_equity", "required_equity", "pot_odds", "threshold", "pct", "penalty", "fold_to_bet", "spr")}
    display(Markdown(
        f"#### Case {number}: {case.title}  (`{case.key}`)\n\n*{case.why}*\n\n**Situation.** {case.description}\n\n"
        f"| | action | probability of the chosen action | note |\n|---|---|---|---|\n"
        f"| **teacher** (label) | `{case.teacher_action_name}` | - | rule `{' > '.join(case.teacher_rules)}`, margin {case.teacher_margin:.3f}; {', '.join(f'{k} {v:.2f}' for k, v in numbers.items())} |\n"
        f"| **XGBoost** | `{ACTIONS[a_x]}` | {p_x[a_x]:.3f} | {'agrees with the teacher' if a_x == case.teacher_action else 'DISAGREES with the teacher'} |\n"
        f"| **EBM** | `{ACTIONS[a_e]}` | {p_e[a_e]:.3f} | {'agrees with the teacher' if a_e == case.teacher_action else 'DISAGREES with the teacher'} |\n\n"
        f"*Teacher's own words:* {case.teacher_summary}\n\n"
        f"**Features that can flip the teacher's decision (R_cf):** {', '.join(f'`{f}`' for f in sorted(decisive)) or 'none'}; **XGBoost's three largest SHAP contributions to `{ACTIONS[a_x]}`:** "
        + ", ".join(f"`{FEATURE_NAMES[j]}` = {case.features[j]:.3g} ({contributions[a_x, j]:+.2f})" for j in order[:3]) + "."
    ))
    explanation = shap.Explanation(values=contributions[a_x, :NUM_FEATURES], base_values=contributions[a_x, NUM_FEATURES], data=case.features, feature_names=list(FEATURE_NAMES))
    shap.plots.waterfall(explanation, max_display=9, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Case {number} ({case.key}): why XGBoost plays {ACTIONS[a_x]} (p = {p_x[a_x]:.2f}; teacher: {case.teacher_action_name} by {case.teacher_rule})", x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=nbp.TEXT, y=1.0)
    retint_axes(fig)
    nbp.savefig(f"nb02_case_{number}_{case.key}", fig)
    plt.show()
    case_summary.append((number, case, a_x, a_e, p_x, p_e, reads, decisive, top3))

# %% [markdown]
# ### 6.3 Explanation-fidelity study `[recomputed]`
#
# **Question.** For many decisions, does the feature that SHAP ranks first belong to the features that the teacher's *firing rule* needs (R_cf)? And does an EBM's own term list do better?
# Both models are asked about the same states and the same action: validation states on which XGBoost, the EBM **and** the teacher all choose the same action, so that the teacher's rule explains what each model did
# (`FIDELITY_STATES` random such states; `XGBoostAgent.top_features` and `EBMAgent.top_features` are the explanations under test; EBM interaction terms are split equally between their two features).
#
# **Why raw hit rates mislead - and what we do about it.**
#
# 1. *Almost every rule reads `equity`.* A model that always answers "equity" would score high; we report that constant answer and a **null** that keeps each model's explanations but gives each state the rule set of a *random other* state (pairing destroyed).
# 2. *Redundant features share credit.* `equity_minus_pot_odds` is `equity - pot_odds` and equals `equity` when there is nothing to call; before the flop `equity` is a monotone function of `preflop_percentile`. Crediting a twin of a feature the rule reads is not an error of the model,
#    so we also report a **family-aware** score in which a feature counts as a hit if it belongs to the same *redundancy family* as a feature in R_cf (the families and the numbers that justify them are in the table below).
# 3. *The literal score is still shown.* Nothing is hidden: literal (strict R_cf), literal with R_read (lenient), and family-aware are all in the results.
#
# (Notebook 1 runs a related study of the EBM with its own four-class scheme, a contrastive variant and 20,000 states; the two studies use different definitions, so their hit rates are not interchangeable - the comparison *between* the two models is the one made here, on identical states and definitions.)

# %%
FAMILIES = {
    "hand strength": ["equity", "equity_minus_pot_odds", "preflop_percentile"],
    "price of the call": ["facing_bet", "to_call_bb", "pot_odds", "bet_fraction_of_pot", "call_stack_fraction", "equity_minus_pot_odds"],
    "stack depth": ["effective_stack_bb", "opp_stack_bb", "my_stack_bb", "spr", "my_commit_frac", "opp_commit_frac"],
    "raise count": ["num_raises_street", "num_raises_hand", "opp_raised_this_street", "i_raised_this_street"],
}
FAMILY_MATRIX = np.zeros((NUM_FEATURES, len(FAMILIES)), dtype=bool)
for k, members in enumerate(FAMILIES.values()):
    for name in members:
        FAMILY_MATRIX[FEATURE_INDEX[name], k] = True
V = VAL[list(FEATURE_NAMES)].astype(float)
pre_mask, post_mask = STREET_VAL == 0, STREET_VAL > 0
spearman = lambda a, b, mask=slice(None): stats.spearmanr(V[a][mask], V[b][mask]).statistic  # noqa: E731
evidence = [
    ("hand strength", "Spearman(`equity`, `preflop_percentile`), preflop states", spearman("equity", "preflop_percentile", pre_mask)),
    ("hand strength", "max abs(`equity_minus_pot_odds` - (`equity` - `pot_odds`))", float(np.abs(V["equity_minus_pot_odds"] - (V["equity"] - V["pot_odds"])).max())),
    ("hand strength", "share of no-bet states where `equity_minus_pot_odds` == `equity`", float((V["equity_minus_pot_odds"] == V["equity"])[V["facing_bet"] == 0].mean())),
    ("price of the call", "share of states where `facing_bet` == (`to_call_bb` > 0)", float((V["facing_bet"] == (V["to_call_bb"] > 0)).mean())),
    ("price of the call", "Spearman(`pot_odds`, `bet_fraction_of_pot`)", spearman("pot_odds", "bet_fraction_of_pot")),
    ("price of the call", "Spearman(`to_call_bb`, `call_stack_fraction`)", spearman("to_call_bb", "call_stack_fraction")),
    ("stack depth", "Spearman(`spr`, `opp_commit_frac`), post-flop", spearman("spr", "opp_commit_frac", post_mask)),
    ("stack depth", "Spearman(`effective_stack_bb`, `opp_stack_bb`)", spearman("effective_stack_bb", "opp_stack_bb")),
    ("raise count", "Spearman(`num_raises_street`, `opp_raised_this_street`), post-flop", spearman("num_raises_street", "opp_raised_this_street", post_mask)),
    ("raise count", "Spearman(`num_raises_street`, `num_raises_hand`), preflop", spearman("num_raises_street", "num_raises_hand", pre_mask)),
]
display(Markdown("**Redundancy families** (a feature may belong to two: `equity_minus_pot_odds` is built from a member of each) and the validation-set evidence that their members are interchangeable:"))
display(md_table(pd.DataFrame({"family": [e[0] for e in evidence], "evidence": [e[1] for e in evidence], "value": [e[2] for e in evidence]}), "{:.4f}"))
display(Markdown("; ".join(f"**{k}**: " + ", ".join(f"`{m}`" for m in v) for k, v in FAMILIES.items())))

# %%
def ebm_feature_attributions(entries: list[dict]) -> np.ndarray:
    """EBM term contributions (main effects and interactions) -> one value per feature; an interaction's contribution is split equally."""
    phi = np.zeros(NUM_FEATURES)
    for entry in entries:
        parts = entry["feature"].split(" x ")
        for part in parts:
            phi[FEATURE_INDEX[part]] += entry["contribution"] / len(parts)
    return phi


agreeing = np.flatnonzero((PRED_XGB == Y_VAL) & (PRED_EBM == Y_VAL))
fidelity_rng = np.random.default_rng(ANALYSIS_SEED + 2)
fid_idx = np.sort(fidelity_rng.choice(agreeing, size=FIDELITY_STATES, replace=False))
started = time.perf_counter()
PHI_X, PHI_E = np.zeros((len(fid_idx), NUM_FEATURES)), np.zeros((len(fid_idx), NUM_FEATURES))
R_CF, R_READ = np.zeros((len(fid_idx), NUM_FEATURES), dtype=bool), np.zeros((len(fid_idx), NUM_FEATURES), dtype=bool)
fid_rules = []
for k, i in enumerate(fid_idx):
    decision, reads = teacher_reads(X_VAL[i], L_VAL[i])
    action = AbstractAction(int(decision.action))
    for e in AGENT.top_features(X_VAL[i], action):
        PHI_X[k, FEATURE_INDEX[e["feature"]]] = e["contribution"]
    PHI_E[k] = ebm_feature_attributions(EBM.top_features(X_VAL[i], action))
    R_READ[k, [FEATURE_INDEX[f] for f in reads]] = True
    R_CF[k, [FEATURE_INDEX[f] for f in decisive_features(X_VAL[i], L_VAL[i], reads, int(decision.action))]] = True
    fid_rules.append(decision.rule)
fid_rules = np.array(fid_rules)
fid_seconds = time.perf_counter() - started


def covered(R: np.ndarray) -> np.ndarray:
    """Features that count as a hit for a state: members of R, plus every member of a family that R touches."""
    touched = (R.astype(int) @ FAMILY_MATRIX.astype(int)) > 0
    return R | ((touched.astype(int) @ FAMILY_MATRIX.T.astype(int)) > 0)


COVER = {"literal (strict, R_cf)": R_CF, "literal, lenient (R_read)": R_READ, "family-aware (R_cf)": covered(R_CF)}
ROWS = np.arange(len(fid_idx))
null_rng = np.random.default_rng(ANALYSIS_SEED + 3)
NULL_PERMUTATIONS = [null_rng.permutation(len(fid_idx)) for _ in range(50)]
HAND_STRENGTH = [FEATURE_INDEX[f] for f in FAMILIES["hand strength"]]


def fidelity_scores(phi: np.ndarray, cover: np.ndarray, drop: list[int] | None = None) -> dict[str, np.ndarray]:
    """Per-state top-1 hit, top-3 hit and share of |SHAP| mass on the covered features (optionally ignoring the features in ``drop``)."""
    magnitude = np.abs(phi).copy()
    cover = cover.copy()
    if drop:
        magnitude[:, drop] = 0.0
        cover[:, drop] = False
    order = np.argsort(-magnitude, axis=1, kind="stable")
    return {
        "top1": cover[ROWS, order[:, 0]].astype(float),
        "top3": cover[ROWS[:, None], order[:, :3]].any(axis=1).astype(float),
        "mass": (magnitude * cover).sum(axis=1) / np.maximum(magnitude.sum(axis=1), 1e-12),
        "eligible": cover.any(axis=1).astype(float),
    }


def null_per_state(phi: np.ndarray, cover: np.ndarray, key: str, drop=None) -> np.ndarray:
    """Expected per-state score if every state were judged against the rule set of a random other state (50 permutations)."""
    return np.mean([fidelity_scores(phi, cover[p], drop)[key] for p in NULL_PERMUTATIONS], axis=0)


def boot_ci(values: np.ndarray, seed: int = ANALYSIS_SEED + 4) -> tuple[float, float, float]:
    idx = np.random.default_rng(seed).integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    return float(values.mean()), *np.percentile(values[idx].mean(axis=1), [2.5, 97.5])


MEASURES = [
    ("top-1 feature is in the set", "literal (strict, R_cf)", "top1", None),
    ("a top-3 feature is in the set", "literal (strict, R_cf)", "top3", None),
    ("share of |SHAP| mass in the set", "literal (strict, R_cf)", "mass", None),
    ("top-1 in the set, lenient (R_read)", "literal, lenient (R_read)", "top1", None),
    ("a top-3 feature in the set, lenient (R_read)", "literal, lenient (R_read)", "top3", None),
    ("top-1 in the family-aware set", "family-aware (R_cf)", "top1", None),
    ("a top-3 feature in the family-aware set", "family-aware (R_cf)", "top3", None),
    ("top-1 beyond hand strength, family-aware", "family-aware (R_cf)", "top1", HAND_STRENGTH),
]
fid = {}
for model, phi in (("XGBoost", PHI_X), ("EBM", PHI_E)):
    for label, level, key, drop in MEASURES:
        scores = fidelity_scores(phi, COVER[level], drop)
        null = null_per_state(phi, COVER[level], key, drop)
        keep = scores["eligible"] > 0 if drop else np.ones(len(fid_idx), dtype=bool)
        fid[model, label] = {"score": scores[key][keep].mean(), "null": null[keep].mean(), "lift": boot_ci((scores[key] - null)[keep]), "per_state": (scores[key] - null), "keep": keep}
eligible_share = float(fidelity_scores(PHI_X, COVER["family-aware (R_cf)"], HAND_STRENGTH)["eligible"].mean())
constant_equity = {level: float(c[:, FEATURE_INDEX["equity"]].mean()) for level, c in COVER.items()}
diff_lift = {label: boot_ci((fid["XGBoost", label]["per_state"] - fid["EBM", label]["per_state"])[fid["XGBoost", label]["keep"]]) for label, *_ in MEASURES}
fid_table = pd.DataFrame(
    {
        "measure": [m_[0] for m_ in MEASURES],
        "XGBoost score": [pct(fid["XGBoost", m_[0]]["score"], 0) for m_ in MEASURES],
        "null": [pct(fid["XGBoost", m_[0]]["null"], 0) for m_ in MEASURES],
        "XGBoost lift (pp) [95% CI]": [f"{100 * fid['XGBoost', m_[0]]['lift'][0]:+.1f} [{100 * fid['XGBoost', m_[0]]['lift'][1]:+.1f}, {100 * fid['XGBoost', m_[0]]['lift'][2]:+.1f}]" for m_ in MEASURES],
        "EBM score": [pct(fid["EBM", m_[0]]["score"], 0) for m_ in MEASURES],
        "EBM null": [pct(fid["EBM", m_[0]]["null"], 0) for m_ in MEASURES],
        "EBM lift (pp) [95% CI]": [f"{100 * fid['EBM', m_[0]]['lift'][0]:+.1f} [{100 * fid['EBM', m_[0]]['lift'][1]:+.1f}, {100 * fid['EBM', m_[0]]['lift'][2]:+.1f}]" for m_ in MEASURES],
        "XGBoost - EBM lift [95% CI]": [f"{100 * diff_lift[m_[0]][0]:+.1f} [{100 * diff_lift[m_[0]][1]:+.1f}, {100 * diff_lift[m_[0]][2]:+.1f}]" for m_ in MEASURES],
    }
)
display(Markdown(
    f"**[recomputed]** {len(fid_idx):,} validation states on which XGBoost, the EBM and the teacher agree ({len(agreeing):,} such states exist), {fid_seconds:.0f} s for the {2 * len(fid_idx):,} SHAP / term explanations. "
    f"The strict rule set R_cf has on average {R_CF.sum(axis=1).mean():.1f} of the 45 features (R_read: {R_READ.sum(axis=1).mean():.1f}; family-aware: {COVER['family-aware (R_cf)'].sum(axis=1).mean():.1f}). "
    f"**Reference: the constant answer 'always `equity`'** would be a top-1 hit in {pct(constant_equity['literal (strict, R_cf)'], 0)} of the states under the strict set and in {pct(constant_equity['family-aware (R_cf)'], 0)} under the family-aware set "
    f"(every rule reads equity, pot odds or the price, so `equity` or a twin is always covered). *score* = share of states (or mean mass); *null* = the same measure when every state is judged against another state's rule set; "
    f"*lift* = score minus null, with a state-level bootstrap interval; the last measure ignores the hand-strength family and uses the {pct(eligible_share, 0)} of states where the rule needs something else too."
))
display(md_table(fid_table))

# %%
fig, ax = plt.subplots(figsize=(11.6, 5.0))
labels_m = [m_[0] for m_ in MEASURES]
series = {
    model: ([100 * fid[model, l]["lift"][0] for l in labels_m], [100 * fid[model, l]["lift"][1] for l in labels_m], [100 * fid[model, l]["lift"][2] for l in labels_m])
    for model in ("XGBoost", "EBM")
}
nbp.forest_plot(ax, labels_m, series, x_label="lift over the null (percentage points, 95% CI over states)", legend=False, colors={"XGBoost": XGB_COLOR, "EBM": EBM_COLOR})
ax.legend(handles=[Line2D([0], [0], color=c, marker="o", markersize=7, linewidth=1.8, label=n) for n, c in (("XGBoost (SHAP)", XGB_COLOR), ("EBM (own terms)", EBM_COLOR))] + [Line2D([0], [0], color=nbp.MUTED, marker="o", markersize=7, markerfacecolor=nbp.SURFACE, linewidth=1.8, label="hollow: CI contains 0")], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, fontsize=8.5)
nbp.title(ax, "Does the explanation point at what the teacher's rule uses?", "right of 0 = better aligned with the firing rule than with a random other rule")
fig.tight_layout()
nbp.savefig("nb02_fidelity", fig)
plt.show()

# per rule: which feature does SHAP put first, and is it a hit?
rule_rows = []
cover_family = COVER["family-aware (R_cf)"]
top1_x, top1_e = np.argmax(np.abs(PHI_X), axis=1), np.argmax(np.abs(PHI_E), axis=1)
for rule, count in pd.Series(fid_rules).value_counts().head(10).items():
    mask = fid_rules == rule
    leaders = Counter(FEATURE_NAMES[j] for j in top1_x[mask]).most_common(2)
    rule_rows.append(
        {
            "teacher rule": f"`{rule}`", "states": int(count),
            "R_cf (typical)": ", ".join(f"`{FEATURE_NAMES[j]}`" for j in np.argsort(-R_CF[mask].mean(axis=0))[: int(round(R_CF[mask].sum(axis=1).mean()))] if R_CF[mask][:, j].mean() > 0.5),
            "XGBoost top-1 (share)": ", ".join(f"`{f}` {n / count:.0%}" for f, n in leaders),
            "XGB family-aware hit": pct(cover_family[mask][np.arange(mask.sum()), top1_x[mask]].mean(), 0),
            "EBM family-aware hit": pct(cover_family[mask][np.arange(mask.sum()), top1_e[mask]].mean(), 0),
        }
    )
display(Markdown("**By teacher rule** (the ten most frequent in the sample; `R_cf (typical)` = features that flip the decision in more than half of the states of that rule):"))
display(md_table(pd.DataFrame(rule_rows)))

# the eight example cases through the same lens
case_rows = []
for number, case, a_x, a_e, p_x, p_e, reads, decisive, top3 in case_summary:
    cover_case = covered(np.isin(FEATURE_NAMES, list(decisive))[None, :])[0]
    phi_e_case = ebm_feature_attributions(EBM.top_features(case.features, AbstractAction(case.teacher_action)))
    ebm_top = FEATURE_NAMES[int(np.argmax(np.abs(phi_e_case)))]
    case_rows.append(
        {
            "case": f"{number} `{case.key}`", "teacher rule": f"`{case.teacher_rule}`", "R_cf": ", ".join(f"`{f}`" for f in sorted(decisive)),
            "XGBoost top-1": f"`{top3[0]}`" + (" (hit)" if cover_case[FEATURE_INDEX[top3[0]]] else " (miss)"),
            "EBM top-1": f"`{ebm_top}`" + (" (hit)" if cover_case[FEATURE_INDEX[ebm_top]] else " (miss)"),
        }
    )
display(Markdown("**The eight example cases through the same lens** (family-aware hit; `R_cf` = features that can flip that decision; EBM top-1 by the feature-level split of its terms):"))
display(md_table(pd.DataFrame(case_rows)))

# %%
def lift_text(model: str, label: str) -> str:
    v = fid[model, label]["lift"]
    return f"{100 * v[0]:+.1f} pp [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"


top1_names = Counter(FEATURE_NAMES[j] for j in top1_x)
pre_rules = np.array([r.startswith("pre_") for r in fid_rules])
pre_equity_family = float(np.isin(top1_x[pre_rules], [FEATURE_INDEX[f] for f in FAMILIES["hand strength"]]).mean())
pre_percentile_named = float((top1_x[pre_rules] == FEATURE_INDEX["preflop_percentile"]).mean())
M0, M1, M2, M5, M7 = (MEASURES[k][0] for k in (0, 1, 2, 5, 7))
display(Markdown(
    f"**What the fidelity study says `[recomputed]`.**\n\n\n\n"
    f"1. *The top-1 feature is a weak test here.* For {pct(top1_names.most_common(1)[0][1] / len(top1_x), 0)} of the decisions XGBoost's top feature is `{top1_names.most_common(1)[0][0]}`; the next most frequent are "
    f"{', '.join(f'`{f}` ({pct(n / len(top1_x), 0)})' for f, n in top1_names.most_common(4)[1:])} - the teacher's rules are built on equity, so a hand-strength feature is nearly always right *and* nearly always first. "
    f"Strictly (R_cf) XGBoost's top-1 is in the rule's set in {pct(fid['XGBoost', M0]['score'], 0)} of the states against a null of {pct(fid['XGBoost', M0]['null'], 0)} (lift {lift_text('XGBoost', M0)}); the EBM's is {pct(fid['EBM', M0]['score'], 0)} against {pct(fid['EBM', M0]['null'], 0)}. "
    f"Mostly this is the twin problem: the model's first feature is `equity_minus_pot_odds` or `equity` where the rule reads `equity` or `preflop_percentile`. "
    f"Before the flop **{pct(pre_equity_family, 0)}** of XGBoost's top-1 features are `equity`, `equity_minus_pot_odds` or `preflop_percentile`, yet {pct(pre_percentile_named, 0)} of them name `preflop_percentile`, the one number the teacher's preflop rules compare. "
    f"Counting a twin as a hit raises the score to {pct(fid['XGBoost', M5]['score'], 0)} (null {pct(fid['XGBoost', M5]['null'], 0)}) - and the constant answer 'equity' reaches {pct(constant_equity['family-aware (R_cf)'], 0)}, so that score cannot separate a faithful explanation from a lazy one.\n\n"
    f"2. *Where the explanations carry information.* Strict top-3 test: XGBoost {pct(fid['XGBoost', M1]['score'], 0)} vs null {pct(fid['XGBoost', M1]['null'], 0)} ({lift_text('XGBoost', M1)}); EBM {pct(fid['EBM', M1]['score'], 0)} vs {pct(fid['EBM', M1]['null'], 0)} ({lift_text('EBM', M1)}). "
    f"Share of the |SHAP| mass on the rule's features: XGBoost lift {lift_text('XGBoost', M2)}, EBM {lift_text('EBM', M2)}. "
    f"With the hand-strength family set aside, the first remaining feature is a decisive one in {pct(fid['XGBoost', M7]['score'], 0)} (XGBoost; null {pct(fid['XGBoost', M7]['null'], 0)}) and {pct(fid['EBM', M7]['score'], 0)} (EBM; null {pct(fid['EBM', M7]['null'], 0)}) of the states: "
    f"lifts {lift_text('XGBoost', M7)} and {lift_text('EBM', M7)}, a difference of {100 * diff_lift[M7][0]:+.1f} pp [{100 * diff_lift[M7][1]:+.1f}, {100 * diff_lift[M7][2]:+.1f}] - "
    f"{'no resolvable difference between the two explanation types on the most discriminating test' if diff_lift[M7][1] <= 0 <= diff_lift[M7][2] else 'a resolvable difference on the most discriminating test'}, and about half of the time neither points at a feature that can flip the decision.\n\n"
    f"3. *What this does and does not measure.* SHAP is exact for the model (additivity, 5.1); the study measures whether the *route the model takes* matches the *route the teacher takes*. A model that reads the same information through a twin feature is faithfully explained by SHAP and still scores a miss "
    f"on the literal tests - that is why both are shown. R_cf is conservative (a feature that flips the decision only for values outside the grid, or only in some states, is missing), the sample is {len(fid_idx):,} states of one validation split, and states on which the models disagree with the teacher are excluded by design."
))

# %% [markdown]
# ## 7. Comparison with the EBM
#
# Same data, same features, same labels, same hand-level split (section 2.1) - so the differences below are differences of the *model class*. This section compares **decisions**, **cost** and **explanations**; it makes no claim about
# which agent plays better poker beyond the interval-qualified results of 4.4 (the official tournament, Notebook 4, is where the agents meet the full field).
#
# ### 7.1 Do the two models decide alike? `[recomputed]`

# %%
same = PRED_XGB == PRED_EBM
xgb_right, ebm_right = PRED_XGB == Y_VAL, PRED_EBM == Y_VAL
four_way = {
    "both right": int((xgb_right & ebm_right).sum()), "only XGBoost right": int((xgb_right & ~ebm_right).sum()),
    "only EBM right": int((~xgb_right & ebm_right).sum()), "both wrong": int((~xgb_right & ~ebm_right).sum()),
}
disagree = ~same
display(Markdown(
    f"**[recomputed]** The two agents choose the same action in **{pct(same.mean(), 2)}** of the {N_VAL:,} validation states ({int(disagree.sum()):,} disagreements). Against the teacher's label: "
    + ", ".join(f"{k} {v:,} ({pct(v / N_VAL, 2)})" for k, v in four_way.items()) + f". When they disagree, XGBoost is the one that matches the teacher in {pct((xgb_right & disagree).sum() / disagree.sum(), 1)} of the cases."
))
by_street = pd.DataFrame(
    {
        "street": list(STREET_NAMES),
        "decisions": [int((STREET_VAL == s).sum()) for s in range(4)],
        "agree": [pct(same[STREET_VAL == s].mean(), 2) for s in range(4)],
        "disagreements": [int(disagree[STREET_VAL == s].sum()) for s in range(4)],
        "XGBoost errors": [int((~xgb_right)[STREET_VAL == s].sum()) for s in range(4)],
        "EBM errors": [int((~ebm_right)[STREET_VAL == s].sum()) for s in range(4)],
    }
)
by_class = pd.DataFrame(
    {
        "teacher action": [f"`{a}`" for a in ACTIONS],
        "decisions": [int((Y_VAL == c).sum()) for c in range(5)],
        "agree": [pct(same[Y_VAL == c].mean(), 2) for c in range(5)],
        "disagreements": [int(disagree[Y_VAL == c].sum()) for c in range(5)],
        "XGBoost errors": [int((~xgb_right)[Y_VAL == c].sum()) for c in range(5)],
        "EBM errors": [int((~ebm_right)[Y_VAL == c].sum()) for c in range(5)],
    }
)
display(md_table(by_street))
display(md_table(by_class))
pairs = Counter((ACTIONS[a], ACTIONS[b]) for a, b in zip(PRED_XGB[disagree], PRED_EBM[disagree]))
rule_dis = pd.DataFrame({"rule": RULE_VAL, "disagree": disagree}).groupby("rule").agg(n=("disagree", "size"), disagreements=("disagree", "sum")).sort_values("disagreements", ascending=False)
rule_dis["rate"] = rule_dis["disagreements"] / rule_dis["n"]
display(Markdown(
    f"**Where do they disagree?** The most frequent (XGBoost, EBM) pairs of actions are " + ", ".join(f"({a}, {b}) {n}" for (a, b), n in pairs.most_common(4)) + ". By teacher rule (largest absolute number of disagreements):"
))
display(md_table(pd.DataFrame(
    {"rule": [f"`{r}`" for r in rule_dis.head(6).index], "decisions": rule_dis.head(6)["n"], "disagreements": rule_dis.head(6)["disagreements"].astype(int), "rate": [pct(v, 1) for v in rule_dis.head(6)["rate"]]}
)))
margin_dis = (np.median(MARGIN_VAL[disagree]), np.median(MARGIN_VAL))
ebm_deviates = float((~ebm_right & disagree).sum() / disagree.sum())
display(Markdown(
    f"The disagreements are lopsided: in {pct(ebm_deviates, 1)} of them it is the EBM that differs from the teacher (XGBoost: {pct(float((~xgb_right & disagree).sum() / disagree.sum()), 1)}). They concentrate in rules with sharp, multi-feature boundaries and in the pure bluffs - "
    f"`cbet_bluff` ({pct(rule_dis.loc['cbet_bluff', 'rate'], 0)} of its decisions) and `semi_bluff_bet` ({pct(rule_dis.loc['semi_bluff_bet', 'rate'], 0)}), where the teacher bets with a weak hand because of the opponent's fold rate and the board - and in `call_odds` ({pct(rule_dis.loc['call_odds', 'rate'], 1)}). "
    f"They are not rare corner cases: the median teacher margin of a disagreement is {margin_dis[0]:.3f} against {margin_dis[1]:.3f} overall, i.e. closer to a threshold than a typical decision, though not on top of one. "
    f"*Hypothesis, not tested here:* an additive model with a few dozen pairwise tables cannot express a rule that needs three or four features at once; the toy fits of section 3.5 show the capacity effect on XGBoost's own model class, and Notebook 1 studies the EBM's side."
))

# %% [markdown]
# ### 7.2 Accuracy, size and speed `[saved]` `[recomputed]` `[live]`
#
# Latency is timed here, now, on this (shared) machine: the median over 300 validation states of one model call (`predict_proba`) and of one local explanation (`top_features`: SHAP for XGBoost, the term decomposition for the EBM).
# Saved numbers from the two `metadata.json` files are shown next to them for comparison; they were timed on a different, busier machine state.

# %%
tree_table = AGENT.booster.trees_to_dataframe()
split_nodes = int((tree_table["Feature"] != "Leaf").sum())
timing_rows = X_VAL[np.random.default_rng(ANALYSIS_SEED + 5).choice(N_VAL, size=300, replace=False)]


def median_ms(function, rows) -> float:
    samples = []
    for row in rows:
        t0 = time.perf_counter()
        function(row)
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


live = {
    "XGBoost": (median_ms(AGENT.predict_proba, timing_rows), median_ms(lambda r: AGENT.top_features(r, AbstractAction.CHECK_CALL), timing_rows[:100])),
    "EBM": (median_ms(EBM.predict_proba, timing_rows), median_ms(lambda r: EBM.top_features(r, AbstractAction.CHECK_CALL), timing_rows[:100])),
}
row_teacher = {"XGBoost": saved_val["teacher"], "EBM": ebm_saved_val["teacher"]}
compare = pd.DataFrame(
    [
        ("model class", "gradient-boosted trees (non-additive), explained post hoc by SHAP", "additive shape functions + pairwise surfaces (glass box)"),
        ("structure", f"{5 * meta['n_trees_per_class']:,} trees, {split_nodes:,} split nodes", f"{ebm_meta['n_terms']} terms ({ebm_meta['n_terms'] - ebm_meta['n_interaction_terms']} main + {ebm_meta['n_interaction_terms']} pairwise)"),
        ("file size", f"{meta['model_file_bytes'] / 1e6:.1f} MB", f"{ebm_meta['model_file_bytes'] / 1e6:.1f} MB"),
        ("training rows / time of the final fit", f"{meta['n_train_rows']:,} / {meta['training_time_s']:.0f} s ({cfg['N_JOBS']} threads)", f"{ebm_meta['n_train_rows']:,} / {ebm_meta['training_time_s']:.0f} s ({ebm_meta['config']['n_jobs']} threads)"),
        ("whole pipeline (4 rounds + validation)", f"{meta['total_pipeline_time_s'] / 60:.1f} min", f"{ebm_meta['total_pipeline_time_s'] / 60:.1f} min"),
        ("masked accuracy, validation `[recomputed]`", f"{pct(1 - err_ci['XGBoost'][0], 2)} [{pct(1 - err_ci['XGBoost'][2], 2)}, {pct(1 - err_ci['XGBoost'][1], 2)}]", f"{pct(1 - err_ci['EBM'][0], 2)} [{pct(1 - err_ci['EBM'][2], 2)}, {pct(1 - err_ci['EBM'][1], 2)}]"),
        ("log-loss, validation `[recomputed]`", f"{M_XGB['log_loss']:.4f}", f"{M_EBM['log_loss']:.4f}"),
        ("on-policy agreement `[saved]`", pct(meta["imitation_on_policy"]["masked_accuracy"], 2), pct(ebm_meta["imitation_on_policy"]["masked_accuracy"], 2)),
        ("saved validation BB/100 vs teacher (legacy protocol, informational)", f"{ci_text(row_teacher['XGBoost']['bb_per_100'], *row_teacher['XGBoost']['ci95_pairs'])} ({row_teacher['XGBoost']['pairs']} pairs)", f"{ci_text(row_teacher['EBM']['bb_per_100'], *row_teacher['EBM']['ci95_pairs'])} ({row_teacher['EBM']['pairs']} pairs)"),
        ("fresh BB/100 vs teacher, two-table `[fresh]`", result_text(fresh["XGBoost", "teacher"]) + f" ({fresh['XGBoost', 'teacher'].n_pairs} pairs)", result_text(fresh["EBM", "teacher"]) + f" ({fresh['EBM', 'teacher'].n_pairs} pairs)"),
        ("latency `predict_proba`, median `[live]` (saved)", f"{live['XGBoost'][0]:.2f} ms ({meta['inference']['predict_proba_median_ms']:.2f})", f"{live['EBM'][0]:.2f} ms ({ebm_meta['inference']['predict_proba_median_ms']:.2f})"),
        ("latency of one local explanation, median `[live]` (saved)", f"{live['XGBoost'][1]:.1f} ms ({meta['inference']['shap_explanation_median_ms']:.1f})", f"{live['EBM'][1]:.2f} ms ({ebm_meta['inference']['term_decomposition_median_ms']:.2f})"),
    ],
    columns=["", "XGBoost", "EBM"],
)
display(md_table(compare))
display(Markdown(
    f"**Reading the table.** XGBoost is {meta['model_file_bytes'] / ebm_meta['model_file_bytes']:.1f}x larger on disk, took {meta['training_time_s'] / ebm_meta['training_time_s']:.2f}x the final-fit time of the EBM (different thread counts and a loaded machine: indicative only), "
    f"predicts about as fast per state ({live['XGBoost'][0]:.2f} vs {live['EBM'][0]:.2f} ms live) - and explains **{live['XGBoost'][1] / live['EBM'][1]:.0f}x slower** ({live['XGBoost'][1]:.1f} vs {live['EBM'][1]:.2f} ms), because every SHAP call builds a `DMatrix` and walks all "
    f"{5 * meta['n_trees_per_class']:,} trees; that is invisible for one decision in the UI and relevant for explaining millions of decisions (batching helps: the 3,000-state SHAP matrix of section 5 took seconds). "
    f"The EBM's explanation is its prediction path, so it is nearly free."
))

# %% [markdown]
# ### 7.3 Do the explanations tell the same story? `[recomputed]`
#
# Two comparisons: **global** (how do the two models rank the 45 features?) and **local** (for the same decision, do they point at the same features?). EBM term importances come from `explain.ebm.term_importance` on the same SHAP sample (mean |term score| averaged over the
# five actions; an interaction term's importance is split equally between its two features); XGBoost's are the mean |SHAP| of section 5. The two are in different units, so only **ranks** are compared.

# %%
ebm_terms = term_importance(EBM, X_SHAP)
ebm_feature = {f: 0.0 for f in FEATURE_NAMES}
ebm_main_only = {f: 0.0 for f in FEATURE_NAMES}
for _, row in ebm_terms.iterrows():
    for f in row["features"]:
        ebm_feature[f] += row["importance"] / len(row["features"])
    if row["kind"] == "main":
        ebm_main_only[row["features"][0]] += row["importance"]
xgb_importance = {f: mean_abs[f] for f in FEATURE_NAMES}
vec = lambda d: np.array([d[f] for f in FEATURE_NAMES])  # noqa: E731
rank_rows = []
for name, d in (("EBM, main + half of each interaction", ebm_feature), ("EBM, main-effect terms only", ebm_main_only)):
    rho, tau = stats.spearmanr(vec(xgb_importance), vec(d)).statistic, stats.kendalltau(vec(xgb_importance), vec(d)).statistic
    top_x, top_e = set(sorted(FEATURE_NAMES, key=lambda f: -xgb_importance[f])[:10]), set(sorted(FEATURE_NAMES, key=lambda f: -d[f])[:10])
    rank_rows.append({"EBM importance": name, "Spearman": rho, "Kendall tau": tau, "top-10 overlap": f"{len(top_x & top_e)} of 10"})
display(md_table(pd.DataFrame(rank_rows), "{:.2f}"))

# local agreement: same decisions, same action
local_rho = np.array([stats.spearmanr(np.abs(PHI_X[k]), np.abs(PHI_E[k])).statistic for k in range(len(fid_idx))])
same_top1 = float((top1_x == top1_e).mean())
family_same = float(np.mean([bool((FAMILY_MATRIX[a] & FAMILY_MATRIX[b]).any()) or a == b for a, b in zip(top1_x, top1_e)]))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
x_values = np.array([xgb_importance[f] for f in FEATURE_NAMES])
y_values = np.array([ebm_feature[f] for f in FEATURE_NAMES])
ax.scatter(x_values, y_values, color=[GROUP_COLORS[FEATURE_GROUP_OF[f]] for f in FEATURE_NAMES], s=46, zorder=3)
label_offsets = [(-8, 8), (6, -12), (6, 6), (-10, -14), (6, 6), (6, -12), (-8, 8), (6, 6)]
for rank_, j in enumerate(np.argsort(-np.maximum(x_values / x_values.max(), y_values / y_values.max()))[:8]):
    dx, dy = label_offsets[rank_]
    ax.annotate(FEATURE_NAMES[j], (x_values[j], y_values[j]), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=nbp.TEXT, ha="right" if dx < 0 else "left")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(max(x_values[x_values > 0].min(), 1e-3) * 0.7, x_values.max() * 2)
ax.set_ylim(max(y_values[y_values > 0].min(), 1e-3) * 0.7, y_values.max() * 2)
ax.set_xlabel("XGBoost: mean |SHAP|, summed over actions (log scale)")
ax.set_ylabel("EBM: term importance, interactions split (log scale)")
legend_groups = [Patch(color=GROUP_COLORS[g]) for g in dict.fromkeys(FEATURE_GROUP_OF[f] for f in FEATURE_NAMES)]
ax.legend(legend_groups, [g.replace("_", " ") for g in dict.fromkeys(FEATURE_GROUP_OF[f] for f in FEATURE_NAMES)], fontsize=8, loc="lower right", title="feature group")
nbp.title(ax, "Global importance: two models, one ranking?", f"Spearman {rank_rows[0]['Spearman']:.2f}; each dot is one of the 45 features")
ax2.hist(local_rho, bins=np.linspace(-0.2, 1.0, 25), color=nbp.SERIES[3], alpha=0.9)
ax2.axvline(np.median(local_rho), color=nbp.TEXT, linestyle="--", linewidth=1.2)
ax2.set_xlabel("Spearman correlation of |attribution| over the 45 features, same decision")
ax2.set_ylabel("decisions")
nbp.title(ax2, "Local agreement of the two explanations", f"median {np.median(local_rho):.2f}; same top-1 feature in {pct(same_top1, 0)} of {len(fid_idx):,} decisions")
fig.tight_layout()
nbp.savefig("nb02_explanation_comparison", fig)
plt.show()
ebm_top = sorted(FEATURE_NAMES, key=lambda f: -ebm_feature[f])[:10]
xgb_top = sorted(FEATURE_NAMES, key=lambda f: -xgb_importance[f])[:10]
display(Markdown(
    f"**[recomputed]** *Global*: the two rankings are only **moderately** alike (Spearman {rank_rows[0]['Spearman']:.2f}, Kendall {rank_rows[0]['Kendall tau']:.2f}; {rank_rows[0]['top-10 overlap']} features in both top-10s). Both put `equity` and `equity_minus_pot_odds` first. "
    f"Top 10 by XGBoost SHAP: {', '.join(f'`{f}`' for f in xgb_top)}; top 10 by EBM term importance: {', '.join(f'`{f}`' for f in ebm_top)}. "
    f"The two models also spread credit between the redundant features differently, and an additive model has to give an explicit term to things that trees can fold into a split deeper down (legality flags, board texture). "
    f"*Local*: for the same decision and the same action the two explanations have a median rank correlation of {np.median(local_rho):.2f} (quartiles {np.percentile(local_rho, 25):.2f} / {np.percentile(local_rho, 75):.2f}) and name the *same* top-1 feature in {pct(same_top1, 0)} of the {len(fid_idx):,} decisions "
    f"({pct(family_same, 0)} if features of the same redundancy family count as the same). The gap between the two numbers is the share explained by redundancy; the remaining {pct(1 - family_same, 0)} are decisions on which the two explanations name different things."
))

# %% [markdown]
# ## 8. Did the model learn the teacher's *structure*?
#
# Agreement on 99.8% of states says the model's outputs match; it does not say *how*. Two independent checks look inside the model for the teacher's rules, which are known exactly (`TeacherParams`):
#
# 1. **Where do the trees split?** If the model has recovered the thresholds, its biggest splits on `preflop_percentile` and `equity` should sit at the teacher's thresholds.
# 2. **Does the model's decision boundary move like the teacher's?** Take real validation states, change one thing at a time, and compare *where the action flips* for the model and for the teacher itself.
#
# ### 8.1 Split thresholds of the trees `[recomputed]`
#
# Every split node of the model's trees names a feature and a threshold and has a *gain* (how much it reduces the loss). The figure shows, per feature, the gain-weighted distribution of thresholds with the teacher's thresholds as vertical lines.

# %%
splits = tree_table[tree_table["Feature"] != "Leaf"]
# the 169 starting-hand classes (also used by the ladders of 8.2): weakest first
PREFLOP = np.flatnonzero(X_VAL[:, FEATURE_INDEX["street"]] == 0)
HAND_COLUMNS = [FEATURE_INDEX[c] for c in ("equity", "preflop_percentile", "made_hand_category", "hole_high_rank", "hole_low_rank", "is_pair", "is_suited", "hole_gap")]
hand_table = pd.DataFrame(X_VAL[PREFLOP][:, HAND_COLUMNS]).drop_duplicates(subset=[0, 1]).sort_values(1)
HANDS = hand_table.to_numpy(dtype=np.float32)
PERCENTILES = HANDS[:, 1]
PRE_THRESHOLDS = {"limp": PARAMS.limp_pct, "open": PARAMS.open_pct, "push": PARAMS.push_pct, "iso": PARAMS.iso_pct, "3-bet": PARAMS.threebet_pct, "open pot": PARAMS.open_pot_pct, "iso pot": PARAMS.iso_pot_pct, "4-bet": PARAMS.fourbet_pct, "shove": PARAMS.shove_pct}
# before the flop equity is a monotone function of the percentile, so each preflop threshold also exists as an equity value: that of the first hand class above it
PRE_AS_EQUITY = {name: float(HANDS[np.flatnonzero(PERCENTILES >= value)[0], 0]) for name, value in PRE_THRESHOLDS.items()}
POST_THRESHOLDS = {"c-bet": PARAMS.cbet_min_equity, "value floor": MIN_VALUE_THRESHOLD, "relief": PARAMS.penalty_relief_equity, "wet pot": PARAMS.wet_pot_equity, "pot": PARAMS.pot_bet_equity, "raise": PARAMS.raise_equity, "monster": PARAMS.monster_equity}
LINES = {"preflop_percentile": [(PRE_THRESHOLDS, nbp.TEXT, "teacher threshold")], "equity": [(POST_THRESHOLDS, nbp.TEXT, "post-flop threshold"), (PRE_AS_EQUITY, nbp.SERIES[2], "preflop threshold expressed as equity")]}
HALF_WIDTH = 0.01
GRID = np.linspace(0, 1, 2001)
fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.8))
split_stats = {}
for ax, (feature, groups) in zip(axes, LINES.items()):
    chosen = splits[splits["Feature"] == feature]
    edges = np.arange(0, 1.0001, 0.005)
    weights = chosen["Gain"].to_numpy()
    histogram, _ = np.histogram(chosen["Split"], bins=edges, weights=weights)
    ax.bar(edges[:-1], 100 * histogram / histogram.sum(), width=0.0046, align="edge", color=GROUP_COLORS[FEATURE_GROUP_OF[feature]])
    top_y = ax.get_ylim()[1]
    all_values = []
    for g, (lines, color, label) in enumerate(groups):
        for k, (name, value) in enumerate(sorted(lines.items(), key=lambda kv: kv[1])):
            all_values.append(value)
            ax.axvline(value, color=color, linestyle=":" if g == 0 else "--", linewidth=1.1)
            ax.text(value, top_y * (0.97 - 0.065 * (k % 3) - 0.2 * g), f" {name} {value:.3g}", ha="left", va="top", fontsize=7.8, color=color)
    near_line = np.zeros(len(chosen), dtype=bool)
    for value in all_values:
        near_line |= np.abs(chosen["Split"].to_numpy() - value) <= HALF_WIDTH
    axis_share = float((np.min(np.abs(GRID[:, None] - np.array(all_values)[None, :]), axis=1) <= HALF_WIDTH).mean())
    split_stats[feature] = {"gain_share": float(weights[near_line].sum() / weights.sum()), "count_share": float(near_line.mean()), "axis": axis_share, "n": len(chosen), "lines": len(all_values)}
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"split threshold on {feature}")
    ax.set_ylabel("share of the feature's total gain (%)")
    nbp.title(ax, f"{feature}: {split_stats[feature]['n']:,} splits", f"{pct(split_stats[feature]['gain_share'], 0)} of the gain sits within {HALF_WIDTH} of one of {len(all_values)} teacher thresholds (lines); those windows cover {pct(axis_share, 0)} of the axis")
fig.tight_layout()
nbp.savefig("nb02_split_thresholds", fig)
plt.show()
pre_only = np.zeros(len(splits[splits["Feature"] == "equity"]), dtype=bool)
equity_splits = splits[splits["Feature"] == "equity"]
for value in PRE_AS_EQUITY.values():
    pre_only |= np.abs(equity_splits["Split"].to_numpy() - value) <= HALF_WIDTH
display(Markdown(
    f"**[recomputed]** {len(splits):,} split nodes in total. On `preflop_percentile`, {pct(split_stats['preflop_percentile']['gain_share'], 0)} of the gain (and {pct(split_stats['preflop_percentile']['count_share'], 0)} of the splits) lie within {HALF_WIDTH} of one of the teacher's {split_stats['preflop_percentile']['lines']} thresholds - "
    f"windows that together cover only {pct(split_stats['preflop_percentile']['axis'], 0)} of the axis. The biggest spikes are at the high thresholds (3-bet, open-pot, 4-bet, shove); the *low* thresholds (limp, open, push, iso) are hardly split on `preflop_percentile` at all. "
    f"They are on `equity` instead: before the flop `equity` is a monotone function of the percentile, so the teacher's preflop thresholds also exist as equity values (orange dashed lines, e.g. `open` 0.18 = equity {PRE_AS_EQUITY['open']:.3f}), and "
    f"{pct(float(equity_splits['Gain'].to_numpy()[pre_only].sum() / equity_splits['Gain'].sum()), 0)} of the gain on `equity` sits at those translated preflop thresholds alone; together with the fixed post-flop thresholds the share is {pct(split_stats['equity']['gain_share'], 0)} of the gain "
    f"({pct(split_stats['equity']['count_share'], 0)} of the splits) in windows covering {pct(split_stats['equity']['axis'], 0)} of the axis. The model uses the two redundant features interchangeably - the same redundancy that spreads SHAP credit in section 5.2. "
    f"The post-flop shares are a *lower bound* of the structure copied, because several boundaries of the teacher are not fixed numbers: the value line slides between {MIN_VALUE_THRESHOLD:.2f} and {MAX_VALUE_THRESHOLD:.2f} with the opponent's fold-to-bet rate, and the call/fold boundary depends on the price, the bet size, the raises and the opponent's aggression. "
    f"(The percentile takes only 169 distinct values, so a threshold can only be placed in the gap between two hand classes.)"
))

# %% [markdown]
# ### 8.2 Do the model's decision boundaries move like the teacher's? `[live]`
#
# Counterfactual scans on **real validation states used as templates**: only one quantity is changed, the teacher's own function and the model are both asked about every variant, and we compare *where the action flips*.
# The teacher is a pure function of the features, so it can answer for any feature vector; the caveat is that a scan can leave the region where the dataset has data (e.g. the equity of a hand changed without changing its cards),
# where a learned model owes nothing. Whatever the scans show holds for states of this kind only.
#
# * **Preflop ladders** - keep the context of a real state (position, raises, stack, opponent) and *replace the hand* by each of the 169 starting-hand classes (all hand features swapped consistently: equity, percentile, ranks, pairs, suitedness). The teacher's thresholds are `limp_pct`, `open_pct`, `open_pot_pct` (first to act),
#   `iso_pct`, `iso_pot_pct` (big blind after a limp), `threebet_pct` (facing one raise), `fourbet_pct`, `shove_pct` (facing a 3-bet or more). Quantity compared: the weakest starting hand whose action reaches the higher level.
# * **Post-flop equity scans** - vary `equity` (and the derived `equity_minus_pot_odds`) on a grid of 0.005. (a) *Checked to*, dry board, no draw: where does *check* turn into *bet*? The teacher's value line is `0.57 + 0.4 x (opponent fold-to-bet - 0.4)`, clipped to [0.52, 0.92]: a flat floor at 0.52 for opponents that fold less than 27.5% of the time and a *rising* line above that - a threshold that moves with an opponent statistic. Half of the templates are drawn from opponents above 27.5% so that the rising part is tested. (b) *Facing a bet*: where does *fold* turn into *call*? That boundary depends on price, bet size, raises and opponent aggression.

# %%
scan_rng = np.random.default_rng(ANALYSIS_SEED + 6)
col = lambda name: X_VAL[:, FEATURE_INDEX[name]]  # noqa: E731
CONTEXTS = {
    "first to act on the button (>= 40 bb)": (np.flatnonzero((col("street") == 0) & (col("is_button") == 1) & (col("num_raises_street") == 0) & (col("effective_stack_bb") >= 40)), [("limp_pct", PARAMS.limp_pct, 1), ("open_pct", PARAMS.open_pct, 2), ("open_pot_pct", PARAMS.open_pot_pct, 3)]),
    "big blind after a limp (>= 40 bb)": (np.flatnonzero((col("street") == 0) & (col("is_button") == 0) & (col("num_raises_street") == 0) & (col("effective_stack_bb") >= 40)), [("iso_pct", PARAMS.iso_pct, 2), ("iso_pot_pct", PARAMS.iso_pot_pct, 3)]),
    "facing one raise (>= 40 bb)": (np.flatnonzero((col("street") == 0) & (col("num_raises_street") == 1) & (col("effective_stack_bb") >= 40)), [("threebet_pct", PARAMS.threebet_pct, 3)]),
    "facing a 3-bet or more (>= 20 bb)": (np.flatnonzero((col("street") == 0) & (col("num_raises_street") >= 2) & (col("effective_stack_bb") >= 20)), [("fourbet_pct", PARAMS.fourbet_pct, 3), ("shove_pct", PARAMS.shove_pct, 4)]),
}


def model_actions(matrix: np.ndarray, legal: np.ndarray) -> np.ndarray:
    return masked_argmax(class_probabilities(AGENT.booster, matrix), legal)


def teacher_actions(matrix: np.ndarray, legal: np.ndarray) -> np.ndarray:
    return np.array([int(TEACHER.decide_features(row, mask).action) for row, mask in zip(matrix, legal)])


def first_hand_at_level(actions: np.ndarray, level: int) -> float:
    """Percentile of the weakest starting hand whose action is at least ``level`` (nan if none)."""
    hit = np.flatnonzero(actions >= level)
    return float(PERCENTILES[hit[0]]) if len(hit) else float("nan")


ladder_rows, ladder_agreement = [], []
started = time.perf_counter()
for context_name, (candidates, thresholds) in CONTEXTS.items():
    chosen = scan_rng.choice(candidates, size=min(len(candidates), COUNTERFACTUAL_TEMPLATES // 15), replace=False)
    teacher_first = {t[0]: [] for t in thresholds}
    model_first = {t[0]: [] for t in thresholds}
    for i in chosen:
        matrix = np.repeat(X_VAL[i][None], len(HANDS), axis=0)
        matrix[:, HAND_COLUMNS] = HANDS
        matrix[:, FEATURE_INDEX["equity_minus_pot_odds"]] = matrix[:, FEATURE_INDEX["equity"]] - matrix[:, FEATURE_INDEX["pot_odds"]]
        legal = np.repeat(L_VAL[i][None], len(HANDS), axis=0)
        by_teacher, by_model = teacher_actions(matrix, legal), model_actions(matrix, legal)
        ladder_agreement.append((context_name, float((by_teacher == by_model).mean()), len(HANDS)))
        for name, _, level in thresholds:
            teacher_first[name].append(first_hand_at_level(by_teacher, level))
            model_first[name].append(first_hand_at_level(by_model, level))
    for name, value, level in thresholds:
        t_arr, m_arr = np.array(teacher_first[name]), np.array(model_first[name])
        both = ~np.isnan(t_arr) & ~np.isnan(m_arr)
        ladder_rows.append(
            {
                "context": context_name, "threshold": f"`{name}`", "teacher parameter": value, "templates": int(len(chosen)),
                "weakest hand playing it: teacher": float(np.nanmedian(t_arr)) if both.any() else float("nan"),
                "XGBoost": float(np.nanmedian(m_arr)) if both.any() else float("nan"),
                "templates with the same hand": f"{int((t_arr[both] == m_arr[both]).sum())} of {int(both.sum())}",
            }
        )
ladder_seconds = time.perf_counter() - started
context_agreement = pd.DataFrame(ladder_agreement, columns=["context", "agreement", "n"]).groupby("context").agg(ladders=("n", "size"), agreement=("agreement", "mean"))
display(Markdown(f"**[live]** ({ladder_seconds:.0f} s) Preflop ladders: for each context, real validation states act as templates and are replayed with all 169 starting-hand classes."))
display(md_table(pd.DataFrame(ladder_rows), "{:.3f}"))
total_agreement = float(np.mean([a[1] for a in ladder_agreement]))
ladder_gap = float(np.median(np.diff(PERCENTILES)))
display(Markdown(
    f"**[live]** Over all {len(ladder_agreement)} ladders ({sum(a[2] for a in ladder_agreement):,} counterfactual states) the model picks the teacher's action in {pct(total_agreement, 2)} of the cases"
    f" (by context: " + ", ".join(f"{c.split(' (')[0]} {pct(r['agreement'], 1)}" for c, r in context_agreement.iterrows()) + "). "
    f"The table lists the percentile of the weakest hand class that reaches each level; because hand classes are discrete (median spacing {ladder_gap:.3f} in percentile), a teacher parameter of 0.18 shows up as the first class above it. "
    f"Identical entries mean the model switches at exactly the same hand class as the teacher."
))

# %%
def scan_switch(candidates: np.ndarray, grid: np.ndarray, base_action: int, label: str, n: int, extra_levels: tuple[int, ...] = ()):
    """For ``n`` template states: the first equity on ``grid`` at which the action leaves ``base_action`` (teacher and model)."""
    chosen = scan_rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    records = []
    for i in chosen:
        matrix = np.repeat(X_VAL[i][None], len(grid), axis=0)
        matrix[:, FEATURE_INDEX["equity"]] = grid
        matrix[:, FEATURE_INDEX["equity_minus_pot_odds"]] = grid - X_VAL[i, FEATURE_INDEX["pot_odds"]]
        legal = np.repeat(L_VAL[i][None], len(grid), axis=0)
        by_teacher, by_model = teacher_actions(matrix, legal), model_actions(matrix, legal)

        def first_leaving(actions: np.ndarray) -> float:
            hit = np.flatnonzero(actions != base_action)
            return float(grid[hit[0]]) if len(hit) else float("nan")

        def first_reaching(actions: np.ndarray, level: int) -> float:
            hit = np.flatnonzero(actions >= level)
            return float(grid[hit[0]]) if len(hit) else float("nan")

        record = {"row": int(i), "teacher": first_leaving(by_teacher), "model": first_leaving(by_model)}
        for level in extra_levels:
            record[f"teacher_{level}"], record[f"model_{level}"] = first_reaching(by_teacher, level), first_reaching(by_model, level)
        records.append(record)
    frame_ = pd.DataFrame(records)
    frame_["label"] = label
    return frame_


GRID_STEP = 0.005
dry_checked_to = np.flatnonzero(
    (col("street") > 0) & (col("facing_bet") == 0) & (col("effective_stack_bb") >= 20) & (col("flush_draw") == 0) & (col("straight_outs") < 8)
    & (col("board_max_suit_count") <= 2) & (col("board_straightness") <= 2) & (col("i_have_initiative") == 0) & (col("is_button") == 0)
)  # fmt: skip
facing_single_bet = np.flatnonzero(
    (col("street") > 0) & (col("facing_bet") == 1) & (col("facing_all_in") == 0) & (col("effective_stack_bb") >= 20) & (col("num_raises_street") == 1) & (col("flush_draw") == 0)
)  # fmt: skip
assert (PARAMS.value_equity, PARAMS.value_fold_slope, FOLD_TO_BET_NEUTRAL, MIN_VALUE_THRESHOLD, MAX_VALUE_THRESHOLD) == (0.57, 0.4, 0.4, 0.52, 0.92)  # the numbers quoted in the text above
fold_rate = col("opp_fold_to_bet")
FLOOR_FOLD_RATE = FOLD_TO_BET_NEUTRAL - (PARAMS.value_equity - MIN_VALUE_THRESHOLD) / PARAMS.value_fold_slope  # fold rate at which the line leaves its 0.52 floor
rising = dry_checked_to[fold_rate[dry_checked_to] > FLOOR_FOLD_RATE]
flat = dry_checked_to[fold_rate[dry_checked_to] <= FLOOR_FOLD_RATE]
value_templates = np.concatenate([scan_rng.choice(flat, size=min(len(flat), COUNTERFACTUAL_TEMPLATES // 2), replace=False), scan_rng.choice(rising, size=min(len(rising), COUNTERFACTUAL_TEMPLATES // 2), replace=False)])
started = time.perf_counter()
value_scan = scan_switch(value_templates, np.round(np.arange(0.30, 1.0 + 1e-9, GRID_STEP), 3), 1, "check -> bet (value line)", len(value_templates), extra_levels=(3,))
call_scan = scan_switch(facing_single_bet, np.round(np.arange(0.05, 0.80 + 1e-9, GRID_STEP), 3), 0, "fold -> call", COUNTERFACTUAL_TEMPLATES)
scan_seconds = time.perf_counter() - started
value_line = np.clip(PARAMS.value_equity + PARAMS.value_fold_slope * (X_VAL[value_scan["row"], FEATURE_INDEX["opp_fold_to_bet"]] - FOLD_TO_BET_NEUTRAL), MIN_VALUE_THRESHOLD, MAX_VALUE_THRESHOLD)
value_scan["analytic"] = value_line
value_scan["fold"] = X_VAL[value_scan["row"], FEATURE_INDEX["opp_fold_to_bet"]]
valid_value = value_scan.dropna(subset=["teacher", "model"])
valid_value = valid_value[(valid_value["teacher"] > 0.30) & (valid_value["teacher"] < 0.83)]
pot_valid = value_scan.dropna(subset=["teacher_3", "model_3"])
valid_call = call_scan.dropna(subset=["teacher", "model"])
valid_call = valid_call[valid_call["teacher"] > 0.05 + 1e-9]
assert (valid_value["teacher"] - valid_value["analytic"]).abs().max() <= GRID_STEP + 1e-6  # the teacher's switch IS its documented value line

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
ax = axes[0]
curve_x = np.linspace(0, max(0.8, float(valid_value["fold"].max()) + 0.02), 200)
ax.plot(curve_x, np.clip(PARAMS.value_equity + PARAMS.value_fold_slope * (curve_x - FOLD_TO_BET_NEUTRAL), MIN_VALUE_THRESHOLD, MAX_VALUE_THRESHOLD), color=nbp.TEXT, linewidth=1.6, label="teacher's value line")
ax.scatter(valid_value["fold"], valid_value["model"], s=14, color=XGB_COLOR, alpha=0.75, label="XGBoost switch point")
ax.set_xlabel("opponent's fold-to-bet rate (the template's own statistic)")
ax.set_ylabel("equity at which the agent starts betting")
ax.legend(loc="upper left", fontsize=8.5)
nbp.title(ax, "Check -> bet: a threshold that moves", f"{len(valid_value)} real checked-to states, equity scanned in steps of {GRID_STEP}")
ax = axes[1]
ax.plot([0.0, 0.8], [0.0, 0.8], color=nbp.MUTED, linestyle="--", linewidth=1.2)
ax.scatter(valid_call["teacher"], valid_call["model"], s=14, color=XGB_COLOR, alpha=0.75)
ax.set_xlabel("equity at which the teacher starts calling")
ax.set_ylabel("equity at which XGBoost starts calling")
nbp.title(ax, "Fold -> call", f"{len(valid_call)} real states facing one bet (price, size, aggression vary)")
ax = axes[2]
errors = {"check -> bet": valid_value["model"] - valid_value["teacher"], "fold -> call": valid_call["model"] - valid_call["teacher"]}
if len(pot_valid):
    errors["bet half pot -> pot (0.84, dry)"] = pot_valid["model_3"] - pot_valid["teacher_3"]
bins_e = np.arange(-0.0775, 0.0776, GRID_STEP)
for (name, values), color in zip(errors.items(), (nbp.ACCENT, XGB_COLOR, nbp.SERIES[2])):
    ax.hist(np.clip(values, -0.075, 0.075), bins=bins_e, histtype="stepfilled", alpha=0.45, color=color, label=f"{name} (n={len(values)})")
ax.set_yscale("log")
ax.set_xlabel("model switch point minus teacher switch point (equity points; clipped at +-0.075)")
ax.set_ylabel("templates (log scale)")
ax.legend(fontsize=8.2, loc="upper left")
nbp.title(ax, "How far off is the boundary?", "0 = identical to within one grid step")
fig.tight_layout()
nbp.savefig("nb02_threshold_recovery", fig)
plt.show()
scan_table = pd.DataFrame(
    [
        {
            "boundary": name, "templates": len(values), "median error": float(np.median(values)), "mean abs error": float(np.abs(values).mean()),
            "90th percentile abs error": float(np.percentile(np.abs(values), 90)), "within one grid step": pct(float((np.abs(values) <= GRID_STEP + 1e-9).mean()), 0), "largest abs error": float(np.abs(values).max()),
        }
        for name, values in errors.items()
    ]
)
display(Markdown(f"**[live]** ({scan_seconds:.0f} s) Equity scans (grid step {GRID_STEP}); errors in equity points, model minus teacher:"))
display(md_table(scan_table, "{:.4f}"))
value_corr = float(np.corrcoef(valid_value["analytic"], valid_value["model"])[0, 1])
rising_rows = valid_value[valid_value["fold"] > FLOOR_FOLD_RATE]
slope_model = float(np.polyfit(rising_rows["fold"], rising_rows["model"], 1)[0]) if len(rising_rows) > 2 else float("nan")
display(Markdown(
    f"**[live]** The teacher's check-to-bet switch equals its documented value line to within one grid step (asserted). XGBoost's switch point follows that line - the floor of {MIN_VALUE_THRESHOLD:.2f} for {len(valid_value) - len(rising_rows)} templates and the rising part for {len(rising_rows)} templates with a fold-to-bet rate above {FLOOR_FOLD_RATE:.3f} (fitted slope on that part {slope_model:.2f}, teacher {PARAMS.value_fold_slope:.2f}) - "
    f"with correlation {value_corr:.3f} and a median error of {np.median(errors['check -> bet']):+.4f} equity points: it learned the *dependence on the opponent's fold-to-bet rate*, not a fixed threshold. "
    f"The price-versus-adjusted-equity boundary of the call/fold rule is reproduced with a mean absolute error of {scan_table.loc[1, 'mean abs error']:.4f} equity points ({scan_table.loc[1, 'within one grid step']} of the templates within one grid step), the worst template off by {scan_table.loc[1, 'largest abs error']:.3f}. "
    f"These are scans of *real* states with a single quantity changed; they say nothing about states unlike any in the dataset."
))

# %% [markdown]
# ## 9. Interpretation
#
# The blocks below are kept strictly apart: **empirical findings** are numbers computed or read in this notebook (every number is generated from the variables above, so the text stays true if the artifacts are regenerated);
# **assumptions** are things taken as given; **speculation** are hypotheses the data suggest but that were not tested; **limitations** say what the evidence cannot support.

# %%
phrase = {"wins": "**beats**", "loses": "**loses to**", "no significant difference": "is **not distinguishable from**"}
t_fresh, t_saved = fresh["XGBoost", "teacher"], saved_val["teacher"]
ebm_t_fresh = fresh["EBM", "teacher"]
h_word = verdict(head.ci_low, head.ci_high)
others_marked = [o for o in EVAL_OPPONENTS if verdict(paired_vs_ebm[o].ci_low, paired_vs_ebm[o].ci_high) != "no significant difference"]
xgb_acc, ebm_acc = 1 - err_ci["XGBoost"][0], 1 - err_ci["EBM"][0]
share_near = float(near[err_mask].mean())
ladder_same = sum(int(r["templates with the same hand"].split(" of ")[0]) for r in ladder_rows)
ladder_total = sum(int(r["templates with the same hand"].split(" of ")[1]) for r in ladder_rows)
top_pairs = ", ".join(f"`{a}` x `{b}`" for a, b, _ in pair_list[:3])
FINDINGS_FIDELITY_ROW = MEASURES[7][0]
if OFFICIAL is not None:
    row_xe_, row_xt_ = OFFICIAL["XGBoost", "EBM"], OFFICIAL["XGBoost", "Teacher"]
    official_verdict = verdict(row_xe_["ci_low"], row_xe_["ci_high"])
    fresh_contains = head.ci_low <= row_xe_["bb_per_100"] <= head.ci_high
    official_sentence = (
        f"The official tournament (Notebook 4; {int(row_xe_['units']):,} pairs per pairing, same protocol) is the larger sample: XGBoost vs EBM {ci_text(row_xe_['bb_per_100'], row_xe_['ci_low'], row_xe_['ci_high'])} BB/100 ({official_verdict}), "
        f"XGBoost vs Teacher {ci_text(row_xt_['bb_per_100'], row_xt_['ci_low'], row_xt_['ci_high'])} ({verdict(row_xt_['ci_low'], row_xt_['ci_high'])}). The fresh head-to-head interval {'contains' if fresh_contains else 'does not contain'} the official estimate and "
        f"{'excludes' if (head.ci_low > 0 or head.ci_high < 0) else 'does not exclude'} zero itself, so this notebook's sample is {'compatible with the official result but too small to resolve it' if fresh_contains and official_verdict != 'no significant difference' else 'consistent with the official verdict'}. "
        + ("In the larger sample the more accurate copy is therefore, if anything, the *weaker* player against the EBM - one pairing, an association and not a mechanism (speculation 1 below)." if official_verdict == "loses" else "")
    )
else:
    official_sentence = "(No official tournament run is committed, so there is no larger sample to compare with.)"
display(Markdown(f"""
### 9.1 Empirical findings (computed in this notebook)

1. **XGBoost reproduces the teacher almost perfectly on held-out states `[recomputed]`.** Masked accuracy {pct(xgb_acc, 2)} (95% hand-level CI [{pct(1 - err_ci['XGBoost'][2], 2)}, {pct(1 - err_ci['XGBoost'][1], 2)}]), {int(row_error['XGBoost'].sum())} errors in {N_VAL:,} decisions, log-loss {M_XGB['log_loss']:.4f}; the EBM on the same states {pct(ebm_acc, 2)} and {M_EBM['log_loss']:.4f}
   ({100 * err_diff_ci[0]:.2f} pp more errors, CI [{100 * err_diff_ci[1]:.2f}, {100 * err_diff_ci[2]:.2f}]). On fresh states the model itself visits it agrees on {pct(meta['imitation_on_policy']['masked_accuracy'], 2)} (EBM {pct(ebm_meta['imitation_on_policy']['masked_accuracy'], 2)}). {pct(share_near, 0)} of its errors lie within 0.01 of a threshold of the rule that decided
   (only {pct(near.mean(), 1)} of all decisions do), they are concentrated in the price-versus-adjusted-equity rules (`{top_rules.index[0]}`, `{top_rules.index[1]}`), and when it is wrong it is unsure (mean confidence {pct(calibration['XGBoost']['conf_wrong'], 0)} against {pct(calibration['XGBoost']['conf_right'], 1)} when right).
2. **What mattered in training `[saved]`.** Among model settings only the histogram resolution (`max_bin` 256 vs 1024: {bins.loc['max_bin=256', 'd_acc_pp']:+.2f} pp, log-loss x{bins.loc['max_bin=256', 'log_loss'] / base['log_loss']:.1f}) and the amount of data ({more_rows['d_acc_pp']:+.3f} pp with all behaviour-cloning rows) moved imitation noticeably; depth, learning rate, regularisation and sub-sampling stay within {rest['d_acc_pp'].abs().max():.2f} pp.
   DAgger shrank the covariate-shift gap from {100 * shift_bc:.2f} to {100 * shift_last:.2f} pp, but the artifacts cannot separate that from the effect of {last['n_train_rows'] - bc['n_train_rows']:,} extra rows, and {len(resolved)} of {3 * len(opps)} paired BB/100 comparisons with round 0 (legacy protocol, informational) are resolved; the candidates are near-tied, so the choice of round 3 is not evidence that it is the best one.
3. **Imitation accuracy does not show up as a resolved strength difference `[fresh]`.** XGBoost imitates the teacher far more accurately than the EBM ({pct(xgb_acc, 2)} vs {pct(ebm_acc, 2)}), yet in play under the two-table duplicate protocol ({t_fresh.n_pairs} fresh pairs against the teacher, {head.n_pairs} head-to-head) the two are
   {'not distinguishable' if h_word == 'no significant difference' else 'distinguishable'}: against the teacher XGBoost scores {result_text(t_fresh)} BB/100 (it {phrase[verdict(t_fresh.ci_low, t_fresh.ci_high)]} the teacher) and the EBM {result_text(ebm_t_fresh)}; head-to-head XGBoost scores {result_text(head)} (conservative match-level interval [{head.match_ci_low:+.0f}, {head.match_ci_high:+.0f}]): XGBoost {phrase[h_word]} the EBM.
   Both clearly beat {', '.join(f'`{o}`' for o in EVAL_OPPONENTS[1:] if fresh['XGBoost', o].ci_low > 0 and fresh['EBM', o].ci_low > 0) or 'none of the exploitable baselines'} (XGBoost: {', '.join(f"{o} {result_text(fresh['XGBoost', o])}" for o in EVAL_OPPONENTS[1:])}). The paired XGBoost-minus-EBM intervals against the four opponents {'all contain 0' if not others_marked else 'exclude 0 only for ' + ', '.join(f'`{o}` ({result_text(paired_vs_ebm[o])})' for o in others_marked)}
   (five comparisons were made, counting the head-to-head, so a lone marginal interval is weak evidence). Samples of this size resolve only differences of about {(head.ci_high - head.ci_low) / 2:.0f} BB/100 or more. {official_sentence}
   The saved validation BB/100 of the two models ({ci_text(t_saved['bb_per_100'], *t_saved['ci95_pairs'])} and {ci_text(ebm_saved_val['teacher']['bb_per_100'], *ebm_saved_val['teacher']['ci95_pairs'])} against the teacher) come from the legacy single-table protocol and are informational: replaying the same deals under that protocol moves the head-to-head by
   {legacy_gap:+.0f} BB/100 (section 4.4), which is why they are not used for any conclusion here. The teacher is the reference being copied, not a ceiling that was reached or surpassed.
4. **What SHAP says the model uses `[recomputed]`.** `equity` and `equity_minus_pot_odds` dominate every view (top two by mean |SHAP|; {pct(shap_share['equity'] + shap_share['equity_minus_pot_odds'], 0)} of the total), followed by `{ranking.loc[2, 'feature']}`, `{ranking.loc[3, 'feature']}` and `{ranking.loc[4, 'feature']}`. Gain, SHAP and permutation importance agree on the leaders (Spearman {rank_corr['gain', 'shap'][0]:.2f} / {rank_corr['gain', 'perm'][0]:.2f} / {rank_corr['shap', 'perm'][0]:.2f})
   but not on the tail, and not on how credit is shared between the redundant `equity`, `equity_minus_pot_odds` and `pot_odds`: shuffled together they cost {perm_groups['equity + equity_minus_pot_odds'][0] / (perm['equity'][0] + perm['equity_minus_pot_odds'][0]):.1f}x the sum of their separate costs; dropping `equity_minus_pot_odds` from a toy refit changes accuracy by {100 * (acc_without - acc_all):+.2f} pp while `equity`'s mean |SHAP| rises from {imp_all['equity']:.1f} to {imp_without['equity']:.1f}.
   Interventional instead of path-dependent SHAP changes magnitudes substantially (`spr` x{independent[FEATURE_INDEX['spr']] / path_dependent[FEATURE_INDEX['spr']]:.2f}, `equity` x{independent[FEATURE_INDEX['equity']] / path_dependent[FEATURE_INDEX['equity']]:.2f}) at a similar ranking (Spearman {rho_variants:.2f}), and the base value is not the average prediction (for `raise_pot`: {SH.base_values[0, 3]:+.2f} against a mean margin of {native_margin[:, 3].mean():+.2f}).
5. **The model is far from additive `[recomputed]`.** {pct(off_diagonal_mass, 0)} of the SHAP attribution mass is in interactions (largest pairs {top_pairs}); a toy booster restricted to depth 1 (additive) reaches {pct(acc[LIVE_FIT_DEPTHS[0]], 1)}, depth 2 {pct(acc[LIVE_FIT_DEPTHS[1]], 1)} and depth {LIVE_FIT_DEPTHS[-1]} {pct(acc[LIVE_FIT_DEPTHS[-1]], 1)} on {len(toy_train):,} rows.
   The same feature does different jobs per action: `spr` is {pct(focus['spr'][1], 0)} about `{focus['spr'][0]}`, `opp_aggression` {pct(focus['opp_aggression'][1], 0)} about `{focus['opp_aggression'][0]}`.
6. **Explanation fidelity is modest, and the top-1 test is nearly uninformative `[recomputed]`.** A constant answer 'equity' would be covered in {pct(constant_equity['family-aware (R_cf)'], 0)} of the states once twins count; XGBoost's strict top-1 hit rate is {pct(fid['XGBoost', MEASURES[0][0]]['score'], 0)} (null {pct(fid['XGBoost', MEASURES[0][0]]['null'], 0)}), mostly because SHAP names a *twin* of the teacher's feature
   (before the flop {pct(pre_percentile_named, 0)} of the top-1 features are `preflop_percentile`, the number the rules compare). The informative tests are the strict top-3 (XGBoost lift {lift_text('XGBoost', MEASURES[1][0])}, EBM {lift_text('EBM', MEASURES[1][0])}) and the first feature beyond hand strength
   (lift {lift_text('XGBoost', FINDINGS_FIDELITY_ROW)} vs {lift_text('EBM', FINDINGS_FIDELITY_ROW)}: {'the same for both' if diff_lift[FINDINGS_FIDELITY_ROW][1] <= 0 <= diff_lift[FINDINGS_FIDELITY_ROW][2] else 'different'}); on that test roughly half of the explanations point at a feature that can flip the decision.
7. **It copied the teacher's structure, not just its outputs `[recomputed]` `[live]`.** {pct(split_stats['preflop_percentile']['gain_share'], 0)} of the gain on `preflop_percentile` sits within 0.01 of one of the nine teacher thresholds (windows covering {pct(split_stats['preflop_percentile']['axis'], 0)} of the axis; the low thresholds are placed on the twin `equity` instead). Replaying 169 starting hands in {len(ladder_agreement)} real preflop contexts, the model matches the teacher's action in {pct(total_agreement, 2)}
   and switches at the *same hand class* as the teacher in {ladder_same} of {ladder_total} (threshold, template) cases. Post-flop, the check-to-bet switch follows the teacher's moving value line with correlation {value_corr:.3f} (mean abs. error {scan_table.loc[0, 'mean abs error']:.4f} equity points) and the call/fold boundary with a mean abs. error of {scan_table.loc[1, 'mean abs error']:.4f} ({scan_table.loc[1, 'within one grid step']} within one grid step).
8. **Compared with the EBM `[recomputed]` `[live]`.** The two agents choose the same action in {pct(same.mean(), 2)} of the validation states; where they differ, XGBoost matches the teacher in {pct(float((xgb_right & disagree).sum() / disagree.sum()), 0)}. XGBoost's file is {meta['model_file_bytes'] / ebm_meta['model_file_bytes']:.1f}x larger and its local explanation takes {live['XGBoost'][1]:.1f} ms against {live['EBM'][1]:.2f} ms
   (a prediction about {live['XGBoost'][0]:.2f} vs {live['EBM'][0]:.2f} ms). The global importance rankings correlate moderately (Spearman {rank_rows[0]['Spearman']:.2f}) and the two models name the same top-1 feature for the same decision in {pct(same_top1, 0)} of the cases ({pct(family_same, 0)} counting redundant twins as equal).
"""))

# %% [markdown]
# ### 9.2 Assumptions (taken as given, not tested here)
#
# 1. **The teacher is a heuristic reference, not a solver and not a ceiling.** Agreement with it says "copies this rule set"; it says nothing about how far the rule set is from good poker. Equity *against a random hand* (the backbone of the rules) is not equity against a betting range.
# 2. **One game, one abstraction.** Heads-up no-limit with blinds 5/10 and 100 big blinds reset every hand (the dataset has a few short-stack episodes), five abstract actions (fold, check/call, half pot, pot, all-in), no other bet sizes.
# 3. **The validation split is representative of the states that matter.** It is hand-disjoint from training but produced by the same behaviour and opponent mixture; the on-policy states of section 3.4 and the fresh duplicate-deal games probe states outside it, within the same opponent families.
# 4. **Duplicate pairs are treated as independent units** in the fresh BB/100 intervals. The two-table protocol removes the leak between the two plays of a deal, but adaptive agents (the teacher, and both learned agents through their opponent-statistics features) still drift *within* a match, which makes consecutive pairs of a match slightly dependent; this is why the conservative match-level intervals are shown next to the headline results. The saved training-time BB/100 are *not* covered by this assumption: they come from the legacy protocol, whose second-mover bias is a known artifact.
# 5. **TreeSHAP with the default (path-dependent) expectation is the explanation under study**; section 5.7 shows the interventional alternative, not as a replacement.
# 6. **Counterfactual scans and the fidelity sets (R_cf) use the teacher as an oracle** on feature vectors that are not always on the data manifold; their grids (deciles, 0.005 equity steps) are choices.
#
# ### 9.3 Speculation (hypotheses the data suggest, not tested)
#
# 1. **Could a better copy be a weaker player? (finding 3).** The fresh head-to-head of 4.4 resolves only what its interval shows; the much larger official tournament of Notebook 4 is the better place to look (4.4 reads its XGBoost-vs-EBM pairing when the run is committed).
#    If the EBM is ahead there, one candidate explanation is that the two models differ most on the rules where the teacher *bluffs* with weak hands (`cbet_bluff`, `semi_bluff_bet`: by far the highest disagreement rates, section 7.1): a model that cannot reproduce those bluffs plays a *less* bluffy version of the teacher, and those bluffs may not pay against near-copies of the same rule set.
#    Nothing in this notebook tests that. An experiment would play the teacher against a variant with its bluff rules switched off (`TeacherParams`) under the two-table protocol.
# 2. **SHAP names `equity_minus_pot_odds` rather than `equity` because the trees found it first.** Boosting picks the split with the largest gain and the two features are interchangeable in most states; which one wins at a given node may be an accident of the order of the search. Re-fitting with different seeds or feature orders, or removing the derived feature, would test it (the toy refit in 5.2 is one data point).
# 3. **Grouped attributions would raise the explanation fidelity.** Summing SHAP values over redundancy families (hand strength, price of the call, stack depth, raise count) before ranking would remove most of the twin misses of finding 6. Not run here.
# 4. **DAgger helps mainly because it adds data.** Finding 2 is compatible with "more rows" explaining the whole gain; an equal-size plain behaviour-cloning model evaluated on-policy would decide it.
# 5. **The residual 0.2% of errors are a resolution limit.** They sit next to thresholds; more rows near those thresholds (or a finer `max_bin`, as the sweep suggests) would shrink them. Whether the remaining errors matter for play is not shown: the fresh results against the teacher do not resolve a loss.
#
# ### 9.4 Limitations
#
# * **Imitation, not skill.** Nothing here shows that the agent is good at poker; the BB/100 results are against four fixed opponents on one game, with wide intervals, and against exploitable baselines they mostly measure how hard a weakness is punished.
# * **A single training run and seed** (42): no estimate of run-to-run variability of the model, the hyper-parameter sweep (one fit per row), the DAgger round choice or the SHAP rankings. The fits are deterministic (no sub-sampling), so a re-run reproduces them, but a different seed for the data would not be the same experiment.
# * **Hyper-parameters and the tree count were chosen on the validation split**, so the dataset-validation metrics are slightly optimistic; the on-policy and duplicate-deal numbers are independent of that choice but were used to *select the DAgger round*.
# * **Equity vs a random hand and a 5-action abstraction** limit what is learnable: the model copies a policy that ignores bet-size nuances and range reasoning.
# * **Fixed stack depth** (about 100 big blinds, plus some short stacks) and a fixed pair of blinds: no claim about other depths or tournaments.
# * **SHAP is a model of the model.** Additive attribution of a non-additive function; the split of credit between correlated features is not unique (sections 5.2, 5.7); the base value is not the average prediction; SHAP values are not causal effects, and they explain the *copy*, not poker.
# * **The fidelity study measures route alignment with the teacher, not SHAP's faithfulness to the model** (which is exact); R_cf and the redundancy families are conservative, hand-defined choices, and the study uses a sample of states (`FIDELITY_STATES`) of one split.
# * **Artifact detail.** The saved `policy_validation` and `dagger[*]` BB/100 tables of both models come from the legacy single-table protocol (`evaluation_protocol_note`); they are informational, and the EBM and XGBoost artifacts also use different validation pair counts (500 vs 1,000 per opponent), so they are not directly comparable (the fresh runs here use identical deals and the corrected protocol for both). `tuning.json` records a script configuration that differs from the final `config.json` in two settings that the sweep does not touch (section 3.2): it was produced by an earlier version of the training script on behaviour-cloning rows only, so its numbers are unaffected.
# * **The library's `TreeExplainer.expected_value` attribute does not satisfy additivity** for this model (it differs from the native base value, section 5.1); this notebook always uses the native base value.
#
# ### 9.5 Ideas for future work
#
# * **Outcome-based learning**: reinforcement learning or value learning on chips won (self-play, a league), so that the agent is no longer bounded by a teacher; keep the teacher as a held-out opponent and compare with the genetic agent (Notebook 3), which never saw it.
# * **Better opponent models**: the 6 opponent-statistics features are crude; a learned range model (what does this bet say about the cards) would address the equity-vs-random limitation directly.
# * **Richer actions and positions**: more bet sizes, variable stacks, and a check against a solver-derived policy as a second, independent reference.
# * **Explanations that respect redundancy**: grouped SHAP, interventional SHAP with a principled background, SHAP on a *refit without redundant features*, and a user study of whether people find waterfalls or EBM term lists more useful.
# * **Matched-size ablations**: plain behaviour cloning with the DAgger row count, several seeds, and a bluff-ablated teacher under the two-table protocol (speculations 1 and 4), and a retrained pipeline whose DAgger rounds are selected under the corrected protocol.
# * **Calibrated uncertainty**: the confidence of the model is informative (wrong decisions are the unsure ones); an agent that defers to a slower, exact rule evaluation when unsure is a cheap hybrid.

# %%
runtime = time.perf_counter() - NOTEBOOK_STARTED
figure_files = sorted((ARTIFACTS_DIR / "figures").glob("nb02_*.png"))
display(Markdown(
    f"Notebook run time: **{runtime / 60:.1f} min** on {N_WORKERS} worker processes / {N_THREADS} threads. Figures written to `artifacts/figures/` ({len(figure_files)} files, "
    f"{sum(f.stat().st_size for f in figure_files) / 1e6:.1f} MB): " + ", ".join(f"`{f.name}`" for f in figure_files) + "."
))
