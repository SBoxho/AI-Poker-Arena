# EBM agent - training notes

An `ExplainableBoostingClassifier` (InterpretML 0.7.8) that **imitates the transparent equity teacher**
(`agents/teacher.py`) on the shared decision dataset `teacher_bc_v2` (432,652 labelled decisions, split by
hand id). Imitation accuracy is **not** poker skill: agreement with the teacher and duplicate-deal BB/100 are
measured separately and are reported separately below. Everything here is also stored in `metadata.json`.

Reproduce: `.venv/Scripts/python scripts/train_ebm.py` (about 14 minutes on 10 cores, wall time 846 s). The run
is deterministic: a second run reproduced every metric, BB/100, importance and the model file byte for byte (only
timings differ). `--smoke` takes ~40 s.

## The saved model

| | |
|---|---|
| file | `ebm.joblib` (3.7 MB, joblib zlib level 3), `metadata.json`, `config.json` |
| terms | 85: 45 shape functions (one per feature) + 40 pairwise interaction surfaces |
| training rows | 245,346 = 150,000 behaviour-cloning rows (whole hands, `MAX_TRAIN_ROWS`) + 3 x ~32k DAgger rows |
| hyper-parameters | 10 outer bags, learning rate 0.1, 300 rounds, 128 bins (24 inside pair tables), `max_leaves=2`, class-weight power 0.5 |
| selected round | 3 (best mean validation BB/100; rule: ties go to the earlier round) |
| latency (this machine) | `predict_proba` 0.6 ms median, `act()` incl. feature building 1.6 ms median / 2.8 ms p95, term decomposition 0.6 ms |

Training details that matter: rows are class-weighted (`(n / (K * count)) ** 0.5`: all-in decisions weigh 3.7x,
check/call 0.63x); early stopping uses **hand-level** hold-out bags taken from the training hands, so the
dataset's validation split never steers training and near-duplicate decisions of one hand cannot leak; binary
flags stay `continuous` (one cut at 0.5). Boosting never stopped early (the teacher's labels are noise free, so
the held-out loss keeps falling): `max_rounds` is the effective regulariser, and the shape functions are
correspondingly sharp (logit contributions up to about +-30).

## What was tried (validation split only; the tournament `eval` seeds were never used)

Behaviour-cloning fits (no DAgger), scored on the full validation split (73,777 decisions), legal-masked. They
were run with throw-away scripts that fit `ExplainableBoostingClassifier` exactly like `fit_ebm` does (seed 0);
the CLI equivalent of a row is e.g. `--set DAGGER_ROUNDS=0 --set MAX_TRAIN_ROWS=100000 --set EBM_INTERACTIONS=15
--set EBM_MAX_INTERACTION_BINS=64 --set EBM_MAX_BINS=256 --set EBM_OUTER_BAGS=8 --set EBM_LEARNING_RATE=0.05
--set EBM_MAX_ROUNDS=500 --set OUTPUT_DIR=...` (seed 42, so the digits differ a little).
One run per row (one seed): differences of a few tenths of a point are within run-to-run noise (the four
rounds of the final run differ by 0.5 points on this metric although they only differ by extra DAgger rows).
Times were measured on a shared machine (the XGBoost model trained at the same time; the 40-pair/64-bin run also
overlapped with another fit of mine), so treat them as +-30 %. The first two rows used InterpretML's default
row-wise early-stopping split.

| rows | pair terms (pair bins) | bins | lr x rounds | class-weight power | fit | masked acc. | macro-F1 | balanced acc. | log-loss | file |
|---|---|---|---|---|---|---|---|---|---|---|
| 40k | 0 | 256 | 0.03 x 300, 4 bags | 0.5 | 10 s | 0.9613 | 0.9487 | 0.9675 | 0.1253 | 0.6 MB |
| 359k | 0 | 256 | 0.03 x 1000, 4 bags | 0.5 | 343 s | 0.9676 | 0.9636 | 0.9753 | 0.1021 | 0.6 MB |
| 100k | 0 | 256 | 0.05 x 500, 8 bags | 0.5 | 77 s | 0.9669 | 0.9605 | 0.9736 | 0.1066 | 0.9 MB |
| 100k | 0, row-wise early-stopping split | 256 | 0.05 x 500, 8 bags | 0.5 | 72 s | 0.9670 | 0.9614 | 0.9741 | 0.1063 | 0.9 MB |
| 100k | 15 (64) | 256 | 0.05 x 500, 8 bags | 0.5 | 97 s | 0.9780 | 0.9702 | 0.9812 | 0.0708 | 6.2 MB |
| 100k | 40 (64) | 256 | 0.05 x 500, 8 bags | 0.5 | 141 s | 0.9826 | 0.9747 | 0.9834 | 0.0524 | 16.4 MB |
| 100k | 30 (32) | 128 | 0.05 x 500, 8 bags | 0.5 | 128 s | 0.9795 | 0.9715 | 0.9809 | 0.0604 | 5.5 MB |
| 100k | 30 (16) | 128 | 0.05 x 500, 8 bags | 0.5 | 109 s | 0.9783 | 0.9702 | 0.9790 | 0.0670 | 2.0 MB |
| 100k | 30 (32) | 128 | 0.05 x 500, 8 bags | 0 | 86 s | 0.9756 | 0.9696 | 0.9697 | 0.0742 | 2.5 MB |
| 100k | 30 (32) | 128 | 0.05 x 500, 8 bags | 1 | 91 s | 0.9734 | 0.9604 | 0.9819 | 0.0772 | 5.8 MB |
| 150k | 30 (32) | 128 | 0.1 x 300, 10 bags | 0.5 | 119 s | 0.9792 | 0.9705 | 0.9804 | 0.0594 | 5.6 MB |
| 150k | 30 (32) | 128 | 0.05 x 500, 10 bags | 0.5 | 175 s | 0.9795 | 0.9711 | 0.9810 | 0.0590 | 5.7 MB |
| **150k** | **40 (24)** | **128** | **0.1 x 300, 10 bags** | **0.5** | **133 s** | **0.9798** | **0.9719** | **0.9818** | **0.0572** | **4.0 MB** |

Decisions drawn from it:

* **Interactions on.** A purely additive model of single features plateaus at about 96.7 % agreement; 15
  pairwise terms reach 97.8 %, 30-40 reach 98.0-98.3 %. The teacher is a threshold rule on derived quantities
  (equity against pot odds, per street, per bet size, opponent statistics), which single-feature curves can
  only approximate. Pairs stay readable (heat maps in `explain/ebm.py`).
* **Size is driven by the pair tables**, not by the number of terms: 64 bins per feature made 40 pairs 16 MB
  (over the 15 MB budget), 24 bins gives 4 MB at the same accuracy. 128 main-effect bins lose nothing against 256.
* **More rows do not help an additive model much**: 40k -> 100k -> 359k rows give 96.1 -> 96.7 -> 96.8 %, so the
  behaviour-cloning part is capped at 150k rows (the same budget as the XGBoost agent).
* **Class-weight power 0.5** has the best accuracy, macro-F1 and log-loss; power 1 buys balanced accuracy
  (0.9819) at the price of precision on the frequent classes (macro-F1 0.9604), power 0 under-serves all-in/fold.
* **Learning rate 0.1 x 300 rounds** matches 0.05 x 500 at two thirds of the time.
* A row-wise or a hand-wise early-stopping split makes no measurable difference here (boosting never stops
  early); the hand-wise split is kept because it is the safe choice whenever labels carry noise.

## DAgger: did it help? (validation, final run; `dagger` table in `metadata.json`)

| round | rows | fit | dataset-val agreement | on-policy agreement | mean BB/100 | BB/100 vs teacher [95 % CI] | paired vs round 0, vs teacher |
|---|---|---|---|---|---|---|---|
| 0 (behaviour cloning) | 150,000 | 114 s | 0.9792 | 0.9781 | +761.5 | +106 [+45, +167] | |
| 1 (beta 0.5) | 180,458 | 146 s | 0.9807 | 0.9813 | +767.2 | +37 [-31, +105] | -69 [-124, -15] |
| 2 (beta 0.25) | 212,917 | 183 s | 0.9833 | 0.9812 | +728.0 | +15 [-46, +75] | -91 [-152, -31] |
| **3 (beta 0.125), selected** | 245,346 | 205 s | 0.9786 | 0.9792 | **+772.3** | +61 [+7, +114] | -46 [-98, +7] |

Validation BB/100 of the EBM per opponent, `[95 % CI over pairs]` (500 duplicate pairs = 1,000 hands each):

| round | teacher | call_station | maniac | random | mean |
|---|---|---|---|---|---|
| 0 | +106 [+45, +167] | +665 [+561, +769] | +1,634 [+1,377, +1,892] | +641 [+429, +852] | +761.5 |
| 1 | +37 [-31, +105] | +662 [+559, +766] | +1,643 [+1,386, +1,900] | +727 [+515, +938] | +767.2 |
| 2 | +15 [-46, +75] | +657 [+555, +759] | +1,655 [+1,396, +1,914] | +585 [+377, +794] | +728.0 |
| **3** | +61 [+7, +114] | +654 [+552, +755] | +1,652 [+1,396, +1,908] | +723 [+516, +930] | **+772.3** |

"On-policy agreement" is measured on fresh states visited by that round's own EBM (4,000 validation hands,
seeds no training step used); the duplicate-deal results use 500 pairs per opponent in the `dataset_val` seed
namespace (identical deals in every round, so rounds are compared on the same cards).

Observations (not claims of cause):

* The covariate shift of plain behaviour cloning is **small** for this teacher/dataset: the round-0 EBM agrees
  with the teacher on 97.8 % of its own states against 97.9 % on the dataset's validation states.
* DAgger moved on-policy agreement by at most +0.3 points (97.8 -> 98.1 %), not monotonically; the fluctuation
  of the dataset-validation agreement between rounds (97.9-98.3 %) is as large, i.e. within training noise.
* Against the three baselines BB/100 is unchanged within the confidence intervals (table below; for orientation the
  teacher scores +716, +1,347 and +706 against call_station, maniac and random in `artifacts/teacher/evaluation.json`,
  on different deals - not a paired comparison). Against the teacher the behaviour-cloning EBM is *ahead*
  (+106 BB/100, CI excludes 0) while the DAgger rounds sit closer to zero; the paired differences for rounds 1 and 2
  exclude 0. Why an imperfect copy beats its teacher was not investigated - it should not be read as "the EBM is
  stronger than the rule it imitates" without a fresh evaluation (adaptation to opponent statistics also
  makes consecutive pairs of one match dependent; `ci95_matches` in the metadata is the conservative interval).
* The final round was picked by the stated rule on a noisy mean (+772 vs +767 vs +762 vs +728 BB/100): treat the
  selection as a near tie, and note that the stored validation numbers of the winner are slightly optimistic.

## What the EBM learned (observations from the saved model)

* Mean |term score| (log-odds, averaged over the five actions, on 20,000 validation states): `equity` 5.1,
  `equity_minus_pot_odds` 3.7, `spr` 1.4, `board_straightness` 1.3, `board_max_suit_count` 1.1,
  `preflop_percentile x to_call_bb` 1.1, `facing_bet` 1.0. Interaction terms carry 30 % of the total importance.
* `spr` matters almost only for **all-in** (mean |score| 6.2 against about 0.2 for the other actions): the
  push/fold rule at short stacks. `facing_bet` matters almost only for **fold** (4.0 against <= 0.5), `can_raise`
  moves the three raise actions by 6-10 log-odds: legality is learned (probability on illegal actions is 8e-5 on
  average), yet the agent still masks.
* The raise_pot curve over `equity` runs from about -19 (equity 0.1) through -8 (0.5) to +18 (0.7) and +23 (0.9);
  fold falls from +7 to -32 - sharp, threshold-like steps, as in the teacher's rules.
* Opponent statistics matter mainly **through interactions with hand strength**: the main effects of
  `opp_vpip`, `opp_log_hands`, `opp_pfr`, `opp_fold_to_bet` are among the weakest terms (0.02-0.13), while
  `equity x opp_fold_to_bet` (raise_pot 2.3), `equity_minus_pot_odds x opp_fold_to_bet` (fold 2.0),
  `equity x opp_aggression` and `spr x opp_aggression` are among the strongest pairs - the teacher's documented
  rules slide value-bet and call thresholds with exactly those statistics.
* `board_straightness` and `board_max_suit_count` rank high although they are not hand strength: the teacher
  has an explicit wet-board rule (3 suited board cards / 3 connected ranks), and the EBM reproduces it.

Caveat when reading shape functions: `equity`, `equity_minus_pot_odds`, `pot_odds` and `preflop_percentile` are
strongly correlated, so an EBM spreads their joint effect over several curves (large opposite-signed values are
normal). Only the *sum* of the terms is identified - read related curves together. For a multi-class model
InterpretML's `explain_global()` drops the pairwise terms; use `poker_arena.explain.ebm` for them.

## Reading the model in a notebook

```python
from poker_arena.agents.ebm_agent import EBMAgent
from poker_arena.explain import ebm as ebm_explain

agent = EBMAgent.load()                                   # artifacts/models/ebm/
ebm_explain.importance_figure(ebm_explain.importance_frame(agent.metadata))
ebm_explain.shape_figure(agent, "equity")                 # one curve per action, with a data-density strip
ebm_explain.interaction_figure(agent, "equity x opp_fold_to_bet", "raise_pot")
ebm_explain.local_explanation_figure(agent, x, "check_call")   # x = build_features(obs)
# InterpretML's own widgets work on agent.model: show(agent.model.explain_global()); for multi-class it drops the pairs
# (one UserWarning per pair) - the interaction functions above draw them
```

## Known noise

On Windows, joblib's loky resource tracker may print a harmless `KeyError: ...joblib_memmapping_folder...`
traceback on stderr after an EBM fit; it does not affect the result.
