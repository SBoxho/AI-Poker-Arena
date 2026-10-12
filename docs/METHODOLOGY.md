# Research methodology, decisions and limitations

This document records *why* the experiments are designed the way they are, what each experiment does and does not demonstrate, and the main design decisions (including the ones we tested and found to make little difference). Numbers quoted here come from saved artifacts (`artifacts/models/*/metadata.json`, `artifacts/tournaments/<run>/`); the notebooks recompute them from those artifacts.

## 1. Research questions

1. Which approaches learn effective heads-up No-Limit Hold'em strategies on a laptop budget?
2. How do the learned strategies differ in *style* and in *performance*?
3. Can their decisions be explained, and does the explanation match what the policy actually does?

The project is deliberately **not** a claim of superhuman play: all opponents come from a small pool, the action space is abstracted, and nothing here measures distance from equilibrium.

## 2. Game, engine and fairness

* Heads-up No-Limit Texas Hold'em, simulated chips only. [PokerKit](https://github.com/uoftcprg/pokerkit) enforces the rules (blinds, button, betting order, minimum raises, all-ins, side pots, showdown, settlement). Our wrapper adds deterministic dealing from a pre-set 12-card `Deal`, the observation boundary, and one convention: *folding when a free check exists is not allowed* (a dominated action that only adds noise).
* Default depth: 1,000 chips = 100 big blinds, blinds 5/10, **stacks reset to the starting stack every hand** (cash-game evaluation at a fixed depth). Everything is configurable, but the models were trained at this depth only.
* **Information boundary.** Agents receive only an `Observation`: own hole cards, revealed board, pot, stacks, bets, blinds, position, legal actions, the public action history of the hand, and smoothed opponent statistics (VPIP, PFR, postflop aggression, fold-to-bet, showdown rate) computed *only* from previous hands' public actions. Dataset generation, GA fitness evaluation, tournaments and live play all go through the same `HandSession`, so there is exactly one code path that builds observations. Tests (differential, card-universe, pickle audit, AST import rules, negative controls) guard it; see README.
* **Action abstraction.** `fold`, `check/call`, `raise ≈ ½ pot`, `raise ≈ pot`, `all-in`. The mapping to concrete raise-to amounts respects minimum raises and stack limits, removes duplicates deterministically (ALL_IN > ½-pot > pot) and the engine rejects anything illegal. A trained model chooses the arg-max *among legal actions* (probabilities are masked and renormalised) — it never falls back to random play and never emits an illegal action.
* **Two-table duplicate deals.** A `Deal` ties cards to *positions*. Every deal is played twice, at two independent *tables*, with the two agents swapped between button and big blind. Each table has its own agent copies and its own opponent-statistics trackers, so nothing learned at one table about how a deal was played can reach the other table (exactly like real duplicate poker). Two identical agents therefore sum to exactly zero on every deal and the luck of the cards cancels. (This replaced an earlier protocol in which the two plays of a deal were played back to back by the same agents with shared trackers - see section 5, *Protocol history*.)

## 3. The supervised pipeline (EBM, XGBoost)

### 3.1 Labels come from a transparent teacher
Learning to play from scratch with supervised models needs a target. We built an explicit reference policy (`agents/teacher.py`): a **deterministic function of the 45-feature vector and the legal mask** (equity against a random hand, preflop percentile, pot odds, SPR, position, draw features, a bet-size/aggression-dependent range correction, opponent-statistic adjustments). All thresholds live in one documented dataclass. Because it is a pure function of the features, a flexible model *can* reproduce it almost exactly — which is why the imitation accuracies are very high (EBM 97.9%, XGBoost 99.8% legal-masked agreement on 73,777 held-out validation decisions, majority-class baseline 51%) and why **imitation accuracy says nothing about poker skill**.

Teacher strength was measured with duplicate deals against the baselines and its own style presets (see `scripts/evaluate_teacher.py`, `artifacts/teacher/evaluation.json`): it beats Random, CallStation and Maniac by several hundred BB/100 and is roughly even with its tight preset and its mirror. It is a heuristic, not a solver.

### 3.2 Dataset
`scripts/generate_dataset.py` plays hands in which the acting seat follows a *behaviour mixture* (teacher 30%, ε-noisy teacher 20%, uniformly random 10%, four style presets 10% each) against an opponent mixture (random, call-station, maniac, teacher and style presets), with match lengths from 30 to 1,000 hands so that opponent statistics take a range of values, and 10% of hands at random shorter stacks. Every visited decision state is labelled by the **canonical** teacher. This covers far more of the state space than the teacher's own play would. Train/validation are split **by hand id** with disjoint seed namespaces. The committed models use `teacher_bc_v2` (432,652 rows from ≈ 133k hands; label distribution fold 10.9%, check/call 51.0%, half-pot 27.3%, pot 9.5%, all-in 1.3%).

### 3.3 Class imbalance, legality, evaluation
* Sample weights `(n / (K · count))^0.5` for every model (power chosen on validation data; 0 = no weighting, 1 = fully balanced).
* Legal-action masking at inference and in all reported *masked* metrics (accuracy, macro-F1, balanced accuracy, per-class recall, confusion matrix, accuracy per street) plus log-loss and the probability mass placed on illegal actions.
* Both models use the *same* dataset, feature schema, sample weights, training-row budget, DAgger settings and validation protocol (150k behaviour-cloning rows + three DAgger rounds of 10k hands each).

### 3.4 Beyond pure imitation: the DAgger decision
The task asked us to investigate a practical alternative to pure imitation. Options considered:

| Option | Verdict |
|---|---|
| Behaviour cloning only | Baseline; suffers covariate shift (learner drifts into states the teacher-driven data under-represents). |
| **DAgger** (learner plays, teacher relabels visited states, aggregate, refit) | **Implemented** for both models (`training/common.py::run_dagger`): cheap, fits laptop budgets, directly targets covariate shift, and keeps the labels interpretable. |
| Outcome-based value regression (Monte-Carlo returns of actions) | Not chosen: poker returns are extremely noisy; reliable value estimates need orders of magnitude more simulated hands than a laptop budget allows. |
| Counterfactual-regret / solver methods | Out of scope for full-game NLHE at this budget; relevant future work for stronger baselines. |
| Deep RL self-play | Out of scope for the first version (planned architecture slot). |

Result, measured per round on held-out validation data (agreement on the dataset's validation split, agreement on fresh *on-policy* states, and duplicate-deal BB/100 against teacher/random/call-station/maniac with validation seeds only): DAgger shrinks the on-policy agreement gap for XGBoost (0.25 → 0.09 percentage points) and moves it by at most a few tenths of a point for the EBM, but **neither model's BB/100 changed measurably** (all paired differences against round 0 are within noise for XGBoost; for the EBM the round-to-round BB/100 swings are of the same size as the noise). We kept DAgger in the pipeline because it is the principled fix and costs little, but the data say it was not what mattered. Per-round tables are stored in each model's `metadata.json` under `dagger`; the rule for choosing the final round (best mean validation BB/100, ties to the earlier round) is near-tied and therefore subject to a winner's-curse bias, which is noted in the metadata.

## 4. The genetic algorithm (no teacher, no labels)

* **Genome:** 18 bounded real genes (`agents/ga_agent.py::GENE_SPECS`) map to a readable rule cascade over equity, pot odds, position, SPR and opponent statistics.
* **Fitness:** average profit (BB/100) in simulated matches against a diverse pool — Random, CallStation, Maniac and five hand-written style presets (nit, tag, lag, passive, maniac-ish) plus a *hall of fame* of earlier champions (mild co-evolution) — using duplicate pairs, **common random numbers within a generation** (all genomes face the same deals, giving low-variance comparisons) and **fresh deals each generation** (no overfitting to a fixed deal set). Per-opponent profits pass through a saturating transform and are blended with the worst-opponent profit, because an un-saturated fitness let genomes over-exploit weak bots and lose to a held-out style in exploratory runs.
* **Operators:** tournament selection, elitism, BLX-α/uniform crossover, Gaussian mutation with decaying step.
* **Held-out checks:** each generation's best genome is validated on separate deals against a validation set containing opponents *not* in the training pool; the **Teacher is never in the training pool** and is used only as a held-out opponent. Final selection re-evaluates the top candidates on fresh deals.
* Standard run: population 40, 40 generations, 3,000 hands per genome per generation; recorded per-generation fitness statistics, diversity and every gene's mean/std/best.

## 5. Tournament statistics

* Round-robin over all pairs; each pairing is one match of 40,000 hands = 20,000 duplicate deals = 200 **sessions** of 200 hands (two tables of 100 hands each, see section 2). At the start of each session every agent is rebuilt and reset and the opponent trackers are cleared, so sessions are independent and results do not depend on the number of worker processes.
* **Why sessions are the independent unit.** The learned agents and the teacher adapt to opponent statistics accumulated *within* a session, so consecutive pairs of one session are dependent. In the first official run, the pair-level interval for EBM vs. Teacher had half-width 14.0 BB/100 whereas the session-level interval had 24.6; a simulation with a session-shared random effect shows pair intervals covering the truth < 75% of the time and session intervals ≈ 95%. All headline intervals and verdicts therefore use 95% t-intervals over per-session BB/100. (Pair-level numbers are stored as `pair_*` columns.)
* **Protocol history (a bug we found and fixed).** The first version played the two hands of a deal back to back with shared agents and opponent trackers. Because the learned agents and the teacher adapt to opponent statistics, the second play of a deal was played against an opponent model that already contained the first play of the *same* deal, which gave the second mover a systematic edge: two identical Teachers (an A/A test whose true difference is zero) showed -13.0 +/- 3.7 BB/100 for the first mover over 800 sessions (z = -3.5; the tournament engineer measured -9.5 +/- 3.8 on other seeds). With opponent statistics switched off the pair cancelled exactly, which located the leak. Notebook 4's A/A section discovered it; it was replicated independently and fixed with the two-table protocol, which is now the default (identical Teachers sum to exactly zero on every deal). The legacy protocol survives as `protocol="single_table"` for diagnostics. Consequences: all tournament numbers in the README and notebooks use the corrected protocol; the training-time validation BB/100 stored in the EBM/XGBoost metadata (and the DAgger round selection that used it) and the GA's fitness during evolution used the legacy protocol. The DAgger candidates were near-tied, and the GA's held-out evaluation under the corrected protocol (vs. Teacher -106 [-155, -57]) agrees with the legacy-era figure (-96) within noise, so we kept those models; retraining the GA under the corrected protocol is listed as future work.
* "Significant" means the interval excludes zero — nothing more. The helper never calls one agent "better" otherwise. The leaderboard ranks by BB/100 pooled over opponents with equal weight per opponent and also reports the pooled-over-pairs variant.
* Evaluation seeds live in the `eval` namespace of `derive_seed`; training data uses `dataset_train`/`dataset_val`/`dagger`, the GA uses `ga_fitness`/`ga_validation`; Notebook 3 verifies that no evaluation deal coincides with a training or validation deal.
* Replays: `HandRecord` stores the deal, button, stacks and every action; `record.verify()` re-runs it through the real engine. The tournament keeps the first hands, the largest pots and each side's biggest wins per match.

## 6. Explainability: three kinds of insight

| Agent | Explanation | Exact? | Caveat |
|---|---|---|---|
| EBM | Sum of per-term (feature or pair) log-odds contributions; global shape functions | exact for the model | credit is shared among correlated features (e.g. equity, equity − pot odds, pot odds) |
| XGBoost | TreeSHAP values of the class margin | exact Shapley values for the tree model | additive attribution of a non-additive model; not causal |
| GA | Rule trace with the evolved gene values and the numbers that fired the rule | exact (the policy *is* the rule cascade) | rules reflect equity-vs-random thresholds, not opponent ranges |

Because the teacher's rules are known, the notebooks can test explanation *fidelity* against ground truth (does the feature the explanation credits belong to the rule that actually fired?) — a check that is normally impossible, with the standard caveat that correlated features make "the right feature" ambiguous.

## 7. Known limitations (read before quoting any result)

* **Imitation ceiling.** EBM and XGBoost copy a heuristic; their strengths and weaknesses are the teacher's (plus the copy errors). That the EBM *outperforms* its teacher head-to-head is an observation to be investigated (Notebook 4), not evidence of deeper understanding.
* **The teacher's opponent model is miscalibrated.** Its adjustments assume a neutral opponent aggression of 0.40 whereas competent agents show about 0.70 in these games, so it calls too lightly against them; Notebook 4 measured that a teacher that ignores opponent statistics beats the canonical teacher by roughly 150 BB/100. "Beats the teacher" is therefore a weaker statement than it sounds. Notebook 4 (section 9) tested why the EBM beats its teacher: the edge is absent in a static game where both agents ignore opponent statistics and absent against the simple bots, random-noise copies of the teacher do not reproduce it, and it appears whenever either side reads opponent statistics - an adaptation effect, consistent with the EBM being a slightly *damped* copy of a costly reaction (a hypothesis, not a tested mechanism). Replications on two new seeds pool to a smaller edge (about +28 BB/100) than the official run, so the official estimate is probably at the high end. The teacher was frozen before this was known because every label, model and evaluation depends on it; recalibrating it and regenerating everything is the obvious next experiment.
* **Opponent pool dependence.** All results are relative to this pool; no best-response or exploitability measurement exists, and no human-strength reference.
* **Equity vs. a random hand** is the core feature. It ignores opponent ranges, fold equity and future betting — equity is not profitability.
* **Abstractions:** five actions, heads-up only, fixed depth, no antes/straddles, all showdown hands are revealed, uncalled-bet portions are included in the visible pot (pot-odds computed from `pot` can be slightly off after an unmatched overbet).
* **Single training seed per model** and a modest GA budget; the GA's saved held-out evaluation is a point estimate with a CI, not a distribution over independent GA runs.
* **Statistics:** intervals quantify sampling noise of this experiment only; they do not cover model-selection noise, hyper-parameter tuning on validation data (tuning used validation metrics — the `eval` seeds were untouched), or distribution shift to other stack depths or blind levels.
* **The boundary is in-process.** It stops accidental leakage and honest agents; code in `act()` could inspect the Python call stack. Run untrusted agents (e.g. LLMs with tools) in a subprocess.
* **Performance:** PokerKit's showdown evaluation limits throughput; the tournament framework compensates with process-level parallelism.

## 8. How to extend the experiments responsibly

New agents (RL, LLM, specialised models) should be evaluated through the same tournament runner with new `eval` seeds, against the same pool, with session-level intervals and a replication on at least one further seed, and should report imitation-style metrics (if any) separately from tournament results.
