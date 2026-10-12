# AI Poker Arena - Architecture & Contracts

This file is the shared contract between all workstreams. **Read it fully before writing code.**
Python >= 3.11, package `poker_arena` in `src/` (editable install in `.venv`). Always run Python via
`.venv/Scripts/python` (Windows, git-bash). Never `git commit`.

## 1. Goals (one paragraph)
Heads-up No-Limit Texas Hold'em with three independently trained AI architectures - **EBM**
(InterpretML `ExplainableBoostingClassifier`), **XGBoost** (+SHAP) and a **Genetic Algorithm**
evolved through gameplay - plus rule-based baselines, a reproducible headless tournament engine, four
research notebooks and a Streamlit app (AI vs AI, AI Championship, Human vs AI). It is a research /
education project about *explainable* decision making under uncertainty. Prediction accuracy is
**not** poker skill; poker outcomes are noisy; never claim superiority without confidence
intervals; no fabricated numbers - everything shown must come from saved experiment artifacts.

## 2. Package layout and ownership

```
src/poker_arena/
  cards.py                 [core]    card <-> int/str helpers (ints 0..51 = rank*4+suit, ranks 23456789TJQKA, suits cdhs)
  config.py                [core]    GameConfig(starting_stack, small_blind, big_blind), repo paths (ARTIFACTS_DIR, MODELS_DIR ...)
  engine/                  [core]    PokerKit-backed rules engine + information boundary (see 3)
    actions.py  deals.py  observation.py  game.py  history.py  opponent.py  session.py
  equity/                  [equity]  fast numpy hand evaluator, Monte-Carlo equity vs random hand, preflop table, hand/board descriptors
  agents/
    base.py registry.py baselines.py     [core]  Agent ABC, Decision, Explanation, ModelArtifactError, AgentSpec registry
    teacher.py                           [ml-data]  transparent equity/pot-odds rule policy ("TeacherAgent", kind "teacher")
    ebm_agent.py xgb_agent.py            [ml-models] trained-agent inference (kinds "ebm", "xgboost")
    ga_agent.py                          [ga]       evolved-strategy agent (kind "ga")
  features/                [ml-data]  feature schema + build_features(obs) shared by training AND inference
  data/                    [ml-data]  dataset generation (teacher-labelled decisions, DAgger aggregation)
  training/                [ml-models | ga]  library code: training/ebm.py, xgb.py (ml-models), ga.py (ga)
  tournament/              [tournament]  headless match/tournament runner, statistics, results IO/loader
  explain/                 [ml-models | ga | ui]  explanation helpers (e.g. EBM local explanation -> plotly figure)
  ui/                      [ui]      reusable Streamlit components (poker table renderer, charts)
app/                       [ui]      Streamlit entry `app/Home.py` + `app/pages/*.py`
scripts/                   training / dataset / tournament CLIs - config block at the top (see 6)
notebooks/                 [notebooks]  4 research notebooks (use package code + saved artifacts only)
artifacts/                 models/{ebm,xgb,ga}/  tournaments/<run_id>/  figures/  datasets/ (git-ignored)
tests/                     pytest suites
```
Each workstream edits ONLY files it owns. If you need a change in another workstream's file, make the
smallest possible change and mention it in your final report.

## 3. Engine, observations, and the anti-cheating boundary (MOST IMPORTANT RULE)

* `engine/game.py::HandEngine` wraps PokerKit (heads-up NLHE, deterministic dealing from a `Deal`).
  It is the **only** thing that holds hidden information (opponent cards, burn cards, undealt board).
  PokerKit is the rules authority; our wrapper adds one rule: *folding when a free check exists is not allowed*.
* **Agents, feature code, dataset generators, trainers and model code must NEVER import
  `poker_arena.engine.game`, `poker_arena.engine.history`, `poker_arena.engine.deals` or
  `poker_arena.engine.session`** (an automated AST test enforces this for `agents/`, `features/`,
  `training/`, `explain/`). They only see `engine.observation.Observation` (frozen, plain values).
  Orchestration code (`data/`, `tournament/`, `ui/`, `app/`, `scripts/`) may use `HandSession` / `play_hand`.
* `Observation` (see file) contains: street, is_button (HU: button = SB, acts first preflop / last postflop), own hole cards,
  revealed board, pot, stacks, bets, committed chips, blinds, `legal` (`LegalActions`), public action `history`
  (`PublicAction`, `by_me` flag), and `opponent_stats` (`OpponentStats`, smoothed VPIP/PFR/aggression/fold-to-bet/
  showdown rate from *previous* hands' public actions). No seeds, no deck, no opponent cards.
  Derived helpers: `obs.to_call`, `obs.num_raises`, `obs.effective_stack`, `obs.abstract_mask`.
* `HandSession` (engine/session.py) is where agents meet the engine: builds obs -> `agent.act(obs)` -> applies -> at hand end
  calls `agent.end_hand(PublicHandSummary)` and updates `OpponentTracker`s. Seats may be `None` for a human.
* `play_hand(config, deal, button_seat, agents, ...) -> HandRecord` plays a complete hand. `HandRecord` is JSON-serialisable
  and replayable (`record.replay()` yields `SpectatorView`s - these contain hidden cards, UI/analysis only).
* **Two-table duplicate deals**: a `Deal` ties cards to *positions*. Each deal is played twice, at two INDEPENDENT tables
  (own agent copies, own `OpponentTracker`s; `poker_arena.tournament.duplicate.play_duplicate_deals`), with the two
  players swapped between button and big blind. Card luck cancels exactly (identical agents sum to 0 on every deal)
  and no information about a deal can cross between its two plays. The earlier single-table protocol (back-to-back
  plays with shared trackers) leaked the first play to the second and is kept only as `protocol="single_table"`
  for diagnostics - see docs/METHODOLOGY.md section 5.
* Seeds: use `poker_arena.config.derive_seed(namespace, *parts)` (namespaces: dataset_train, dataset_val, dagger, ga_fitness, ga_validation, eval, ui, test)
  so training / validation / evaluation randomness can never overlap. `Deal.from_seed(int)`; agents get a private RNG through `agent.reset(seed)`. All simulation must be deterministic
  given (config, seeds, agent artifacts).

## 4. Action space
Concrete `Action`: `FOLD`, `CHECK_CALL`, `RAISE_TO(total street bet)`. Discretised `AbstractAction` (IntEnum):
`FOLD=0, CHECK_CALL=1, RAISE_SMALL=2 (~1/2 pot), RAISE_POT=3 (~pot), ALL_IN=4`. `obs.abstract_mask` is a length-5 bool
numpy array of legal abstract actions; distinct abstract actions always map to distinct concrete amounts (duplicates
removed in the order ALL_IN > RAISE_SMALL > RAISE_POT; short stacks / min-raise clamps handled in `engine/actions.py`).
`Decision.from_abstract(obs, abstract, explanation)` builds a legal concrete action. A trained model must only ever
choose from `obs.abstract_mask` (mask the probabilities/logits - never fall back to random, never return an illegal action).

## 5. Agent interface (`agents/base.py`)
`Agent.act(obs) -> Decision`, `reset(seed)`, `end_hand(summary)`, `explain(obs, decision) -> Explanation|None`,
`describe()`, attributes `name`, `architecture`, `description`, `supports_explanations`.
Agents are built through the registry: `create_agent(AgentSpec(kind, params, label))` or `create_agent("ebm")`.
Kinds: `random`, `call_station`, `maniac` (core), `teacher` (ml-data), `ebm`, `xgboost` (ml-models), `ga` (ga).
Trained agents expose a classmethod `load(model_dir: Path | None = None, **kw)` (default dir `artifacts/models/<subdir>/`)
and raise `ModelArtifactError` (clear message: which file is missing / which version mismatch / how to train) - never fall back silently.
**Copy semantics:** `reset(seed)` must *rebind* every mutable per-match field (RNG, memory, caches) to fresh objects, and the loaded
model object is treated as read-only - so `copy.copy(agent)` + `reset()` yields an independent agent that shares only the heavy,
immutable model (the Streamlit app caches loaded models and gives every browser session its own agent copies).
Hot-path rule: `act()` must be fast (target < 5 ms typical). Explanations (SHAP, local EBM) are computed lazily in `explain()`.

## 6. Training-script convention (user requirement - very important)
Every script in `scripts/` (`generate_dataset.py`, `train_ebm.py`, `train_xgboost.py`, `train_ga.py`, `run_tournament.py`) starts
with imports, then IMMEDIATELY a block like:

```python
# ==========================================
# TRAINING CONFIGURATION
# ==========================================
RANDOM_SEED = 42
# Game
STARTING_STACK = 1000
SMALL_BLIND = 5
BIG_BLIND = 10
# ... every important parameter of THIS script, with a one-line comment each ...
SMOKE_OVERRIDES = {"NUM_TRAINING_HANDS": 300, ...}   # applied by --smoke (runs in ~1-2 min)
```
Rules: no magic numbers elsewhere in the file; `--smoke` applies `SMOKE_OVERRIDES`; `--set KEY=VALUE` (repeatable) overrides any
config variable from the CLI; the script writes the *effective* configuration, seeds, library versions and metrics to
`artifacts/models/<x>/config.json` / `metadata.json`. The heavy lifting lives in package modules (`poker_arena.training.*`);
scripts only read the config block, call the library, print a summary. Notebooks call the same library functions or load saved artifacts.

## 7. Artifact contracts
* Models: `artifacts/models/ebm/`, `artifacts/models/xgb/`, `artifacts/models/ga/` each with the model file(s),
  `metadata.json` (schema_version, feature_names, training config, seeds, library versions, training metrics, created_at,
  game config used) and `config.json`. Small enough to commit (< ~15 MB each). Agent `load()` validates schema_version + feature list.
* Datasets (`artifacts/datasets/`, git-ignored, regenerable): `parquet` or `npz` with feature matrix, label, legal mask, hand id/split
  (split by HAND id - never by decision - to avoid leakage between train/validation).
* Tournaments: `artifacts/tournaments/<run_id>/` written by `poker_arena.tournament` (the tournament workstream defines
  `tournament/results.py` with `load_results(run_dir) -> TournamentResults` offering DataFrame accessors; the UI and the notebooks
  use ONLY that loader). `artifacts/tournaments/latest.json` points to the newest "official" run. Replayable sample hands are
  stored as `HandRecord.to_dict()` JSON lines.

## 8. Research principles to keep visible in docs / UI text
1. Imitation accuracy != poker skill. 2. Poker is high variance: report BB/100 with confidence intervals, use duplicate deals.
3. Different explainability: EBM additive shape functions, XGBoost SHAP, GA readable genes. 4. Equity (vs a random hand) is not
profitability. 5. Legal != good. 6. Same observation interface for everybody. 7. Reproducibility: save seeds + configs.
8. No unsupported claims.

## 9. Style
Type hints, small modules, docstrings that explain *why*, no dead code, no premature abstraction, named constants instead of magic numbers.
pytest tests per module under `tests/` (`test_<module>.py`). Mark slow tests with `@pytest.mark.slow`.

## 10. Research notebooks (authoring + build)
* **Source of truth:** `notebooks/src/<NN_name>.py` in py:percent format (`# %%` code cells, `# %% [markdown]` cells whose lines are
  prefixed `# `). The `.ipynb` in `notebooks/` is generated (executed outputs embedded) - never edit it by hand.
* **Build:** `.venv/Scripts/python scripts/build_notebooks.py [--execute] [--timeout SEC] [names...]` (names = stems, default all).
  `--execute` runs every cell in a fresh kernel of the venv interpreter with the repo root as working directory and fails loudly
  (cell source + traceback) on the first error; a failed build never overwrites an existing `.ipynb`. Outputs above `--max-mb` are refused.
* **Conventions** (checked by `tests/test_notebooks_build.py`): the first markdown cell has a `Reproduce:` line; the first code cell is a
  clearly marked `NOTEBOOK CONFIGURATION` block with the few adjustable numbers (seeds, pairs, workers); notebooks use package code + saved
  artifacts only, every number shown is computed or read (quote numbers through f-strings / `display(Markdown(...))`), seeded fresh
  evaluations use the `eval` seed namespace with a seed different from training, and results are reported with CIs.
* **Plots:** `poker_arena.explain.notebook_plots` (dark felt style, palette validated for the dark surface, `apply_style()`,
  `savefig(name)` -> `artifacts/figures/<name>.png`, `bar_with_ci`, `forest_plot`, `title`, `footnote`; optional `use_plotly`/`style_plotly`).
  Static matplotlib is the default (GitHub renders it); name figure files `nb<NN>_<topic>` and keep embedded images at moderate DPI.
