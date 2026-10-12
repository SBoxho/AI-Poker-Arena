"""The evolved poker strategy: a vector of readable genes and the agent that plays it.

Why this design
---------------
The genetic algorithm (``training/ga.py`` + ``evolution/``) searches over a *small, named* parameter
vector - the :class:`Genome` - instead of an opaque network, so that the evolved result can be read
like a strategy guide ("open-raise the button with equity above 0.47", "bluff 14% of the time when
checked to").  Every gene has bounds and a plain-English description (:data:`GENE_SPECS`), and every
decision of :class:`GeneticAgent` explains which rules fired with the gene values and the numbers
involved.

The strategy is built from three cheap quantities: *equity against a random hand* (an O(1) table
lookup preflop, a small Monte-Carlo estimate postflop - the same scale the whole project uses), pot
odds and the stack-to-pot ratio.  Equity against a random hand is *not* the equity against a betting
range, so several genes are deliberate corrections for that (call margin, fold-to-big-bet, deep-stack
caution) that evolution is free to set.

Anti-cheating: the agent only ever reads the :class:`~poker_arena.engine.observation.Observation`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from poker_arena.agents.base import Agent, Decision, Explanation, ModelArtifactError
from poker_arena.config import MODELS_DIR
from poker_arena.engine.actions import AbstractAction
from poker_arena.engine.observation import Observation, OpponentStats, Street
from poker_arena.engine.opponent import PublicHandSummary
from poker_arena.equity import draw_features, equity_vs_random, preflop_class, preflop_equity

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
GA_SCHEMA_VERSION = 1
MODEL_SUBDIR = "ga"
BEST_GENOME_FILE = "best_genome.json"
METADATA_FILE = "metadata.json"
TRAIN_COMMAND = ".venv/Scripts/python scripts/train_ga.py"

#: Monte-Carlo deals per postflop equity estimate. Standard error ~0.045 at equity 0.5. Evolved
#: genomes are trained with the same sample count they are deployed with (stored in metadata).
DEFAULT_EQUITY_SAMPLES = 120
#: Stack-to-pot ratio at/above which the deep-stack caution gene applies in full.
SPR_CAP = 8.0
#: Largest opponent-bet size (as a fraction of the pot before the bet) counted by ``fold_big_bet``.
BET_FRACTION_CAP = 2.0
#: Outs (flush draw = 9, open-ended straight draw = 8) needed to count a hand as a semi-bluff draw.
SEMI_BLUFF_MIN_OUTS = 8
FLUSH_DRAW_OUTS = 9
#: Neutral opponent profile the adaptation genes measure deviations from.
_NEUTRAL_STATS = OpponentStats()

_SIZE_LABEL = {
    AbstractAction.RAISE_SMALL: "1/2 pot",
    AbstractAction.RAISE_POT: "pot-size",
    AbstractAction.ALL_IN: "all-in",
}
_STREET_LABEL = {Street.PREFLOP: "preflop", Street.FLOP: "flop", Street.TURN: "turn", Street.RIVER: "river"}


# --------------------------------------------------------------------------------------
# Genome
# --------------------------------------------------------------------------------------
def _gene(default: float, low: float, high: float, group: str, doc: str) -> Any:
    return field(default=default, metadata={"low": low, "high": high, "group": group, "doc": doc})


@dataclass(frozen=True, slots=True)
class Genome:
    """Eighteen bounded real-valued genes that fully define a strategy.

    Thresholds are expressed as *equity against a random hand* (0.5 = average hand, 0.85 = pocket
    aces preflop). The defaults form a sensible tight-aggressive baseline; evolution moves them.
    Construction validates the bounds; use :meth:`from_vector` with ``clip=True`` to project a
    mutated vector back into range.
    """

    # ---- preflop -----------------------------------------------------------------
    pf_play_btn: float = _gene(
        0.42, 0.33, 0.65, "preflop",
        "Button (first to act): minimum equity to enter the pot; weaker hands fold.",
    )
    pf_open_raise: float = _gene(
        0.50, 0.38, 0.80, "preflop",
        "Button: equity at/above which it raises first-in; between pf_play_btn and this it limps.",
    )
    pf_bb_raise: float = _gene(
        0.58, 0.40, 0.85, "preflop",
        "Big blind after a limp: raise (isolate) with at least this equity, otherwise check.",
    )
    pf_defend: float = _gene(
        0.50, 0.33, 0.80, "preflop",
        "Minimum equity to call a preflop raise (the pot-odds requirement must be met as well).",
    )
    pf_3bet: float = _gene(
        0.68, 0.50, 0.95, "preflop",
        "Equity at/above which the agent re-raises (3-bets) a raise.",
    )
    pf_shove: float = _gene(
        0.80, 0.62, 0.95, "preflop",
        "Equity at/above which the agent jams all-in over a preflop raise (4-bet/shove).",
    )
    # ---- postflop value ----------------------------------------------------------
    value_bet: float = _gene(
        0.62, 0.50, 0.92, "postflop",
        "Bet for value when checked to with adjusted equity at/above this.",
    )
    value_raise: float = _gene(
        0.78, 0.55, 0.98, "postflop",
        "Raise a bet with adjusted equity at/above this.",
    )
    shove_threshold: float = _gene(
        0.92, 0.70, 1.00, "postflop",
        "Go all-in postflop with adjusted equity at/above this (1.0 = practically never).",
    )
    # ---- postflop defence --------------------------------------------------------
    call_margin: float = _gene(
        0.03, -0.10, 0.20, "defence",
        "Equity required to call = pot odds + this margin (negative = calls light, positive = tight).",
    )
    fold_big_bet: float = _gene(
        0.08, 0.0, 0.30, "defence",
        "Extra equity required per pot-size of the bet faced: big bets show a stronger range than "
        "'random hand' equity assumes.",
    )
    deep_stack_caution: float = _gene(
        0.04, 0.0, 0.15, "defence",
        "Equity subtracted when stacks are deep relative to the pot (scaled by SPR up to "
        f"{SPR_CAP:g}): more chips behind = more to lose.",
    )
    position_bonus: float = _gene(
        0.03, -0.05, 0.12, "defence",
        "Equity added when in position (button postflop) and subtracted out of position.",
    )
    # ---- aggression / mixing -----------------------------------------------------
    bluff_freq: float = _gene(
        0.12, 0.0, 0.70, "aggression",
        "Probability of betting a weak hand when checked to (pure bluff).",
    )
    semi_bluff_freq: float = _gene(
        0.50, 0.0, 1.0, "aggression",
        "Probability of betting/raising a flush or open-ended straight draw instead of calling.",
    )
    big_bet_prob: float = _gene(
        0.35, 0.0, 1.0, "aggression",
        "Probability that a bet or raise is pot-sized instead of half-pot.",
    )
    # ---- opponent adaptation -----------------------------------------------------
    adapt_fold: float = _gene(
        0.30, 0.0, 1.0, "adaptation",
        "Bluff/semi-bluff frequency added per +1.0 of opponent fold-to-bet above neutral "
        "(bluff more against folders, less against callers).",
    )
    adapt_aggression: float = _gene(
        0.0, -0.40, 0.40, "adaptation",
        "Call margin added per +1.0 of opponent aggression above neutral "
        "(>0 tightens against aggressive opponents, <0 calls them down lighter).",
    )

    def __post_init__(self) -> None:
        for spec in GENE_SPECS:
            value = getattr(self, spec.name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
                raise ValueError(f"Gene {spec.name} must be a real number, got {value!r}")
            value = float(value)
            if not math.isfinite(value) or not spec.low <= value <= spec.high:
                raise ValueError(f"Gene {spec.name}={value!r} outside its bounds [{spec.low}, {spec.high}]")
            object.__setattr__(self, spec.name, value)

    # ---- conversions -------------------------------------------------------------
    def to_vector(self) -> np.ndarray:
        """Genes as a float64 array in :data:`GENE_NAMES` order (what the GA operators work on)."""
        return np.array([getattr(self, name) for name in GENE_NAMES], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in GENE_NAMES}

    @classmethod
    def from_vector(cls, values: Sequence[float] | np.ndarray, *, clip: bool = False) -> "Genome":
        """Build a genome from a vector in :data:`GENE_NAMES` order; ``clip=True`` projects into bounds."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(GENE_NAMES),):
            raise ValueError(f"Expected {len(GENE_NAMES)} gene values, got shape {array.shape}")
        if clip:
            array = np.clip(array, GENE_LOWS, GENE_HIGHS)
        return cls(**dict(zip(GENE_NAMES, array.tolist(), strict=True)))

    @classmethod
    def from_dict(cls, genes: Mapping[str, float]) -> "Genome":
        """Strict inverse of :meth:`to_dict`: missing or unknown gene names are an error."""
        missing = [n for n in GENE_NAMES if n not in genes]
        unknown = [n for n in genes if n not in GENE_NAMES]
        if missing or unknown:
            raise ValueError(f"Gene names do not match: missing {missing}, unknown {unknown}")
        return cls(**{n: genes[n] for n in GENE_NAMES})

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Genome":
        """Uniformly random genome inside the bounds."""
        return cls.from_vector(rng.uniform(GENE_LOWS, GENE_HIGHS))

    def strategy_card(self) -> list[str]:
        """The genome rewritten as readable strategy rules (for the UI, notebooks and reports)."""
        g = self
        if g.pf_open_raise <= g.pf_play_btn:
            button_rule = f"Button, first to act: fold below equity {g.pf_play_btn:.2f}, raise everything else (never limps)."
        else:
            button_rule = (
                f"Button, first to act: fold below equity {g.pf_play_btn:.2f}, limp up to {g.pf_open_raise:.2f}, "
                f"raise at or above {g.pf_open_raise:.2f}."
            )
        return [
            button_rule,
            f"Big blind vs a limp: raise with equity >= {g.pf_bb_raise:.2f}, otherwise check.",
            f"Preflop vs a raise: call with equity >= {g.pf_defend:.2f} (and pot odds {g.call_margin:+.2f}), "
            f"{_tiered('3-bet', g.pf_3bet, 'jam all-in', g.pf_shove)}.",
            f"Postflop (adjusted equity): bet for value at >= {g.value_bet:.2f}; facing a bet, "
            f"{_tiered('raise', g.value_raise, 'go all-in', g.shove_threshold)}.",
            f"Facing a bet: call when equity >= pot odds {g.call_margin:+.2f} + {g.fold_big_bet:.2f} per "
            f"pot-size of the bet.",
            f"Adjusted equity = equity {g.position_bonus:+.2f} in position ({-g.position_bonus:+.2f} out of "
            f"position) minus up to {g.deep_stack_caution:.2f} for deep stacks.",
            f"Bluffs {g.bluff_freq:.0%} of weak hands when checked to; semi-bluffs {g.semi_bluff_freq:.0%} "
            f"of flush/straight draws.",
            f"Bet sizing: pot-size {g.big_bet_prob:.0%} of the time, otherwise half pot.",
            f"Adaptation: bluff frequency {g.adapt_fold:+.2f} per +1.0 opponent fold-to-bet above neutral; "
            f"call margin {g.adapt_aggression:+.2f} per +1.0 opponent aggression above neutral.",
        ]


@dataclass(frozen=True)
class GeneSpec:
    """Name, bounds and documentation of one gene (the single source of truth is :class:`Genome`)."""

    name: str
    low: float
    high: float
    default: float
    group: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "default": self.default,
            "group": self.group,
            "description": self.description,
        }


GENE_SPECS: tuple[GeneSpec, ...] = tuple(
    GeneSpec(f.name, f.metadata["low"], f.metadata["high"], f.default, f.metadata["group"], f.metadata["doc"])
    for f in fields(Genome)
)
GENE_NAMES: tuple[str, ...] = tuple(spec.name for spec in GENE_SPECS)
GENE_LOWS = np.array([spec.low for spec in GENE_SPECS], dtype=np.float64)
GENE_HIGHS = np.array([spec.high for spec in GENE_SPECS], dtype=np.float64)


#: Fixed playing styles built from the same genes: sparring partners for the GA (the first five are in
#: the default training pool) and seeds of the initial population. ``balanced`` and ``bluffer`` are
#: reserved as *held-out* styles that never appear in the training pool.
PRESETS: dict[str, Genome] = {
    "nit": Genome(
        pf_play_btn=0.56, pf_open_raise=0.62, pf_bb_raise=0.72, pf_defend=0.60, pf_3bet=0.78, pf_shove=0.84,
        value_bet=0.72, value_raise=0.86, shove_threshold=0.97, call_margin=0.10, fold_big_bet=0.15,
        deep_stack_caution=0.08, position_bonus=0.0, bluff_freq=0.02, semi_bluff_freq=0.10, big_bet_prob=0.30,
        adapt_fold=0.0, adapt_aggression=0.0,
    ),
    "tag": Genome(),  # the baseline defaults: tight-aggressive
    "lag": Genome(
        pf_play_btn=0.35, pf_open_raise=0.40, pf_bb_raise=0.48, pf_defend=0.42, pf_3bet=0.58, pf_shove=0.75,
        value_bet=0.55, value_raise=0.70, shove_threshold=0.88, call_margin=-0.02, fold_big_bet=0.02,
        deep_stack_caution=0.0, position_bonus=0.06, bluff_freq=0.35, semi_bluff_freq=0.85, big_bet_prob=0.55,
        adapt_fold=0.30, adapt_aggression=0.0,
    ),
    "passive": Genome(
        pf_play_btn=0.36, pf_open_raise=0.75, pf_bb_raise=0.82, pf_defend=0.42, pf_3bet=0.92, pf_shove=0.93,
        value_bet=0.80, value_raise=0.96, shove_threshold=0.99, call_margin=-0.03, fold_big_bet=0.0,
        deep_stack_caution=0.0, position_bonus=0.0, bluff_freq=0.0, semi_bluff_freq=0.05, big_bet_prob=0.20,
        adapt_fold=0.0, adapt_aggression=0.0,
    ),
    "maniac-ish": Genome(
        pf_play_btn=0.33, pf_open_raise=0.38, pf_bb_raise=0.40, pf_defend=0.36, pf_3bet=0.50, pf_shove=0.70,
        value_bet=0.50, value_raise=0.57, shove_threshold=0.78, call_margin=0.0, fold_big_bet=0.0,
        deep_stack_caution=0.0, position_bonus=0.0, bluff_freq=0.60, semi_bluff_freq=1.0, big_bet_prob=0.80,
        adapt_fold=0.0, adapt_aggression=0.0,
    ),
    "balanced": Genome(
        pf_play_btn=0.40, pf_open_raise=0.52, pf_bb_raise=0.60, pf_defend=0.46, pf_3bet=0.72, pf_shove=0.82,
        value_bet=0.58, value_raise=0.74, shove_threshold=0.90, call_margin=0.05, fold_big_bet=0.10,
        deep_stack_caution=0.03, position_bonus=0.04, bluff_freq=0.20, semi_bluff_freq=0.60, big_bet_prob=0.50,
        adapt_fold=0.40, adapt_aggression=0.10,
    ),
    "bluffer": Genome(
        pf_play_btn=0.46, pf_open_raise=0.52, pf_bb_raise=0.62, pf_defend=0.52, pf_3bet=0.64, pf_shove=0.85,
        value_bet=0.66, value_raise=0.82, shove_threshold=0.95, call_margin=0.06, fold_big_bet=0.05,
        deep_stack_caution=0.02, position_bonus=0.05, bluff_freq=0.50, semi_bluff_freq=0.90, big_bet_prob=0.75,
        adapt_fold=0.50, adapt_aggression=-0.10,
    ),
}
TRAINING_PRESETS: tuple[str, ...] = ("nit", "tag", "lag", "passive", "maniac-ish")
HELD_OUT_PRESETS: tuple[str, ...] = ("balanced", "bluffer")


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
@dataclass(slots=True)
class _Plan:
    """The outcome of the rule evaluation: what to do, why, and the numbers behind it."""

    abstract: AbstractAction
    rules: list[str]
    genes_used: tuple[str, ...]
    facts: dict[str, Any] = field(default_factory=dict)


class GeneticAgent(Agent):
    """Plays the strategy encoded by a :class:`Genome`.

    ``reset(seed)`` rebinds the private RNGs and the per-hand equity cache, so
    ``copy.copy(agent)`` followed by ``reset()`` yields an independent player that shares only the
    immutable genome (ARCHITECTURE.md section 5).
    """

    architecture = "Genetic Algorithm"
    supports_explanations = True

    def __init__(
        self,
        genome: Genome | None = None,
        name: str = "GA",
        equity_samples: int = DEFAULT_EQUITY_SAMPLES,
        seed: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if equity_samples < 2:
            raise ValueError("equity_samples must be at least 2")
        self.genome = genome if genome is not None else Genome()
        self.name = name
        self.equity_samples = equity_samples
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.description = "Strategy evolved through self-play: a few readable genes (thresholds, bluff rates, sizing)."
        self.reset(seed)

    # ---- construction helpers ----------------------------------------------------
    @classmethod
    def from_preset(cls, preset: str, **kwargs: Any) -> "GeneticAgent":
        """A fixed-style player (``nit``, ``tag``, ``lag``, ``passive``, ``maniac-ish``, ...)."""
        if preset not in PRESETS:
            raise KeyError(f"Unknown GA preset {preset!r}; known: {sorted(PRESETS)}")
        kwargs.setdefault("name", f"GA-{preset}")
        return cls(PRESETS[preset], **kwargs)

    @classmethod
    def load(cls, model_dir: Path | str | None = None, **kwargs: Any) -> "GeneticAgent":
        """Load the evolved genome from ``artifacts/models/ga/`` (or ``model_dir``).

        Raises :class:`ModelArtifactError` with an actionable message if a file is missing, corrupt,
        or was written for a different schema version / gene set.
        """
        directory = Path(model_dir) if model_dir is not None else MODELS_DIR / MODEL_SUBDIR
        genome_doc = _read_json(directory / BEST_GENOME_FILE)
        metadata = _read_json(directory / METADATA_FILE)
        for label, document in ((BEST_GENOME_FILE, genome_doc), (METADATA_FILE, metadata)):
            version = document.get("schema_version")
            if version != GA_SCHEMA_VERSION:
                raise ModelArtifactError(
                    f"{directory / label} has schema_version {version!r} but this code expects "
                    f"{GA_SCHEMA_VERSION}. Retrain with `{TRAIN_COMMAND}`."
                )
        stored_names = metadata.get("gene_names")
        if stored_names != list(GENE_NAMES):
            raise ModelArtifactError(
                f"{directory / METADATA_FILE} was trained with genes {stored_names!r}, but the code defines "
                f"{list(GENE_NAMES)}. Retrain with `{TRAIN_COMMAND}`."
            )
        try:
            genome = Genome.from_dict(genome_doc["genes"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ModelArtifactError(
                f"{directory / BEST_GENOME_FILE} is not a valid genome ({exc}). Retrain with `{TRAIN_COMMAND}`."
            ) from exc
        equity_samples = metadata.get("equity_samples")
        if not isinstance(equity_samples, int) or equity_samples < 2:
            raise ModelArtifactError(
                f"{directory / METADATA_FILE} lacks a valid 'equity_samples'. Retrain with `{TRAIN_COMMAND}`."
            )
        params: dict[str, Any] = {"equity_samples": equity_samples, "metadata": metadata}
        params.update(kwargs)
        return cls(genome, **params)

    # ---- Agent interface ---------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Re-seed the private RNGs and drop per-hand memory (rebinds - never mutates - the old ones)."""
        decision_stream, equity_stream = np.random.SeedSequence(0 if seed is None else seed).spawn(2)
        self._rng = np.random.default_rng(decision_stream)  # bluff / sizing coin flips
        self._equity_rng = np.random.default_rng(equity_stream)  # Monte-Carlo equity
        self._equity_cache: dict[tuple[tuple[str, str], tuple[str, ...]], float] = {}

    def end_hand(self, summary: PublicHandSummary) -> None:
        self._equity_cache = {}

    def act(self, obs: Observation) -> Decision:
        mask = obs.abstract_mask
        # Both coins are drawn on every decision so the random stream never depends on which rule fires.
        mix_coin, size_coin = self._rng.random(2)
        if obs.street is Street.PREFLOP:
            plan = self._decide_preflop(obs, mask, float(mix_coin), float(size_coin))
        else:
            plan = self._decide_postflop(obs, mask, float(mix_coin), float(size_coin))
        return Decision.from_abstract(obs, plan.abstract, self._explanation(obs, plan))

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["genome"] = self.genome.to_dict()
        info["strategy"] = self.genome.strategy_card()
        info["equity_samples"] = self.equity_samples
        training = self.metadata.get("training")
        if training:
            info["training"] = training
        return info

    # ---- shared quantities -------------------------------------------------------
    def _equity(self, obs: Observation) -> float:
        """Postflop equity vs a random hand, estimated once per (hand, board) and cached."""
        key = (obs.my_hole, obs.board)
        equity = self._equity_cache.get(key)
        if equity is None:
            equity = equity_vs_random(obs.my_hole, obs.board, self.equity_samples, self._equity_rng)
            self._equity_cache[key] = equity
        return equity

    def _pick_raise(self, mask: np.ndarray, size_coin: float) -> AbstractAction | None:
        """Half-pot or pot-size according to ``big_bet_prob``; falls back to whatever raise is legal."""
        if size_coin < self.genome.big_bet_prob:
            order = (AbstractAction.RAISE_POT, AbstractAction.RAISE_SMALL, AbstractAction.ALL_IN)
        else:
            order = (AbstractAction.RAISE_SMALL, AbstractAction.RAISE_POT, AbstractAction.ALL_IN)
        return next((a for a in order if mask[a]), None)

    def _call_requirement(self, obs: Observation) -> tuple[float, dict[str, float]]:
        """Equity needed to continue against the current bet: pot odds plus the risk genes."""
        g = self.genome
        stats = obs.opponent_stats
        pot_odds = obs.to_call / (obs.pot + obs.to_call)
        bet_fraction = min(obs.to_call / max(obs.pot - obs.to_call, 1), BET_FRACTION_CAP)
        parts = {
            "pot_odds": pot_odds,
            "call_margin": g.call_margin,
            "big_bet": g.fold_big_bet * bet_fraction,
            "opponent_aggression": g.adapt_aggression * (stats.aggression - _NEUTRAL_STATS.aggression),
        }
        return sum(parts.values()), parts

    @staticmethod
    def _passive(mask: np.ndarray, prefer_fold: bool) -> AbstractAction:
        """Fold (if allowed and wanted) or check/call - always legal when it is our turn."""
        if prefer_fold and mask[AbstractAction.FOLD]:
            return AbstractAction.FOLD
        return AbstractAction.CHECK_CALL

    # ---- preflop -----------------------------------------------------------------
    def _decide_preflop(self, obs: Observation, mask: np.ndarray, mix_coin: float, size_coin: float) -> _Plan:
        g = self.genome
        hand = preflop_class(obs.my_hole)
        equity = preflop_equity(obs.my_hole)
        facts: dict[str, Any] = {"street": "preflop", "hand": hand, "equity": round(equity, 3)}
        head = f"{hand} (equity {equity:.2f})"

        if obs.to_call == 0:  # big blind with the option after the button limped
            raise_action = self._pick_raise(mask, size_coin)
            if equity >= g.pf_bb_raise and raise_action is not None:
                rule = f"{head} >= pf_bb_raise {g.pf_bb_raise:.2f} -> raise over the limp ({_SIZE_LABEL[raise_action]})"
                return _Plan(raise_action, [rule], ("pf_bb_raise", "big_bet_prob"), facts)
            rule = f"{head} < pf_bb_raise {g.pf_bb_raise:.2f} -> check"
            return _Plan(AbstractAction.CHECK_CALL, [rule], ("pf_bb_raise",), facts)

        if obs.num_raises == 0:  # button, first to act
            if equity < g.pf_play_btn:
                rule = f"{head} < pf_play_btn {g.pf_play_btn:.2f} -> fold"
                return _Plan(self._passive(mask, True), [rule], ("pf_play_btn",), facts)
            raise_action = self._pick_raise(mask, size_coin)
            if equity >= g.pf_open_raise and raise_action is not None:
                rule = (
                    f"{head} >= pf_play_btn {g.pf_play_btn:.2f} and >= pf_open_raise {g.pf_open_raise:.2f} "
                    f"-> open-raise ({_SIZE_LABEL[raise_action]})"
                )
                return _Plan(raise_action, [rule], ("pf_play_btn", "pf_open_raise", "big_bet_prob"), facts)
            rule = f"{head} >= pf_play_btn {g.pf_play_btn:.2f} but below pf_open_raise {g.pf_open_raise:.2f} -> limp"
            return _Plan(AbstractAction.CHECK_CALL, [rule], ("pf_play_btn", "pf_open_raise"), facts)

        # facing a raise
        required, parts = self._call_requirement(obs)
        facts.update(required_equity=round(required, 3), pot_odds=round(parts["pot_odds"], 3))
        if equity >= g.pf_shove and mask[AbstractAction.ALL_IN]:
            rule = f"{head} >= pf_shove {g.pf_shove:.2f} -> jam all-in over the raise"
            return _Plan(AbstractAction.ALL_IN, [rule], ("pf_shove",), facts)
        raise_action = self._pick_raise(mask, size_coin)
        if equity >= g.pf_3bet and obs.num_raises == 1 and raise_action is not None:
            rule = f"{head} >= pf_3bet {g.pf_3bet:.2f} -> 3-bet ({_SIZE_LABEL[raise_action]})"
            return _Plan(raise_action, [rule], ("pf_3bet", "big_bet_prob"), facts)
        needed = max(g.pf_defend, required)
        if equity >= needed:
            rule = (
                f"{head} >= max(pf_defend {g.pf_defend:.2f}, required {required:.2f}: {_describe_parts(parts)}) -> call"
            )
            return _Plan(AbstractAction.CHECK_CALL, [rule], ("pf_defend", "call_margin", "fold_big_bet"), facts)
        rule = f"{head} < max(pf_defend {g.pf_defend:.2f}, required {required:.2f}: {_describe_parts(parts)}) -> fold"
        return _Plan(self._passive(mask, True), [rule], ("pf_defend", "call_margin", "fold_big_bet"), facts)

    # ---- postflop ----------------------------------------------------------------
    def _decide_postflop(self, obs: Observation, mask: np.ndarray, mix_coin: float, size_coin: float) -> _Plan:
        g = self.genome
        stats = obs.opponent_stats
        street = _STREET_LABEL[obs.street]
        equity = self._equity(obs)
        position = g.position_bonus if obs.is_button else -g.position_bonus
        spr = obs.effective_stack / max(obs.pot, 1)
        caution = g.deep_stack_caution * min(spr, SPR_CAP) / SPR_CAP
        adjusted = equity + position - caution
        facts: dict[str, Any] = {
            "street": street,
            "equity": round(equity, 3),
            "adjusted_equity": round(adjusted, 3),
            "in_position": obs.is_button,
            "spr": round(spr, 2),
        }
        head = f"{street}: equity {equity:.2f} -> adjusted {adjusted:.2f} ({position:+.2f} position, {-caution:+.2f} deep stacks)"
        used = ["position_bonus", "deep_stack_caution"]
        fold_deviation = stats.fold_to_bet - _NEUTRAL_STATS.fold_to_bet
        bluff_probability = min(max(g.bluff_freq + g.adapt_fold * fold_deviation, 0.0), 1.0)
        semi_probability = min(max(g.semi_bluff_freq + g.adapt_fold * fold_deviation, 0.0), 1.0)
        facts["opponent_fold_to_bet"] = round(stats.fold_to_bet, 3)

        if obs.to_call == 0:
            return self._decide_unopened(obs, mask, adjusted, head, used, facts, bluff_probability, semi_probability, mix_coin, size_coin)

        required, parts = self._call_requirement(obs)
        facts.update(required_equity=round(required, 3), pot_odds=round(parts["pot_odds"], 3))
        if adjusted >= g.shove_threshold and mask[AbstractAction.ALL_IN]:
            rule = f"{head}; adjusted >= shove_threshold {g.shove_threshold:.2f} -> all-in"
            return _Plan(AbstractAction.ALL_IN, [rule], (*used, "shove_threshold"), facts)
        raise_action = self._pick_raise(mask, size_coin)
        if adjusted >= g.value_raise and raise_action is not None:
            rule = f"{head}; adjusted >= value_raise {g.value_raise:.2f} -> raise ({_SIZE_LABEL[raise_action]})"
            return _Plan(raise_action, [rule], (*used, "value_raise", "big_bet_prob"), facts)
        if raise_action is not None and obs.num_raises_this_street == 1 and self._has_draw(obs):
            if mix_coin < semi_probability:
                rule = (
                    f"{head}; draw and coin {mix_coin:.2f} < semi-bluff probability {semi_probability:.2f} "
                    f"(semi_bluff_freq {g.semi_bluff_freq:.2f} {g.adapt_fold * fold_deviation:+.2f} adaptation) "
                    f"-> semi-bluff raise ({_SIZE_LABEL[raise_action]})"
                )
                return _Plan(raise_action, [rule], (*used, "semi_bluff_freq", "adapt_fold", "big_bet_prob"), facts)
        detail = f"required {required:.2f}: {_describe_parts(parts)}"
        used_defence = (*used, "call_margin", "fold_big_bet", "adapt_aggression")
        if adjusted >= required:
            return _Plan(AbstractAction.CHECK_CALL, [f"{head}; adjusted >= {detail} -> call"], used_defence, facts)
        return _Plan(self._passive(mask, True), [f"{head}; adjusted < {detail} -> fold"], used_defence, facts)

    def _decide_unopened(
        self, obs: Observation, mask: np.ndarray, adjusted: float, head: str, used: list[str], facts: dict[str, Any],
        bluff_probability: float, semi_probability: float, mix_coin: float, size_coin: float,
    ) -> _Plan:
        """Checked to (or first to act): value bet, shove, semi-bluff, bluff or check."""
        g = self.genome
        if adjusted >= g.shove_threshold and mask[AbstractAction.ALL_IN]:
            rule = f"{head}; adjusted >= shove_threshold {g.shove_threshold:.2f} -> all-in"
            return _Plan(AbstractAction.ALL_IN, [rule], (*used, "shove_threshold"), facts)
        raise_action = self._pick_raise(mask, size_coin)
        if raise_action is None:
            return _Plan(AbstractAction.CHECK_CALL, [f"{head}; no bet is possible -> check"], tuple(used), facts)
        if adjusted >= g.value_bet:
            rule = f"{head}; adjusted >= value_bet {g.value_bet:.2f} -> value bet ({_SIZE_LABEL[raise_action]})"
            return _Plan(raise_action, [rule], (*used, "value_bet", "big_bet_prob"), facts)
        if obs.street in (Street.FLOP, Street.TURN) and self._has_draw(obs):
            if mix_coin < semi_probability:
                rule = (
                    f"{head}; draw and coin {mix_coin:.2f} < semi-bluff probability {semi_probability:.2f} "
                    f"-> semi-bluff bet ({_SIZE_LABEL[raise_action]})"
                )
                return _Plan(raise_action, [rule], (*used, "semi_bluff_freq", "adapt_fold", "big_bet_prob"), facts)
            rule = f"{head}; draw but coin {mix_coin:.2f} >= semi-bluff probability {semi_probability:.2f} -> check"
            return _Plan(AbstractAction.CHECK_CALL, [rule], (*used, "semi_bluff_freq", "adapt_fold"), facts)
        if mix_coin < bluff_probability:
            rule = (
                f"{head}; weak hand, coin {mix_coin:.2f} < bluff probability {bluff_probability:.2f} "
                f"-> bluff ({_SIZE_LABEL[raise_action]})"
            )
            return _Plan(raise_action, [rule], (*used, "bluff_freq", "adapt_fold", "big_bet_prob"), facts)
        rule = f"{head}; below value_bet {g.value_bet:.2f}, coin {mix_coin:.2f} >= bluff probability {bluff_probability:.2f} -> check"
        return _Plan(AbstractAction.CHECK_CALL, [rule], (*used, "value_bet", "bluff_freq"), facts)

    @staticmethod
    def _has_draw(obs: Observation) -> bool:
        """Flush draw or open-ended straight draw on the flop/turn (cheap, public-information features)."""
        if obs.street not in (Street.FLOP, Street.TURN):
            return False
        features = draw_features(obs.my_hole, obs.board)
        outs = FLUSH_DRAW_OUTS * features["flush_draw"] + features["straight_outs"]
        return outs >= SEMI_BLUFF_MIN_OUTS

    # ---- explanation -------------------------------------------------------------
    def _explanation(self, obs: Observation, plan: _Plan) -> Explanation:
        label = plan.abstract.name.lower()
        genes = {name: round(getattr(self.genome, name), 3) for name in dict.fromkeys(plan.genes_used)}
        return Explanation(
            summary=f"{plan.rules[0]}.",
            details={"action": label, "rules": plan.rules, "genes": genes, "facts": plan.facts},
        )


def _tiered(raise_name: str, raise_threshold: float, shove_name: str, shove_threshold: float) -> str:
    """'raise at >= 0.78, go all-in at >= 0.92' - or only the all-in when it pre-empts the raise."""
    if raise_threshold >= shove_threshold:
        return f"{shove_name} at >= {shove_threshold:.2f} (never just {raise_name})"
    return f"{raise_name} at >= {raise_threshold:.2f}, {shove_name} at >= {shove_threshold:.2f}"


def _describe_parts(parts: Mapping[str, float]) -> str:
    """'pot odds 0.33 + margin 0.03 + big bet 0.04 + aggression 0.00' for explanations."""
    return (
        f"pot odds {parts['pot_odds']:.2f} + call_margin {parts['call_margin']:+.2f} + "
        f"fold_big_bet {parts['big_bet']:.2f} + adapt_aggression {parts['opponent_aggression']:+.2f}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ModelArtifactError(f"Missing GA artifact {path}. Train it with `{TRAIN_COMMAND}`.")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"Cannot read GA artifact {path} ({exc}). Retrain with `{TRAIN_COMMAND}`.") from exc
    if not isinstance(document, dict):
        raise ModelArtifactError(f"GA artifact {path} must contain a JSON object. Retrain with `{TRAIN_COMMAND}`.")
    return document
