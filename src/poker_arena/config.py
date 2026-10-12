"""Shared configuration primitives and filesystem layout."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

# <repo>/src/poker_arena/config.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TOURNAMENTS_DIR = ARTIFACTS_DIR / "tournaments"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"


@dataclass(frozen=True)
class GameConfig:
    """Rules of the simulated game. All chips are simulated integers."""

    starting_stack: int = 1000
    small_blind: int = 5
    big_blind: int = 10

    def __post_init__(self) -> None:
        # Strict: with equal blinds PokerKit lets the BIG blind open the preflop action heads-up
        # (it breaks the "last blind" tie by seat index), violating "the button acts first preflop".
        if not 0 < self.small_blind < self.big_blind:
            raise ValueError("Require 0 < small_blind < big_blind")
        if self.starting_stack < 2 * self.big_blind:
            raise ValueError("starting_stack must be at least 2 big blinds")

    @property
    def stack_in_bb(self) -> float:
        return self.starting_stack / self.big_blind

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GameConfig":
        return cls(**{k: int(d[k]) for k in ("starting_stack", "small_blind", "big_blind")})


# Seed namespaces keep training, validation and evaluation randomness disjoint *by construction*:
# the same (namespace, parts) always gives the same seed, different namespaces never collide in practice.
SEED_NAMESPACES = ("dataset_train", "dataset_val", "dagger", "ga_fitness", "ga_validation", "eval", "ui", "test")


def derive_seed(namespace: str, *parts: int | str) -> int:
    """Deterministic 63-bit seed from a namespace and arbitrary parts (e.g. master seed, hand index).

    Example: ``derive_seed("eval", 42, match_index, pair_index)``. Use a distinct namespace for every
    purpose so that e.g. tournament deals can never coincide with training deals.
    """
    if namespace not in SEED_NAMESPACES:
        raise ValueError(f"Unknown seed namespace {namespace!r}; known: {SEED_NAMESPACES}")
    digest = hashlib.sha256("|".join([namespace, *map(str, parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def portable_path(path: Path | str) -> str:
    """Repo-relative POSIX string for a path inside the repository (otherwise the path unchanged).

    Saved artifacts must never embed a local absolute directory (such as a user's home folder).
    """
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()


def portable_json_default(value: object) -> object:
    """``json.dumps(default=...)`` handler: numpy scalars/arrays become Python values, paths become portable."""
    import numpy as np  # local import keeps this module dependency-free for lightweight callers

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return portable_path(value)
    return str(value)
