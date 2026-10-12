"""Reading and writing datasets (``<name>.parquet`` + ``<name>.meta.json``) and small table helpers.

Layout of the table (see :data:`poker_arena.data.generation.COLUMNS`): identifier columns
(``hand_id``, ``hand_index``, ``decision_idx``, ``split``, ``source``, ``behavior``, ``opponent``),
the features in :data:`~poker_arena.features.FEATURE_NAMES` order (``float32``), then ``label``
(teacher's abstract action 0..4), five ``legal_*`` booleans, ``behavior_action`` (what the acting
seat actually did, -1 if unknown), ``teacher_margin`` and ``teacher_rule``.

Always split train/validation by ``hand_id`` (the ``split`` column does exactly that); never
shuffle decisions of one hand across folds.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from poker_arena.config import DATASETS_DIR
from poker_arena.data.columns import COLUMNS, LEGAL_COLUMNS
from poker_arena.features import FEATURE_NAMES, schema_hash

_LIBRARIES = ("numpy", "pandas", "pyarrow", "pokerkit", "scikit-learn", "interpret", "xgboost")
_COLUMN_SEPARATOR = b"\x1f"
_PARQUET_COMPRESSION = "zstd"


def library_versions() -> dict[str, str]:
    """Versions of the interpreter and the libraries that influence the generated numbers."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in _LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def content_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of the table *content* (independent of parquet encoding): equal iff the data are equal."""
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(str(column).encode())
        values = frame[column]
        if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object or pd.api.types.is_string_dtype(values):
            digest.update(_COLUMN_SEPARATOR.join(str(v).encode() for v in values.tolist()))
        else:
            digest.update(str(values.dtype).encode())
            digest.update(np.ascontiguousarray(values.to_numpy()).tobytes())
    return digest.hexdigest()


def dataset_paths(name: str, directory: Path | None = None) -> tuple[Path, Path]:
    """``(parquet path, meta path)`` of a named dataset."""
    base = Path(directory) if directory is not None else DATASETS_DIR
    return base / f"{name}.parquet", base / f"{name}.meta.json"


def write_dataset(frame: pd.DataFrame, meta: dict, name: str, directory: Path | None = None) -> tuple[Path, Path]:
    """Write ``<name>.parquet`` and ``<name>.meta.json``; returns both paths."""
    parquet_path, meta_path = dataset_paths(name, directory)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, engine="pyarrow", compression=_PARQUET_COMPRESSION, index=False)
    full_meta = {
        **meta,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {
            "parquet": parquet_path.name,
            "parquet_bytes": parquet_path.stat().st_size,
            "parquet_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        },
    }
    meta_path.write_text(json.dumps(full_meta, indent=2) + "\n", encoding="utf-8")
    return parquet_path, meta_path


def load_meta(name: str, directory: Path | None = None) -> dict:
    """Parsed ``<name>.meta.json``."""
    return json.loads(dataset_paths(name, directory)[1].read_text(encoding="utf-8"))


def load_dataset(
    name: str,
    directory: Path | None = None,
    split: str | None = None,
    check_schema: bool = True,
) -> pd.DataFrame:
    """Read a dataset table (optionally only one ``split``) and verify it matches the current feature schema."""
    parquet_path, _ = dataset_paths(name, directory)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} does not exist; generate it with `.venv/Scripts/python scripts/generate_dataset.py`"
        )
    if check_schema:
        stored = load_meta(name, directory)["schema_hash"]
        if stored != schema_hash():
            raise ValueError(
                f"Dataset {name!r} was built with feature schema {stored}, current is {schema_hash()}; regenerate it."
            )
    frame = pd.read_parquet(parquet_path, engine="pyarrow")
    if split is not None:
        frame = frame[frame["split"] == split].reset_index(drop=True)
    return frame


def feature_matrix(frame: pd.DataFrame) -> NDArray[np.float32]:
    """Features as a ``(rows, NUM_FEATURES)`` float32 matrix in schema order."""
    return frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)


def label_vector(frame: pd.DataFrame) -> NDArray[np.int64]:
    """Teacher labels (abstract action indices 0..4)."""
    return frame["label"].to_numpy(dtype=np.int64)


def legal_mask_matrix(frame: pd.DataFrame) -> NDArray[np.bool_]:
    """Legal abstract actions as a ``(rows, 5)`` boolean matrix."""
    return frame[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)


def validate_frame(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``frame`` does not have the dataset layout or labels violate the masks."""
    if list(frame.columns) != list(COLUMNS):
        raise ValueError("Dataset columns differ from poker_arena.data.generation.COLUMNS")
    labels, masks = label_vector(frame), legal_mask_matrix(frame)
    if len(frame) and not masks[np.arange(len(frame)), labels].all():
        raise ValueError("Some teacher labels are not legal actions")
