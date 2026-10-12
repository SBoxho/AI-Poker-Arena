"""Session-state helpers: namespaced keys so a page can reset "its" state without touching others.

Nothing is stored at module level - every browser session keeps its own objects in
``st.session_state``. The store is injectable so the helpers can be unit-tested with a plain dict.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, MutableMapping
from typing import Any, TypeVar

import streamlit as st

T = TypeVar("T")
KEY_PREFIX = "pa"
SEED_BITS = 31


def random_seed() -> int:
    """Fresh non-deterministic seed for "random" games (a full-range int fits every widget)."""
    return secrets.randbits(SEED_BITS)


class SessionNamespace:
    """A prefixed view on the session store (``pa.<namespace>.<name>``)."""

    def __init__(self, namespace: str, store: MutableMapping[str, Any] | None = None) -> None:
        self.namespace = namespace
        self._store = store

    @property
    def store(self) -> MutableMapping[str, Any]:
        return self._store if self._store is not None else st.session_state  # type: ignore[return-value]

    def key(self, name: str) -> str:
        """Full key; also use it as the ``key=`` of widgets so ``clear()`` resets them too."""
        return f"{KEY_PREFIX}.{self.namespace}.{name}"

    def __contains__(self, name: str) -> bool:
        return self.key(name) in self.store

    def get(self, name: str, default: Any = None) -> Any:
        return self.store.get(self.key(name), default)

    def set(self, name: str, value: Any) -> None:
        self.store[self.key(name)] = value

    def setdefault(self, name: str, factory: Callable[[], T]) -> T:
        """Return the stored value, creating it with ``factory()`` on first use."""
        full = self.key(name)
        if full not in self.store:
            self.store[full] = factory()
        return self.store[full]

    def pop(self, name: str) -> None:
        self.store.pop(self.key(name), None)

    def forget(self, *names: str) -> None:
        for name in names:
            self.pop(name)

    def clear(self, *, keep: tuple[str, ...] = ()) -> None:
        """Delete every key of this namespace except the names in ``keep``."""
        prefix = f"{KEY_PREFIX}.{self.namespace}."
        kept = {self.key(name) for name in keep}
        for full in [k for k in list(self.store.keys()) if k.startswith(prefix) and k not in kept]:
            del self.store[full]

    def drop_prefix(self, name_prefix: str, *, keep: str | None = None) -> None:
        """Delete keys whose name starts with ``name_prefix`` (e.g. stale per-decision widgets)."""
        prefix = self.key(name_prefix)
        kept = self.key(keep) if keep is not None else None
        for full in [k for k in list(self.store.keys()) if k.startswith(prefix) and k != kept]:
            del self.store[full]
