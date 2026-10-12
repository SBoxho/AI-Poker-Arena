"""Namespaced session-state helpers (tested against a plain dict store)."""

from __future__ import annotations

from poker_arena.ui.state import KEY_PREFIX, SEED_BITS, SessionNamespace, random_seed


def test_keys_are_prefixed_and_namespaced():
    ns = SessionNamespace("ava", {})
    assert ns.key("match") == f"{KEY_PREFIX}.ava.match"
    assert SessionNamespace("hva", {}).key("match") != ns.key("match")


def test_get_set_contains_pop():
    store: dict = {}
    ns = SessionNamespace("ava", store)
    assert "x" not in ns and ns.get("x") is None and ns.get("x", 5) == 5
    ns.set("x", 1)
    assert "x" in ns and ns.get("x") == 1 and store == {ns.key("x"): 1}
    ns.pop("x")
    ns.pop("x")  # popping a missing key is fine
    assert "x" not in ns


def test_setdefault_calls_the_factory_once():
    calls = []
    ns = SessionNamespace("ava", {})
    first = ns.setdefault("v", lambda: calls.append(1) or [])
    second = ns.setdefault("v", lambda: calls.append(1) or [])
    assert first is second and len(calls) == 1


def test_clear_only_touches_its_own_namespace_and_honours_keep():
    store: dict = {"other": 1}
    a, b = SessionNamespace("ava", store), SessionNamespace("hva", store)
    for name in ("match", "playing", "speed"):
        a.set(name, name)
    b.set("match", "keep me")
    a.clear(keep=("speed",))
    assert "match" not in a and "playing" not in a and a.get("speed") == "speed"
    assert b.get("match") == "keep me" and store["other"] == 1


def test_forget_and_drop_prefix():
    ns = SessionNamespace("hva", {})
    for name in ("amount-0-1", "amount-0-2", "amount-0-3", "fold-0-1", "error"):
        ns.set(name, 1)
    ns.drop_prefix("amount-", keep="amount-0-3")
    assert "amount-0-1" not in ns and "amount-0-2" not in ns and "amount-0-3" in ns and "fold-0-1" in ns
    ns.forget("error", "fold-0-1", "missing")
    assert "error" not in ns and "fold-0-1" not in ns


def test_random_seed_range_and_variety():
    seeds = {random_seed() for _ in range(20)}
    assert len(seeds) > 1 and all(0 <= s < 2**SEED_BITS for s in seeds)
