"""checks.evaluate: every operator, path forms, and fail-closed behavior."""

import pytest

from sandbox.checks import evaluate, resolve, strict_eq

STATE = {
    "a": {"flag": True, "n": 5, "s": "My meter is BROKEN", "lst": ["2026-09", "2026-08"], "d": {"k": 1}},
    "items": [{"id": "x", "v": 1}, {"id": "y", "v": 2}],
    "none": None,
}


def ok(cond):
    return evaluate(STATE, {"all": [cond]})[0]


@pytest.mark.parametrize("path,expected", [
    ("a.flag", True), ("a.n", 5), ("items.0.id", "x"), ("items.-1.v", 2), ("none", None), ("a.lst.1", "2026-08"),
])
def test_resolve_found(path, expected):
    assert resolve(STATE, path) == expected


@pytest.mark.parametrize("path", ["a.missing", "items.2", "items.-3", "items.x", "a.n.deeper", "a.flag.0"])
def test_resolve_missing(path):
    assert ok({"path": path, "op": "exists", "value": False})
    assert not ok({"path": path, "op": "exists"})


def test_eq_strict_bools_and_numbers():
    assert ok({"path": "a.flag", "op": "eq", "value": True})
    assert not ok({"path": "a.flag", "op": "eq", "value": 1})
    assert not ok({"path": "a.n", "op": "eq", "value": True})
    assert ok({"path": "a.n", "op": "eq", "value": 5.0})
    assert ok({"path": "a.lst", "op": "eq", "value": ["2026-09", "2026-08"]})
    assert not ok({"path": "a.lst", "op": "eq", "value": ["2026-09"]})
    assert ok({"path": "a.d", "op": "eq", "value": {"k": 1}})
    assert not ok({"path": "a.d", "op": "eq", "value": {"k": True}})
    assert ok({"path": "none", "op": "eq", "value": None})
    assert not ok({"path": "a.missing", "op": "eq", "value": None})
    assert not strict_eq("5", 5)


def test_neq():
    assert ok({"path": "a.n", "op": "neq", "value": 6})
    assert not ok({"path": "a.n", "op": "neq", "value": 5})
    assert not ok({"path": "a.missing", "op": "neq", "value": 5}), "missing path fails neq"


def test_contains():
    assert ok({"path": "a.s", "op": "contains", "value": "broken"}), "case-insensitive substring"
    assert ok({"path": "a.s", "op": "contains", "value": "meter"})
    assert not ok({"path": "a.s", "op": "contains", "value": "fine"})
    assert ok({"path": "a.lst", "op": "contains", "value": "2026-09"})
    assert not ok({"path": "a.lst", "op": "contains", "value": "2026"}), "lists match whole elements"
    assert ok({"path": "a.d", "op": "contains", "value": "k"})
    assert not ok({"path": "a.n", "op": "contains", "value": 5})
    assert not ok({"path": "a.s", "op": "contains", "value": 5})


def test_len_gte():
    assert ok({"path": "a.lst", "op": "len_gte", "value": 2})
    assert not ok({"path": "a.lst", "op": "len_gte", "value": 3})
    assert ok({"path": "a.s", "op": "len_gte", "value": 1})
    assert not ok({"path": "a.n", "op": "len_gte", "value": 1})
    assert not ok({"path": "a.lst", "op": "len_gte", "value": True})


def test_exists_default_true():
    assert ok({"path": "a.flag", "op": "exists"})
    assert ok({"path": "none", "op": "exists"}), "a key holding null exists"


def test_all_requires_every_condition_and_explains():
    passed, why = evaluate(STATE, {"all": [
        {"path": "a.flag", "op": "eq", "value": True},
        {"path": "a.n", "op": "eq", "value": 7},
    ]})
    assert not passed
    assert "a.n" in why and "expected 7" in why and "got 5" in why
    passed, why = evaluate(STATE, {"all": [{"path": "a.flag", "op": "eq", "value": True}]})
    assert passed and "1 conditions passed" in why


def test_nested_any():
    check = {"all": [{"any": [{"path": "a.n", "op": "eq", "value": 1}, {"path": "a.n", "op": "eq", "value": 5}]}]}
    assert evaluate(STATE, check)[0]


@pytest.mark.parametrize("check", [
    None, [], {}, {"all": []}, {"all": "x"}, {"all": [None]}, {"all": [{"op": "eq"}]},
    {"all": [{"path": "a.n", "op": "gt", "value": 1}]}, {"some": []},
])
def test_malformed_fails_closed(check):
    passed, why = evaluate(STATE, check)
    assert passed is False and why


def test_never_raises_on_odd_state():
    assert evaluate(None, {"all": [{"path": "a", "op": "exists"}]})[0] is False
    assert evaluate([1, 2], {"all": [{"path": "0", "op": "eq", "value": 1}]})[0] is True


def test_pure_does_not_mutate():
    import copy
    before = copy.deepcopy(STATE)
    evaluate(STATE, {"all": [{"path": "a.lst", "op": "contains", "value": "2026-09"}]})
    assert STATE == before
