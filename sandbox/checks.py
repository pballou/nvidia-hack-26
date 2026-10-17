"""Task checks over sandbox session state (docs/tasks/contracts.md §1).

`evaluate(state, check)` is pure: no I/O, never raises. Used by sandbox/tests, the
Gym `web_agent` verify() and the console's verified banner.

Check format: {"all": [cond, ...]}; every condition must pass. (Nested {"all": ...}
and {"any": ...} groups are also accepted as conditions; tasks.json uses only "all".)

A condition is {"path": "a.b.0.c", "op": <op>, "value": <expected>}.

Paths are dot-separated. A segment indexes a dict by key, or a list by integer
(negative counts from the end, so "messages.sent.-1" is the newest sent message).

Ops:
- eq        found and equal. Strict about bools (true != 1); ints and floats compare
            numerically (50 == 50.0); lists and dicts compare element-wise.
- neq       found and not equal. A missing path fails (it means a schema mismatch).
- contains  string target: case-insensitive substring of a string value.
            list target: some element equals the value (eq semantics).
            dict target: the value is a key.
- len_gte   target is a list, dict or string with len >= value.
- exists    value defaults to true: passes iff the path resolves. With value false,
            passes iff it does not (e.g. "billing.payments.1" + false = at most one
            payment).

An empty or malformed check fails closed, so a broken task can't award reward.
"""

from __future__ import annotations

from typing import Any

OPS = ("eq", "neq", "contains", "len_gte", "exists")
_MISSING = object()


def resolve(state: Any, path: str) -> Any:
    """Return the value at a dotted path, or the module-private _MISSING sentinel."""
    cur = state
    if path == "":
        return cur
    for seg in path.split("."):
        if isinstance(cur, dict):
            if seg not in cur:
                return _MISSING
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError:
                return _MISSING
            if not -len(cur) <= idx < len(cur):
                return _MISSING
            cur = cur[idx]
        else:
            return _MISSING
    return cur


def strict_eq(a: Any, b: Any) -> bool:
    """Equality that does not treat bools as ints."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(strict_eq(a[k], b[k]) for k in a)
    if type(a) is not type(b):
        return False
    return a == b


def _short(v: Any) -> str:
    s = repr(v)
    return s if len(s) <= 80 else s[:77] + "..."


def _eval_condition(state: Any, cond: Any) -> tuple[bool, str]:
    if not isinstance(cond, dict):
        return False, f"malformed condition {_short(cond)}"
    if "all" in cond or "any" in cond:
        return evaluate(state, cond)
    path, op = cond.get("path"), cond.get("op")
    if not isinstance(path, str):
        return False, f"condition without a string path: {_short(cond)}"
    if op not in OPS:
        return False, f"{path}: unknown op {op!r}"
    expected = cond.get("value", True if op == "exists" else None)
    actual = resolve(state, path)
    found = actual is not _MISSING

    if op == "exists":
        want = bool(expected)
        ok = found == want
        return ok, f"{path} {'exists' if found else 'missing'} (want {'exists' if want else 'missing'})"
    if not found:
        return False, f"{path}: path not found (op {op})"
    if op == "eq":
        ok = strict_eq(actual, expected)
        return ok, f"{path} == {_short(expected)}" if ok else f"{path}: expected {_short(expected)}, got {_short(actual)}"
    if op == "neq":
        ok = not strict_eq(actual, expected)
        return ok, f"{path} != {_short(expected)}" if ok else f"{path}: expected anything but {_short(expected)}"
    if op == "contains":
        if isinstance(actual, str):
            ok = isinstance(expected, str) and expected.casefold() in actual.casefold()
        elif isinstance(actual, list):
            ok = any(strict_eq(x, expected) for x in actual)
        elif isinstance(actual, dict):
            ok = isinstance(expected, str) and expected in actual
        else:
            return False, f"{path}: contains on a {type(actual).__name__}"
        return ok, f"{path} contains {_short(expected)}" if ok else f"{path}: {_short(actual)} does not contain {_short(expected)}"
    # len_gte
    if not isinstance(actual, (list, dict, str)):
        return False, f"{path}: len_gte on a {type(actual).__name__}"
    if isinstance(expected, bool) or not isinstance(expected, int):
        return False, f"{path}: len_gte needs an integer value, got {_short(expected)}"
    ok = len(actual) >= expected
    return ok, f"{path} has {len(actual)} >= {expected}" if ok else f"{path}: length {len(actual)} < {expected}"


def evaluate(state: dict, check: dict) -> tuple[bool, str]:
    """Return (passed, explanation). Pure; never raises."""
    try:
        if not isinstance(check, dict):
            return False, "malformed check: not an object"
        if "all" in check:
            mode, conds = "all", check["all"]
        elif "any" in check:
            mode, conds = "any", check["any"]
        else:
            return False, "malformed check: needs an 'all' list"
        if not isinstance(conds, list) or not conds:
            return False, f"malformed check: '{mode}' must be a non-empty list"
        results = [_eval_condition(state, c) for c in conds]
        passed = [r for r in results if r[0]]
        if mode == "all":
            ok = len(passed) == len(results)
            failed = [msg for good, msg in results if not good]
            if ok:
                return True, f"all {len(results)} conditions passed"
            return False, f"{len(failed)} of {len(results)} conditions failed: " + "; ".join(failed)
        ok = bool(passed)
        if ok:
            return True, f"any: passed ({passed[0][1]})"
        return False, "any: none passed: " + "; ".join(msg for _, msg in results)
    except Exception as exc:  # pure function, but never let a bad check crash a verifier
        return False, f"check raised {type(exc).__name__}: {exc}"
