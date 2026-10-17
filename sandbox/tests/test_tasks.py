"""tasks.json matches contracts.md §1, and every check is meaningful on its initial state."""

import json
from pathlib import Path

import pytest

from sandbox import state as st
from sandbox.checks import OPS, evaluate, resolve, _MISSING

TASKS = json.loads((Path(__file__).resolve().parents[1] / "tasks.json").read_text())
FIELDS = {"id", "request", "start_path", "initial_state_overrides", "check", "steps_min", "tags"}


def initial_state(task):
    return st.apply_overrides(st.default_state(), task["initial_state_overrides"])


def test_schema_exact():
    assert isinstance(TASKS, list) and len(TASKS) == 12
    for t in TASKS:
        assert set(t) == FIELDS, t["id"]
        assert isinstance(t["id"], str) and isinstance(t["request"], str) and t["request"].strip()
        assert t["start_path"].startswith("/") and "sid=" not in t["start_path"]
        assert isinstance(t["initial_state_overrides"], dict)
        assert isinstance(t["steps_min"], int) and t["steps_min"] >= 1
        assert isinstance(t["tags"], list) and all(isinstance(x, str) for x in t["tags"])
        assert set(t["check"]) == {"all"} and t["check"]["all"]
        for c in t["check"]["all"]:
            assert set(c) <= {"path", "op", "value"} and c["op"] in OPS
    assert len({t["id"] for t in TASKS}) == 12


def test_mix():
    level = lambda t: [x for x in t["tags"] if x in ("easy", "medium", "hard")]
    assert all(len(level(t)) == 1 for t in TASKS)
    counts = {k: sum(level(t) == [k] for t in TASKS) for k in ("easy", "medium", "hard")}
    assert counts == {"easy": 4, "medium": 5, "hard": 3}
    for t in TASKS:
        lv = level(t)[0]
        assert (t["steps_min"] == 1) if lv == "easy" else (2 <= t["steps_min"] <= 3 if lv == "medium" else t["steps_min"] >= 4), t["id"]
    assert sum("icon_only" in t["tags"] for t in TASKS) >= 3
    assert sum("canvas" in t["tags"] for t in TASKS) >= 2
    assert sum("scroll" in t["tags"] for t in TASKS) >= 2


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_check_fails_on_initial_state(task):
    passed, why = evaluate(initial_state(task), task["check"])
    assert not passed, f"{task['id']} is already satisfied at start: {why}"


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_check_paths_exist_in_schema(task):
    """Every path resolves on the initial state, except appended items and exists:false probes."""
    s = initial_state(task)
    for c in task["check"]["all"]:
        if c["op"] == "exists" and c.get("value") is False:
            continue
        p = c["path"]
        if p.startswith(("billing.payments.", "outages.reports.", "messages.sent.")):
            continue  # list items created by the task
        assert resolve(s, p) is not _MISSING, f"{task['id']}: {p}"


def test_start_paths_render():
    from fastapi.testclient import TestClient
    from sandbox.app import app
    client = TestClient(app)
    for t in TASKS:
        client.post("/api/reset", json={"sid": f"render-{t['id']}", "task_id": t["id"]})
        sep = "&" if "?" in t["start_path"] else "?"
        r = client.get(f"{t['start_path']}{sep}sid=render-{t['id']}")
        assert r.status_code == 200, t["id"]
