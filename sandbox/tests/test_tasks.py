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


# --- reward correctness: the solution passes; solution + a collateral change fails --------------

SOLUTIONS = {
    "text-size-large": [("settings.accessibility", {"key": "text_size", "value": "L"})],
    "download-september-statement": [("billing.download", {"statement_id": "2026-09"})],
    "mark-all-read": [("messages.mark_all_read", {})],
    "turn-on-autopay": [("billing.autopay", {"enabled": True})],
    "high-contrast": [("settings.accessibility", {"key": "high_contrast", "value": True})],
    "sms-billing-off": [("settings.notification", {"category": "billing", "channel": "sms", "enabled": False})],
    "max-usage-month": [("usage.select_month", {"month": "2026-07"})],
    "report-outage-riverside": [("outages.report", {"district": "riverside"})],
    "star-landlord-message": [("messages.star", {"id": "msg-landlord", "starred": True})],
    "pay-50": [("billing.pay", {"amount_cents": 5000})],
    "message-support-meter": [("messages.send", {"to": "support", "subject": "Meter", "body": "My meter broke this morning."})],
    "stop-promotions": [("settings.notification", {"category": "promotions", "channel": c, "enabled": False})
                        for c in ("email", "sms", "push")],
}

# Costly side effects no task may cause unless it is that task.
GENERIC = {
    "pays": ("billing.pay", {"amount_cents": 100}),
    "turns-on-autopay": ("billing.autopay", {"enabled": True}),
    "sends-a-message": ("messages.send", {"to": "billing", "body": "hello"}),
    "reports-an-outage": ("outages.report", {"district": "hillcrest"}),
}
SPECIFIC = {
    "text-size-large": [("settings.accessibility", {"key": "high_contrast", "value": True})],
    "high-contrast": [("settings.accessibility", {"key": "reduce_motion", "value": True})],
    "sms-billing-off": [("settings.notification", {"category": "service", "channel": "push", "enabled": True})],
    "stop-promotions": [("settings.notification", {"category": "surveys", "channel": "email", "enabled": False})],
    "star-landlord-message": [("messages.star", {"id": "msg-outage-hillcrest", "starred": True})],
    "download-september-statement": [("billing.download", {"statement_id": "2026-08"})],
}


def solved(task, extra=()):
    s = initial_state(task)
    for action, payload in [*SOLUTIONS[task["id"]], *extra]:
        st.apply_action(s, action, payload)
    return s


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_solution_passes(task):
    passed, why = evaluate(solved(task), task["check"])
    assert passed, why


COLLATERAL = [(t, name, act) for t in TASKS
              for name, act in [*GENERIC.items(), *((f"specific-{i}", a) for i, a in enumerate(SPECIFIC.get(t["id"], [])))]
              if (t["id"], name) != ("turn-on-autopay", "turns-on-autopay")]  # that one is the task itself


@pytest.mark.parametrize("task,name,extra", COLLATERAL, ids=[f"{t['id']}+{n}" for t, n, _ in COLLATERAL])
def test_solution_plus_collateral_change_fails(task, name, extra):
    passed, why = evaluate(solved(task, [extra]), task["check"])
    assert not passed, f"{task['id']} still passes after {name}: {why}"


@pytest.mark.parametrize("body", ["My meter is broken.", "my METER broke", "The meter seems broken to me"])
def test_meter_message_wording_variants_pass(body):
    task = next(t for t in TASKS if t["id"] == "message-support-meter")
    s = initial_state(task)
    st.apply_action(s, "messages.send", {"to": "support", "body": body})
    assert evaluate(s, task["check"])[0]
