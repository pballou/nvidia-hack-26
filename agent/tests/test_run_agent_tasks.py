"""scripts/run_agent_tasks.py batch logic: repeats, concurrency, the adapter pin, memory pause (no models)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_agent_tasks.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_agent_tasks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rat = _load()
TASKS = [{"id": f"t{i}", "request": f"do {i}", "start_path": "/", "check": {"all": []}, "tags": []} for i in range(3)]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(TASKS))
    state = {"in_flight": 0, "max_in_flight": 0, "calls": []}

    async def fake_run_one(task, cfg, sandbox, http, pool, quiet):
        state.setdefault("cfgs", []).append(cfg)
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        state["calls"].append(task["id"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        return {"task_id": task["id"], "tags": [], "verified_success": True, "success_claimed": True, "steps": 1,
                "wall_s": 0.0, "reason": "done", "run_id": f"r{len(state['calls'])}", "trace_dir": None,
                "explanation": "ok", "stats": {}}

    async def fake_pool(cfg):
        return object()

    async def no_close():
        return None

    monkeypatch.setattr(rat, "run_one", fake_run_one)
    from agent import browser

    monkeypatch.setattr(browser, "get_pool", fake_pool)
    monkeypatch.setattr(browser, "close_pool", no_close)

    def run(*extra):
        return rat.main(["--tasks", str(tasks_file), "--sandbox-url", "http://sandbox.test", "--quiet", *extra]), None

    monkeypatch.setenv("AGENT_RUNS_DIR", str(tmp_path / "runs"))  # summaries land in tmp
    state["run"] = run
    state["tmp"] = tmp_path
    return state


def _summary(tmp: Path) -> dict:
    (path,) = sorted((tmp / "runs").glob("summary_*.json"))
    return json.loads(path.read_text())


def test_repeats_and_concurrency(harness):
    code, _ = harness["run"]("--repeats", "3", "--concurrency", "2")
    s = _summary(harness["tmp"])
    assert code == 0 and s["n_tasks"] == 9 and s["repeats"] == 3 and s["concurrency"] == 2
    assert [r["repeat"] for r in s["tasks"]] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert [r["task_id"] for r in s["tasks"][:3]] == ["t0", "t1", "t2"]
    assert harness["max_in_flight"] == 2


def test_clarify_is_forced_off(harness, monkeypatch):
    """The harness measures the browser loop on the fixed phrasings: no pre-flight ask (agent/clarify.py)."""
    monkeypatch.setenv("AGENT_CLARIFY", "on")
    code, _ = harness["run"]()
    assert code == 0 and harness["cfgs"] and all(c.clarify is False for c in harness["cfgs"])
    assert _summary(harness["tmp"])["config"]["clarify"] is False


def test_adapter_change_aborts_the_arm(harness, monkeypatch):
    roots = iter(["/ckpt/a", "/ckpt/a", "/ckpt/a", "/ckpt/a", "/ckpt/b"])

    async def fake_root(http, cfg, name):
        return next(roots, "/ckpt/b")

    monkeypatch.setattr(rat, "served_root", fake_root)
    code, _ = harness["run"]("--repeats", "3", "--pin-grounder-adapter")
    s = _summary(harness["tmp"])
    assert code == 3 and s["grounder_adapter"] == "/ckpt/a" and "changed" in s["aborted"]
    assert s["n_tasks"] < 9 and s["tasks"][-1].get("adapter_changed") is True


def test_missing_adapter_refuses_to_start(harness, monkeypatch):
    async def none_root(http, cfg, name):
        return None

    monkeypatch.setattr(rat, "served_root", none_root)
    with pytest.raises(SystemExit):
        harness["run"]("--pin-grounder-adapter")
    assert harness["calls"] == []


def test_memory_pressure_pauses_then_resumes(harness, monkeypatch):
    levels = iter([2, 2, 1])
    monkeypatch.setattr(rat, "memory_pressure_level", lambda: next(levels, 1))
    real_sleep = asyncio.sleep

    async def fast_sleep(s):
        await real_sleep(0)

    monkeypatch.setattr(rat.asyncio, "sleep", fast_sleep)
    code, _ = harness["run"]("--ids", "t0", "--pause-on-memory-pressure")
    assert code == 0 and harness["calls"] == ["t0"]


def test_bad_counts_are_rejected(harness):
    with pytest.raises(SystemExit):
        harness["run"]("--repeats", "0")
