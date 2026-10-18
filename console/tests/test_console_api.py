"""Console API tests: fake agents for the API contract, plus mock agent units (headless Chromium).

The 12 sandbox tasks end to end through the console with the real mock agent are in test_mock_e2e.py."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from PIL import Image

from console import mock_agent
from console.runner import VERDICT_LABEL
from console.tests.conftest import messages, read_sse

TASK = "text-size-large"


def fake_agent(*, steps: int = 2, block: bool = False, error: Exception | None = None,
               set_text_size: str | None = None, trace_root: Path | None = None):
    """(run_task, make_cfg) emitting `steps` click steps with real PNGs under trace_root/<agent run id>/."""

    async def run_task(request, *, start_url, cfg, on_event=None):
        rid = f"fake-{time.time_ns()}"
        d = Path(cfg["trace_root"]) / rid
        d.mkdir(parents=True)

        async def ev(step, kind, payload, **extra):
            await on_event({"run_id": rid, "step": step, "kind": kind, "ts": time.time(), "payload": payload, **extra})

        await ev(0, "start", {"request": request, "start_url": start_url, "config": cfg["options"]})
        Image.new("RGB", (32, 20), "white").save(d / "step_00.png")
        await ev(0, "observation", {"text": "start"}, screenshot_path=str(d / "step_00.png"))
        if set_text_size:  # stands in for the agent's clicks: the UI's own mutation endpoint, with the session cookie
            sid = parse_qs(urlsplit(start_url).query)["sid"][0]
            base = f"{urlsplit(start_url).scheme}://{urlsplit(start_url).netloc}"
            async with httpx.AsyncClient(cookies={"sid": sid}, trust_env=False) as c:
                r = await c.post(f"{base}/api/action", json={"type": "settings.accessibility",
                                                             "payload": {"key": "text_size", "value": set_text_size}})
                r.raise_for_status()
        for i in range(1, steps + 1):
            Image.new("RGB", (32, 20), "red").save(d / f"step_{i:02d}_overlay.png")
            Image.new("RGB", (32, 20), "blue").save(d / f"step_{i:02d}.png")
            await ev(i, "grounding", {"target": f"target {i}", "x": 10, "y": 10}, overlay_path=str(d / f"step_{i:02d}_overlay.png"),
                     timings_ms={"ground": 5.0})
            await ev(i, "observation", {"text": f"after {i}"}, screenshot_path=str(d / f"step_{i:02d}.png"))
        if block:
            await asyncio.sleep(60)
        if error:
            raise error
        await ev(steps + 1, "done", {"success_claimed": True, "summary": "did it", "steps": steps + 1, "reason": "done"})
        return {"run_id": rid, "success_claimed": True, "summary": "did it", "steps": steps + 1, "trace_dir": str(d)}

    def make_cfg(options):
        return {"options": options, "trace_root": str(trace_root)}

    return run_task, make_cfg


def start(base, request="Make the text bigger", task_id=None, options=None, expect=200):
    r = httpx.post(f"{base}/api/run", json={"request": request, "task_id": task_id, "options": options or {}})
    assert r.status_code == expect, r.text
    return r.json()


def wait_status(base, run_id, statuses, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = httpx.get(f"{base}/api/runs/{run_id}").json()
        if info["status"] in statuses:
            return info
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached {statuses}: {info}")


# --- config, tasks, static -----------------------------------------------------------------


def test_config_tasks_and_index(console, tmp_path):
    static = tmp_path / "static"
    base, _ = console(fake_agent(trace_root=tmp_path), static_dir=static)
    cfg = httpx.get(f"{base}/api/config").json()
    assert cfg == {"agent": "mock", "sandbox_url": cfg["sandbox_url"], "headed": False,
                   "defaults": {"grounder": "base", "thinking": "retry", "lightning": "local"},
                   "asr": {"available": True, "model": "stub", "label": "stub"}}
    tasks = httpx.get(f"{base}/api/tasks").json()
    assert len(tasks) == 12 and all(set(t) == {"id", "request", "start_path", "tags", "steps_min"} for t in tasks)

    assert httpx.get(f"{base}/").status_code == 404  # not built yet
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>console</title>")
    (static / "app.js").write_text("console.log(1)")
    r = httpx.get(f"{base}/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "console" in r.text
    js = httpx.get(f"{base}/static/app.js")
    assert js.text == "console.log(1)"
    # /static/* revalidates on every load (an edited console.js must not stay cached under a fresh index.html)
    assert js.headers["cache-control"] == "no-cache" and js.headers["etag"]
    again = httpx.get(f"{base}/static/app.js", headers={"If-None-Match": js.headers["etag"]})
    assert again.status_code == 304 and again.headers["cache-control"] == "no-cache"
    (static / "app.js").write_text("console.log('edited')")  # another size: a new ETag even within one mtime tick
    edited = httpx.get(f"{base}/static/app.js", headers={"If-None-Match": js.headers["etag"]})
    assert edited.status_code == 200 and edited.text == "console.log('edited')"
    assert r.headers["cache-control"] == "no-store"


# --- runs and the SSE stream -----------------------------------------------------------------


def test_stream_order_replay_and_last_event_id(console, tmp_path):
    base, _ = console(fake_agent(steps=2, trace_root=tmp_path / "elsewhere"))
    run = start(base, options={"grounder": "grounder", "thinking": "off", "lightning": "local"})
    assert run["run_id"].startswith("c-") and run["sid"] == "console-" + run["run_id"]
    assert run["start_url"].endswith("/?sid=" + run["sid"])

    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    assert [m["id"] for m in stream] == list(range(1, len(stream) + 1))
    kinds = [m["event"] if m["event"] != "step" else m["data"]["kind"] for m in stream]
    assert kinds == ["status", "status", "start", "observation", "grounding", "observation", "grounding",
                     "observation", "done", "result", "status", "end"]
    assert [m["data"]["status"] for m in stream if m["event"] == "status"] == ["starting", "running", "done"]
    assert stream[2]["data"]["payload"]["config"] == {"grounder": "grounder", "thinking": "off", "lightning": "local"}
    assert stream[-3]["data"]["success_claimed"] is True and "verdict" not in [m["event"] for m in stream]

    info = httpx.get(f"{base}/api/runs/{run['run_id']}").json()
    assert info["status"] == "done" and info["n_events"] == len(stream) and info["verdict"] is None
    assert info["agent_run_id"] == stream[2]["data"]["run_id"] and info["result"]["summary"] == "did it"
    assert info["options"] == {"grounder": "grounder", "thinking": "off", "lightning": "local"}

    # replay after the fact is identical; Last-Event-ID skips what the client already has
    assert messages(read_sse(f"{base}/api/runs/{run['run_id']}/events")) == stream
    tail = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events", headers={"Last-Event-ID": "5"}))
    assert tail == stream[5:]
    # nothing left after `end`: 204 tells EventSource to stop reconnecting
    r = httpx.get(f"{base}/api/runs/{run['run_id']}/events", headers={"Last-Event-ID": str(len(stream))})
    assert r.status_code == 204


def test_live_subscriber_sees_everything_once(console, tmp_path):
    """Subscribe while the run is blocked mid-way: backlog + live messages, no gaps or duplicates."""
    base, _ = console(fake_agent(steps=1, block=True, trace_root=tmp_path), ping_s=0.2)
    run = start(base)
    wait_status(base, run["run_id"], {"running"})
    time.sleep(0.3)
    result: list = []
    reader = threading.Thread(
        target=lambda: result.extend(read_sse(f"{base}/api/runs/{run['run_id']}/events")))
    reader.start()
    time.sleep(0.6)  # at least one ping while blocked
    assert httpx.post(f"{base}/api/runs/{run['run_id']}/stop").json() == {"ok": True}
    reader.join(10)
    assert any(m.get("comment") == "ping" for m in result)
    stream = messages(result)
    assert [m["id"] for m in stream] == list(range(1, len(stream) + 1))
    assert stream[-2]["data"]["status"] == "stopped" and stream[-1]["event"] == "end"


def test_images_served_and_validated(console, tmp_path):
    base, _ = console(fake_agent(steps=1, trace_root=tmp_path / "agent-runs"))
    run = start(base)
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    steps = [m["data"] for m in stream if m["event"] == "step"]
    shot = next(s for s in steps if s.get("screenshot_url"))
    over = next(s for s in steps if s.get("overlay_url"))
    rid = steps[0]["run_id"]
    assert shot["screenshot_url"] == f"/runs/{rid}/step_00.png"
    assert over["overlay_url"] == f"/runs/{rid}/step_01_overlay.png"
    for url in (shot["screenshot_url"], over["overlay_url"]):
        r = httpx.get(base + url)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    (tmp_path / "agent-runs" / rid / "notes.txt").write_text("secret")
    for bad in (f"/runs/{rid}/notes.txt", f"/runs/{rid}/step_1.png", f"/runs/{rid}/step_0001.png",
                f"/runs/{rid}/step_09.png", f"/runs/{rid}/..%2Fstep_00.png", "/runs/..%2F..%2Fetc/step_00.png",
                "/runs/unknown-run/step_00.png", f"/runs/{rid}/step_00.jpg"):
        assert httpx.get(base + bad).status_code == 404, bad


def test_one_run_at_a_time_and_stop(console, tmp_path):
    base, _ = console(fake_agent(steps=1, block=True, trace_root=tmp_path))
    first = start(base)
    wait_status(base, first["run_id"], {"running"})
    busy = start(base, expect=409)
    assert busy["active_run_id"] == first["run_id"] and busy["error"]

    t = time.time()
    assert httpx.post(f"{base}/api/runs/{first['run_id']}/stop", timeout=15).json() == {"ok": True}
    assert time.time() - t < 5
    info = httpx.get(f"{base}/api/runs/{first['run_id']}").json()
    assert info["status"] == "stopped" and info["result"] is None
    assert httpx.post(f"{base}/api/runs/{first['run_id']}/stop").json() == {"ok": True}  # idempotent
    stream = messages(read_sse(f"{base}/api/runs/{first['run_id']}/events"))
    assert stream[-2]["data"] == {"status": "stopped", "message": "stopped by the user"} and stream[-1]["event"] == "end"

    second = start(base)  # the slot is free again
    assert second["run_id"] != first["run_id"]
    httpx.post(f"{base}/api/runs/{second['run_id']}/stop", timeout=15)


def test_agent_error_is_reported(console, tmp_path):
    base, _ = console(fake_agent(steps=1, error=RuntimeError("grounder exploded"), trace_root=tmp_path))
    run = start(base)
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    assert stream[-2]["data"] == {"status": "error", "message": "RuntimeError: grounder exploded"}
    assert "result" not in [m["event"] for m in stream]


# --- verdicts against the live sandbox ------------------------------------------------------


@pytest.mark.parametrize("size, passed", [("L", True), ("S", False), (None, False)])
def test_verdict_from_sandbox_state(console, tmp_path, size, passed):
    base, _ = console(fake_agent(steps=1, set_text_size=size, trace_root=tmp_path))
    run = start(base, request="Make the text bigger, I can't read this.", task_id=TASK)
    assert run["start_url"].endswith("/settings?tab=accessibility&sid=" + run["sid"])
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    assert [m["event"] for m in stream][-4:] == ["result", "verdict", "status", "end"]
    verdict = stream[-3]["data"]
    assert verdict["task_id"] == TASK and verdict["passed"] is passed and verdict["label"] == VERDICT_LABEL
    assert verdict["explanation"]
    assert httpx.get(f"{base}/api/runs/{run['run_id']}").json()["verdict"] == verdict


def test_verdict_after_stop(console, tmp_path):
    base, _ = console(fake_agent(steps=1, block=True, set_text_size="L", trace_root=tmp_path))
    run = start(base, task_id=TASK)
    wait_status(base, run["run_id"], {"running"})
    httpx.post(f"{base}/api/runs/{run['run_id']}/stop", timeout=15)
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    assert [m["event"] for m in stream][-3:] == ["verdict", "status", "end"]
    assert stream[-3]["data"]["passed"] is True and stream[-2]["data"]["status"] == "stopped"


# --- errors -------------------------------------------------------------------------------------


@pytest.mark.parametrize("body", [
    {"request": ""}, {"request": "   "}, {"request": "x" * 501}, {"request": 5}, {},
    {"request": "hi", "task_id": "no-such-task"}, {"request": "hi", "task_id": 3},
    {"request": "hi", "options": {"grounder": "big"}}, {"request": "hi", "options": {"thinking": True}},
    {"request": "hi", "options": {"lightning": "cloud"}}, {"request": "hi", "options": {"headed": True}},
    {"request": "hi", "options": "fast"}, ["hi"],
])
def test_bad_run_requests_400(console, tmp_path, body):
    base, _ = console(fake_agent(trace_root=tmp_path))
    r = httpx.post(f"{base}/api/run", json=body)
    assert r.status_code == 400 and r.json()["error"], body


def test_bad_json_400_and_limits(console, tmp_path):
    base, _ = console(fake_agent(trace_root=tmp_path))
    r = httpx.post(f"{base}/api/run", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    run = start(base, request="x" * 500)  # exactly at the limit is fine
    read_sse(f"{base}/api/runs/{run['run_id']}/events")


def test_unknown_run_404(console, tmp_path):
    base, _ = console(fake_agent(trace_root=tmp_path))
    for method, path in (("GET", "/api/runs/c-nope"), ("GET", "/api/runs/c-nope/events"), ("POST", "/api/runs/c-nope/stop")):
        r = httpx.request(method, base + path)
        assert r.status_code == 404 and r.json()["error"]


def test_sandbox_unreachable_503(console, tmp_path):
    from console.tests.conftest import free_port

    base, _ = console(fake_agent(trace_root=tmp_path), sandbox_url=f"http://127.0.0.1:{free_port()}")
    r = httpx.post(f"{base}/api/run", json={"request": "hi"})
    assert r.status_code == 503 and "sandbox" in r.json()["error"]
    assert httpx.post(f"{base}/api/run", json={"request": "hi"}).status_code == 503  # not 409: no run is left behind


def test_real_agent_unavailable_503(console, monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.core", None)  # makes `from agent.core import run_task` fail
    base, _ = console(None, agent="real")
    r = httpx.post(f"{base}/api/run", json={"request": "hi"})
    assert r.status_code == 503 and "agent.core not available yet (T2); use --agent mock" in r.json()["error"]
    assert httpx.get(f"{base}/api/config").json()["agent"] == "real"


# --- mock agent units (headless Chromium against the live sandbox; the 12 tasks are in test_mock_e2e.py) ---


def _write_script(tmp_path, steps, request="Mock test: click the middle", **extra):
    path = tmp_path / "scripts.json"
    path.write_text(json.dumps({"tasks": {"t": {"request": request, "start_screen": "A page.", "steps": steps,
                                                **extra}}}))
    return path


def _run_mock(tmp_path, url, scripts, request="mock TEST:  click the middle", cancel_after: str | None = None):
    cfg = mock_agent.MockConfig(runs_dir=str(tmp_path / "runs"), step_delay_s=0.0 if not cancel_after else 0.5,
                                scripts_path=str(scripts))
    events: list[dict] = []

    async def go():
        async def on_event(ev):
            events.append(ev.to_dict())

        task = asyncio.ensure_future(mock_agent.run_task(request, start_url=url, cfg=cfg, on_event=on_event))
        if cancel_after:
            while not any(e["kind"] == cancel_after for e in events):
                await asyncio.sleep(0.02)
            task.cancel()
        return await task

    try:
        return asyncio.run(go()), events
    except asyncio.CancelledError:
        return None, events


def test_mock_agent_scripted_click_and_overlay(tmp_path, sandbox_url):
    scripts = _write_script(tmp_path, [
        {"tool": "click", "arguments": {"target_description": "the middle of the page"}, "x": 640, "y": 400,
         "reasoning": "Testing.", "screen_after": "Still a page.", "path_after": "/"},
        {"tool": "scroll", "arguments": {"direction": "down", "amount": "small"}},
        {"tool": "done", "arguments": {"summary": "Clicked it.", "success": True}},
    ])
    result, events = _run_mock(tmp_path, f"{sandbox_url}/?sid=mock-test-1", scripts)
    assert result["success_claimed"] is True and result["steps"] == 3 and result["summary"] == "Clicked it."
    kinds = [e["kind"] for e in events]
    assert kinds == ["start", "observation", "plan", "tool_call", "grounding", "action", "observation",
                     "plan", "tool_call", "action", "observation", "plan", "tool_call", "action", "done"]
    grounding = next(e for e in events if e["kind"] == "grounding")
    assert grounding["payload"]["point_1000"] == [500, 500] and Path(grounding["overlay_path"]).is_file()
    action = next(e for e in events if e["kind"] == "action")
    assert action["payload"]["click"] == {"x": 640.0, "y": 400.0} and isinstance(action["payload"]["screen_changed"], bool)
    assert next(e for e in events if e["kind"] == "tool_call")["payload"]["description"] == "Clicking: the middle of the page"
    # timings split like agent/session.py: act (the action alone) + settle on the action, screenshot on the observation
    for e in events:
        if e["kind"] == "action" and e["payload"]["name"] != "done":
            assert set(e["timings_ms"]) == {"act", "settle"} and e["timings_ms"]["settle"] >= 250, e
        elif e["kind"] == "observation":
            assert set(e["timings_ms"]) == {"screenshot"}, e
        else:
            assert "timings_ms" not in e, e  # no plan / ground / summary latency: no model ran
    # the prompt fields agent/trace.py documents, declared as "mock"
    assert all((e["payload"]["prompt_version"], e["payload"]["prompt_sha"]) == ("mock", "mock")
               for e in events if e["kind"] in ("plan", "grounding"))
    assert all(e["payload"]["summary"] == {"model": "mock (no VLM)", "prompt_version": "mock", "prompt_sha": "mock"}
               for e in events if e["kind"] == "observation")
    trace = Path(result["trace_dir"])
    assert {p.name for p in trace.iterdir()} == {"events.jsonl", "step_00.png", "step_01.png", "step_01_overlay.png",
                                                 "step_02.png"}


def test_mock_script_start_page_matching():
    """A script replays only from the page it was recorded on: same path and query, sid aside."""
    script = {"start_path": "/settings?tab=accessibility"}
    for url in ("http://s/settings?tab=accessibility&sid=x", "http://s/settings?sid=x&tab=accessibility"):
        assert mock_agent.recorded_elsewhere(script, url) is None, url
    for url in ("http://s/?sid=x", "http://s/settings?sid=x", "http://s/settings?tab=billing&sid=x"):
        assert mock_agent.recorded_elsewhere(script, url) == "/settings?tab=accessibility", url
    assert mock_agent.recorded_elsewhere({"start_path": "/"}, "http://s/?sid=x") is None
    assert mock_agent.recorded_elsewhere({}, "http://s/messages?sid=x") is None  # older scripts: not checked
    assert mock_agent.show_start("http://s:1/settings?sid=x&tab=accessibility") == "/settings?tab=accessibility"


def test_mock_agent_replays_a_script_only_from_its_start_page(tmp_path, sandbox_url):
    scripts = _write_script(tmp_path, [
        {"tool": "scroll", "arguments": {"direction": "down", "amount": "small"}, "reasoning": "Testing.",
         "screen_after": "The inbox, scrolled."},
        {"tool": "done", "arguments": {"summary": "Scrolled the inbox.", "success": True}},
    ], start_path="/messages")
    result, events = _run_mock(tmp_path, f"{sandbox_url}/messages?sid=mock-test-6", scripts)
    assert result["success_claimed"] is True and result["summary"] == "Scrolled the inbox."

    result, events = _run_mock(tmp_path, f"{sandbox_url}/?sid=mock-test-7", scripts)  # the typed-text start page
    assert [e["kind"] for e in events] == ["start", "observation", "plan", "tool_call", "action", "done"]
    assert result["success_claimed"] is False and result["steps"] == 1
    assert "Its script for this request starts on /messages, but this run started on /," in result["summary"]
    assert events[1]["payload"]["observation"]["screen_summary"] == mock_agent.NO_SUMMARY
    assert events[-1]["payload"]["reason"] == "done" and not [e for e in events if e["kind"] == "error"]


def _driver_pids() -> set[int]:
    """Playwright node drivers started by this test process (`node .../cli.js run-driver`)."""
    psutil = pytest.importorskip("psutil")
    out = set()
    for proc in psutil.Process().children(recursive=True):
        try:
            if "run-driver" in " ".join(proc.cmdline()):
                out.add(proc.pid)
        except psutil.Error:
            pass
    return out


def test_open_page_cancelled_while_starting_leaks_no_driver():
    """A Stop in a mock run's first ~100 ms cancels open_page while Playwright's node driver is starting. That
    used to skip the driver's cleanup: one ~130 MB node process left per early stop, for the life of the console."""
    before = _driver_pids()

    async def go():
        for delay in (0.0, 0.01, 0.03, 0.06, 0.1, 0.3):
            async def use():
                async with mock_agent.open_page():
                    await asyncio.sleep(30)

            task = asyncio.ensure_future(use())
            await asyncio.sleep(delay)
            task.cancel()
            if delay == 0.03:  # a second Stop must not cut the cleanup short either
                await asyncio.sleep(0.005)
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        deadline = time.time() + 5
        while _driver_pids() - before and time.time() < deadline:  # a stopped driver exits on its own
            await asyncio.sleep(0.1)
        return _driver_pids() - before

    assert asyncio.run(go()) == set()


def test_mock_agent_stale_and_malformed_scripts(tmp_path, sandbox_url):
    stale = _write_script(tmp_path, [
        {"tool": "click", "arguments": {"target_description": "x"}, "x": 10, "y": 10, "path_after": "/nowhere"},
        {"tool": "done", "arguments": {"summary": "never", "success": True}},
    ])
    result, events = _run_mock(tmp_path, f"{sandbox_url}/?sid=mock-test-2", stale)
    assert result["success_claimed"] is False
    err = next(e for e in events if e["kind"] == "error")
    assert "stale" in err["payload"]["message"] and events[-1]["payload"]["reason"] == "error"

    bad = _write_script(tmp_path, [{"tool": "click", "arguments": {"target_description": "x"}}])  # no x/y
    result, events = _run_mock(tmp_path, f"{sandbox_url}/?sid=mock-test-3", bad)
    assert events[-1]["kind"] == "done" and events[-1]["payload"]["reason"] == "error"

    missing = tmp_path / "missing.json"
    result, events = _run_mock(tmp_path, f"{sandbox_url}/?sid=mock-test-4", missing)
    assert [e["kind"] for e in events if e["kind"] == "error"] == ["error"]
    assert events[-1]["payload"]["reason"] == "done" and result["success_claimed"] is False


def test_mock_agent_cancel_emits_stopped(tmp_path, sandbox_url):
    scripts = _write_script(tmp_path, [
        {"tool": "click", "arguments": {"target_description": "x"}, "x": 10, "y": 10},
        {"tool": "done", "arguments": {"summary": "never", "success": True}},
    ])
    result, events = _run_mock(tmp_path, f"{sandbox_url}/?sid=mock-test-5", scripts, cancel_after="observation")
    assert result is None
    assert events[-1]["kind"] == "done" and events[-1]["payload"]["reason"] == "stopped"
