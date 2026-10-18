"""The request check (agent/clarify.py) in the console backend: defaults and flags, follow-up runs (`clarifies`),
the restated-request late task match (POST /api/runs/{id}/task_match), what the logs make of them (needs_input,
clarify, follow-up links) and the mock agent's clarify flows.

Fake agents for the API rules (no browser, no model). The mock's ask flows open no browser either; only the
follow-up that replays pay-50 drives headless Chromium (marked e2e). The Logs page itself (the clarify row, the
Needs input chip, the follow-up links) is checked in a browser in test_logs_e2e.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from agent import clarify as agent_clarify
from agent.clarify import follow_up_request
from console import __main__ as console_main
from console import logs, mock_agent
from console.agent_bridge import get_agent, real_overrides
from console.app import ConsoleSettings, parse_run_body
from console.runner import VERDICT_LABEL
from console.tests.conftest import free_port, messages, read_sse

REPO = Path(__file__).resolve().parents[2]
DEMO_UP = REPO / "scripts" / "demo_up.sh"
TASKS = json.loads((REPO / "sandbox" / "tasks.json").read_text())
TASKS_BY_ID = {t["id"]: t for t in TASKS}
GARBLED, ASK = "Allors towards my bill.", "Pay something toward my bill"
QUESTIONS = {GARBLED: "What would you like to do with your bill?", ASK: "How much would you like to pay toward your bill?"}
RESTATED = "Pay $50 toward my bill."
RESTATEMENTS = {follow_up_request(GARBLED, "fifty dollars"): RESTATED, follow_up_request(ASK, "fifty dollars"): RESTATED}
CLARIFY_SHAPE = {"model": "lightning", "prompt_version": "clarify-v4", "prompt_sha": "0123456789"}
ASK_USAGE = {"prompt_tokens": 900, "completion_tokens": 20}
RESTATED_LINE = 'Checked as task: pay-50 (matched from the agent\'s restatement: "Pay $50 toward my bill.")'


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No AGENT_* from this machine (AGENT_CLARIFY would change the real config); real traces go to tmp."""
    for key in list(os.environ):
        if key.startswith("AGENT_") or key == "CONSOLE_VLM_LABEL":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENT_RUNS_DIR", str(tmp_path / "agent-runs"))
    return tmp_path / "agent-runs"


class Cfg:
    def __init__(self, root: Path, options: dict):
        self.run_dir_root, self.options = root, options


def clarifying_agent(root: Path, seen: list, *, gate: threading.Event | None = None, pay_cents: int | None = None):
    """(run_task, make_cfg) in agent/core.py's shapes: start (with parent_run_id), then with options.clarify "on" a
    step-0 clarify event: ask for QUESTIONS (done reason needs_input, as soon as it asked), else proceed with
    RESTATEMENTS' text (or the request trimmed). Then it waits on `gate`, optionally pays `pay_cents` through the
    sandbox UI's own mutation endpoint (standing in for the agent's clicks), and ends with done. It writes
    events.jsonl under cfg.run_dir_root and records what it was given in `seen`."""

    async def run_task(request, *, start_url, cfg, on_event=None, **kwargs):
        seen.append({"request": request, "start_url": start_url, "options": dict(cfg.options), "kwargs": kwargs})
        rid = f"20261018-{time.strftime('%H%M%S')}-{secrets.token_hex(3)}"
        d = Path(cfg.run_dir_root) / rid
        d.mkdir(parents=True)
        f = (d / "events.jsonl").open("a")

        async def emit(step, kind, payload, **extra):
            e = {"run_id": rid, "step": step, "kind": kind, "ts": round(time.time(), 3), "payload": payload, **extra}
            f.write(json.dumps(e) + "\n")
            f.flush()
            await on_event(e)

        def result(reason, summary, steps, question=None, success=False):
            return {"run_id": rid, "success_claimed": success, "summary": summary, "steps": steps, "trace_dir": str(d),
                    "reason": reason, "question": question}

        try:
            start = {"request": request, "start_url": start_url,
                     "config": {"lightning": {"model": "lightning"}, "clarify": cfg.options.get("clarify") == "on"}}
            if kwargs.get("parent_run_id"):
                start["parent_run_id"] = kwargs["parent_run_id"]
            await emit(0, "start", start)
            if cfg.options.get("clarify") == "on":
                q = QUESTIONS.get(request)
                if q:
                    await emit(0, "clarify", {"action": "ask", "request": request, "question": q, **CLARIFY_SHAPE,
                                              "usage": ASK_USAGE,
                                              "tool_call": {"name": "ask_user", "arguments": {"question": q}}},
                               timings_ms={"clarify": 812.5})
                    await emit(0, "done", {"success_claimed": False, "summary": q, "steps": 0, "reason": "needs_input",
                                           "stats": {"latency_ms": {"clarify": [812.5]}}, "question": q,
                                           "request": request})
                    return result("needs_input", q, 0, q)
                restated = RESTATEMENTS.get(request, " ".join(request.split()))
                await emit(0, "clarify", {"action": "proceed", "request": request, "normalized_request": restated,
                                          **CLARIFY_SHAPE}, timings_ms={"clarify": 640.0})
            while gate is not None and not gate.is_set():
                await asyncio.sleep(0.02)
            if pay_cents:
                sid = parse_qs(urlsplit(start_url).query)["sid"][0]
                base = f"{urlsplit(start_url).scheme}://{urlsplit(start_url).netloc}"
                async with httpx.AsyncClient(cookies={"sid": sid}, trust_env=False) as c:
                    r = await c.post(f"{base}/api/action", json={"type": "billing.pay",
                                                                 "payload": {"amount_cents": pay_cents}})
                    r.raise_for_status()
            await emit(1, "done", {"success_claimed": True, "summary": "did it", "steps": 1, "reason": "done"})
            return result("done", "did it", 1, success=True)
        finally:
            f.close()

    return run_task, lambda options: Cfg(root, options)


def post(base, body, expect=200):
    r = httpx.post(f"{base}/api/run", json=body)
    assert r.status_code == expect, r.text
    return r.json()


def run_to_end(base, body) -> tuple[dict, list[dict]]:
    run = post(base, body)
    return run, messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))


def steps_of(stream) -> list[dict]:
    return [m["data"] for m in stream if m["event"] == "step"]


def agent_id(stream) -> str:
    return steps_of(stream)[0]["run_id"]


def wait_kind(app, console_run_id: str, kind: str, timeout: float = 15.0) -> dict:
    """The first step of `kind` in the run's stream, once it has arrived (read from the runner, no SSE)."""
    run = app.state.runner.get(console_run_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for m in list(run.messages):
            if m.event == "step" and m.data.get("kind") == kind:
                return m.data
        time.sleep(0.02)
    raise AssertionError(f"run {console_run_id} never had a {kind} step")


def late_match(base, console_run_id, task_id="pay-50", matched_from=RESTATED, **over):
    body = {"task_id": task_id, "method": "restated", "score": 0.9, "matched_from": matched_from, **over}
    return httpx.post(f"{base}/api/runs/{console_run_id}/task_match", json=body)


# --- A) defaults, flags, the per-run option -------------------------------------------------------------------


def test_config_defaults_and_settings(console, tmp_path):
    base, _ = console(clarifying_agent(tmp_path / "runs", []))
    assert httpx.get(f"{base}/api/config").json()["defaults"] == {
        "grounder": "grounder", "thinking": "off", "lightning": "local", "clarify": "on"}
    base, _ = console(clarifying_agent(tmp_path / "runs", []), default_clarify="off", default_thinking="retry")
    assert httpx.get(f"{base}/api/config").json()["defaults"] == {
        "grounder": "grounder", "thinking": "retry", "lightning": "local", "clarify": "off"}
    for bad in ({"default_clarify": "maybe"}, {"default_clarify": True}, {"default_thinking": "always"}):
        with pytest.raises(ValueError):
            ConsoleSettings(**bad)


def test_cli_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(console_main, "create_app", lambda settings: seen.setdefault("settings", settings))
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    monkeypatch.setattr(console_main, "get_agent", lambda settings: (None, lambda options: {}))
    sandbox = f"http://127.0.0.1:{free_port()}"
    assert console_main.main(["--sandbox-url", sandbox]) == 0
    s = seen.pop("settings")
    assert (s.default_clarify, s.default_thinking) == ("on", "off")
    assert console_main.main(["--clarify", "off", "--thinking", "retry", "--sandbox-url", sandbox]) == 0
    s = seen.pop("settings")
    assert (s.default_clarify, s.default_thinking) == ("off", "retry")
    assert s.default_options() == {"grounder": "grounder", "thinking": "retry", "lightning": "local", "clarify": "off"}
    for bad in (["--clarify", "maybe"], ["--thinking", "always"]):
        with pytest.raises(SystemExit):
            console_main.main(bad)


def _port_in_range() -> int:
    """A free port in 19100-19199 (this suite's range for servers it starts by hand)."""
    for port in range(19140, 19200):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    pytest.skip("no free port in 19140-19199")


@pytest.mark.parametrize("args, want", [([], ["--thinking", "off", "--clarify", "on"]),
                                        (["--clarify", "off", "--thinking=retry"], ["--thinking", "retry", "--clarify",
                                                                                    "off"])])
def test_demo_up_passes_thinking_and_clarify_to_the_console(sandbox_url, tmp_path, args, want):
    """demo_up.sh starts the console with --thinking / --clarify (defaults off / on). PYTHON is a stand-in that
    records its arguments and sleeps, so nothing real starts; the sandbox is the suite's."""
    fake, record = tmp_path / "fake-python", tmp_path / "args.txt"
    fake.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{record}"\nexec sleep 30\n')
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_", "CONSOLE_"))}
    env.update(SANDBOX_PORT=sandbox_url.rsplit(":", 1)[1], CONSOLE_PORT=str(_port_in_range()), PYTHON=str(fake))
    proc = subprocess.Popen(["bash", str(DEMO_UP), "--skip-models", "--asr", "stub", *args], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline = time.time() + 20
        while not (record.exists() and record.read_text().strip()) and time.time() < deadline:
            time.sleep(0.05)
        got = record.read_text().split("\n")
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(10)
    assert got[:2] == ["-m", "console"], got
    flags = got[2:]
    for i in range(0, len(want), 2):
        assert flags[flags.index(want[i]) + 1] == want[i + 1], flags
    out = subprocess.run(["bash", str(DEMO_UP), "--help"], capture_output=True, text=True, timeout=20).stdout
    assert "--thinking off|retry|on" in out and "--clarify on|off" in out
    for bad in (["--clarify", "maybe"], ["--thinking", "x"]):
        r = subprocess.run(["bash", str(DEMO_UP), *bad], capture_output=True, text=True, timeout=20)
        assert r.returncode == 2 and r.stderr, bad


def test_clarify_option_reaches_the_agent_configs(clean_env):
    real = ConsoleSettings(agent="real")
    _, make_cfg = get_agent(real)
    defaults = real.default_options()
    cfg = make_cfg(defaults)
    assert cfg.clarify is True and cfg.thinking == "off"  # the console's defaults beat configs/agent.yaml's off/retry
    assert make_cfg({**defaults, "clarify": "off"}).clarify is False
    assert real_overrides(real, defaults)["clarify"] == "on"
    assert "clarify" not in real_overrides(real, {"grounder": "base", "thinking": "off", "lightning": "local"})

    _, mock_cfg = get_agent(ConsoleSettings(agent="mock"))
    assert mock_cfg(defaults).clarify_on is True and mock_cfg(defaults).public_dict()["clarify"] is True
    assert mock_cfg({**defaults, "clarify": "off"}).clarify_on is False
    assert mock_agent.MockConfig().clarify_on is False  # the mock alone keeps it off, like agent/config.py


def test_per_run_clarify_through_post_run(console, sandbox_url, tmp_path, clean_env):
    """The real make_cfg (load_config + the console's overrides) behind POST /api/run, with a fake run_task."""
    seen: list = []

    async def run_task(request, *, start_url, cfg, on_event=None):
        seen.append(cfg)
        rid = f"20261018-000000-{secrets.token_hex(3)}"
        d = Path(cfg.run_dir_root) / rid
        d.mkdir(parents=True)
        e = {"run_id": rid, "step": 0, "kind": "start", "ts": time.time(), "payload": {"config": cfg.public_dict()}}
        (d / "events.jsonl").write_text(json.dumps(e) + "\n")
        await on_event(e)
        return {"run_id": rid, "success_claimed": False, "summary": "", "steps": 0, "trace_dir": str(d)}

    kw = {"agent": "real", "runs_dir": tmp_path / "runs", "static_dir": tmp_path / "static"}
    _, make_cfg = get_agent(ConsoleSettings(sandbox_url=sandbox_url, **kw))
    base, _ = console((run_task, make_cfg), **kw)
    for options, want in (({}, True), ({"clarify": "off"}, False), ({"clarify": "on", "thinking": "retry"}, True)):
        _, stream = run_to_end(base, {"request": "Turn on high contrast.", "options": options})
        assert seen[-1].clarify is want and steps_of(stream)[0]["payload"]["config"]["clarify"] is want
        doc = json.loads((clean_env / agent_id(stream) / "console.json").read_text())
        assert doc["options"]["clarify"] == ("on" if want else "off")
        assert doc["options"]["thinking"] == options.get("thinking", "off")


# --- B) follow-up runs -----------------------------------------------------------------------------------------


def test_follow_up_builds_the_request_and_links_the_runs(console, tmp_path):
    runs, seen = tmp_path / "runs", []
    base, _ = console(clarifying_agent(runs, seen), runs_dir=runs)
    run, stream = run_to_end(base, {"request": ASK})
    steps = steps_of(stream)
    assert [s["kind"] for s in steps] == ["start", "clarify", "done"]
    done = steps[-1]["payload"]
    assert (done["reason"], done["question"], done["request"], done["steps"]) == ("needs_input", QUESTIONS[ASK], ASK, 0)
    parent = agent_id(stream)
    assert seen[0]["kwargs"] == {}  # not a follow-up: run_task gets no parent_run_id at all
    parent_doc = json.loads((runs / parent / "console.json").read_text())
    assert parent_doc["parent_run_id"] is None and parent_doc["clarifies"] is None

    want = follow_up_request(ASK, "fifty dollars")
    preview = ASK + " (clarified: " + "fifty dollars" + ")"  # how the page previews it (for its matcher)
    assert want == preview
    child_run, child_stream = run_to_end(base, {"request": preview,
                                                "clarifies": {"parent_run_id": parent, "answer": "  fifty   dollars "}})
    assert child_run["request"] == want
    assert seen[1]["request"] == want and seen[1]["kwargs"] == {"parent_run_id": parent}
    child = agent_id(child_stream)
    assert steps_of(child_stream)[0]["payload"]["parent_run_id"] == parent
    doc = json.loads((runs / child / "console.json").read_text())
    clarifies = {"parent_run_id": parent, "answer": "fifty dollars", "previous_request": ASK}
    assert (doc["request"], doc["parent_run_id"], doc["clarifies"]) == (want, parent, clarifies)
    info = httpx.get(f"{base}/api/runs/{child_run['run_id']}").json()
    assert (info["request"], info["parent_run_id"], info["clarifies"]) == (want, parent, clarifies)

    # no request in the body: the backend builds it
    second, second_stream = run_to_end(base, {"clarifies": {"parent_run_id": parent, "answer": "$50"}})
    assert second["request"] == follow_up_request(ASK, "$50") == ASK + " (clarified: $50)"
    second_id = agent_id(second_stream)

    # the logs: the parent asked and was answered twice; each child is a follow-up of it
    listed = {s["run_id"]: s for s in httpx.get(f"{base}/api/logs/runs").json()["runs"]}
    p = listed[parent]
    assert (p["status"], p["question"], p["parent_run_id"], p["child_run_ids"]) == (
        "needs_input", QUESTIONS[ASK], None, [child, second_id])
    assert p["clarify"] == {"action": "ask", "question": QUESTIONS[ASK], "normalized_request": None, **CLARIFY_SHAPE,
                            "usage": ASK_USAGE, "latency_ms": 812.5, "error": None, "skipped": None}
    assert p["models"]["lightning"] == "lightning" and p["steps"] == 0 and p["success_claimed"] is False
    c = listed[child]
    assert (c["parent_run_id"], c["child_run_ids"], c["status"], c["question"]) == (parent, [], "done", None)
    assert c["clarify"]["action"] == "proceed" and c["clarify"]["normalized_request"] == RESTATED
    assert c["clarify"]["latency_ms"] == 640.0
    detail = httpx.get(f"{base}/api/logs/runs/{parent}").json()
    assert detail["summary"]["child_run_ids"] == [child, second_id]
    assert [e["kind"] for e in detail["events"]] == ["start", "clarify", "done"]
    assert detail["events"][1]["model"] == "lightning" and detail["events"][1]["prompt_sha"] == "0123456789"
    assert httpx.get(f"{base}/api/logs/runs/{child}").json()["summary"]["parent_run_id"] == parent


def test_follow_up_parent_from_its_trace(console, tmp_path):
    """Another console (e.g. after a restart) finds the parent through its trace, not its own memory."""
    runs, seen = tmp_path / "runs", []
    first, _ = console(clarifying_agent(runs, seen), runs_dir=runs)
    _, stream = run_to_end(first, {"request": GARBLED})
    parent = agent_id(stream)
    second, app = console(clarifying_agent(runs, seen), runs_dir=runs)
    assert app.state.runner.agent_run_events(parent) == (None, None)  # not this console's run
    run, stream = run_to_end(second, {"clarifies": {"parent_run_id": parent, "answer": "fifty dollars"}})
    assert run["request"] == follow_up_request(GARBLED, "fifty dollars")
    assert seen[-1]["kwargs"] == {"parent_run_id": parent}
    clar = steps_of(stream)[1]["payload"]
    assert (clar["action"], clar["normalized_request"]) == ("proceed", RESTATED)


def test_bad_follow_ups_400(console, tmp_path):
    runs, seen, gate = tmp_path / "runs", [], threading.Event()
    base, app = console(clarifying_agent(runs, seen, gate=gate), runs_dir=runs)
    _, stream = run_to_end(base, {"request": ASK})
    parent = agent_id(stream)
    gate.set()
    _, stream = run_to_end(base, {"request": "Turn on high contrast."})
    clear = agent_id(stream)
    n_runs = len(seen)

    def bad(body, error):
        r = httpx.post(f"{base}/api/run", json=body)
        assert r.status_code == 400, (body, r.text)
        assert error in r.json()["error"], (body, r.json())

    ok = {"parent_run_id": parent, "answer": "fifty dollars"}
    bad({"clarifies": {**ok, "parent_run_id": "20261018-000000-abcdef"}}, "unknown parent run '20261018-000000-abcdef'")
    bad({"clarifies": {**ok, "parent_run_id": clear}}, "did not end with a question (done reason 'done'")
    for pid in ("../etc", "", 5, None, ".."):
        bad({"clarifies": {**ok, "parent_run_id": pid}}, "clarifies.parent_run_id must be the agent run id")
    for answer in ("", "   ", "x" * 301, 5, None):
        bad({"clarifies": {**ok, "answer": answer}}, "clarifies.answer must be a string of 1 to 300 characters")
    bad({"clarifies": {"parent_run_id": parent}}, "clarifies.answer must be")
    bad({"clarifies": {**ok, "why": "x"}}, "unknown clarifies field(s): why")
    bad({"clarifies": "fifty dollars"}, "clarifies must be an object")
    built = follow_up_request(ASK, "fifty dollars")
    bad({"request": ASK + " (clarified: forty dollars)", "clarifies": ok},
        f"request does not match the follow-up request the console builds from the question's request and the "
        f"answer: {built!r}")
    bad({"request": ASK + " [answer: fifty dollars]", "clarifies": ok}, "request does not match")
    bad({"request": 5, "clarifies": ok}, "request does not match")
    bad({"request": "", "clarifies": ok}, "request does not match")
    assert len(seen) == n_runs  # nothing started

    # whitespace aside, the page's text is accepted; a 300-character answer is fine
    r = post(base, {"request": "  " + built.replace(" ", "  ") + " ", "clarifies": ok})
    assert r["request"] == built
    read_sse(f"{base}/api/runs/{r['run_id']}/events")
    r = post(base, {"clarifies": {**ok, "answer": "x" * 300}})
    read_sse(f"{base}/api/runs/{r['run_id']}/events")

    # a parent that hasn't ended yet
    gate.clear()
    live = post(base, {"request": "Turn on high contrast."})
    live_id = wait_kind(app, live["run_id"], "clarify")["run_id"]
    bad({"clarifies": {"parent_run_id": live_id, "answer": "x"}}, f"parent run {live_id!r} has not ended")
    gate.set()
    read_sse(f"{base}/api/runs/{live['run_id']}/events")


def test_parse_run_body_follow_ups():
    seen = []

    def resolve(pid):
        seen.append(pid)
        return ("Pay something toward my bill", None) if pid == "p-1" else (None, f"unknown parent run {pid!r}")

    clean, err = parse_run_body({"clarifies": {"parent_run_id": "p-1", "answer": "fifty\tdollars"}}, TASKS_BY_ID,
                                resolve_parent=resolve)
    assert err is None and clean["request"] == "Pay something toward my bill (clarified: fifty dollars)"
    assert clean["clarifies"] == {"parent_run_id": "p-1", "answer": "fifty dollars",
                                  "previous_request": "Pay something toward my bill"}
    assert parse_run_body({"clarifies": {"parent_run_id": "p-2", "answer": "x"}}, TASKS_BY_ID,
                          resolve_parent=resolve) == (None, "unknown parent run 'p-2'")
    assert parse_run_body({"clarifies": {"parent_run_id": "p-1", "answer": "x"}}, TASKS_BY_ID)[1] == \
        "follow-up runs (clarifies) are not available here"
    # a plain run: clarifies null, the request validated as before
    clean, err = parse_run_body({"request": " hi "}, TASKS_BY_ID)
    assert err is None and (clean["request"], clean["clarifies"]) == ("hi", None)
    # a follow-up may also be checked as a task (the page matched its preview), with the page's task_match
    match = {"method": "fuzzy", "score": 0.8, "matched_from": "Pay something toward my bill (clarified: $50)"}
    clean, err = parse_run_body({"clarifies": {"parent_run_id": "p-1", "answer": "$50"}, "task_id": "pay-50",
                                 "task_match": match}, TASKS_BY_ID, resolve_parent=resolve)
    assert err is None and (clean["task_id"], clean["task_match"]) == ("pay-50", match)
    # a follow-up's request may be longer than 500 characters (the question's request plus the answer)
    long = "Pay " + "x" * 490

    def resolve_long(pid):
        return long, None

    clean, err = parse_run_body({"clarifies": {"parent_run_id": "p-1", "answer": "y" * 300}}, TASKS_BY_ID,
                                resolve_parent=resolve_long)
    assert err is None and len(clean["request"]) > 500


# --- C) the restated-request late task match -------------------------------------------------------------------


def test_restated_match_checks_the_follow_up_as_that_task(console, tmp_path):
    """The demo flow with fake agents: a garbled request gets a question; the answer's run is free text (no task
    matches the user's words), the agent restates it as pay-50's text, the page reports that match, and the verdict
    at the end checks pay-50 (here the fake pays $50, so it passes)."""
    runs, seen, gate = tmp_path / "runs", [], threading.Event()
    base, app = console(clarifying_agent(runs, seen, gate=gate, pay_cents=5000), runs_dir=runs)
    _, stream = run_to_end(base, {"request": GARBLED})
    parent = agent_id(stream)

    run = post(base, {"clarifies": {"parent_run_id": parent, "answer": "fifty dollars"}})
    assert run["start_url"].endswith("/?sid=" + run["sid"])  # free text: the dashboard, the default state
    clar = wait_kind(app, run["run_id"], "clarify")
    assert clar["payload"]["normalized_request"] == RESTATED
    r = late_match(base, run["run_id"], score=1)
    assert r.status_code == 200, r.text
    match = {"method": "restated", "score": 1.0, "matched_from": RESTATED}
    assert r.json() == {"ok": True, "run_id": run["run_id"], "task_id": "pay-50", "task_match": match}
    info = httpx.get(f"{base}/api/runs/{run['run_id']}").json()
    assert (info["task_id"], info["task_match"]) == ("pay-50", match)
    rid = clar["run_id"]
    doc = json.loads((runs / rid / "console.json").read_text())  # rewritten at once
    assert (doc["status"], doc["task_id"], doc["task_match"]) == ("running", "pay-50", match)
    again = late_match(base, run["run_id"])
    assert again.status_code == 409 and again.json() == {"error": "this run is already checked as task pay-50"}

    gate.set()
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    verdict = next(m["data"] for m in stream if m["event"] == "verdict")
    assert verdict == {"task_id": "pay-50", "passed": True, "explanation": verdict["explanation"],
                       "label": VERDICT_LABEL}
    doc = json.loads((runs / rid / "console.json").read_text())
    assert (doc["task_id"], doc["task_match"], doc["verdict"]) == ("pay-50", match, verdict)
    s = httpx.get(f"{base}/api/logs/runs/{rid}").json()["summary"]
    assert (s["task_id"], s["task_match"], s["verified"]) == ("pay-50", match, True)
    assert s["task_line"] == RESTATED_LINE
    late = late_match(base, run["run_id"])
    assert late.status_code == 409 and "already checked" in late.json()["error"]


def test_restated_match_rules(console, tmp_path):
    runs, seen, gate = tmp_path / "runs", [], threading.Event()
    base, app = console(clarifying_agent(runs, seen, gate=gate), runs_dir=runs)
    run = post(base, {"request": " Pay  $50 toward my bill. "})  # free text; restated as RESTATED
    wait_kind(app, run["run_id"], "clarify")

    assert late_match(base, "c-nope").status_code == 404
    for body, error in (
        ({"task_id": "pay-50", "method": "fuzzy", "score": 1, "matched_from": RESTATED},
         "method must be restated (other methods are sent with POST /api/run)"),
        ({"task_id": "no-such", "method": "restated", "score": 1, "matched_from": RESTATED}, "unknown task_id 'no-such'"),
        ({"task_id": "pay-50", "method": "restated", "score": 2, "matched_from": RESTATED},
         "score must be a number from 0 to 1, or null"),
        ({"task_id": "pay-50", "method": "restated", "score": True, "matched_from": RESTATED},
         "score must be a number from 0 to 1, or null"),
        ({"task_id": "pay-50", "method": "restated", "matched_from": RESTATED}, "needs score"),
        ({"task_id": "pay-50", "method": "restated", "score": 1, "matched_from": RESTATED, "x": 1}, "unknown field(s): x"),
        ({"task_id": "pay-50", "method": "restated", "score": 1, "matched_from": ""},
         "matched_from must be the agent's restatement"),
        ({"task_id": "pay-50", "method": "restated", "score": 1, "matched_from": "x" * 501},
         "matched_from must be a string of at most 500 characters"),
        (["pay-50"], "body must be a JSON object"),
    ):
        r = httpx.post(f"{base}/api/runs/{run['run_id']}/task_match", json=body)
        assert r.status_code == 400 and r.json() == {"error": error}, (body, r.text)

    r = late_match(base, run["run_id"], matched_from="Pay fifty dollars toward my bill")
    assert r.status_code == 400
    assert r.json()["error"] == f"matched_from must be the agent's restatement of this run's request: {RESTATED!r}"
    r = late_match(base, run["run_id"], task_id="mark-all-read", matched_from=RESTATED)
    assert r.status_code == 409 and r.json()["error"] == "task mark-all-read starts on /messages, but this run started on /"
    r = late_match(base, run["run_id"], task_id="max-usage-month")
    assert r.status_code == 409
    assert r.json()["error"] == ("task max-usage-month starts from its own state (ui.nav_collapsed), but this run "
                                 "started from the default state")
    assert httpx.get(f"{base}/api/runs/{run['run_id']}").json()["task_id"] is None  # nothing changed so far

    r = late_match(base, run["run_id"], matched_from="Pay $50  toward my bill.")  # whitespace aside: it is the text
    assert r.status_code == 200, r.text
    gate.set()
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    verdict = next(m["data"] for m in stream if m["event"] == "verdict")
    assert verdict["task_id"] == "pay-50" and verdict["passed"] is False  # nothing was paid: the check says so
    r = late_match(base, run["run_id"])
    assert r.status_code == 409


def test_restated_match_refused_when_it_could_change_nothing_honestly(console, tmp_path):
    runs, seen, gate = tmp_path / "runs", [], threading.Event()
    base, app = console(clarifying_agent(runs, seen, gate=gate), runs_dir=runs)

    # a run that already has a task (picked from the list)
    picked = post(base, {"request": RESTATED, "task_id": "pay-50"})
    wait_kind(app, picked["run_id"], "clarify")
    r = late_match(base, picked["run_id"], task_id="high-contrast")
    assert r.status_code == 409 and r.json()["error"] == "this run is already checked as task pay-50"
    gate.set()
    read_sse(f"{base}/api/runs/{picked['run_id']}/events")

    # the request check is off: no restatement to match
    gate.clear()
    off = post(base, {"request": RESTATED, "options": {"clarify": "off"}})
    wait_kind(app, off["run_id"], "start")
    r = late_match(base, off["run_id"])
    assert r.status_code == 409
    assert r.json()["error"] == "the agent has not restated this run's request (no clarify event that proceeded with one)"
    gate.set()
    stream = messages(read_sse(f"{base}/api/runs/{off['run_id']}/events"))
    assert "clarify" not in [s["kind"] for s in steps_of(stream)] and "verdict" not in [m["event"] for m in stream]

    # the run is over: too late (and a needs_input run never restated anything)
    _, stream = run_to_end(base, {"request": "Turn on high contrast."})
    ended = next(r for r in app.state.runner.runs.values() if r.agent_run_id == agent_id(stream))
    r = late_match(base, ended.run_id, task_id="high-contrast", matched_from="Turn on high contrast.")
    assert r.status_code == 409 and r.json()["error"] == "this run has ended: too late to choose a task for its verdict"


# --- E) the mock agent's request check -------------------------------------------------------------------------


def test_mock_clarify_decisions():
    d = mock_agent.clarify_decision
    common = {"model": "mock (no Lightning)", "prompt_version": "clarify-v4 (mock)", "prompt_sha": "mock"}
    for request, question in QUESTIONS.items():
        assert d(request) == {"action": "ask", "request": request, "question": question, **common,
                              "tool_call": {"name": "ask_user", "arguments": {"question": question}}}
    assert d("pay SOMETHING toward my bill.")["question"] == QUESTIONS[ASK]  # case and the final period aside
    for answer in ("fifty dollars", "$50", "Fifty dollars.", " $50 "):
        for previous in QUESTIONS:
            request = follow_up_request(previous, answer)
            assert d(request) == {"action": "proceed", "request": request, "normalized_request": RESTATED, **common,
                                  "tool_call": {"name": "proceed", "arguments": {"normalized_request": RESTATED}}}
    other = follow_up_request(ASK, "forty dollars")
    assert d(other)["normalized_request"] == other  # the mock only understands $50
    assert d(follow_up_request("Turn on high contrast.", "fifty dollars"))["normalized_request"] == \
        "Turn on high contrast. (clarified: fifty dollars)"
    assert d("  Turn on   high contrast. ")["normalized_request"] == "Turn on high contrast."
    twice = follow_up_request(follow_up_request(ASK, "um"), "fifty dollars")
    assert d(twice) == {"action": "proceed", "request": twice, **common, "skipped": "already clarified 2 times"}
    # the same follow-up format and limit as the real request check
    assert mock_agent.CLARIFIED_MARK == agent_clarify.CLARIFIED_MARK
    assert mock_agent.MAX_CLARIFICATIONS == agent_clarify.MAX_CLARIFICATIONS


def test_mock_ask_ends_with_needs_input_without_a_browser(console, tmp_path):
    base, _ = console(None, runs_dir=tmp_path / "runs", step_delay_s=0.0)  # --agent mock, clarify on by default
    for request, question in QUESTIONS.items():
        _, stream = run_to_end(base, {"request": request})
        steps = steps_of(stream)
        assert [(s["step"], s["kind"]) for s in steps] == [(0, "start"), (0, "clarify"), (0, "done")]
        assert steps[1]["payload"] == {"action": "ask", "request": request, "question": question,
                                       "model": "mock (no Lightning)", "prompt_version": "clarify-v4 (mock)",
                                       "prompt_sha": "mock", "mock": True,
                                       "tool_call": {"name": "ask_user", "arguments": {"question": question}}}
        assert "timings_ms" not in steps[1]  # no model ran: no latency
        done = steps[2]["payload"]
        assert {k: v for k, v in done.items() if k != "stats"} == {
            "success_claimed": False, "summary": question, "question": question, "request": request, "steps": 0,
            "reason": "needs_input", "mock": True}
        assert isinstance(done["stats"], dict)
        result = next(m["data"] for m in stream if m["event"] == "result")
        assert (result["reason"], result["question"], result["steps"]) == ("needs_input", question, 0)
        assert sorted(p.name for p in Path(result["trace_dir"]).iterdir()) == ["console.json", "events.jsonl"]  # no shots
        assert stream[-2]["data"] == {"status": "done"} and "verdict" not in [m["event"] for m in stream]
        s = httpx.get(f"{base}/api/logs/runs/{result['run_id']}").json()["summary"]
        assert (s["status"], s["question"], s["agent"], s["models"]["lightning"]) == (
            "needs_input", question, "mock", "mock (no Lightning)")


@pytest.mark.e2e
def test_mock_follow_up_replays_pay_50_through_the_restated_match(console, tmp_path):
    """The whole flow with the real mock agent: ask -> needs_input; the answer's follow-up -> proceed with the
    restatement "Pay $50 toward my bill." -> the page's restated match -> the pay-50 script in Chromium -> ✓."""
    runs = tmp_path / "runs"
    base, app = console(None, runs_dir=runs, step_delay_s=0.0)
    _, stream = run_to_end(base, {"request": GARBLED})
    parent = agent_id(stream)
    assert steps_of(stream)[-1]["payload"]["reason"] == "needs_input"

    preview = GARBLED + " (clarified: fifty dollars)"
    run = post(base, {"request": preview, "clarifies": {"parent_run_id": parent, "answer": "fifty dollars"}})
    clar = wait_kind(app, run["run_id"], "clarify")
    assert (clar["payload"]["action"], clar["payload"]["normalized_request"]) == ("proceed", RESTATED)
    r = late_match(base, run["run_id"], score=1)
    assert r.status_code == 200, r.text
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events", timeout=120))
    steps = steps_of(stream)
    assert steps[0]["payload"]["parent_run_id"] == parent and steps[0]["payload"]["request"] == preview
    script = json.loads(mock_agent.SCRIPTS_PATH.read_text())["tasks"]["pay-50"]
    calls = [s["payload"]["tool_calls"][0]["name"] for s in steps if s["kind"] == "plan"]
    assert calls == [st["tool"] for st in script["steps"]]  # the pay-50 script, picked by the restatement
    result = next(m["data"] for m in stream if m["event"] == "result")
    verdict = next(m["data"] for m in stream if m["event"] == "verdict")
    assert result["success_claimed"] is True and (verdict["task_id"], verdict["passed"]) == ("pay-50", True), verdict
    s = httpx.get(f"{base}/api/logs/runs/{result['run_id']}").json()["summary"]
    assert (s["task_line"], s["parent_run_id"], s["verified"]) == (RESTATED_LINE, parent, True)
    assert httpx.get(f"{base}/api/logs/runs/{parent}").json()["summary"]["child_run_ids"] == [result["run_id"]]

    # the request check off: the same words reach the script lookup as they are (no script: the mock gives up)
    _, stream = run_to_end(base, {"request": GARBLED, "options": {"clarify": "off"}})
    kinds = [s["kind"] for s in steps_of(stream)]
    assert kinds == ["start", "observation", "plan", "tool_call", "action", "done"]
    assert steps_of(stream)[0]["payload"]["config"]["clarify"] is False
    assert steps_of(stream)[-1]["payload"]["reason"] == "done"
