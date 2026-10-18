"""Logs page backend (console/logs.py), console.json from the runner, and the optional `voice` of POST /api/run.

No browser except the one e2e test, which runs the real mock agent (headless Chromium) through the console.

fixtures/real_traces/ holds two real agent traces (Lightning + Qwen VLM, console runs on 2026-10-17):
runs/20261017-175025-e9fe8e (done, 2 steps) and runs/20261017-174924-f75115 (stopped after the first
plan). They are byte-for-byte copies except that the local checkout prefix of screenshot_path /
overlay_path was replaced with /path/to/repo/runs/ (runs/ is gitignored, so the tests can't read it).
The PNGs aren't committed; tests write placeholders with the same names.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import shutil
import statistics
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from PIL import Image

from console import logs
from console.app import parse_run_body
from console.tests.conftest import messages, read_sse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "real_traces"
DONE_RUN = "20261017-175025-e9fe8e"
STOPPED_RUN = "20261017-174924-f75115"
# fixtures/stuck_trace/: a synthetic real-agent trace in agent/core.py's exact shapes for the "stuck" stop, with the
# console.json a fully local (Mac) console run writes. Its stuck verdicts, reason and done summary come from
# agent.progress.ProgressTracker itself (test_stuck_fixture_matches_progress_tracker keeps them in sync). Six clicks
# on the Text size control that change nothing: a warning at step 3 (the same target 3 times), the stop at step 6.
STUCK_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stuck_trace"
STUCK_RUN = "20261017-211502-5f7c1d"
TASK = "download-september-statement"
VOICE = {"text": "VOICE-ONLY-MARKER download my september statement", "model": "parakeet-tdt-0.6b-v3",
         "latency_ms": 812.4, "duration_s": 2.5}


def ev(kind, step, ts, payload=None, **extra):
    return {"run_id": "r1", "step": step, "kind": kind, "ts": ts, "payload": payload or {}, **extra}


def hand_trace() -> list[dict]:
    """A real-agent-shaped trace with known numbers."""
    config = {"lightning": {"base_url": "http://x/v1", "model": "lightning-served", "root": "nvidia/ckpt"},
              "grounder_model": "grounder", "summary_model": "base", "prompt_version": "agent-v9"}
    return [
        ev("start", 0, 1000.0, {"request": "Do the thing.", "start_url": "http://s/?sid=x", "config": config}),
        ev("observation", 0, 1001.0, {"observation": {}}, timings_ms={"screenshot": 5.0, "summary": 100.0}),
        ev("plan", 1, 1002.0, {"tool_calls": []}, timings_ms={"plan": 300.0}),
        ev("tool_call", 1, 1002.0, {"name": "click", "valid": True}),
        ev("grounding", 1, 1003.0, {"model": "grounder-from-payload"}, timings_ms={"ground": 40.0}),
        ev("action", 1, 1003.5, {"ok": True}, timings_ms={"act": 10.0, "settle": 500.0}),
        ev("observation", 1, 1004.0, {"summary": {"model": "summary-from-payload"}}, timings_ms={"summary": 200.0}),
        ev("telemetry", 1, 1005.0, {"anything": 1}),  # unknown kind: kept, no model
        ev("plan", 2, 1006.0, {"model": "lightning-from-payload"}, timings_ms={"plan": 500.0}),
        ev("tool_call", 2, 1006.0, {"name": "clik", "valid": False, "error": "unknown tool"}),
        ev("error", 2, 1006.1, {"message": "boom", "where": "planner", "recoverable": True}),
        ev("plan", 3, 1009.0, {}, timings_ms={"plan": 400.0}),
        ev("tool_call", 3, 1009.0, {"name": "done", "valid": True}),
        ev("action", 3, 1009.0, {"done": True}),
        ev("done", 3, 1010.5, {"success_claimed": True, "summary": "ok", "steps": 3, "reason": "done"}),
    ]


CURRENT_RUN = "20261017-191633-1a67b1"
SHA = {"plan": "cc1d9edb42", "summary": "b00fd27100", "grounding": "f9ed0fa621"}
SUMMARY_USAGE = {"prompt_tokens": 1702, "completion_tokens": 233}
GROUND_USAGE = {"prompt_tokens": 1650, "completion_tokens": 14}
SUMMARY_ERROR = "base: request timed out after 45 s"


def current_agent_trace(run_dir: str = f"/path/to/repo/runs/{CURRENT_RUN}") -> list[dict]:
    """A real-agent trace with the fields agent/core.py writes since b6b0db2 (documented at the top of agent/trace.py):
    plan {model, prompt_version, prompt_sha}, observation summary {model, prompt_sha, usage?, error?, reused?},
    grounding {prompt_sha, usage}. Built from that field list (no real trace with them was recorded yet), shaped
    like core.py's output. The first summary call fails (summary.error); step 1 clicks; step 2's grounding fails,
    so its observation carries step 1's description over (reused: no new summarizer call); step 3 is done."""
    rid = Path(run_dir).name
    cfg = {"lightning": {"base_url": "http://127.0.0.1:8000/v1", "model": "lightning",
                         "root": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"},
           "grounder_model": "grounder", "summary_model": "base", "model_family": "qwen3_5", "max_steps": 12,
           "thinking": "retry", "prompt_version": "agent-v5"}

    def e(kind, step, ts, payload, **extra):
        return {"run_id": rid, "step": step, "kind": kind, "ts": ts, "payload": payload, **extra}

    def plan(step, ts, name, args, prompt_tokens, ms):
        return e("plan", step, ts, {"content": "", "reasoning": "", "thinking": False,
                                    "tool_calls": [{"name": name, "arguments": args}],
                                    "usage": {"completion_tokens": 20, "prompt_tokens": prompt_tokens,
                                              "total_tokens": prompt_tokens + 20},
                                    "model": "lightning", "prompt_version": "agent-v5", "prompt_sha": SHA["plan"]},
                 timings_ms={"plan": ms})

    def obs(step, ts, path, changed, last, text, summary, timings):
        o = {"step": step, "url": f"http://127.0.0.1:8090{path}?sid=console-c-0f0f0f0f", "title": "Acme Utilities",
             "screen_changed": changed, "last_action_result": last, "screen_summary": text}
        return e("observation", step, ts, {"observation": o, "text": f"Step {step} | URL: {path}", "summary": summary},
                 screenshot_path=f"{run_dir}/step_{step:02d}.png", timings_ms=timings)

    gear, unicorn = {"target_description": "the gear icon"}, {"target_description": "the purple unicorn button"}
    fresh = {"model": "base", "prompt_sha": SHA["summary"], "usage": SUMMARY_USAGE}
    failed = {"model": "base", "prompt_sha": SHA["summary"], "error": SUMMARY_ERROR}
    return [
        e("start", 0, 2000.0, {"request": "Turn on high contrast.", "start_url": "http://127.0.0.1:8090/?sid=x",
                               "config": cfg}),
        obs(0, 2001.0, "/", None, "", f"(no screen description: {SUMMARY_ERROR})", failed,
            {"screenshot": 40.0, "summary": 45000.0, "settle": 350.0}),
        plan(1, 2002.0, "click", gear, 1800, 1500.0),
        e("tool_call", 1, 2002.0, {"name": "click", "arguments": gear, "valid": True, "description": "Clicking: the gear icon"}),
        e("grounding", 1, 2002.5, {"target": "the gear icon", "x": 1180.0, "y": 32.0, "point_1000": [922, 40],
                                   "raw": "(922, 40)", "model": "grounder", "prompt_sha": SHA["grounding"],
                                   "usage": GROUND_USAGE},
          overlay_path=f"{run_dir}/step_01_overlay.png", timings_ms={"ground": 600.0}),
        e("action", 1, 2003.0, {"name": "click", "arguments": gear, "ok": True, "screen_changed": True, "done": False,
                                "click": {"x": 1180, "y": 32}}, timings_ms={"act": 20.0, "settle": 500.0}),
        obs(1, 2004.0, "/settings", True, 'clicked "the gear icon" at (1180, 32)', "PAGE: Settings", fresh,
            {"screenshot": 50.0, "summary": 900.0}),
        plan(2, 2005.0, "click", unicorn, 2300, 1700.0),
        e("tool_call", 2, 2005.0, {"name": "click", "arguments": unicorn, "valid": True,
                                   "description": "Clicking: the purple unicorn button"}),
        e("grounding", 2, 2005.5, {"target": "the purple unicorn button", "x": None, "y": None, "point_1000": None,
                                   "raw": "I can't see it", "model": "grounder", "prompt_sha": SHA["grounding"],
                                   "usage": GROUND_USAGE, "error": "no point in the reply"},
          overlay_path=f"{run_dir}/step_02_overlay.png", timings_ms={"ground": 700.0}),
        e("action", 2, 2006.0, {"name": "click", "arguments": unicorn, "ok": False, "screen_changed": None, "done": False,
                                "error": "could not locate 'the purple unicorn button': no point in the reply"}),
        obs(2, 2006.0, "/settings", None, "click failed: could not locate 'the purple unicorn button'",
            "PAGE: Settings", {**fresh, "reused": True}, {"screenshot": 30.0}),
        plan(3, 2007.0, "done", {"summary": "I could not finish.", "success": False}, 2800, 1300.0),
        e("tool_call", 3, 2007.0, {"name": "done", "arguments": {"summary": "I could not finish.", "success": False},
                                   "valid": True, "description": "Finishing: I could not finish."}),
        e("action", 3, 2007.0, {"name": "done", "arguments": {"summary": "I could not finish.", "success": False},
                                "ok": True, "screen_changed": None, "done": True}),
        e("done", 3, 2008.0, {"success_claimed": False, "summary": "I could not finish.", "steps": 3, "reason": "done",
                              "stats": {"n_actions": 2}}),
    ]


def console_doc(**over) -> dict:
    doc = {"console_run_id": "c-1234abcd", "agent": "real", "request": "Do the thing.", "task_id": TASK,
           "options": {"grounder": "base", "thinking": "retry", "lightning": "local"}, "sid": "console-c-1234abcd",
           "start_url": "http://s/?sid=console-c-1234abcd", "voice": dict(VOICE), "status": "done",
           "result": {"success_claimed": False}, "verdict": {"task_id": TASK, "passed": False,
                                                             "explanation": "the statement was not downloaded",
                                                             "label": "verified by sandbox state, not seen by the agent"},
           "started_ts": 999.0, "ended_ts": 1011.0}
    doc.update(over)
    return doc


def write_run(root: Path, run_id: str, events: list[dict], console: dict | None = None, raw_tail: bytes = b"") -> Path:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "events.jsonl").write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in events) + raw_tail)
    if console is not None:
        (d / "console.json").write_text(json.dumps(console))
    return d


def png(path: Path) -> None:
    Image.new("RGB", (8, 6), "white").save(path)


def copy_real_traces(root: Path) -> None:
    for rid in (DONE_RUN, STOPPED_RUN):
        shutil.copytree(FIXTURES / rid, root / rid)
    for name in ("step_00.png", "step_01.png", "step_01_overlay.png"):
        png(root / DONE_RUN / name)
    png(root / STOPPED_RUN / "step_00.png")


def raw_get(base: str, path: str) -> tuple[int, bytes]:
    """GET with the path sent verbatim (httpx would normalize '..')."""
    u = urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


# --- summary math ---------------------------------------------------------------------------------


def test_summary_math_on_a_hand_made_trace():
    s = logs.summarize_run(hand_trace(), None, run_id="r1")
    assert s["run_id"] == "r1" and s["agent"] == "real" and s["console_run_id"] is None and s["task_id"] is None
    assert s["request"] == "Do the thing."
    assert (s["started_ts"], s["ended_ts"], s["total_ms"]) == (1000.0, 1010.5, 10500)
    # p50 per stage from the events' timings_ms; screenshot/settle are not stages; voice needs console.json
    assert s["stage_p50_ms"] == {"voice": None, "summary": 150.0, "plan": 400.0, "ground": 40.0, "act": 10.0}
    assert (s["status"], s["steps"], s["success_claimed"]) == ("done", 3, True)
    assert (s["n_events"], s["n_errors"], s["n_invalid_tool_calls"], s["n_bad_lines"]) == (15, 1, 1, 0)
    # payload-declared names win over the start config
    assert s["models"] == {"lightning": "lightning-from-payload", "grounder": "grounder-from-payload",
                           "summary": "summary-from-payload", "asr": None}
    assert (s["prompt_version"], s["lightning_root"]) == ("agent-v9", "nvidia/ckpt")
    assert s["prompt_sha"] == {"lightning": None, "grounder": None, "summary": None}  # the run config doesn't record it: nothing to fall back to
    assert (s["verified"], s["verdict_explanation"], s["voice"]) == (None, None, None)
    json.dumps(s, allow_nan=False)


def test_models_fall_back_to_the_start_config():
    events = [e for e in hand_trace() if e["kind"] not in ("grounding",)]
    for e in events:
        e["payload"].pop("model", None)
        e["payload"].pop("summary", None)
    s = logs.summarize_run(events)
    assert s["models"] == {"lightning": "lightning-served", "grounder": "grounder", "summary": "base", "asr": None}
    assert s["run_id"] == "r1"  # without a dir name: the events' run_id
    cfg = events[0]["payload"]["config"]
    assert logs.event_model(events[2], cfg) == "lightning-served"
    assert logs.event_model({"kind": "grounding", "payload": {}}, cfg) == "grounder"
    assert logs.event_model({"kind": "observation", "payload": {"summary": "a string"}}, cfg) == "base"
    assert logs.event_model({"kind": "action", "payload": {"model": "x"}}, cfg) is None
    # no start event at all: nothing to fall back to
    assert logs.summarize_run(events[1:])["models"] == {"lightning": None, "grounder": None, "summary": None, "asr": None}


def test_console_json_fields_voice_and_verdict():
    s = logs.summarize_run(hand_trace(), console_doc(), run_id="r1")
    assert (s["console_run_id"], s["task_id"]) == ("c-1234abcd", TASK)
    assert s["voice"] == VOICE and s["models"]["asr"] == VOICE["model"]
    assert s["stage_p50_ms"]["voice"] == 812.4
    assert (s["verified"], s["verdict_explanation"]) == (False, "the statement was not downloaded")
    assert s["success_claimed"] is True  # the agent's own done event beats console.json's copy of the result
    assert (s["started_ts"], s["ended_ts"]) == (1000.0, 1010.5)  # from the events, not console.json
    no_verdict = logs.summarize_run(hand_trace(), console_doc(verdict=None, voice=None))
    assert (no_verdict["verified"], no_verdict["verdict_explanation"], no_verdict["voice"]) == (None, None, None)
    assert no_verdict["stage_p50_ms"]["voice"] is None and no_verdict["models"]["asr"] is None


def _done(reason):
    return [ev("start", 0, 1.0, {"request": "x"}), ev("plan", 1, 2.0),
            ev("done", 1, 3.0, {"reason": reason, "steps": 1, "success_claimed": False})]


@pytest.mark.parametrize("events, console, status", [
    (_done("done"), None, "done"),
    (_done("max_steps"), None, "max_steps"),
    (_done("stuck"), console_doc(status="done"), "stuck"),          # agent/progress.py's stop: its own status
    (_done("stopped"), console_doc(status="stopped"), "stopped"),
    (_done("error"), None, "error"),
    (_done("weird"), None, "unknown"),
    (_done("done"), console_doc(status="error"), "done"),           # the agent's done event wins
    (_done("done")[:2], console_doc(status="running"), "running"),  # console.json while the run is live
    (_done("done")[:2], console_doc(status="stopped"), "stopped"),  # e.g. an agent that ignored the stop
    (_done("done")[:2], console_doc(status="error"), "error"),
    (_done("done")[:2], console_doc(status="done"), "unknown"),     # returned without a done event: odd, say so
    (_done("done")[:2], None, "unknown"),                           # killed, no console.json
    ([], None, "unknown"),
])
def test_status(events, console, status):
    assert logs.summarize_run(events, console)["status"] == status


def test_partial_trace_fallbacks():
    events = _done("done")[:2]  # start + plan, no done
    s = logs.summarize_run(events, console_doc(result={"success_claimed": True}))
    assert (s["steps"], s["success_claimed"], s["total_ms"]) == (1, True, 1000)
    s = logs.summarize_run([], None)
    assert (s["steps"], s["success_claimed"], s["started_ts"], s["total_ms"], s["request"]) == (None, None, None, None, None)
    assert s["stage_p50_ms"] == dict.fromkeys(logs.STAGES)
    s = logs.summarize_run([{"kind": "start"}, {"kind": "plan", "ts": "soon", "step": True, "timings_ms": {"plan": "x"}}],
                           {"request": "from console", "status": 3, "voice": {"text": 5, "latency_ms": True}})
    assert s["request"] == "from console" and s["started_ts"] is None and s["steps"] is None
    assert s["stage_p50_ms"]["plan"] is None and s["voice"] == {"text": None, "model": None, "latency_ms": None,
                                                               "duration_s": None}


def test_live_status():
    running = logs.summarize_run(_done("done")[:2], console_doc(status="running"), run_id="a")
    assert logs.apply_live(running, {"a"})["status"] == "running"
    assert logs.apply_live(running, set())["status"] == "unknown"  # a console that died mid-run
    no_console = logs.summarize_run(_done("done")[:2], None, run_id="a")
    assert logs.apply_live(no_console, {"a"})["status"] == "running"
    finished = logs.summarize_run(_done("stopped"), None, run_id="a")
    assert logs.apply_live(finished, {"a"})["status"] == "stopped"


def test_mock_runs_are_labelled_and_never_borrow_config_models():
    config = {"agent": "mock", "grounder_model": "base", "lightning": "hosted", "thinking": "retry"}
    old_mock = [ev("start", 0, 1.0, {"request": "x", "config": config, "mock": True}),
                ev("observation", 0, 2.0, {"mock": True}), ev("plan", 1, 3.0, {"mock": True}),
                ev("done", 1, 4.0, {"reason": "done", "steps": 1, "mock": True})]
    s = logs.summarize_run(old_mock)
    assert s["agent"] == "mock"
    assert s["models"] == {"lightning": None, "grounder": None, "summary": None, "asr": None}  # "—", not "base"
    assert s["prompt_version"] is None and s["lightning_root"] is None and s["prompt_sha"] == {"lightning": None, "grounder": None, "summary": None}
    new_mock = [dict(e, payload=dict(e["payload"])) for e in old_mock]
    new_mock[1]["payload"]["summary"] = {"model": "mock (no VLM)", "prompt_version": "mock", "prompt_sha": "mock"}
    new_mock[2]["payload"].update(model="mock (no Lightning)", prompt_version="mock", prompt_sha="mock")
    s = logs.summarize_run(new_mock)
    assert s["models"]["lightning"] == "mock (no Lightning)" and s["models"]["summary"] == "mock (no VLM)"
    assert s["prompt_version"] == "mock"
    assert s["prompt_sha"] == {"lightning": "mock", "grounder": None, "summary": "mock"}
    # console.json alone can mark a run as mock (the console ran --agent mock)
    assert logs.summarize_run(_done("done"), {"agent": "mock"})["agent"] == "mock"
    assert logs.summarize_run(_done("done"), {"agent": "real"})["agent"] == "real"


def test_prompt_identity_tokens_and_reused_summaries(tmp_path):
    """The fields agent/core.py writes since b6b0db2: prompt_version + prompt_sha per plan, summary model / prompt_sha /
    usage / error / reused per observation, prompt_sha + usage per grounding."""
    events = current_agent_trace()
    s = logs.summarize_run(events, run_id=CURRENT_RUN)
    assert s["prompt_version"] == "agent-v5"
    assert s["prompt_sha"] == {"lightning": SHA["plan"], "grounder": SHA["grounding"], "summary": SHA["summary"]}
    assert s["models"] == {"lightning": "lightning", "grounder": "grounder", "summary": "base", "asr": None}
    # the summary p50 counts real summarizer calls only: the reused observation made none (no summary timing)
    assert s["stage_p50_ms"]["summary"] == round(statistics.median([45000.0, 900.0]), 1)
    assert (s["status"], s["steps"], s["n_errors"]) == ("done", 3, 0)

    d = write_run(tmp_path, CURRENT_RUN, events)
    evs = logs.run_detail(d, CURRENT_RUN)["events"]
    assert [(e["kind"], e["prompt_sha"]) for e in evs if e["prompt_sha"]] == [
        ("observation", SHA["summary"]), ("plan", SHA["plan"]), ("grounding", SHA["grounding"]),
        ("observation", SHA["summary"]), ("plan", SHA["plan"]), ("grounding", SHA["grounding"]),
        ("observation", SHA["summary"]), ("plan", SHA["plan"])]
    assert all(e["prompt_sha"] is None for e in evs if e["kind"] in ("start", "tool_call", "action", "done"))
    obs = [e for e in evs if e["kind"] == "observation"]
    assert [e["model"] for e in obs] == ["base"] * 3  # a reused description still names the model that wrote it
    assert [e["payload"]["summary"].get("reused") for e in obs] == [None, None, True]
    assert all(e["payload"] == events[i]["payload"] for i, e in enumerate(evs))  # kept as written
    # payload-declared names beat the config, even when the config disagrees
    assert logs.event_model(events[2], {"lightning": {"model": "other"}}) == "lightning"
    assert logs.event_prompt_sha({"kind": "observation", "payload": {"summary": "a string"}}) is None
    assert logs.event_prompt_sha({"kind": "action", "payload": {"prompt_sha": "x"}}) is None
    json.dumps(s, allow_nan=False)


# --- stuck (agent/progress.py) and endpoints -----------------------------------------------------------


def stuck_events() -> list[dict]:
    return logs.read_events(STUCK_FIXTURES / STUCK_RUN / "events.jsonl")[0]


def stuck_console() -> dict:
    return json.loads((STUCK_FIXTURES / STUCK_RUN / "console.json").read_text())


def test_stuck_fixture_matches_progress_tracker():
    """The fixture's stuck verdicts, reason and question are what agent/progress.py produces for its actions."""
    from agent.progress import ProgressTracker

    events = stuck_events()
    tr = ProgressTracker()
    verdicts = []
    for e in (e for e in events if e["kind"] == "action"):
        p = e["payload"]
        c = p.get("click") or {}
        verdicts.append(tr.record(p["name"], p["arguments"], ok=p["ok"], screen_changed=p["screen_changed"],
                                  click_xy=(c["x"], c["y"]) if "x" in c else None))
        assert p.get("stuck") == ({"verdict": verdicts[-1], "reason": tr.reason} if verdicts[-1] else None), e
    assert verdicts == [None, None, "warn", None, None, "stuck"]
    done = events[-1]["payload"]
    assert done["reason"] == "stuck" and done["success_claimed"] is False and done["summary"] == tr.summary()
    assert done["summary"].startswith("I'm stuck: ") and done["summary"].endswith("ask me to try something else.")
    # every stat agent/core.py documents, plus latency_ms
    assert set(done["stats"]) == {"n_tool_calls", "n_invalid_calls", "n_actions", "n_clicks", "n_failed_actions",
                                  "n_no_tool_call", "n_truncated", "n_no_change", "n_recoveries", "n_thinking_calls",
                                  "n_api_errors", "n_stuck_warnings", "n_stuck_stops", "latency_ms"}
    assert all(e["kind"] in ("start", "observation", "plan", "tool_call", "grounding", "action", "done") for e in events)


def test_stuck_summary_status_and_counts():
    events, console = stuck_events(), stuck_console()
    s = logs.summarize_run(events, console, run_id=STUCK_RUN)
    assert (s["status"], s["steps"], s["success_claimed"], s["verified"]) == ("stuck", 6, False, False)
    assert (s["n_stuck_warnings"], s["n_stuck_stops"], s["n_errors"], s["n_invalid_tool_calls"]) == (1, 1, 0, 0)
    assert s["models"] == {"lightning": "lightning", "grounder": "grounder", "summary": "grounder", "asr": None}
    assert s["agent"] == "real" and s["task_id"] == "text-size-large"
    json.dumps(s, allow_nan=False)

    # without the done event's stats (a run cut off, or an older writer): counted from the action verdicts
    no_stats = [dict(e, payload={k: v for k, v in e["payload"].items() if k != "stats"}) for e in events]
    assert logs.stuck_counts(no_stats, no_stats[-1]["payload"]) == (1, 1)
    still_going = [e for e in events if e["kind"] != "done"][:20]  # up to step 3's warning, no done yet
    assert logs.stuck_counts(still_going, {}) == (1, 0)  # the progress check ran (a verdict) and hasn't stopped it
    assert logs.summarize_run(still_going, None)["status"] == "unknown"
    # traces from before the progress check: not recorded (null), not 0
    assert logs.stuck_counts(hand_trace(), hand_trace()[-1]["payload"]) == (None, None)
    assert logs.summarize_run(_done("stuck"))["n_stuck_stops"] == 1  # the reason alone says it stopped as stuck
    zero = {"reason": "done", "stats": {"n_stuck_warnings": 0, "n_stuck_stops": 0}}
    assert logs.stuck_counts([], zero) == (0, 0)
    assert logs.stuck_counts([], {"stats": {"n_stuck_warnings": True, "n_stuck_stops": -1}}) == (None, None)


def test_endpoints_from_console_json_else_the_start_event():
    s = logs.summarize_run(stuck_events(), stuck_console(), run_id=STUCK_RUN)
    assert s["endpoints"] == {
        "vlm": {"base_url": "http://127.0.0.1:8011/v1", "label": "Mac (vllm-metal)", "host": "127.0.0.1:8011",
                "summary_model": "grounder", "served": ["grounder"], "reachable": True},
        "lightning": {"base_url": "http://127.0.0.1:8000/v1", "model": "lightning", "label": "local BF16 (:8000)",
                      "host": "127.0.0.1:8000"},
        "source": "console"}

    # no console.json: from the start event's config (the label follows the port; nobody checked what it served)
    derived = logs.summarize_run(stuck_events(), None)["endpoints"]
    assert derived["source"] == "trace"
    assert derived["vlm"] == {**s["endpoints"]["vlm"], "served": None, "reachable": None}
    assert derived["lightning"] == s["endpoints"]["lightning"]
    old = json.loads((FIXTURES / DONE_RUN / "events.jsonl").read_text().splitlines()[0])  # recorded on :8001
    ep = logs.summarize_run([old], console_doc())["endpoints"]  # an older console.json: no endpoints key
    assert ep["source"] == "trace" and (ep["vlm"]["label"], ep["vlm"]["host"], ep["vlm"]["summary_model"]) == (
        "H200 via tunnel", "127.0.0.1:8001", "base")
    assert ep["lightning"]["label"] == "local BF16 (:8000)" and "root" not in ep["lightning"]

    # mock runs ran no models; junk in console.json is ignored, not shown
    mock = [ev("start", 0, 1.0, {"config": {"agent": "mock", "lightning": "hosted"}, "mock": True})]
    assert logs.summarize_run(mock, {"agent": "mock", "endpoints": {"vlm": None, "lightning": None}})["endpoints"] == {
        "vlm": None, "lightning": None, "source": None}
    junk = console_doc(endpoints={"vlm": {"base_url": 5}, "lightning": "x"})
    assert logs.summarize_run(hand_trace(), junk)["endpoints"]["source"] == "trace"  # hand_trace: lightning only
    assert logs.summarize_run([], None)["endpoints"] == {"vlm": None, "lightning": None, "source": None}
    # the hosted API
    hosted = [ev("start", 0, 1.0, {"config": {"lightning": {"base_url": "https://integrate.api.nvidia.com/v1",
                                                             "model": "nvidia/nemotron-3.5-lightning-30b-a3b"}}})]
    assert logs.summarize_run(hosted)["endpoints"]["lightning"]["label"] == "hosted API"


def test_stuck_run_over_http(console, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(STUCK_FIXTURES / STUCK_RUN, runs / STUCK_RUN)
    base, _ = console(idle_agent(), runs_dir=runs)
    listed = httpx.get(f"{base}/api/logs/runs").json()["runs"]
    assert [(r["run_id"], r["status"], r["n_stuck_warnings"], r["n_stuck_stops"]) for r in listed] == [
        (STUCK_RUN, "stuck", 1, 1)]
    detail = httpx.get(f"{base}/api/logs/runs/{STUCK_RUN}").json()
    assert detail["summary"]["endpoints"]["source"] == "console" and detail["console"] == stuck_console()
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds[-2:] == ["done", "verdict"] and kinds.count("action") == 6
    stuck = [e["payload"]["stuck"]["verdict"] for e in detail["events"] if "stuck" in e.get("payload", {})]
    assert stuck == ["warn", "stuck"]


# --- reading and details --------------------------------------------------------------------------


def test_read_events_skips_and_counts_junk(tmp_path):
    good = [ev("start", 0, 1.0, {"request": "x"}), ev("plan", 1, 2.0, timings_ms={"plan": 7.0})]
    junk = (b"\xff\xfe\x00 not utf-8\n" b"[1, 2]\n" b"\"just a string\"\n" b"\n   \n"
            b'{"kind": "plan", "step": 2, "ts": NaN, "timings_ms": {"plan": Infinity}}\n'
            b'{"run_id": "r1", "kind": "act')  # a run killed mid-write
    d = write_run(tmp_path, "r1", good, raw_tail=junk)
    events, bad = logs.read_events(d / "events.jsonl")
    assert [e["kind"] for e in events] == ["start", "plan", "plan"] and bad == 4
    assert events[2]["ts"] is None and events[2]["timings_ms"]["plan"] is None  # NaN/Infinity -> null
    detail = logs.run_detail(d, "r1")
    assert detail["summary"]["n_bad_lines"] == 4 and detail["summary"]["n_events"] == 3
    assert detail["summary"]["stage_p50_ms"]["plan"] == 7.0
    json.dumps(detail, allow_nan=False)
    assert logs.read_events(tmp_path / "missing.jsonl") == ([], 0)
    (d / "console.json").write_bytes(b"\xff not json")
    assert logs.read_console_json(d / "console.json") is None
    (d / "console.json").write_text("[1]")
    assert logs.read_console_json(d / "console.json") is None


def test_detail_derived_fields_and_console_events(tmp_path):
    events = hand_trace()
    events[1]["screenshot_path"] = "/somewhere/else/r1/step_00.png"   # recorded on another machine: by basename
    events[4]["overlay_path"] = "/somewhere/else/r1/step_01_overlay.png"  # not in the run dir: no URL
    events[6]["screenshot_path"] = "/somewhere/else/r1/notes.txt"         # not a step image: no URL
    d = write_run(tmp_path, "r1", events, console_doc())
    png(d / "step_00.png")
    (d / "notes.txt").write_text("x")
    out = logs.run_detail(d, "r1")
    assert set(out) == {"summary", "console", "events"} and out["console"] == console_doc()
    evs = out["events"]
    assert len(evs) == len(events) + 2
    voice, verdict, agent = evs[0], evs[-1], evs[1:-1]
    assert voice == {"run_id": "r1", "step": None, "kind": "voice", "ts": None, "t_ms": None, "payload": VOICE,
                     "model": VOICE["model"], "source": "console"}
    assert verdict["kind"] == "verdict" and verdict["source"] == "console" and verdict["model"] is None
    assert verdict["payload"] == {"passed": False, "explanation": "the statement was not downloaded",
                                  "label": "verified by sandbox state, not seen by the agent", "task_id": TASK}
    assert verdict["ts"] == 1011.0 and verdict["t_ms"] == 11000.0
    assert [e["kind"] for e in agent] == [e["kind"] for e in events] and all(e["source"] == "agent" for e in agent)
    assert [e["t_ms"] for e in agent][:3] == [0.0, 1000.0, 2000.0] and agent[-1]["t_ms"] == 10500.0
    assert [e["model"] for e in agent[:5]] == [None, "base", "lightning-served", None, "grounder-from-payload"]
    assert agent[6]["model"] == "summary-from-payload" and agent[7]["model"] is None  # unknown kind
    assert agent[1]["screenshot_url"] == "/runs/r1/step_00.png"
    assert "overlay_url" not in agent[4] and "screenshot_url" not in agent[6]
    assert agent[1]["payload"] == events[1]["payload"]  # the original event is kept as is

    bare = logs.run_detail(write_run(tmp_path, "r2", events, console_doc(voice=None, verdict=None)), "r2")
    assert [e["source"] for e in bare["events"]] == ["agent"] * len(events)
    assert logs.run_detail(write_run(tmp_path, "r3", events), "r3")["console"] is None


def test_run_ids_and_find_run_dir(tmp_path):
    for good in ("20261017-175025-e9fe8e", "a", "x.y_z-1", "a" * 80):
        assert logs.valid_run_id(good), good
    for bad in ("", ".", "..", "...", "a/b", "..%2F", "a b", "a" * 81, None, 5, "ü"):
        assert not logs.valid_run_id(bad), bad
    write_run(tmp_path / "runs", "r1", hand_trace())
    (tmp_path / "events.jsonl").write_text("{}\n")  # what '..' would reach
    assert logs.find_run_dir([tmp_path / "runs"], "r1") == (tmp_path / "runs" / "r1").resolve()
    assert logs.find_run_dir([tmp_path / "runs"], "..") is None
    assert logs.find_run_dir([tmp_path / "runs"], "nope") is None


def test_list_runs_roots_order_limit_and_cache(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, "old", [ev("start", 0, 100.0)])
    write_run(a, "new", [ev("start", 0, 300.0)])
    write_run(a, "shared", [ev("start", 0, 200.0, {"request": "from a"})])
    write_run(b, "shared", [ev("start", 0, 250.0, {"request": "from b"})])   # same id: the first root wins
    write_run(b, "only-b", [ev("start", 0, 150.0)])
    (a / "no-events").mkdir()                                                 # not a run
    write_run(a, "bad name", [ev("start", 0, 400.0)])                         # can't be addressed: skipped
    (a / "stray.txt").write_text("x")
    write_run(a, "no-ts", [{"kind": "start"}])                                # sorted by the file's mtime
    roots = [a, b, a, tmp_path / "b" / ".." / "a", tmp_path / "missing"]
    assert logs.dedupe_roots(roots + [None]) == [a.resolve(), b.resolve(), (tmp_path / "missing").resolve()]

    index = logs.RunIndex()
    runs = logs.list_runs(roots, index=index)
    assert [r["run_id"] for r in runs] == ["no-ts", "new", "shared", "only-b", "old"]
    assert next(r for r in runs if r["run_id"] == "shared")["request"] == "from a"
    assert [r["run_id"] for r in logs.list_runs(roots, limit=2, index=index)] == ["no-ts", "new"]
    assert index.reads == 5  # the second listing came from the cache

    time.sleep(0.01)
    with (a / "old" / "events.jsonl").open("a") as f:  # a run still being written
        f.write(json.dumps(ev("done", 1, 500.0, {"reason": "done", "steps": 1})) + "\n")
    (a / "new" / "console.json").write_text(json.dumps(console_doc()))
    runs = logs.list_runs(roots, index=index)
    assert index.reads == 7
    old = next(r for r in runs if r["run_id"] == "old")
    assert old["status"] == "done" and old["ended_ts"] == 500.0
    assert next(r for r in runs if r["run_id"] == "new")["console_run_id"] == "c-1234abcd"
    shutil.rmtree(a / "old")
    assert "old" not in [r["run_id"] for r in logs.list_runs(roots, index=index)]


def test_list_runs_is_fast_for_500_runs(tmp_path):
    root = tmp_path / "runs"
    trace = hand_trace()
    for e in trace:  # a realistic line length
        e["payload"]["text"] = "PAGE: Acme Utilities | Billing | Amount Due $142.37 " * 20
    for i in range(500):
        write_run(root, f"20261017-{i:06d}-abc", trace, console_doc() if i % 2 else None)
    index = logs.RunIndex()
    t = time.perf_counter()
    assert len(logs.list_runs([root], limit=200, index=index)) == 200
    first = time.perf_counter() - t
    t = time.perf_counter()
    logs.list_runs([root], limit=200, index=index)
    cached = time.perf_counter() - t
    assert index.reads == 500
    assert first < 5.0 and cached < 0.5, (first, cached)


def test_real_traces(tmp_path):
    root = tmp_path / "runs"
    copy_real_traces(root)
    runs = logs.list_runs([root])
    assert [r["run_id"] for r in runs] == [DONE_RUN, STOPPED_RUN]
    done, stopped = runs

    assert done["agent"] == "real" and done["request"] == "Download my September statement."
    assert (done["status"], done["steps"], done["success_claimed"]) == ("done", 2, True)
    assert (done["started_ts"], done["ended_ts"], done["total_ms"]) == (1792273825.135, 1792273839.482, 14347)
    assert done["models"] == {"lightning": "lightning", "grounder": "base", "summary": "base", "asr": None}
    assert done["lightning_root"] == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    assert done["prompt_version"] == "agent-v4" and done["prompt_sha"] == {"lightning": None, "grounder": None, "summary": None}  # recorded before b6b0db2
    assert done["stage_p50_ms"] == {"voice": None, "summary": round(statistics.median([2102.1, 2101.6]), 1),
                                    "plan": round(statistics.median([3025.3, 4465.6]), 1), "ground": 675.7,
                                    "act": 18.2}
    assert (done["n_events"], done["n_errors"], done["n_invalid_tool_calls"], done["n_bad_lines"]) == (11, 0, 0, 0)
    assert (done["console_run_id"], done["task_id"], done["verified"], done["voice"]) == (None, None, None, None)

    assert (stopped["status"], stopped["steps"], stopped["success_claimed"]) == ("stopped", 1, False)
    assert stopped["stage_p50_ms"] == {"voice": None, "summary": 1571.1, "plan": 2864.6, "ground": None, "act": None}
    assert stopped["total_ms"] == round((1792273772.882 - 1792273766.236) * 1000) and stopped["n_events"] == 5

    detail = logs.run_detail(root / DONE_RUN, DONE_RUN)
    evs = detail["events"]
    assert [e["kind"] for e in evs] == ["start", "observation", "plan", "tool_call", "grounding", "action",
                                        "observation", "plan", "tool_call", "action", "done"]
    assert detail["console"] is None and all(e["source"] == "agent" for e in evs)
    assert evs[1]["screenshot_url"] == f"/runs/{DONE_RUN}/step_00.png"
    assert evs[4]["overlay_url"] == f"/runs/{DONE_RUN}/step_01_overlay.png"
    assert evs[6]["screenshot_url"] == f"/runs/{DONE_RUN}/step_01.png"
    assert [e["model"] for e in evs[1:5]] == ["base", "lightning", None, "base"]
    assert all(e["prompt_sha"] is None for e in evs)
    assert evs[0]["t_ms"] == 0.0 and evs[-1]["t_ms"] == 14347.0


# --- HTTP ------------------------------------------------------------------------------------------


def idle_agent():
    async def run_task(request, *, start_url, cfg, on_event=None):
        raise AssertionError("not used")

    return run_task, lambda options: {}


def test_logs_api_list_detail_download_and_page(console, tmp_path):
    runs = tmp_path / "runs"
    copy_real_traces(runs)
    static = tmp_path / "static"
    base, app = console(idle_agent(), runs_dir=runs, static_dir=static)

    body = httpx.get(f"{base}/api/logs/runs").json()
    assert [r["run_id"] for r in body["runs"]] == [DONE_RUN, STOPPED_RUN] and body["total"] == 2
    assert body["runs"][0] == logs.list_runs([runs])[0]
    one = httpx.get(f"{base}/api/logs/runs", params={"limit": 1}).json()
    assert [r["run_id"] for r in one["runs"]] == [DONE_RUN] and one["total"] == 2
    for bad in ("0", "-1", "abc", "1001", "1.5"):
        r = httpx.get(f"{base}/api/logs/runs", params={"limit": bad})
        assert r.status_code == 400 and r.json()["error"], bad

    detail = httpx.get(f"{base}/api/logs/runs/{DONE_RUN}").json()
    assert set(detail) == {"summary", "console", "events"} and detail["summary"]["status"] == "done"
    assert detail == json.loads(json.dumps(logs.run_detail(runs / DONE_RUN, DONE_RUN)))

    r = httpx.get(f"{base}/api/logs/runs/{DONE_RUN}/download")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.headers["content-disposition"] == f'attachment; filename="run-{DONE_RUN}.json"'
    assert r.json() == detail

    # images of older runs are served from the run roots (the runner never saw these runs)
    for url in (detail["events"][1]["screenshot_url"], detail["events"][4]["overlay_url"]):
        img = httpx.get(base + url)
        assert img.status_code == 200 and img.headers["content-type"] == "image/png" and img.content[:4] == b"\x89PNG"

    assert httpx.get(f"{base}/logs").status_code == 404  # page not built
    static.mkdir()
    (static / "logs.html").write_text("<!doctype html><title>logs</title>")
    page = httpx.get(f"{base}/logs")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"] and "logs" in page.text


def test_logs_api_rejects_bad_ids(console, tmp_path):
    runs = tmp_path / "runs"
    copy_real_traces(runs)
    write_run(tmp_path, "outside", hand_trace())          # a sibling of runs/: '..' tricks must not reach it
    (tmp_path / "events.jsonl").write_text(json.dumps(hand_trace()[0]) + "\n")
    base, _ = console(idle_agent(), runs_dir=runs)
    for path in ("/api/logs/runs/..", "/api/logs/runs/%2E%2E", "/api/logs/runs/%2e%2e/download",
                 "/api/logs/runs/..%2Foutside", "/api/logs/runs/..%2F..%2Fetc%2Fpasswd",
                 "/api/logs/runs/%2E%2E%2Foutside/download", "/api/logs/runs/.", "/api/logs/runs/a%20b",
                 "/api/logs/runs/" + "a" * 81, "/api/logs/runs/nope", "/api/logs/runs/nope/download",
                 "/api/logs/runs/outside", f"/api/logs/runs/{DONE_RUN}%2F..",
                 "/runs/..%2Foutside/step_00.png", "/runs/%2E%2E/step_00.png"):
        status, body = raw_get(base, path)
        assert status == 404, (path, status, body[:200])
        assert b"Do the thing." not in body, path


@pytest.mark.parametrize("voice", [
    "heard it", 5, ["text"], {"text": "x"},
    {**VOICE, "extra": 1},
    {**VOICE, "text": None}, {**VOICE, "text": "x" * 501}, {**VOICE, "model": 3}, {**VOICE, "model": "m" * 121},
    {**VOICE, "latency_ms": "12"}, {**VOICE, "latency_ms": -1}, {**VOICE, "latency_ms": True},
    {**VOICE, "duration_s": None}, {**VOICE, "duration_s": float("nan")}, {**VOICE, "duration_s": float("inf")},
])
def test_bad_voice_400(console, tmp_path, voice):
    base, _ = console(idle_agent())
    r = httpx.post(f"{base}/api/run", content=json.dumps({"request": "hi", "voice": voice}),
                   headers={"content-type": "application/json"})
    assert r.status_code == 400 and "voice" in r.json()["error"], voice


def test_parse_run_body_voice():
    tasks = {TASK: {}}
    clean, err = parse_run_body({"request": "hi"}, tasks)
    assert err is None and clean["voice"] is None
    clean, err = parse_run_body({"request": "hi", "voice": None}, tasks)
    assert err is None and clean["voice"] is None
    edge = {"text": "", "model": "stub", "latency_ms": 0, "duration_s": 60}
    clean, err = parse_run_body({"request": "hi", "voice": edge}, tasks)
    assert err is None and clean["voice"] == edge
    clean, err = parse_run_body({"request": "hi", "voice": {**VOICE, "text": "x" * 500, "model": "m" * 120}}, tasks)
    assert err is None


class RecordingCfg:
    def __init__(self, root: Path, options: dict):
        self.run_dir_root, self.options = root, options


def recording_agent(root: Path, gate: threading.Event, seen: list):
    """Writes a real-agent-shaped events.jsonl under cfg.run_dir_root and records everything it is given.
    It waits on `gate` after its start event, so the test can look at the live state."""

    async def run_task(request, *, start_url, cfg, on_event=None, **kwargs):
        seen.append({"request": request, "start_url": start_url, "cfg": vars(cfg), "kwargs": kwargs})
        rid = f"20261017-000000-{time.time_ns() % 10**8:08d}"
        d = Path(cfg.run_dir_root) / rid
        d.mkdir(parents=True)
        f = (d / "events.jsonl").open("a")

        async def emit(step, kind, payload, **extra):
            e = {"run_id": rid, "step": step, "kind": kind, "ts": round(time.time(), 3), "payload": payload, **extra}
            f.write(json.dumps(e) + "\n")
            f.flush()
            await on_event(e)

        try:
            await emit(0, "start", {"request": request, "start_url": start_url,
                                    "config": {"lightning": {"model": "lightning"}, "grounder_model": "base",
                                               "summary_model": "base"}})
            while not gate.is_set():
                await asyncio.sleep(0.02)
            png(d / "step_00.png")
            await emit(0, "observation", {"observation": {}}, screenshot_path=str(d / "step_00.png"),
                       timings_ms={"summary": 20.0})
            await emit(1, "plan", {"tool_calls": [{"name": "done"}]}, timings_ms={"plan": 30.0})
            await emit(1, "tool_call", {"name": "done", "arguments": {}, "valid": True})
            await emit(1, "done", {"success_claimed": True, "summary": "done", "steps": 1, "reason": "done"})
        finally:
            f.close()
        return {"run_id": rid, "success_claimed": True, "summary": "done", "steps": 1, "trace_dir": str(d)}

    return run_task, lambda options: RecordingCfg(root, options)


def wait_for(fn, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = fn()
        if out:
            return out
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_console_json_through_post_run_and_voice_never_reaches_the_agent(console, tmp_path):
    runs = tmp_path / "runs"
    gate, seen = threading.Event(), []
    base, _ = console(recording_agent(runs, gate, seen), runs_dir=runs, agent="real")
    r = httpx.post(f"{base}/api/run", json={"request": "Download my September statement.", "task_id": TASK,
                                            "voice": VOICE})
    assert r.status_code == 200, r.text
    run = r.json()

    # console.json exists as soon as the agent's run id is known (start event, no screenshot yet)
    path = wait_for(lambda: next(runs.glob("*/console.json"), None))
    live = json.loads(path.read_text())
    rid = path.parent.name
    assert live["status"] == "running" and live["console_run_id"] == run["run_id"] and live["voice"] == VOICE
    assert live["verdict"] is None and live["ended_ts"] is None and live["agent"] == "real"
    assert not list(path.parent.glob("*.png"))  # written before any screenshot, from cfg.run_dir_root
    assert {k: live[k] for k in ("task_id", "sid", "start_url", "request")} == {
        "task_id": TASK, "sid": run["sid"], "start_url": run["start_url"], "request": "Download my September statement."}
    listed = httpx.get(f"{base}/api/logs/runs").json()["runs"]
    assert [(x["run_id"], x["status"]) for x in listed] == [(rid, "running")]

    gate.set()
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    final = json.loads(path.read_text())  # already final when `end` arrives
    assert final["status"] == "done" and final["voice"] == VOICE and final["result"]["run_id"] == rid
    assert final["verdict"]["task_id"] == TASK and final["verdict"]["passed"] is False  # nothing was downloaded
    assert final["started_ts"] <= final["ended_ts"]
    assert not list(path.parent.glob(".console.json.*"))  # no temp files left behind

    # the voice transcript went nowhere near the agent
    assert len(seen) == 1 and seen[0]["kwargs"] == {}
    assert VOICE["text"] not in json.dumps(seen, default=str)
    assert VOICE["text"] not in (path.parent / "events.jsonl").read_text()
    assert VOICE["text"] not in json.dumps([m for m in stream if m["event"] == "step"])

    detail = httpx.get(f"{base}/api/logs/runs/{rid}").json()
    s = detail["summary"]
    assert (s["status"], s["console_run_id"], s["task_id"], s["verified"]) == ("done", run["run_id"], TASK, False)
    assert s["agent"] == "real" and s["models"] == {"lightning": "lightning", "grounder": "base", "summary": "base",
                                                    "asr": VOICE["model"]}
    assert s["stage_p50_ms"] == {"voice": 812.4, "summary": 20.0, "plan": 30.0, "ground": None, "act": None}
    assert s["voice"] == VOICE and s["models"]["asr"] == VOICE["model"] and s["stage_p50_ms"]["voice"] == 812.4
    assert detail["console"] == final
    assert detail["events"][0]["kind"] == "voice" and detail["events"][0]["source"] == "console"
    assert detail["events"][-1]["kind"] == "verdict" and detail["events"][-1]["payload"]["passed"] is False
    assert detail["events"][2]["screenshot_url"] == f"/runs/{rid}/step_00.png"


def test_stale_running_console_json_is_unknown(console, tmp_path):
    """A console killed mid-run leaves console.json at 'running': the logs say 'unknown', not 'running'."""
    runs = tmp_path / "runs"
    copy_real_traces(runs)
    events = (runs / STOPPED_RUN / "events.jsonl").read_text().splitlines()
    (runs / STOPPED_RUN / "events.jsonl").write_text("\n".join(events[:-1]) + "\n")  # no done event
    (runs / STOPPED_RUN / "console.json").write_text(json.dumps(console_doc(status="running", verdict=None)))
    base, _ = console(idle_agent(), runs_dir=runs)
    summary = httpx.get(f"{base}/api/logs/runs/{STOPPED_RUN}").json()["summary"]
    assert summary["status"] == "unknown" and summary["console_run_id"] == "c-1234abcd"


@pytest.mark.e2e
def test_real_mock_run_with_voice(console, tmp_path):
    """The real mock agent (headless Chromium) through POST /api/run: console.json, the logs API, and the
    voice transcript never reaching the agent."""
    runs = tmp_path / "runs"
    base, _ = console(None, runs_dir=runs, step_delay_s=0.0)
    r = httpx.post(f"{base}/api/run", json={"request": "Download my September statement.", "task_id": TASK,
                                            "voice": {**VOICE, "model": "stub"}})
    assert r.status_code == 200, r.text
    run = r.json()
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events", timeout=60))
    result = next(m["data"] for m in stream if m["event"] == "result")
    assert next(m["data"] for m in stream if m["event"] == "verdict")["passed"] is True
    rid = result["run_id"]

    doc = json.loads((runs / rid / "console.json").read_text())
    assert doc["status"] == "done" and doc["agent"] == "mock" and doc["console_run_id"] == run["run_id"]
    assert doc["voice"] == {**VOICE, "model": "stub"} and doc["verdict"]["passed"] is True
    assert VOICE["text"] not in (runs / rid / "events.jsonl").read_text()
    assert VOICE["text"] not in json.dumps([m for m in stream if m["event"] == "step"])

    detail = httpx.get(f"{base}/api/logs/runs/{rid}").json()
    s = detail["summary"]
    assert (s["agent"], s["status"], s["verified"], s["success_claimed"]) == ("mock", "done", True, True)
    assert s["models"] == {"lightning": "mock (no Lightning)", "grounder": "grounder, mock", "summary": "mock (no VLM)",
                           "asr": "stub"}
    assert s["prompt_version"] == "mock" and s["lightning_root"] is None
    assert s["prompt_sha"] == {"lightning": "mock", "grounder": "mock", "summary": "mock"}
    p50 = s["stage_p50_ms"]
    assert (p50["voice"], p50["plan"], p50["summary"], p50["ground"]) == (812.4, None, None, None)  # nothing invented
    assert p50["act"] > 0 and s["n_errors"] == 0 and s["n_bad_lines"] == 0
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds[0] == "voice" and kinds[1] == "start" and kinds[-2:] == ["done", "verdict"]
    grounding = next(e for e in detail["events"] if e["kind"] == "grounding")
    assert grounding["payload"]["prompt_version"] == "mock" and grounding["payload"]["mock"] is True
    assert grounding["payload"]["prompt_sha"] == grounding["prompt_sha"] == "mock"
    acts = [e for e in detail["events"] if e["kind"] == "action" and e["payload"]["name"] != "done"]
    assert acts and all(set(e["timings_ms"]) == {"act", "settle"} for e in acts)  # split like the real agent
    assert p50["act"] == round(statistics.median(e["timings_ms"]["act"] for e in acts), 1)
    shot = next(e for e in detail["events"] if e.get("screenshot_url"))
    img = httpx.get(base + shot["screenshot_url"])
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    assert [x["run_id"] for x in httpx.get(f"{base}/api/logs/runs").json()["runs"]] == [rid]
