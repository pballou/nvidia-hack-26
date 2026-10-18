"""End to end with the REAL mock agent: all 12 sandbox tasks through POST /api/run and the SSE stream.

The mock drives headless Chromium from pixels only: mouse clicks and wheel and keys at the coordinates in
console/mock_scripts.json (measured by console/tools/calibrate_mock.py). It never reads the DOM or the sandbox state,
and the console checks the verdict against /api/state only after the run. So a passing verdict here shows the task is
solvable by pixel clicks alone, and that the stream carries everything the console UI renders: tool calls in plain
words, grounding points with overlay images, screenshots, measured act/settle/screenshot timings, the done summary
and the verdict.

The console's request check is on by default (python -m console --clarify on), so every run here starts with the
mock's step-0 clarify event (proceed, the request restated as itself) before the browser opens; one task also runs
with the check off (no clarify event at all). The ask flows and the follow-up are in test_clarify_api.py.

The browser-driving tests are marked `e2e` and run in the default suite (about 45 s for all 12 tasks plus the
free-text runs). Skip them with -m "not e2e"; the fast script-integrity checks below still run.
If it fails after a sandbox layout change, re-run python -m console.tools.calibrate_mock.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from console import mock_agent
from console.app import ConsoleSettings, create_app
from console.runner import VERDICT_LABEL
from console.tests.conftest import ServerThread, free_port, messages, read_sse
from console.tools import calibrate_mock

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO / "sandbox" / "tasks.json").read_text())
SCRIPTS_FILE = json.loads(mock_agent.SCRIPTS_PATH.read_text())
SCRIPTS = SCRIPTS_FILE["tasks"]
W, H = mock_agent.VIEWPORT["width"], mock_agent.VIEWPORT["height"]
OPTIONS = {"grounder": "base", "thinking": "retry", "lightning": "local"}  # clarify: the console's default (on)


@pytest.fixture(scope="module")
def mock_console(sandbox_url, tmp_path_factory):
    """One console (--agent mock, no step delay, headless) for the whole module: (base_url, runs_dir)."""
    runs = tmp_path_factory.mktemp("e2e-runs")
    app = create_app(ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, step_delay_s=0.0,
                                     headed=False, static_dir=runs / "no-static", stop_timeout_s=10.0))
    srv = ServerThread(app)
    yield srv.start(), runs
    srv.stop()


def run_via_console(base: str, request: str, task_id: str | None = None,
                    options: dict | None = None) -> tuple[dict, list[dict]]:
    r = httpx.post(f"{base}/api/run", json={"request": request, "task_id": task_id, "options": options or OPTIONS})
    assert r.status_code == 200, r.text
    run = r.json()
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events", timeout=120))
    assert [m["id"] for m in stream] == list(range(1, len(stream) + 1))
    return run, stream


def assert_png(base: str, url: str, size=(W, H)) -> None:
    r = httpx.get(base + url)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png", url
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.size == size, url


def needs_grounding(step: dict) -> bool:
    return step["tool"] == "click" or (step["tool"] == "type_text" and bool(step["arguments"].get("target_description")))


# --- the 12 tasks -------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_task_solved_by_pixel_clicks(mock_console, task):
    base, _ = mock_console
    script = SCRIPTS[task["id"]]
    run, stream = run_via_console(base, task["request"], task["id"])

    kinds = [m["event"] for m in stream]
    assert kinds[:2] == ["status", "status"] and kinds[-4:] == ["result", "verdict", "status", "end"], kinds
    assert [m["data"]["status"] for m in stream if m["event"] == "status"] == ["starting", "running", "done"]
    result, verdict = stream[-4]["data"], stream[-3]["data"]
    assert result["success_claimed"] is True, result
    assert verdict == {"task_id": task["id"], "passed": True, "explanation": verdict["explanation"],
                       "label": VERDICT_LABEL}, verdict
    assert result["steps"] == len(script["steps"]) and result["summary"] == script["steps"][-1]["arguments"]["summary"]

    steps = [m["data"] for m in stream if m["event"] == "step"]
    assert all(s["payload"]["mock"] is True and s["run_id"] == result["run_id"] for s in steps)
    assert not [s["payload"] for s in steps if s["kind"] == "error"]
    by_step: dict[int, dict[str, dict]] = {}
    for s in steps:
        assert s["kind"] not in by_step.setdefault(s["step"], {}), f"two {s['kind']} events in step {s['step']}"
        by_step[s["step"]][s["kind"]] = s
    assert set(by_step) == set(range(len(script["steps"]) + 1))

    start = by_step[0]
    assert start["start"]["payload"]["start_url"] == run["start_url"]
    # the request check: step 0, right after start and before the browser's first observation
    assert [s["kind"] for s in steps[:3]] == ["start", "clarify", "observation"]
    clar = start["clarify"]["payload"]
    assert (clar["action"], clar["normalized_request"], clar["prompt_version"]) == (
        "proceed", task["request"], "clarify-v4 (mock)")
    assert start["start"]["payload"]["config"]["clarify"] is True and "timings_ms" not in start["clarify"]
    assert start["observation"]["payload"]["observation"]["screen_summary"] == script["start_screen"]
    assert start["observation"]["payload"]["summary"] == mock_agent.MOCK_SUMMARY_INFO
    assert set(start["observation"]["timings_ms"]) == {"screenshot"}  # measured; no summary latency invented
    assert_png(base, start["observation"]["screenshot_url"])

    for n, want in enumerate(script["steps"], 1):
        ev = by_step[n]
        plan, call = ev["plan"], ev["tool_call"]["payload"]
        assert plan["payload"]["reasoning"] == want["reasoning"] and "timings_ms" not in plan  # nothing simulated
        assert (plan["payload"]["model"], plan["payload"]["prompt_version"], plan["payload"]["prompt_sha"]) == \
            (mock_agent.MOCK_PLANNER, "mock", "mock")
        assert plan["payload"]["tool_calls"] == [{"name": want["tool"], "arguments": want["arguments"]}]
        assert call["name"] == want["tool"] and call["arguments"] == want["arguments"] and call["valid"] is True
        assert call["description"] == mock_agent.describe(want["tool"], want["arguments"])
        if want["tool"] == "done":
            assert set(ev) == {"plan", "tool_call", "action", "done"} and n == len(script["steps"])
            assert ev["action"]["payload"]["done"] is True
            assert ev["done"]["payload"]["reason"] == "done" and ev["done"]["payload"]["success_claimed"] is True
            assert ev["done"]["payload"]["summary"] == result["summary"]
            continue
        action = ev["action"]
        assert action["payload"]["screen_changed"] is True, f"step {n} {want['tool']}: the screen did not change"
        # split like the real agent (agent/session.py): act = the mouse/keyboard action alone, settle = the wait;
        # the screenshot of the settled screen belongs to the observation that follows
        assert set(action["timings_ms"]) == {"act", "settle"} and action["timings_ms"]["act"] > 0
        assert action["timings_ms"]["settle"] >= mock_agent.MockConfig().settle_ms
        obs = ev["observation"]
        assert obs["payload"]["observation"]["screen_summary"] == want["screen_after"]
        assert obs["payload"]["observation"]["screen_changed"] is True
        assert set(obs["timings_ms"]) == {"screenshot"} and obs["timings_ms"]["screenshot"] > 0
        assert obs["screenshot_url"] == f"/runs/{result['run_id']}/step_{n:02d}.png"
        assert_png(base, obs["screenshot_url"])
        if needs_grounding(want):
            g = ev["grounding"]
            assert g["payload"]["target"] == want["arguments"]["target_description"]
            assert (g["payload"]["x"], g["payload"]["y"]) == (want["x"], want["y"])
            assert g["payload"]["point_1000"] == [round(want["x"] / W * 1000), round(want["y"] / H * 1000)]
            assert "timings_ms" not in g and Path(g["overlay_path"]).is_file()
            assert g["payload"]["prompt_sha"] == "mock" and g["payload"]["model"] == "base, mock"
            assert g["overlay_url"] == f"/runs/{result['run_id']}/step_{n:02d}_overlay.png"
            assert_png(base, g["overlay_url"])
            assert action["payload"]["click"] == {"x": want["x"], "y": want["y"]}
        else:
            assert "grounding" not in ev and "click" not in action["payload"] and "overlay_url" not in action

    info = httpx.get(f"{base}/api/runs/{run['run_id']}").json()
    assert info["status"] == "done" and info["verdict"] == verdict and info["result"] == result
    assert info["agent_run_id"] == result["run_id"] and info["n_events"] == len(stream)
    lines = (Path(result["trace_dir"]) / "events.jsonl").read_text().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == [s["kind"] for s in steps]


@pytest.mark.e2e
def test_free_text_request_has_no_verdict(mock_console):
    """No task_id: the mock has no script, looks at the screen, gives up honestly, and the console adds no verdict."""
    base, runs = mock_console
    run, stream = run_via_console(base, "Please water my plants")
    kinds = [m["event"] for m in stream]
    assert "verdict" not in kinds and kinds[-3:] == ["result", "status", "end"], kinds
    assert stream[-2]["data"] == {"status": "done"}
    result = stream[-3]["data"]
    assert result["success_claimed"] is False and result["summary"] == mock_agent.FALLBACK_SUMMARY
    assert run["start_url"].endswith("/?sid=" + run["sid"])

    steps = [m["data"] for m in stream if m["event"] == "step"]
    assert [s["kind"] for s in steps] == ["start", "clarify", "observation", "plan", "tool_call", "action", "done"]
    assert all(s["payload"]["mock"] is True for s in steps)
    assert steps[1]["payload"]["normalized_request"] == "Please water my plants"  # clear: it proceeds
    plan = steps[3]["payload"]
    assert plan["content"] == mock_agent.FALLBACK_PLAN and plan["tool_calls"][0]["name"] == "done"
    assert steps[4]["payload"]["description"].startswith("Finishing: ")
    assert steps[-1]["payload"]["success_claimed"] is False and steps[-1]["payload"]["reason"] == "done"
    obs = steps[2]
    assert obs["payload"]["observation"]["url"].startswith(run["start_url"].split("?")[0])
    assert set(obs["timings_ms"]) == {"screenshot"}
    assert_png(base, obs["screenshot_url"])

    info = httpx.get(f"{base}/api/runs/{run['run_id']}").json()
    assert info["status"] == "done" and info["task_id"] is None and info["verdict"] is None
    lines = (runs / result["run_id"] / "events.jsonl").read_text().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == [s["kind"] for s in steps]


@pytest.mark.e2e
def test_typed_task_text_from_another_page_falls_back_honestly(mock_console):
    """A task's exact text without its task_id starts at '/', not at the task's start page. The script's points
    were measured there, so the mock must not replay them (it used to click the dashboard and then fail with a
    developer-facing "mock script is stale" error). It says why it did nothing instead."""
    base, _ = mock_console
    task = next(t for t in TASKS if t["id"] == "mark-all-read")
    assert task["start_path"] == "/messages"
    run, stream = run_via_console(base, task["request"])
    assert run["start_url"].endswith("/?sid=" + run["sid"])
    kinds = [m["event"] for m in stream]
    assert "verdict" not in kinds and kinds[-3:] == ["result", "status", "end"], kinds
    steps = [m["data"] for m in stream if m["event"] == "step"]
    assert [s["kind"] for s in steps] == ["start", "clarify", "observation", "plan", "tool_call", "action",
                                          "done"]  # no error
    result = stream[-3]["data"]
    assert result["success_claimed"] is False and result["steps"] == 1
    assert "starts on /messages, but this run started on /," in result["summary"]
    assert "stale" not in json.dumps(steps) and "calibrate" not in json.dumps(steps)
    assert steps[2]["payload"]["observation"]["screen_summary"] == mock_agent.NO_SUMMARY  # not the script's text
    assert steps[3]["payload"]["content"].startswith("Mock agent: the scripted plan for this request starts on")


@pytest.mark.e2e
def test_task_with_the_request_check_off(mock_console):
    """--clarify off (or the per-run option): no clarify event at all, and the task still replays and verifies."""
    base, _ = mock_console
    task = next(t for t in TASKS if t["id"] == "high-contrast")
    run, stream = run_via_console(base, task["request"], task["id"], {**OPTIONS, "clarify": "off"})
    steps = [m["data"] for m in stream if m["event"] == "step"]
    assert "clarify" not in [s["kind"] for s in steps] and [s["kind"] for s in steps[:2]] == ["start", "observation"]
    assert steps[0]["payload"]["config"]["clarify"] is False
    verdict = next(m["data"] for m in stream if m["event"] == "verdict")
    assert verdict["task_id"] == "high-contrast" and verdict["passed"] is True, verdict
    assert httpx.get(f"{base}/api/runs/{run['run_id']}").json()["options"]["clarify"] == "off"


# --- what makes the run above meaningful: scripts and mock are pixel-only -----------------------


def test_scripts_cover_every_task_and_hold_no_selectors():
    assert list(SCRIPTS) == [t["id"] for t in TASKS]
    assert SCRIPTS_FILE["viewport"] == mock_agent.VIEWPORT
    for task in TASKS:
        script = SCRIPTS[task["id"]]
        assert script["request"] == task["request"] and set(script) == {"request", "start_path", "start_screen", "steps"}
        assert script["start_path"] == (task["start_path"] or "/")  # the mock replays it only from that page
        calibrate_mock.validate_script(script)  # tool schema, viewport bounds, done last, only replay keys
    text = json.dumps(SCRIPTS)  # the replayed part (the file's _about note mentions "no selectors")
    for needle in ("testid", "selector", "locator", "AcmeTestHooks", "/api/state", "xpath"):
        assert needle not in text, needle


def test_mock_agent_is_pixel_only():
    """The mock touches the page only through goto, screenshots, mouse, keyboard, url and title."""
    tree = ast.parse(Path(mock_agent.__file__).read_text())
    page_attrs = {n.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "page"}
    assert page_attrs <= {"goto", "screenshot", "mouse", "keyboard", "url", "title", "wait_for_load_state"}, page_attrs
    forbidden = {"locator", "evaluate", "evaluate_handle", "query_selector", "query_selector_all", "eval_on_selector",
                 "eval_on_selector_all", "wait_for_selector", "frame_locator", "inner_text", "inner_html",
                 "text_content", "get_attribute", "dispatch_event", "fill", "select_option", "set_input_files",
                 "add_init_script", "expose_function", "route"}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & forbidden and not {a for a in attrs if a.startswith("get_by_")}
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"httpx", "requests", "aiohttp", "sandbox", "agent"}, imported
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [s for s in strings if "/api/" in s or "testid" in s]


def test_validate_script_rejects_bad_scripts():
    ok = {"steps": [{"tool": "click", "arguments": {"target_description": "the gear icon"}, "x": 10, "y": 10,
                     "reasoning": "r", "screen_after": "s", "path_after": "/"},
                    {"tool": "done", "arguments": {"summary": "ok", "success": True}, "reasoning": "r"}]}
    calibrate_mock.validate_script(ok)
    bad = [
        [{**ok["steps"][0], "testid": "topbar-settings"}, ok["steps"][1]],  # a selector leaked
        [{**ok["steps"][0], "x": W + 5}, ok["steps"][1]],  # off screen
        [{**ok["steps"][0], "arguments": {"target": "the gear icon"}}, ok["steps"][1]],  # not the tool schema
        [{"tool": "scroll", "arguments": {"direction": "sideways", "amount": "page"}, "reasoning": "r",
          "screen_after": "s"}, ok["steps"][1]],
        [ok["steps"][0]],  # no done
        [ok["steps"][1], ok["steps"][0]],  # done is not last
    ]
    for steps in bad:
        with pytest.raises(calibrate_mock.CalibrationError):
            calibrate_mock.validate_script({"steps": steps})


def test_calibrate_exits_2_when_the_sandbox_is_down(tmp_path, capsys):
    out = tmp_path / "scripts.json"
    assert calibrate_mock.main(["--sandbox-url", f"http://127.0.0.1:{free_port()}", "--out", str(out)]) == 2
    assert "sandbox not reachable" in capsys.readouterr().err and not out.exists()
