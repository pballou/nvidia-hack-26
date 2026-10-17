"""End to end with the REAL mock agent: all 12 sandbox tasks through POST /api/run and the SSE stream.

The mock drives headless Chromium from pixels only: mouse clicks and wheel and keys at the coordinates in
console/mock_scripts.json (measured by console/tools/calibrate_mock.py). It never reads the DOM or the sandbox state,
and the console checks the verdict against /api/state only after the run. So a passing verdict here shows the task is
solvable by pixel clicks alone, and that the stream carries everything the console UI renders: tool calls in plain
words, grounding points with overlay images, screenshots, measured act timings, the done summary and the verdict.

The two browser-driving tests are marked `e2e` and run in the default suite (about 45 s for all 12 tasks plus the
free-text run). Skip them with -m "not e2e"; the fast script-integrity checks below still run.
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
OPTIONS = {"grounder": "base", "thinking": "retry", "lightning": "local"}


@pytest.fixture(scope="module")
def mock_console(sandbox_url, tmp_path_factory):
    """One console (--agent mock, no step delay, headless) for the whole module: (base_url, runs_dir)."""
    runs = tmp_path_factory.mktemp("e2e-runs")
    app = create_app(ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, step_delay_s=0.0,
                                     headed=False, static_dir=runs / "no-static", stop_timeout_s=10.0))
    srv = ServerThread(app)
    yield srv.start(), runs
    srv.stop()


def run_via_console(base: str, request: str, task_id: str | None = None) -> tuple[dict, list[dict]]:
    r = httpx.post(f"{base}/api/run", json={"request": request, "task_id": task_id, "options": OPTIONS})
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
    assert start["observation"]["payload"]["observation"]["screen_summary"] == script["start_screen"]
    assert_png(base, start["observation"]["screenshot_url"])

    for n, want in enumerate(script["steps"], 1):
        ev = by_step[n]
        plan, call = ev["plan"], ev["tool_call"]["payload"]
        assert plan["payload"]["reasoning"] == want["reasoning"] and "timings_ms" not in plan  # nothing simulated
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
        assert action["timings_ms"]["act"] > 0
        obs = ev["observation"]
        assert obs["payload"]["observation"]["screen_summary"] == want["screen_after"]
        assert obs["payload"]["observation"]["screen_changed"] is True and "timings_ms" not in obs
        assert obs["screenshot_url"] == f"/runs/{result['run_id']}/step_{n:02d}.png"
        assert_png(base, obs["screenshot_url"])
        if needs_grounding(want):
            g = ev["grounding"]
            assert g["payload"]["target"] == want["arguments"]["target_description"]
            assert (g["payload"]["x"], g["payload"]["y"]) == (want["x"], want["y"])
            assert g["payload"]["point_1000"] == [round(want["x"] / W * 1000), round(want["y"] / H * 1000)]
            assert "timings_ms" not in g and Path(g["overlay_path"]).is_file()
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
    assert [s["kind"] for s in steps] == ["start", "observation", "plan", "tool_call", "action", "done"]
    assert all(s["payload"]["mock"] is True for s in steps)
    plan = steps[2]["payload"]
    assert plan["content"] == mock_agent.FALLBACK_PLAN and plan["tool_calls"][0]["name"] == "done"
    assert steps[3]["payload"]["description"].startswith("Finishing: ")
    assert steps[-1]["payload"]["success_claimed"] is False and steps[-1]["payload"]["reason"] == "done"
    obs = steps[1]
    assert obs["payload"]["observation"]["url"].startswith(run["start_url"].split("?")[0])
    assert "timings_ms" not in obs
    assert_png(base, obs["screenshot_url"])

    info = httpx.get(f"{base}/api/runs/{run['run_id']}").json()
    assert info["status"] == "done" and info["task_id"] is None and info["verdict"] is None
    lines = (runs / result["run_id"] / "events.jsonl").read_text().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == [s["kind"] for s in steps]


# --- what makes the run above meaningful: scripts and mock are pixel-only -----------------------


def test_scripts_cover_every_task_and_hold_no_selectors():
    assert list(SCRIPTS) == [t["id"] for t in TASKS]
    assert SCRIPTS_FILE["viewport"] == mock_agent.VIEWPORT
    for task in TASKS:
        script = SCRIPTS[task["id"]]
        assert script["request"] == task["request"] and set(script) == {"request", "start_screen", "steps"}
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
