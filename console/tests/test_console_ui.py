"""The console page in a real browser, driven by keyboard only, against a live sandbox and the real mock agent.

Headless Chromium loads console/static from a live console (--agent mock, no step delay) and uses only the
keyboard: Tab to the task picker, type-ahead to pick a task, Enter to run, Escape to stop, arrow keys through the
step thumbnails. Checks: the step log reads in plain words ("Clicking: ..."), thumbnails replay steps (arrows work
straight from the banner too), the final banner carries the sandbox verdict with its exact label and takes focus,
typed task text runs as that task, free text gets no sandbox check, focus never drops to <body> when the form locks,
and the option toggles show the server defaults with clear labels. A replayed T2 event stream checks the step
headers for replies without an action, sub-second timings in ms, accessible names and one announcement per step.
A route-fulfilled /api/config gives the page a real agent's endpoints: the header says where the VLM and Lightning
run, an unreachable VLM gets a warning chip, and a grounder model the VLM doesn't serve is disabled (keyboard
included). Fake agents replay agent/progress.py's stuck stop (the banner leads with the agent's question, the
verdict below it, a warning line in the log) and a 25-step run (max_steps), which must fit at 1280x800 with nothing
cropped. Every test also checks that the page logs no console errors and makes no request that leaves 127.0.0.1.

DOM selectors are fine here (tests only). The agent under test never sees the DOM: the mock clicks pixels.
Marked `e2e` like test_mock_e2e.py (a browser plus real mock runs, about 20 s); skip with -m "not e2e".
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from PIL import Image

from agent.progress import ProgressTracker
from console.app import CONSOLE_DIR, ConsoleSettings, create_app
from console.runner import VERDICT_LABEL
from console.tests.conftest import ServerThread

pytestmark = pytest.mark.e2e

STATIC = CONSOLE_DIR / "static"
SCRIPTS = json.loads((CONSOLE_DIR / "mock_scripts.json").read_text())["tasks"]
TASK = "high-contrast"
TASK_REQUEST = "Turn on high contrast."
VIEWPORT = {"width": 1280, "height": 800}
LOCAL_HOST = "127.0.0.1"
FREE_TEXT_LABEL = "Free-text request: no sandbox check"
WARN_RGB = "rgb(122, 74, 0)"  # console.css --warn (light theme); --bad is rgb(179, 38, 30)

# /api/config "endpoints" of a real agent (console/agent_bridge.py endpoints_for), for route-fulfilled configs
MAC_VLM = {"base_url": "http://127.0.0.1:8011/v1", "label": "Mac (vllm-metal)", "summary_model": "grounder",
           "served": ["grounder"], "reachable": True}
H200_VLM = {"base_url": "http://127.0.0.1:8001/v1", "label": "H200 via tunnel", "summary_model": "base",
            "served": ["base", "grounder"], "reachable": True}
LOCAL_LIGHTNING = {"base_url": "http://127.0.0.1:8000/v1", "model": "lightning", "label": "local BF16 (:8000)"}
REAL_DEFAULTS = {"grounder": "grounder", "thinking": "retry", "lightning": "local"}


def _console(sandbox_url: str, runs: Path, **kw) -> ServerThread:
    settings = ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, headed=False,
                               static_dir=STATIC, stop_timeout_s=10.0, **kw)
    return ServerThread(create_app(settings))


@pytest.fixture(scope="module")
def ui_console(sandbox_url, tmp_path_factory):
    """Console URL: the real mock agent, no step delay (a run takes a few seconds)."""
    srv = _console(sandbox_url, tmp_path_factory.mktemp("ui-runs"), step_delay_s=0.0)
    yield srv.start()
    srv.stop()


@pytest.fixture(scope="module")
def slow_console(sandbox_url, tmp_path_factory):
    """Console URL with a 1 s step delay, so a run is still going when the test presses Escape."""
    srv = _console(sandbox_url, tmp_path_factory.mktemp("ui-slow-runs"), step_delay_s=1.0)
    yield srv.start()
    srv.stop()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@dataclass
class Watch:
    """What the page did: every request URL, off-host ones (aborted), console errors, page errors, /api/run bodies."""
    requests: list[str] = field(default_factory=list)
    offhost: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    run_bodies: list[dict] = field(default_factory=list)

    def assert_clean(self, base: str) -> None:
        assert not self.offhost, f"requests left {LOCAL_HOST}: {self.offhost}"
        assert not self.errors, self.errors
        origin = urlsplit(base).netloc
        assert self.requests and all(urlsplit(u).netloc == origin for u in self.requests), self.requests


@pytest.fixture
def page(browser):
    """(page, watch): a fresh 1280x800 context whose route aborts anything not on 127.0.0.1."""
    watch = Watch()
    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="en-US",
                              timezone_id="America/New_York")

    def route(r):
        url = r.request.url
        watch.requests.append(url)
        if r.request.method == "POST" and urlsplit(url).path == "/api/run":
            watch.run_bodies.append(json.loads(r.request.post_data or "{}"))
        if urlsplit(url).hostname != LOCAL_HOST:
            watch.offhost.append(url)
            r.abort("blockedbyclient")
        else:
            r.continue_()

    ctx.route("**/*", route)
    pg = ctx.new_page()
    pg.on("console", lambda m: watch.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: watch.errors.append(f"pageerror: {e}"))
    yield pg, watch
    ctx.close()


# --- helpers ----------------------------------------------------------------------------


def open_console(pg, base: str) -> None:
    pg.goto(base + "/")
    pg.wait_for_function("document.querySelector('#task-select').options.length > 12")  # Free text + 12 tasks
    pg.wait_for_function("document.querySelector('#agent-badge').dataset.agent === 'mock'")


def focused(pg) -> dict:
    return pg.evaluate("""() => { const e = document.activeElement;
        return {id: e.id, tag: e.tagName, cls: e.className, step: e.dataset ? e.dataset.step : undefined}; }""")


def tab_to(pg, pred, limit: int = 12, key: str = "Tab") -> list[dict]:
    """Press Tab until pred(focused element) holds; returns the elements passed on the way."""
    seen = []
    for _ in range(limit):
        pg.keyboard.press(key)
        f = focused(pg)
        seen.append(f)
        if pred(f):
            return seen
    raise AssertionError(f"never reached the target with {key}; focus went through {seen}")


def wait_banner(pg, timeout_s: float = 90) -> None:
    pg.wait_for_selector("#banner:not([hidden])", timeout=timeout_s * 1000)


def step_actions(pg) -> list[str]:
    return pg.eval_on_selector_all("#step-log .step .step-action", "els => els.map(e => e.textContent)")


def shown_image(pg) -> str:
    return urlsplit(pg.get_attribute("#screen-img", "src") or "").path


def record_announcements(pg) -> None:
    """Collect every text #sr-status (the page's one status region) shows, into window.__said."""
    pg.evaluate("""() => { window.__said = []; const el = document.querySelector('#sr-status');
        new MutationObserver(() => { if (el.textContent) window.__said.push(el.textContent); })
          .observe(el, {childList: true, characterData: true, subtree: true}); }""")


def button_names(pg) -> list[str]:
    """Accessible names of the page's buttons, as Chromium computes them (CSS generated content included)."""
    cdp = pg.context.new_cdp_session(pg)
    nodes = cdp.send("Accessibility.getFullAXTree")["nodes"]
    return [n["name"]["value"] for n in nodes
            if n.get("role", {}).get("value") == "button" and n.get("name", {}).get("value")]


def fake_config(pg, base: str, **changes) -> dict:
    """Serve the page's /api/config from the live console's, with `changes` merged in (e.g. a real agent's
    endpoints; the mock has none). Returns {"cfg": ..., "hits": n}: assign a new "cfg" to change what the next
    GET sees. Page routes take precedence over the context's (which still blocks off-host requests)."""
    state = {"cfg": {**httpx.get(f"{base}/api/config").json(), **changes}, "hits": 0}

    def handle(route):
        state["hits"] += 1
        route.fulfill(status=200, content_type="application/json", body=json.dumps(state["cfg"]))

    pg.route("**/api/config", handle)
    return state


def open_real_console(pg, base: str) -> None:
    pg.goto(base + "/")
    pg.wait_for_function("document.querySelector('#agent-badge').dataset.agent === 'real'")


def no_page_overflow(pg) -> None:
    """At 1280x800 the page itself never scrolls: the step log and the thumbnail strip scroll inside their boxes."""
    size = pg.evaluate("""() => { const d = document.documentElement;
        return [d.scrollWidth, d.scrollHeight, innerWidth, innerHeight]; }""")
    assert size[0] <= size[2] and size[1] <= size[3], size


# --- tests ---------------------------------------------------------------------------------


def test_keyboard_only_sandbox_task(page, ui_console):
    pg, watch = page
    open_console(pg, ui_console)

    # Tab: skip link, Logs link, high-contrast toggle, mic, request box, then the task picker
    path = tab_to(pg, lambda f: f["id"] == "task-select", limit=6)
    assert [f["id"] for f in path][-2:] == ["request", "task-select"], path
    pg.keyboard.type("Turn on")  # the native select's type-ahead picks "Turn on high contrast. (medium)"
    assert pg.input_value("#task-select") == TASK
    assert pg.input_value("#request") == TASK_REQUEST
    assert "Medium" in pg.inner_text("#task-meta")

    pg.keyboard.press("Shift+Tab")
    assert focused(pg)["id"] == "request"
    pg.keyboard.press("Enter")
    pg.wait_for_selector("#run-status[data-state='running'], #run-status[data-state='starting']")
    wait_banner(pg)

    # the banner: the sandbox's verdict with its exact label, and it has focus
    assert focused(pg)["id"] == "banner"
    assert pg.get_attribute("#banner", "data-tone") == "pass"
    assert pg.inner_text("#banner-icon") == "✓"
    assert pg.inner_text("#banner-title") == "Verified: the task is done"
    assert pg.text_content("#banner-label") == VERDICT_LABEL == "verified by sandbox state, not seen by the agent"
    assert pg.inner_text("#banner-expl").startswith("Sandbox check: ")
    assert "The agent claimed success." in pg.inner_text("#banner-claim")
    assert pg.inner_text("#run-status").startswith("Done")
    defaults = httpx.get(f"{ui_console}/api/config").json()["defaults"]  # the page runs with the server's defaults
    assert defaults == {"grounder": "grounder", "thinking": "retry", "lightning": "local"}
    assert watch.run_bodies == [{"request": TASK_REQUEST, "task_id": TASK, "options": defaults,  # picked: says so
                                 "task_match": {"method": "picked", "score": None, "matched_from": TASK_REQUEST}}]
    assert pg.inner_text("#banner-task") == f"Checked as task: {TASK} (picked from the list)"

    # the step log, in plain words: the start, one line per tool call, the done summary
    script = SCRIPTS[TASK]["steps"]
    want = [f"Clicking: {s['arguments']['target_description']}" for s in script if s["tool"] == "click"]
    actions = step_actions(pg)
    assert actions[0].startswith("Started at /")
    assert actions[1:-1] == want, actions
    assert actions[-1] == f"Done: {script[-1]['arguments']['summary']}"
    assert not [a for a in actions if "{" in a or "target_description" in a], actions  # no raw tool-call JSON
    assert pg.locator("#step-log .step.is-compact").count() == len(actions) - 1  # only the newest is expanded
    assert pg.locator("#log-empty").is_hidden() and pg.locator("#pending").is_hidden()

    # thumbnails: one per step with a screenshot (the start + each action; done has none)
    thumbs = pg.locator("#thumbs .thumb")
    assert thumbs.count() == len(script)
    assert [t.get_attribute("tabindex") for t in thumbs.all()].count("0") == 1  # one tab stop (roving tabindex)
    pg.wait_for_function("[...document.querySelectorAll('#thumbs .thumb img')]"
                         ".every(i => i.complete && i.naturalWidth === 1280)")
    assert shown_image(pg).endswith(f"step_{len(script) - 1:02d}.png")  # live: the newest screenshot
    assert pg.get_attribute("#step-log", "aria-live") is None  # steps are announced one sentence each instead

    # ← straight from the banner (where focus lands after a run) replays the previous step, as the hint says;
    # in the request box it only moves the caret
    pg.focus("#request")
    pg.keyboard.press("ArrowLeft")
    assert pg.get_attribute("#live-btn", "aria-pressed") == "true"
    pg.focus("#banner")
    pg.keyboard.press("ArrowLeft")
    f = focused(pg)
    assert "thumb" in f["cls"].split() and f["step"] == str(len(script) - 2), f
    assert "replay" in pg.inner_text("#screen-caption")
    assert pg.get_attribute("#live-btn", "aria-pressed") == "false"
    pg.focus("#banner")

    # from the banner, Tab reaches the thumbnail toolbar; arrows and Home/End replay steps
    tab_to(pg, lambda f: "thumb" in f["cls"].split())
    pg.keyboard.press("Home")
    f = focused(pg)
    assert "thumb" in f["cls"] and f["step"] == "0"
    assert shown_image(pg).endswith("step_00.png")
    assert pg.get_attribute("#thumbs .thumb[data-step='0']", "aria-current") == "true"
    assert pg.inner_text("#screen-caption").startswith("Start")
    assert pg.get_attribute("#live-btn", "aria-pressed") == "false"

    pg.keyboard.press("ArrowRight")
    assert focused(pg)["step"] == "1"
    assert shown_image(pg).endswith("step_01_overlay.png")  # a click step shows what the grounder saw first
    caption = pg.inner_text("#screen-caption")
    assert caption.startswith(f"Step 1 · {want[0]}") and "replay" in caption
    assert "before the click" in pg.get_attribute("#screen-img", "alt")
    step1 = "#step-log .step[data-step='1']"
    assert set(pg.get_attribute(step1, "class").split()) == {"step", "is-shown"}  # replayed: expanded, highlighted
    assert "Grounder (grounder, mock) chose" in pg.inner_text(step1)
    assert pg.inner_text(step1).count("“") >= 1  # its reasoning, quoted

    pg.keyboard.press("End")
    assert focused(pg)["step"] == str(len(script) - 1)
    pg.keyboard.press("ArrowLeft")
    assert focused(pg)["step"] == str(len(script) - 2)

    # back to live with the Live button, by keyboard
    tab_to(pg, lambda f: f["id"] == "live-btn", key="Shift+Tab", limit=6)
    pg.keyboard.press("Enter")
    assert pg.get_attribute("#live-btn", "aria-pressed") == "true"
    assert shown_image(pg).endswith(f"step_{len(script) - 1:02d}.png")

    # (mouse, too) clicking a compact step in the log replays it, like its thumbnail
    pg.click("#step-log .step[data-step='2'] .step-action")
    assert shown_image(pg).endswith("step_02_overlay.png")
    assert "is-compact" not in pg.get_attribute("#step-log .step[data-step='2']", "class")
    watch.assert_clean(ui_console)


def test_free_text_run_via_run_button(page, ui_console):
    pg, watch = page
    open_console(pg, ui_console)
    tab_to(pg, lambda f: f["id"] == "request", limit=5)  # skip link, Logs, high contrast, mic, request
    pg.keyboard.type("Please water my plants")
    assert pg.input_value("#task-select") == ""  # Free text
    tab_to(pg, lambda f: f["id"] == "run-btn")  # through the picker and the option groups
    pg.keyboard.press("Enter")
    wait_banner(pg)

    assert focused(pg)["id"] == "banner"
    assert pg.get_attribute("#banner", "data-tone") == "neutral"
    assert pg.text_content("#banner-label") == FREE_TEXT_LABEL
    assert pg.inner_text("#banner-title") == "Run finished"
    assert "The agent did not claim success." in pg.inner_text("#banner-claim")
    assert "mock agent" in pg.inner_text("#banner-summary")
    assert VERDICT_LABEL not in pg.inner_text("#banner")
    assert watch.run_bodies[0]["task_id"] is None and watch.run_bodies[0]["request"] == "Please water my plants"
    actions = step_actions(pg)
    assert actions[0].startswith("Started at /") and actions[-1].startswith("Done: This is the mock agent.")
    watch.assert_clean(ui_console)


def test_typed_task_text_runs_as_that_task(page, ui_console):
    """Typed text that is a sandbox task's text runs as that task (its start page and verdict), as voice does."""
    pg, watch = page
    open_console(pg, ui_console)

    def match(text):
        return pg.evaluate("(t) => { const m = window.AcmeConsole.matchTask(t); return m ? m.id : null; }", text)

    # case, punctuation and how an amount is written don't matter (speech-to-text spells "$50" out)...
    for text in ("Pay $50 toward my bill.", "pay 50 dollars toward my bill", "Pay fifty dollars toward my bill.",
                 "PAY $50.00 TOWARD MY BILL", "Pay fifty  dollars, toward my bill!"):
        assert match(text) == "pay-50", text
    # ...but other words or another amount do
    for text in ("Pay $60 toward my bill.", "Pay fifteen dollars toward my bill.", "Pay $500 toward my bill.",
                 "Pay $50 toward my bill twice.", "Pay toward my bill.", "Please water my plants"):
        assert match(text) is None, text
    tasks = httpx.get(f"{ui_console}/api/tasks").json()
    assert [match(t["request"]) for t in tasks] == [t["id"] for t in tasks]  # each task's text is its own key

    # mark-all-read starts at /messages: typed as free text it would start at / (and the mock's script fails)
    tab_to(pg, lambda f: f["id"] == "request", limit=5)
    pg.keyboard.type("mark all my messages as read")
    # typing alone doesn't switch the picker (the note says what it matches); leaving the box or running does
    assert pg.input_value("#task-select") == ""
    assert pg.inner_text("#request-note").startswith("This matches sandbox task mark-all-read")
    pg.keyboard.press("Enter")
    wait_banner(pg)
    assert watch.run_bodies[-1]["task_id"] == "mark-all-read"
    # an exact match runs the task's own spelling of the same words, and its task_match is left to the backend
    assert watch.run_bodies[-1]["request"] == pg.input_value("#request") == "Mark all my messages as read."
    assert "task_match" not in watch.run_bodies[-1]
    assert pg.input_value("#task-select") == "mark-all-read"
    assert step_actions(pg)[0] == "Started at /messages"
    assert pg.get_attribute("#banner", "data-tone") == "pass"
    assert pg.inner_text("#banner-task") == "Checked as task: mark-all-read (matched from your request)"
    assert pg.text_content("#banner-label") == VERDICT_LABEL

    # picking a task and then changing only case or punctuation keeps the task (no "runs as free text" note); so
    # does a close paraphrase (now matched from the request); a second request in the same words doesn't
    pg.select_option("#task-select", TASK)
    pg.fill("#request", "turn on high contrast")
    assert pg.input_value("#task-select") == TASK and pg.inner_text("#request-note") == ""
    pg.fill("#request", "please switch on high contrast now")
    assert pg.input_value("#task-select") == TASK
    assert pg.inner_text("#request-note").startswith("Will be checked as task high-contrast (matched from your request")
    pg.fill("#request", "turn on high contrast and reduce motion")
    assert pg.input_value("#task-select") == "" and "free text" in pg.inner_text("#request-note")
    watch.assert_clean(ui_console)


def test_focus_stays_visible_when_the_form_locks(page, slow_console):
    """Controls that get disabled when a run starts or stops don't drop focus to <body> (invisible, and Space there
    starts the mic): it moves to the request box."""
    pg, watch = page
    open_console(pg, slow_console)
    pg.focus("#task-select")
    pg.keyboard.type("Turn on")

    # Enter on an option radio submits the form; the option radios are then disabled
    pg.focus("#thinking-retry")
    pg.keyboard.press("ArrowRight")
    assert pg.is_checked("#thinking-on")
    pg.keyboard.press("Enter")
    pg.wait_for_selector("#run-status[data-state='running'], #run-status[data-state='starting']")
    assert watch.run_bodies[-1]["options"]["thinking"] == "on"
    assert focused(pg)["id"] == "request"

    # Stop by keyboard while the stop request is slow (the real agent can take seconds): Stop shows "Stopping…"
    pg.evaluate("""() => { const f = window.fetch;
        window.fetch = (u, o) => (String(u).endsWith('/stop') ? new Promise((r) => setTimeout(r, 1500)).then(() => f(u, o))
          : f(u, o)); }""")
    pg.focus("#stop-btn")
    pg.keyboard.press("Enter")
    assert pg.inner_text("#stop-btn") == "Stopping…" and not pg.is_enabled("#stop-btn")
    assert focused(pg)["id"] == "request"
    wait_banner(pg, 30)
    assert focused(pg)["id"] == "banner"

    # the next run starts while the banner has focus (a voice run does this); the banner hides, focus moves on
    pg.evaluate("window.AcmeConsole.run()")
    assert pg.locator("#banner").is_hidden()
    assert focused(pg)["id"] == "request"
    pg.keyboard.press("Escape")
    wait_banner(pg, 30)
    watch.assert_clean(slow_console)


def replay_agent(trace_root: Path, gap_s: float = 0.3):
    """(run_task, make_cfg) replaying T2's event shapes (agent/core.py): a reply cut off at the token limit, a reply
    without a tool call, a failed Lightning call, one click with T2's measured timings, then done."""

    async def run_task(request, *, start_url, cfg, on_event=None):
        rid = f"replay-{time.time_ns()}"
        d = trace_root / rid
        d.mkdir(parents=True)

        async def ev(step, kind, payload, **extra):
            await on_event({"run_id": rid, "step": step, "kind": kind, "ts": time.time(), "payload": payload, **extra})

        def png(name, color):
            Image.new("RGB", (64, 40), color).save(d / name)
            return str(d / name)

        def planner_error(message, **kw):
            return {"message": message, "where": "planner", "recoverable": True, **kw}

        await ev(0, "start", {"request": request, "start_url": start_url, "config": {}})
        await ev(0, "observation", {"text": "Step 0 | URL: /"}, screenshot_path=png("step_00.png", "white"),
                 timings_ms={"screenshot": 31.0, "summary": 1500.0})
        no_action = [
            ({"tool_calls": [], "thinking": True, "finish_reason": "length"},
             planner_error("reply was cut off at the token limit before a tool call; asked Lightning to call one",
                           finish_reason="length")),
            ({"tool_calls": [], "thinking": False, "finish_reason": "stop"},
             planner_error("reply had no tool call; asked Lightning to call one", finish_reason="stop")),
            ({"tool_calls": [], "thinking": False}, planner_error("APIConnectionError: Connection error.")),
        ]
        for step, (plan, error) in enumerate(no_action, start=1):
            await asyncio.sleep(gap_s)
            await ev(step, "plan", plan, timings_ms={"plan": 2100.0})
            await ev(step, "error", error)
        click = {"name": "click", "arguments": {"target_description": "the gear icon in the top bar"}}
        await asyncio.sleep(gap_s)
        await ev(4, "plan", {"tool_calls": [click], "thinking": False, "finish_reason": "tool_calls"},
                 timings_ms={"plan": 2100.0})
        await ev(4, "tool_call", {**click, "valid": True})
        await ev(4, "grounding", {"target": "the gear icon in the top bar", "x": 1110, "y": 32, "model": "base"},
                 overlay_path=png("step_04_overlay.png", "red"), timings_ms={"ground": 676.4})
        await ev(4, "action", {**click, "ok": True, "screen_changed": True, "done": False},
                 timings_ms={"act": 18.2, "settle": 507.0})
        await ev(4, "observation", {"text": "Step 4 | URL: /settings"}, screenshot_path=png("step_04.png", "blue"),
                 timings_ms={"screenshot": 25.0, "summary": 1900.0})
        done = {"name": "done", "arguments": {"summary": "Opened the settings.", "success": True}}
        await asyncio.sleep(gap_s)
        await ev(5, "plan", {"tool_calls": [done]}, timings_ms={"plan": 900.0})
        await ev(5, "tool_call", {**done, "valid": True})
        await ev(5, "action", {**done, "ok": True, "screen_changed": None, "done": True})
        await ev(5, "done", {"success_claimed": True, "summary": "Opened the settings.", "steps": 5, "reason": "done"})
        return {"run_id": rid, "success_claimed": True, "summary": "Opened the settings.", "steps": 5,
                "trace_dir": str(d)}

    return run_task, lambda options: {"options": options}


def test_replayed_real_agent_run(page, console, tmp_path):
    """T2's real event shapes: step headers for replies without an action, timings, accessible names, announcements."""
    pg, watch = page
    base, _ = console(replay_agent(tmp_path / "trace"), agent="mock", static_dir=STATIC)
    open_console(pg, base)
    record_announcements(pg)
    pg.fill("#request", "Open the settings")
    pg.press("#request", "Enter")
    wait_banner(pg, 30)

    assert step_actions(pg) == [
        "Started at /",
        "No action: Lightning’s reply was cut off before it chose one",  # not "the Lightning call did not succeed"
        "No action: Lightning replied without a tool call",
        "Planning failed: the Lightning call did not succeed",
        "Clicking: the gear icon in the top bar",
        "Done: Opened the settings.",
    ]
    # sub-second timings in ms: a real click acts in ~20 ms, which used to read "act 0.0 s"
    timing = pg.text_content("#step-log .step[data-step='4'] .step-meta > span").replace("\u00a0", " ")
    assert timing == "plan 2.1 s · ground 676 ms · act 18 ms · summary 1.9 s ·"
    assert pg.text_content("#step-log .step[data-step='0'] .step-meta > span").replace("\u00a0", " ") == "summary 1.5 s ·"

    # "Screen text": the accessible name starts with the visible words (voice control: "click Screen text")
    btn = "#step-log .step[data-step='4'] .step-meta button"
    assert pg.text_content(btn) == "Screen text"
    assert pg.get_attribute(btn, "aria-label") == "Screen text: what Lightning read after step 4"
    pg.click("#step-log .step[data-step='4'] .step-action")  # open step 4 in the viewer, then its screen text
    pg.click(btn)
    assert pg.text_content(btn) == "Hide screen text"
    assert pg.get_attribute(btn, "aria-label") == "Hide screen text: what Lightning read after step 4"
    assert pg.get_attribute("#step-log .step[data-step='0'] .step-meta button", "aria-label") \
        == "Screen text: what Lightning read at the start"

    # one sentence per step in the status region: no timings, no pixel coordinates
    assert pg.evaluate("window.__said") == [
        "Run started: Open the settings",
        "Step 1: No action: Lightning’s reply was cut off before it chose one. Asking again.",
        "Step 2: No action: Lightning replied without a tool call. Asking again.",
        "Step 3: Planning failed: the Lightning call did not succeed. Asking again.",
        "Step 4: Clicking: the gear icon in the top bar. Screen changed.",
        "Step 5: Done: Opened the settings.",
    ]

    # toggle buttons: their glyphs (✓, ●, ▶) stay out of the accessible names
    pg.click("#hc-toggle")
    assert pg.get_attribute("#hc-toggle", "aria-pressed") == "true"
    names = button_names(pg)
    assert "High contrast" in names and "Live" in names, names
    assert not [n for n in names if re.match(r"[✓●▶]", n)], names
    mic = re.search(r'<button id="mic-btn"[^>]*>', (STATIC / "index.html").read_text()).group(0)
    assert "aria-pressed" not in mic  # its label says what pressing it does ("Speak" / "Stop listening")
    watch.assert_clean(base)


def test_escape_stops_a_running_task(page, slow_console):
    pg, watch = page
    open_console(pg, slow_console)
    pg.focus("#task-select")
    pg.keyboard.type("Turn on")
    pg.keyboard.press("Shift+Tab")
    pg.keyboard.press("Enter")
    pg.wait_for_function("[...document.querySelectorAll('#step-log .step-action')]"
                         ".some(e => e.textContent.startsWith('Clicking: '))", timeout=30000)
    assert pg.is_enabled("#stop-btn") and not pg.is_enabled("#run-btn")
    pg.keyboard.press("Escape")
    wait_banner(pg, 30)

    assert focused(pg)["id"] == "banner"
    assert pg.get_attribute("#banner", "data-tone") == "stopped"
    assert pg.inner_text("#banner-title").startswith("Stopped")
    assert pg.inner_text("#banner-icon") == "■"
    assert pg.text_content("#banner-label") == VERDICT_LABEL  # a sandbox task still gets its verdict after a stop
    assert pg.inner_text("#banner-expl").startswith("✗ Sandbox check: 1 of ")
    # the failed conditions are one keyboard press away, not dumped into the banner
    assert pg.locator("#banner-details").is_hidden()
    pg.keyboard.press("Tab")
    assert pg.evaluate("document.activeElement.getAttribute('aria-controls')") == "banner-details"
    pg.keyboard.press("Enter")
    assert pg.locator("#banner-details").is_visible()
    assert "settings.accessibility.high_contrast" in pg.inner_text("#banner-details")
    assert pg.evaluate("document.activeElement.getAttribute('aria-expanded')") == "true"
    assert "You stopped the run." in pg.inner_text("#banner-claim")
    assert pg.get_attribute("#run-status", "data-state") == "stopped"
    assert pg.inner_text("#run-status").startswith("Stopped")
    assert pg.is_enabled("#run-btn") and not pg.is_enabled("#stop-btn")
    assert len(step_actions(pg)) < len(SCRIPTS[TASK]["steps"]) + 1  # it really stopped early

    run_id = pg.evaluate("sessionStorage.getItem('gui-assist.run')")
    info = httpx.get(f"{slow_console}/api/runs/{run_id}").json()
    assert info["status"] == "stopped" and info["verdict"]["passed"] is False
    watch.assert_clean(slow_console)


@pytest.mark.parametrize("default_lightning,default_grounder", [("local", "grounder"), ("hosted", "base")])
def test_options_show_server_defaults_with_clear_labels(page, console, default_lightning, default_grounder):
    pg, watch = page
    base, _ = console(None, agent="mock", static_dir=STATIC, default_lightning=default_lightning,
                      default_grounder=default_grounder)
    defaults = httpx.get(f"{base}/api/config").json()["defaults"]
    assert (defaults["lightning"], defaults["grounder"]) == (default_lightning, default_grounder)
    open_console(pg, base)

    labels = pg.eval_on_selector_all(".options .seg label", "els => els.map(l => [l.htmlFor, l.textContent])")
    assert labels == [["grounder-base", "base (adapter off)"], ["grounder-grounder", "grounder (LoRA)"],
                      ["thinking-off", "off"], ["thinking-retry", "retry"], ["thinking-on", "on"],
                      ["lightning-local", "Local (BF16 on :8000)"], ["lightning-hosted", "Hosted API"]]
    checked = pg.eval_on_selector_all(".options input:checked", "els => els.map(i => i.id)")
    assert checked == [f"grounder-{default_grounder}", "thinking-retry", f"lightning-{default_lightning}"]
    assert not pg.eval_on_selector_all(".options input:disabled", "els => els.map(i => i.id)")  # mock: no models
    summary = "local Lightning" if default_lightning == "local" else "hosted Lightning"
    grounder = "LoRA grounder" if default_grounder == "grounder" else "base"
    assert pg.text_content("#options-summary") == f"{grounder} · thinking retry · {summary}"
    # the mock runs no models: its badge says so, and the header lists no endpoints
    assert pg.inner_text("#agent-badge") == "Agent: mock (scripted, no models)"
    assert pg.locator("#endpoints").is_hidden()

    # keyboard: the radio group is one tab stop, arrows change the choice, and the summary follows
    tab_to(pg, lambda f: f["id"] == f"lightning-{default_lightning}", limit=10)
    pg.keyboard.press("ArrowRight" if default_lightning == "local" else "ArrowLeft")
    other = "hosted" if default_lightning == "local" else "local"
    assert pg.is_checked(f"#lightning-{other}")
    assert pg.text_content("#options-summary").endswith(f"{other} Lightning")
    watch.assert_clean(base)


# --- a real agent's endpoints (route-fulfilled /api/config) -----------------------------------------------------


def test_endpoints_header_and_unserved_grounder(page, ui_console):
    """The header says where the VLM and Lightning run; a grounder model the VLM doesn't serve can't be picked."""
    pg, watch = page
    cfg = fake_config(pg, ui_console, agent="real", defaults=REAL_DEFAULTS,
                      endpoints={"vlm": MAC_VLM, "lightning": LOCAL_LIGHTNING})
    open_real_console(pg, ui_console)
    assert pg.inner_text("#agent-badge") == "Agent: real"
    assert pg.inner_text("#vlm-endpoint-text") == "VLM: Mac (vllm-metal) · 127.0.0.1:8011 · serves grounder"
    assert pg.inner_text("#lightning-endpoint") == "Lightning: local BF16 (:8000)"
    assert pg.locator("#vlm-warning").is_hidden()
    assert pg.evaluate("document.querySelector('.app-header').getBoundingClientRect().height") < 80  # one row
    no_page_overflow(pg)

    # base isn't served at :8011: disabled, and its label says why; the served default stays checked
    assert pg.is_disabled("#grounder-base") and pg.is_enabled("#grounder-grounder")
    assert pg.text_content("label[for='grounder-base']") == "base (adapter off) (not served at 127.0.0.1:8011)"
    assert pg.text_content("label[for='grounder-grounder']") == "grounder (LoRA)"
    assert pg.is_checked("#grounder-grounder") and pg.locator("#grounder-note").is_hidden()
    assert pg.text_content("#options-summary") == "LoRA grounder · thinking retry · local Lightning"
    # the two segments still sit side by side (the note wraps inside its own segment)
    tops = pg.eval_on_selector_all("#opt-grounder .seg label", "els => els.map(l => l.getBoundingClientRect().top)")
    assert tops[0] == tops[1], tops

    # keyboard: the group's one tab stop is the served model, and the arrow keys never land on the disabled one
    pg.focus("#task-select")
    tab_to(pg, lambda f: f["id"] == "grounder-grounder", limit=3)  # task picker, Options summary, grounder
    for key in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"):
        pg.keyboard.press(key)
        assert focused(pg)["id"] == "grounder-grounder" and pg.is_checked("#grounder-grounder"), key
    pg.keyboard.press("Tab")
    assert focused(pg)["id"] == "thinking-retry"

    # the Lightning line follows the toggle (the endpoints describe the default; hosted is the other preset)
    pg.click("label[for='lightning-hosted']")
    assert pg.inner_text("#lightning-endpoint") == "Lightning: hosted API"
    pg.click("label[for='lightning-local']")
    assert pg.inner_text("#lightning-endpoint") == "Lightning: local BF16 (:8000)"

    # the VLM doesn't answer: a warning chip, and nothing is disabled (what it serves is unknown)
    cfg["cfg"] = {**cfg["cfg"], "endpoints": {"vlm": {**H200_VLM, "served": None, "reachable": False},
                                              "lightning": LOCAL_LIGHTNING}}
    open_real_console(pg, ui_console)
    assert pg.inner_text("#vlm-endpoint-text") == "VLM: H200 via tunnel · 127.0.0.1:8001"
    assert pg.is_visible("#vlm-warning") and pg.inner_text("#vlm-warning") == "⚠ not reachable"
    assert pg.eval_on_selector_all("#opt-grounder input:disabled", "els => els.length") == 0
    assert pg.evaluate("document.querySelector('.app-header').getBoundingClientRect().height") < 80

    # it comes up: the page looks again (on a timer, or here when the tab becomes visible) and says so once
    record_announcements(pg)
    cfg["cfg"] = {**cfg["cfg"], "endpoints": {"vlm": H200_VLM, "lightning": LOCAL_LIGHTNING}}
    pg.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    pg.wait_for_selector("#vlm-warning", state="hidden")
    assert pg.inner_text("#vlm-endpoint-text") == "VLM: H200 via tunnel · 127.0.0.1:8001 · serves base, grounder"
    pg.wait_for_function("window.__said.length > 0")
    assert pg.evaluate("window.__said") == ["The VLM is reachable again."]

    # the VLM stopped serving base since the page looked: the backend refuses the run (400), the page shows its
    # message, looks again, and moves the choice to the model that is served
    pg.click("label[for='grounder-base']")
    msg = "grounder model 'base' is not served at http://127.0.0.1:8001/v1 (serves: grounder)"
    pg.route("**/api/run", lambda r: r.fulfill(status=400, content_type="application/json",
                                               body=json.dumps({"error": msg})))
    cfg["cfg"] = {**cfg["cfg"], "endpoints": {"vlm": {**H200_VLM, "served": ["grounder"]}, "lightning": LOCAL_LIGHTNING}}
    pg.fill("#request", "Open the settings")
    pg.press("#request", "Enter")
    pg.wait_for_function("document.querySelector('#grounder-base').disabled")
    assert pg.inner_text("#form-error") == f"Can’t start: {msg}"
    assert pg.is_checked("#grounder-grounder")
    assert pg.text_content("label[for='grounder-base']").endswith("(not served at 127.0.0.1:8001)")

    # serves neither model: both disabled, and a note says runs can't start
    cfg["cfg"] = {**cfg["cfg"], "endpoints": {"vlm": {**H200_VLM, "served": []}, "lightning": LOCAL_LIGHTNING}}
    open_real_console(pg, ui_console)
    assert pg.inner_text("#vlm-endpoint-text") == "VLM: H200 via tunnel · 127.0.0.1:8001 · serves no models"
    assert pg.eval_on_selector_all("#opt-grounder input:disabled", "els => els.length") == 2
    assert pg.inner_text("#grounder-note") == ("The VLM at 127.0.0.1:8001 serves neither model, "
                                               "so runs can’t start until it does.")

    # the screen-summary model isn't served there (--summary-model base on a grounder-only VLM): a warning chip
    cfg["cfg"] = {**cfg["cfg"], "endpoints": {"vlm": {**MAC_VLM, "summary_model": "base"}, "lightning": LOCAL_LIGHTNING}}
    open_real_console(pg, ui_console)
    assert pg.inner_text("#vlm-warning") == "⚠ summary model base not served"
    assert pg.locator("#grounder-note").is_hidden() and pg.is_checked("#grounder-grounder")
    assert pg.evaluate("document.querySelector('.app-header').getBoundingClientRect().height") < 80
    assert cfg["hits"] == 6  # 4 page loads, the visibility change, the 400: no other polling in a short test
    # Chromium logs every 4xx as a console error; this one 400 is the test's own
    expected = [e for e in watch.errors if "status of 400" in e]
    assert len(expected) == 1, watch.errors
    watch.errors.remove(expected[0])
    watch.assert_clean(ui_console)


# --- 'stuck' (agent/progress.py) ------------------------------------------------------------------------------------


def png_file(d: Path, name: str, color: str) -> str:
    Image.new("RGB", (160, 100), color).save(d / name)
    return str(d / name)


def stuck_agent(trace_root: Path, seen: dict, target: str = "the save button", gap_s: float = 0.15):
    """(run_task, make_cfg) with T2's stuck stop: the same click changes nothing, agent/progress.py's real
    ProgressTracker warns (an action with stuck.verdict 'warn'), then stops it (verdict 'stuck', done reason
    'stuck' with the tracker's own summary: a question to the user). `seen` gets the summary and the reason."""

    async def run_task(request, *, start_url, cfg, on_event=None):
        rid = f"stuck-{time.time_ns()}"
        d = trace_root / rid
        d.mkdir(parents=True)

        async def ev(step, kind, payload, **extra):
            await on_event({"run_id": rid, "step": step, "kind": kind, "ts": time.time(), "payload": payload, **extra})

        await ev(0, "start", {"request": request, "start_url": start_url, "config": {}})
        await ev(0, "observation", {"text": "Step 0 | URL: /settings"}, screenshot_path=png_file(d, "step_00.png", "white"))
        tracker = ProgressTracker()
        click = {"name": "click", "arguments": {"target_description": target}}
        step, verdict = 0, None
        while verdict != "stuck":
            step += 1
            await asyncio.sleep(gap_s)
            await ev(step, "plan", {"tool_calls": [click], "thinking": verdict == "warn", "reasoning": "Save it."})
            await ev(step, "tool_call", {**click, "valid": True, "description": f"Clicking: {target}"})
            await ev(step, "grounding", {"target": target, "x": 640, "y": 400, "model": "grounder"},
                     overlay_path=png_file(d, f"step_{step:02d}_overlay.png", "red"))
            verdict = tracker.record("click", click["arguments"], ok=True, screen_changed=False, click_xy=(640, 400))
            action = {**click, "ok": True, "screen_changed": False, "done": False, "click": {"x": 640, "y": 400}}
            if verdict:
                action["stuck"] = {"verdict": verdict, "reason": tracker.reason}
            await ev(step, "action", action, timings_ms={"act": 20.0})
            await ev(step, "observation", {"text": f"Step {step} | URL: /settings"},
                     screenshot_path=png_file(d, f"step_{step:02d}.png", "white"))
        seen.update(summary=tracker.summary(), reason=tracker.reason, steps=step)
        stats = {"n_clicks": step, "n_no_change": step, "n_stuck_warnings": tracker.n_warnings, "n_stuck_stops": 1}
        await ev(step, "done", {"success_claimed": False, "summary": seen["summary"], "steps": step, "reason": "stuck",
                                "stats": stats})
        return {"run_id": rid, "success_claimed": False, "summary": seen["summary"], "steps": step, "trace_dir": str(d)}

    return run_task, lambda options: {"options": options}


def test_stuck_run_asks_the_user(page, console, tmp_path):
    """done.reason 'stuck': the banner leads with the agent's question (amber, not a red failure, clearly not
    done), the sandbox verdict still shows below it, and the step log marks the warning and the stop."""
    pg, watch = page
    seen: dict = {}
    base, _ = console(stuck_agent(tmp_path / "trace", seen), agent="mock", static_dir=STATIC)
    open_console(pg, base)
    record_announcements(pg)
    pg.select_option("#task-select", TASK)
    pg.press("#request", "Enter")
    wait_banner(pg, 30)

    summary, reason = seen["summary"], seen["reason"]
    assert reason == 'click "the save button" made no progress 3 times'  # agent/progress.py's own words
    cut = summary.index("What would you like")
    question, context = summary[cut:], summary[:cut].strip()
    assert summary.startswith("I'm stuck: ") and question.endswith("something else.")

    # the banner: the question is the headline, and it is the banner's accessible name with the title
    assert focused(pg)["id"] == "banner"
    assert pg.get_attribute("#banner", "data-tone") == "stuck"
    assert pg.inner_text("#banner-icon") == "?"
    assert pg.inner_text("#banner-title") == "The agent is stuck and asks:"
    assert pg.inner_text("#banner-question") == f"“{question}”"
    assert pg.get_attribute("#banner", "aria-labelledby") == "banner-title banner-question"
    assert pg.evaluate("getComputedStyle(document.querySelector('#banner')).borderTopColor") == WARN_RGB  # not red
    # not done: the sandbox check (it still runs after a stuck stop) sits below the question
    assert pg.text_content("#banner-label") == VERDICT_LABEL
    assert pg.inner_text("#banner-expl").startswith("✗ Not done. Sandbox check: 1 of ")
    assert pg.inner_text("#banner-summary") == f"Agent’s summary: “{context}”"  # what it tried, without the question
    claim = pg.inner_text("#banner-claim")
    assert "The agent did not claim success." in claim and "run it again" in claim
    assert "1 stuck warning" in pg.inner_text("#banner-extra")
    assert pg.get_attribute("#run-status", "data-state") == "stuck"
    assert pg.inner_text("#run-status").startswith("Stuck")
    y = pg.evaluate("""() => ['#banner', '#screen', '#thumbs'].map(s => document.querySelector(s)
        .getBoundingClientRect()).map(r => [r.top, r.bottom])""")
    assert y[0][1] <= y[1][0] and y[1][1] <= y[2][0], y  # banner, screen, thumbnails: stacked, none overlapping
    no_page_overflow(pg)

    # the step log: a warning line on the step that triggered it (still shown when that step is compact)...
    n = seen["steps"]
    assert n == 6
    warn_step = "#step-log .step[data-step='3']"
    assert pg.inner_text(f"{warn_step} .step-warn") == f"Stuck warning: {reason}"
    assert pg.locator(f"{warn_step} .step-warn").is_visible() and "is-compact" in pg.get_attribute(warn_step, "class")
    assert [pg.locator(f"#step-log .step[data-step='{i}'] .step-warn").count() for i in (1, 2, 4, 5)] == [0] * 4
    # ...and "Stopped: stuck" plus the question on the last one, amber (warn), not red (bad)
    last = f"#step-log .step[data-step='{n}']"
    lines = pg.eval_on_selector_all(f"{last} .step-lines > p", "els => els.map(e => [e.className, e.textContent])")
    assert ["step-line step-warn", f"Stopped: stuck ({reason})"] in lines, lines
    assert ["step-line step-ask", f"The agent asks: “{question}”"] in lines, lines
    assert pg.get_attribute(last, "data-state") == "warn"

    # one announcement per step; the warning and the stop are each said once, with their step
    said = pg.evaluate("window.__said")
    assert said[0] == f"Run started: {TASK_REQUEST}"
    assert said[1:] == [f"Step {i}: Clicking: the save button. No change on screen."
                        + {3: f" Stuck warning: {reason}.", 6: f" Stopped: stuck ({reason})."}.get(i, "")
                        for i in range(1, n + 1)]
    watch.assert_clean(base)


# --- a 25-step run (configs/agent.yaml max_steps) -------------------------------------------------------------------


def long_agent(trace_root: Path, n: int = 25):
    """(run_task, make_cfg): n screen-changing clicks, each with an overlay and a screenshot, then done(max_steps)."""

    async def run_task(request, *, start_url, cfg, on_event=None):
        rid = f"long-{time.time_ns()}"
        d = trace_root / rid
        d.mkdir(parents=True)

        async def ev(step, kind, payload, **extra):
            await on_event({"run_id": rid, "step": step, "kind": kind, "ts": time.time(), "payload": payload, **extra})

        await ev(0, "start", {"request": request, "start_url": start_url, "config": {}})
        await ev(0, "observation", {"text": "Step 0 | URL: /"}, screenshot_path=png_file(d, "step_00.png", "white"))
        colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
        for step in range(1, n + 1):
            click = {"name": "click", "arguments": {"target_description": f"the row {step} details link"}}
            await ev(step, "plan", {"tool_calls": [click], "reasoning": f"Row {step} is next. " * 6})
            await ev(step, "tool_call", {**click, "valid": True})
            await ev(step, "grounding", {"target": click["arguments"]["target_description"], "x": 600, "y": 30 * step,
                                         "model": "grounder"},
                     overlay_path=png_file(d, f"step_{step:02d}_overlay.png", "red"), timings_ms={"ground": 400.0})
            await ev(step, "action", {**click, "ok": True, "screen_changed": True, "done": False},
                     timings_ms={"act": 20.0})
            await ev(step, "observation", {"text": f"Step {step} | URL: /usage"},
                     screenshot_path=png_file(d, f"step_{step:02d}.png", colors[step % len(colors)]))
        summary = f"Stopped: reached the {n}-step limit before finishing."
        await ev(n, "done", {"success_claimed": False, "summary": summary, "steps": n, "reason": "max_steps",
                             "stats": {"n_clicks": n}})
        return {"run_id": rid, "success_claimed": False, "summary": summary, "steps": n, "trace_dir": str(d)}

    return run_task, lambda options: {"options": options}


THUMB_GEOMETRY = """() => {
  const box = (e) => e.getBoundingClientRect();
  const strip = document.querySelector('#thumbs');
  const s = box(strip);
  const inner = {top: s.top + strip.clientTop, bottom: s.top + strip.clientTop + strip.clientHeight,
                 left: s.left + strip.clientLeft, right: s.left + strip.clientLeft + strip.clientWidth};
  const thumbs = [...strip.querySelectorAll('.thumb')].map((t) => {
    const r = box(t);
    const num = t.querySelector('.thumb-num');
    const img = t.querySelector('img');
    return {step: t.dataset.step, top: r.top, bottom: r.bottom, left: r.left, right: r.right,
            current: t.getAttribute('aria-current') === 'true', num: num.textContent,
            numFits: box(num).right <= r.right && box(num).bottom <= r.bottom,
            filled: img.offsetHeight >= t.clientHeight - 1};  // no letterbox band under the image
  });
  return {strip: {...inner, scrollWidth: strip.scrollWidth, clientWidth: strip.clientWidth},
          panel: box(document.querySelector('.panel-right')).toJSON(), thumbs};
}"""


def visible_in_strip(geo: dict, step: str, ring: float = 6) -> bool:
    """The thumb for `step` is fully inside the strip's visible box, with room for its focus ring."""
    t = next(t for t in geo["thumbs"] if t["step"] == step)
    s = geo["strip"]
    return t["left"] >= s["left"] + ring - 1 and t["right"] <= s["right"] - ring + 1


def test_25_step_run_fits(page, console, tmp_path):
    """max_steps is 25: start + 25 thumbnails scroll sideways inside their strip, never the page; none is cropped
    (rings included); the log scrolls; replaying any step keeps its thumbnail in view."""
    pg, watch = page
    base, _ = console(long_agent(tmp_path / "trace"), agent="mock", static_dir=STATIC)
    open_console(pg, base)
    pg.fill("#request", "Look at every row")
    pg.press("#request", "Enter")
    wait_banner(pg, 60)

    assert len(step_actions(pg)) == 26 and step_actions(pg)[-1] == "Clicking: the row 25 details link"
    assert pg.inner_text("#banner-summary") == "Agent’s summary: “Stopped: reached the 25-step limit before finishing.”"
    assert "25 steps" in pg.inner_text("#banner-claim")
    pg.wait_for_function("[...document.querySelectorAll('#thumbs .thumb img')].every(i => i.complete && i.naturalWidth)")
    no_page_overflow(pg)

    geo = pg.evaluate(THUMB_GEOMETRY)
    s, panel, thumbs = geo["strip"], geo["panel"], geo["thumbs"]
    assert [t["step"] for t in thumbs] == [str(i) for i in range(26)]
    assert [t["num"] for t in thumbs] == ["Start", *[str(i) for i in range(1, 26)]] and all(t["numFits"] for t in thumbs)
    assert s["scrollWidth"] > s["clientWidth"]  # it scrolls sideways...
    assert panel["left"] <= s["left"] and s["right"] <= panel["right"] and s["bottom"] <= panel["bottom"]  # ...inside
    for t in thumbs:  # none is cut at the top or bottom (nor by a scrollbar), the focus ring fits (high contrast:
        # 4 px + 2 px offset), and none is letterboxed
        assert t["top"] >= s["top"] + 6 and t["bottom"] <= s["bottom"] - 6 and t["filled"], (t, s)
    assert visible_in_strip(geo, "25")  # live: the newest is in view

    # the log scrolls inside its box, and following live left the newest step in view
    log = pg.evaluate("""() => { const l = document.querySelector('#log-scroll').getBoundingClientRect();
        const n = document.querySelector('#step-log .step[data-step="25"]').getBoundingClientRect();
        const el = document.querySelector('#log-scroll');
        return {scrolls: el.scrollHeight > el.clientHeight, top: n.top >= l.top - 1, bottom: n.top < l.bottom}; }""")
    assert log == {"scrolls": True, "top": True, "bottom": True}, log
    assert pg.evaluate("(() => { const e = document.querySelector('#step-log .step[data-step=\"25\"] .step-num');"
                       " return e.scrollWidth <= e.clientWidth; })()")

    # keyboard replay from the banner: Home, End and arrows keep the focused thumbnail in view
    pg.focus("#banner")
    pg.keyboard.press("Home")
    assert focused(pg)["step"] == "0" and visible_in_strip(pg.evaluate(THUMB_GEOMETRY), "0")
    pg.keyboard.press("End")
    assert focused(pg)["step"] == "25" and visible_in_strip(pg.evaluate(THUMB_GEOMETRY), "25")
    for _ in range(3):
        pg.keyboard.press("ArrowLeft")
    assert focused(pg)["step"] == "22" and visible_in_strip(pg.evaluate(THUMB_GEOMETRY), "22")

    # replaying from the log (mouse) scrolls the strip to that step's thumbnail too
    pg.evaluate("document.querySelector('#log-scroll').scrollTop = 0")
    pg.click("#step-log .step[data-step='2'] .step-action")
    geo = pg.evaluate(THUMB_GEOMETRY)
    assert next(t for t in geo["thumbs"] if t["step"] == "2")["current"] and visible_in_strip(geo, "2")
    assert shown_image(pg).endswith("step_02_overlay.png")
    no_page_overflow(pg)
    watch.assert_clean(base)
