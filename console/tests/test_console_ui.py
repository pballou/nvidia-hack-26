"""The console page in a real browser, driven by keyboard only, against a live sandbox and the real mock agent.

Headless Chromium loads console/static from a live console (--agent mock, no step delay) and uses only the
keyboard: Tab to the task picker, type-ahead to pick a task, Enter to run, Escape to stop, arrow keys through the
step thumbnails. Checks: the step log reads in plain words ("Clicking: ..."), thumbnails replay steps (arrows work
straight from the banner too), the final banner carries the sandbox verdict with its exact label and takes focus,
typed task text runs as that task, free text gets no sandbox check, focus never drops to <body> when the form locks,
and the option toggles show the server defaults with clear labels. A replayed T2 event stream checks the step
headers for replies without an action, sub-second timings in ms, accessible names and one announcement per step.
Every test also checks that the page logs no console errors and makes no request that leaves 127.0.0.1.

DOM selectors are fine here (tests only). The agent under test never sees the DOM: the mock clicks pixels.
Marked `e2e` like test_mock_e2e.py (a browser plus real mock runs, about 10 s); skip with -m "not e2e".
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
    assert watch.run_bodies == [{"request": TASK_REQUEST, "task_id": TASK,
                                 "options": {"grounder": "base", "thinking": "retry", "lightning": "local"}}]

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
    assert "Grounder (base, mock) chose" in pg.inner_text(step1)
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
    assert pg.input_value("#task-select") == ""  # typing alone doesn't switch the picker; running does
    pg.keyboard.press("Enter")
    wait_banner(pg)
    assert watch.run_bodies[-1]["task_id"] == "mark-all-read"
    assert watch.run_bodies[-1]["request"] == pg.input_value("#request") == "Mark all my messages as read."
    assert pg.input_value("#task-select") == "mark-all-read"
    assert step_actions(pg)[0] == "Started at /messages"
    assert pg.get_attribute("#banner", "data-tone") == "pass"
    assert pg.text_content("#banner-label") == VERDICT_LABEL

    # picking a task and then changing only case or punctuation keeps the task (no "runs as free text" note)
    pg.select_option("#task-select", TASK)
    pg.fill("#request", "turn on high contrast")
    assert pg.input_value("#task-select") == TASK and pg.inner_text("#request-note") == ""
    pg.fill("#request", "turn on high contrast now")
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


@pytest.mark.parametrize("default_lightning", ["local", "hosted"])
def test_options_show_server_defaults_with_clear_labels(page, console, default_lightning):
    pg, watch = page
    base, _ = console(None, agent="mock", static_dir=STATIC, default_lightning=default_lightning)
    assert httpx.get(f"{base}/api/config").json()["defaults"]["lightning"] == default_lightning
    open_console(pg, base)

    labels = pg.eval_on_selector_all(".options .seg label", "els => els.map(l => [l.htmlFor, l.textContent])")
    assert labels == [["grounder-base", "base (adapter off)"], ["grounder-grounder", "grounder (LoRA)"],
                      ["thinking-off", "off"], ["thinking-retry", "retry"], ["thinking-on", "on"],
                      ["lightning-local", "Local (BF16 on :8000)"], ["lightning-hosted", "Hosted API"]]
    checked = pg.eval_on_selector_all(".options input:checked", "els => els.map(i => i.id)")
    assert checked == ["grounder-base", "thinking-retry", f"lightning-{default_lightning}"]
    summary = "local Lightning" if default_lightning == "local" else "hosted Lightning"
    assert pg.text_content("#options-summary") == f"base · thinking retry · {summary}"

    # keyboard: the radio group is one tab stop, arrows change the choice, and the summary follows
    tab_to(pg, lambda f: f["id"] == f"lightning-{default_lightning}", limit=10)
    pg.keyboard.press("ArrowRight" if default_lightning == "local" else "ArrowLeft")
    other = "hosted" if default_lightning == "local" else "local"
    assert pg.is_checked(f"#lightning-{other}")
    assert pg.text_content("#options-summary").endswith(f"{other} Lightning")
    watch.assert_clean(base)
