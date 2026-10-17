"""The console page in a real browser, driven by keyboard only, against a live sandbox and the real mock agent.

Headless Chromium loads console/static from a live console (--agent mock, no step delay) and uses only the
keyboard: Tab to the task picker, type-ahead to pick a task, Enter to run, Escape to stop, arrow keys through the
step thumbnails. Checks: the step log reads in plain words ("Clicking: ..."), thumbnails replay steps, the final
banner carries the sandbox verdict with its exact label and takes focus, free text gets no sandbox check, and the
option toggles show the server defaults with clear labels. Every test also checks that the page logs no console
errors and makes no request that leaves 127.0.0.1.

DOM selectors are fine here (tests only). The agent under test never sees the DOM: the mock clicks pixels.
Marked `e2e` like test_mock_e2e.py (a browser plus real mock runs, about 10 s); skip with -m "not e2e".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

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
