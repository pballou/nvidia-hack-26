"""The "Ask when unclear" step (agent/clarify.py) in the console page: headless Chromium against a live sandbox and
the real mock agent (console/mock_agent.py, which plays the request check from scripted cases), plus fake agents for
the shapes the mock doesn't make.

- An unclear request ("Allors towards my bill.") ends with done reason needs_input before any browser opens: the banner
  asks the question, focus lands in the answer box, and the answer (typed, keyboard only) starts a follow-up run:
  POST /api/run {request: "<previous> (clarified: <answer>)", clarifies: {parent_run_id, answer}}. That run's request
  check restates it ("Pay $50 toward my bill."), the page matches the restatement to pay-50 and asks the backend to
  check it as that task (POST /api/runs/{id}/task_match, method "restated"), and the verdict is ✓, with a banner line
  that says the agent's restatement chose the check (the Logs page says the same).
- The same by voice: while the question waits, the next transcript fills the answer box and the countdown sends it
  (Esc cancels it first). Transcripts are route-mocked; the fake microphone is Chromium's beep source.
- "Make the text bigger" proceeds without a question; the step log and the line above it say what the agent understood.
- The options: "Ask when unclear" on and Thinking off by default (from /api/config), both sent with every run; with the
  check off there is no clarify event and nothing about it shows.
- Fake agents: a refused late match (the task starts elsewhere) says why in the log; a failed check says it went ahead.
Every test checks for no console errors and no request off 127.0.0.1. Marked `e2e`; skip with -m "not e2e".
"""

from __future__ import annotations

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
VIEWPORT = {"width": 1280, "height": 800}
FAKE_MIC = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]
DEFAULTS = {"grounder": "grounder", "thinking": "off", "clarify": "on", "lightning": "local"}  # the console's
GARBLED, GARBLED_Q = "Allors towards my bill.", "What would you like to do with your bill?"
VAGUE, VAGUE_Q = "Pay something toward my bill", "How much would you like to pay toward your bill?"
RESTATED = "Pay $50 toward my bill."  # the mock's restatement of either one, answered with fifty dollars
ANSWER_ESC = "Cancelled: the answer wasn’t sent. It’s in the answer box: press Enter there to send it."


def _console(sandbox_url: str, runs: Path, **kw) -> ServerThread:
    settings = ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, headed=False,
                               static_dir=STATIC, stop_timeout_s=10.0, step_delay_s=0.0, **kw)
    return ServerThread(create_app(settings))


@pytest.fixture(scope="module")
def clarify_console(sandbox_url, tmp_path_factory):
    """The real mock agent with the console's defaults (the request check on), no step delay."""
    srv = _console(sandbox_url, tmp_path_factory.mktemp("clarify-runs"))
    yield srv.start()
    srv.stop()


@pytest.fixture(scope="module")
def browser():
    """One Chromium for the module, with the fake microphone (a beep source) for the voice answer."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(args=FAKE_MIC)
        yield b
        b.close()


@dataclass
class Watch:
    """What the page did: POST bodies by path, off-host requests (aborted), console errors. `transcript` set: the
    next POST /api/transcribe answers that text (a route mock; no recognizer runs)."""
    posts: dict = field(default_factory=dict)
    replies: dict = field(default_factory=dict)
    offhost: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    transcript: str | None = None

    def bodies(self, path_re: str) -> list[dict]:
        return [b for p, bs in self.posts.items() if re.fullmatch(path_re, p) for b in bs]

    def assert_clean(self, allow: tuple[int, ...] = ()) -> None:
        assert not self.offhost, f"requests left 127.0.0.1: {self.offhost}"
        errors = [e for e in self.errors
                  if not any(f"status of {code}" in e for code in allow)]  # HTTP errors the test provokes
        assert not errors, errors


@pytest.fixture
def page(browser):
    watch = Watch()
    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="en-US",
                              timezone_id="America/New_York")

    def route(r):
        url = urlsplit(r.request.url)
        if url.hostname != "127.0.0.1":
            watch.offhost.append(r.request.url)
            return r.abort("blockedbyclient")
        if r.request.method == "POST" and url.path.startswith("/api/") and url.path != "/api/transcribe":
            watch.posts.setdefault(url.path, []).append(json.loads(r.request.post_data or "{}"))
        if r.request.method == "POST" and url.path == "/api/transcribe" and watch.transcript is not None:
            return r.fulfill(status=200, content_type="application/json", body=json.dumps(
                {"text": watch.transcript, "duration_s": 0.9, "latency_ms": 12.0, "model": "test"}))
        return r.continue_()

    ctx.route("**/*", route)
    pg = ctx.new_page()
    pg.on("console", lambda m: watch.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: watch.errors.append(f"pageerror: {e}"))
    pg.on("response", lambda r: watch.replies.setdefault(urlsplit(r.url).path, []).append(r)
          if r.request.method == "POST" else None)
    yield pg, watch
    ctx.close()


# --- helpers --------------------------------------------------------------------------------------------------------


def open_console(pg, base: str) -> None:
    pg.goto(base + "/")
    pg.wait_for_function("document.querySelector('#task-select').options.length > 12")
    pg.wait_for_function("document.querySelector('#agent-badge').dataset.agent === 'mock'")


def focused(pg) -> str:
    return pg.evaluate("document.activeElement.id || document.activeElement.tagName")


def wait_banner(pg, timeout_s: float = 60) -> None:
    pg.wait_for_selector("#banner:not([hidden])", timeout=timeout_s * 1000)


def wait_new_banner(pg, timeout_s: float = 60) -> None:
    """The banner went away (a new run) and came back (that run ended)."""
    pg.wait_for_selector("#banner", state="hidden", timeout=10000)
    wait_banner(pg, timeout_s)


def step_lines(pg, step: int = 0) -> list[tuple[str, str]]:
    return [tuple(x) for x in pg.eval_on_selector_all(
        f"#step-log .step[data-step='{step}'] .step-lines > p", "els => els.map(e => [e.className, e.textContent])")]


def no_page_overflow(pg) -> None:
    size = pg.evaluate("""() => { const d = document.documentElement;
        return [d.scrollWidth, d.scrollHeight, innerWidth, innerHeight]; }""")
    assert size[0] <= size[2] and size[1] <= size[3], size


def run_id(pg) -> str:
    return pg.evaluate("sessionStorage.getItem('gui-assist.run')")


def ask_flow(pg, request: str, question: str) -> None:
    """Type `request` by keyboard, run it, and check the needs_input banner."""
    if focused(pg) != "request":
        pg.focus("#request")
    pg.keyboard.type(request)
    pg.keyboard.press("Enter")
    wait_banner(pg)
    assert pg.get_attribute("#banner", "data-tone") == "ask"
    assert pg.inner_text("#banner-icon") == "?"
    assert pg.inner_text("#banner-title") == "The agent asks:"
    assert pg.inner_text("#banner-question") == f"“{question}”"
    assert pg.is_visible("#banner-answer")


# --- tests ------------------------------------------------------------------------------------------------------------


def test_ask_answer_follow_up_by_keyboard(page, clarify_console):
    """Garbled request -> the agent asks -> typed answer -> follow-up run -> restated late match -> ✓."""
    pg, watch = page
    base = clarify_console
    open_console(pg, base)
    for _ in range(5):  # skip link, Logs, high contrast, mic, request box
        pg.keyboard.press("Tab")
    assert focused(pg) == "request"
    ask_flow(pg, GARBLED, GARBLED_Q)
    assert watch.bodies("/api/run") == [{"request": GARBLED, "task_id": None, "options": DEFAULTS}]

    # the banner: the question is its name, focus is in the answer box (whose description reads the question too)
    assert focused(pg) == "answer-input"
    assert pg.get_attribute("#banner", "aria-labelledby") == "banner-title banner-question"
    assert pg.get_attribute("#answer-input", "aria-describedby") == "banner-question answer-hint"
    assert pg.text_content("label[for='answer-input']") == "Your answer"
    assert "Nothing was done yet: it asked before opening the app." in pg.inner_text("#banner-claim")
    assert "0 steps" in pg.inner_text("#banner-claim")
    assert pg.locator("#banner-label").is_hidden() and pg.locator("#banner-task").is_hidden()  # no task, no verdict yet
    assert "Open this session in the sandbox" not in pg.inner_text("#banner-extra")  # no browser was opened
    assert pg.is_visible("#banner-logs-link")
    assert pg.get_attribute("#run-status", "data-state") == "needs_input"
    assert pg.inner_text("#run-status").startswith("Needs your answer")
    no_page_overflow(pg)  # at 1280x800 the answer box fits: the page itself never scrolls
    assert pg.inner_text("#screen-empty") == "No screenshots: the agent asked you first, before opening the app."
    # the step log: step 0 only, the check and its question
    assert pg.eval_on_selector_all("#step-log .step-action", "els => els.map(e => e.textContent)") == [
        "Asked you first, before opening the app"]
    lines = step_lines(pg)
    assert ("step-line", f"You asked: “{GARBLED}”") in lines
    assert ("step-line", "Request check: unclear, so the agent asks before doing anything.") in lines
    assert ("step-line step-ask", f"The agent asks: “{GARBLED_Q}”") == lines[-1], lines
    assert "ask when unclear: on" in pg.inner_text("#step-log .step[data-step='0']")
    assert pg.get_attribute("#step-log .step[data-step='0'] .step-lines > p:nth-child(3)", "title") \
        == "mock (no Lightning) · clarify-v4 (mock) · sha mock"
    assert pg.locator("#understood").is_hidden()
    parent = pg.evaluate("window.AcmeConsole.pendingQuestion()")
    assert parent["question"] == GARBLED_Q and parent["request"] == GARBLED
    assert re.fullmatch(r"[A-Za-z0-9._-]+", parent["parentRunId"])

    # keyboard only: Enter on an empty answer says what's missing; Tab goes to Answer; type and Enter sends it
    pg.keyboard.press("Enter")
    assert pg.inner_text("#answer-error") == "Type your answer first."
    assert focused(pg) == "answer-input" and len(watch.bodies("/api/run")) == 1
    pg.keyboard.press("Tab")
    assert focused(pg) == "answer-btn"
    pg.keyboard.press("Shift+Tab")
    pg.keyboard.type("fifty  dollars")
    assert pg.locator("#answer-error").is_hidden()  # typing clears it
    pg.keyboard.press("Enter")
    wait_new_banner(pg)

    follow_up = f"{GARBLED} (clarified: fifty dollars)"
    assert watch.bodies("/api/run")[1] == {
        "request": follow_up, "task_id": None, "options": DEFAULTS,  # the page's text, for the backend to compare
        "clarifies": {"parent_run_id": parent["parentRunId"], "answer": "fifty dollars"}}
    reply = watch.replies["/api/run"][1].json()
    assert reply["request"] == follow_up  # the backend built the same text (agent.clarify.follow_up_request)
    # the late match: the restatement is pay-50's text, and the run started where pay-50 starts (/)
    assert watch.bodies(r"/api/runs/[^/]+/task_match") == [
        {"task_id": "pay-50", "method": "restated", "score": 1, "matched_from": RESTATED}]

    # the verdict, and the line that says the agent's restatement chose the check
    assert pg.get_attribute("#banner", "data-tone") == "pass"
    assert pg.inner_text("#banner-title") == "Verified: the task is done"
    assert pg.text_content("#banner-label") == VERDICT_LABEL
    task_line = f"Checked as task: pay-50 (matched from the agent's restatement: \"{RESTATED}\")"
    assert pg.inner_text("#banner-task") == task_line
    assert focused(pg) == "banner"
    # the follow-up says what it follows; the request box holds what the agent got
    lines = step_lines(pg)
    assert lines[:2] == [("step-line step-followup", f"Follow-up to: “{GARBLED}”"),
                         ("step-line", "Your answer: “fifty dollars”")]
    step0 = "#step-log .step[data-step='0']"
    assert "is-compact" in pg.get_attribute(step0, "class")  # later steps came; it still says what it follows
    assert pg.is_visible(f"{step0} .step-followup") and pg.is_hidden(f"{step0} .step-understood")
    assert ("step-line", "Request check: clear, so the agent goes ahead.") in lines
    assert ("step-line step-understood", f"What the agent understood: “{RESTATED}”") in lines
    assert ("step-line", "Checked as task pay-50: it matches the agent’s restatement, not your words.") in lines
    assert pg.inner_text("#understood") == f"What the agent understood: “{RESTATED}” · checked as task pay-50"
    assert pg.input_value("#request") == follow_up and pg.input_value("#task-select") == ""
    assert pg.eval_on_selector_all("#step-log .step-action", "els => els.map(e => e.textContent)")[0] == "Started at /"

    # the backend and the Logs page agree: the follow-up's parent, its task and the same task line
    info = httpx.get(f"{base}/api/runs/{run_id(pg)}").json()
    assert info["task_id"] == "pay-50" and info["task_match"]["method"] == "restated"
    assert info["parent_run_id"] == parent["parentRunId"] and info["request"] == follow_up
    logs = httpx.get(f"{base}/api/logs/runs/{info['agent_run_id']}").json()["summary"]
    assert logs["task_line"] == task_line and logs["parent_run_id"] == parent["parentRunId"]

    # shown again after a reload: the follow-up line comes from the backend, the task line from its task_match
    pg.reload()
    wait_banner(pg)
    assert pg.inner_text("#banner-task") == task_line
    assert step_lines(pg)[:2] == [("step-line step-followup", f"Follow-up to: “{GARBLED}”"),
                                  ("step-line", "Your answer: “fifty dollars”")]
    assert pg.input_value("#task-select") == ""  # the restatement's task isn't the request box's
    watch.assert_clean()


def test_voice_answers_the_question(page, clarify_console):
    """While a question waits, the next transcript is the answer: Esc cancels its countdown, the next one sends it."""
    pg, watch = page
    open_console(pg, clarify_console)
    pg.wait_for_function("document.querySelector('#mic-btn').dataset.state === 'idle'")
    ask_flow(pg, VAGUE, VAGUE_Q)
    assert focused(pg) == "answer-input"
    assert pg.text_content("#answer-voice").strip() == "Or hold Right Option and say it: your next voice input answers."
    assert pg.evaluate("window.AcmeConsole.pendingQuestion().question") == VAGUE_Q

    def say(text: str, countdown: bool = True) -> None:
        watch.transcript = text
        pg.keyboard.down("AltRight")  # push-to-talk works in a text field (after ~200 ms held alone)
        pg.wait_for_function("document.querySelector('#mic-btn').dataset.state === 'listening'", timeout=10000)
        time.sleep(0.7)
        pg.keyboard.up("AltRight")
        if countdown:
            pg.wait_for_function("!document.querySelector('#voice-countdown').hidden", timeout=10000)
        else:
            pg.wait_for_function("document.querySelector('#mic-btn').dataset.state === 'done'", timeout=10000)

    say("fifty dollars")
    assert pg.input_value("#answer-input") == "fifty dollars"
    assert pg.input_value("#request") == VAGUE  # the request box keeps the question's request
    assert pg.evaluate("document.querySelector('#voice-status').firstChild.textContent") == "Answer: “fifty dollars”"
    assert pg.text_content("#voice-status .visually-hidden") == " Sends in 1.5 seconds. Escape stops it."
    assert re.fullmatch(r"Answering in \d\.\d s — press Esc to cancel", pg.inner_text("#voice-countdown-text"))
    assert pg.get_attribute("#voice-cancel", "aria-label") == "Cancel sending the answer"
    pg.keyboard.press("Escape")
    assert pg.locator("#voice-countdown").is_hidden()
    assert pg.inner_text("#voice-status") == ANSWER_ESC
    time.sleep(2.0)  # past the countdown: nothing was sent, the question still waits
    assert len(watch.bodies("/api/run")) == 1 and pg.get_attribute("#banner", "data-tone") == "ask"
    assert pg.input_value("#answer-input") == "fifty dollars"

    # a stop phrase is never an answer (nothing is running, so it does nothing)
    say("Stop.", countdown=False)
    assert pg.inner_text("#voice-status") == "Heard: “Stop.”. Nothing is running, so there’s nothing to stop."
    assert pg.input_value("#answer-input") == "fifty dollars" and pg.locator("#voice-countdown").is_hidden()

    # editing the answer during the countdown cancels it
    say("fifty dollars")
    pg.focus("#answer-input")
    pg.keyboard.press("End")
    pg.keyboard.type("!")
    assert pg.locator("#voice-countdown").is_hidden()
    assert pg.inner_text("#voice-status") == ("Cancelled because you edited the answer: press Enter in the answer box "
                                              "to send it.")
    time.sleep(1.8)
    assert len(watch.bodies("/api/run")) == 1

    say("fifty dollars")  # the next take answers again; this time the countdown runs out and sends it
    wait_new_banner(pg)
    follow_up = f"{VAGUE} (clarified: fifty dollars)"
    body = dict(watch.bodies("/api/run")[1])
    voice = body.pop("voice")  # what was heard, for the Logs page
    assert voice["text"] == "fifty dollars" and voice["model"] == "test"
    match = body.pop("task_match")  # the follow-up's own words match pay-50 ("pay" and 50): checked as it
    assert match["method"] == "fuzzy" and match["matched_from"] == follow_up and 0.5 <= match["score"] < 1, match
    assert body == {"request": follow_up, "task_id": "pay-50", "options": DEFAULTS,
                    "clarifies": {"parent_run_id": body["clarifies"]["parent_run_id"], "answer": "fifty dollars"}}
    assert not watch.bodies(r"/api/runs/[^/]+/task_match")  # it has a task already: no late match
    assert pg.get_attribute("#banner", "data-tone") == "pass"
    assert pg.inner_text("#banner-task") == "Checked as task: pay-50 (matched from your request)"
    assert pg.input_value("#request") == follow_up and pg.input_value("#task-select") == "pay-50"
    assert step_lines(pg)[:2] == [("step-line step-followup", f"Follow-up to: “{VAGUE}”"),
                                  ("step-line", "Your answer: “fifty dollars”")]
    assert pg.inner_text("#heard").endswith(": fifty dollars")  # the take that started this run stays shown
    assert pg.inner_text("#understood") == f"What the agent understood: “{RESTATED}”"
    watch.assert_clean()


def test_a_clear_request_proceeds_without_a_question(page, clarify_console):
    pg, watch = page
    open_console(pg, clarify_console)
    pg.focus("#request")
    pg.keyboard.type("Make the text bigger")
    pg.keyboard.press("Enter")  # matches text-size-large (fuzzy): it starts on that task's page
    pg.wait_for_selector("#understood:not([hidden])", timeout=30000)
    assert pg.inner_text("#understood") == "What the agent understood: “Make the text bigger”"
    wait_banner(pg)
    assert pg.get_attribute("#banner", "data-tone") != "ask" and pg.locator("#banner-answer").is_hidden()
    assert pg.locator("#banner-question").is_hidden()
    assert pg.evaluate("window.AcmeConsole.pendingQuestion()") is None
    lines = step_lines(pg)
    assert ("step-line", "Request check: clear, so the agent goes ahead.") in lines
    assert ("step-line step-understood", "What the agent understood: “Make the text bigger”") in lines
    assert not [t for _, t in lines if "asks" in t]
    actions = pg.eval_on_selector_all("#step-log .step-action", "els => els.map(e => e.textContent)")
    assert actions[0] == "Started at /settings?tab=accessibility" and len(actions) >= 2  # the browser loop ran
    assert watch.bodies("/api/run")[0]["task_id"] == "text-size-large"
    assert not watch.bodies(r"/api/runs/[^/]+/task_match")
    assert pg.inner_text("#banner-task") == "Checked as task: text-size-large (matched from your request)"
    watch.assert_clean()


def test_toggles_default_from_config_and_are_sent(page, clarify_console, console):
    """On / off from /api/config, next to Thinking (off by default); the choice goes with the run; off: no check."""
    pg, watch = page
    open_console(pg, clarify_console)
    assert httpx.get(f"{clarify_console}/api/config").json()["defaults"] == DEFAULTS
    assert pg.is_checked("#clarify-on") and pg.is_checked("#thinking-off")
    assert pg.text_content("#opt-clarify legend") == "Ask when unclear one question first"
    tops = pg.evaluate("['#opt-thinking', '#opt-clarify'].map(s => document.querySelector(s).getBoundingClientRect().top)")
    assert tops[0] == tops[1], tops  # side by side
    assert pg.text_content("#options-summary") == "LoRA grounder · thinking off · asks if unclear · local Lightning"

    # keyboard: Thinking's tab stop, then this group's; an arrow turns it off
    pg.focus("#thinking-off")
    pg.keyboard.press("Tab")
    assert focused(pg) == "clarify-on"
    pg.keyboard.press("ArrowRight")
    assert pg.is_checked("#clarify-off")
    assert pg.text_content("#options-summary") == "LoRA grounder · thinking off · never asks first · local Lightning"
    pg.fill("#request", GARBLED)
    pg.press("#request", "Enter")
    wait_banner(pg)
    assert watch.bodies("/api/run")[-1]["options"] == {**DEFAULTS, "clarify": "off"}
    # no clarify event: no question, nothing about a check, the run goes on as before (the mock has no plan for it)
    assert pg.get_attribute("#banner", "data-tone") == "neutral" and pg.locator("#banner-answer").is_hidden()
    assert pg.locator("#understood").is_hidden()
    assert not [t for _, t in step_lines(pg) if "Request check" in t]
    assert "ask when unclear: off" in pg.text_content("#step-log .step[data-step='0']")  # (compact: hidden lines)
    assert "mock agent" in pg.inner_text("#banner-summary")

    # a console started with --clarify off --thinking retry: the page starts there
    base, _ = console(None, agent="mock", static_dir=STATIC, default_clarify="off", default_thinking="retry")
    open_console(pg, base)
    assert pg.is_checked("#clarify-off") and pg.is_checked("#thinking-retry")
    assert pg.text_content("#options-summary") == "LoRA grounder · thinking retry · never asks first · local Lightning"
    watch.assert_clean()


# --- fake agents: shapes the mock doesn't make ------------------------------------------------------------------------


def clarify_agent(trace_root: Path, clarify: dict, timing_ms: float | None = None):
    """(run_task, make_cfg): start, the given clarify payload, one observation, done (no success)."""

    async def run_task(request, *, start_url, cfg, on_event=None, parent_run_id=None):
        rid = f"clarify-{time.time_ns()}"
        d = trace_root / rid
        d.mkdir(parents=True)

        async def ev(step, kind, payload, **extra):
            await on_event({"run_id": rid, "step": step, "kind": kind, "ts": time.time(), "payload": payload, **extra})

        await ev(0, "start", {"request": request, "start_url": start_url, "config": {"clarify": True}})
        await ev(0, "clarify", {"request": request, "model": "lightning", "prompt_version": "clarify-v4",
                                "prompt_sha": "c1adfb0587", **clarify},
                 **({"timings_ms": {"clarify": timing_ms}} if timing_ms is not None else {}))
        img = d / "step_00.png"
        Image.new("RGB", (160, 100), "white").save(img)
        await ev(0, "observation", {"text": "Step 0 | URL: /"}, screenshot_path=str(img))
        await ev(0, "done", {"success_claimed": False, "summary": "Did nothing.", "steps": 0, "reason": "done"})
        return {"run_id": rid, "success_claimed": False, "summary": "Did nothing.", "steps": 0, "trace_dir": str(d)}

    return run_task, lambda options: {"options": options}


def test_a_refused_late_match_says_why(page, console, tmp_path):
    """The restatement matches mark-all-read, which starts at /messages; this free-text run started at /: refused."""
    pg, watch = page
    agent = clarify_agent(tmp_path / "trace", {"action": "proceed", "normalized_request": "Mark all my messages as read."},
                          timing_ms=1612.0)
    base, _ = console(agent, agent="mock", static_dir=STATIC)
    open_console(pg, base)
    # the user's own words support mark-all-read (so the restatement guard lets it through), but the direct match is
    # ruled out (a second request), and the task starts at /messages while this run started at /: the backend refuses
    pg.fill("#request", "mark all my messages as read and then archive them")
    pg.press("#request", "Enter")
    wait_banner(pg, 30)
    assert watch.bodies(r"/api/runs/[^/]+/task_match") == [
        {"task_id": "mark-all-read", "method": "restated", "score": 1, "matched_from": "Mark all my messages as read."}]
    status = watch.replies[next(p for p in watch.replies if p.endswith("/task_match"))][0].status
    assert status in (400, 409), status
    line = next(t for c, t in step_lines(pg) if c == "step-line step-warn")
    assert line.startswith("The agent’s restatement matches task mark-all-read, but this run can’t be checked as it: ")
    assert pg.locator("#banner-task").is_hidden() and pg.text_content("#banner-label") == "Free-text request: no sandbox check"
    assert pg.inner_text("#understood") == "What the agent understood: “Mark all my messages as read.”"
    # the check's latency, first on the step's timing line
    assert pg.text_content("#step-log .step[data-step='0'] .step-meta > span").replace(" ", " ").startswith(
        "clarify 1.6 s")
    watch.assert_clean(allow=(status,))


def test_free_text_over_a_match_is_kept_and_a_failed_check_goes_ahead(page, console, tmp_path):
    """The user chose Free text over high-contrast: its restatement doesn't bring the task back. A check that failed
    says the agent went ahead anyway."""
    pg, watch = page
    agent = clarify_agent(tmp_path / "trace", {"action": "proceed", "normalized_request": "Turn on high contrast.",
                                               "error": "APITimeoutError: Request timed out."})
    base, _ = console(agent, agent="mock", static_dir=STATIC)
    open_console(pg, base)
    pg.fill("#request", "Turn on high contrast.")
    pg.dispatch_event("#request", "change")
    assert pg.input_value("#task-select") == "high-contrast"
    pg.select_option("#task-select", "")  # Free text
    pg.press("#request", "Enter")
    wait_banner(pg, 30)
    assert watch.bodies("/api/run")[0]["task_id"] is None
    assert not watch.bodies(r"/api/runs/[^/]+/task_match")
    assert ("step-line step-warn",
            "Request check failed (APITimeoutError: Request timed out), so the agent went ahead anyway.") in step_lines(pg)
    assert pg.text_content("#banner-label") == "Free-text request: no sandbox check"
    watch.assert_clean()


@pytest.mark.parametrize("task_id,user_words,ok", [
    ("pay-50", "Allors towards my bill. (clarified: fifty dollars)", True),    # the stage case: the user said 50
    ("pay-50", "Allors towards my bill. (clarified: $50)", True),
    ("pay-50", "Allors towards my bill.", False),                               # the amount came only from the agent
    ("pay-50", "Pay something toward my bill (clarified: the full amount)", False),
    ("high-contrast", "the high contrast thing please", True),
    ("high-contrast", "make it easier to see (clarified: yes please)", False),  # 'contrast' only in the restatement
    ("download-september-statement", "download my statement (clarified: september)", True),
    ("download-september-statement", "download my statement", False),          # the month came only from the agent
    ("report-outage-riverside", "report an outage (clarified: riverside)", True),
    ("report-outage-riverside", "report an outage (clarified: hillcrest)", False),
])
def test_restatement_guard_needs_the_users_own_distinguishing_words(page, clarify_console, task_id, user_words, ok):
    """Orchestrator guard: a restated match is adopted only if the task's distinguishing words are the user's own."""
    pg, watch = page
    open_console(pg, clarify_console)
    got = pg.evaluate("([t, w]) => window.AcmeConsole.matchSupport(t, w)", [task_id, user_words])
    assert got["ok"] is ok, got
    if not ok:
        assert got["missing"]


def test_a_restatement_that_adds_the_amount_is_not_checked(page, console, tmp_path):
    """'Allors towards my bill.' restated as 'Pay $50 toward my bill.': the user never said 50, so no task, no ✓."""
    pg, watch = page
    agent = clarify_agent(tmp_path / "trace", {"action": "proceed", "normalized_request": "Pay $50 toward my bill."})
    base, _ = console(agent, agent="mock", static_dir=STATIC)
    open_console(pg, base)
    pg.fill("#request", "Allors towards my bill.")
    pg.press("#request", "Enter")
    wait_banner(pg, 30)
    assert watch.bodies(r"/api/runs/[^/]+/task_match") == [], "the guard must refuse before asking the backend"
    line = next(t for c, t in step_lines(pg) if c == "step-line step-warn")
    assert line.startswith("The agent’s restatement matches task pay-50, but this run can’t be checked as it: "), line
    assert "50" in line and "only the agent’s restatement" in line, line
    assert pg.locator("#banner-task").is_hidden() and pg.text_content("#banner-label") == "Free-text request: no sandbox check"
    watch.assert_clean()
