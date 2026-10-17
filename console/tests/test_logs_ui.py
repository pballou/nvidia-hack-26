"""The logs page (/logs) in a real browser, against a live sandbox and a console running the real mock agent.

Two mock runs are made first: one started by voice through the console page (Chromium's fake microphone; the
/api/transcribe reply is route-fulfilled, so its model and latency are known), and one free-text run through
POST /api/run. A third console with a 1 s step delay checks the auto-refresh while a run is still going.

Checks: the runs table and the event rows work by keyboard only (one tab stop each, arrows, Home/End, Enter);
the kind / model / text filters and deep links (?run=&kind=&model=) agree with the URL; Copy JSON puts exactly the
API's JSON on the clipboard (and falls back to a selected textarea); the download link sends an attachment; the
voice row shows the model, the latency and the text; the verdict row says what the console banner said; mock runs
say "mock"; missing values show "—". Every test also checks the page logs no console errors and makes no request
that leaves 127.0.0.1. DOM selectors are fine here (tests only).

Marked `e2e` (a browser plus real mock runs, about 30 s); skip with -m "not e2e".
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from console import asr
from console.app import CONSOLE_DIR, ConsoleSettings, create_app
from console.runner import VERDICT_LABEL
from console.tests.conftest import ServerThread

pytestmark = pytest.mark.e2e

STATIC = CONSOLE_DIR / "static"
VIEWPORT = {"width": 1280, "height": 800}
LOCAL_HOST = "127.0.0.1"
FAKE_MIC = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]
TASK, TASK_REQUEST = "high-contrast", "Turn on high contrast."
HEARD = "turn on high contrast"  # no capital, no period: the console still matches the task
TRANSCRIPT = {"text": HEARD, "model": "test-asr", "latency_ms": 123.4, "duration_s": 1.25}
FREE_TEXT = "Please water my plants"
FINAL = {"done", "stopped", "error"}


def _console(sandbox_url: str, runs: Path, **kw) -> ServerThread:
    settings = ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, headed=False,
                               static_dir=STATIC, stop_timeout_s=10.0, **kw)
    return ServerThread(create_app(settings))


@pytest.fixture(scope="module")
def stub_asr():
    asr.set_engine(asr.StubEngine(HEARD))  # voice input is available; the reply itself is route-fulfilled
    yield
    asr.set_engine(None)


@pytest.fixture(scope="module")
def logs_console(sandbox_url, tmp_path_factory, stub_asr):
    srv = _console(sandbox_url, tmp_path_factory.mktemp("logs-runs"), step_delay_s=0.0)
    yield srv.start()
    srv.stop()


@pytest.fixture(scope="module")
def slow_console(sandbox_url, tmp_path_factory, stub_asr):
    """1 s per step, so a run is still going while the logs page watches it."""
    srv = _console(sandbox_url, tmp_path_factory.mktemp("logs-slow-runs"), step_delay_s=1.0)
    yield srv.start()
    srv.stop()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(args=FAKE_MIC)
        yield b
        b.close()


@dataclass
class Watch:
    """What a page did: every request URL, off-host ones (aborted), console errors, /api/run bodies."""
    requests: list[str] = field(default_factory=list)
    offhost: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    run_bodies: list[dict] = field(default_factory=list)

    def assert_clean(self, base: str, allow_404: bool = False) -> None:
        assert not self.offhost, f"requests left {LOCAL_HOST}: {self.offhost}"
        errors = [e for e in self.errors if not (allow_404 and "404" in e and "Failed to load resource" in e)]
        assert not errors, errors
        origin = urlsplit(base).netloc
        assert self.requests and all(urlsplit(u).netloc == origin for u in self.requests), self.requests

    def detail_fetches(self, run_id: str) -> int:
        return sum(1 for u in self.requests if urlsplit(u).path == f"/api/logs/runs/{run_id}")


def new_page(browser, **ctx_kw):
    watch = Watch()
    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="en-US",
                              timezone_id="America/New_York", **ctx_kw)

    def route(r):
        url = r.request.url
        path = urlsplit(url).path
        watch.requests.append(url)
        if urlsplit(url).hostname != LOCAL_HOST:
            watch.offhost.append(url)
            return r.abort("blockedbyclient")
        if r.request.method == "POST" and path == "/api/run":
            watch.run_bodies.append(json.loads(r.request.post_data or "{}"))
        if r.request.method == "POST" and path == "/api/transcribe":
            return r.fulfill(status=200, content_type="application/json", body=json.dumps(TRANSCRIPT))
        return r.continue_()

    ctx.route("**/*", route)
    pg = ctx.new_page()
    pg.on("console", lambda m: watch.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: watch.errors.append(f"pageerror: {e}"))
    return ctx, pg, watch


@pytest.fixture
def page(browser):
    """(page, watch): a fresh 1280x800 context with clipboard access, aborting anything not on 127.0.0.1."""
    ctx, pg, watch = new_page(browser, permissions=["clipboard-read", "clipboard-write"])
    yield pg, watch
    ctx.close()


def wait_final(base: str, console_run_id: str, timeout_s: float = 60) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = httpx.get(f"{base}/api/runs/{console_run_id}").json()
        if info["status"] in FINAL:
            return info
        time.sleep(0.2)
    raise AssertionError(f"run {console_run_id} did not finish")


@dataclass
class Runs:
    voice_id: str          # agent run id of the voice-triggered sandbox-task run
    text_id: str           # agent run id of the free-text run (newer)
    banner: dict           # the console banner after the voice run
    voice_body: dict       # what the console POSTed to /api/run for it


@pytest.fixture(scope="module")
def runs(browser, logs_console) -> Runs:
    base = logs_console
    ctx, pg, watch = new_page(browser)
    try:
        # 1) by voice, through the console page: hold Right Option, the transcript selects the task, it runs
        pg.goto(base + "/")
        pg.wait_for_function("document.querySelector('#task-select').options.length > 12")
        pg.wait_for_function("document.querySelector('#voice-hint').textContent !== 'Checking voice input…'")
        pg.focus("#request")
        pg.keyboard.down("AltRight")
        pg.wait_for_function("document.querySelector('#mic-btn').dataset.state === 'listening'")
        time.sleep(0.6)
        pg.keyboard.up("AltRight")
        pg.wait_for_selector("#banner:not([hidden])", timeout=60000)
        banner = {k: content(pg, f"#banner-{k}") for k in ("title", "label", "expl")}
        banner["tone"] = pg.get_attribute("#banner", "data-tone")
        href = pg.get_attribute("#banner-logs-link", "href")
        assert content(pg, "#banner-logs-link") == "View this run in Logs"
        m = re.fullmatch(r"/logs\?run=([A-Za-z0-9._-]+)", href or "")
        assert m, href
        voice_id = m.group(1)
        assert len(watch.run_bodies) == 1, watch.run_bodies
        watch.assert_clean(base)
    finally:
        ctx.close()

    # 2) free text, straight through the API (no voice, no sandbox check)
    r = httpx.post(f"{base}/api/run", json={"request": FREE_TEXT})
    assert r.status_code == 200, r.text
    info = wait_final(base, r.json()["run_id"])
    assert info["agent_run_id"] and info["agent_run_id"] != voice_id
    return Runs(voice_id=voice_id, text_id=info["agent_run_id"], banner=banner, voice_body=watch.run_bodies[0])


# --- helpers ----------------------------------------------------------------------------------------------------


def focused(pg) -> dict:
    return pg.evaluate("""() => { const e = document.activeElement;
        return {id: e.id, tag: e.tagName, cls: String(e.className), runId: e.dataset ? e.dataset.runId : undefined,
                idx: e.dataset ? e.dataset.idx : undefined, text: (e.textContent || '').trim().slice(0, 80)}; }""")


def tab_to(pg, pred, limit: int = 12, key: str = "Tab") -> list[dict]:
    seen = []
    for _ in range(limit):
        pg.keyboard.press(key)
        f = focused(pg)
        seen.append(f)
        if pred(f):
            return seen
    raise AssertionError(f"never reached the target with {key}; focus went through {seen}")


def params(pg) -> dict:
    """The page's query now (pg.url catches up with history.replaceState only later)."""
    return {k: v[0] for k, v in parse_qs(pg.evaluate("location.search").lstrip("?")).items()}


def open_logs(pg, base: str, query: str = "") -> None:
    pg.goto(f"{base}/logs{query}")
    pg.wait_for_function("document.querySelectorAll('#runs-body tr').length > 0")
    if "run=" in query:
        pg.wait_for_function("document.querySelectorAll('#event-list li.ev').length > 0")


def visible_kinds(pg) -> list[str]:
    return pg.evaluate("[...document.querySelectorAll('#event-list li.ev')].filter(li => !li.hidden)"
                       ".map(li => li.dataset.kind)")


def all_kinds(pg) -> list[str]:
    return pg.evaluate("[...document.querySelectorAll('#event-list li.ev')].map(li => li.dataset.kind)")


def row(pg, idx: int) -> str:
    return f"#event-list li.ev[data-idx='{idx}']"


def norm(s: str | None) -> str:
    return (s or "").replace("\xa0", " ")  # the page keeps "123 ms" together with no-break spaces


def inner(pg, sel: str) -> str:
    return norm(pg.inner_text(sel))


def content(pg, sel: str) -> str:
    return norm(pg.text_content(sel))


def cell_text(pg, sel: str) -> str:
    """Visible text only (the '—' for a missing value, without its screen-reader words)."""
    return norm(pg.evaluate("""(sel) => { const el = document.querySelector(sel).cloneNode(true);
        el.querySelectorAll('.visually-hidden').forEach(n => n.remove()); return el.textContent.trim(); }""", sel))


# --- tests ----------------------------------------------------------------------------------------------------------


def test_voice_run_body_and_runs_table_by_keyboard(page, logs_console, runs):
    pg, watch = page
    # the console sent what it heard with the run (console.json only; never passed to the agent)
    assert runs.voice_body == {"request": TASK_REQUEST, "task_id": TASK,
                               "options": {"grounder": "base", "thinking": "retry", "lightning": "local"},
                               "voice": TRANSCRIPT}
    open_logs(pg, logs_console)
    pg.wait_for_function("document.querySelectorAll('#runs-body tr').length === 2")
    assert pg.eval_on_selector_all("#runs-body tr", "trs => trs.map(t => t.dataset.runId)") == [runs.text_id, runs.voice_id]
    assert pg.eval_on_selector_all("#runs-body .agent-tag", "els => els.map(e => e.textContent)") == ["mock", "mock"]
    assert pg.locator("#run-section").is_hidden() and pg.locator("#run-placeholder").is_visible()

    voice_tr = f"#runs-body tr[data-run-id='{runs.voice_id}']"
    text_tr = f"#runs-body tr[data-run-id='{runs.text_id}']"
    assert cell_text(pg, f"{voice_tr} .run-req") == TASK_REQUEST
    assert cell_text(pg, f"{voice_tr} .task-id") == TASK
    assert cell_text(pg, f"{text_tr} .task-id") == "free text"
    assert content(pg, f"{voice_tr} .status-chip") == "Done"
    checks = lambda tr: pg.eval_on_selector_all(f"{tr} td.c-check .tick", "els => els.map(e => e.className)")
    assert checks(voice_tr) == ["tick tick-yes", "tick tick-yes"]   # claimed ✓, verified ✓
    assert checks(text_tr) == ["tick tick-no", "tick tick-none"]    # claimed ✗, no sandbox check —
    # the mock runs no Lightning or VLM: no plan/summary/ground latency, only the measured act (and voice) p50
    p50 = [norm(t) for t in pg.eval_on_selector_all(
        f"{voice_tr} td.p50", "els => els.map(e => e.textContent.replace(/not recorded/, ''))")]
    assert p50[0] == "123 ms" and p50[1:4] == ["—", "—", "—"] and re.fullmatch(r"[\d.,]+ (ms|s)", p50[4]), p50
    assert "ASR test-asr" in inner(pg, f"{voice_tr} .models")

    # keyboard only: the table is one tab stop; arrows move, Enter opens, the URL follows
    path = tab_to(pg, lambda f: "run-link" in f["cls"], limit=8)
    assert [f["id"] for f in path][:-1][-1] == "refresh-btn", path
    assert focused(pg)["runId"] == runs.text_id
    pg.keyboard.press("ArrowDown")
    assert focused(pg)["runId"] == runs.voice_id
    assert "run" not in params(pg)  # moving doesn't open
    pg.keyboard.press("Enter")
    pg.wait_for_function(f"document.querySelector('#run-heading').textContent === 'Run {runs.voice_id}'")
    assert params(pg) == {"run": runs.voice_id}
    assert focused(pg)["runId"] == runs.voice_id  # focus stays in the table
    assert "is-selected" in pg.get_attribute(voice_tr, "class")
    assert pg.get_attribute(f"{voice_tr} .run-link", "aria-current") == "true"
    assert pg.locator("#run-placeholder").is_hidden()

    pg.keyboard.press("Home")
    assert focused(pg)["runId"] == runs.text_id
    pg.keyboard.press("Space")  # Space opens too
    pg.wait_for_function(f"document.querySelector('#run-heading').textContent === 'Run {runs.text_id}'")
    assert params(pg) == {"run": runs.text_id}
    assert "Free text" in inner(pg, "#run-summary") or "free text" in inner(pg, "#run-summary")
    pg.keyboard.press("End")
    assert focused(pg)["runId"] == runs.voice_id
    pg.keyboard.press("Shift+Tab")
    assert focused(pg)["id"] == "refresh-btn"
    pg.keyboard.press("Tab")
    assert focused(pg)["runId"] == runs.voice_id  # the tab stop stays where the arrows left it

    # Back returns to the run opened before
    pg.go_back()
    pg.wait_for_function(f"document.querySelector('#run-heading').textContent === 'Run {runs.voice_id}'")
    assert params(pg) == {"run": runs.voice_id}
    watch.assert_clean(logs_console)


def test_event_rows_by_keyboard_voice_row_and_verdict(page, logs_console, runs):
    pg, watch = page
    open_logs(pg, logs_console, f"?run={runs.voice_id}")
    detail = httpx.get(f"{logs_console}/api/logs/runs/{runs.voice_id}").json()
    kinds = all_kinds(pg)
    assert kinds == [e["kind"] for e in detail["events"]]
    assert kinds[0] == "voice" and kinds[1] == "start" and kinds[-2:] == ["done", "verdict"], kinds
    assert content(pg, "#events-meta") == f"{len(kinds)} events"
    badges = content(pg, "#run-badges")
    assert "mock" in badges and "Done" in badges and "Verified" in badges

    # the voice row: from the console, with the speech model, its latency and what it heard
    v = row(pg, 0)
    assert pg.get_attribute(v, "data-source") == "console"
    assert cell_text(pg, f"{v} .ev-src") == "console"
    assert cell_text(pg, f"{v} .ev-model") == "test-asr"
    assert cell_text(pg, f"{v} .ev-lat") == "123 ms"
    assert cell_text(pg, f"{v} .ev-desc") == f"Heard: “{HEARD}”"
    assert cell_text(pg, f"{v} .ev-t") == "—"  # the backend has no timestamp for it: shown as missing, not guessed

    # keyboard only: the event log is one tab stop, on the first event
    tab_to(pg, lambda f: "ev-row" in f["cls"], limit=40)
    assert focused(pg)["idx"] == "0"
    pg.keyboard.press("Enter")
    assert pg.get_attribute(f"{v} .ev-row", "aria-expanded") == "true"
    det = inner(pg, f"{v} .ev-details")
    assert "Speech model" in det and "test-asr" in det and "123 ms" in det and HEARD in det and "1.25 s" in det
    pg.keyboard.press("ArrowDown")
    assert focused(pg)["idx"] == "1"
    pg.keyboard.press("Enter")
    assert "Full run config (JSON)" in inner(pg, f"{row(pg, 1)} .ev-details")
    pg.keyboard.press("Enter")  # and closed again
    assert pg.get_attribute(f"{row(pg, 1)} .ev-row", "aria-expanded") == "false"
    assert pg.locator(f"{row(pg, 1)} .ev-details").is_hidden()

    # the verdict row (End) says what the console banner said
    pg.keyboard.press("End")
    last = len(kinds) - 1
    assert focused(pg)["idx"] == str(last)
    pg.keyboard.press("Enter")
    vr = row(pg, last)
    verdict = detail["events"][-1]["payload"]
    assert runs.banner["tone"] == "pass" and verdict["passed"] is True
    assert runs.banner["title"] in cell_text(pg, f"{vr} .ev-desc")  # "Verified: the task is done"
    assert cell_text(pg, f"{vr} .ev-src") == "console"
    vdet = inner(pg, f"{vr} .ev-details")
    assert runs.banner["label"] == VERDICT_LABEL and VERDICT_LABEL in vdet
    assert verdict["explanation"] in vdet and "✓ passed" in vdet
    n = re.fullmatch(r"all (\d+) conditions? passed", verdict["explanation"]).group(1)
    assert runs.banner["expl"].startswith(f"Sandbox check: {n}/{n} condition"), runs.banner
    pg.keyboard.press("Home")
    assert focused(pg)["idx"] == "0"

    # the summary card: the voice record and what's missing for a mock run
    summary = inner(pg, "#run-summary")
    assert HEARD in summary and "test-asr" in summary and "123 ms" in summary
    assert "mock (no Lightning)" in summary and "mock (no VLM)" in summary
    assert "scripted replay" in summary

    # a grounding row: what the grounder saw, as a thumbnail that opens full size (keyboard: Enter, Escape)
    gi = kinds.index("grounding")
    pg.focus(f"{row(pg, gi)} .ev-row")
    pg.keyboard.press("Enter")
    g = detail["events"][gi]
    gdet = inner(pg, f"{row(pg, gi)} .ev-details")
    assert g["payload"]["target"] in gdet and ", ".join(str(n) for n in g["payload"]["point_1000"]) in gdet
    assert "Mock script (pre-measured point)" in cell_text(pg, f"{row(pg, gi)} .ev-desc")
    thumb = f"{row(pg, gi)} .thumb-btn"
    assert pg.get_attribute(thumb, "data-src") == g["overlay_url"]
    pg.locator(thumb).scroll_into_view_if_needed()  # the thumbnail loads lazily
    pg.wait_for_function(f"document.querySelector(\"{thumb} img\").naturalWidth === 1280")
    pg.focus(thumb)
    pg.keyboard.press("Enter")
    pg.wait_for_selector("#img-dialog[open]")
    assert focused(pg)["id"] == "img-dialog-close"
    pg.wait_for_function("document.querySelector('#img-dialog-img').naturalWidth === 1280")
    pg.keyboard.press("Escape")
    pg.wait_for_selector("#img-dialog:not([open])", state="attached")
    assert "thumb-btn" in focused(pg)["cls"]  # focus back where it was
    watch.assert_clean(logs_console)


def test_filters_by_kind_model_and_search_follow_the_url(page, logs_console, runs):
    pg, watch = page
    rid = runs.voice_id
    open_logs(pg, logs_console, f"?run={rid}")
    kinds = all_kinds(pg)
    counts = {k: kinds.count(k) for k in kinds}
    chips = pg.eval_on_selector_all("#kind-chips .kind-chip", """els => els.map(e => [e.dataset.kind,
        e.querySelector('.chip-count').textContent, e.querySelector('input').checked])""")
    assert {k: int(n) for k, n, _ in chips} == counts and all(c for *_, c in chips)

    # Space on a kind chip hides that kind; focus stays on the chip; the URL lists the kinds still shown
    pg.focus("#kind-voice")
    pg.keyboard.press("Space")
    assert focused(pg)["id"] == "kind-voice"
    assert "voice" not in visible_kinds(pg) and len(visible_kinds(pg)) == len(kinds) - 1
    assert params(pg)["kind"].split(",") == [k for k in counts if k != "voice"]
    assert content(pg, "#events-meta") == f"Showing {len(kinds) - 1} of {len(kinds)} events"
    pg.keyboard.press("Space")
    assert "kind" not in params(pg) and visible_kinds(pg) == kinds

    # the model filter lists the models in this run; one model shows only its events
    options = pg.eval_on_selector_all("#model-filter option", "os => os.map(o => o.value)")
    assert set(options) == {"", "test-asr", "mock (no Lightning)", "mock (no VLM)", "base, mock"}, options
    pg.select_option("#model-filter", "base, mock")
    assert visible_kinds(pg) == ["grounding"] * counts["grounding"]
    assert params(pg)["model"] == "base, mock"

    # text search (with the model filter still on), then "Show all events" clears everything
    pg.fill("#event-search", "accessibility tab")
    assert visible_kinds(pg) == ["grounding"]
    assert params(pg)["q"] == "accessibility tab"
    pg.fill("#event-search", "no event says this")
    assert visible_kinds(pg) == [] and pg.locator("#events-empty").is_visible()
    pg.focus("#clear-filters")
    pg.keyboard.press("Enter")
    assert visible_kinds(pg) == kinds and params(pg) == {"run": rid}
    assert pg.input_value("#event-search") == "" and pg.input_value("#model-filter") == ""

    # deep links set the filters
    open_logs(pg, logs_console, f"?run={rid}&kind=plan,grounding")
    assert set(visible_kinds(pg)) == {"plan", "grounding"}
    checked = pg.eval_on_selector_all("#kind-chips input:checked", "els => els.map(e => e.value)")
    assert checked == ["plan", "grounding"]
    assert content(pg, "#events-meta") == (
        f"Showing {counts['plan'] + counts['grounding']} of {len(kinds)} events")
    open_logs(pg, logs_console, f"?run={rid}&model=mock (no Lightning)")
    assert visible_kinds(pg) == ["plan"] * counts["plan"]
    assert pg.input_value("#model-filter") == "mock (no Lightning)"
    open_logs(pg, logs_console, f"?run={rid}&q=Accessibility")
    assert 0 < len(visible_kinds(pg)) < len(kinds) and pg.input_value("#event-search") == "Accessibility"
    watch.assert_clean(logs_console)


def test_copy_json_and_download(page, logs_console, runs):
    pg, watch = page
    rid = runs.voice_id
    open_logs(pg, logs_console, f"?run={rid}")
    api = httpx.get(f"{logs_console}/api/logs/runs/{rid}").json()
    assert pg.get_attribute("#copy-status", "role") == "status"

    pg.focus("#copy-btn")
    pg.keyboard.press("Enter")
    pg.wait_for_function("document.querySelector('#copy-status').textContent.startsWith('Copied')")
    assert json.loads(pg.evaluate("navigator.clipboard.readText()")) == api
    assert pg.locator("#copy-fallback-wrap").is_hidden()

    # the download: a server attachment (managed Chrome blocks blob: downloads)
    href = pg.get_attribute("#download-link", "href")
    assert href == f"/api/logs/runs/{rid}/download"
    r = httpx.get(logs_console + href)
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.headers["content-disposition"] == f'attachment; filename="run-{rid}.json"'
    assert r.json() == api
    pg.focus("#download-link")
    with pg.expect_download() as dl:
        pg.keyboard.press("Enter")
    assert dl.value.suggested_filename == f"run-{rid}.json"
    assert json.loads(Path(dl.value.path()).read_text()) == api

    # no clipboard: the JSON is shown selected in a textarea instead
    pg.add_init_script("navigator.clipboard.writeText = () => Promise.reject(new Error('blocked by policy'));")
    open_logs(pg, logs_console, f"?run={rid}")
    pg.click("#copy-btn")
    pg.wait_for_selector("#copy-fallback-wrap:not([hidden])")
    assert focused(pg)["id"] == "copy-fallback"
    sel = pg.evaluate("(() => { const t = document.querySelector('#copy-fallback');"
                      " return [t.selectionStart, t.selectionEnd, t.value.length]; })()")
    assert sel[0] == 0 and sel[1] == sel[2] > 0
    assert json.loads(pg.input_value("#copy-fallback")) == api
    status = content(pg, "#copy-status")
    assert status.startswith("Copied") or "Ctrl+C" in status, status
    watch.assert_clean(logs_console)


def test_auto_refresh_while_running_then_stops(page, slow_console):
    pg, watch = page
    r = httpx.post(f"{slow_console}/api/run", json={"request": TASK_REQUEST, "task_id": TASK})
    assert r.status_code == 200, r.text
    crid = r.json()["run_id"]
    deadline = time.time() + 30
    aid = None
    while time.time() < deadline and not aid:
        aid = httpx.get(f"{slow_console}/api/runs/{crid}").json()["agent_run_id"]
        time.sleep(0.1)
    assert aid

    open_logs(pg, slow_console, f"?run={aid}")
    assert "Running" in content(pg, "#run-badges")
    assert "refreshing every 3 s" in content(pg, "#refresh-note")
    first = len(all_kinds(pg))
    pg.focus(f"{row(pg, 0)} .ev-row")
    pg.keyboard.press("Enter")  # open the start event; the refreshes must keep it open and focused

    pg.wait_for_function("document.querySelector('#run-badges').textContent.includes('Done')", timeout=40000)
    pg.wait_for_function("[...document.querySelectorAll('#event-list li.ev')].some(li => li.dataset.kind === 'verdict')")
    assert len(all_kinds(pg)) > first
    assert pg.get_attribute(f"{row(pg, 0)} .ev-row", "aria-expanded") == "true"
    assert focused(pg)["idx"] == "0"
    assert "refreshing" not in content(pg, "#refresh-note")
    assert content(pg, f"#runs-body tr[data-run-id='{aid}'] .status-chip") == "Done"

    n = watch.detail_fetches(aid)
    assert n >= 2  # the first load plus at least one 3 s refresh
    time.sleep(3.6)
    assert watch.detail_fetches(aid) == n  # done: no timer left
    pg.click("#refresh-btn")
    pg.wait_for_function(f"document.querySelector('#refresh-btn').disabled === false")
    assert watch.detail_fetches(aid) == n + 1
    watch.assert_clean(slow_console)


def test_theme_is_shared_and_missing_runs_say_so(page, logs_console, runs):
    pg, watch = page
    pg.goto(logs_console + "/")
    pg.wait_for_function("document.querySelector('#task-select').options.length > 12")
    assert pg.get_attribute("#logs-link", "href") == "/logs" and content(pg, "#logs-link") == "Logs"
    pg.focus("#hc-toggle")
    pg.keyboard.press("Enter")
    assert pg.evaluate("document.documentElement.classList.contains('hc')")
    pg.focus("#logs-link")
    pg.keyboard.press("Enter")
    pg.wait_for_url(re.compile(r"/logs$"))
    pg.wait_for_function("document.querySelectorAll('#runs-body tr').length > 0")
    assert pg.evaluate("document.documentElement.classList.contains('hc')")  # the same localStorage key
    assert pg.get_attribute("#hc-toggle", "aria-pressed") == "true"
    pg.focus("#hc-toggle")
    pg.keyboard.press("Enter")
    assert pg.evaluate("localStorage.getItem('gui-assist.high-contrast')") == "0"
    assert pg.get_attribute("a.nav-link", "href") == "/"

    pg.goto(f"{logs_console}/logs?run=no-such-run")
    pg.wait_for_function("document.querySelector('#run-error').textContent !== ''")
    assert "There is no run “no-such-run”" in content(pg, "#run-error")
    assert pg.is_disabled("#copy-btn") and pg.locator("#download-link").is_hidden()
    fetched = len(watch.requests)
    pg.goto(f"{logs_console}/logs?run=../../etc")  # not a run id: not even asked for
    pg.wait_for_function("document.querySelector('#run-error').textContent.includes('../../etc')")
    assert not [u for u in watch.requests[fetched:] if "/api/logs/runs/" in urlsplit(u).path]
    watch.assert_clean(logs_console, allow_404=True)
