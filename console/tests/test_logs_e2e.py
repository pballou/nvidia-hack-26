"""Logs page acceptance, end to end: real servers, headless Chromium, the real mock agent and the real agent's traces.

1) Three runs made through the console page by keyboard, against a live sandbox and the real mock agent:
   a typed sandbox task (pick the task, Enter), a voice-triggered task (Chromium's fake microphone, the real
   /api/transcribe with the stub engine, the transcript selects the task, the countdown starts the run) and a run
   stopped mid-way with Escape. Then /logs lists every run with the right status, claimed, verified and steps;
   each run's event log has every events.jsonl line in file order plus the console's voice row (first: model, ms,
   text) and verdict row (last); the p50 cells match the timings in events.jsonl (and nothing the mock didn't
   measure); Copy JSON and Download JSON are byte-identical and hold events.jsonl's lines plus the 2 console events.
2) The two real agent traces in fixtures/real_traces (Lightning + the Qwen VLM; runs/ is gitignored) in a temp runs
   dir behind a console: Lightning plan rows (model or "—", tokens from usage, thinking, tool-call arguments),
   observation rows with the raw screen summary, grounding rows with the raw output, the point and an overlay
   thumbnail, action rows with the click x, y, and no "mock" anywhere on the page. Text contrast is checked
   against WCAG AA in both themes. Screenshots go to the test's tmp dir (look at them after layout changes).
3) A trace with an overflowing number (1e999 -> inf) still opens and downloads (as null), over HTTP.
4) A run watched while it is going: the auto-refresh says once, in the live region, how it ended.

Every page also checks: no console errors, no request that leaves 127.0.0.1. DOM selectors are fine here (tests only).
Marked `e2e` (about 25 s); skip with -m "not e2e".
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from PIL import Image

from console import asr
from console.app import CONSOLE_DIR, ConsoleSettings, create_app
from console.runner import VERDICT_LABEL
from console.tests.conftest import ServerThread

pytestmark = pytest.mark.e2e

STATIC = CONSOLE_DIR / "static"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "real_traces"
DONE_RUN, STOPPED_RUN = "20261017-175025-e9fe8e", "20261017-174924-f75115"
VIEWPORT = {"width": 1440, "height": 900}
FAKE_MIC = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]
LOCAL = "127.0.0.1"
TYPED_TASK = "download-september-statement"
VOICE_TASK, HEARD = "mark-all-read", "mark all my messages as read"  # lower case, no period: still the task
STOP_TASK = "stop-promotions"  # 6 steps: there is time to stop it after the first


# --- servers, browser, pages -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub_asr():
    asr.set_engine(asr.StubEngine(HEARD))  # the real /api/transcribe runs; the stub engine says what was "heard"
    yield
    asr.set_engine(None)


def _serve(sandbox_url: str, runs: Path, **kw) -> ServerThread:
    settings = ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, static_dir=STATIC,
                               stop_timeout_s=10.0, **kw)
    return ServerThread(create_app(settings))


@pytest.fixture(scope="module")
def mock_console(sandbox_url, tmp_path_factory, stub_asr):
    runs = tmp_path_factory.mktemp("e2e-runs")
    srv = _serve(sandbox_url, runs, step_delay_s=0.4)  # paced, so a run can be stopped after its first step
    yield srv.start(), runs
    srv.stop()


@pytest.fixture(scope="module")
def real_console(sandbox_url, tmp_path_factory):
    """A console whose runs dir holds the two real traces (the mock is the agent; it is never started)."""
    runs = tmp_path_factory.mktemp("e2e-real-runs")
    for rid in (DONE_RUN, STOPPED_RUN):
        shutil.copytree(FIXTURES / rid, runs / rid)
    # the PNGs aren't committed: 1280x800 stand-ins with the trace's file names
    for rid, names in ((DONE_RUN, ("step_00.png", "step_01.png", "step_01_overlay.png")), (STOPPED_RUN, ("step_00.png",))):
        for name in names:
            Image.new("RGB", (1280, 800), (240, 244, 248)).save(runs / rid / name)
    srv = _serve(sandbox_url, runs)
    yield srv.start(), runs
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
    requests: list[str] = field(default_factory=list)
    offhost: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def assert_clean(self) -> None:
        assert not self.offhost, f"requests left {LOCAL}: {self.offhost}"
        assert not self.errors, self.errors


@pytest.fixture
def page(browser):
    """(page, watch): 1440x900, clipboard access, anything not on 127.0.0.1 aborted and recorded."""
    watch = Watch()
    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="en-US",
                              timezone_id="America/New_York", permissions=["clipboard-read", "clipboard-write"],
                              accept_downloads=True)

    def route(r):
        watch.requests.append(r.request.url)
        if urlsplit(r.request.url).hostname != LOCAL:
            watch.offhost.append(r.request.url)
            return r.abort("blockedbyclient")
        return r.continue_()

    ctx.route("**/*", route)
    pg = ctx.new_page()
    pg.on("console", lambda m: watch.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: watch.errors.append(f"pageerror: {e}"))
    yield pg, watch
    ctx.close()


# --- helpers ----------------------------------------------------------------------------------------------------


def lines_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def visible(pg, sel: str) -> str:
    """Visible text of one element (a '—' without its screen-reader words), no-break spaces as spaces."""
    return pg.evaluate("""(sel) => { const el = document.querySelector(sel); if (!el) return null;
        const c = el.cloneNode(true); c.querySelectorAll('.visually-hidden').forEach(n => n.remove());
        return c.textContent.replace(/\\u00a0/g, ' ').replace(/\\s+/g, ' ').trim(); }""", sel)


def text_of(pg, sel: str) -> str:
    return pg.inner_text(sel).replace("\xa0", " ")


def ms_of(text: str) -> float:
    """'933 ms' / '2.10 s' / '24.1 s' (the page's formats) -> ms."""
    m = re.fullmatch(r"([\d,.]+) (ms|s)", text.strip())
    assert m, text
    v = float(m.group(1).replace(",", ""))
    return v if m.group(2) == "ms" else v * 1000


def close_to(shown: str, ms: float) -> bool:
    """The page rounds: whole ms below 1 s, 2 decimals below 10 s, 1 decimal below 60 s."""
    tol = 0.5 if ms < 1000 else 5 if ms < 10000 else 50
    return abs(ms_of(shown) - ms) <= tol + 1e-6


def open_logs(pg, base: str, run_id: str | None = None) -> None:
    pg.goto(f"{base}/logs" + (f"?run={run_id}" if run_id else ""))
    pg.wait_for_function("document.querySelectorAll('#runs-body tr').length > 0")
    if run_id:
        pg.wait_for_function("(id) => document.querySelector('#run-heading').textContent === 'Run ' + id"
                             " && document.querySelectorAll('#event-list li.ev').length > 0", arg=run_id)


def ev_sel(i: int) -> str:
    return f"#event-list li.ev[data-idx='{i}']"


def expand_all(pg) -> None:
    """Open every event's details (by keyboard: Home, then Enter and ArrowDown down the list)."""
    n = pg.locator("#event-list li.ev").count()
    pg.focus(f"{ev_sel(0)} .ev-row")
    pg.keyboard.press("Home")
    for _ in range(n):
        pg.keyboard.press("Enter")
        pg.keyboard.press("ArrowDown")
    assert pg.locator("#event-list .ev-row[aria-expanded='true']").count() == n


def details_block(pg, idx: int, label_start: str) -> dict | None:
    """{open, text} of the <details> in event idx whose summary starts with label_start."""
    return pg.evaluate("""([sel, start]) => {
        const d = [...document.querySelectorAll(sel + ' details.d-more')]
          .find(x => x.querySelector('summary').textContent.startsWith(start));
        return d ? {open: d.open, text: d.querySelector('pre').textContent} : null; }""", [ev_sel(idx), label_start])


CONTRAST_JS = r"""
() => {
  const parse = (c) => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null;
    const p = m[1].split(',').map(parseFloat); return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1}; };
  const lin = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const lum = (c) => 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
  const over = (fg, bg) => ({r: fg.r * fg.a + bg.r * (1 - fg.a), g: fg.g * fg.a + bg.g * (1 - fg.a),
                             b: fg.b * fg.a + bg.b * (1 - fg.a), a: 1});
  const bgOf = (el) => { const layers = [];
    for (let e = el; e; e = e.parentElement) { const c = parse(getComputedStyle(e).backgroundColor);
      if (c && c.a > 0) { layers.push(c); if (c.a >= 1) break; } }
    let bg = {r: 255, g: 255, b: 255, a: 1}; for (const l of layers.reverse()) bg = over(l, bg); return bg; };
  const low = []; let n = 0;
  for (const el of document.querySelectorAll('body *')) {
    const own = [...el.childNodes].filter(c => c.nodeType === 3 && c.textContent.trim()).map(c => c.textContent).join(' ').trim();
    if (!own || el.closest('.visually-hidden, [hidden], .thumb-btn, button:disabled')) continue;
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
    const cs = getComputedStyle(el); if (cs.visibility === 'hidden') continue;
    const bg = bgOf(el), fg = over(parse(cs.color), bg);
    const ratio = (Math.max(lum(fg), lum(bg)) + 0.05) / (Math.min(lum(fg), lum(bg)) + 0.05);
    const size = parseFloat(cs.fontSize), large = size >= 24 || (size >= 18.66 && parseInt(cs.fontWeight, 10) >= 700);
    n += 1;
    if (ratio < (large ? 3 : 4.5)) low.push(`${own.slice(0, 40)} (${ratio.toFixed(2)}: ${cs.color} on rgb(${bg.r|0},${bg.g|0},${bg.b|0}))`);
  }
  return {n, low};
}
"""


# --- 1) three runs through the console page, then /logs -------------------------------------------------------------


@dataclass
class UiRun:
    label: str
    task_id: str
    agent_run_id: str
    banner_title: str


def _wait_final(pg, timeout_ms: int = 60000) -> None:
    pg.wait_for_function("['done', 'stopped', 'error'].includes(document.querySelector('#run-status').dataset.state)",
                         timeout=timeout_ms)
    pg.wait_for_selector("#banner-logs-link", timeout=timeout_ms)


def _agent_run_id(pg) -> str:
    m = re.fullmatch(r"/logs\?run=([A-Za-z0-9._-]+)", pg.get_attribute("#banner-logs-link", "href") or "")
    assert m, pg.get_attribute("#banner-logs-link", "href")
    return m.group(1)


@pytest.fixture(scope="module")
def ui_runs(browser, mock_console) -> list[UiRun]:
    """typed, voice, stopped (oldest first), all made through the console page by keyboard."""
    base, _ = mock_console
    ctx = browser.new_context(viewport=VIEWPORT)
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    out: list[UiRun] = []
    try:
        pg.goto(base + "/")
        pg.wait_for_function("document.querySelector('#task-select').options.length > 12")
        pg.wait_for_function("document.querySelector('#voice-hint').textContent !== 'Checking voice input…'")

        # a) typed: pick the sandbox task, then Enter in the request box
        pg.select_option("#task-select", TYPED_TASK)
        pg.focus("#request")
        pg.keyboard.press("Enter")
        _wait_final(pg)
        out.append(UiRun("typed", TYPED_TASK, _agent_run_id(pg), visible(pg, "#banner-title")))

        # b) voice: hold Right Option, the transcript selects the task, the countdown starts the run
        pg.select_option("#task-select", "")
        pg.fill("#request", "")
        pg.focus("#request")
        pg.keyboard.down("AltRight")
        pg.wait_for_function("document.querySelector('#mic-btn').dataset.state === 'listening'")
        time.sleep(0.7)
        pg.keyboard.up("AltRight")
        pg.wait_for_function(f"document.querySelector('#task-select').value === '{VOICE_TASK}'", timeout=15000)
        pg.wait_for_function("document.querySelector('#run-status').dataset.state === 'running'", timeout=15000)
        _wait_final(pg)
        out.append(UiRun("voice", VOICE_TASK, _agent_run_id(pg), visible(pg, "#banner-title")))

        # c) stopped mid-way: Escape once the agent has taken its first step
        pg.select_option("#task-select", STOP_TASK)
        pg.focus("#request")
        pg.keyboard.press("Enter")
        pg.wait_for_function("document.querySelector('#run-status').dataset.state === 'running'"
                             " && document.querySelectorAll('#step-log > li').length >= 2", timeout=30000)
        pg.keyboard.press("Escape")
        pg.wait_for_function("document.querySelector('#run-status').dataset.state === 'stopped'", timeout=20000)
        _wait_final(pg)
        out.append(UiRun("stopped", STOP_TASK, _agent_run_id(pg), visible(pg, "#banner-title")))
        assert not errors, errors
    finally:
        ctx.close()
    assert len({r.agent_run_id for r in out}) == 3, out
    return out


def test_runs_list_matches_the_traces(page, mock_console, ui_runs):
    pg, watch = page
    base, runs = mock_console
    typed, voice, stopped = ui_runs
    api = httpx.get(f"{base}/api/logs/runs").json()
    assert [r["run_id"] for r in api["runs"]] == [stopped.agent_run_id, voice.agent_run_id, typed.agent_run_id]

    open_logs(pg, base)
    pg.wait_for_function("document.querySelectorAll('#runs-body tr').length === 3")
    assert pg.eval_on_selector_all("#runs-body tr", "trs => trs.map(t => t.dataset.runId)") == \
        [r["run_id"] for r in api["runs"]]
    assert visible(pg, "#runs-meta") == "3 runs · 3 mock · newest first"

    want = {typed.agent_run_id: ("Done", "tick-yes", "tick-yes"), voice.agent_run_id: ("Done", "tick-yes", "tick-yes"),
            stopped.agent_run_id: ("Stopped", "tick-no", "tick-no")}
    for run in ui_runs:
        rid = run.agent_run_id
        evs = lines_of(runs / rid / "events.jsonl")
        doc = json.loads((runs / rid / "console.json").read_text())
        done = next(e for e in reversed(evs) if e["kind"] == "done")["payload"]
        s = next(r for r in api["runs"] if r["run_id"] == rid)
        # the summary is what the trace and console.json say
        assert (s["agent"], s["task_id"], s["console_run_id"]) == ("mock", run.task_id, doc["console_run_id"])
        assert s["status"] == done["reason"] == doc["status"] == ("stopped" if run is stopped else "done")
        assert s["steps"] == done["steps"] and s["success_claimed"] is done["success_claimed"]
        assert s["verified"] is doc["verdict"]["passed"] is (run is not stopped)
        assert s["n_events"] == len(evs) and s["n_errors"] == 0
        assert s["started_ts"] == evs[0]["ts"] and s["ended_ts"] == evs[-1]["ts"]
        # p50s: only what the mock measured (act) plus the voice transcription; nothing invented
        acts = [e["timings_ms"]["act"] for e in evs if "act" in (e.get("timings_ms") or {})]
        assert s["stage_p50_ms"]["act"] == (round(statistics.median(acts), 1) if acts else None)
        assert (s["stage_p50_ms"]["plan"], s["stage_p50_ms"]["summary"], s["stage_p50_ms"]["ground"]) == (None,) * 3
        if run is voice:
            assert s["voice"] == doc["voice"] and s["voice"]["text"] == HEARD and s["models"]["asr"] == "stub"
            assert s["stage_p50_ms"]["voice"] == doc["voice"]["latency_ms"]
        else:
            assert s["voice"] is None and s["stage_p50_ms"]["voice"] is None and s["models"]["asr"] is None

        # the table row says the same
        tr = f"#runs-body tr[data-run-id='{rid}']"
        status, claimed, verified = want[rid]
        assert visible(pg, f"{tr} .status-chip") == status
        assert pg.eval_on_selector_all(f"{tr} td.c-check .tick", "els => els.map(e => e.classList[1])") == \
            [claimed, verified]
        assert visible(pg, f"{tr} td:nth-child(7)") == str(s["steps"])
        assert visible(pg, f"{tr} .agent-tag") == "mock"
        assert visible(pg, f"{tr} .task-id") == run.task_id
        cells = pg.eval_on_selector_all(f"{tr} td.p50", """els => els.map(e => { const c = e.cloneNode(true);
            c.querySelectorAll('.visually-hidden').forEach(n => n.remove()); return c.textContent.replace(/\\u00a0/g, ' '); })""")
        for stage, shown in zip(("voice", "summary", "plan", "ground", "act"), cells):
            v = s["stage_p50_ms"][stage]
            assert shown == "—" if v is None else close_to(shown, v), (rid, stage, shown, v)
    pg.screenshot(path=str(runs / "logs-list.png"))
    watch.assert_clean()


def test_event_logs_voice_and_verdict_rows(page, mock_console, ui_runs):
    pg, watch = page
    base, runs = mock_console
    for run in ui_runs:
        rid = run.agent_run_id
        open_logs(pg, base, rid)
        evs = lines_of(runs / rid / "events.jsonl")
        doc = json.loads((runs / rid / "console.json").read_text())
        is_voice = run.label == "voice"
        kinds = pg.eval_on_selector_all("#event-list li.ev", "els => els.map(e => e.dataset.kind)")
        sources = pg.eval_on_selector_all("#event-list li.ev", "els => els.map(e => e.dataset.source)")
        # every events.jsonl line, in file order, between the console's voice row (voice runs) and verdict row
        assert kinds == (["voice"] if is_voice else []) + [e["kind"] for e in evs] + ["verdict"], (run.label, kinds)
        assert sources == (["console"] if is_voice else []) + ["agent"] * len(evs) + ["console"]
        assert visible(pg, "#events-meta") == f"{len(kinds)} events"
        first_agent = 1 if is_voice else 0
        for i, e in enumerate(evs):  # step numbers and the t column (ms since the first event) per row
            row = ev_sel(first_agent + i)
            assert visible(pg, f"{row} .ev-step") == str(e["step"])
            shown_t = visible(pg, f"{row} .ev-t")
            assert shown_t.startswith("+") and abs(float(shown_t[1:-3].replace(",", "")) - (e["ts"] - evs[0]["ts"]) * 1000) <= 1

        if is_voice:  # the voice row: speech model, its latency and what it heard; no time or step (not guessed)
            v = doc["voice"]
            assert (visible(pg, f"{ev_sel(0)} .ev-model"), visible(pg, f"{ev_sel(0)} .ev-src")) == ("stub", "console")
            assert close_to(visible(pg, f"{ev_sel(0)} .ev-lat"), v["latency_ms"])
            assert visible(pg, f"{ev_sel(0)} .ev-desc") == f"Heard: “{HEARD}”"
            assert (visible(pg, f"{ev_sel(0)} .ev-t"), visible(pg, f"{ev_sel(0)} .ev-step")) == ("—", "—")
            pg.focus(f"{ev_sel(0)} .ev-row")
            pg.keyboard.press("Enter")
            det = text_of(pg, f"{ev_sel(0)} .ev-details")
            assert "stub" in det and HEARD in det
            assert any(f"{d:.2f} s" in det for d in (v["duration_s"], v["duration_s"] + 5e-4)), det  # toFixed(2)
            assert "never the audio" in det

        # the verdict row: what the console recorded in console.json (and showed in its banner)
        last = len(kinds) - 1
        verdict = doc["verdict"]
        desc = visible(pg, f"{ev_sel(last)} .ev-desc")
        # the banner's "Verified: the task is done" / "Stopped: the task is not done", after its lead word
        assert run.banner_title.split(": ", 1)[-1] in desc and verdict["explanation"] in desc, (desc, run.banner_title)
        assert desc.startswith("✓ Verified" if verdict["passed"] else "✗ Not verified")
        pg.focus(f"{ev_sel(0)} .ev-row")
        pg.keyboard.press("End")
        pg.keyboard.press("Enter")
        det = text_of(pg, f"{ev_sel(last)} .ev-details")
        assert VERDICT_LABEL in det and verdict["explanation"] in det and run.task_id in det
        badges = visible(pg, "#run-badges")
        assert "mock" in badges and ("Stopped" if run.label == "stopped" else "Done") in badges
        assert ("Not verified" if run.label == "stopped" else "✓ Verified") in badges

        # mock runs say so, and claim no model work: no plan / summary / ground latency on any row
        for i, e in enumerate(evs):
            if e["kind"] in ("plan", "observation", "grounding"):
                assert visible(pg, f"{ev_sel(first_agent + i)} .ev-lat") == "—"
        assert "scripted replay" in text_of(pg, "#run-summary")
        if run.label == "stopped":
            assert visible(pg, f"{ev_sel(last - 1)} .ev-desc").startswith("Run ended: stopped by the user")
        pg.screenshot(path=str(runs / f"logs-{run.label}.png"), full_page=True)
    watch.assert_clean()


def test_copy_json_and_download_are_identical(page, mock_console, ui_runs):
    pg, watch = page
    base, runs = mock_console
    run = next(r for r in ui_runs if r.label == "voice")
    rid = run.agent_run_id
    open_logs(pg, base)
    # keyboard only: Tab into the runs table, arrow to the run, Enter opens it
    for _ in range(10):
        pg.keyboard.press("Tab")
        if pg.evaluate("document.activeElement.classList.contains('run-link')"):
            break
    while pg.evaluate("document.activeElement.dataset.runId") != rid:
        pg.keyboard.press("ArrowDown")
    pg.keyboard.press("Enter")
    pg.wait_for_function(f"document.querySelector('#run-heading').textContent === 'Run {rid}'")

    pg.focus("#copy-btn")
    pg.keyboard.press("Enter")
    pg.wait_for_function("document.querySelector('#copy-status').textContent.startsWith('Copied')")
    assert pg.get_attribute("#copy-status", "aria-live") == "polite"
    copied = pg.evaluate("navigator.clipboard.readText()")

    pg.focus("#download-link")
    with pg.expect_download() as dl:
        pg.keyboard.press("Enter")
    assert dl.value.suggested_filename == f"run-{rid}.json"
    downloaded = Path(dl.value.path()).read_bytes()
    assert copied.encode("utf-8") == downloaded  # byte-identical, not just the same JSON
    server = httpx.get(f"{base}/api/logs/runs/{rid}/download")
    assert server.headers["content-disposition"] == f'attachment; filename="run-{rid}.json"'
    assert server.content == downloaded

    doc = json.loads(copied)
    lines = lines_of(runs / rid / "events.jsonl")
    assert len(doc["events"]) == len(lines) + 2  # + the console's voice and verdict
    assert [e["source"] for e in doc["events"]] == ["console"] + ["agent"] * len(lines) + ["console"]
    for line, e in zip(lines, doc["events"][1:-1]):
        assert {k: e[k] for k in line} == line  # every line as written, plus the derived fields
    assert doc["console"] == json.loads((runs / rid / "console.json").read_text())
    watch.assert_clean()


# --- 2) the real agent's traces -----------------------------------------------------------------------------------


def test_real_traces_render_every_model_step(page, real_console):
    pg, watch = page
    base, runs = real_console
    evs = lines_of(runs / DONE_RUN / "events.jsonl")
    kinds = [e["kind"] for e in evs]
    open_logs(pg, base, DONE_RUN)
    assert pg.eval_on_selector_all("#runs-body .agent-tag", "els => els.map(e => e.textContent)") == ["real", "real"]
    assert pg.eval_on_selector_all("#event-list li.ev", "els => els.map(e => e.dataset.kind)") == kinds  # no console rows
    assert "No verdict recorded" in visible(pg, "#run-badges")  # no console.json: no verdict, and none invented
    expand_all(pg)

    # plan rows: Lightning's model (from the run config: the payload has none), tokens, thinking, tool-call args
    for i in [i for i, k in enumerate(kinds) if k == "plan"]:
        p = evs[i]["payload"]
        u = p["usage"]
        assert visible(pg, f"{ev_sel(i)} .ev-model") == "lightning"
        assert visible(pg, f"{ev_sel(i)} .ev-desc").startswith("Lightning chose: ")
        assert close_to(visible(pg, f"{ev_sel(i)} .ev-lat"), evs[i]["timings_ms"]["plan"])
        det = text_of(pg, f"{ev_sel(i)} .ev-details")
        assert "lightning (from the run config)" in det
        assert (f"{u['prompt_tokens']:,} prompt · {u['completion_tokens']} completion · 0 reasoning · "
                f"{u['total_tokens']:,} total") in det
        assert "off for this call" in det and "Reasoning: none (thinking was off)." in det
        call = p["tool_calls"][0]
        args = details_block(pg, i, f"{call['name']} arguments (JSON)")
        assert args and args["open"] and json.loads(args["text"]) == call["arguments"]

    # observation rows: the raw screen summary, open, exactly as the VLM wrote it; summary model from the config
    for i in [i for i, k in enumerate(kinds) if k == "observation"]:
        block = details_block(pg, i, "The screen summary Lightning read")
        assert block and block["open"] and block["text"] == evs[i]["payload"]["observation"]["screen_summary"]
        assert visible(pg, f"{ev_sel(i)} .ev-model") == "base"
        assert close_to(visible(pg, f"{ev_sel(i)} .ev-lat"), evs[i]["timings_ms"]["summary"])
        assert "base (from the run config)" in text_of(pg, f"{ev_sel(i)} .ev-details")
        assert pg.get_attribute(f"{ev_sel(i)} .thumb-btn", "data-src") == f"/runs/{DONE_RUN}/step_0{evs[i]['step']}.png"

    # the grounding row: raw output, the 0-1000 point, pixels, and the overlay the grounder saw
    gi = kinds.index("grounding")
    g = evs[gi]["payload"]
    det = text_of(pg, f"{ev_sel(gi)} .ev-details")
    assert "base (from this event)" in det and "935, 677" in det and "1196.8, 541.6" in det
    raw = details_block(pg, gi, "Raw model output")
    assert raw and raw["open"] and raw["text"] == g["raw"]
    thumb = f"{ev_sel(gi)} .thumb-btn"
    assert pg.get_attribute(thumb, "data-src") == f"/runs/{DONE_RUN}/step_01_overlay.png"
    pg.locator(thumb).scroll_into_view_if_needed()
    pg.wait_for_function(f"document.querySelector(\"{thumb} img\").naturalWidth === 1280")
    assert visible(pg, f"{ev_sel(gi)} .ev-desc") == f"Grounder placed “{g['target']}” at (1197, 542)"

    # the click action: x, y as clicked
    ai = kinds.index("action")
    assert visible(pg, f"{ev_sel(ai)} .ev-desc") == "Clicked at (1197, 542) · screen changed"
    assert "click · click at 1197, 542" in text_of(pg, f"{ev_sel(ai)} .ev-details")
    for i, k in enumerate(kinds):  # no model behind these: "—"
        if k in ("start", "tool_call", "action", "done"):
            assert visible(pg, f"{ev_sel(i)} .ev-model") == "—"

    # nothing about a real run says "mock" (text, hidden details, attributes)
    pg.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
    hits = pg.evaluate("""() => [...document.body.textContent.matchAll(/.{0,30}mock.{0,30}/gi)].map(m => m[0])
        .concat([...document.querySelectorAll('*')].flatMap(e => [...e.attributes]
          .filter(a => /mock/i.test(a.value)).map(a => `${e.tagName} ${a.name}=${a.value}`)))""")
    assert hits == [], hits
    pg.evaluate("document.querySelectorAll('details').forEach(d => d.open = false)")
    pg.screenshot(path=str(runs / "real-done.png"), full_page=True)

    # the stopped run: the plan is there, the click never happened ("—" for ground/act), no verdict
    tr = f"#runs-body tr[data-run-id='{STOPPED_RUN}']"
    assert visible(pg, f"{tr} .status-chip") == "Stopped"
    assert pg.eval_on_selector_all(f"{tr} td.c-check .tick", "els => els.map(e => e.classList[1])") == \
        ["tick-no", "tick-none"]
    p50 = pg.eval_on_selector_all(f"{tr} td.p50", "els => els.map(e => e.firstChild && e.firstChild.nodeType === 3"
                                                  " ? e.textContent.replace(/\\u00a0/g, ' ') : '—')")
    assert p50[0] == "—" and close_to(p50[1], 1571.1) and close_to(p50[2], 2864.6) and p50[3:] == ["—", "—"], p50
    open_logs(pg, base, STOPPED_RUN)
    skinds = pg.eval_on_selector_all("#event-list li.ev", "els => els.map(e => e.dataset.kind)")
    assert skinds == [e["kind"] for e in lines_of(runs / STOPPED_RUN / "events.jsonl")]
    assert visible(pg, f"{ev_sel(len(skinds) - 1)} .ev-desc").startswith("Run ended: stopped by the user")
    watch.assert_clean()


@pytest.mark.parametrize("theme", ["light", "hc"])
def test_real_trace_contrast_in_both_themes(page, real_console, theme):
    pg, watch = page
    base, runs = real_console
    pg.goto(f"{base}/logs")
    pg.evaluate(f"localStorage.setItem('gui-assist.high-contrast', '{'1' if theme == 'hc' else '0'}')")
    open_logs(pg, base, DONE_RUN)
    assert pg.evaluate("document.documentElement.classList.contains('hc')") is (theme == "hc")
    expand_all(pg)
    pg.evaluate("document.activeElement.blur()")
    res = pg.evaluate(CONTRAST_JS)
    assert res["n"] > 200 and res["low"] == [], res["low"]
    pg.screenshot(path=str(runs / f"real-{theme}.png"), full_page=True)
    watch.assert_clean()


def test_overflowing_numbers_in_a_trace_become_null(console, tmp_path):
    """1e999 parses to inf, which strict JSON can't carry: the detail and the download still answer (null)."""
    runs = tmp_path / "runs"
    (runs / "r-inf").mkdir(parents=True)
    (runs / "r-inf" / "events.jsonl").write_text(
        '{"run_id": "r-inf", "step": 0, "kind": "start", "ts": 1000.0, "payload": {"request": "x"}}\n'
        '{"run_id": "r-inf", "step": 1, "kind": "plan", "ts": 1001.0, "payload": {"wait_ms": 1e999},'
        ' "timings_ms": {"plan": -1e999}}\n')
    base, _ = console(None, runs_dir=runs)
    for path in ("", "/download"):
        r = httpx.get(f"{base}/api/logs/runs/r-inf{path}")
        assert r.status_code == 200, r.text
        plan = r.json()["events"][1]
        assert plan["payload"]["wait_ms"] is None and plan["timings_ms"]["plan"] is None
    assert httpx.get(f"{base}/api/logs/runs/r-inf").content == httpx.get(f"{base}/api/logs/runs/r-inf/download").content


def test_a_watched_live_run_announces_its_end(page, console, tmp_path):
    """The 3 s auto-refresh of a running run: its end is said once, in the one live region (status, verdict)."""
    pg, watch = page
    base, _ = console(None, runs_dir=tmp_path / "runs", static_dir=STATIC, step_delay_s=0.4)
    r = httpx.post(f"{base}/api/run", json={"request": "Turn on high contrast.", "task_id": "high-contrast"})
    assert r.status_code == 200, r.text
    deadline, aid = time.time() + 20, None
    while time.time() < deadline and not aid:
        aid = httpx.get(f"{base}/api/runs/{r.json()['run_id']}").json()["agent_run_id"]
        time.sleep(0.1)
    assert aid
    open_logs(pg, base, aid)
    assert "Running" in visible(pg, "#run-badges") and "refreshing every 3 s" in visible(pg, "#refresh-note")
    assert pg.get_attribute("#sr-status", "aria-live") == "polite"
    pg.wait_for_function("document.querySelector('#sr-status').textContent.includes('finished')", timeout=40000)
    said = visible(pg, "#sr-status")
    n = pg.locator("#event-list li.ev").count()
    assert said == f"Run {aid} finished: Done, verified, {n} events.", said
    assert pg.get_attribute("#events-meta", "aria-live") is None  # one live region for this, not two
    watch.assert_clean()
