"""Matching a request to a sandbox task (console.js createTaskMatcher), and what the console records about it.

A spoken "Pay fifty dollars towards my bill." has to run as the pay-50 task ("Pay $50 toward my bill."), or the run
gets no sandbox verdict. The matcher is console-side and fuzzy but conservative: a near miss ("Pay my bill",
"Download my August statement", "Turn off high contrast", "Report an outage in Hillcrest", "Turn on SMS for billing")
must stay free text, because a wrong task gives a wrong verdict.

- The table: headless Chromium loads the real page from a live console (fake agent) and evaluates
  window.AcmeConsole.matchTaskDetailed over all 12 task texts (method "exact"), natural spoken variants of each
  (TASK_VARIANTS) and near misses (NEAR_MISSES, null). Python's request_key (console/app.py) must call exactly the
  same texts exact. Run with -s to print the table with scores and runner-ups.
- The page: typed text is matched when the box loses focus or on Run, voice text at once (AcmeConsole.setRequest);
  the picker shows the match and Free text undoes it; the agent gets the user's own words for a fuzzy match while
  task_id sets the start page and the verdict; the banner says "Checked as task: <id> (matched from your request)"
  or "(picked from the list)" above the verdict; the Logs page shows the same line.
- The API: POST /api/run task_match validation (400s), console.json and GET /api/runs/{id} task_match, an exact match
  recorded by the backend when the page leaves it out, RunSummary task_match / task_line.

DOM selectors are fine here (tests only). The browser tests carry the e2e marker; skip them with -m "not e2e".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from console.app import CONSOLE_DIR, ConsoleSettings, create_app, parse_task_match, request_key
from console.runner import VERDICT_LABEL
from console.tests.conftest import ServerThread, messages, read_sse
from console.tests.test_console_api import fake_agent

STATIC = CONSOLE_DIR / "static"
TASKS = json.loads((CONSOLE_DIR.parent / "sandbox" / "tasks.json").read_text())
TASKS_BY_ID = {t["id"]: t for t in TASKS}
VIEWPORT = {"width": 1280, "height": 800}
LOCAL_HOST = "127.0.0.1"
FREE_TEXT_LABEL = "Free-text request: no sandbox check"

# How people say each task (what speech-to-text gives back: spelled-out numbers, "towards", filler, synonyms).
TASK_VARIANTS = {
    "text-size-large": [
        "please make the text bigger i can't read this", "Make the text bigger.",
        "can you make the font larger, I can't read this", "Um, make the text big please, I cannot read this.",
    ],
    "download-september-statement": [
        "download the september statement", "Download my statement for September.",
        "could you download my September bill", "download my sept 2026 statement",
    ],
    "mark-all-read": [
        "mark all messages read", "Mark all of my messages as read, please.", "mark every message as read",
        "mark everything in my inbox as read",
    ],
    "turn-on-autopay": [
        "turn autopay on", "Turn on auto pay.", "I keep forgetting to pay, can you turn on autopay", "enable autopay",
        "Turn on autopay so I don't forget to pay",
    ],
    "high-contrast": [
        "Can you turn on high contrast?",  # polite question = a request
        "switch on high contrast", "turn high contrast on", "Please turn on high-contrast mode.", "enable high contrast",
    ],
    "sms-billing-off": [
        "turn off text messages for billing alerts", "Turn off SMS for my billing alerts.",
        "stop texting me about my bill", "disable sms billing alerts",
    ],
    "max-usage-month": [
        "which month did i use the most power open it", "Which month did I use the most electricity? Open it.",
        "open the month with the highest usage", "show me the month where I used the most energy",
    ],
    "report-outage-riverside": [
        "report an outage in riverside", "Report a power outage in the Riverside district.",
        "I want to report a blackout in the riverside area",
    ],
    "star-landlord-message": [
        "star the landlord's message", "Star the message from my landlord.", "favorite the landlord message",
    ],
    "pay-50": [
        "Could you pay fifty dollars toward my bill?",  # polite question = a request
        "Allocate fifty dollars towards my bill.", "Put $50 toward my bill.", "Apply 50 dollars to my bill",  # money verbs + amount
        "pay fifty dollars towards my bill", "Pay fifty dollars towards my bill.", "pay 50 dollars toward my bill",
        "Pay $50 toward my bill please", "make a payment of fifty dollars toward my bill",
        "can you pay fifty bucks toward my bill",
    ],
    "message-support-meter": [
        "send support a message that my meter is broken",
        "Send a message to customer support saying my meter is broken.",
        "message support that my meter isn't working", "tell support my meter is broken",
    ],
    "stop-promotions": [
        "stop all the promotions, no emails texts or push", "Stop sending me promotions.",
        "unsubscribe me from all promotions", "no more promotional emails, texts or push notifications",
    ],
}

# Close to a task, but not that task: each must stay free text (no task, so no verdict that would be wrong).
NEAR_MISSES = [
    # opposite intent and questions (orchestrator review, 2026-10-18): a verdict here would be wrong
    "Cancel my outage report in Riverside", "Remove the star from the landlord's message",
    "Undo the $50 payment toward my bill", "Delete my September statement download",
    "Should I pay $50 toward my bill?", "How do I pay $50 toward my bill?", "Is high contrast on?",
    "Did I report an outage in Riverside?", "Is autopay turned on?",
    "Pay my bill",                                    # no amount
    "Pay $60 toward my bill.", "Pay fifteen dollars toward my bill.", "Pay $500 toward my bill.",  # another amount
    "Allocate $60 toward my bill.", "Put the text on large",  # money verb, wrong amount / no amount
    "Pay $50 toward my bill twice.", "Pay $50 toward my bill every month",
    "Download my August statement",                   # another month
    "Download my September and August statements", "Open my September statement",
    "Turn off high contrast",                         # the opposite
    "Don't turn on high contrast", "Turn on low contrast",
    "Report an outage in Hillcrest",                  # another district
    "Report an outage in Riverside and Hillcrest", "Is there an outage in Riverside?",
    "Turn on SMS for billing",                        # the opposite
    "Turn off email for billing alerts", "Turn off SMS for outage alerts", "Turn off SMS and email for billing alerts",
    "Make the text smaller", "Mark all messages as unread", "Mark the landlord's message as read",
    "Turn off autopay", "I keep forgetting to pay", "Star the message from support", "Unstar the landlord's message",
    "Send support a message that my meter is fine", "Send the landlord a message that my meter is broken",
    "My meter is broken", "Stop sending me promotional emails", "Turn on promotions",
    "Which month did I use the least power?", "Open my July usage",
    # two requests in one
    "Make the text bigger and turn on high contrast", "Pay $50 toward my bill and turn on autopay",
    "Star the landlord message and mark all as read", "Turn on high contrast and reduce motion",
    # nothing like a task
    "Please water my plants", "stop", "turn it on", "hello",
]

MATCH_JS = """(texts) => texts.map((t) => {
  const m = window.AcmeConsole.matchTaskDetailed(t);
  const e = window.AcmeConsole.explainMatch(t);
  return {text: t, id: m ? m.task.id : null, method: m ? m.method : null, score: m ? m.score : null,
          runner_up: m ? m.runner_up : null, best: e.tasks[0], legacy: (window.AcmeConsole.matchTask(t) || {}).id || null};
})"""


def _console(sandbox_url: str, runs: Path, fake) -> ServerThread:
    settings = ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, static_dir=STATIC,
                               stop_timeout_s=5.0)
    return ServerThread(create_app(settings, agent=fake))


def recording(fake, seen: list | None = None):
    """(run_task, make_cfg) of test_console_api's fake agent that also writes <trace_root>/<run_id>/events.jsonl, as
    the agents do (the Logs page reads it), and records what the agent was given (request, start_url) in `seen`."""
    run_task, make_cfg = fake

    async def wrapped(request, *, start_url, cfg, on_event=None):
        if seen is not None:
            seen.append({"request": request, "start_url": start_url})

        async def logged(ev):
            with (Path(cfg["trace_root"]) / ev["run_id"] / "events.jsonl").open("a") as f:
                f.write(json.dumps(ev) + "\n")
            await on_event(ev)

        return await run_task(request, start_url=start_url, cfg=cfg, on_event=logged)

    return wrapped, make_cfg


@pytest.fixture(scope="module")
def match_console(sandbox_url, tmp_path_factory):
    """(console URL, runs dir, seen): a fake agent that sets the text size to L (text-size-large passes) and records
    what it was given. The mock agent can't serve here: it only has scripts for the tasks' exact texts."""
    runs = tmp_path_factory.mktemp("match-runs")
    seen: list = []
    srv = _console(sandbox_url, runs, recording(fake_agent(steps=1, set_text_size="L", trace_root=runs), seen))
    yield srv.start(), runs, seen
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
    requests: list[str] = field(default_factory=list)
    offhost: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    run_bodies: list[dict] = field(default_factory=list)

    def assert_clean(self) -> None:
        assert not self.offhost, f"requests left {LOCAL_HOST}: {self.offhost}"
        assert not self.errors, self.errors


@pytest.fixture
def page(browser, match_console):
    watch = Watch()
    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="en-US")

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
    pg.goto(match_console[0] + "/")
    pg.wait_for_function("document.querySelector('#task-select').options.length > 12")
    yield pg, watch
    ctx.close()


def state(pg) -> dict:
    return pg.evaluate("""() => ({request: document.querySelector('#request').value,
        task: document.querySelector('#task-select').value, note: document.querySelector('#request-note').textContent})""")


def run_and_wait(pg, start, timeout_s: float = 30) -> None:
    """start() a run, then wait for that run's banner (not the previous run's, still up until the POST returns)."""
    prev = pg.evaluate("sessionStorage.getItem('gui-assist.run')")
    start()
    pg.wait_for_function("(prev) => sessionStorage.getItem('gui-assist.run') !== prev"
                         " && !document.querySelector('#banner').hidden", arg=prev, timeout=timeout_s * 1000)


def run_info(base: str, pg) -> dict:
    """GET /api/runs/{id} of the run the page last started (it keeps the id in sessionStorage)."""
    run_id = pg.evaluate("sessionStorage.getItem('gui-assist.run')")
    return httpx.get(f"{base}/api/runs/{run_id}").json()


# --- the table ---------------------------------------------------------------------------------


def _fmt_runner_up(r: dict | None) -> str:
    if not r:
        return "-"
    out = f"{r['id']} {r['score']:.2f}"
    if r.get("ruled_out"):
        out += f" (sim {r['similarity']:.2f}; {r['ruled_out']})"
    return out


def print_table(rows: list[dict], want: dict[str, str | None]) -> None:
    print(f"\n{'expected':<29} {'got':<29} {'method':<6} {'score':>5}  runner-up | text")
    for r in rows:
        score = f"{r['score']:.2f}" if r["score"] is not None else "-"
        if r["id"] is None:  # free text: the closest task and what ruled it out
            b = r["best"]
            ru = f"closest {b['id']} {b['score']:.2f}" + (f" ({b['why'][0]})" if b["why"] else " (below threshold/margin)")
        else:
            ru = _fmt_runner_up(r["runner_up"])
        print(f"{str(want[r['text']]):<29} {str(r['id']):<29} {str(r['method'] or '-'):<6} {score:>5}  {ru} | {r['text']}")


@pytest.mark.e2e
def test_match_table(page):
    pg, watch = page
    want: dict[str, str | None] = {t["request"]: t["id"] for t in TASKS}
    for tid, texts in TASK_VARIANTS.items():
        assert len(texts) >= 3, tid
        for text in texts:
            want[text] = tid
    assert len(NEAR_MISSES) >= 15
    for text in NEAR_MISSES:
        want[text] = None
    assert set(TASK_VARIANTS) == set(TASKS_BY_ID)

    rows = pg.evaluate(MATCH_JS, list(want))
    print_table(rows, want)
    wrong = [(r["text"], want[r["text"]], r["id"], r["best"]) for r in rows if r["id"] != want[r["text"]]]
    assert not wrong, wrong

    for r in rows:
        exact_py = any(request_key(r["text"]) == request_key(t["request"]) for t in TASKS)
        assert (r["method"] == "exact") == exact_py, r  # console.js requestKey == console/app.py request_key
        assert r["legacy"] == r["id"], r  # matchTask(text) is matchTaskDetailed(text).task
        if r["id"] is None:
            continue
        assert r["method"] in ("exact", "fuzzy") and 0.5 <= r["score"] <= 1, r
        ru = r["runner_up"]
        assert ru is None or (ru["id"] != r["id"] and r["score"] - ru["score"] >= 0.15), r  # a clear margin
    for t in TASKS:  # every task's own text is an exact match
        (r,) = [x for x in rows if x["text"] == t["request"]]
        assert (r["id"], r["method"], r["score"]) == (t["id"], "exact", 1), r
    # the moment that went missing on stage
    (r,) = [x for x in rows if x["text"] == "Pay fifty dollars towards my bill."]
    assert (r["id"], r["method"], r["score"]) == ("pay-50", "fuzzy", 1), r
    assert pg.evaluate("window.AcmeConsole.matchTaskDetailed('')") is None
    watch.assert_clean()


# --- the page ----------------------------------------------------------------------------------


@pytest.mark.e2e
def test_typed_fuzzy_match_runs_your_words_as_the_task(page, match_console):
    pg, watch = page
    base, runs, seen = match_console
    said = "please make the text bigger i can't read this"
    pg.focus("#request")
    pg.keyboard.type(said)
    s = state(pg)
    assert s["task"] == "" and s["note"].startswith("This matches sandbox task text-size-large"), s  # typing: a hint

    pg.keyboard.press("Tab")  # leaving the box: the picker shows the match (and Free text would undo it)
    s = state(pg)
    assert s["task"] == "text-size-large" and s["request"] == said, s
    assert s["note"].startswith("Will be checked as task text-size-large (matched from your request; the agent gets "
                                "your words as they are)"), s
    assert "Easy" in pg.inner_text("#task-meta")

    pg.focus("#request")
    run_and_wait(pg, lambda: pg.keyboard.press("Enter"))
    body = watch.run_bodies[-1]
    assert body["request"] == said and body["task_id"] == "text-size-large", body  # nothing swapped in
    assert body["task_match"] == {"method": "fuzzy", "score": 1, "matched_from": said}, body
    assert seen[-1]["request"] == said  # the agent got the user's words...
    start = urlsplit(seen[-1]["start_url"])
    assert start.path == "/settings" and "tab=accessibility" in start.query  # ...at the task's start page
    assert state(pg)["request"] == said

    # the banner: the line above the verdict, then the verdict checked against the portal's state
    assert pg.get_attribute("#banner", "data-tone") == "pass"
    assert pg.inner_text("#banner-title") == "Verified: the task is done"
    assert pg.inner_text("#banner-task") == "Checked as task: text-size-large (matched from your request)"
    assert pg.text_content("#banner-label") == VERDICT_LABEL
    ys = pg.evaluate("""() => ['#banner-title', '#banner-task', '#banner-label', '#banner-expl']
        .map((s) => document.querySelector(s).getBoundingClientRect().top)""")
    assert ys == sorted(ys) and len(set(ys)) == 4, ys
    assert (pg.get_attribute("#banner-task", "title") or "").startswith("Close match (score 1.00)")

    info = run_info(base, pg)
    assert info["task_match"] == {"method": "fuzzy", "score": 1.0, "matched_from": said}
    assert info["verdict"]["passed"] is True

    # the Logs page says the same
    pg.click("#banner-logs-link")
    pg.wait_for_selector("#run-summary .task-line")
    summary = pg.inner_text("#run-summary")
    assert "Checked as task: text-size-large (matched from your request)" in summary
    assert f"close match, score 1.00 · from “{said}”" in summary
    watch.assert_clean()


@pytest.mark.e2e
def test_voice_match_shows_in_the_picker_and_free_text_undoes_it(page, match_console):
    pg, watch = page
    base, _, seen = match_console
    said = "Pay fifty dollars towards my bill."
    out = pg.evaluate("(t) => window.AcmeConsole.setRequest(t)", said)  # what voice.js does with a transcript
    assert out == {"taskId": "pay-50", "request": said}
    s = state(pg)
    assert s["task"] == "pay-50" and s["request"] == said, s  # matched, your words kept
    assert s["note"].startswith("Will be checked as task pay-50 (matched from your request"), s

    # run it as matched: the banner names the task; the verdict comes from the portal (the fake paid nothing)
    run_and_wait(pg, lambda: pg.evaluate("window.AcmeConsole.run()"))
    assert watch.run_bodies[-1]["task_id"] == "pay-50" and watch.run_bodies[-1]["request"] == said
    assert watch.run_bodies[-1]["task_match"]["method"] == "fuzzy"
    assert seen[-1]["request"] == said
    assert pg.inner_text("#banner-task") == "Checked as task: pay-50 (matched from your request)"
    assert pg.inner_text("#banner-title") == "Not verified: the task is not done"

    # again, but undo the match with Free text: the words stay, no task, no check; small edits don't re-match
    pg.evaluate("(t) => window.AcmeConsole.setRequest(t)", said)
    pg.select_option("#task-select", "")
    s = state(pg)
    assert s == {"request": said, "task": "", "note": "Runs as free text: no sandbox check."}, s
    pg.focus("#request")
    pg.keyboard.press("End")
    pg.keyboard.type(" please")
    pg.keyboard.press("Tab")
    assert state(pg)["task"] == ""
    pg.focus("#request")
    run_and_wait(pg, lambda: pg.keyboard.press("Enter"))
    body = watch.run_bodies[-1]
    assert body["task_id"] is None and "task_match" not in body and body["request"] == f"{said} please", body
    assert pg.text_content("#banner-label") == FREE_TEXT_LABEL and pg.locator("#banner-task").is_hidden()

    # a near miss never selects a task
    for text in ("Pay my bill", "Turn off high contrast"):
        out = pg.evaluate("(t) => window.AcmeConsole.setRequest(t)", text)
        assert out == {"taskId": None, "request": text} and state(pg)["task"] == ""
    watch.assert_clean()


@pytest.mark.e2e
def test_picked_task_says_so(page, match_console):
    pg, watch = page
    base, _, _ = match_console
    pg.select_option("#task-select", "text-size-large")
    run_and_wait(pg, lambda: pg.click("#run-btn"))
    request = TASKS_BY_ID["text-size-large"]["request"]
    assert watch.run_bodies[-1]["task_match"] == {"method": "picked", "score": None, "matched_from": request}
    assert pg.inner_text("#banner-task") == "Checked as task: text-size-large (picked from the list)"
    assert run_info(base, pg)["task_match"] == {"method": "picked", "score": None, "matched_from": request}

    # after a reload the page shows that run again, with the same line
    pg.reload()
    pg.wait_for_selector("#banner:not([hidden])")
    assert pg.inner_text("#banner-task") == "Checked as task: text-size-large (picked from the list)"
    watch.assert_clean()


# --- the API -----------------------------------------------------------------------------------

GOOD = {"method": "fuzzy", "score": 0.8, "matched_from": "pay fifty dollars towards my bill"}


@pytest.mark.parametrize("match, error", [
    ("fuzzy", "task_match must be an object"),
    ([], "task_match must be an object"),
    ({**GOOD, "why": "x"}, "unknown task_match field(s): why"),
    ({"method": "fuzzy", "score": 0.8}, "task_match needs matched_from"),
    ({"method": "fuzzy"}, "task_match needs score, matched_from"),
    ({**GOOD, "method": "guessed"}, "task_match.method must be one of picked, exact, fuzzy"),
    ({**GOOD, "score": 1.5}, "task_match.score must be a number from 0 to 1, or null"),
    ({**GOOD, "score": -0.1}, "task_match.score must be a number from 0 to 1, or null"),
    ({**GOOD, "score": "0.8"}, "task_match.score must be a number from 0 to 1, or null"),
    ({**GOOD, "score": True}, "task_match.score must be a number from 0 to 1, or null"),
    ({**GOOD, "score": float("nan")}, "task_match.score must be a number from 0 to 1, or null"),
    ({**GOOD, "matched_from": 3}, "task_match.matched_from must be a string of at most 500 characters"),
    ({**GOOD, "matched_from": "x" * 501}, "task_match.matched_from must be a string of at most 500 characters"),
])
def test_bad_task_match_400(console, tmp_path, match, error):
    base, _ = console(fake_agent(trace_root=tmp_path))
    body = {"request": "pay fifty dollars towards my bill", "task_id": "pay-50", "task_match": match}
    r = httpx.post(f"{base}/api/run", content=json.dumps(body), headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json() == {"error": error}, r.text


def test_task_match_needs_a_task_id(console, tmp_path):
    base, _ = console(fake_agent(trace_root=tmp_path))
    r = httpx.post(f"{base}/api/run", json={"request": "hi", "task_match": GOOD})
    assert r.status_code == 400 and r.json() == {"error": "task_match needs a task_id"}


def test_parse_task_match():
    pay = TASKS_BY_ID["pay-50"]
    assert parse_task_match(GOOD, "pay-50", "x", TASKS_BY_ID) == (GOOD, None)
    picked = {"method": "picked", "score": None, "matched_from": pay["request"]}
    assert parse_task_match(picked, "pay-50", pay["request"], TASKS_BY_ID) == (picked, None)
    assert parse_task_match({**GOOD, "score": 1}, "pay-50", "x", TASKS_BY_ID)[0]["score"] == 1.0
    assert parse_task_match({**GOOD, "matched_from": ""}, "pay-50", "x", TASKS_BY_ID)[1] is None
    # left out: an exact match is recorded as one (the page doesn't send those); anything else records nothing
    said = "pay fifty dollars toward my bill"
    assert parse_task_match(None, "pay-50", said, TASKS_BY_ID) == (
        {"method": "exact", "score": 1.0, "matched_from": said}, None)
    assert parse_task_match(None, "pay-50", "pay fifty dollars towards my bill", TASKS_BY_ID) == (None, None)
    assert parse_task_match(None, None, said, TASKS_BY_ID) == (None, None)
    assert parse_task_match(None, "pay-50", said, {"pay-50": {}}) == (None, None)  # a task without its text


def test_request_key_matches_the_page():
    assert request_key("Pay $50 toward my bill.") == request_key("pay fifty dollars toward my bill") == "pay50towardmybill"
    assert request_key("PAY $50.00 TOWARD MY BILL") == "pay50towardmybill"
    assert request_key("Pay fifty-five dollars") == "pay55"
    assert request_key("two hundred five") == "205"
    assert request_key("Café ½") == "cafe12"
    assert request_key("Pay $500 toward my bill.") != request_key("Pay $50 toward my bill.")


def _run_and_read(base: str, runs: Path, body: dict) -> tuple[dict, dict, dict]:
    """POST /api/run, wait for the end; (console.json, GET /api/runs/{id}, the Logs RunSummary)."""
    r = httpx.post(f"{base}/api/run", json=body)
    assert r.status_code == 200, r.text
    run = r.json()
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    rid = next(m["data"]["run_id"] for m in stream if m["event"] == "result")
    doc = json.loads((runs / rid / "console.json").read_text())
    info = httpx.get(f"{base}/api/runs/{run['run_id']}").json()
    summary = httpx.get(f"{base}/api/logs/runs/{rid}").json()["summary"]
    listed = next(s for s in httpx.get(f"{base}/api/logs/runs").json()["runs"] if s["run_id"] == rid)
    assert listed["task_match"] == summary["task_match"] and listed["task_line"] == summary["task_line"]
    return doc, info, summary


def test_task_match_in_console_json_runs_and_logs(console, tmp_path):
    runs = tmp_path / "runs"
    base, _ = console(recording(fake_agent(steps=1, set_text_size="L", trace_root=runs)), runs_dir=runs)
    said = "please make the text bigger i can't read this"

    # fuzzy, as the page sends it
    match = {"method": "fuzzy", "score": 1, "matched_from": said}
    doc, info, s = _run_and_read(base, runs, {"request": said, "task_id": "text-size-large", "task_match": match})
    want = {"method": "fuzzy", "score": 1.0, "matched_from": said}
    assert doc["task_match"] == info["task_match"] == s["task_match"] == want
    assert doc["request"] == said and doc["verdict"]["passed"] is True
    assert s["task_line"] == "Checked as task: text-size-large (matched from your request)"

    # picked
    text = TASKS_BY_ID["text-size-large"]["request"]
    match = {"method": "picked", "score": None, "matched_from": text}
    doc, info, s = _run_and_read(base, runs, {"request": text, "task_id": "text-size-large", "task_match": match})
    assert doc["task_match"] == info["task_match"] == s["task_match"] == match
    assert s["task_line"] == "Checked as task: text-size-large (picked from the list)"

    # left out with the task's own words (an exact match: the page doesn't send it): the backend records it
    said = "make the text bigger, i can't read this"
    doc, info, s = _run_and_read(base, runs, {"request": said, "task_id": "text-size-large"})
    assert doc["task_match"] == {"method": "exact", "score": 1.0, "matched_from": said}
    assert s["task_line"] == "Checked as task: text-size-large (matched from your request)"

    # left out with other words: nothing is claimed about how the task was chosen
    doc, info, s = _run_and_read(base, runs, {"request": "Do the thing.", "task_id": "text-size-large"})
    assert doc["task_match"] is None and info["task_match"] is None and s["task_match"] is None
    assert s["task_line"] == "Checked as task: text-size-large"

    # free text: no task, no line
    doc, info, s = _run_and_read(base, runs, {"request": "Please water my plants"})
    assert doc["task_id"] is None and doc["task_match"] is None and s["task_line"] is None


def test_task_line_from_console_json():
    from console import logs

    fuzzy = {"method": "fuzzy", "score": 0.83, "matched_from": "x"}
    s = logs.summarize_run([], {"task_id": "pay-50", "task_match": fuzzy, "verdict": {"passed": False}})
    assert (s["task_match"], s["task_line"]) == (fuzzy, "Checked as task: pay-50 (matched from your request)")
    s = logs.summarize_run([], {"task_id": "pay-50", "task_match": {"method": "exact", "score": 1}})  # no verdict yet
    assert s["task_line"] == "Sandbox task: pay-50 (matched from your request)"
    assert s["task_match"] == {"method": "exact", "score": 1.0, "matched_from": None}
    s = logs.summarize_run([], {"task_id": "pay-50", "task_match": {"method": "magic"}, "verdict": {"passed": True}})
    assert (s["task_match"], s["task_line"]) == (None, "Checked as task: pay-50")  # unknown method: not trusted
    s = logs.summarize_run([], {"task_id": None, "task_match": fuzzy})
    assert (s["task_match"], s["task_line"]) == (None, None)
    assert logs.summarize_run([], None)["task_line"] is None
