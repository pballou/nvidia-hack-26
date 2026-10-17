"""Voice input in a real browser: headless Chromium with a fake microphone (Chromium's beeping test device).

What runs for real: the page's capture (getUserMedia -> AudioWorklet -> wav.js), the 16 kHz mono PCM16 WAV upload,
the console backend's POST /api/transcribe with the stub engine (console/asr.py), the mock agent and the sandbox
verdict. Other transcripts, errors and a missing recognizer are route-mocked, so these tests don't depend on a real
ASR model. Every test checks that nothing leaves 127.0.0.1 and that the page logs no console errors (HTTP errors
the test itself provokes excepted). Marked `e2e`; about 20 s.
"""

from __future__ import annotations

import io
import json
import time
import wave
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import pytest

from console import asr
from console.app import CONSOLE_DIR, ConsoleSettings, create_app
from console.tests.conftest import ServerThread

pytestmark = pytest.mark.e2e

STATIC = CONSOLE_DIR / "static"
VIEWPORT = {"width": 1280, "height": 800}
STUB_TEXT = "turn on high contrast"  # no capital, no period: the task match ignores case and punctuation
TASK, TASK_REQUEST = "high-contrast", "Turn on high contrast."
FAKE_MIC = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]

# Keeps every getUserMedia stream and AudioContext, so a test can check the mic was released after each recording.
SPY = """(() => {
  const md = navigator.mediaDevices, gum = md.getUserMedia.bind(md), AC = window.AudioContext;
  window.__streams = []; window.__contexts = [];
  md.getUserMedia = async (c) => { window.__gumArgs = c; const s = await gum(c); window.__streams.push(s); return s; };
  window.AudioContext = function (...a) { const c = new AC(...a); window.__contexts.push(c); return c; };
  window.AudioContext.prototype = AC.prototype;
})();"""

STATE_JS = """() => {
  const q = (s) => document.querySelector(s);
  return {
    mic: q('#mic-btn').dataset.state, pressed: q('#mic-btn').getAttribute('aria-pressed'),
    label: q('#mic-label').textContent, disabled: q('#mic-btn').disabled,
    status: q('#voice-status').textContent, tone: q('#voice-status').dataset.tone || '',
    hint: q('#voice-hint').textContent,
    countdown: q('#voice-countdown').hidden ? null : q('#voice-countdown-text').textContent,
    heard: q('#heard').hidden ? null : q('#heard').textContent,
    request: q('#request').value, task: q('#task-select').value, run: q('#run-status').dataset.state,
    active: document.activeElement ? (document.activeElement.id || document.activeElement.tagName) : null,
  };
}"""


def _console(sandbox_url, runs, **kw) -> ServerThread:
    settings = ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, headed=False,
                               static_dir=STATIC, stop_timeout_s=10.0, **kw)
    return ServerThread(create_app(settings))


@pytest.fixture(scope="module")
def stub_asr():
    asr.set_engine(asr.StubEngine(STUB_TEXT))
    yield
    asr.set_engine(None)  # back to the configured engine, loaded lazily


@pytest.fixture(scope="module")
def fast_console(sandbox_url, tmp_path_factory, stub_asr):
    srv = _console(sandbox_url, tmp_path_factory.mktemp("voice-runs"), step_delay_s=0.0)
    yield srv.start()
    srv.stop()


@pytest.fixture(scope="module")
def slow_console(sandbox_url, tmp_path_factory, stub_asr):
    """1 s per step, so a run is still going while the test speaks."""
    srv = _console(sandbox_url, tmp_path_factory.mktemp("voice-slow-runs"), step_delay_s=1.0)
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
class Voice:
    """The page plus what it sent: WAV uploads, /api/run bodies, off-host requests, console errors.

    `reply` decides /api/transcribe: None = the real backend (stub engine); ("json", body), ("status", code, error),
    ("abort",) or ("hold",) = keep the request pending until the test calls release()."""
    pg: object
    wavs: list = field(default_factory=list)
    content_types: list = field(default_factory=list)
    run_bodies: list = field(default_factory=list)
    offhost: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    reply: tuple | None = None
    held: list = field(default_factory=list)

    def state(self) -> dict:
        return self.pg.evaluate(STATE_JS)

    def wait(self, js: str, timeout: float = 10) -> None:
        self.pg.wait_for_function(js, timeout=timeout * 1000)

    def wait_mic(self, state: str, timeout: float = 10) -> None:
        self.wait(f"document.querySelector('#mic-btn').dataset.state === {json.dumps(state)}", timeout)

    def wait_status(self, text: str, timeout: float = 10) -> None:
        self.wait(f"document.querySelector('#voice-status').textContent.includes({json.dumps(text)})", timeout)

    def hold(self, key: str, seconds: float) -> None:
        self.pg.keyboard.down(key)
        self.wait_mic("listening")
        time.sleep(seconds)
        self.pg.keyboard.up(key)

    def say_text(self, text: str) -> None:
        self.reply = ("json", {"text": text, "duration_s": 0.6, "latency_ms": 12.3, "model": "test"})

    def release(self, body: dict) -> None:
        for r in self.held:
            r.fulfill(status=200, content_type="application/json", body=json.dumps(body))
        self.held.clear()

    def assert_clean(self, allow_http_errors: bool = False) -> None:
        assert not self.offhost, f"requests left 127.0.0.1: {self.offhost}"
        errors = [e for e in self.errors if not (allow_http_errors and "Failed to load resource" in e)]
        assert not errors, errors


@pytest.fixture
def voice(browser):
    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="en-US")
    ctx.add_init_script(SPY)
    v = Voice(pg=None)

    def route(r):
        url = urlsplit(r.request.url)
        if url.hostname != "127.0.0.1":
            v.offhost.append(r.request.url)
            return r.abort("blockedbyclient")
        if r.request.method == "POST" and url.path == "/api/run":
            v.run_bodies.append(json.loads(r.request.post_data or "{}"))
        if r.request.method == "POST" and url.path == "/api/transcribe":
            v.wavs.append(r.request.post_data_buffer or b"")
            v.content_types.append(r.request.headers.get("content-type"))
            kind = v.reply[0] if v.reply else None
            if kind == "json":
                return r.fulfill(status=200, content_type="application/json", body=json.dumps(v.reply[1]))
            if kind == "status":
                return r.fulfill(status=v.reply[1], content_type="application/json",
                                 body=json.dumps({"error": v.reply[2]}))
            if kind == "abort":
                return r.abort("connectionrefused")
            if kind == "hold":
                v.held.append(r)
                return None
        return r.continue_()

    ctx.route("**/*", route)
    v.pg = ctx.new_page()
    v.pg.on("console", lambda m: v.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    v.pg.on("pageerror", lambda e: v.errors.append(f"pageerror: {e}"))
    yield v
    ctx.close()


def open_console(v: Voice, base: str) -> None:
    v.pg.goto(base + "/")
    v.wait("document.querySelector('#task-select').options.length > 12")
    v.wait("document.querySelector('#voice-hint').textContent !== 'Checking voice input…'")


def wav_format(data: bytes) -> tuple[int, int, int, float]:
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE" and data[36:40] == b"data"  # 44-byte header
    with wave.open(io.BytesIO(data)) as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes() / w.getframerate()


# --- tests ----------------------------------------------------------------------------------------------


def test_push_to_talk_fills_selects_and_runs_a_sandbox_task(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    s = v.state()
    assert s["mic"] == "idle" and s["label"] == "Speak" and s["pressed"] == "false" and not s["disabled"]
    assert s["hint"] == "Or hold Right Option to talk, release to send."

    # hold Right Option in the request box: listening, with a live level from the fake mic's beeps
    v.reply = ("hold",)
    v.pg.focus("#request")
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    s = v.state()
    assert s["pressed"] == "true" and s["label"] == "Stop listening" and s["status"] == "Listening… release to send"
    v.wait("parseFloat(document.querySelector('#mic-btn').style.getPropertyValue('--level')) > 0.05")
    time.sleep(0.8)
    v.pg.keyboard.up("AltRight")
    v.wait_mic("transcribing")
    s = v.state()
    assert s["label"] == "Transcribing…" and s["status"] == "Transcribing…" and s["request"] == ""  # Option typed nothing
    assert v.pg.get_attribute("#mic-btn", "aria-disabled") == "true"

    # the upload: 16 kHz mono PCM16 WAV, raw body, Content-Type audio/wav; the mic is released
    assert len(v.wavs) == 1 and v.content_types == ["audio/wav"]
    channels, width, rate, seconds = wav_format(v.wavs[0])
    assert (channels, width, rate) == (1, 2, 16000) and 0.6 < seconds < 2.0, (channels, width, rate, seconds)
    spy = v.pg.evaluate("""() => ({tracks: window.__streams.flatMap(s => s.getTracks().map(t => t.readyState)),
                                    contexts: window.__contexts.map(c => c.state), args: window.__gumArgs})""")
    assert spy["tracks"] and set(spy["tracks"]) == {"ended"}, spy
    assert spy["contexts"] and set(spy["contexts"]) == {"closed"}, spy
    assert spy["args"] == {"audio": {"channelCount": 1, "echoCancellation": True, "noiseSuppression": True}}

    # send the held upload on to the real backend (stub engine): the task is matched and selected
    req = v.held.pop()
    v.reply = None
    req.continue_()
    v.wait_mic("done")
    s = v.state()
    assert s["request"] == TASK_REQUEST and s["task"] == TASK, s
    assert s["heard"].startswith("Heard (stub, ") and s["heard"].endswith(f" ms): {STUB_TEXT}"), s
    assert s["status"].startswith(f"Heard: “{TASK_REQUEST}” (a sandbox task)"), s
    assert s["countdown"].startswith("Running in ") and s["countdown"].endswith(" s — press Esc to cancel"), s
    assert v.pg.is_visible("#voice-cancel")

    # 1.5 s later it runs by itself; the heard line stays above the run's steps; the sandbox verdict shows
    v.wait("document.querySelector('#run-status').dataset.state !== 'idle'", timeout=5)
    assert v.state()["countdown"] is None
    v.pg.wait_for_selector("#banner:not([hidden])", timeout=60000)
    assert v.pg.get_attribute("#banner", "data-tone") == "pass"
    assert v.run_bodies == [{"request": TASK_REQUEST, "task_id": TASK,
                             "options": {"grounder": "base", "thinking": "retry", "lightning": "local"}}]
    s = v.state()
    assert s["heard"].startswith("Heard (stub, ") and s["status"] == ""  # old status cleared for the step log
    log = v.pg.inner_text(".log")
    assert log.index("Heard (stub, ") < log.index("Started at /")
    v.assert_clean()


def test_mic_button_by_keyboard_and_space_only_off_controls(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    # Tab order: skip link, high contrast, the mic, the request box (the mic sits left of the box)
    for _ in range(3):
        v.pg.keyboard.press("Tab")
    assert v.state()["active"] == "mic-btn"
    v.pg.keyboard.press("Tab")
    assert v.state()["active"] == "request"

    # Space types in the request box; Space on a button or a radio keeps its meaning (no recording)
    v.pg.keyboard.type("a b")
    assert v.state()["request"] == "a b" and v.state()["mic"] == "idle"
    v.pg.fill("#request", "")
    v.pg.focus("#hc-toggle")
    v.pg.keyboard.press("Space")
    assert v.pg.evaluate("document.documentElement.classList.contains('hc')") and v.state()["mic"] == "idle"
    v.pg.keyboard.press("Space")
    v.pg.focus("#thinking-retry")
    v.pg.keyboard.down("Space")
    time.sleep(0.3)
    assert v.state()["mic"] == "idle"
    v.pg.keyboard.up("Space")
    assert not v.wavs

    # the Speak button toggles: Space starts, Enter stops; an empty transcript is "Didn't catch that"
    v.say_text("")
    v.pg.focus("#mic-btn")
    v.pg.keyboard.press("Space")
    v.wait_mic("listening")
    assert v.state()["status"] == "Listening… press again to send"
    time.sleep(0.6)
    v.pg.keyboard.press("Enter")
    v.wait_status("Didn’t catch that")
    s = v.state()
    assert s["mic"] == "idle" and s["countdown"] is None and s["heard"] is None and s["active"] == "mic-btn", s

    # Space held with focus on the page itself is push-to-talk
    v.say_text("Please water my plants")
    v.pg.evaluate("document.activeElement.blur()")
    v.hold("Space", 0.6)
    v.wait_mic("done")
    s = v.state()
    assert s["request"] == "Please water my plants" and s["task"] == "" and s["countdown"], s  # free text
    assert s["heard"] == "Heard (stub, 12 ms): Please water my plants", s  # the config's asr.label, rounded ms
    v.pg.click("#voice-cancel")  # Cancel: no run, focus to the request box
    s = v.state()
    assert s["countdown"] is None and s["status"].startswith("Auto-run cancelled") and s["active"] == "request", s
    time.sleep(1.8)
    assert not v.run_bodies and v.state()["run"] == "idle"
    assert len(v.wavs) == 2
    v.assert_clean()


def test_short_taps_repeats_chords_escape_and_errors(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    v.pg.focus("#request")
    # a tap shorter than 0.25 s is not sent
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    v.pg.keyboard.up("AltRight")
    v.wait_status("Too short")
    assert not v.wavs

    # auto-repeat keydowns don't restart; one recording, one upload. Esc cancels the countdown, no run
    v.say_text("TURN ON HIGH CONTRAST!")
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    for _ in range(4):
        v.pg.keyboard.down("AltRight")  # Playwright marks these repeat=true
        time.sleep(0.1)
    time.sleep(0.2)
    v.pg.keyboard.up("AltRight")
    v.wait_mic("done")
    assert len(v.wavs) == 1 and v.state()["task"] == TASK
    v.pg.keyboard.press("Escape")
    s = v.state()
    assert s["countdown"] is None and s["status"].startswith("Auto-run cancelled"), s
    time.sleep(1.8)
    assert not v.run_bodies

    # Right Option used with another key is a shortcut, not speech; Esc while listening cancels
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    v.pg.keyboard.press("KeyE")
    v.wait_mic("idle")
    v.pg.keyboard.up("AltRight")
    assert "used with another key" in v.state()["status"]
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    v.pg.keyboard.press("Escape")
    v.wait_mic("idle")
    v.pg.keyboard.up("AltRight")
    assert v.state()["status"].startswith("Cancelled") and len(v.wavs) == 1

    # backend errors are shown under the mic (the live region) and never break the page
    for reply, want in [
        (("status", 400, "Recording is too short"), "Couldn’t use that recording: Recording is too short."),
        (("status", 413, "recording is too large (max 60 s)"), "That recording is too long: recording is too large"),
        (("status", 503, "ASR engine 'parakeet' failed to load"), "Speech recognition is unavailable: ASR engine"),
        (("abort",), "Couldn’t reach the console backend to transcribe."),
    ]:
        v.reply = reply
        v.hold("AltRight", 0.4)
        v.wait_mic("error")
        s = v.state()
        assert s["status"].startswith(want) and s["tone"] == "error" and s["countdown"] is None, (s, want)
    v.reply = None  # and the next try works
    v.hold("AltRight", 0.4)
    v.wait_mic("done")
    v.pg.keyboard.press("Escape")
    assert not v.run_bodies
    v.assert_clean(allow_http_errors=True)


def test_speaking_during_a_run(voice, slow_console):
    v = voice
    open_console(v, slow_console)
    v.pg.select_option("#task-select", TASK)
    v.pg.click("#run-btn")
    v.wait("[...document.querySelectorAll('#step-log .step-action')].some(e => e.textContent.startsWith('Clicking: '))",
           timeout=30)

    # anything but stop/cancel only fills the box (a task's text still selects it); no second run
    v.say_text("mark all my messages as read")
    v.hold("AltRight", 0.4)
    v.wait_mic("done")
    s = v.state()
    assert s["request"] == "Mark all my messages as read." and s["task"] == "mark-all-read" and s["run"] == "running"
    assert s["status"].startswith('A run is in progress — say "stop" to stop it.') and s["countdown"] is None, s

    # "Stop." calls the Stop action
    v.say_text(" Stop. ")
    v.hold("AltRight", 0.4)
    v.wait_status("Stopping")
    v.pg.wait_for_selector("#banner:not([hidden])", timeout=30000)
    assert v.pg.get_attribute("#banner", "data-tone") == "stopped"
    assert "You stopped the run." in v.pg.inner_text("#banner-claim")
    assert v.state()["heard"] == "Heard (stub, 12 ms): Stop."
    assert [b["task_id"] for b in v.run_bodies] == [TASK]
    v.assert_clean()


def test_no_recognizer_disables_the_mic(voice, fast_console):
    v = voice

    def patch(r):
        resp = r.fetch()
        cfg = resp.json()
        cfg["asr"] = {"available": False, "model": "parakeet", "label": "Parakeet",
                      "error": "ASR engine 'parakeet' failed to load: No module named 'nemo'"}
        r.fulfill(response=resp, body=json.dumps(cfg), content_type="application/json")

    v.pg.route("**/api/config", patch)
    open_console(v, fast_console)
    s = v.state()
    assert s["disabled"] and s["mic"] == "off", s
    assert s["hint"] == ("Voice input is off: the speech recognizer (Parakeet) isn’t available: "
                         "ASR engine 'parakeet' failed to load: No module named 'nemo'.")
    v.pg.focus("#request")
    v.pg.keyboard.down("AltRight")
    time.sleep(0.3)
    v.pg.keyboard.up("AltRight")
    assert v.state()["mic"] == "off" and not v.wavs
    assert v.pg.is_hidden("#kbd-ptt-wrap")
    v.assert_clean()


def test_push_to_talk_key_setting(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    msg = "document.querySelector('#ptt-msg').textContent"
    assert v.pg.text_content("#ptt-key") == "Right Option" and v.pg.is_hidden("#ptt-reset")
    assert "The Fn key can’t be used" in v.pg.text_content("#ptt-note")
    v.pg.click("#ptt-change")
    assert v.pg.text_content("#ptt-change") == "Press a key…"
    v.pg.keyboard.press("Escape")  # cancels
    v.wait(f"{msg}.startsWith('Kept Right Option')")
    v.pg.keyboard.press("Enter")  # focus is still on Change
    v.pg.keyboard.press("Enter")  # Enter can't be the key
    v.wait(f"{msg}.startsWith('Enter runs the request')")
    v.pg.keyboard.press("KeyQ")
    v.wait("document.querySelector('#ptt-key').textContent === 'Q'")
    assert v.pg.evaluate("localStorage.getItem('gui-assist.ptt-key')") == "KeyQ"
    assert v.pg.text_content("#voice-hint") == "Or hold Q to talk, release to send."
    assert v.pg.text_content("#kbd-ptt") == "Q"

    # a typing key types in fields and records elsewhere; Right Option no longer records
    v.pg.focus("#request")
    v.pg.keyboard.press("KeyQ")
    v.pg.keyboard.down("AltRight")
    time.sleep(0.3)
    v.pg.keyboard.up("AltRight")
    s = v.state()
    assert s["request"] == "q" and s["mic"] == "idle", s
    v.pg.fill("#request", "")
    v.pg.evaluate("document.activeElement.blur()")
    v.say_text("")
    v.hold("KeyQ", 0.4)
    v.wait_status("Didn’t catch that")

    # it survives a reload; Reset brings back Right Option
    v.pg.reload()
    open_console(v, fast_console)
    assert v.pg.text_content("#ptt-key") == "Q"
    v.pg.click("#ptt-reset")
    assert v.pg.text_content("#ptt-key") == "Right Option" and v.pg.is_hidden("#ptt-reset")
    assert v.pg.evaluate("localStorage.getItem('gui-assist.ptt-key')") is None
    assert v.state()["active"] == "ptt-change"
    v.assert_clean()
