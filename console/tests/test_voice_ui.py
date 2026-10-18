"""Voice input in a real browser: headless Chromium with a fake microphone that plays a spoken phrase.

The fake microphone (--use-fake-device-for-media-stream with --use-file-for-fake-audio-capture) loops "Pay fifty
dollars toward my bill", made with macOS `say` + `afconvert` and trimmed of leading and trailing silence (elsewhere:
a synthetic voiced signal with no silence), so every slice of the recording should hold sound.

What runs for real: the page's capture (getUserMedia -> AudioWorklet -> wav.js), the warm microphone with its
pre-roll, the 16 kHz mono PCM16 WAV upload, the console backend's POST /api/transcribe with the stub engine
(console/asr.py), the mock agent and the sandbox verdict. Other transcripts, errors and a missing recognizer are
route-mocked, so these tests don't depend on a real ASR model. Every test checks that nothing leaves 127.0.0.1 and
that the page logs no console errors (HTTP errors the test itself provokes excepted). Marked `e2e`; about 70 s.
"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import subprocess
import time
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
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
PHRASE = "Pay fifty dollars toward my bill"
WARM_TEXT = "Microphone ready (releases after 5 min idle)"
COLD_PTT = "Listening… start speaking now, release to send"  # the first press: nothing was buffered before it

# Keeps every getUserMedia stream and AudioContext, so a test can check the mic was released after each recording.
SPY = """(() => {
  const md = navigator.mediaDevices, gum = md.getUserMedia.bind(md), AC = window.AudioContext;
  window.__streams = []; window.__contexts = []; window.__gumCalls = 0;
  md.getUserMedia = async (c) => {
    window.__gumCalls += 1; window.__gumArgs = c; const s = await gum(c); window.__streams.push(s); return s;
  };
  window.AudioContext = function (...a) { const c = new AC(...a); window.__contexts.push(c); return c; };
  window.AudioContext.prototype = AC.prototype;
  // every change of the mic button's state or the status line, with the frames the gate (if any) let through
  window.__log = [];
  document.addEventListener('DOMContentLoaded', () => {
    const mic = document.getElementById('mic-btn'), status = document.getElementById('voice-status');
    const rec = () => {
      const e = { state: mic.dataset.state, status: status.textContent, frames: window.__delivered ?? null,
                  bg: getComputedStyle(mic).backgroundColor };
      const last = window.__log[window.__log.length - 1];
      if (!last || last.state !== e.state || last.status !== e.status) window.__log.push(e);
    };
    new MutationObserver(rec).observe(status, { childList: true, characterData: true, subtree: true });
    new MutationObserver(rec).observe(mic, { attributes: true, attributeFilter: ['data-state'] });
  });
})();"""

SPY_STATE = """() => ({tracks: window.__streams.flatMap(s => s.getTracks().map(t => t.readyState)),
                       contexts: window.__contexts.map(c => c.state), args: window.__gumArgs,
                       gum: window.__gumCalls})"""

# Holds the worklet's audio messages ('preroll', 'chunk') while window.__gate is true and lets them through, in order,
# on window.__openGate(): the page sees a microphone that is open but whose first frames are late.
FRAME_GATE = """(() => {
  const AWN = window.AudioWorkletNode;
  const desc = Object.getOwnPropertyDescriptor(MessagePort.prototype, 'onmessage');
  window.__gate = true; window.__queued = []; window.__delivered = 0;
  window.__openGate = () => {
    window.__gate = false; const q = window.__queued; window.__queued = []; q.forEach((f) => f());
  };
  window.AudioWorkletNode = function (...a) {
    const node = new AWN(...a);
    const port = node.port;
    Object.defineProperty(port, 'onmessage', {
      configurable: true,
      get() { return desc.get.call(port); },
      set(h) {
        desc.set.call(port, h && ((e) => {
          const t = e.data && e.data.type;
          if (t !== 'chunk' && t !== 'preroll') return h(e);
          const go = () => { if (e.data.samples.length) window.__delivered += 1; h(e); };
          if (window.__gate) window.__queued.push(go); else go();
        }));
      },
    });
    return node;
  };
  window.AudioWorkletNode.prototype = AWN.prototype;
})();"""

# navigator.permissions.query('microphone') answers a status the test controls: window.__perm('denied').
FAKE_PERMISSION = """(() => {
  const status = { state: 'granted', onchange: null };
  window.__perm = (s) => { status.state = s; if (status.onchange) status.onchange(); };
  const q = navigator.permissions.query.bind(navigator.permissions);
  navigator.permissions.query = async (d) => (d && d.name === 'microphone' ? status : q(d));
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


def shorten_gaps(pcm: array, rate: int, quiet: int = 500, keep_ms: int = 20) -> array:
    """Cut every quiet stretch (10 ms windows under `quiet` RMS) down to `keep_ms`, so any 150 ms of the loop holds
    speech. `say` leaves up to ~120 ms between words, and Chromium's noise suppression makes those near-silent."""
    n = rate // 100
    out, run = array("h"), 0
    for i in range(0, len(pcm) - n + 1, n):
        w = pcm[i:i + n]
        run = run + 1 if math.sqrt(sum(x * x for x in w) / n) < quiet else 0
        if run * 10 <= keep_ms:
            out.extend(w)
    return out


def make_speech_wav(path: Path) -> str:
    """Write the fake microphone's audio: PHRASE spoken by macOS `say`, 48 kHz mono PCM16, leading and trailing
    silence cut and gaps shortened (Chromium loops the file, so it never goes quiet for long). Without
    `say`/`afconvert`: a voiced synthetic signal (a 140 Hz buzz with a slow swell) with no silence at all.
    Returns "say" or "synthetic"."""
    rate, pcm, how = 48000, None, "synthetic"
    say, afconvert = shutil.which("say"), shutil.which("afconvert")
    if say and afconvert:
        aiff, raw = path.with_suffix(".aiff"), path.with_name(path.stem + "-raw.wav")
        try:
            subprocess.run([say, "-o", str(aiff), PHRASE], check=True, timeout=60, capture_output=True)
            subprocess.run([afconvert, "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1", str(aiff), str(raw)],
                           check=True, timeout=60, capture_output=True)
            with wave.open(str(raw)) as w:
                rate, pcm = w.getframerate(), array("h", w.readframes(w.getnframes()))
            loud = [i for i, s in enumerate(pcm) if abs(s) > 1000]
            pcm = shorten_gaps(pcm[loud[0]:loud[-1] + 1], rate) if loud else None
            how = "say"
        except (OSError, subprocess.SubprocessError, wave.Error):
            pcm = None
    if not pcm:
        rate, how = 48000, "synthetic"
        pcm = array("h", [int(7000 * (0.7 + 0.3 * math.sin(2 * math.pi * 3 * i / rate))
                              * sum(math.sin(2 * math.pi * 140 * k * i / rate) / k for k in (1, 2, 3, 5)) / 2)
                          for i in range(2 * rate)])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return how


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fake-mic") / "phrase.wav"
    make_speech_wav(path)
    return path


@pytest.fixture(scope="module")
def browser(speech_wav):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(args=FAKE_MIC + [f"--use-file-for-fake-audio-capture={speech_wav}"])
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

    def warm(self) -> str | None:
        """The warm-microphone line's text, or None while it's hidden."""
        return self.pg.evaluate("""() => {
          const p = document.getElementById('mic-warm');
          return p && !p.hidden ? p.querySelector('#mic-warm-text').textContent : null;
        }""")

    def announced(self) -> str:
        return self.pg.text_content("#mic-warm-status")

    def last_take(self) -> dict:
        return self.pg.evaluate("window.AcmeVoice.lastTake()")

    def wait_released(self, timeout: float = 5) -> None:
        """Every getUserMedia track stopped and every AudioContext closed."""
        self.wait("""window.__streams.length > 0
                     && window.__streams.every(s => s.getTracks().every(t => t.readyState === 'ended'))
                     && window.__contexts.every(c => c.state === 'closed')""", timeout)

    def log(self) -> list[dict]:
        return self.pg.evaluate("window.__log")

    def wait_uploads(self, n: int, timeout: float = 10) -> None:
        deadline = time.monotonic() + timeout
        while len(self.wavs) < n:
            assert time.monotonic() < deadline, f"{len(self.wavs)} uploads, wanted {n}"
            self.pg.wait_for_timeout(50)  # lets Playwright run the route handler

    def bg(self) -> str:
        return self.pg.evaluate("getComputedStyle(document.querySelector('#mic-btn')).backgroundColor")

    def release(self, body: dict) -> None:
        for r in self.held:
            try:
                r.fulfill(status=200, content_type="application/json", body=json.dumps(body))
            except Exception:  # the page aborted it (a cancelled transcription)
                pass
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
    assert s["mic"] == "idle" and s["label"] == "Speak" and s["pressed"] is None and not s["disabled"]
    assert s["hint"] == "Or hold Right Option to talk, release to send."

    # hold Right Option in the request box: listening, with a live level from the fake mic's beeps
    v.reply = ("hold",)
    v.pg.focus("#request")
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    s = v.state()
    assert s["pressed"] is None and s["label"] == "Stop listening" and s["status"] == COLD_PTT, s
    v.wait("parseFloat(document.querySelector('#mic-btn').style.getPropertyValue('--level')) > 0.05")
    time.sleep(0.8)
    v.pg.keyboard.up("AltRight")
    v.wait_mic("transcribing")
    s = v.state()
    assert s["label"] == "Transcribing…" and s["status"] == "Transcribing…" and s["request"] == ""  # Option typed nothing
    # the button cancels the transcription now, and its name says so
    assert v.pg.get_attribute("#mic-btn", "aria-disabled") is None
    assert v.pg.get_attribute("#mic-btn", "aria-label") == "Transcribing… press to cancel"

    # the upload: 16 kHz mono PCM16 WAV, raw body, Content-Type audio/wav; the mic stays warm, and says so
    assert len(v.wavs) == 1 and v.content_types == ["audio/wav"]
    channels, width, rate, seconds = wav_format(v.wavs[0])
    assert (channels, width, rate) == (1, 2, 16000) and 0.6 < seconds < 2.0, (channels, width, rate, seconds)
    spy = v.pg.evaluate(SPY_STATE)
    assert spy["tracks"] == ["live"] and spy["contexts"] == ["running"], spy
    assert spy["args"] == {"audio": {"channelCount": 1, "echoCancellation": True, "noiseSuppression": True}}
    assert v.warm() == WARM_TEXT

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
    assert len(v.run_bodies) == 1
    body = dict(v.run_bodies[0])
    voice = body.pop("voice")  # what was heard, for the logs page (console.json only; the agent never gets it)
    assert body == {"request": TASK_REQUEST, "task_id": TASK,
                    "options": {"grounder": "grounder", "thinking": "retry", "lightning": "local"}}
    assert voice["text"] == STUB_TEXT and voice["model"] == "stub" and voice["latency_ms"] >= 0, voice
    assert 0.6 < voice["duration_s"] < 2.0, voice
    s = v.state()
    assert s["heard"].startswith("Heard (stub, ") and s["status"] == ""  # old status cleared for the step log
    log = v.pg.inner_text(".log")
    assert log.index("Heard (stub, ") < log.index("Started at /")
    # Turn off releases the microphone: every track stopped, the context closed
    v.pg.click("#mic-off")
    assert v.pg.evaluate(SPY_STATE)["tracks"] == ["ended"] and v.warm() is None  # at once
    v.wait_released()
    v.assert_clean()


def test_mic_button_by_keyboard_and_space_only_off_controls(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    # Tab order: skip link, Logs, high contrast, the mic, the request box (the mic sits left of the box)
    for _ in range(4):
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
    assert v.state()["status"] == "Listening… start speaking now, press again to send"  # the first: cold
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
                         "ASR engine 'parakeet' failed to load: No module named 'nemo'. "
                         "This checks again every few seconds and turns voice input on by itself.")
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


def patch_asr(v: Voice, asr_info: dict | None, calls: list | None = None):
    """Route /api/config through the live backend with its "asr" replaced (None = left alone). Returns a setter."""
    current = {"asr": asr_info}

    def patch(r):
        resp = r.fetch()
        cfg = resp.json()
        if calls is not None:
            calls.append(time.time())
        if current["asr"] is not None:
            cfg["asr"] = current["asr"]
        r.fulfill(response=resp, body=json.dumps(cfg), content_type="application/json")

    v.pg.route("**/api/config", patch)
    return lambda value: current.update(asr=value)


def test_recognizer_coming_up_turns_voice_on_by_itself(voice, fast_console):
    v = voice
    calls: list = []
    set_asr = patch_asr(v, {"available": False, "model": "parakeet", "label": "Parakeet TDT 0.6B v3",
                            "error": "Parakeet server at http://127.0.0.1:8096 not ready yet (HTTP 503)"}, calls)
    open_console(v, fast_console)
    s = v.state()
    assert s["mic"] == "off" and s["disabled"], s
    assert s["hint"] == ("Voice input is off: the speech recognizer (Parakeet TDT 0.6B v3) isn’t available: "
                         "Parakeet server at http://127.0.0.1:8096 not ready yet (HTTP 503). "
                         "This checks again every few seconds and turns voice input on by itself.")
    set_asr(None)  # the sidecar finished loading: the backend's own /api/config now says available
    v.wait_mic("idle", timeout=8)
    s = v.state()
    assert not s["disabled"] and s["status"] == "Voice input is available again." and s["tone"] == "ok", s
    assert s["hint"] == "Or hold Right Option to talk, release to send."
    assert len(calls) == 2, calls  # the page load, then one check about 4 s later
    assert 3.5 < calls[1] - calls[0] < 6, calls
    time.sleep(4.5)
    assert len(calls) == 2, calls  # voice is on: no more checks
    v.assert_clean()


def test_explicit_stub_is_labelled_and_never_runs_by_itself(voice, fast_console):
    """`--asr stub` (/api/config asr.fixed_text): the page says it's fixed text, and never auto-runs it."""
    v = voice
    patch_asr(v, {"available": True, "model": "stub", "label": "stub (fixed text)", "fixed_text": True})
    open_console(v, fast_console)
    s = v.state()
    assert s["mic"] == "idle" and s["hint"] == ("Voice input is a stub: it fills in fixed text, whatever you say, "
                                                "and never runs it by itself. Hold Right Option to try it."), s
    assert v.pg.text_content("#asr-note").startswith("Voice input is the stub (fixed text) engine: it recognizes no")
    v.pg.evaluate("document.activeElement.blur()")
    v.hold("AltRight", 0.4)
    v.wait_mic("done")
    s = v.state()
    assert s["request"] == TASK_REQUEST and s["task"] == TASK and s["countdown"] is None, s
    assert re.fullmatch(rf"Stub \(fixed text, \d+ ms\): {STUB_TEXT}", s["heard"]), s
    assert s["status"] == (f"The stub’s fixed text (not what you said), “{STUB_TEXT}” (a sandbox task) is in the "
                           "request box. The stub never runs by itself: press Enter in the request box to run it."), s
    time.sleep(2.0)  # past any countdown
    assert not v.run_bodies and v.state()["run"] == "idle"
    v.assert_clean()


def test_stop_phrases(voice, slow_console):
    v = voice
    open_console(v, slow_console)
    v.pg.evaluate("document.activeElement.blur()")

    # with nothing running, a stop phrase starts nothing
    for phrase in ["Stop it.", "Cancel that.", "Please stop.", "stop stop", "Stop, stop!", "STOP!", "stop the run"]:
        v.say_text(phrase)
        v.hold("AltRight", 0.3)
        v.wait_status(f"Heard: “{phrase}”. Nothing is running, so there’s nothing to stop.")
        s = v.state()
        assert s["countdown"] is None and s["request"] == "", (phrase, s)
    time.sleep(1.8)  # past a 1.5 s countdown
    assert not v.run_bodies

    # a request that only starts with "stop" is a request
    v.say_text("Stop sending me promotions. No emails, no texts, no push notifications.")
    v.hold("AltRight", 0.3)
    v.wait_mic("done")
    s = v.state()
    assert s["task"] == "stop-promotions" and s["countdown"], s
    v.pg.keyboard.press("Escape")
    assert v.state()["countdown"] is None and not v.run_bodies

    # during a run, "Stop it." and "Cancel that." stop it
    for phrase in ["Stop it.", "Cancel that."]:
        v.pg.select_option("#task-select", TASK)
        v.pg.click("#run-btn")
        v.wait("[...document.querySelectorAll('#step-log .step-action')]"
               ".some(e => e.textContent.startsWith('Clicking: '))", timeout=30)
        v.say_text(phrase)
        v.pg.evaluate("document.activeElement.blur()")
        v.hold("AltRight", 0.3)
        v.wait_status("Stopping")
        v.pg.wait_for_selector("#banner:not([hidden])", timeout=30000)
        assert v.pg.get_attribute("#banner", "data-tone") == "stopped", phrase
        assert v.state()["request"] == TASK_REQUEST, phrase  # the stop phrase didn't touch the box
    assert [b["task_id"] for b in v.run_bodies] == [TASK, TASK]
    v.assert_clean()


def test_slow_transcription_can_be_cancelled_and_never_replaces_typing(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    late = {"text": STUB_TEXT, "duration_s": 0.6, "latency_ms": 5000.0, "model": "test"}

    # Esc cancels a transcription that is taking long; its late reply changes nothing
    v.reply = ("hold",)
    v.pg.evaluate("document.activeElement.blur()")
    v.hold("AltRight", 0.4)
    v.wait_mic("transcribing")
    v.pg.keyboard.press("Escape")
    s = v.state()
    assert s["mic"] == "idle" and s["status"] == "Transcription cancelled. Nothing will run.", s
    v.release(late)
    time.sleep(0.3)
    s = v.state()
    assert s["request"] == "" and s["heard"] is None and s["countdown"] is None and s["mic"] == "idle", s

    # so does the mic button (not the second half of a double-click, though)
    v.hold("AltRight", 0.4)
    v.wait_mic("transcribing")
    time.sleep(0.7)
    v.pg.click("#mic-btn")
    s = v.state()
    assert s["mic"] == "idle" and s["status"] == "Transcription cancelled. Nothing will run.", s
    v.release(late)

    # typing meanwhile: the late transcript neither replaces the text nor runs
    v.hold("AltRight", 0.4)
    v.wait_mic("transcribing")
    v.pg.click("#request")
    v.pg.keyboard.type("Download my September statement.")
    v.release(late)
    v.wait_mic("done")
    s = v.state()
    assert s["request"] == "Download my September statement." and s["task"] == "" and s["countdown"] is None, s
    assert s["status"] == (f"Heard: “{STUB_TEXT}”. You changed the request meanwhile, so yours was kept and nothing "
                           "runs by itself."), s
    assert s["heard"] == f"Heard (stub, 5000 ms): {STUB_TEXT}", s
    time.sleep(1.8)
    assert not v.run_bodies and v.state()["run"] == "idle" and len(v.wavs) == 3
    v.assert_clean(allow_http_errors=True)


def test_a_transcript_over_500_characters_is_cut_and_not_run(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    said = " ".join(f"part {i} of my very long request" for i in range(25))
    assert len(said) > 500
    v.say_text(said)
    v.pg.evaluate("document.activeElement.blur()")
    v.hold("AltRight", 0.4)
    v.wait_mic("done")
    s = v.state()
    kept = s["request"]
    assert 450 < len(kept) <= 500 and said.startswith(kept) and said[len(kept)] == " ", kept  # cut between words
    assert s["countdown"] is None and s["active"] == "request", s
    assert s["status"] == (f"Too long: you said {len(said)} characters and a request holds 500 at most. The request "
                           f"box has the first {len(kept)}; check it, then press Enter to run it."), s
    assert s["heard"].endswith(f"{said} [too long: the request box kept the first {len(kept)} characters]"), s
    time.sleep(1.8)
    assert not v.run_bodies

    # Enter runs exactly what the box shows; the logs get the text that was used
    v.pg.keyboard.press("Enter")
    v.pg.wait_for_selector("#banner:not([hidden])", timeout=60000)
    (body,) = v.run_bodies
    assert body["request"] == kept and body["voice"]["text"] == kept, body
    v.assert_clean()


def test_option_chords_in_a_text_field_never_touch_the_mic(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    v.pg.focus("#request")
    # Option+E (é on a Mac), Option+Shift+2 (€) and a quick tap: silent, and no getUserMedia at all
    v.pg.keyboard.down("AltRight")
    v.pg.keyboard.press("KeyE")
    v.pg.keyboard.up("AltRight")
    v.pg.keyboard.down("AltRight")
    time.sleep(0.08)
    v.pg.keyboard.down("Shift")
    v.pg.keyboard.press("Digit2")
    v.pg.keyboard.up("Shift")
    v.pg.keyboard.up("AltRight")
    v.pg.keyboard.down("AltRight")
    time.sleep(0.08)
    v.pg.keyboard.up("AltRight")
    time.sleep(0.4)
    s = v.state()
    assert v.pg.evaluate("window.__gumCalls") == 0 and s["mic"] == "idle" and s["status"] == "", s
    assert s["active"] == "request", s

    # held alone past the short wait, it is push-to-talk as before
    v.say_text("")
    v.hold("AltRight", 0.4)
    v.wait_status("Didn’t catch that")
    assert v.pg.evaluate("window.__gumCalls") == 1 and len(v.wavs) == 1
    v.assert_clean()


def test_after_voice_setting(voice, fast_console):
    v = voice
    open_console(v, fast_console)
    sel = "#voice-autorun"
    assert v.pg.evaluate("document.getElementById('voice-autorun').closest('#options') !== null")
    assert v.pg.text_content("label[for='voice-autorun']") == "After voice input:"
    opts = v.pg.eval_on_selector_all(f"{sel} option", "els => els.map(o => [o.value, o.textContent])")
    assert opts == [["1.5", "run in 1.5 s"], ["5", "run in 5 s"], ["off", "don’t auto-run"]]
    assert v.pg.input_value(sel) == "1.5"
    stored = "localStorage.getItem('gui-assist.voice-autorun')"

    # don't auto-run: the transcript waits in the box
    v.pg.select_option(sel, "off")
    assert v.pg.evaluate(stored) == "off"
    v.say_text("Please water my plants")
    v.pg.evaluate("document.activeElement.blur()")
    v.hold("AltRight", 0.4)
    v.wait_mic("done")
    s = v.state()
    assert s["request"] == "Please water my plants" and s["countdown"] is None, s
    assert s["status"] == "Heard: “Please water my plants”. Press Enter in the request box to run it.", s
    time.sleep(1.8)
    assert not v.run_bodies

    # 5 s: still counting after 2 s; the screen-reader text says 5 seconds
    v.pg.select_option(sel, "5")
    v.pg.evaluate("document.activeElement.blur()")
    v.hold("AltRight", 0.4)
    v.wait_mic("done")
    s = v.state()
    assert re.fullmatch(r"Running in (5\.0|4\.\d) s — press Esc to cancel", s["countdown"] or ""), s
    assert v.pg.text_content("#voice-status .visually-hidden") == " Runs in 5 seconds. Escape stops it."
    time.sleep(2.0)
    assert v.state()["countdown"] is not None and not v.run_bodies
    v.pg.keyboard.press("Escape")
    assert v.state()["countdown"] is None

    # it survives a reload; the default removes the stored choice
    v.pg.reload()
    open_console(v, fast_console)
    assert v.pg.input_value(sel) == "5"
    v.pg.select_option(sel, "1.5")
    assert v.pg.evaluate(stored) is None
    time.sleep(0.5)
    assert not v.run_bodies
    v.assert_clean()


# --- the warm microphone --------------------------------------------------------------------------------


def pcm_of(data: bytes) -> tuple[array, int]:
    with wave.open(io.BytesIO(data)) as w:
        return array("h", w.readframes(w.getnframes())), w.getframerate()


def loudest(pcm: array, rate: int, start_s: float, end_s: float) -> float:
    """The highest RMS (PCM16 units) of any 10 ms window in [start_s, end_s)."""
    n = rate // 100
    best = 0.0
    for i in range(int(start_s * rate), int(end_s * rate) - n + 1, n):
        w = pcm[i:i + n]
        best = max(best, math.sqrt(sum(x * x for x in w) / len(w)))
    return best


SOUND = 300  # PCM16 RMS: speech is thousands; the phrase's short gaps and a dead mic are tens


def test_warm_press_starts_with_the_pre_roll(voice, fast_console):
    """The fix for "Pay fifty dollars toward my bill" heard as "Allors towards my bill" (the first ~0.6 s went to
    getUserMedia, the AudioContext and the worklet). After the first take the microphone stays open: the next press
    calls no getUserMedia, is listening at once, and its WAV starts with the ~400 ms before the key went down (sound
    in its first 150 ms). After a stall (a suspended context, which the press resumes) nothing old is sent."""
    v = voice
    open_console(v, fast_console)
    v.say_text("")
    v.pg.evaluate("document.activeElement.blur()")

    # cold: one getUserMedia, recorded from the first frame, "start speaking now"
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    assert v.state()["status"] == COLD_PTT
    time.sleep(0.6)
    v.pg.keyboard.up("AltRight")
    v.wait_status("Didn’t catch that")
    cold = v.last_take()
    assert cold["cold"] and cold["preRollMs"] < 100 and cold["sent"], cold
    spy = v.pg.evaluate(SPY_STATE)
    assert spy["gum"] == 1 and spy["tracks"] == ["live"] and spy["contexts"] == ["running"], spy
    # the warm microphone says so, and announces it once, politely
    assert v.warm() == WARM_TEXT
    v.wait("document.querySelector('#mic-warm-status').textContent !== ''")
    assert v.announced() == f"{WARM_TEXT}. Turn off releases it now."
    assert v.pg.get_attribute("#mic-warm-status", "role") == "status"
    assert v.pg.get_attribute("#mic-warm-status", "aria-live") == "polite"
    assert v.pg.get_attribute("#mic-off", "aria-label") == "Turn off the microphone"
    assert v.pg.text_content("#mic-off") == "Turn off" and v.pg.is_visible("#mic-off")
    time.sleep(0.6)  # the pre-roll ring fills

    # warm: listening at once, no getUserMedia, and the recording begins before the key went down
    n_log = len(v.log())
    t0 = time.monotonic()
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    to_live = time.monotonic() - t0
    assert v.state()["status"] == "Listening… release to send"
    assert v.warm() == "Microphone on (recording)"
    time.sleep(0.8)
    v.pg.keyboard.up("AltRight")
    held = time.monotonic() - t0
    v.wait_uploads(2)
    v.wait_mic("idle")
    assert to_live < 0.35, to_live
    assert not [e for e in v.log()[n_log:] if e["status"].startswith("Starting")], v.log()[n_log:]
    warm = v.last_take()
    assert not warm["cold"] and 300 <= warm["preRollMs"] <= 500 and warm["sent"], warm
    assert v.pg.evaluate("window.__gumCalls") == 1
    pcm, rate = pcm_of(v.wavs[1])
    seconds = len(pcm) / rate
    assert rate == 16000 and 0.25 < seconds - held < 0.65, (seconds, held, warm)
    assert seconds > held + warm["preRollMs"] / 1000 - 0.15, (seconds, held, warm)
    assert loudest(pcm, rate, 0, 0.15) > SOUND, loudest(pcm, rate, 0, 0.15)  # from before the press
    assert v.warm() == WARM_TEXT

    # a stalled context: the press resumes it (as Safari needs), and audio from before the stall is not sent
    v.pg.evaluate("() => Promise.all(window.__contexts.filter(c => c.state === 'running').map(c => c.suspend()))")
    assert v.pg.evaluate(SPY_STATE)["contexts"] == ["suspended"]
    time.sleep(1.0)
    t0 = time.monotonic()
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    time.sleep(0.6)
    v.pg.keyboard.up("AltRight")
    held = time.monotonic() - t0
    v.wait_uploads(3)
    v.wait_mic("idle")
    stale = v.last_take()
    assert not stale["cold"] and stale["preRollMs"] < 150, stale
    pcm, rate = pcm_of(v.wavs[2])
    assert len(pcm) / rate < held + 0.2, (len(pcm) / rate, held, stale)
    spy = v.pg.evaluate(SPY_STATE)
    assert spy["gum"] == 1 and spy["contexts"] == ["running"] and spy["tracks"] == ["live"], spy
    v.assert_clean()


def test_cold_press_says_starting_until_frames_arrive(voice, fast_console):
    """The first press: "Starting the microphone…" (not red) while the microphone opens and no frames have come;
    "Listening… start speaking now" (red) only once they do. The frames are held back here (FRAME_GATE) for 0.5 s,
    and none is lost: the take is recorded from the first frame."""
    v = voice
    v.pg.context.add_init_script(FRAME_GATE)
    open_console(v, fast_console)
    v.say_text("")
    v.pg.evaluate("document.activeElement.blur()")
    idle_bg = v.bg()
    v.pg.keyboard.down("AltRight")
    v.wait_status("Starting the microphone…")
    v.wait("window.__queued.length > 0")  # the microphone is open and sending; its frames are held back
    time.sleep(0.5)
    s = v.state()
    assert s["mic"] == "opening" and s["status"] == "Starting the microphone…" and s["label"] == "Stop listening", s
    assert v.bg() == idle_bg and v.pg.is_hidden("#voice-timer")  # not the red listening state, no clock
    assert v.pg.evaluate("window.__delivered") == 0
    v.pg.evaluate("window.__openGate()")
    v.wait_mic("listening")
    s = v.state()
    assert s["status"] == COLD_PTT and s["tone"] == "live", s
    assert v.bg() != idle_bg
    time.sleep(0.4)
    v.pg.keyboard.up("AltRight")
    v.wait_status("Didn’t catch that")

    log = v.log()
    first = next(i for i, e in enumerate(log) if e["status"].startswith("Listening"))
    assert log[first]["state"] == "listening" and log[first]["frames"] > 0, log
    before = [e for e in log[:first] if e["state"] in ("opening", "listening")]
    assert before and all(e["state"] == "opening" and e["frames"] == 0 and e["bg"] == idle_bg for e in before), log
    assert {e["status"] for e in before} <= {"Starting the microphone…", ""}, log
    take = v.last_take()
    assert take["cold"] and take["seconds"] > 0.85 and take["sent"], take  # 0.5 s held back + 0.4 s listening
    assert len(v.wavs) == 1 and v.warm() == WARM_TEXT
    v.assert_clean()


def test_the_warm_microphone_is_released(voice, fast_console):
    """Released after the idle time (5 min, shortened here with AcmeVoice.setIdleReleaseMs), at once by Turn off (by
    keyboard: the next Tab stop after the request box; also during a take), when the page is hidden, and when voice
    input goes off. Each time every track stops and the context closes; the next press opens it again."""
    v = voice
    v.pg.context.add_init_script(FAKE_PERMISSION)
    open_console(v, fast_console)
    v.say_text("")

    def take(cold: bool = True) -> None:
        n = len(v.wavs)
        v.pg.evaluate("document.activeElement.blur()")
        v.pg.keyboard.down("AltRight")
        v.wait_mic("listening")
        assert v.state()["status"] == (COLD_PTT if cold else "Listening… release to send")
        time.sleep(0.4)
        v.pg.keyboard.up("AltRight")
        v.wait_uploads(n + 1)
        v.wait_mic("idle")

    # idle: released after the (shortened) idle time, and not before
    take()
    assert v.warm() == WARM_TEXT
    assert v.pg.evaluate("window.AcmeVoice.setIdleReleaseMs(1500)") == 1500
    assert v.warm() == "Microphone ready (releases after 1.5 s idle)"
    time.sleep(0.8)
    assert v.pg.evaluate("window.AcmeVoice.mic().warm") and v.pg.evaluate(SPY_STATE)["tracks"] == ["live"]
    v.wait_released(timeout=4)
    assert v.warm() is None and not v.pg.evaluate("window.AcmeVoice.mic().warm")
    v.wait("document.querySelector('#mic-warm-status').textContent.startsWith('Microphone off')")
    assert v.announced() == "Microphone off: released after 1.5 s without a recording."
    assert v.pg.evaluate("window.AcmeVoice.setIdleReleaseMs(300000)") == 300000

    # the next press opens it again (cold); Turn off by keyboard releases it at once, focus goes to Speak
    take(cold=True)
    assert v.pg.evaluate("window.__gumCalls") == 2 and v.warm() == WARM_TEXT
    v.pg.focus("#request")
    v.pg.keyboard.press("Tab")
    assert v.state()["active"] == "mic-off"
    v.pg.keyboard.press("Enter")
    assert v.pg.evaluate(SPY_STATE)["tracks"] == ["ended", "ended"] and v.warm() is None
    v.wait_released()
    assert v.state()["active"] == "mic-btn"
    v.wait("document.querySelector('#mic-warm-status').textContent === 'Microphone off.'")

    # Turn off during a take drops the take
    take(cold=True)
    n = len(v.wavs)
    v.pg.evaluate("document.activeElement.blur()")
    v.pg.keyboard.down("AltRight")
    v.wait_mic("listening")
    v.pg.click("#mic-off")
    s = v.state()
    assert s["mic"] == "idle" and s["status"] == "Cancelled: the microphone was turned off. Nothing was sent.", s
    v.pg.keyboard.up("AltRight")
    v.wait_released()
    time.sleep(0.3)
    assert len(v.wavs) == n and v.pg.evaluate("window.__gumCalls") == 3

    # the page is hidden
    take(cold=True)
    assert v.pg.evaluate(SPY_STATE)["tracks"][-1] == "live"
    v.pg.evaluate("""() => {
      Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
      document.dispatchEvent(new Event('visibilitychange'));
      delete document.hidden;
    }""")
    v.wait_released()
    assert v.warm() is None

    # voice input goes off (the microphone permission is blocked), then comes back: the mic stays released
    take(cold=True)
    v.pg.evaluate("window.__perm('denied')")
    v.wait_released()
    s = v.state()
    assert s["mic"] == "off" and s["hint"].startswith("Voice input is off: microphone access is blocked"), s
    v.pg.evaluate("window.__perm('granted')")
    v.wait_mic("idle")
    assert v.warm() is None and v.pg.evaluate("window.__gumCalls") == 5
    v.assert_clean()
