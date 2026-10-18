"""Voice input, part 1: the in-page WAV encoder, POST /api/transcribe, and the mic flow end to end.

Three layers, checked separately:
- wav.js in a real page (window.AcmeWav): a known Float32 signal goes in, the bytes come back to Python and are read
  with the stdlib `wave` module and checked field by field against the 44-byte RIFF/WAVE PCM16 header; clipping
  saturates; resample() keeps lengths and DC levels.
- The endpoint, with no browser (FastAPI TestClient, stub engine): raw audio/wav and multipart "audio" both work;
  non-WAV, stereo, 44.1 kHz, 8-bit, float, empty, truncated and too-short uploads are 400; oversize is 413; a broken
  engine is 503; /api/config reports `asr`.
- The page in headless Chromium with Chromium's fake microphone (--use-fake-device-for-media-stream, a beeping test
  source) against a live console running the real mock agent and the real sandbox: click the mic, click again, the
  stub transcript fills the box and selects its sandbox task, a countdown runs it, and the verdict banner shows ✓.
  Also: Esc during the countdown, push-to-talk (Right Option, Space off controls but not on the Run button or a
  focused scroller), and "stop" spoken during a slow run. The capture fallbacks (a refused 16 kHz context, no
  AudioWorklet) also get a warm second press with its pre-roll, then Turn off (the warm microphone's own tests are in
  test_voice_ui.py). Every browser test checks for no console errors, no request off 127.0.0.1 and
  no touch of the Web Speech API or MediaRecorder at runtime; a static scan checks console/static never names them.

Browser tests are marked `e2e` like test_console_ui.py (skip with -m "not e2e").
"""

from __future__ import annotations

import io
import json
import math
import re
import struct
import time
import wave
from array import array
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from console import asr
from console.app import CONSOLE_DIR, ConsoleSettings, create_app
from console.runner import VERDICT_LABEL
from console.tests.conftest import ServerThread

STATIC = CONSOLE_DIR / "static"
STUB_TEXT = "Turn on high contrast."  # the stub's default, and a sandbox task's exact request
TASK = "high-contrast"
LONG_TASK = "message-support-meter"  # 7 steps: at 1 s per step it is still running while the test speaks
VIEWPORT = {"width": 1280, "height": 800}
LOCAL_HOST = "127.0.0.1"
FAKE_MIC = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]
# the console's defaults (--grounder, --thinking, --clarify)
DEFAULT_OPTIONS = {"grounder": "grounder", "thinking": "off", "lightning": "local", "clarify": "on"}
HEARD_STUB = re.compile(r"^Heard \(stub, \d+ ms\): Turn on high contrast\.$")
# The first press opens the microphone and records from its first frame; later presses find it warm.
COLD_TOGGLE = "Listening… start speaking now, press again to send"
COLD_PTT = "Listening… start speaking now, release to send"
FORBIDDEN_APIS = ("SpeechRecognition", "webkitSpeechRecognition", "MediaRecorder")

e2e = pytest.mark.e2e


# --- WAV builders (Python side) ------------------------------------------------------------------------


def make_wav(seconds: float = 1.0, rate: int = 16000, channels: int = 1, width: int = 2, tone: bool = True) -> bytes:
    """A WAV via the stdlib writer: a quiet 440 Hz tone (or silence) at any rate, channel count and sample width."""
    n = int(round(seconds * rate))
    if tone and width == 2:
        one = [int(6000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]
        frames = array("h", [s for s in one for _ in range(channels)]).tobytes()
    elif tone and width == 1:
        frames = bytes(128 + int(40 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n) for _ in range(channels))
    else:
        frames = (b"\x80" if width == 1 else b"\x00") * (n * channels * width)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def float32_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """IEEE float WAV (format 3): a real WAV, but not PCM16."""
    size = int(seconds * rate) * 4
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + size, b"WAVE", b"fmt ", 16, 3, 1, rate, rate * 4, 4, 32,
                       b"data", size) + b"\x00" * size


def header(data: bytes) -> dict:
    keys = ("riff", "riff_size", "wave", "fmt", "fmt_size", "format", "channels", "rate", "byte_rate", "block_align",
            "bits", "data_id", "data_size")
    return dict(zip(keys, struct.unpack("<4sI4s4sIHHIIHH4sI", data[:44])))


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def pcm16_expected(x: float) -> int:
    """What a correct encoder writes for one Float32 sample: NaN -> 0, clamp to [-1, 1], asymmetric scale, round half up."""
    s = f32(x)
    if math.isnan(s):
        return 0
    s = max(-1.0, min(1.0, s))
    return math.floor((s * 32768 if s < 0 else s * 32767) + 0.5)


# --- fixtures ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub_engine():
    """The stub engine with its default text, whatever CONSOLE_ASR / CONSOLE_ASR_STUB_TEXT say on this machine."""
    asr.set_engine(asr.StubEngine(STUB_TEXT))
    yield
    asr.set_engine(None)  # back to the configured engine, loaded lazily


async def _never_run(*a, **kw):  # the endpoint tests never start a run
    raise AssertionError("no runs in the endpoint tests")


@pytest.fixture(scope="module")
def client(stub_engine, tmp_path_factory):
    settings = ConsoleSettings(agent="mock", sandbox_url="http://127.0.0.1:9", static_dir=STATIC,
                               runs_dir=tmp_path_factory.mktemp("voice-api-runs"))
    with TestClient(create_app(settings, agent=(_never_run, lambda **kw: {}))) as c:
        yield c


def _console(sandbox_url, runs, **kw) -> ServerThread:
    settings = ConsoleSettings(agent="mock", sandbox_url=sandbox_url, runs_dir=runs, headed=False,
                               static_dir=STATIC, stop_timeout_s=10.0, **kw)
    return ServerThread(create_app(settings))


@pytest.fixture(scope="module")
def fast_console(sandbox_url, tmp_path_factory, stub_engine):
    """Live console, real mock agent, no step delay."""
    srv = _console(sandbox_url, tmp_path_factory.mktemp("voice-fast-runs"), step_delay_s=0.0)
    yield srv.start()
    srv.stop()


@pytest.fixture(scope="module")
def slow_console(sandbox_url, tmp_path_factory, stub_engine):
    """Live console with 1 s per step, so a run is still going while the test speaks."""
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


# Before any page script: trap the forbidden speech/recording APIs (any read is recorded) and keep a trace of the
# mic button's states with the status line's text at each change.
INIT_JS = """(() => {
  window.__forbidden = [];
  for (const name of %s) {
    const orig = window[name];
    try {
      Object.defineProperty(window, name, { configurable: true,
        get() { window.__forbidden.push(name); return orig; }, set() { window.__forbidden.push(name + '='); } });
    } catch (e) { window.__forbidden.push('trap failed: ' + name); }
  }
  window.__trace = [];
  document.addEventListener('DOMContentLoaded', () => {
    const mic = document.getElementById('mic-btn');
    const status = document.getElementById('voice-status');
    let last = '';
    const rec = () => {
      const key = mic.dataset.state + '|' + mic.getAttribute('aria-pressed');
      if (key === last) return;
      last = key;
      window.__trace.push({ state: mic.dataset.state, pressed: mic.getAttribute('aria-pressed'),
                            status: status.textContent });
    };
    new MutationObserver(rec).observe(mic, { attributes: true, attributeFilter: ['data-state', 'aria-pressed'] });
    rec();
  });
})();""" % json.dumps(list(FORBIDDEN_APIS))

STATE_JS = """() => {
  const q = (s) => document.querySelector(s);
  return {
    mic: q('#mic-btn').dataset.state, pressed: q('#mic-btn').getAttribute('aria-pressed'),
    label: q('#mic-label').textContent, disabled: q('#mic-btn').disabled,
    status: q('#voice-status').textContent,
    countdown: q('#voice-countdown').hidden ? null : q('#voice-countdown-text').textContent,
    heard: q('#heard').hidden ? null : q('#heard').textContent,
    request: q('#request').value, task: q('#task-select').value, run: q('#run-status').dataset.state,
    formError: q('#form-error').textContent,
    active: document.activeElement ? (document.activeElement.id || document.activeElement.tagName) : null,
  };
}"""


@dataclass
class Mic:
    """A page plus what it did: WAV uploads, /api/run bodies, every request URL, off-host requests, console errors.

    `reply` = None lets /api/transcribe through to the live backend (stub engine); a dict is fulfilled as its JSON."""
    pg: object = None
    wavs: list = field(default_factory=list)
    content_types: list = field(default_factory=list)
    run_bodies: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    offhost: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    reply: dict | None = None

    def state(self) -> dict:
        return self.pg.evaluate(STATE_JS)

    def wait(self, js: str, timeout: float = 10) -> None:
        self.pg.wait_for_function(js, timeout=timeout * 1000)

    def wait_mic(self, state: str, timeout: float = 10) -> None:
        self.wait(f"document.querySelector('#mic-btn').dataset.state === {json.dumps(state)}", timeout)

    def trace(self) -> list[dict]:
        return self.pg.evaluate("window.__trace")

    def assert_clean(self, base: str) -> None:
        assert not self.offhost, f"requests left {LOCAL_HOST}: {self.offhost}"
        assert not self.errors, self.errors
        origin = urlsplit(base).netloc
        assert self.requests and all(urlsplit(u).netloc == origin for u in self.requests), self.requests
        assert self.pg.evaluate("window.__forbidden") == [], "the page touched a forbidden speech/recording API"


@pytest.fixture
def mic(browser):
    m = Mic()
    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="en-US",
                              timezone_id="America/New_York")
    ctx.add_init_script(INIT_JS)

    def route(r):
        url = urlsplit(r.request.url)
        m.requests.append(r.request.url)
        if url.hostname != LOCAL_HOST:
            m.offhost.append(r.request.url)
            return r.abort("blockedbyclient")
        if r.request.method == "POST" and url.path == "/api/run":
            m.run_bodies.append(json.loads(r.request.post_data or "{}"))
        if r.request.method == "POST" and url.path == "/api/transcribe":
            m.wavs.append(r.request.post_data_buffer or b"")
            m.content_types.append(r.request.headers.get("content-type"))
            if m.reply is not None:
                return r.fulfill(status=200, content_type="application/json", body=json.dumps(m.reply))
        return r.continue_()

    ctx.route("**/*", route)
    m.pg = ctx.new_page()
    m.pg.on("console", lambda msg: m.errors.append(f"console.{msg.type}: {msg.text}") if msg.type == "error" else None)
    m.pg.on("pageerror", lambda e: m.errors.append(f"pageerror: {e}"))
    yield m
    ctx.close()


def open_console(m: Mic, base: str) -> None:
    m.pg.goto(base + "/")
    m.wait("document.querySelector('#task-select').options.length > 12")  # Free text + 12 tasks
    m.wait("document.querySelector('#mic-btn').dataset.state === 'idle'")  # /api/config said asr is available


def read_wav(data: bytes) -> tuple[int, int, int, int, bytes]:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes(), w.readframes(w.getnframes())


def states(trace: list[dict]) -> list[str]:
    return [t["state"] for t in trace]


def is_subsequence(want: list[str], got: list[str]) -> bool:
    it = iter(got)
    return all(s in it for s in want)


# --- static scan (no browser) --------------------------------------------------------------------------


def test_static_files_use_no_cloud_speech_or_media_recorder():
    files = sorted(STATIC.glob("*.js")) + [STATIC / "index.html"]
    assert {"voice.js", "wav.js", "pcm-worklet.js", "console.js"} <= {f.name for f in files}
    for f in files:
        text = f.read_text()
        for name in FORBIDDEN_APIS + ("webm", "speechSynthesis", "XMLHttpRequest", "WebSocket", "sendBeacon"):
            assert name not in text, f"{f.name} mentions {name}"
        # every fetch goes to this console's own /api (relative URLs only; no host anywhere)
        for target in re.findall(r"fetch\(\s*([`'\"])(.*?)\1", text):
            assert target[1].startswith("/api/"), f"{f.name}: fetch({target[1]!r})"
    voice = (STATIC / "voice.js").read_text()
    # the WAV upload, and /api/config again while the recognizer is down
    assert sorted(re.findall(r"fetch\(\s*'([^']*)'", voice)) == ["/api/config", "/api/transcribe"]
    assert "'Content-Type': 'audio/wav'" in voice
    assert "addModule('/static/pcm-worklet.js')" in voice
    html = (STATIC / "index.html").read_text()
    assert re.findall(r'<script src="([^"]+)"', html) == ["/static/console.js", "/static/wav.js", "/static/voice.js"]
    assert "http://" not in html and "https://" not in html  # no CDN


# --- the endpoint (no browser) -------------------------------------------------------------------------


def test_config_reports_asr(client):
    assert client.get("/api/config").json()["asr"] == {"available": True, "model": "stub", "label": "stub"}


def test_transcribe_raw_wav(client):
    r = client.post("/api/transcribe", content=make_wav(1.25), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"text", "duration_s", "latency_ms", "model"}
    assert body["text"] == STUB_TEXT and body["model"] == "stub"
    assert body["duration_s"] == pytest.approx(1.25, abs=1e-3)
    assert isinstance(body["latency_ms"], (int, float)) and body["latency_ms"] >= 0


def test_transcribe_multipart(client):
    r = client.post("/api/transcribe", files={"audio": ("speech.wav", make_wav(0.75), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == STUB_TEXT and body["model"] == "stub" and body["latency_ms"] >= 0
    assert body["duration_s"] == pytest.approx(0.75, abs=1e-3)
    # the wrong field name is a 400, not a crash
    r = client.post("/api/transcribe", files={"file": ("speech.wav", make_wav(0.75), "audio/wav")})
    assert r.status_code == 400 and "audio" in r.json()["error"]


@pytest.mark.parametrize("seconds", [0.2, 60.0])
def test_transcribe_duration_limits_are_inclusive(client, seconds):
    r = client.post("/api/transcribe", content=make_wav(seconds, tone=False), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200, r.text
    assert r.json()["duration_s"] == pytest.approx(seconds, abs=1e-3)


BAD_AUDIO = {
    "not-wav": (b"this is not audio at all. " * 40, "WAV"),
    "riff-but-not-wave": (b"RIFF" + struct.pack("<I", 1000) + b"AVI " + b"\x00" * 1000, "WAV"),
    "stereo": (make_wav(1.0, channels=2), "2 channel"),
    "44.1kHz": (make_wav(1.0, rate=44100), "44100 Hz"),
    "8-bit": (make_wav(1.0, width=1), "8-bit"),
    "float32": (float32_wav(), "format"),
    "empty": (b"", "No audio"),
    "truncated-header": (make_wav(1.0)[:30], "Unreadable"),
    "header-only": (make_wav(1.0)[:44], "too short"),
    "too-short": (make_wav(0.1), "too short"),
    "just-under-0.2s": (make_wav(0.199), "too short"),
}


@pytest.mark.parametrize("case", list(BAD_AUDIO))
def test_transcribe_rejects_bad_audio(client, case):
    data, words = BAD_AUDIO[case]
    r = client.post("/api/transcribe", content=data, headers={"Content-Type": "audio/wav"})
    assert r.status_code == 400, (case, r.status_code, r.text)
    err = r.json()["error"]
    assert isinstance(err, str) and words in err, (case, err)


def test_transcribe_rejects_oversize(client):
    big = make_wav(61.0, tone=False)
    assert len(big) > asr.MAX_BYTES + 4096  # refused from Content-Length, before reading the body
    r = client.post("/api/transcribe", content=big, headers={"Content-Type": "audio/wav"})
    assert r.status_code == 413 and "too large" in r.json()["error"]
    r = client.post("/api/transcribe", files={"audio": ("speech.wav", big, "audio/wav")})
    assert r.status_code == 413 and "too large" in r.json()["error"]
    # just over MAX_BYTES but under the Content-Length slack: refused after reading
    r = client.post("/api/transcribe", content=make_wav(60.0, tone=False) + b"\x00" * 2048,
                    headers={"Content-Type": "audio/wav"})
    assert r.status_code == 413, r.text


def test_transcribe_engine_unavailable_is_503(client, monkeypatch):
    def broken():
        raise ImportError("No module named 'nemo'")

    saved = asr._engine_name
    monkeypatch.setitem(asr.ENGINES, "broken", broken)
    asr.configure("broken")
    try:
        cfg = client.get("/api/config").json()["asr"]
        assert cfg["available"] is False and cfg["label"] == "broken" and "No module named 'nemo'" in cfg["error"]
        r = client.post("/api/transcribe", content=make_wav(0.5), headers={"Content-Type": "audio/wav"})
        assert r.status_code == 503 and "failed to load" in r.json()["error"]
        # bad audio is still a 400 (validated before the engine loads)
        assert client.post("/api/transcribe", content=b"nope", headers={"Content-Type": "audio/wav"}).status_code == 400
    finally:
        asr.configure(saved)
        asr.set_engine(asr.StubEngine(STUB_TEXT))
    assert client.get("/api/config").json()["asr"]["available"] is True


# --- wav.js in the page --------------------------------------------------------------------------------


@e2e
def test_wav_encoder_header_samples_and_clipping(mic, fast_console):
    m = mic
    open_console(m, fast_console)
    sine = [0.5 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(4000)]  # 0.25 s at 16 kHz
    extra = [1.5, -1.5, 1.0, -1.0, 0.0, 1e9, -1e9, 0.99999, -0.99999, float("nan"), float("inf"), float("-inf")]
    signal = sine + extra
    data = bytes(m.pg.evaluate(
        "(xs) => Array.from(new Uint8Array(window.AcmeWav.encodeWav(Float32Array.from(xs), 16000)))", signal))
    n = len(signal)

    # the stdlib reader agrees
    channels, rate, width, frames, pcm = read_wav(data)
    assert (channels, rate, width, frames) == (1, 16000, 2, n)

    # the 44-byte header, field by field
    assert len(data) == 44 + 2 * n
    h = header(data)
    assert h == {"riff": b"RIFF", "riff_size": len(data) - 8, "wave": b"WAVE", "fmt": b"fmt ", "fmt_size": 16,
                 "format": 1, "channels": 1, "rate": 16000, "byte_rate": 32000, "block_align": 2, "bits": 16,
                 "data_id": b"data", "data_size": 2 * n}
    assert data[:4] == b"RIFF" and data[8:16] == b"WAVEfmt " and data[36:40] == b"data"
    assert data[16:20] == b"\x10\x00\x00\x00" and data[20:22] == b"\x01\x00" and data[22:24] == b"\x01\x00"
    assert data[24:28] == b"\x80\x3e\x00\x00" and data[28:32] == b"\x00\x7d\x00\x00"  # 16000, 32000 little-endian
    assert data[32:34] == b"\x02\x00" and data[34:36] == b"\x10\x00"

    # every sample, little-endian int16, exactly as a correct encoder writes it
    got = array("h", pcm)
    assert got.tolist() == [pcm16_expected(x) for x in signal]
    tail = got.tolist()[len(sine):]
    assert tail[:7] == [32767, -32768, 32767, -32768, 0, 32767, -32768]  # clipping saturates, no wraparound
    assert tail[9:] == [0, 32767, -32768]  # NaN -> silence, +-Infinity saturate
    assert max(got) == 32767 and min(got) == -32768
    assert abs(max(got[:4000]) - 16384) <= 2  # the 0.5-amplitude sine peaks at half scale

    # odd rates are written as given; empty input is a valid empty WAV; bad rates throw
    other = bytes(m.pg.evaluate("() => Array.from(new Uint8Array(AcmeWav.encodeWav(new Float32Array(3), 44100)))"))
    assert header(other)["rate"] == 44100 and header(other)["byte_rate"] == 88200 and len(other) == 50
    empty = bytes(m.pg.evaluate("() => Array.from(new Uint8Array(AcmeWav.encodeWav(new Float32Array(0), 16000)))"))
    assert len(empty) == 44 and header(empty)["data_size"] == 0 and header(empty)["riff_size"] == 36
    assert read_wav(empty)[3] == 0
    errs = m.pg.evaluate("""() => [0, -1, NaN, 'x'].map((r) => {
        try { AcmeWav.encodeWav(new Float32Array(4), r); return 'no error'; } catch (e) { return e.name; } })""")
    assert errs == ["RangeError"] * 4

    # and the backend accepts what the page encodes (the real endpoint, stub engine)
    r = httpx.post(f"{fast_console}/api/transcribe", content=data, headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200 and r.json()["duration_s"] == pytest.approx(n / 16000, abs=1e-3)
    m.assert_clean(fast_console)


@e2e
def test_wav_resample_lengths_dc_and_tone(mic, fast_console):
    m = mic
    open_console(m, fast_console)
    out = m.pg.evaluate("""() => {
      const W = window.AcmeWav;
      const dc = (n, v) => new Float32Array(n).fill(v);
      const sine = (n, rate, f) => Float32Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * f * i / rate));
      const stats = (a) => ({ len: a.length, min: Math.min(...a), max: Math.max(...a), type: a.constructor.name });
      const r48 = W.resample(dc(48000, 0.3), 48000, 16000);
      const r48b = W.resample(dc(4800, -0.7), 48000);  // toRate defaults to 16 kHz
      const r441 = W.resample(dc(44100, 0.25), 44100, 16000);
      const r8 = W.resample(dc(8000, 0.4), 8000, 16000);
      const src = dc(1600, 0.1);
      const same = W.resample(src, 16000, 16000);
      const tone = W.resample(sine(4800, 48000, 440), 48000, 16000);
      const ref = sine(1600, 16000, 440);  // the same tone sampled at 16 kHz
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < 1600; i += 1) { dot += tone[i] * ref[i]; na += tone[i] ** 2; nb += ref[i] ** 2; }
      return {
        r48: stats(r48), r48b: stats(r48b), r441: stats(r441), r8: stats(r8),
        same: { ...stats(same), copy: same !== src },
        empty: W.resample(new Float32Array(0), 48000, 16000).length,
        corr: dot / Math.sqrt(na * nb), toneRms: W.rms(tone),
        rms: [W.rms(dc(100, 0.5)), W.rms(new Float32Array(0)), W.rms(null)],
        bad: [0, -5, NaN].map((r) => { try { W.resample(dc(4, 0), r); return 'no error'; } catch (e) { return e.name; } }),
        target: W.TARGET_RATE,
      };
    }""")
    assert out["r48"]["len"] == 16000 and out["r48"]["type"] == "Float32Array"
    assert out["r48"]["min"] == pytest.approx(0.3, abs=1e-6) and out["r48"]["max"] == pytest.approx(0.3, abs=1e-6)
    assert out["r48b"]["len"] == 1600 and out["r48b"]["min"] == pytest.approx(-0.7, abs=1e-6)
    assert out["r48b"]["max"] == pytest.approx(-0.7, abs=1e-6)
    assert out["r441"]["len"] == 16000, out["r441"]  # 1 s at 44.1 kHz is 1 s at 16 kHz
    assert out["r441"]["min"] == pytest.approx(0.25, abs=1e-6) and out["r441"]["max"] == pytest.approx(0.25, abs=1e-6)
    assert out["r8"]["len"] == 16000 and out["r8"]["min"] == pytest.approx(0.4, abs=1e-6)
    assert out["r8"]["max"] == pytest.approx(0.4, abs=1e-6)
    assert out["same"]["len"] == 1600 and out["same"]["copy"] and out["same"]["min"] == pytest.approx(0.1, abs=1e-7)
    assert out["empty"] == 0
    assert out["corr"] > 0.99, out["corr"]  # a 440 Hz tone stays a 440 Hz tone (no aliasing, no time warp)
    assert out["toneRms"] == pytest.approx(1 / math.sqrt(2), rel=0.01)
    assert out["rms"] == [pytest.approx(0.5), 0, 0]
    assert out["bad"] == ["RangeError"] * 3 and out["target"] == 16000
    m.assert_clean(fast_console)


# --- the mic flow with the fake microphone --------------------------------------------------------------


@e2e
def test_mic_click_transcribes_and_runs_a_verified_task(mic, fast_console):
    m = mic
    open_console(m, fast_console)
    s = m.state()
    # a plain action button: the label says what pressing it does, and there is no aria-pressed to contradict it
    assert (s["mic"], s["pressed"], s["label"], s["disabled"]) == ("idle", None, "Speak", False), s
    assert m.pg.get_attribute("#voice-status", "role") == "status"
    assert m.pg.get_attribute("#voice-status", "aria-live") == "polite"

    m.pg.click("#mic-btn")
    m.wait_mic("listening")
    s = m.state()
    assert s["pressed"] is None and s["label"] == "Stop listening", s
    assert s["status"] == COLD_TOGGLE, s
    time.sleep(1.2)
    m.pg.click("#mic-btn")
    m.wait_mic("done")

    # every state on the way, each announced in the live region ("opening": the mic is starting, no frames yet)
    trace = m.trace()
    assert is_subsequence(["idle", "opening", "listening", "transcribing", "done"], states(trace)), trace
    by_state = {t["state"]: t for t in trace}
    assert all(t["pressed"] is None for t in trace), trace
    assert by_state["opening"]["status"] == "Starting the microphone…"
    assert by_state["transcribing"]["status"] == "Transcribing…"
    assert by_state["listening"]["status"] == COLD_TOGGLE

    # the upload: one raw 16 kHz mono PCM16 WAV about as long as the recording, with the fake mic's beeps in it
    assert len(m.wavs) == 1 and m.content_types == ["audio/wav"]
    channels, rate, width, frames, pcm = read_wav(m.wavs[0])
    assert (channels, rate, width) == (1, 16000, 2)
    assert 1.0 <= frames / rate <= 3.0, frames / rate
    assert header(m.wavs[0])["data_size"] == len(m.wavs[0]) - 44
    assert max(abs(x) for x in array("h", pcm)) > 500  # not silence

    # the transcript fills the box and selects its sandbox task; the heard line sits in the step log
    s = m.state()
    assert s["request"] == STUB_TEXT and s["task"] == TASK, s
    assert HEARD_STUB.match(s["heard"] or ""), s
    assert m.pg.locator(".log #heard").is_visible()
    assert s["status"].startswith(f"Heard: “{STUB_TEXT}” (a sandbox task)"), s
    # the screen-reader part is short, and still true when it's read after the run started
    assert m.pg.text_content("#voice-status .visually-hidden") == " Runs in 1.5 seconds. Escape stops it."
    assert s["countdown"] and re.fullmatch(r"Running in \d\.\d s — press Esc to cancel", s["countdown"]), s

    # the countdown runs it; the sandbox verdict shows ✓
    m.wait("document.querySelector('#run-status').dataset.state !== 'idle'", timeout=5)
    assert m.state()["countdown"] is None
    m.pg.wait_for_selector("#banner:not([hidden])", timeout=60000)
    assert m.pg.get_attribute("#banner", "data-tone") == "pass"
    assert m.pg.inner_text("#banner-icon") == "✓"
    assert m.pg.inner_text("#banner-title") == "Verified: the task is done"
    assert m.pg.text_content("#banner-label") == VERDICT_LABEL
    assert len(m.run_bodies) == 1
    body = dict(m.run_bodies[0])
    voice = body.pop("voice")  # what was heard, for the logs page (console.json only; the agent never gets it)
    assert body == {"request": STUB_TEXT, "task_id": TASK, "options": DEFAULT_OPTIONS}
    assert voice["text"] == STUB_TEXT and voice["model"] == "stub" and voice["latency_ms"] >= 0, voice
    assert 1.0 <= voice["duration_s"] <= 3.0, voice
    s = m.state()
    assert s["run"] == "done" and HEARD_STUB.match(s["heard"] or ""), s
    log = m.pg.inner_text(".log")
    assert log.index("Heard (stub, ") < log.index("Started at /")
    m.assert_clean(fast_console)


@e2e
def test_escape_during_the_countdown_cancels_the_run(mic, fast_console):
    m = mic
    open_console(m, fast_console)
    m.pg.click("#mic-btn")
    m.wait_mic("listening")
    time.sleep(0.6)
    m.pg.click("#mic-btn")
    m.wait("!document.querySelector('#voice-countdown').hidden")
    m.pg.keyboard.press("Escape")
    s = m.state()
    assert s["countdown"] is None and s["status"].startswith("Auto-run cancelled"), s
    time.sleep(2.0)  # past the 1.5 s countdown
    s = m.state()
    assert not m.run_bodies and s["run"] == "idle", (m.run_bodies, s)
    assert s["request"] == STUB_TEXT and s["task"] == TASK  # still there, to run by hand
    assert len(m.wavs) == 1
    m.assert_clean(fast_console)


@e2e
def test_push_to_talk_keys(mic, fast_console):
    m = mic
    open_console(m, fast_console)

    # hold Right Option (in the request box: it's a modifier, so it works anywhere and types nothing)
    m.pg.focus("#request")
    m.pg.keyboard.down("AltRight")
    m.wait_mic("listening")
    assert m.state()["status"] == COLD_PTT
    time.sleep(0.6)
    m.pg.keyboard.up("AltRight")
    m.wait_mic("done")
    s = m.state()
    assert len(m.wavs) == 1 and s["request"] == STUB_TEXT and s["task"] == TASK and s["countdown"], s
    assert is_subsequence(["listening", "transcribing", "done"], states(m.trace()))
    m.pg.keyboard.press("Escape")
    assert m.state()["countdown"] is None

    # hold Space with focus on the page itself
    m.pg.fill("#request", "")
    m.pg.evaluate("document.activeElement.blur()")
    assert m.pg.evaluate("document.activeElement === document.body")
    m.pg.keyboard.down("Space")
    m.wait_mic("listening")
    assert m.state()["status"] == "Listening… release to send"  # the microphone is warm now
    time.sleep(0.6)
    m.pg.keyboard.up("Space")
    m.wait_mic("done")
    s = m.state()
    assert len(m.wavs) == 2 and s["request"] == STUB_TEXT and s["countdown"], s
    for w in m.wavs:
        channels, rate, width, frames, _ = read_wav(w)
        assert (channels, rate, width) == (1, 16000, 2) and frames >= 0.5 * 16000
    m.pg.keyboard.press("Escape")
    assert m.state()["countdown"] is None
    m.wait_mic("idle", timeout=5)  # "done" rests back to idle

    # Space on the Run button is the button's Space: no recording, and its release presses Run
    m.pg.fill("#request", "")
    m.pg.focus("#run-btn")
    n_trace = len(m.trace())
    m.pg.keyboard.down("Space")
    time.sleep(0.5)
    s = m.state()
    assert s["mic"] == "idle" and s["active"] == "run-btn", s
    m.pg.keyboard.up("Space")
    m.wait("document.querySelector('#form-error').textContent !== ''", timeout=5)
    time.sleep(0.3)
    s = m.state()
    assert s["formError"] == "Type what you want to do first." and s["mic"] == "idle", s
    assert len(m.trace()) == n_trace and len(m.wavs) == 2 and not m.run_bodies
    m.assert_clean(fast_console)


@e2e
def test_space_on_a_focused_scroller_scrolls_it(mic, fast_console):
    """Chromium lets Tab focus a scroller that overflows, with no tabindex (a step's "Screen text" <pre>, the banner
    details). Space there scrolls it, as on any control; it is not push-to-talk."""
    m = mic
    open_console(m, fast_console)
    m.pg.evaluate("""() => {
      const before = document.createElement('button');
      before.id = 't-before'; before.type = 'button'; before.textContent = 'before';
      const box = document.createElement('pre');
      box.id = 't-scroller'; box.style.cssText = 'overflow: auto; max-height: 4rem; white-space: pre';
      box.textContent = Array.from({ length: 60 }, (_, i) => 'line ' + i).join('\\n');
      document.querySelector('.log').append(before, box);
    }""")
    m.pg.focus("#t-before")
    m.pg.keyboard.press("Tab")
    focus = m.pg.evaluate("() => [document.activeElement.id, document.activeElement.hasAttribute('tabindex')]")
    assert focus == ["t-scroller", False], focus
    n_trace = len(m.trace())
    m.pg.keyboard.down("Space")
    time.sleep(0.5)
    s = m.state()
    m.pg.keyboard.up("Space")
    time.sleep(0.2)
    assert s["mic"] == "idle" and m.state()["mic"] == "idle", s
    assert m.pg.evaluate("document.getElementById('t-scroller').scrollTop") > 0  # Space scrolled it
    assert not m.wavs and len(m.trace()) == n_trace  # the mic never left idle

    # with the scroller gone, focus is on the page itself: Space is push-to-talk again
    m.pg.evaluate("document.getElementById('t-scroller').remove()")
    assert m.pg.evaluate("document.activeElement === document.body")
    m.pg.keyboard.down("Space")
    m.wait_mic("listening")
    m.pg.keyboard.press("Escape")
    m.pg.keyboard.up("Space")
    m.wait_mic("idle")
    assert not m.wavs
    m.assert_clean(fast_console)


@e2e
def test_saying_stop_stops_a_running_task(mic, slow_console):
    m = mic
    open_console(m, slow_console)
    m.pg.select_option("#task-select", LONG_TASK)
    m.pg.click("#run-btn")
    m.wait("[...document.querySelectorAll('#step-log .step-action')].some(e => e.textContent.startsWith('Clicking: '))",
           timeout=30)
    assert m.state()["run"] == "running"

    m.reply = {"text": "stop", "duration_s": 0.6, "latency_ms": 41.6, "model": "stub"}
    m.pg.click("#mic-btn")
    m.wait_mic("listening")
    time.sleep(0.5)
    m.pg.click("#mic-btn")
    m.wait("document.querySelector('#voice-status').textContent.startsWith('Stopping')")
    m.pg.wait_for_selector("#banner:not([hidden])", timeout=30000)
    assert m.pg.get_attribute("#banner", "data-tone") == "stopped"
    assert "You stopped the run." in m.pg.inner_text("#banner-claim")
    s = m.state()
    assert s["run"] == "stopped" and s["heard"] == "Heard (stub, 42 ms): stop", s
    tasks = {t["id"]: t for t in httpx.get(f"{slow_console}/api/tasks").json()}
    assert s["countdown"] is None and s["request"] == tasks[LONG_TASK]["request"]  # "stop" didn't touch the box

    run_id = m.pg.evaluate("sessionStorage.getItem('gui-assist.run')")
    info = httpx.get(f"{slow_console}/api/runs/{run_id}").json()
    assert info["status"] == "stopped" and info["task_id"] == LONG_TASK and info["verdict"]["passed"] is False
    assert [b["task_id"] for b in m.run_bodies] == [LONG_TASK]  # "stop" started nothing
    assert len(m.wavs) == 1
    m.assert_clean(slow_console)


# Audio spies for the fallback tests. `__mode`: 'resample' makes a 16 kHz context refuse the microphone (what
# Firefox does), so voice.js records at the default rate and resamples with wav.js; 'script' removes
# AudioWorkletNode, so it captures with a ScriptProcessor. Every stream and context is kept to check the release.
AUDIO_SPY = """(() => {
  const md = navigator.mediaDevices, gum = md.getUserMedia.bind(md), AC = window.AudioContext;
  window.__streams = []; window.__contexts = []; window.__scriptProcessors = 0; window.__gumCalls = 0;
  md.getUserMedia = async (c) => {
    window.__gumCalls += 1; const s = await gum(c); window.__streams.push(s); return s;
  };
  window.AudioContext = function (...a) { const c = new AC(...a); window.__contexts.push(c); return c; };
  window.AudioContext.prototype = AC.prototype;
  const csp = AC.prototype.createScriptProcessor;
  AC.prototype.createScriptProcessor = function (...a) { window.__scriptProcessors += 1; return csp.apply(this, a); };
  if (window.__mode === 'resample') {
    const cms = AC.prototype.createMediaStreamSource;
    AC.prototype.createMediaStreamSource = function (s) {
      if (this.sampleRate === 16000) throw new DOMException('different sample-rate not supported', 'NotSupportedError');
      return cms.call(this, s);
    };
  }
  if (window.__mode === 'script') window.AudioWorkletNode = undefined;
})();"""

# getUserMedia that waits while window.__hold is set, until window.__release().
GUM_GATE = """(() => {
  const md = navigator.mediaDevices, gum = md.getUserMedia.bind(md);
  window.__hold = false; window.__waiters = [];
  window.__release = () => { const w = window.__waiters; window.__waiters = []; w.forEach((f) => f()); };
  md.getUserMedia = async (c) => { if (window.__hold) await new Promise((r) => window.__waiters.push(r)); return gum(c); };
})();"""


@e2e
@pytest.mark.parametrize("mode", ["resample", "script"])
def test_capture_fallbacks_still_send_16k_mono_wav(mic, fast_console, mode):
    m = mic
    m.pg.context.add_init_script(f"window.__mode = {json.dumps(mode)};")
    m.pg.context.add_init_script(AUDIO_SPY)
    open_console(m, fast_console)
    m.pg.click("#mic-btn")
    m.wait_mic("listening")
    t0 = time.time()
    time.sleep(1.5)
    m.pg.click("#mic-btn")
    held = time.time() - t0
    m.wait_mic("done")
    m.pg.keyboard.press("Escape")  # no run

    assert len(m.wavs) == 1 and m.content_types == ["audio/wav"]
    channels, rate, width, frames, pcm = read_wav(m.wavs[0])
    assert (channels, rate, width) == (1, 16000, 2)
    assert abs(frames / rate - held) < 0.35, (frames / rate, held)  # a wrong resample ratio would show here
    assert max(abs(x) for x in array("h", pcm)) > 500  # the fake mic's beeps survived
    spy_js = """() => ({ rates: window.__contexts.map((c) => c.sampleRate),
        states: window.__contexts.map((c) => c.state), sp: window.__scriptProcessors, gum: window.__gumCalls,
        tracks: window.__streams.flatMap((s) => s.getTracks().map((t) => t.readyState)) })"""
    spy = m.pg.evaluate(spy_js)
    assert spy["tracks"] == ["live"] and spy["gum"] == 1, spy  # warm: kept for the next press
    if mode == "resample":
        assert spy["rates"][0] == 16000 and len(spy["rates"]) == 2 and spy["rates"][1] != 16000, spy
        assert spy["states"] == ["closed", "running"] and spy["sp"] == 0, spy  # the refused one is closed
    else:
        assert spy["rates"] == [16000] and spy["states"] == ["running"] and spy["sp"] == 1, spy
    assert m.state()["request"] == STUB_TEXT
    assert m.pg.evaluate("window.AcmeVoice.mic()")["kind"] == ("worklet" if mode == "resample" else "script")

    # the second press: no getUserMedia, and the take starts with the ~400 ms pre-roll (kept at the context's
    # rate in the worklet, or in the page for the ScriptProcessor), resampled like the rest
    time.sleep(0.6)
    m.pg.click("#mic-btn")
    m.wait_mic("listening")
    t0 = time.time()
    time.sleep(1.0)
    m.pg.click("#mic-btn")
    held = time.time() - t0
    m.wait_mic("done")
    m.pg.keyboard.press("Escape")
    take = m.pg.evaluate("window.AcmeVoice.lastTake()")
    assert not take["cold"] and 300 <= take["preRollMs"] <= 500 and take["rate"] == spy["rates"][-1], take
    channels, rate, width, frames, pcm = read_wav(m.wavs[1])
    assert (channels, rate, width) == (1, 16000, 2)
    assert abs(frames / rate - held - take["preRollMs"] / 1000) < 0.35, (frames / rate, held, take)
    assert max(abs(x) for x in array("h", pcm)) > 500
    assert m.pg.evaluate(spy_js)["gum"] == 1

    # Turn off: every track stopped, every context closed
    m.pg.click("#mic-off")
    m.wait("window.__contexts.every((c) => c.state === 'closed')")
    spy = m.pg.evaluate(spy_js)
    assert spy["tracks"] == ["ended"] and set(spy["states"]) == {"closed"}, spy
    m.assert_clean(fast_console)


@e2e
def test_status_line_never_shows_stale_text(mic, fast_console):
    """Regression: say() re-announces repeated text after 60 ms; a newer message must not be overwritten by it.

    A recording cancelled while the mic is starting leaves "Starting the microphone…" on screen. Starting again says
    the same words (cleared, rewritten 60 ms later) and the mic goes live in ~30 ms; the late write used to replace
    "Listening… release to send" while recording. Each round turns the warm microphone off, so the next starts cold."""
    m = mic
    m.pg.context.add_init_script(GUM_GATE)
    open_console(m, fast_console)
    for _ in range(3):
        m.pg.evaluate("window.__hold = true; document.activeElement.blur()")
        m.pg.keyboard.down("AltRight")
        m.wait_mic("opening")
        m.pg.click("#ptt-change")  # changing the key cancels the starting recording, quietly
        m.pg.keyboard.up("AltRight")
        m.pg.keyboard.press("Escape")  # leave key capture, keep Right Option
        s = m.state()
        assert s["mic"] == "idle" and s["status"] == "Starting the microphone…", s
        m.pg.evaluate("window.__hold = false; window.__release()")
        time.sleep(0.2)
        assert not m.pg.evaluate("window.AcmeVoice.mic().warm")  # cancelled before any audio: nothing kept open
        m.pg.evaluate("document.activeElement.blur()")
        m.pg.keyboard.down("AltRight")
        m.wait_mic("listening")
        time.sleep(0.3)  # well past the 60 ms re-announce
        s = m.state()
        assert s["mic"] == "listening" and s["status"] == COLD_PTT, s
        m.pg.keyboard.press("Escape")  # cancel; nothing sent
        m.pg.keyboard.up("AltRight")
        m.wait_mic("idle")
        assert m.pg.evaluate("window.AcmeVoice.mic().warm")
        m.pg.click("#mic-off")
        assert not m.pg.evaluate("window.AcmeVoice.mic().warm")
    assert not m.wavs and not m.run_bodies
    m.assert_clean(fast_console)
