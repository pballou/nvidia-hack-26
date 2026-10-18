"""Speech-to-text for the console's voice input: transcribe(wav_bytes) -> dict.

The browser records 16 kHz mono PCM16 WAV (console/static/wav.js) and POSTs it to
/api/transcribe. This module validates the WAV and runs the active engine. Engines are
local only (no cloud speech):
- "parakeet" (the default) is NVIDIA Parakeet TDT 0.6B v3 served by NeMo-Speech.cpp on
  127.0.0.1:8096 (scripts/run_asr.sh). It counts toward the 33B cap (contracts.md §6):
  30 + 2 + 0.6 = 32.6B by model card. Until the sidecar answers, /api/config reports voice
  input as unavailable: the page shows the mic as off and checks again by itself.
- "stub" returns a fixed string and recognizes nothing. Only an explicit `--asr stub` or
  CONSOLE_ASR=stub selects it. It is labelled "stub (fixed text)", and /api/config says
  `fixed_text: true`, so the page never runs its text by itself. Tests install StubEngine
  directly (set_engine, console/tests/conftest.py) as a stand-in for a real recognizer.

An engine is any object with `name` (model id shown in responses), `label` (short UI
name), `transcribe(pcm16: bytes, sample_rate: int) -> str` and optionally `fixed_text`
(True = it returns canned text, not what was said).
"""

from __future__ import annotations

import io
import os
import threading
import time
import wave
from typing import Callable, Protocol

SAMPLE_RATE = 16_000
MIN_SECONDS = 0.2
MAX_SECONDS = 60.0
MAX_BYTES = 44 + int(MAX_SECONDS * SAMPLE_RATE) * 2 + 1024  # header slack for extra chunks


class AudioError(ValueError):
    """The upload isn't a usable 16 kHz mono PCM16 WAV. The message is shown to the user."""


class EngineUnavailable(RuntimeError):
    """The configured engine can't run (not installed, failed to load)."""


class Engine(Protocol):
    name: str
    label: str

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str: ...


class StubEngine:
    """Returns a fixed transcript. CONSOLE_ASR_STUB_TEXT overrides the default.

    This is the tests' stand-in for a recognizer: the page treats its text like real speech."""

    name = "stub"
    label = "stub"
    fixed_text = False

    def __init__(self, text: str | None = None):
        self.text = text if text is not None else os.environ.get("CONSOLE_ASR_STUB_TEXT", "Turn on high contrast.")

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        return self.text


class FixedTextEngine(StubEngine):
    """`--asr stub` / CONSOLE_ASR=stub: the stub chosen by hand, to try the page without the sidecar.

    It says what it is (the label, and `fixed_text` in /api/config), so the page never presents
    its text as recognized speech and never runs it by itself."""

    label = "stub (fixed text)"
    fixed_text = True


ENGINES: dict[str, Callable[[], Engine]] = {"stub": FixedTextEngine}
_lock = threading.Lock()
_engine: Engine | None = None
_engine_name = os.environ.get("CONSOLE_ASR") or "parakeet"  # empty counts as unset (as in demo_up.sh)


def configure(name: str) -> None:
    """Select the engine by name (loaded lazily on first use)."""
    global _engine, _engine_name
    if name not in ENGINES:
        raise EngineUnavailable(f"unknown ASR engine {name!r}; have {sorted(ENGINES)}")
    with _lock:
        _engine_name, _engine = name, None


def set_engine(engine: Engine | None) -> None:
    """Install an engine instance directly (tests)."""
    global _engine
    with _lock:
        _engine = engine


def get_engine() -> Engine:
    global _engine
    with _lock:
        if _engine is None:
            try:
                _engine = ENGINES[_engine_name]()
            except EngineUnavailable:
                raise
            except Exception as exc:  # import errors, missing weights, ...
                raise EngineUnavailable(f"ASR engine {_engine_name!r} failed to load: {exc}") from exc
        return _engine


def info() -> dict:
    """For /api/config: whether voice input can work, and what to call the model."""
    try:
        engine = get_engine()
    except EngineUnavailable as exc:
        label = getattr(ENGINES.get(_engine_name), "label", None)  # e.g. "Parakeet TDT 0.6B v3", not "parakeet"
        return {"available": False, "model": _engine_name, "label": label if isinstance(label, str) else _engine_name,
                "error": str(exc)}
    out = {"available": True, "model": engine.name, "label": engine.label}
    if getattr(engine, "fixed_text", False):
        out["fixed_text"] = True
    return out


def read_wav(data: bytes) -> tuple[bytes, float]:
    """Validate a 16 kHz mono PCM16 WAV; return (pcm16 frames, duration in seconds)."""
    if not data:
        raise AudioError("No audio received")
    if len(data) > MAX_BYTES:
        raise AudioError(f"Recording is too long (max {MAX_SECONDS:.0f} s)")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("Audio must be a WAV file")
    try:
        with wave.open(io.BytesIO(data)) as w:
            channels, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            frames = w.readframes(n)
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"Unreadable WAV: {exc or 'the file is truncated'}") from exc
    if channels != 1 or width != 2 or rate != SAMPLE_RATE:
        raise AudioError(f"Audio must be 16 kHz mono 16-bit PCM (got {rate} Hz, {channels} channel(s), {8 * width}-bit)")
    duration = len(frames) / 2 / SAMPLE_RATE
    if duration < MIN_SECONDS:
        raise AudioError("Recording is too short")
    if duration > MAX_SECONDS:
        raise AudioError(f"Recording is too long (max {MAX_SECONDS:.0f} s)")
    return frames, duration


def transcribe(wav_bytes: bytes) -> dict:
    """-> {text, duration_s, latency_ms, model}. Raises AudioError or EngineUnavailable."""
    frames, duration = read_wav(wav_bytes)
    engine = get_engine()
    t0 = time.perf_counter()
    text = engine.transcribe(frames, SAMPLE_RATE)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {"text": (text or "").strip(), "duration_s": round(duration, 3), "latency_ms": round(latency_ms, 1),
            "model": engine.name}


class NemoSpeechEngine:
    """NVIDIA Parakeet TDT 0.6B v3 (official Q8_0 GGUF) via NeMo-Speech.cpp at CONSOLE_ASR_URL.

    The sidecar runs locally (scripts/run_asr.sh, default http://127.0.0.1:8096): POST
    /v1/audio/transcriptions with multipart file=<16 kHz mono PCM16 WAV>, response_format=json.
    """

    name = "nvidia/parakeet-tdt-0.6b-v3"
    label = "Parakeet TDT 0.6B v3"

    def __init__(self, url: str | None = None):
        import httpx

        self.url = (url or os.environ.get("CONSOLE_ASR_URL", "http://127.0.0.1:8096")).rstrip("/")
        self._http = httpx.Client(base_url=self.url, timeout=httpx.Timeout(30.0, connect=1.0), trust_env=False)
        try:
            r = self._http.get("/health", timeout=2.0)
        except httpx.HTTPError as exc:
            self._http.close()
            raise EngineUnavailable(f"Parakeet server not reachable at {self.url} ({exc}). "
                                    "Start it with scripts/run_asr.sh") from exc
        if r.status_code != 200:
            self._http.close()
            raise EngineUnavailable(f"Parakeet server at {self.url} not ready yet (HTTP {r.status_code})")

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        import httpx

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm16)
        try:
            r = self._http.post("/v1/audio/transcriptions", data={"response_format": "json"},
                                files={"file": ("speech.wav", buf.getvalue(), "audio/wav")})
        except httpx.HTTPError as exc:
            raise EngineUnavailable(f"Parakeet server failed: {exc}") from exc
        if 400 <= r.status_code < 500:
            raise AudioError(f"Parakeet rejected the audio: {r.text[:200]}")
        if r.status_code != 200:
            raise EngineUnavailable(f"Parakeet error {r.status_code}: {r.text[:200]}")
        return r.json().get("text", "")


ENGINES["parakeet"] = NemoSpeechEngine
